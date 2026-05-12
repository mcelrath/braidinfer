//! Model weight loading and initialization.
//! Extracted from model.rs for maintainability.

use std::path::Path;

use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;

use super::Model;
use crate::config::*;
use crate::kernel::AllKernels;
use crate::weights::*;

impl Model {
    /// Default max_seq_len cap for flat KV cache (limits VRAM usage).
    /// Override with `load_with_max_seq_len`. Paged KV grows dynamically.
    const DEFAULT_MAX_SEQ_LEN: usize = 8192;

    pub fn load(model_dir: &Path, device: DeviceId) -> Result<Self, ModelError> {
        Self::load_with_max_seq_len(model_dir, device, None)
    }

    pub fn load_with_max_seq_len(
        model_dir: &Path,
        device: DeviceId,
        max_seq_len: Option<usize>,
    ) -> Result<Self, ModelError> {
        let config_path = model_dir.join("config.json");
        let mut config = if config_path.exists() {
            ModelConfig::from_config_json(&config_path)
                .map_err(|e| ModelError::MissingWeight(format!("config.json: {e}")))?
        } else {
            ModelConfig::qwen35_0_8b()
        };
        // Cap max_seq_len: model may claim 262144 but flat KV can't afford that.
        // User override takes priority, otherwise cap at DEFAULT_MAX_SEQ_LEN.
        config.max_seq_len =
            max_seq_len.unwrap_or(config.max_seq_len.min(Self::DEFAULT_MAX_SEQ_LEN));

        // Weight quantization mode from env var
        config.weight_quant = match std::env::var("WEIGHT_QUANT").as_deref() {
            Ok("rnf4") => WeightQuantMode::Rnf4,
            Ok("mixed") => WeightQuantMode::Mixed,
            _ => WeightQuantMode::Bf16,
        };

        let multi_gpu = std::env::var("MULTI_GPU").is_ok();

        let st = SafeTensorSet::open_directory(model_dir)?;

        // Locate .bqnt file: BQNT_PATH env var takes priority; else auto-derive from model_dir.
        // Auto path: sibling to model_dir named "{model_dir_name}.q4.bqnt"
        let explicit_bqnt_path = std::env::var("BQNT_PATH")
            .ok()
            .map(std::path::PathBuf::from);
        let auto_bqnt_path = model_dir.file_name().map(|n| {
            model_dir
                .parent()
                .unwrap_or(model_dir)
                .join(format!("{}.q4.bqnt", n.to_string_lossy()))
        });
        let bqnt_path_to_try = explicit_bqnt_path.as_ref().or(auto_bqnt_path.as_ref());

        let bqnt = bqnt_path_to_try.and_then(|path| {
            if path.exists() {
                match crate::bqnt::MmapBqnt::open(path) {
                    Ok(b) => {
                        eprintln!(
                            "Loaded pre-quantized weights from {} ({} tensors)",
                            path.display(),
                            b.n_tensors()
                        );
                        Some(b)
                    }
                    Err(e) => {
                        eprintln!("WARNING: Failed to open {}: {e}", path.display());
                        None
                    }
                }
            } else {
                None
            }
        });

        // If no bqnt found and quantizing, create a writer to cache for next launch.
        // Only create writer when BQNT_PATH is not explicitly set (avoid overwriting user files).
        let save_bqnt_path = if bqnt.is_none()
            && explicit_bqnt_path.is_none()
            && config.weight_quant != WeightQuantMode::Bf16
        {
            auto_bqnt_path.clone()
        } else {
            None
        };
        let bqnt_writer: std::cell::RefCell<Option<crate::bqnt::BqntWriter>> =
            std::cell::RefCell::new(save_bqnt_path.as_ref().and_then(|p| {
                match crate::bqnt::BqntWriter::create(p, 65536) {
                    Ok(w) => {
                        eprintln!("First-time quantization: caching to {}", p.display());
                        Some(w)
                    }
                    Err(e) => {
                        eprintln!("WARNING: Cannot create bqnt cache at {}: {e}", p.display());
                        None
                    }
                }
            }));

        // Pin mmap'd shard regions so hipMemcpy can DMA directly (avoids bounce buffer).
        // Costs ~300ms upfront to fault in pages, but saves ~500ms on weight copies.
        // Skip when bqnt is present: linear weights come from bqnt, not safetensors,
        // so pinning 200+GB of safetensors shards would wastefully fault in pages.
        // Some models have mmap regions that fail hipHostRegister (non-page-aligned etc.);
        // track which succeeded so we only unregister those.
        let mut pinned: Vec<*mut std::ffi::c_void> = Vec::new();
        if bqnt.is_none() {
            let shard_ptrs: Vec<(*mut std::ffi::c_void, usize)> = st
                .shard_mmaps()
                .map(|m| (m.as_ptr() as *mut std::ffi::c_void, m.len()))
                .collect();
            for &(ptr, len) in &shard_ptrs {
                let rc = unsafe { ffi::hipHostRegister(ptr, len, 0) };
                if rc == 0 {
                    pinned.push(ptr);
                }
            }
            if pinned.len() < shard_ptrs.len() {
                eprintln!(
                    "Warning: {}/{} safetensor shards failed hipHostRegister (slower DMA fallback)",
                    shard_ptrs.len() - pinned.len(),
                    shard_ptrs.len()
                );
            }
        }

        // Discover tensor name prefix by finding "layers.0." in tensor names.
        // Prefer prefixes containing "model" to avoid matching MTP/draft heads.
        let prefix = {
            let names = st.tensor_names();
            let candidates: Vec<&str> = names
                .iter()
                .filter(|n| n.contains("layers.0."))
                .map(|n| &n[..n.find("layers.0.").unwrap()])
                .collect();
            let prefix = candidates
                .iter()
                .find(|p| p.contains("model"))
                .or_else(|| candidates.iter().find(|p| !p.contains("mtp")))
                .or(candidates.first())
                .ok_or_else(|| ModelError::MissingWeight("no layers.0. tensor found".into()))?;
            prefix.to_string()
        };

        let stream = Stream::new(device)?;
        let kernels = AllKernels::load(device)?;

        // Discover model features from tensor names
        let names = st.tensor_names();
        let has_qk_norm = names.iter().any(|n| n.contains("q_norm.weight"));
        config.has_qk_norm = has_qk_norm;

        // Detect gated Q: Qwen3.5 packs Q+gate in q_proj [nqh*hd*2, hidden].
        // Standard models have q_proj [nqh*hd, hidden].
        let first_attn_idx = config
            .layers
            .iter()
            .position(|l| l.layer_type == LayerType::Attention);
        let has_output_gate = if let Some(ai) = first_attn_idx {
            let q_name = format!("{prefix}layers.{ai}.self_attn.q_proj.weight");
            if let Ok(raw) = st.tensor_data(&q_name) {
                let expected_gated =
                    config.num_q_heads * config.head_dim * 2 * config.hidden_size * 2; // bf16
                raw.len() == expected_gated
            } else {
                false
            }
        } else {
            false
        };
        config.has_output_gate = has_output_gate;
        let embed_name = names
            .iter()
            .find(|n| {
                n.starts_with(&prefix)
                    && (n.contains("embed_tokens.weight")
                        || n.contains("tok_embeddings.weight")
                        || n.ends_with("wte.weight")
                        || n.contains("embeddings.weight"))
            })
            .or_else(|| {
                names.iter().find(|n| {
                    n.contains("embed_tokens.weight")
                        || n.contains("tok_embeddings.weight")
                        || n.ends_with("wte.weight")
                        || n.contains("embeddings.weight")
                })
            })
            .ok_or_else(|| ModelError::MissingWeight("embedding tensor not found".into()))?
            .to_string();
        let norm_name = names
            .iter()
            .find(|n| {
                n.starts_with(&prefix)
                    && (n.ends_with("norm.weight")
                        || n.ends_with("ln_f.weight")
                        || n.ends_with("norm_f.weight"))
                    && !n.contains("layers.")
            })
            .or_else(|| {
                names.iter().find(|n| {
                    (n.contains("norm.weight")
                        || n.contains("ln_f.weight")
                        || n.contains("norm_f.weight"))
                        && !n.contains("layers.")
                        && !n.contains("visual")
                        && !n.contains("mtp")
                })
            })
            .ok_or_else(|| ModelError::MissingWeight("final norm tensor not found".into()))?
            .to_string();

        let embed_weight = load_weight_bf16(
            &st,
            &embed_name,
            device,
            config.vocab_size * config.hidden_size,
        )?;
        let lm_head_weight = if config.tie_word_embeddings {
            // Weight-tied: reuse embed_weight pointer (allocate a dummy — the megakernel uses embed_weight)
            DeviceBuffer::<u16>::alloc(device, 0)? // placeholder, megakernel will use embed_weight
        } else {
            let lm_head_name = names
                .iter()
                .find(|n| n.contains("lm_head.weight"))
                .ok_or_else(|| ModelError::MissingWeight("lm_head.weight not found".into()))?
                .to_string();
            load_weight_bf16(
                &st,
                &lm_head_name,
                device,
                config.vocab_size * config.hidden_size,
            )?
        };
        let final_norm_weight = load_weight_bf16(&st, &norm_name, device, config.hidden_size)?;

        // Per-layer quantization control: WEIGHT_QUANT_LAYERS=0-11,20-31 restricts Q4 to those layers
        let quant_layers: Option<std::collections::HashSet<usize>> =
            std::env::var("WEIGHT_QUANT_LAYERS").ok().map(|s| {
                let mut set = std::collections::HashSet::new();
                for part in s.split(',') {
                    let part = part.trim();
                    if let Some((a, b)) = part.split_once('-') {
                        if let (Ok(start), Ok(end)) = (a.parse::<usize>(), b.parse::<usize>()) {
                            for i in start..=end {
                                set.insert(i);
                            }
                        }
                    } else if let Ok(n) = part.parse::<usize>() {
                        set.insert(n);
                    }
                }
                eprintln!(
                    "WEIGHT_QUANT_LAYERS: quantizing {} layers: {:?}",
                    set.len(),
                    {
                        let mut v: Vec<_> = set.iter().copied().collect();
                        v.sort();
                        v
                    }
                );
                set
            });

        // Per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut moe_weights_vec: Vec<Option<MoeWeights>> =
            (0..config.num_layers).map(|_| None).collect();
        let is_caching = save_bqnt_path.is_some() && bqnt_writer.borrow().is_some();
        for i in 0..config.num_layers {
            if is_caching {
                eprint!("\rQuantizing layer {}/{} ...", i + 1, config.num_layers);
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
            let p = format!("{prefix}layers.{i}.");
            let is_moe = matches!(config.layers[i].ffn_type, FfnType::MoE { .. });
            let wq = config.weight_quant;
            let use_quant = quant_layers.as_ref().map_or(true, |s| s.contains(&i));
            // Helper: load linear weight, trying bqnt first if available and layer is quantized.
            // Falls through to quantize-at-load from safetensors, caching to bqnt_writer if set.
            let load_lw =
                |name: &str, out_dim: usize, in_dim: usize| -> Result<LinearWeight, ModelError> {
                    if use_quant {
                        if let Some(ref b) = bqnt {
                            if let Ok(lw) = crate::weights::load_linear_weight_bqnt(b, name, device)
                            {
                                return Ok(lw);
                            }
                        }
                        if bqnt_writer.borrow().is_some() {
                            let mut guard = bqnt_writer.borrow_mut();
                            return crate::weights::load_linear_weight_cached(
                                &st,
                                name,
                                device,
                                out_dim,
                                in_dim,
                                wq,
                                guard.as_mut().unwrap(),
                            );
                        }
                    }
                    load_linear_weight(&st, name, device, out_dim, in_dim, wq)
                };
            let layer_type = &config.layers[i].layer_type;
            if *layer_type == LayerType::Mamba2 {
                // Mamba2 SSM layer (Nemotron-H 'M' layers)
                let hs = config.hidden_size;
                let (nh, hd, _sd, ck, _ng, cd) = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 {
                        num_heads,
                        head_dim,
                        state_dim,
                        conv_kernel,
                        n_groups,
                        conv_dim,
                        ..
                    } => (
                        *num_heads,
                        *head_dim,
                        *state_dim,
                        *conv_kernel,
                        *n_groups,
                        *conv_dim,
                    ),
                    _ => panic!("Mamba2 layer but no Mamba2 recurrent config"),
                };
                let intermediate = nh * hd;
                let in_proj_size = intermediate + cd + nh; // gate + xBC + dt
                // Try Nemotron weight names first, then generic
                let norm_name = find_weight_name(
                    &st,
                    &[
                        format!("{p}norm.weight"),
                        format!("{p}input_layernorm.weight"),
                    ],
                )?;
                let w = Mamba2LayerWeights {
                    input_norm: load_weight_bf16(&st, &norm_name, device, hs)?,
                    in_proj: load_lw(&format!("{p}mixer.in_proj.weight"), in_proj_size, hs)?,
                    conv1d_weight: load_weight_bf16(
                        &st,
                        &format!("{p}mixer.conv1d.weight"),
                        device,
                        cd * ck,
                    )?,
                    conv1d_bias: load_weight_f32(
                        &st,
                        &format!("{p}mixer.conv1d.bias"),
                        device,
                        cd,
                    )?,
                    dt_bias: load_weight_f32(&st, &format!("{p}mixer.dt_bias"), device, nh)?,
                    a_log: load_weight_f32(&st, &format!("{p}mixer.A_log"), device, nh)?,
                    d: load_weight_f32(&st, &format!("{p}mixer.D"), device, nh)?,
                    norm_weight: load_weight_f32(
                        &st,
                        &format!("{p}mixer.norm.weight"),
                        device,
                        intermediate,
                    )?,
                    out_proj: load_lw(&format!("{p}mixer.out_proj.weight"), hs, intermediate)?,
                };
                layers.push(LayerWeights::Mamba2(w));
            } else if *layer_type == LayerType::MoeFfn {
                // Standalone MoE FFN layer (Nemotron-H 'E' layers)
                let hs = config.hidden_size;
                let norm_name = find_weight_name(
                    &st,
                    &[
                        format!("{p}norm.weight"),
                        format!("{p}input_layernorm.weight"),
                    ],
                )?;
                let w = MoeFfnLayerWeights {
                    input_norm: load_weight_bf16(&st, &norm_name, device, hs)?,
                };
                layers.push(LayerWeights::MoeFfn(w));
                // Load MoE weights — Nemotron uses mixer.gate/mixer.experts instead of mlp.gate/mlp.experts
                // Try Nemotron naming first by checking if mixer.gate.weight exists
                let moe_prefix = if st.tensor_data(&format!("{p}mixer.gate.weight")).is_ok() {
                    format!("{p}mixer.")
                } else {
                    format!("{p}mlp.")
                };
                moe_weights_vec[i] = Some(if multi_gpu {
                    if bqnt_writer.borrow().is_some() {
                        let mut g = bqnt_writer.borrow_mut();
                        crate::weights::load_moe_weights_lite_cached(
                            &st,
                            &moe_prefix,
                            &config,
                            &config.layers[i].ffn_type,
                            device,
                            wq,
                            bqnt.as_ref(),
                            g.as_mut().unwrap(),
                        )?
                    } else {
                        load_moe_weights_lite(
                            &st,
                            &moe_prefix,
                            &config,
                            &config.layers[i].ffn_type,
                            device,
                            wq,
                            bqnt.as_ref(),
                        )?
                    }
                } else {
                    if bqnt_writer.borrow().is_some() {
                        let mut g = bqnt_writer.borrow_mut();
                        crate::weights::load_moe_weights_cached(
                            &st,
                            &moe_prefix,
                            &config,
                            &config.layers[i].ffn_type,
                            device,
                            wq,
                            bqnt.as_ref(),
                            g.as_mut().unwrap(),
                        )?
                    } else {
                        load_moe_weights(
                            &st,
                            &moe_prefix,
                            &config,
                            &config.layers[i].ffn_type,
                            device,
                            wq,
                            bqnt.as_ref(),
                        )?
                    }
                });
            } else if config.layers[i].layer_type == LayerType::Attention {
                let hs = config.hidden_size;
                let q_mult = if has_output_gate { 2 } else { 1 };
                let w = AttentionLayerWeights {
                    input_norm: load_weight_bf16(
                        &st,
                        &find_weight_name(
                            &st,
                            &[
                                format!("{p}input_layernorm.weight"),
                                format!("{p}norm.weight"),
                            ],
                        )?,
                        device,
                        hs,
                    )?,
                    w_q_gate: load_lw(
                        &find_weight_name(
                            &st,
                            &[
                                format!("{p}self_attn.q_proj.weight"),
                                format!("{p}mixer.q_proj.weight"),
                            ],
                        )?,
                        config.num_q_heads * config.head_dim * q_mult,
                        hs,
                    )?,
                    w_k: load_lw(
                        &find_weight_name(
                            &st,
                            &[
                                format!("{p}self_attn.k_proj.weight"),
                                format!("{p}mixer.k_proj.weight"),
                            ],
                        )?,
                        config.num_kv_heads * config.head_dim,
                        hs,
                    )?,
                    w_v: load_lw(
                        &find_weight_name(
                            &st,
                            &[
                                format!("{p}self_attn.v_proj.weight"),
                                format!("{p}mixer.v_proj.weight"),
                            ],
                        )?,
                        config.num_kv_heads * config.head_dim,
                        hs,
                    )?,
                    w_o: load_lw(
                        &find_weight_name(
                            &st,
                            &[
                                format!("{p}self_attn.o_proj.weight"),
                                format!("{p}mixer.o_proj.weight"),
                            ],
                        )?,
                        hs,
                        config.num_q_heads * config.head_dim,
                    )?,
                    q_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.q_norm.weight");
                        let raw = st
                            .tensor_data(&name)
                            .map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else {
                        DeviceBuffer::<u16>::alloc(device, 0)?
                    },
                    k_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.k_norm.weight");
                        let raw = st
                            .tensor_data(&name)
                            .map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else {
                        DeviceBuffer::<u16>::alloc(device, 0)?
                    },
                    post_norm: {
                        let name =
                            find_weight_name(&st, &[format!("{p}post_attention_layernorm.weight")]);
                        if let Ok(n) = name {
                            load_weight_bf16(&st, &n, device, hs)?
                        } else {
                            DeviceBuffer::<u16>::alloc(device, 0)?
                        } // no post-norm (Nemotron * layers)
                    },
                    w_gate: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(
                            &format!("{p}mlp.gate_proj.weight"),
                            config.intermediate_size,
                            hs,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                    w_up: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(
                            &format!("{p}mlp.up_proj.weight"),
                            config.intermediate_size,
                            hs,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                    w_down: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(
                            &format!("{p}mlp.down_proj.weight"),
                            hs,
                            config.intermediate_size,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                };
                layers.push(LayerWeights::Attention(w));

                // Load MoE weights if this layer uses MoE FFN
                if is_moe {
                    moe_weights_vec[i] = Some(if multi_gpu {
                        if bqnt_writer.borrow().is_some() {
                            let mut g = bqnt_writer.borrow_mut();
                            crate::weights::load_moe_weights_lite_cached(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                                g.as_mut().unwrap(),
                            )?
                        } else {
                            load_moe_weights_lite(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                            )?
                        }
                    } else {
                        if bqnt_writer.borrow().is_some() {
                            let mut g = bqnt_writer.borrow_mut();
                            crate::weights::load_moe_weights_cached(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                                g.as_mut().unwrap(),
                            )?
                        } else {
                            load_moe_weights(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                            )?
                        }
                    });
                }
            } else {
                let nh = config.linear_num_heads;
                let nvh = config.linear_num_value_heads;
                let kd = config.linear_key_head_dim;
                let vd = config.linear_value_head_dim;
                let qkv_out = nh * kd + nh * kd + nvh * vd;
                let z_out = nvh * vd;
                let ck = config.linear_conv_kernel_dim;
                let q_dim = nh * kd;
                let v_dim = nvh * vd;
                let conv_total = qkv_out * ck;
                let conv_name = format!("{p}linear_attn.conv1d.weight");
                let conv_raw_bytes = st
                    .tensor_data(&conv_name)
                    .map_err(|_| ModelError::MissingWeight(conv_name.clone()))?;
                assert_eq!(conv_raw_bytes.len(), conv_total * 2);
                let conv_raw: &[u16] = unsafe {
                    std::slice::from_raw_parts(conv_raw_bytes.as_ptr() as *const u16, conv_total)
                };
                let trace_path = std::env::var("TRACE").ok();
                let conv1d_weight_buf = if trace_path.is_some() {
                    let mut buf = DeviceBuffer::<u16>::alloc(device, conv_total)?;
                    buf.copy_from_host(conv_raw)?;
                    Some(buf)
                } else {
                    None
                };
                let mut conv_w_q_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_k_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_v_buf = DeviceBuffer::<u16>::alloc(device, v_dim * ck)?;
                conv_w_q_buf.copy_from_host(&conv_raw[..q_dim * ck])?;
                conv_w_k_buf.copy_from_host(&conv_raw[q_dim * ck..2 * q_dim * ck])?;
                conv_w_v_buf.copy_from_host(&conv_raw[2 * q_dim * ck..])?;
                let hs = config.hidden_size;
                let w = GdnLayerWeights {
                    input_norm: load_weight_bf16(
                        &st,
                        &format!("{p}input_layernorm.weight"),
                        device,
                        hs,
                    )?,
                    w_qkv: load_lw(&format!("{p}linear_attn.in_proj_qkv.weight"), qkv_out, hs)?,
                    w_a: load_lw(&format!("{p}linear_attn.in_proj_a.weight"), nvh, hs)?,
                    w_b: load_lw(&format!("{p}linear_attn.in_proj_b.weight"), nvh, hs)?,
                    w_z: load_lw(&format!("{p}linear_attn.in_proj_z.weight"), z_out, hs)?,
                    conv1d_weight: conv1d_weight_buf, // Some only when TRACE env var set
                    conv1d_weight_q: conv_w_q_buf,
                    conv1d_weight_k: conv_w_k_buf,
                    conv1d_weight_v: conv_w_v_buf,
                    a_log: load_weight_f32(&st, &format!("{p}linear_attn.A_log"), device, nvh)?,
                    dt_bias: load_weight_bf16(
                        &st,
                        &format!("{p}linear_attn.dt_bias"),
                        device,
                        nvh,
                    )?,
                    output_norm: load_weight_f32(
                        &st,
                        &format!("{p}linear_attn.norm.weight"),
                        device,
                        vd,
                    )?, // normalizes [nvh, vd] output
                    w_out: load_lw(&format!("{p}linear_attn.out_proj.weight"), hs, z_out)?,
                    post_norm: load_weight_bf16(
                        &st,
                        &format!("{p}post_attention_layernorm.weight"),
                        device,
                        hs,
                    )?,
                    w_gate: if !is_moe {
                        load_lw(
                            &format!("{p}mlp.gate_proj.weight"),
                            config.intermediate_size,
                            hs,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                    w_up: if !is_moe {
                        load_lw(
                            &format!("{p}mlp.up_proj.weight"),
                            config.intermediate_size,
                            hs,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                    w_down: if !is_moe {
                        load_lw(
                            &format!("{p}mlp.down_proj.weight"),
                            hs,
                            config.intermediate_size,
                        )?
                    } else {
                        LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
                    },
                };
                layers.push(LayerWeights::Gdn(w));

                // Load MoE weights for GDN layers with MoE FFN (e.g. Qwen3.5-122B)
                if is_moe {
                    moe_weights_vec[i] = Some(if multi_gpu {
                        if bqnt_writer.borrow().is_some() {
                            let mut g = bqnt_writer.borrow_mut();
                            crate::weights::load_moe_weights_lite_cached(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                                g.as_mut().unwrap(),
                            )?
                        } else {
                            load_moe_weights_lite(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                            )?
                        }
                    } else {
                        if bqnt_writer.borrow().is_some() {
                            let mut g = bqnt_writer.borrow_mut();
                            crate::weights::load_moe_weights_cached(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                                g.as_mut().unwrap(),
                            )?
                        } else {
                            load_moe_weights(
                                &st,
                                &p,
                                &config,
                                &config.layers[i].ffn_type,
                                device,
                                wq,
                                bqnt.as_ref(),
                            )?
                        }
                    });
                }
            }
        }

        // Finish bqnt cache file if we created one
        if let Some(writer) = bqnt_writer.into_inner() {
            if let Some(ref p) = save_bqnt_path {
                eprintln!("\nSaving quantized weights to {} ...", p.display());
                match writer.finish("{}") {
                    Ok(()) => eprintln!("Cached weights saved to {}", p.display()),
                    Err(e) => eprintln!("WARNING: Failed to save bqnt cache: {e}"),
                }
            }
        } else if is_caching {
            eprintln!();
        }

        // Unpin mmap'd regions now that all weights are on GPU
        for ptr in &pinned {
            unsafe { ffi::hipHostUnregister(*ptr) };
        }

        // GDN states: [nh * kd * vd] per GDN layer
        let nh = config.linear_num_heads;
        let nvh = config.linear_num_value_heads;
        let kd = config.linear_key_head_dim;
        let vd = config.linear_value_head_dim;
        let ck = config.linear_conv_kernel_dim;
        let qkv_out = nh * kd * 2 + nvh * vd;

        let mut gdn_states = Vec::new();
        let mut gdn_conv_states = Vec::new();
        for i in 0..config.num_layers {
            if config.layers[i].layer_type == LayerType::Gdn {
                let mut recurrent = DeviceBuffer::<f32>::alloc(device, nvh * kd * vd)?;
                let zeros = vec![0.0f32; nvh * kd * vd];
                recurrent.copy_from_host(&zeros)?;
                gdn_states.push(GdnState { recurrent });

                let mut conv_state = DeviceBuffer::<f32>::alloc(device, qkv_out * (ck - 1))?;
                let zeros = vec![0.0f32; qkv_out * (ck - 1)];
                conv_state.copy_from_host(&zeros)?;
                gdn_conv_states.push(conv_state);
            }
        }

        // Mamba2 states: [num_heads, head_dim, state_size] SSM + [conv_dim, conv_kernel] conv
        let mut mamba2_states = Vec::new();
        if let RecurrentLayerKind::Mamba2 {
            num_heads: m_nh,
            head_dim: m_hd,
            state_dim: m_sd,
            conv_kernel: m_ck,
            conv_dim: m_cd,
            ..
        } = &config.recurrent_kind
        {
            for i in 0..config.num_layers {
                if config.layers[i].layer_type == LayerType::Mamba2 {
                    let ssm_size = m_nh * m_hd * m_sd;
                    let mut ssm = DeviceBuffer::<f32>::alloc(device, ssm_size)?;
                    ssm.copy_from_host(&vec![0.0f32; ssm_size])?;
                    let conv_size = m_cd * (m_ck - 1); // conv state = [conv_dim, kernel-1]
                    let mut conv = DeviceBuffer::<f32>::alloc(device, conv_size)?;
                    conv.copy_from_host(&vec![0.0f32; conv_size])?;
                    mamba2_states.push(Mamba2State { ssm, conv });
                }
            }
        }

        // KV caches
        let kv_size = config.max_seq_len * config.num_kv_heads * config.head_dim;
        let zeros_kv = vec![0.0f32; kv_size];
        let mut kv_caches = Vec::new();
        for i in 0..config.num_layers {
            if config.layers[i].layer_type == LayerType::Attention {
                let mut k = DeviceBuffer::<f32>::alloc(device, kv_size)?;
                let mut v = DeviceBuffer::<f32>::alloc(device, kv_size)?;
                k.copy_from_host(&zeros_kv)?;
                v.copy_from_host(&zeros_kv)?;
                kv_caches.push(KvCache { k, v });
            }
        }

        // inv_freq
        let inv_freq_data = compute_inv_freq(config.rope_dim, config.rope_theta);
        let mut inv_freq_buf = DeviceBuffer::<f32>::alloc(device, inv_freq_data.len())?;
        inv_freq_buf.copy_from_host(&inv_freq_data)?;

        let pos_buf = braidinfer_hip::MappedHostBuffer::<i32>::alloc(3)?;

        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let vs = config.vocab_size;
        let nqh = config.num_q_heads;
        let hd = config.head_dim;
        let nkh = config.num_kv_heads;

        let activations = ActivationBuffers {
            hidden: DeviceBuffer::<f32>::alloc(device, hs)?,
            normed: DeviceBuffer::<f32>::alloc(device, hs)?,
            qkv: DeviceBuffer::<f32>::alloc(device, qkv_out)?,
            q_gdn: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            k_gdn: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            v_gdn: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            a_proj: DeviceBuffer::<f32>::alloc(device, nvh)?,
            b_proj: DeviceBuffer::<f32>::alloc(device, nvh)?,
            z_proj: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            gate_gdn: DeviceBuffer::<f32>::alloc(device, nvh)?,
            recurrent_out: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            normed_gated: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            out_proj: DeviceBuffer::<f32>::alloc(device, hs)?,
            q_gate_attn: DeviceBuffer::<f32>::alloc(
                device,
                nqh * hd * if config.has_output_gate { 2 } else { 1 },
            )?,
            q_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            // 5ax-decode fix: gate_attn is also workers-write → GPU 0-read
            // when has_output_gate is true. Same UC treatment as attn_out.
            gate_attn: DeviceBuffer::<f32>::alloc_uncached(device, nqh * hd)?,
            k_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            v_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            // 5ax-decode fix per GFX1100_ARCH.md §5.4: attn_out is the
            // canonical "pool-cycled scratch reused every decode step" L2-
            // stale candidate. Workers P2P-write via UC peer mapping; GPU 0
            // reads back. With cached alloc, GPU 0's L2 holds the previous
            // step's attn_out — workers' fresh write lands in VRAM but
            // GPU 0 reads stale from L2 (gfx1100 has no buffer_gl2_inv).
            // alloc_uncached forces both write-target and read-source to
            // bypass L2 → GPU 0 reads fresh VRAM. gate_attn has same role
            // when has_output_gate=true (Qwen-style).
            attn_out: DeviceBuffer::<f32>::alloc_uncached(device, nqh * hd)?,
            gated_out: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            ffn_gate: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_up: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_act: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_down: DeviceBuffer::<f32>::alloc(device, hs)?,
            residual: DeviceBuffer::<f32>::alloc(device, hs)?,
            logits: DeviceBuffer::<f32>::alloc(device, vs)?,
            logits_mapped: braidinfer_hip::MappedHostBuffer::<f32>::alloc(vs)?,
            inv_freq: inv_freq_buf,
            position_ids: pos_buf,
            // MoE scratch: sized for per-layer max expert dimensions
            moe_scores: DeviceBuffer::<f32>::alloc(
                device,
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE { num_experts, .. } => Some(*num_experts),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            normed_stage: braidinfer_hip::memory::MappedHostBuffer::<f32>::alloc_portable(hs)?,
            ffn_down_stage: braidinfer_hip::memory::MappedHostBuffer::<f32>::alloc(hs)?,
            moe_expert_ids: braidinfer_hip::memory::MappedHostBuffer::<i32>::alloc(
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE { num_active, .. } => Some(*num_active),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            moe_expert_weights: braidinfer_hip::memory::MappedHostBuffer::<f32>::alloc(
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE { num_active, .. } => Some(*num_active),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            moe_expert_gate: DeviceBuffer::<f32>::alloc(
                device,
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE {
                            expert_intermediate_size,
                            shared_intermediate_size,
                            ..
                        } => Some((*expert_intermediate_size).max(*shared_intermediate_size)),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            moe_expert_up: DeviceBuffer::<f32>::alloc(
                device,
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE {
                            expert_intermediate_size,
                            shared_intermediate_size,
                            ..
                        } => Some((*expert_intermediate_size).max(*shared_intermediate_size)),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            moe_expert_act: DeviceBuffer::<f32>::alloc(
                device,
                config
                    .layers
                    .iter()
                    .filter_map(|l| match &l.ffn_type {
                        FfnType::MoE {
                            expert_intermediate_size,
                            shared_intermediate_size,
                            ..
                        } => Some((*expert_intermediate_size).max(*shared_intermediate_size)),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1),
            )?,
            moe_expert_out: DeviceBuffer::<f32>::alloc(device, hs)?,
            moe_latent: DeviceBuffer::<f32>::alloc(device, config.moe_latent_size.unwrap_or(hs))?,
            // Mamba2 scratch: sized from recurrent_kind if Mamba2
            mamba2_in_proj: {
                let size = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 {
                        num_heads,
                        head_dim,
                        conv_dim,
                        ..
                    } => num_heads * head_dim + conv_dim + num_heads, // gate + xBC + dt
                    _ => 1,
                };
                DeviceBuffer::<f32>::alloc(device, size)?
            },
            mamba2_conv_out: {
                let size = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 { conv_dim, .. } => *conv_dim,
                    _ => 1,
                };
                DeviceBuffer::<f32>::alloc(device, size)?
            },
            mamba2_ssm_out: {
                let size = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 {
                        num_heads,
                        head_dim,
                        ..
                    } => num_heads * head_dim,
                    _ => 1,
                };
                DeviceBuffer::<f32>::alloc(device, size)?
            },
            argmax_result: DeviceBuffer::<i32>::alloc(device, 1)?,
            // Prefill MoE batched scratch
            prefill_moe_normed: {
                let latent = config.moe_latent_size.unwrap_or(hs);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * latent)?
            },
            prefill_moe_expert_input: {
                let latent = config.moe_latent_size.unwrap_or(hs);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * latent)?
            },
            prefill_moe_gate_out: {
                let max_eis = config.layers.iter().filter_map(|l| match &l.ffn_type {
                    FfnType::MoE { expert_intermediate_size, .. } => Some(*expert_intermediate_size),
                    _ => None,
                }).max().unwrap_or(1);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * max_eis)?
            },
            prefill_moe_up_out: {
                let max_eis = config.layers.iter().filter_map(|l| match &l.ffn_type {
                    FfnType::MoE { expert_intermediate_size, .. } => Some(*expert_intermediate_size),
                    _ => None,
                }).max().unwrap_or(1);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * max_eis)?
            },
            prefill_moe_act_out: {
                let max_eis = config.layers.iter().filter_map(|l| match &l.ffn_type {
                    FfnType::MoE { expert_intermediate_size, .. } => Some(*expert_intermediate_size),
                    _ => None,
                }).max().unwrap_or(1);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * max_eis)?
            },
            prefill_moe_down_out: DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * hs)?,
            prefill_moe_ffn_out: DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * hs)?,
            prefill_moe_residual: DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * hs)?,
            prefill_moe_ids_dev: {
                let max_k = config.layers.iter().filter_map(|l| match &l.ffn_type {
                    FfnType::MoE { num_active, .. } => Some(*num_active),
                    _ => None,
                }).max().unwrap_or(1);
                DeviceBuffer::<i32>::alloc(device, crate::megakernel::CHUNK_TOKENS * max_k)?
            },
            prefill_moe_weights_dev: {
                let max_k = config.layers.iter().filter_map(|l| match &l.ffn_type {
                    FfnType::MoE { num_active, .. } => Some(*num_active),
                    _ => None,
                }).max().unwrap_or(1);
                DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS * max_k)?
            },
            prefill_moe_token_indices: DeviceBuffer::<i32>::alloc(device, crate::megakernel::CHUNK_TOKENS)?,
            prefill_moe_token_weights: DeviceBuffer::<f32>::alloc(device, crate::megakernel::CHUNK_TOKENS)?,
        };

        let has_moe = config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        Ok(Model {
            config,
            device,
            stream,
            kernels,
            embed_weight,
            lm_head_weight,
            final_norm_weight,
            layers,
            moe_weights: moe_weights_vec,
            activations,
            gdn_conv_states,
            legacy_kv_caches: Some(kv_caches),
            megakernel_prefill: None,
            megakernel_prefill_partial: None,
            megakernel_prefill_partial_n: 0,
            megakernel_prefill_segments: std::collections::HashMap::new(),
            prefill_bufs: None,
            gdn_states,
            mamba2_states,
            seq_len: 0,
            megakernel_paged: None,
            page_allocator: None,
            quant_allocator: None,
            paged_seq: None,
            paged_page_table: None,
            paged_position_table: None,
            checkpoint_pool: None,
            last_checkpoint_slot: None,
            trace: std::env::var("TRACE")
                .ok()
                .and_then(|path| crate::trace::TraceWriter::open(&path).ok()),
            debug_nan: std::env::var("DEBUG_NAN").is_ok(),
            has_moe,
            persistent: std::env::var("PERSISTENT").as_deref() == Ok("1"),
            kv_quant: std::env::var("KV_QUANT").as_deref() == Ok("1"),
            sync_debug: std::env::var("SYNC_DEBUG").is_ok(),
            debug_p2p_hidden: std::env::var("DEBUG_P2P_HIDDEN").is_ok(),
            weight_prefix: prefix.clone(),
            multi_gpu: None,
            distributed_moe: Vec::new(),
            moe_p2p: None,
            megakernel_multi_gpu_p2p: None,
            persistent_workers: None,
        })
    }

    /// Enable multi-GPU expert parallel dispatch.
    /// Distributes MoE expert weights across available GPUs (round-robin).
    /// Must be called after load, before first decode_step.
    pub fn enable_multi_gpu(&mut self) -> Result<(), ModelError> {
        if !self.has_moe {
            eprintln!("Multi-GPU: model has no MoE layers, skipping");
            return Ok(());
        }

        let max_eis = self
            .config
            .layers
            .iter()
            .filter_map(|l| match &l.ffn_type {
                FfnType::MoE {
                    expert_intermediate_size,
                    ..
                } => Some(*expert_intermediate_size),
                _ => None,
            })
            .max()
            .unwrap_or(1);

        let ctx = crate::multi_gpu::MultiGpuContext::init(self.config.hidden_size, max_eis)?;
        let mut ctx = match ctx {
            Some(c) => c,
            None => {
                // Only 1 GPU available. If expert weights were loaded lite (skipped because
                // MULTI_GPU was set), inference will silently produce wrong output or fault.
                // Callers should check GPU count BEFORE setting MULTI_GPU to avoid this.
                let experts_missing = self.moe_weights.iter().any(|m| {
                    m.as_ref().map_or(false, |moe| {
                        moe.expert_gate_up.raw_data_ptr() == std::ptr::null()
                    })
                });
                if experts_missing {
                    return Err(ModelError::MissingWeight(
                        "MULTI_GPU=1 but only 1 GPU available: expert weights were skipped at \
                         load time and cannot be used. Do not set MULTI_GPU=1 with a single GPU."
                            .into(),
                    ));
                }
                eprintln!("Multi-GPU: only 1 device, skipping");
                return Ok(());
            }
        };

        braidinfer_hip::device::Device::set_current(DeviceId(0))?;

        // Distribute MoE weights across GPUs
        let num_devices = ctx.num_devices;
        let hs = self.config.hidden_size;

        // Check if expert weights were loaded (single-GPU) or skipped (multi-GPU lite load)
        let experts_on_gpu0 = self.moe_weights.iter().any(|m| {
            m.as_ref().map_or(false, |moe| {
                moe.expert_gate_up.raw_data_ptr() != std::ptr::null()
            })
        });

        let bqnt = std::env::var("BQNT_PATH")
            .ok()
            .and_then(|p| crate::bqnt::MmapBqnt::open(std::path::Path::new(&p)).ok());

        let mut distributed = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            if let Some(ref moe) = self.moe_weights[i] {
                // Distribute experts starting at GPU 0 for all paths.
                // Persistent path: GPU 0 runs OP_EXPERT_FFN via fat worker at OP_BARRIER.
                // kbk path: GPU 0 runs experts via hipLaunchKernel (no cooperative kernel).
                let start_gpu = 0usize;
                if experts_on_gpu0 {
                    let dist = crate::weights::distribute_moe_weights_from_ref(
                        moe,
                        num_devices,
                        hs,
                        start_gpu,
                    )?;
                    distributed.push(Some(dist));
                } else if let Some(ref b) = bqnt {
                    let dist = crate::weights::distribute_moe_weights_from_bqnt(
                        moe,
                        b,
                        i,
                        &self.weight_prefix,
                        num_devices,
                        hs,
                        start_gpu,
                    )?;
                    distributed.push(Some(dist));
                } else {
                    return Err(ModelError::MissingWeight(
                        "Multi-GPU requires BQNT_PATH for direct expert loading".into(),
                    ));
                }
            } else {
                distributed.push(None);
            }
        }

        self.distributed_moe = distributed;
        eprintln!("Multi-GPU: experts distributed across all {num_devices} GPUs");

        // Allocate head-parallel attention buffers for all GPUs
        let num_attn_layers = self
            .config
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::config::LayerType::Attention)
            .count();
        // GQA replicates KV heads on every GPU, but Q heads are partitioned evenly.
        if num_attn_layers > 0 {
            if self.config.num_q_heads < num_devices {
                return Err(ModelError::InvalidConfig(format!(
                    "multi-GPU attention requires num_q_heads ({}) >= num_devices ({num_devices})",
                    self.config.num_q_heads
                )));
            }
            if self.config.num_q_heads % num_devices != 0 {
                return Err(ModelError::InvalidConfig(format!(
                    "multi-GPU attention requires num_q_heads ({}) to be divisible by num_devices ({num_devices})",
                    self.config.num_q_heads
                )));
            }
            let local_nqh = self.config.num_q_heads / num_devices;
            let local_nkh = self.config.num_kv_heads; // replicated on every GPU
            let q_mult = if self.config.has_output_gate { 2 } else { 1 };
            ctx.init_attn_buffers(
                num_attn_layers,
                local_nqh,
                local_nkh,
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.hidden_size,
                q_mult,
            )?;
            // Split Q/K/V projection weights onto each GPU
            self.init_split_attn_weights(&mut ctx, local_nqh, local_nkh, q_mult)?;
        }

        self.multi_gpu = Some(ctx);
        Ok(())
    }

    /// Copy row-slices of Q/K/V attention weights onto each GPU for distributed projection.
    /// Each GPU i gets rows [i*local_rows .. (i+1)*local_rows] of each weight matrix.
    fn init_split_attn_weights(
        &self,
        ctx: &mut crate::multi_gpu::MultiGpuContext,
        local_nqh: usize,
        local_nkh: usize,
        q_mult: usize,
    ) -> Result<(), ModelError> {
        use crate::multi_gpu::MultiGpuContext;
        let num_gpus = ctx.num_devices;
        let hs = self.config.hidden_size;
        let hd = self.config.head_dim;

        let attn_layer_indices: Vec<usize> = self
            .config
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.layer_type == crate::config::LayerType::Attention)
            .map(|(i, _)| i)
            .collect();

        for &layer_idx in attn_layer_indices.iter() {
            let w = match &self.layers[layer_idx] {
                LayerWeights::Attention(w) => w,
                _ => continue,
            };
            // GPU 0: skip copy — dispatch_head_parallel_attention reads from self.layers directly.
            // GPUs 1+: copy the row slice to each GPU's VRAM.
            for gpu_i in 1..num_gpus {
                let dst_device = ctx.workers[gpu_i].device;
                let q_row_start = gpu_i * local_nqh * hd * q_mult;
                // KV heads are replicated (GQA): always copy from row 0, copy all local_nkh rows
                let w_q = MultiGpuContext::copy_weight_slice(
                    &w.w_q_gate,
                    dst_device,
                    q_row_start,
                    local_nqh * hd * q_mult,
                    hs,
                )
                .map_err(ModelError::Hip)?;
                let w_k =
                    MultiGpuContext::copy_weight_slice(&w.w_k, dst_device, 0, local_nkh * hd, hs)
                        .map_err(ModelError::Hip)?;
                let w_v =
                    MultiGpuContext::copy_weight_slice(&w.w_v, dst_device, 0, local_nkh * hd, hs)
                        .map_err(ModelError::Hip)?;
                ctx.workers[gpu_i].attn_w_q_gate.push(w_q);
                ctx.workers[gpu_i].attn_w_k.push(w_k);
                ctx.workers[gpu_i].attn_w_v.push(w_v);
            }
        }
        braidinfer_hip::device::Device::set_current(braidinfer_core::types::DeviceId(0))
            .map_err(ModelError::Hip)?;
        eprintln!(
            "Multi-GPU: split QKV weights for {} attn layers across {} GPUs",
            attn_layer_indices.len(),
            num_gpus
        );
        Ok(())
    }
}
