use std::path::Path;

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_core::safetensors::SafeTensorSet;

use crate::megakernel::{MegakernelProgram, PrefillBuffers, CHUNK_TOKENS};
use crate::paged_kv::{self, PageAllocator, RecurrentCheckpointPool, SequenceState};

// Re-export weight types and config for backward compatibility
pub use crate::config::*;
pub use crate::weights::*;
// ---- Main model struct ----

pub struct Model {
    pub(crate) config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) lm_head_weight: DeviceBuffer<u16>,  // separate from embed when tie_word_embeddings=false
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) moe_weights: Vec<Option<MoeWeights>>,  // per-layer MoE FFN (None for dense FFN layers)
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) kv_caches: Vec<KvCache>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) mamba2_states: Vec<Mamba2State>,
    pub(crate) seq_len: u32,
    megakernel: Option<MegakernelProgram>,
    // Paged KV path (lazy-init)
    megakernel_paged: Option<MegakernelProgram>,
    page_allocator: Option<PageAllocator>,
    quant_allocator: Option<PageAllocator>,
    paged_seq: Option<SequenceState>,
    checkpoint_pool: Option<RecurrentCheckpointPool>,
    last_checkpoint_slot: Option<u32>,
    trace: Option<crate::trace::TraceWriter>,
    pub(crate) debug_nan: bool,
}

// ---- Model impl ----

impl Model {
    /// Default max_seq_len cap for flat KV cache (limits VRAM usage).
    /// Override with `load_with_max_seq_len`. Paged KV grows dynamically.
    const DEFAULT_MAX_SEQ_LEN: usize = 8192;

    pub fn load(model_dir: &Path, device: DeviceId) -> Result<Self, ModelError> {
        Self::load_with_max_seq_len(model_dir, device, None)
    }

    pub fn load_with_max_seq_len(model_dir: &Path, device: DeviceId, max_seq_len: Option<usize>) -> Result<Self, ModelError> {
        let config_path = model_dir.join("config.json");
        let mut config = if config_path.exists() {
            ModelConfig::from_config_json(&config_path)
                .map_err(|e| ModelError::MissingWeight(format!("config.json: {e}")))?
        } else {
            ModelConfig::qwen35_0_8b()
        };
        // Cap max_seq_len: model may claim 262144 but flat KV can't afford that.
        // User override takes priority, otherwise cap at DEFAULT_MAX_SEQ_LEN.
        config.max_seq_len = max_seq_len.unwrap_or(config.max_seq_len.min(Self::DEFAULT_MAX_SEQ_LEN));

        // Weight quantization mode from env var
        config.weight_quant = match std::env::var("WEIGHT_QUANT").as_deref() {
            Ok("rnf4") => WeightQuantMode::Rnf4,
            Ok("mixed") => WeightQuantMode::Mixed,
            _ => WeightQuantMode::Bf16,
        };

        let st = SafeTensorSet::open_directory(model_dir)?;

        // Check for pre-quantized .bqnt file (set BQNT_PATH env var)
        let bqnt = std::env::var("BQNT_PATH").ok()
            .and_then(|p| {
                let path = std::path::PathBuf::from(&p);
                match crate::bqnt::MmapBqnt::open(&path) {
                    Ok(b) => {
                        eprintln!("Loaded pre-quantized weights from {} ({} tensors)",
                            p, b.n_tensors());
                        Some(b)
                    }
                    Err(e) => {
                        eprintln!("WARNING: Failed to open BQNT_PATH={p}: {e}");
                        None
                    }
                }
            });

        // Pin mmap'd shard regions so hipMemcpy can DMA directly (avoids bounce buffer).
        // Costs ~300ms upfront to fault in pages, but saves ~500ms on weight copies.
        // Some models have mmap regions that fail hipHostRegister (non-page-aligned etc.);
        // track which succeeded so we only unregister those.
        let shard_ptrs: Vec<(*mut std::ffi::c_void, usize)> = st.shard_mmaps()
            .map(|m| (m.as_ptr() as *mut std::ffi::c_void, m.len()))
            .collect();
        let mut pinned: Vec<*mut std::ffi::c_void> = Vec::new();
        for &(ptr, len) in &shard_ptrs {
            let rc = unsafe { ffi::hipHostRegister(ptr, len, 0) };
            if rc == 0 {
                pinned.push(ptr);
            }
        }

        // Discover tensor name prefix by finding "layers.0." in tensor names.
        // Prefer prefixes containing "model" to avoid matching MTP/draft heads.
        let prefix = {
            let names = st.tensor_names();
            let candidates: Vec<&str> = names.iter()
                .filter(|n| n.contains("layers.0."))
                .map(|n| &n[..n.find("layers.0.").unwrap()])
                .collect();
            let prefix = candidates.iter()
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
        let first_attn_idx = config.layers.iter().position(|l| l.layer_type == LayerType::Attention);
        let has_output_gate = if let Some(ai) = first_attn_idx {
            let q_name = format!("{prefix}layers.{ai}.self_attn.q_proj.weight");
            if let Ok(raw) = st.tensor_data(&q_name) {
                let expected_gated = config.num_q_heads * config.head_dim * 2 * config.hidden_size * 2; // bf16
                raw.len() == expected_gated
            } else { false }
        } else { false };
        config.has_output_gate = has_output_gate;
        let embed_name = names.iter()
            .find(|n| n.starts_with(&prefix) && (n.contains("embed_tokens.weight") || n.contains("tok_embeddings.weight") || n.ends_with("wte.weight") || n.contains("embeddings.weight")))
            .or_else(|| names.iter().find(|n| n.contains("embed_tokens.weight") || n.contains("tok_embeddings.weight") || n.ends_with("wte.weight") || n.contains("embeddings.weight")))
            .ok_or_else(|| ModelError::MissingWeight("embedding tensor not found".into()))?
            .to_string();
        let norm_name = names.iter()
            .find(|n| n.starts_with(&prefix) && (n.ends_with("norm.weight") || n.ends_with("ln_f.weight") || n.ends_with("norm_f.weight")) && !n.contains("layers."))
            .or_else(|| names.iter().find(|n| (n.contains("norm.weight") || n.contains("ln_f.weight") || n.contains("norm_f.weight")) && !n.contains("layers.") && !n.contains("visual") && !n.contains("mtp")))
            .ok_or_else(|| ModelError::MissingWeight("final norm tensor not found".into()))?
            .to_string();

        let embed_weight = load_weight_bf16(&st, &embed_name, device, config.vocab_size * config.hidden_size)?;
        let lm_head_weight = if config.tie_word_embeddings {
            // Weight-tied: reuse embed_weight pointer (allocate a dummy — the megakernel uses embed_weight)
            DeviceBuffer::<u16>::alloc(device, 0)?  // placeholder, megakernel will use embed_weight
        } else {
            let lm_head_name = names.iter()
                .find(|n| n.contains("lm_head.weight"))
                .ok_or_else(|| ModelError::MissingWeight("lm_head.weight not found".into()))?
                .to_string();
            load_weight_bf16(&st, &lm_head_name, device, config.vocab_size * config.hidden_size)?
        };
        let final_norm_weight = load_weight_bf16(&st, &norm_name, device, config.hidden_size)?;

        // Per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut moe_weights_vec: Vec<Option<MoeWeights>> = (0..config.num_layers).map(|_| None).collect();
        for i in 0..config.num_layers {
            let p = format!("{prefix}layers.{i}.");
            let is_moe = matches!(config.layers[i].ffn_type, FfnType::MoE { .. });
            let wq = config.weight_quant;
            // Helper: load linear weight, trying bqnt first if available
            let load_lw = |name: &str, out_dim: usize, in_dim: usize| -> Result<LinearWeight, ModelError> {
                if let Some(ref b) = bqnt {
                    if let Ok(lw) = crate::weights::load_linear_weight_bqnt(b, name, device) {
                        return Ok(lw);
                    }
                }
                load_linear_weight(&st, name, device, out_dim, in_dim, wq)
            };
            let layer_type = &config.layers[i].layer_type;
            if *layer_type == LayerType::Mamba2 {
                // Mamba2 SSM layer (Nemotron-H 'M' layers)
                let hs = config.hidden_size;
                let (nh, hd, _sd, ck, _ng, cd) = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 { num_heads, head_dim, state_dim, conv_kernel, n_groups, conv_dim, .. } =>
                        (*num_heads, *head_dim, *state_dim, *conv_kernel, *n_groups, *conv_dim),
                    _ => panic!("Mamba2 layer but no Mamba2 recurrent config"),
                };
                let intermediate = nh * hd;
                let in_proj_size = intermediate + cd + nh; // gate + xBC + dt
                // Try Nemotron weight names first, then generic
                let norm_name = find_weight_name(&st, &[
                    format!("{p}norm.weight"),
                    format!("{p}input_layernorm.weight"),
                ])?;
                let w = Mamba2LayerWeights {
                    input_norm: load_weight_bf16(&st, &norm_name, device, hs)?,
                    in_proj: load_lw(&format!("{p}mixer.in_proj.weight"), in_proj_size, hs)?,
                    conv1d_weight: load_weight_bf16(&st, &format!("{p}mixer.conv1d.weight"), device, cd * ck)?,
                    conv1d_bias: load_weight_f32(&st, &format!("{p}mixer.conv1d.bias"), device, cd)?,
                    dt_bias: load_weight_f32(&st, &format!("{p}mixer.dt_bias"), device, nh)?,
                    a_log: load_weight_f32(&st, &format!("{p}mixer.A_log"), device, nh)?,
                    d: load_weight_f32(&st, &format!("{p}mixer.D"), device, nh)?,
                    norm_weight: load_weight_f32(&st, &format!("{p}mixer.norm.weight"), device, intermediate)?,
                    out_proj: load_lw(&format!("{p}mixer.out_proj.weight"), hs, intermediate)?,
                };
                layers.push(LayerWeights::Mamba2(w));
            } else if *layer_type == LayerType::MoeFfn {
                // Standalone MoE FFN layer (Nemotron-H 'E' layers)
                let hs = config.hidden_size;
                let norm_name = find_weight_name(&st, &[
                    format!("{p}norm.weight"),
                    format!("{p}input_layernorm.weight"),
                ])?;
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
                moe_weights_vec[i] = Some(load_moe_weights(&st, &moe_prefix, &config, &config.layers[i].ffn_type, device, wq, bqnt.as_ref())?);
            } else if config.layers[i].layer_type == LayerType::Attention {
                let hs = config.hidden_size;
                let q_mult = if has_output_gate { 2 } else { 1 };
                let w = AttentionLayerWeights {
                    input_norm: load_weight_bf16(&st, &find_weight_name(&st, &[
                        format!("{p}input_layernorm.weight"),
                        format!("{p}norm.weight"),
                    ])?, device, hs)?,
                    w_q_gate: load_lw(&find_weight_name(&st, &[
                        format!("{p}self_attn.q_proj.weight"),
                        format!("{p}mixer.q_proj.weight"),
                    ])?, config.num_q_heads * config.head_dim * q_mult, hs)?,
                    w_k: load_lw(&find_weight_name(&st, &[
                        format!("{p}self_attn.k_proj.weight"),
                        format!("{p}mixer.k_proj.weight"),
                    ])?, config.num_kv_heads * config.head_dim, hs)?,
                    w_v: load_lw(&find_weight_name(&st, &[
                        format!("{p}self_attn.v_proj.weight"),
                        format!("{p}mixer.v_proj.weight"),
                    ])?, config.num_kv_heads * config.head_dim, hs)?,
                    w_o: load_lw(&find_weight_name(&st, &[
                        format!("{p}self_attn.o_proj.weight"),
                        format!("{p}mixer.o_proj.weight"),
                    ])?, hs, config.num_q_heads * config.head_dim)?,
                    q_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.q_norm.weight");
                        let raw = st.tensor_data(&name).map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    k_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.k_norm.weight");
                        let raw = st.tensor_data(&name).map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    post_norm: {
                        let name = find_weight_name(&st, &[
                            format!("{p}post_attention_layernorm.weight"),
                        ]);
                        if let Ok(n) = name { load_weight_bf16(&st, &n, device, hs)? }
                        else { DeviceBuffer::<u16>::alloc(device, 0)? } // no post-norm (Nemotron * layers)
                    },
                    w_gate: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(&format!("{p}mlp.gate_proj.weight"),
                            config.intermediate_size, hs)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                    w_up: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(&format!("{p}mlp.up_proj.weight"),
                            config.intermediate_size, hs)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                    w_down: if !is_moe && !matches!(config.layers[i].ffn_type, FfnType::None) {
                        load_lw(&format!("{p}mlp.down_proj.weight"),
                            hs, config.intermediate_size)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                };
                layers.push(LayerWeights::Attention(w));

                // Load MoE weights if this layer uses MoE FFN
                if is_moe {
                    moe_weights_vec[i] = Some(load_moe_weights(&st, &p, &config, &config.layers[i].ffn_type, device, wq, bqnt.as_ref())?);
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
                let conv_raw_bytes = st.tensor_data(&conv_name)
                    .map_err(|_| ModelError::MissingWeight(conv_name.clone()))?;
                assert_eq!(conv_raw_bytes.len(), conv_total * 2);
                let conv_raw: &[u16] = unsafe {
                    std::slice::from_raw_parts(conv_raw_bytes.as_ptr() as *const u16, conv_total)
                };
                let mut conv1d_weight_buf = DeviceBuffer::<u16>::alloc(device, conv_total)?;
                conv1d_weight_buf.copy_from_host(conv_raw)?;
                let mut conv_w_q_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_k_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_v_buf = DeviceBuffer::<u16>::alloc(device, v_dim * ck)?;
                conv_w_q_buf.copy_from_host(&conv_raw[..q_dim * ck])?;
                conv_w_k_buf.copy_from_host(&conv_raw[q_dim * ck..2 * q_dim * ck])?;
                conv_w_v_buf.copy_from_host(&conv_raw[2 * q_dim * ck..])?;
                let hs = config.hidden_size;
                let w = GdnLayerWeights {
                    input_norm: load_weight_bf16(&st, &format!("{p}input_layernorm.weight"), device, hs)?,
                    w_qkv: load_lw(&format!("{p}linear_attn.in_proj_qkv.weight"), qkv_out, hs)?,
                    w_a: load_lw(&format!("{p}linear_attn.in_proj_a.weight"), nvh, hs)?,
                    w_b: load_lw(&format!("{p}linear_attn.in_proj_b.weight"), nvh, hs)?,
                    w_z: load_lw(&format!("{p}linear_attn.in_proj_z.weight"), z_out, hs)?,
                    conv1d_weight: conv1d_weight_buf,
                    conv1d_weight_q: conv_w_q_buf,
                    conv1d_weight_k: conv_w_k_buf,
                    conv1d_weight_v: conv_w_v_buf,
                    a_log: load_weight_f32(&st, &format!("{p}linear_attn.A_log"), device, nvh)?,
                    dt_bias: load_weight_bf16(&st, &format!("{p}linear_attn.dt_bias"), device, nvh)?,
                    output_norm: load_weight_f32(&st, &format!("{p}linear_attn.norm.weight"), device, vd)?,  // normalizes [nvh, vd] output
                    w_out: load_lw(&format!("{p}linear_attn.out_proj.weight"), hs, z_out)?,
                    post_norm: load_weight_bf16(&st, &format!("{p}post_attention_layernorm.weight"), device, hs)?,
                    w_gate: if !is_moe {
                        load_lw(&format!("{p}mlp.gate_proj.weight"), config.intermediate_size, hs)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                    w_up: if !is_moe {
                        load_lw(&format!("{p}mlp.up_proj.weight"), config.intermediate_size, hs)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                    w_down: if !is_moe {
                        load_lw(&format!("{p}mlp.down_proj.weight"), hs, config.intermediate_size)?
                    } else { LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?) },
                };
                layers.push(LayerWeights::Gdn(w));

                // Load MoE weights for GDN layers with MoE FFN (e.g. Qwen3.5-122B)
                if is_moe {
                    moe_weights_vec[i] = Some(load_moe_weights(&st, &p, &config, &config.layers[i].ffn_type, device, wq, bqnt.as_ref())?);
                }
            }
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
        if let RecurrentLayerKind::Mamba2 { num_heads: m_nh, head_dim: m_hd, state_dim: m_sd, conv_kernel: m_ck, conv_dim: m_cd, .. } = &config.recurrent_kind {
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

        let mut pos_buf = DeviceBuffer::<i32>::alloc(device, 3)?;
        pos_buf.copy_from_host(&[0i32, 0, 0])?;

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
            q_gate_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd * if config.has_output_gate { 2 } else { 1 })?,
            q_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            gate_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            k_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            v_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            attn_out: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            gated_out: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            ffn_gate: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_up: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_act: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_down: DeviceBuffer::<f32>::alloc(device, hs)?,
            residual: DeviceBuffer::<f32>::alloc(device, hs)?,
            logits: DeviceBuffer::<f32>::alloc(device, vs)?,
            inv_freq: inv_freq_buf,
            position_ids: pos_buf,
            gdn_cs_q: DeviceBuffer::<f32>::alloc(device, nh * kd * (ck - 1))?,
            gdn_cs_k: DeviceBuffer::<f32>::alloc(device, nh * kd * (ck - 1))?,
            gdn_cs_v: DeviceBuffer::<f32>::alloc(device, nvh * vd * (ck - 1))?,
            gdn_conv_out_q: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_k: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_v: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            // MoE scratch: sized for per-layer max expert dimensions
            moe_scores: DeviceBuffer::<f32>::alloc(device, config.layers.iter().filter_map(|l| match &l.ffn_type {
                FfnType::MoE { num_experts, .. } => Some(*num_experts), _ => None
            }).max().unwrap_or(1))?,
            moe_expert_gate: DeviceBuffer::<f32>::alloc(device, config.layers.iter().filter_map(|l| match &l.ffn_type {
                FfnType::MoE { expert_intermediate_size, shared_intermediate_size, .. } =>
                    Some((*expert_intermediate_size).max(*shared_intermediate_size)), _ => None
            }).max().unwrap_or(1))?,
            moe_expert_up: DeviceBuffer::<f32>::alloc(device, config.layers.iter().filter_map(|l| match &l.ffn_type {
                FfnType::MoE { expert_intermediate_size, shared_intermediate_size, .. } =>
                    Some((*expert_intermediate_size).max(*shared_intermediate_size)), _ => None
            }).max().unwrap_or(1))?,
            moe_expert_act: DeviceBuffer::<f32>::alloc(device, config.layers.iter().filter_map(|l| match &l.ffn_type {
                FfnType::MoE { expert_intermediate_size, shared_intermediate_size, .. } =>
                    Some((*expert_intermediate_size).max(*shared_intermediate_size)), _ => None
            }).max().unwrap_or(1))?,
            moe_expert_out: DeviceBuffer::<f32>::alloc(device, hs)?,
            // Mamba2 scratch: sized from recurrent_kind if Mamba2
            mamba2_in_proj: {
                let size = match &config.recurrent_kind {
                    RecurrentLayerKind::Mamba2 { num_heads, head_dim, conv_dim, .. } =>
                        num_heads * head_dim + conv_dim + num_heads, // gate + xBC + dt
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
                    RecurrentLayerKind::Mamba2 { num_heads, head_dim, .. } => num_heads * head_dim,
                    _ => 1,
                };
                DeviceBuffer::<f32>::alloc(device, size)?
            },
            argmax_result: DeviceBuffer::<i32>::alloc(device, 1)?,
        };

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
            kv_caches,
            gdn_states,
            mamba2_states,
            seq_len: 0,
            megakernel: None,
            megakernel_paged: None,
            page_allocator: None,
            quant_allocator: None,
            paged_seq: None,
            checkpoint_pool: None,
            last_checkpoint_slot: None,
            trace: std::env::var("TRACE").ok().and_then(|path| {
                crate::trace::TraceWriter::open(&path).ok()
            }),
            debug_nan: std::env::var("DEBUG_NAN").is_ok(),
        })
    }

    fn gdn_forward(
        &mut self,
        layer_idx: usize,
        gdn_idx: usize,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nh = cfg.linear_num_heads as u32;
        let nvh = cfg.linear_num_value_heads as u32;
        let kd = cfg.linear_key_head_dim as u32;
        let vd = cfg.linear_value_head_dim as u32;
        let ck = cfg.linear_conv_kernel_dim as u32;
        let eps = cfg.rms_norm_eps;

        let weights = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => panic!("expected GDN layer at {layer_idx}"),
        };

        // 1. RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &weights.input_norm,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;

        // 2. Project QKV [6144]
        weights.w_qkv.forward(&self.kernels.linear_proj,
            &mut self.activations.qkv, &self.activations.normed,
            nh * kd * 2 + nvh * vd, hs, &self.stream)?;

        // 3. Project a [nvh], b [nvh], z [nvh*vd]
        weights.w_a.forward(&self.kernels.linear_proj,
            &mut self.activations.a_proj, &self.activations.normed, nvh, hs, &self.stream)?;
        weights.w_b.forward(&self.kernels.linear_proj,
            &mut self.activations.b_proj, &self.activations.normed, nvh, hs, &self.stream)?;
        weights.w_z.forward(&self.kernels.linear_proj,
            &mut self.activations.z_proj, &self.activations.normed, nvh * vd, hs, &self.stream)?;

        // 4. Causal conv1d: split qkv into q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, nh as usize * kd as usize, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, nh as usize * kd as usize * 2, nvh as usize * vd as usize, &self.stream)?;
        }

        let conv_q_out_len = nh as usize * kd as usize;
        let conv_k_out_len = nh as usize * kd as usize;
        let conv_v_out_len = nvh as usize * vd as usize;
        let ck_usize = ck as usize;

        // Split conv state into q/k/v sub-states
        // gdn_conv_states[gdn_idx] is [6144, ck-1] = [6144 * (ck-1)].
        // Split into 3 sub-states: q=[2048,ck-1], k=[2048,ck-1], v=[2048,ck-1].
        let conv_state_q_len = conv_q_out_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_out_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_out_len * (ck_usize - 1);

        unsafe {
            d2d_copy_f32(&mut self.activations.gdn_cs_q, 0, &self.gdn_conv_states[gdn_idx], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_k, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_v, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
        }

        // Run 3 conv1d operations using pre-split weight buffers from the layer
        // SAFETY: Raw pointers break the borrow on self.layers so we can mutably access
        // self.activations. The pointers remain valid because layers[layer_idx] is not
        // modified or moved during this function call.
        let (conv_w_q_ptr, conv_w_k_ptr, conv_w_v_ptr) = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => (
                &w.conv1d_weight_q as *const DeviceBuffer<u16>,
                &w.conv1d_weight_k as *const DeviceBuffer<u16>,
                &w.conv1d_weight_v as *const DeviceBuffer<u16>,
            ),
            _ => unreachable!(),
        };
        unsafe {
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_q,
                &self.activations.q_gdn,
                &*conv_w_q_ptr,
                &mut self.activations.gdn_conv_out_q,
                conv_q_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_k,
                &self.activations.k_gdn,
                &*conv_w_k_ptr,
                &mut self.activations.gdn_conv_out_k,
                conv_k_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_v,
                &self.activations.v_gdn,
                &*conv_w_v_ptr,
                &mut self.activations.gdn_conv_out_v,
                conv_v_out_len as u32,
                ck,
                &self.stream,
            )?;
        }

        // Write back updated conv states
        unsafe {
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], 0, &self.activations.gdn_cs_q, 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len, &self.activations.gdn_cs_k, 0, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, &self.activations.gdn_cs_v, 0, conv_state_v_len, &self.stream)?;
        }

        // conv_out_q/k/v now hold the post-conv Q,K,V (with SiLU applied inside the kernel)
        // Copy them back to q_gdn, k_gdn, v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.gdn_conv_out_q, 0, conv_q_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.gdn_conv_out_k, 0, conv_k_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.gdn_conv_out_v, 0, conv_v_out_len, &self.stream)?;
        }

        // 5. Compute GDN gate
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &weights_gdn.a_log,
            &self.activations.a_proj,
            &weights_gdn.dt_bias,
            nvh,
            &self.stream,
        )?;

        // 6. GDN recurrent step v2 (nvh heads, GQA group = nvh/nh)
        let gqa_group = nvh / nh;
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[gdn_idx].recurrent,
            &mut self.activations.recurrent_out,
            nvh,
            kd,
            vd,
            gqa_group,
            &self.stream,
        )?;

        // 7. RMSNorm gated (recurrent_out with z gate)
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &weights_gdn.output_norm,
            nvh,  // value heads, not key heads
            vd,
            eps,
            &self.stream,
        )?;

        // 8. Output projection [1024, 2048]
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        weights_gdn.w_out.forward(&self.kernels.linear_proj,
            &mut self.activations.out_proj, &self.activations.normed_gated,
            hs, nvh * vd,  // value heads, not key heads
            &self.stream,
        )?;

        // 9. Residual add: hidden = out_proj + hidden
        // Copy hidden to residual first, then add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        // 10. FFN — extract raw pointers to avoid borrow conflict with &mut self
        Ok(())
    }

    fn ffn_forward(
        &mut self,
        post_norm: &DeviceBuffer<u16>,
        w_gate: &DeviceBuffer<u16>,
        w_up: &DeviceBuffer<u16>,
        w_down: &DeviceBuffer<u16>,
    ) -> HipResult<()> {
        let hs = self.config.hidden_size as u32;
        let is = self.config.intermediate_size as u32;
        let eps = self.config.rms_norm_eps;

        self.kernels.ffn_fused.forward_gate_up(
            &mut self.activations.ffn_act,
            &self.activations.hidden,
            post_norm,
            w_gate,
            w_up,
            hs,
            is,
            eps,
            &self.stream,
        )?;

        // Save residual
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.ffn_fused.forward_down_residual(
            &mut self.activations.hidden,
            &self.activations.residual,
            w_down,
            &self.activations.ffn_act,
            hs,
            is,
            &self.stream,
        )
    }

    /// MoE FFN forward: route to top-k experts, run expert FFNs, combine.
    /// Uses individual kernel launches (no megakernel).
    fn moe_ffn_forward(&mut self, layer_idx: usize) -> HipResult<()> {
        let moe = self.moe_weights[layer_idx].as_ref()
            .expect("moe_ffn_forward called on non-MoE layer");
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let ne = moe.num_experts;
        let eps = self.config.rms_norm_eps;

        // SAFETY: Raw pointer breaks borrow on self.layers to allow mutable access to
        // self.activations. Pointer valid for duration of this function (layers not modified).
        let norm_weight = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::MoeFfn(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("no norm weight for this layer type in MoE FFN"),
        };

        // 1. RMSNorm(hidden) → normed
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*norm_weight,
                1, hs as u32, eps, self.config.rms_norm_one_plus_w, &self.stream,
            )?;
        }

        // Save residual
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
        }

        // 2. Gate projection: normed → scores[num_experts]
        self.kernels.linear_proj.forward(
            &mut self.activations.moe_scores,
            &moe.gate,
            &self.activations.normed,
            ne as u32, hs as u32, &self.stream,
        )?;
        self.stream.synchronize()?;

        // 3. Read scores to CPU and do top-k selection
        let mut scores = vec![0.0f32; ne];
        self.activations.moe_scores.copy_to_host(&mut scores)?;

        let k = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { num_active, .. } => *num_active,
            _ => unreachable!(),
        };

        // Score transformation depends on gate type
        let gate_type = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { gate_type, .. } => gate_type.clone(),
            _ => unreachable!(),
        };

        let topk = match &gate_type {
            GateType::Sigmoid { .. } => {
                // Sigmoid scoring (Nemotron): sigmoid(logits) for weights,
                // sigmoid(logits) + correction_bias for expert SELECTION only
                let sigmoid_scores: Vec<f32> = scores.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect();
                // Selection uses biased scores
                let mut selection_scores = sigmoid_scores.clone();
                if let Some(ref bias) = moe.score_correction_bias {
                    for (s, b) in selection_scores.iter_mut().zip(bias.iter()) {
                        *s += b;
                    }
                }
                let mut indexed: Vec<(usize, f32)> = selection_scores.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                // Weights come from UN-BIASED sigmoid scores
                let topk: Vec<(usize, f32)> = indexed[..k].iter()
                    .map(|&(id, _)| (id, sigmoid_scores[id])).collect();
                topk
            }
            _ => {
                // Softmax scoring (+ optional correction bias for selection)
                if let Some(ref bias) = moe.score_correction_bias {
                    for (s, b) in scores.iter_mut().zip(bias.iter()) {
                        *s += b;
                    }
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                let probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp() / exp_sum).collect();
                let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed[..k].to_vec()
            }
        };

        let (weights, scaling) = match &gate_type {
            GateType::NormTopK { routed_scaling_factor } | GateType::Sigmoid { routed_scaling_factor } => {
                // NormTopK/Sigmoid: renormalize selected weights to sum to 1, then scale
                let sum: f32 = topk.iter().map(|(_, w)| w).sum();
                let w: Vec<f32> = topk.iter().map(|(_, w)| w / sum).collect();
                (w, *routed_scaling_factor)
            }
            GateType::Softmax => {
                // Standard softmax: use raw probabilities (don't renormalize)
                let w: Vec<f32> = topk.iter().map(|(_, w)| *w).collect();
                (w, 1.0)
            }
        };

        // 4. Zero accumulation buffer on GPU (no CPU round-trip)
        unsafe {
            let rc = braidinfer_hip::ffi::hipMemsetAsync(
                self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
                0, hs * 4, self.stream.raw(),
            );
            if rc != 0 {
                return Err(braidinfer_hip::HipError(rc).into());
            }
        }

        // Debug MoE routing
        if self.debug_nan && layer_idx <= 3 {
            eprintln!("  MoE L{layer_idx}: topk={:?} weights={:?} scaling={scaling:.4}",
                topk.iter().map(|(id,_)| *id).collect::<Vec<_>>(), &weights);
        }

        // 5. For each selected expert: run FFN and GPU-accumulate
        for (j, &(expert_id, _)) in topk.iter().enumerate() {
            let w = weights[j] * scaling;

            let down_offset = moe.expert_down.row_byte_offset_dim(expert_id * hs, eis);

            if moe.has_gate_proj {
                // SwiGLU: gate_proj → silu → * up_proj
                let gate_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * 2 * eis, hs);
                let up_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * 2 * eis + eis, hs);

                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_gate.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, gate_offset, &self.stream,
                )?;
                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_up.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, up_offset, &self.stream,
                )?;
                self.kernels.silu_mul.forward(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_gate,
                    &self.activations.moe_expert_up,
                    eis as u32, &self.stream,
                )?;
            } else {
                // relu²: up_proj → relu² (no gate_proj)
                let up_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * eis, hs);
                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_up.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, up_offset, &self.stream,
                )?;
                self.kernels.silu_mul.relu_squared(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_up,
                    eis as u32, &self.stream,
                )?;
            }

            // Debug: check intermediate values
            if self.debug_nan && layer_idx <= 1 && j == 0 {
                self.stream.synchronize()?;
                let up_len = self.activations.moe_expert_up.len();
                let act_len = self.activations.moe_expert_act.len();
                let mut up_buf = vec![0.0f32; up_len];
                let mut act_buf = vec![0.0f32; act_len];
                self.activations.moe_expert_up.copy_to_host(&mut up_buf)?;
                self.activations.moe_expert_act.copy_to_host(&mut act_buf)?;
                let up_max = up_buf[..eis].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let act_max = act_buf[..eis].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("  Expert {expert_id}: up_max={up_max:.4}, act_max(relu²)={act_max:.4}, w={w:.6}");
            }

            // Down projection (pre-allocated buffer)
            moe.expert_down.forward_sub(
                &self.kernels.linear_proj,
                self.activations.moe_expert_out.as_mut_ptr(),
                self.activations.moe_expert_act.as_ptr(),
                hs as u32, eis as u32, down_offset, &self.stream,
            )?;

            // GPU-side weighted accumulate: ffn_down += w * expert_out
            self.kernels.residual_add.weighted_accumulate(
                &mut self.activations.ffn_down,
                &self.activations.moe_expert_out,
                w,
                hs as u32,
                &self.stream,
            )?;
        }

        // 6. Shared expert (always-on, added to output)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &self.config.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } =>
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size },
                _ => eis,
            };
            // Shared expert receives the same normed input as routed experts.
            // HF NemotronHMOE: residuals = hidden_states (already normed by block)
            se.up_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.moe_expert_up, &self.activations.normed,
                se_is as u32, hs as u32, &self.stream)?;

            if moe.has_gate_proj {
                // SwiGLU shared expert
                se.gate_proj.forward(&self.kernels.linear_proj,
                    &mut self.activations.moe_expert_gate, &self.activations.normed,
                    se_is as u32, hs as u32, &self.stream)?;
                self.kernels.silu_mul.forward(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_gate,
                    &self.activations.moe_expert_up,
                    se_is as u32, &self.stream)?;
            } else {
                // relu² shared expert
                self.kernels.silu_mul.relu_squared(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_up,
                    se_is as u32, &self.stream)?;
            }

            se.down_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.moe_expert_out, &self.activations.moe_expert_act,
                hs as u32, se_is as u32, &self.stream)?;

            // Apply shared expert gate: sigmoid(gate @ input) * shared_output
            let se_weight = if let Some(ref gate_buf) = moe.shared_expert_gate {
                // Compute sigmoid(gate @ normed_input) on CPU
                self.stream.synchronize()?;
                let mut normed = vec![0.0f32; hs];
                self.activations.normed.copy_to_host(&mut normed)?;
                let mut gate_w = vec![0u16; hs];
                gate_buf.copy_to_host(&mut gate_w)?;
                let dot: f32 = normed.iter().zip(gate_w.iter())
                    .map(|(&x, &w)| x * f32::from_bits((w as u32) << 16))
                    .sum();
                1.0 / (1.0 + (-dot).exp()) // sigmoid
            } else {
                1.0
            };
            self.kernels.residual_add.weighted_accumulate(
                &mut self.activations.ffn_down,
                &self.activations.moe_expert_out,
                se_weight,
                hs as u32, &self.stream)?;
        }

        // Debug: check ffn_down magnitude
        if self.debug_nan && layer_idx <= 3 {
            self.stream.synchronize()?;
            let ffn_len = self.activations.ffn_down.len();
            let mut buf = vec![0.0f32; ffn_len];
            self.activations.ffn_down.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  MoE L{layer_idx} ffn_down(len={ffn_len}) max_abs={max_abs:.4}");
        }

        // 7. Residual add: hidden = residual + ffn_down
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.residual,
            &self.activations.ffn_down,
            hs as u32, &self.stream,
        )?;

        Ok(())
    }

    /// Mamba2 SSM layer forward pass (Nemotron-H 'M' layers).
    /// Steps: norm → in_proj → split → conv1d → split → ssm_update → norm_gated → out_proj → residual
    fn mamba2_forward(&mut self, layer_idx: usize, mamba2_idx: usize) -> Result<(), ModelError> {
        let w = match &self.layers[layer_idx] {
            LayerWeights::Mamba2(w) => w as *const Mamba2LayerWeights,
            _ => panic!("mamba2_forward called on non-Mamba2 layer"),
        };
        let (nh, hd, sd, _ck, ng, cd) = match &self.config.recurrent_kind {
            RecurrentLayerKind::Mamba2 { num_heads, head_dim, state_dim, conv_kernel, n_groups, conv_dim, .. } =>
                (*num_heads, *head_dim, *state_dim, *conv_kernel, *n_groups, *conv_dim),
            _ => panic!("mamba2_forward but no Mamba2 config"),
        };
        let hs = self.config.hidden_size as u32;
        let intermediate = (nh * hd) as u32;
        let in_proj_size = (nh * hd + cd + nh) as u32;
        let eps = self.config.rms_norm_eps;

        // 1. RMSNorm
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed, &self.activations.hidden,
                &(*w).input_norm, 1, hs, eps,
                self.config.rms_norm_one_plus_w, &self.stream,
            )?;
        }

        // Debug: per-step tracing for layer 0
        let dbg = self.debug_nan && layer_idx == 0;
        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.normed.len();
            let mut buf = vec![0.0f32; n];
            self.activations.normed.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 after norm: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 2. in_proj: normed → [gate(intermediate), xBC(conv_dim), dt(num_heads)]
        unsafe {
            (*w).in_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.mamba2_in_proj, &self.activations.normed,
                in_proj_size, hs, &self.stream)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_in_proj.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_in_proj.copy_to_host(&mut buf)?;
            let gate_max = buf[..nh*hd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let xbc_max = buf[nh*hd..nh*hd+cd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let dt_max = buf[nh*hd+cd..].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 in_proj: gate_max={gate_max:.4e}, xBC_max={xbc_max:.4e}, dt_max={dt_max:.4e}");
        }

        // 3. Conv1d update on xBC with bias + silu activation
        // Input is mamba2_in_proj[intermediate..intermediate+cd], output to mamba2_conv_out
        {
            let state = &mut self.mamba2_states[mamba2_idx];
            let func = self.kernels.causal_conv1d.module.get_function("causal_conv1d_update_bias_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.conv.as_mut_ptr().cast();
            let mut in_ptr: *const std::ffi::c_void = unsafe {
                self.activations.mamba2_in_proj.as_ptr().add(nh * hd).cast()
            };
            let mut w_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_weight.as_ptr().cast() };
            let mut bias_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_bias.as_ptr().cast() };
            let mut out_ptr: *mut std::ffi::c_void = self.activations.mamba2_conv_out.as_mut_ptr().cast();
            let mut i_cd = cd as i32;
            let mut i_ck = _ck as i32;
            let mut args: [*mut std::ffi::c_void; 7] = [
                std::ptr::addr_of_mut!(state_ptr).cast(),
                std::ptr::addr_of_mut!(in_ptr).cast(),
                std::ptr::addr_of_mut!(w_ptr).cast(),
                std::ptr::addr_of_mut!(bias_ptr).cast(),
                std::ptr::addr_of_mut!(out_ptr).cast(),
                std::ptr::addr_of_mut!(i_cd).cast(),
                std::ptr::addr_of_mut!(i_ck).cast(),
            ];
            let block_size = 256u32;
            let grid_size = (cd as u32 + block_size - 1) / block_size;
            func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, &self.stream, &mut args)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_conv_out.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_conv_out.copy_to_host(&mut buf)?;
            let x_max = buf[..nh*hd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let b_max = buf[nh*hd..nh*hd+ng*sd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let c_max = buf[nh*hd+ng*sd..].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 conv1d: x_max={x_max:.4e}, B_max={b_max:.4e}, C_max={c_max:.4e}");
        }

        // 4. Split conv_out → x[intermediate], B[ng*sd], C[ng*sd]
        // x = conv_out[0..intermediate], B = conv_out[intermediate..intermediate+ng*sd], C = conv_out[intermediate+ng*sd..]
        // dt = mamba2_in_proj[intermediate+cd..intermediate+cd+nh]

        // 5. selective_state_update
        let state = &mut self.mamba2_states[mamba2_idx];
        unsafe {
            let x_ptr = self.activations.mamba2_conv_out.as_ptr();
            let b_ptr = self.activations.mamba2_conv_out.as_ptr().add(nh * hd);
            let c_ptr = self.activations.mamba2_conv_out.as_ptr().add(nh * hd + ng * sd);
            let dt_ptr = self.activations.mamba2_in_proj.as_ptr().add(nh * hd + cd);

            // Create temporary DeviceBuffer wrappers pointing to sub-regions
            // We need to call the kernel with raw pointers
            let func = self.kernels.ssm_update.module.get_function("selective_state_update_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.ssm.as_mut_ptr().cast();
            let mut x_p: *const std::ffi::c_void = x_ptr.cast();
            let mut dt_p: *const std::ffi::c_void = dt_ptr.cast();
            let mut dt_bias_p: *const std::ffi::c_void = (*w).dt_bias.as_ptr().cast();
            let mut a_log_p: *const std::ffi::c_void = (*w).a_log.as_ptr().cast();
            let mut b_p: *const std::ffi::c_void = b_ptr.cast();
            let mut c_p: *const std::ffi::c_void = c_ptr.cast();
            let mut d_p: *const std::ffi::c_void = (*w).d.as_ptr().cast();
            let mut out_p: *mut std::ffi::c_void = self.activations.mamba2_ssm_out.as_mut_ptr().cast();
            let mut i_nh = nh as i32;
            let mut i_hd = hd as i32;
            let mut i_sd = sd as i32;
            let mut i_ng = ng as i32;

            let mut args: [*mut std::ffi::c_void; 13] = [
                std::ptr::addr_of_mut!(state_ptr).cast(),
                std::ptr::addr_of_mut!(x_p).cast(),
                std::ptr::addr_of_mut!(dt_p).cast(),
                std::ptr::addr_of_mut!(dt_bias_p).cast(),
                std::ptr::addr_of_mut!(a_log_p).cast(),
                std::ptr::addr_of_mut!(b_p).cast(),
                std::ptr::addr_of_mut!(c_p).cast(),
                std::ptr::addr_of_mut!(d_p).cast(),
                std::ptr::addr_of_mut!(out_p).cast(),
                std::ptr::addr_of_mut!(i_nh).cast(),
                std::ptr::addr_of_mut!(i_hd).cast(),
                std::ptr::addr_of_mut!(i_sd).cast(),
                std::ptr::addr_of_mut!(i_ng).cast(),
            ];

            func.launch(
                (nh as u32, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &mut args,
            )?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_ssm_out.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_ssm_out.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 ssm_out: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 6. rmsnorm_gated: normed_out = rmsnorm(ssm_out) * silu(gate)
        // Mamba2 uses per-group norm (group_size=512, n_groups=8).
        // The rmsnorm_gated kernel applies weight[0..group_size] per group,
        // but Mamba2 needs weight[g*group_size..(g+1)*group_size] per group.
        // Launch one kernel per group with offset pointers.
        let group_size = (nh * hd / ng) as u32;
        {
            let func = self.kernels.rmsnorm_gated.module.get_function("rmsnorm_gated_f32")?;
            for g in 0..ng {
                let off = g * group_size as usize;
                let mut out_p: *mut std::ffi::c_void = unsafe { self.activations.mamba2_conv_out.as_mut_ptr().add(off).cast() };
                let mut x_p: *const std::ffi::c_void = unsafe { self.activations.mamba2_ssm_out.as_ptr().add(off).cast() };
                let mut z_p: *const std::ffi::c_void = unsafe { self.activations.mamba2_in_proj.as_ptr().add(off).cast() };
                let mut w_p: *const std::ffi::c_void = unsafe { (*w).norm_weight.as_ptr().add(off).cast() };
                let mut i_nh = 1i32;
                let mut i_vd = group_size as i32;
                let mut f_eps = eps;
                let mut args: [*mut std::ffi::c_void; 7] = [
                    std::ptr::addr_of_mut!(out_p).cast(),
                    std::ptr::addr_of_mut!(x_p).cast(),
                    std::ptr::addr_of_mut!(z_p).cast(),
                    std::ptr::addr_of_mut!(w_p).cast(),
                    std::ptr::addr_of_mut!(i_nh).cast(),
                    std::ptr::addr_of_mut!(i_vd).cast(),
                    std::ptr::addr_of_mut!(f_eps).cast(),
                ];
                func.launch((1, 1, 1), (256, 1, 1), 256 * 4, &self.stream, &mut args)?;
            }
        }

        // 7. out_proj: normed_out → output[hidden_size]
        unsafe {
            (*w).out_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.out_proj, &self.activations.mamba2_conv_out,
                hs, intermediate, &self.stream)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.out_proj.len();
            let mut buf = vec![0.0f32; n.min(self.config.hidden_size)];
            self.activations.out_proj.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 out_proj: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 8. Residual add
        unsafe {
            crate::model::d2d_copy_f32(
                &mut self.activations.residual, 0,
                &self.activations.hidden, 0,
                self.config.hidden_size, &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs, &self.stream,
        )?;

        Ok(())
    }

    fn attention_forward(
        &mut self,
        layer_idx: usize,
        kv_cache_idx: usize,
        position: u32,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nqh = cfg.num_q_heads as u32;
        let nkh = cfg.num_kv_heads as u32;
        let hd = cfg.head_dim as u32;
        let rd = cfg.rope_dim as u32;
        let s0 = cfg.mrope_section[0] as u32;
        let s1 = cfg.mrope_section[1] as u32;
        let s2 = cfg.mrope_section[2] as u32;
        let eps = cfg.rms_norm_eps;
        let max_sl = cfg.max_seq_len as u32;

        // 1. RMSNorm
        // SAFETY: Raw pointer breaks borrow on self.layers for mutable self.activations access.
        // Pointer valid: layers not modified during attention_forward.
        let input_norm = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("expected attention layer"),
        };
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*input_norm,
                1,
                hs,
                eps,
                cfg.rms_norm_one_plus_w,
                &self.stream,
            )?;
        }


        // 2. Project Q+gate, K, V
        // Use raw pointers to LinearWeight to work around borrow checker
        // (self.layers borrows self, but we need &mut self.activations)
        let (w_q_gate_p, w_k_p, w_v_p, w_o_p, q_norm_w, k_norm_w) =
            match &self.layers[layer_idx] {
                LayerWeights::Attention(w) => (
                    &w.w_q_gate as *const LinearWeight,
                    &w.w_k as *const LinearWeight,
                    &w.w_v as *const LinearWeight,
                    &w.w_o as *const LinearWeight,
                    &w.q_norm as *const DeviceBuffer<u16>,
                    &w.k_norm as *const DeviceBuffer<u16>,
                ),
                _ => unreachable!(),
            };

        let q_mult = if cfg.has_output_gate { 2u32 } else { 1 };
        unsafe {
            (*w_q_gate_p).forward(&self.kernels.linear_proj,
                &mut self.activations.q_gate_attn, &self.activations.normed,
                nqh * hd * q_mult, hs, &self.stream)?;
            (*w_k_p).forward(&self.kernels.linear_proj,
                &mut self.activations.k_attn, &self.activations.normed,
                nkh * hd, hs, &self.stream)?;
            (*w_v_p).forward(&self.kernels.linear_proj,
                &mut self.activations.v_attn, &self.activations.normed,
                nkh * hd, hs, &self.stream)?;
        }


        // 3. Split q_gate_attn → q, gate (gated) or just copy (non-gated)
        let hd_usize = hd as usize;
        if cfg.has_output_gate {
            unsafe {
                for h in 0..nqh as usize {
                    let src_q = h * hd_usize * 2;
                    let src_g = h * hd_usize * 2 + hd_usize;
                    let dst = h * hd_usize;
                    d2d_copy_f32(&mut self.activations.q_attn, dst, &self.activations.q_gate_attn, src_q, hd_usize, &self.stream)?;
                    d2d_copy_f32(&mut self.activations.gate_attn, dst, &self.activations.q_gate_attn, src_g, hd_usize, &self.stream)?;
                }
            }
        } else {
            // Non-gated: q_gate_attn IS q_attn, just copy
            let total = nqh as usize * hd_usize;
            unsafe {
                d2d_copy_f32(&mut self.activations.q_attn, 0, &self.activations.q_gate_attn, 0, total, &self.stream)?;
            }
        }


        // 4a. Write K,V to cache BEFORE QK-norm (pre-norm K has full dynamic range
        //     for quantization; post-norm K is bounded ±0.06 which destroys Q4 quantization).
        //     See exterior_algebra kb-20260328-115542-e172dd.
        {
            let max_sl = self.config.max_seq_len;
            for h in 0..nkh as usize {
                let src_off = h * hd as usize;
                let dst_off = h * max_sl * hd as usize + position as usize * hd as usize;
                unsafe {
                    d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].k, dst_off, &self.activations.k_attn, src_off, hd as usize, &self.stream)?;
                    d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].v, dst_off, &self.activations.v_attn, src_off, hd as usize, &self.stream)?;
                }
            }
        }

        // 4b. QK norm (in-place on q_attn, k_attn — for current token's attention computation)
        if cfg.has_qk_norm {
            let q_norm_len = unsafe { (*q_norm_w).len() };
            if q_norm_len == hd as usize {
                // Per-head QK norm (Qwen3.5 style): weight is [head_dim]
                unsafe {
                    self.kernels.qk_norm.forward(
                        &mut self.activations.q_attn,
                        &mut self.activations.k_attn,
                        &*q_norm_w,
                        &*k_norm_w,
                        nqh,
                        nkh,
                        hd,
                        eps,
                        &self.stream,
                    )?;
                }
            } else {
                // Full-hidden QK norm (OLMoE style): weight is [hidden_size], apply as RMSNorm
                // Use normed buffer as temp to avoid aliasing
                unsafe {
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.q_attn,
                        &*q_norm_w,
                        1, nqh * hd, eps, cfg.rms_norm_one_plus_w, &self.stream,
                    )?;
                    d2d_copy_f32(&mut self.activations.q_attn, 0, &self.activations.normed, 0, (nqh * hd) as usize, &self.stream)?;
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.k_attn,
                        &*k_norm_w,
                        1, nkh * hd, eps, cfg.rms_norm_one_plus_w, &self.stream,
                    )?;
                    d2d_copy_f32(&mut self.activations.k_attn, 0, &self.activations.normed, 0, (nkh * hd) as usize, &self.stream)?;
                }
            }
        }


        // 5. Apply RoPE (skip for Nemotron-H which has no rotary embeddings)
        if cfg.use_rope {
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)?;

        self.kernels.mrope.forward(
            &mut self.activations.q_attn,
            &mut self.activations.k_attn,
            &self.activations.inv_freq,
            &self.activations.position_ids,
            nqh,
            nkh,
            hd,
            rd,
            s0,
            s1,
            s2,
            &self.stream,
        )?;
        } // end if cfg.use_rope

        // 6. KV write already done at step 4a (before QK-norm) for quantization quality.


        // 7. GQA attention
        let seq_len = position + 1;
        self.kernels.gqa_attention.forward(
            &mut self.activations.attn_out,
            &self.activations.q_attn,
            &self.kv_caches[kv_cache_idx].k,
            &self.kv_caches[kv_cache_idx].v,
            nqh,
            nkh,
            hd,
            seq_len,
            max_sl as u32,
            &self.stream,
        )?;


        // 8. Output gate (Qwen3.5 only) or pass-through
        let final_attn = if cfg.has_output_gate {
            self.kernels.output_gate.forward(
                &mut self.activations.gated_out,
                &self.activations.attn_out,
                &self.activations.gate_attn,
                nqh * hd,
                &self.stream,
            )?;
            &self.activations.gated_out as *const DeviceBuffer<f32>
        } else {
            &self.activations.attn_out as *const DeviceBuffer<f32>
        };


        // 9. Output projection
        unsafe {
            (*w_o_p).forward(&self.kernels.linear_proj,
                &mut self.activations.out_proj, &*final_attn,
                hs, nqh * hd, &self.stream)?;
        }


        // 10. Residual add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        Ok(())
    }

    pub fn config(&self) -> &ModelConfig { &self.config }
    pub fn stream(&self) -> &Stream { &self.stream }
    pub fn vocab_size(&self) -> usize { self.config.vocab_size }

    pub fn set_position(&mut self, position: u32) -> HipResult<()> {
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)
    }

    pub fn read_logits(&self) -> Result<Vec<f32>, ModelError> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// GPU-resident argmax: run decode step and return token ID without transferring logits.
    /// Eliminates vocab_size×4 bytes PCIe transfer per token (e.g., 512KB for Nemotron).
    pub fn decode_step_token(&mut self, token_id: u32, position: u32) -> Result<u32, ModelError> {
        // Run the full decode step (logits stay on GPU — decode_step copies them
        // to CPU but that's the current architecture; the argmax avoids the NEXT copy)
        // TODO: add decode_step_no_copy to avoid even the first CPU transfer
        let _ = self.decode_step(token_id, position)?;
        // Argmax on GPU — transfers only 4 bytes back
        let result = self.kernels.argmax.forward(
            &self.activations.logits,
            &mut self.activations.argmax_result,
            self.config.vocab_size as u32,
            &self.stream,
        )?;
        Ok(result)
    }

    /// Run a single decode step. Returns logits [vocab_size].
    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        if has_moe || self.trace.is_some() {
            return self.decode_step_moe(token_id, position);
        }

        // Dense models: use megakernel (handles bf16 + quantized weights, both RMSNorm variants)
        if self.megakernel.is_none() {
            let mut mk = MegakernelProgram::compile(self)?;
            if let Ok(dump_path) = std::env::var("MEGAKERNEL_DUMP") {
                let max_slots: i32 = std::env::var("MEGAKERNEL_DUMP_SLOTS")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(500);
                mk.enable_dump(max_slots)?;
                eprintln!("Megakernel dump enabled: {} slots, output={}", max_slots, dump_path);
            }
            self.megakernel = Some(mk);
        }
        let mk = self.megakernel.as_mut().unwrap();
        mk.update_step(token_id, position, &self.stream)?;
        mk.execute(&self.stream)?;

        self.stream.synchronize()?;

        // Write dump after first decode token if MEGAKERNEL_DUMP is set
        if let Ok(dump_path) = std::env::var("MEGAKERNEL_DUMP") {
            if mk.dump_active() {
                mk.write_dump_btrc(&self.stream, &dump_path)?;
                mk.disable_dump()?;
            }
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// MoE decode step: kernel-by-kernel execution with MoE FFN dispatch.
    fn decode_step_moe(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;

        // Set position_ids for mRoPE/RoPE
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)?;

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;

        if self.debug_nan {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("embed(tok={token_id}): max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("embed", &buf);
        }

        // Process each layer
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        let mut mamba2_idx = 0usize;
        for layer_i in 0..self.config.num_layers {
            match self.config.layers[layer_i].layer_type {
                LayerType::Attention => {
                    self.attention_forward(layer_i, kv_idx, position)?;
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    self.gdn_forward(layer_i, gdn_idx)?;
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    self.mamba2_forward(layer_i, mamba2_idx)?;
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Standalone MoE FFN layer — just norm + MoE dispatch + residual
                    // The norm is applied inside moe_ffn_forward, skip to FFN below
                }
                LayerType::LfmConv => panic!("LfmConv not yet implemented"),
            }

            // Debug: check for NaN in hidden state after each layer
            if self.debug_nan {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                let nan_count = buf.iter().filter(|x| x.is_nan()).count();
                let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("L{layer_i} ({:?}): {nan_count} NaN, max_abs={max_abs:.2e}", self.config.layers[layer_i].layer_type);
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace.as_mut().unwrap().write_checkpoint(&format!("L{layer_i}.post_mixer"), &buf);
            }

            // FFN: dense, MoE, or None (standalone layers like Nemotron M/*)
            if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. }) {
                self.moe_ffn_forward(layer_i)?;
            } else if matches!(self.config.layers[layer_i].ffn_type, FfnType::None) {
                // No FFN for this layer (Nemotron M and * layers)
            } else {
                // Dense FFN: fused (bf16) or unfused (quantized)
                let hs = self.config.hidden_size;
                let is = self.config.intermediate_size;
                let eps = self.config.rms_norm_eps;

                // SAFETY: Raw pointers break borrow on self.layers for mutable self.activations.
                let (post_norm_p, w_gate_p, w_up_p, w_down_p) = match &self.layers[layer_i] {
                    LayerWeights::Attention(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    LayerWeights::Gdn(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    _ => panic!("dense FFN only for Attention/Gdn layers"),
                };

                let all_bf16 = unsafe {
                    matches!(&*w_gate_p, LinearWeight::Bf16(_))
                    && matches!(&*w_up_p, LinearWeight::Bf16(_))
                    && matches!(&*w_down_p, LinearWeight::Bf16(_))
                };

                if all_bf16 {
                    unsafe { self.ffn_forward(&*post_norm_p, (*w_gate_p).as_bf16(), (*w_up_p).as_bf16(), (*w_down_p).as_bf16())?; }
                } else {
                    // Unfused path for quantized weights
                    unsafe {
                        d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
                    }
                    unsafe {
                        self.kernels.rmsnorm.forward(
                            &mut self.activations.normed, &self.activations.hidden, &*post_norm_p,
                            1, hs as u32, eps, self.config.rms_norm_one_plus_w, &self.stream)?;
                    }
                    unsafe {
                        (*w_gate_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_gate, &self.activations.normed,
                            is as u32, hs as u32, &self.stream)?;
                        (*w_up_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_up, &self.activations.normed,
                            is as u32, hs as u32, &self.stream)?;
                    }
                    self.kernels.silu_mul.forward(
                        &mut self.activations.ffn_act, &self.activations.ffn_gate, &self.activations.ffn_up,
                        is as u32, &self.stream)?;
                    unsafe {
                        (*w_down_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_down, &self.activations.ffn_act,
                            hs as u32, is as u32, &self.stream)?;
                    }
                    self.kernels.residual_add.forward(
                        &mut self.activations.hidden, &self.activations.ffn_down, &self.activations.residual,
                        hs as u32, &self.stream)?;
                }
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace.as_mut().unwrap().write_checkpoint(&format!("L{layer_i}.post_ffn"), &buf);
            }
        }

        // Final RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &self.final_norm_weight,
            1, hs, eps, self.config.rms_norm_one_plus_w, &self.stream,
        )?;

        // LM head
        let lm_head_w = if self.config.tie_word_embeddings {
            &self.embed_weight
        } else {
            &self.lm_head_weight
        };
        self.kernels.linear_proj.forward(
            &mut self.activations.logits,
            lm_head_w,
            &self.activations.normed,
            self.config.vocab_size as u32, hs, &self.stream,
        )?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        if self.trace.is_some() {
            // Capture final_norm hidden state
            let mut norm_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.normed.copy_to_host(&mut norm_buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("final_norm", &norm_buf);

            // Capture top-10 logits (token_id + value pairs as f32)
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed.iter().take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace.as_mut().unwrap().write_checkpoint("top10_logits", &top10);
        }

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, false)
    }

    /// Run a single decode step with quantized KV cache (4-bit residual_pc).
    /// Sealed chunks are quantized to int4; active chunk stays f32.
    pub fn decode_step_paged_quantized(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, true)
    }

    fn decode_step_paged_inner(&mut self, token_id: u32, position: u32, quantized: bool) -> Result<Vec<f32>, ModelError> {
        let max_chunks = (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS;

        // Lazy-init: compile paged megakernel
        if self.megakernel_paged.is_none() {
            let mut mk = MegakernelProgram::compile_paged(self)?;
            mk.init_paged_buffers(max_chunks)?;
            if quantized {
                mk.enable_quantized_kv(max_chunks, &self.config)?;
            }
            self.megakernel_paged = Some(mk);
        } else {
            let mk = self.megakernel_paged.as_ref().unwrap();
            assert_eq!(mk.quantized_kv, quantized,
                "cannot mix decode_step_paged and decode_step_paged_quantized on the same model");
        }

        // Lazy-init: f32 PageAllocator (staging) and SequenceState
        if self.page_allocator.is_none() {
            self.page_allocator = Some(PageAllocator::new(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
            self.paged_seq = Some(SequenceState::new(CHUNK_TOKENS as u32));
        }

        // Lazy-init: quantized PageAllocator
        if quantized && self.quant_allocator.is_none() {
            self.quant_allocator = Some(PageAllocator::new_quantized(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
        }

        // append_token
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(alloc_mut)?;
        }

        let stream = &self.stream;
        let mk = self.megakernel_paged.as_mut().unwrap();
        let seq = self.paged_seq.as_ref().unwrap();
        let allocator = self.page_allocator.as_ref().unwrap();

        mk.update_step_paged(token_id, position, seq, allocator, stream)?;
        mk.execute(stream)?;
        stream.synchronize()?;

        // Post-step: handle chunk seal + quantization
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let q_alloc = self.quant_allocator.as_mut();
            mk.post_step_paged(position, seq_mut, alloc_mut, q_alloc, &self.config, &self.stream)?;
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Save the current GDN recurrent states into a checkpoint pool slot.
    /// Lazy-initializes the pool on first call. Returns the slot index.
    pub fn save_recurrent_checkpoint(&mut self) -> Result<u32, ModelError> {
        if self.checkpoint_pool.is_none() {
            // Pool capacity 1: prefill uses ring-buffer overwrite (only most-recent needed).
            // Speculative decode (future) may increase this.
            self.checkpoint_pool = Some(RecurrentCheckpointPool::new(
                self.device,
                &self.config,
                1,
            )?);
        }
        // Free previous slot before allocating new one (ring buffer with capacity 1)
        if let Some(prev) = self.last_checkpoint_slot.take() {
            self.checkpoint_pool.as_mut().unwrap().free(prev);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> = self.gdn_states.iter().map(|s| &s.recurrent).collect();
        let pool = self.checkpoint_pool.as_mut().unwrap();
        let slot = paged_kv::save_checkpoint(pool, &recurrent_bufs, self.stream.raw())?;
        self.last_checkpoint_slot = Some(slot);
        Ok(slot)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    /// Saves GDN checkpoints at each 64-token chunk boundary.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }

        // MoE and quantized-weight models can't use megakernel prefill
        // (batched FFN fused kernel only handles bf16).
        // Fall back to sequential decode.
        let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        let has_quant = self.config.weight_quant != WeightQuantMode::Bf16;
        if has_moe || has_quant {
            let mut logits = vec![];
            for (i, &tok) in tokens.iter().enumerate() {
                logits = self.decode_step_moe(tok, i as u32)?;
            }
            return Ok(logits);
        }

        let mut pos = 0u32;
        for chunk in tokens.chunks(CHUNK_TOKENS) {
            let mut bufs = PrefillBuffers::alloc(self.device, &self.config, chunk.len())?;
            let program = MegakernelProgram::compile_prefill(self, chunk, pos, &mut bufs)?;
            program.execute(&self.stream)?;
            self.stream.synchronize()?;
            pos += chunk.len() as u32;
            if pos < tokens.len() as u32 {
                let _slot = self.save_recurrent_checkpoint()?;
            }
        }
        self.read_logits()
    }

    /// Read all GDN recurrent state to host (for testing).
    pub fn read_gdn_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_states.len());
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            state.recurrent.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Restore GDN recurrent states from a previously saved checkpoint slot.
    pub fn restore_recurrent_checkpoint(&mut self, slot: u32) -> Result<(), ModelError> {
        let pool = self.checkpoint_pool.as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self.gdn_states.iter_mut().map(|s| &mut s.recurrent).collect();
        let stream_raw = self.stream.raw();
        paged_kv::restore_checkpoint(pool, slot, &mut recurrent_bufs, stream_raw)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn read_hidden(&self) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut buf = vec![0.0f32; self.config.hidden_size];
        self.activations.hidden.copy_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn decode_step_traced(&mut self, token_id: u32, position: u32) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..self.config.num_layers {
            if self.config.layers[i].layer_type == LayerType::Attention {
                self.attention_forward(i, kv_idx, position)?;
                kv_idx += 1;
            } else {
                self.gdn_forward(i, gdn_idx)?;
                gdn_idx += 1;
            }
            traces.push((format!("layer_{i}"), self.read_hidden()?));
        }

        unsafe { d2d_copy_f32(&mut self.activations.normed, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden, &self.activations.normed,
            &self.final_norm_weight, 1, hs, self.config.rms_norm_eps, self.config.rms_norm_one_plus_w, &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits, &self.embed_weight,
            &self.activations.hidden, vs, hs, &self.stream,
        )?;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok((logits, traces))
    }

    fn read_buf(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut v = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut v)?;
        Ok(v)
    }

    pub fn gdn_layer0_trace(&mut self, token_id: u32) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed, &self.activations.hidden,
            &w.input_norm, 1, hs, eps, self.config.rms_norm_one_plus_w, &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        w.w_qkv.forward(&self.kernels.linear_proj,
            &mut self.activations.qkv, &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd, hs, &self.stream)?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        w.w_a.forward(&self.kernels.linear_proj,
            &mut self.activations.a_proj, &self.activations.normed, nvh_traced, hs, &self.stream)?;
        w.w_b.forward(&self.kernels.linear_proj,
            &mut self.activations.b_proj, &self.activations.normed, nvh_traced, hs, &self.stream)?;
        w.w_z.forward(&self.kernels.linear_proj,
            &mut self.activations.z_proj, &self.activations.normed, nvh_traced * vd, hs, &self.stream)?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nvh_traced * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, conv_q_len, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, conv_q_len + conv_k_len, conv_v_len, &self.stream)?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            d2d_copy_u16(&mut conv_w_q, 0, &w.conv1d_weight, 0, conv_q_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_k, 0, &w.conv1d_weight, conv_q_len * ck_usize, conv_k_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_v, 0, &w.conv1d_weight, (conv_q_len + conv_k_len) * ck_usize, conv_v_len * ck_usize, &self.stream)?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(&mut cs_q, 0, &self.gdn_conv_states[0], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut cs_k, 0, &self.gdn_conv_states[0], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut cs_v, 0, &self.gdn_conv_states[0], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(&mut cs_q, &self.activations.q_gdn, &conv_w_q, &mut conv_out_q, conv_q_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_k, &self.activations.k_gdn, &conv_w_k, &mut conv_out_k, conv_k_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_v, &self.activations.v_gdn, &conv_w_v, &mut conv_out_v, conv_v_len as u32, ck, &self.stream)?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &conv_out_q, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &conv_out_k, 0, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &conv_out_v, 0, conv_v_len, &self.stream)?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn, &w.a_log, &self.activations.a_proj,
            &w.dt_bias, nh, &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn, &self.activations.k_gdn, &self.activations.v_gdn,
            &self.activations.gate_gdn, &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent, &mut self.activations.recurrent_out,
            nvh_traced, kd, vd, gqa_traced, &self.stream,
        )?;
        traces.push(("recurrent_out".into(), self.read_buf(&self.activations.recurrent_out)?));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated, &self.activations.recurrent_out,
            &self.activations.z_proj, &w.output_norm, nvh_traced, vd, eps, &self.stream,
        )?;
        traces.push(("normed_gated".into(), self.read_buf(&self.activations.normed_gated)?));

        // out_proj
        w.w_out.forward(&self.kernels.linear_proj,
            &mut self.activations.out_proj, &self.activations.normed_gated,
            hs, nvh_traced * vd, &self.stream)?;
        traces.push(("out_proj".into(), self.read_buf(&self.activations.out_proj)?));

        // Residual
        unsafe { d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden, &self.activations.out_proj,
            &self.activations.residual, hs, &self.stream,
        )?;
        traces.push(("after_residual".into(), self.read_hidden()?));

        Ok(traces)
    }

    pub fn reset_state(&mut self) -> Result<(), ModelError> {
        let nh = self.config.linear_num_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let ck = self.config.linear_conv_kernel_dim;
        let nvh_r = self.config.linear_num_value_heads;
        let qkv_out = nh * kd * 2 + nvh_r * vd;

        for state in &mut self.gdn_states {
            let zeros = vec![0.0f32; nvh_r * kd * vd];
            state.recurrent.copy_from_host(&zeros)?;
        }
        for conv_state in &mut self.gdn_conv_states {
            let zeros = vec![0.0f32; qkv_out * (ck - 1)];
            conv_state.copy_from_host(&zeros)?;
        }
        let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
        let zeros_kv = vec![0.0f32; kv_size];
        for cache in &mut self.kv_caches {
            cache.k.copy_from_host(&zeros_kv)?;
            cache.v.copy_from_host(&zeros_kv)?;
        }
        self.seq_len = 0;
        // Free quantized KV slots back to pool
        if let (Some(seq), Some(q_alloc)) = (self.paged_seq.as_mut(), self.quant_allocator.as_mut()) {
            seq.free_quant_slots(q_alloc);
        }
        Ok(())
    }
}
