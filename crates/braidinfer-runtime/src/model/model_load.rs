//! Model weight loading and initialization.
//! Extracted from model.rs for maintainability.

use std::path::{Path, PathBuf};

use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;

use super::Model;
use crate::config::*;
use crate::kernel::AllKernels;
use crate::weights::*;

/// Resolve the .bqnt file path: BQNT_PATH env var takes priority; else auto-derive
/// from `model_dir` as `{parent}/{model_dir_name}.q4.bqnt`. Returns `Some(path)` only
/// when the resolved path exists on disk. Returns `None` when neither env nor auto path
/// resolves to an existing file.
///
/// This is the single source of truth for bqnt path resolution, shared between
/// `Model::load_with_max_seq_len` and `Model::enable_distributed_moe` so that a model
/// loaded via auto-derived path can be run multi-GPU without setting BQNT_PATH
/// (bd braidinfer-abuf).
pub(super) fn resolve_bqnt_path(model_dir: &Path) -> Option<PathBuf> {
    resolve_bqnt_path_with_env(std::env::var("BQNT_PATH").ok(), model_dir)
}

/// Inner helper that takes the env value as a parameter for deterministic unit testing.
fn resolve_bqnt_path_with_env(bqnt_path_env: Option<String>, model_dir: &Path) -> Option<PathBuf> {
    let explicit = bqnt_path_env.map(PathBuf::from);
    let auto = model_dir.file_name().map(|n| {
        model_dir
            .parent()
            .unwrap_or(model_dir)
            .join(format!("{}.q4.bqnt", n.to_string_lossy()))
    });
    explicit.or(auto).filter(|p| p.exists())
}

impl Model {
    /// VRAM headroom reserved above the bqnt data-section size when deciding whether to
    /// build the weight arena (activations + KV cache + the few Bf16->f32 widened tensors).
    const ARENA_VRAM_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

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
        // bd 9gmh: KV_QUANT+PERSISTENT guard removed — quantize_sealed_chunk now dispatches
        // via persistent worker mailbox (quantize_sealed_chunk_via_worker), which is safe
        // under the cooperative kernel. KV_QUANT+MULTI_GPU remains unsupported (multi-GPU
        // paged dispatch not yet implemented).
        let kv_quant = std::env::var("KV_QUANT").as_deref() == Ok("1");
        let multi_gpu_env = std::env::var("MULTI_GPU").is_ok();
        if kv_quant && multi_gpu_env {
            return Err(ModelError::InvalidConfig(
                "KV_QUANT=1 is not supported with MULTI_GPU \
                 (multi-GPU paged KV dispatch not yet implemented). \
                 Either unset KV_QUANT, or unset MULTI_GPU.".into(),
            ));
        }

        let config_path = model_dir.join("config.json");
        // bd b77g: prefer the .bqnt's embedded `model_config` over the HF config.json so
        // a model's architecture does not depend on the HF cache, and to avoid the silent
        // (wrong) qwen35_0_8b() fallback. (Weights still load from the safetensors dir —
        // full HF-independence is follow-on braidinfer-4ayf.) The bqnt is re-opened for
        // weights below; this metadata-only open is cheap (mmap, no tensor read).
        let bqnt_model_config: Option<serde_json::Value> = {
            let explicit = std::env::var("BQNT_PATH").ok().map(std::path::PathBuf::from);
            let auto = model_dir.file_name().map(|n| {
                model_dir
                    .parent()
                    .unwrap_or(model_dir)
                    .join(format!("{}.q4.bqnt", n.to_string_lossy()))
            });
            explicit
                .or(auto)
                .filter(|p| p.exists())
                .and_then(|p| crate::bqnt::MmapBqnt::open(&p).ok())
                .and_then(|b| b.metadata().ok())
                .and_then(|m| serde_json::from_str::<serde_json::Value>(&m).ok())
                .and_then(|v| v.get("model_config").cloned())
                .filter(|v| !v.is_null())
        };
        let mut config = if let Some(v) = &bqnt_model_config {
            ModelConfig::from_config_value(v)
                .map_err(|e| ModelError::MissingWeight(format!("bqnt model_config: {e}")))?
        } else if config_path.exists() {
            ModelConfig::from_config_json(&config_path)
                .map_err(|e| ModelError::MissingWeight(format!("config.json: {e}")))?
        } else {
            eprintln!(
                "WARNING (bd b77g): the .bqnt has no embedded model_config AND there is no \
                 config.json at {} — falling back to hardcoded ModelConfig::qwen35_0_8b(), \
                 which is WRONG for any non-0.8B model (dim mismatch). Re-quantize with a \
                 config-embedding bqnt_quantize, or provide config.json.",
                model_dir.display()
            );
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

        // bd 4ayf A3.2.3b: tolerate a missing/empty safetensors dir — a self-contained .bqnt
        // supplies every weight + name, so st is the legacy fallback only. Empty st degrades
        // gracefully (bqnt-first load paths never hit it for a complete bqnt); an INCOMPLETE
        // bqnt (or no bqnt) + empty st surfaces as MissingWeight at the specific tensor.
        let st = SafeTensorSet::open_directory(model_dir).unwrap_or_else(|_| SafeTensorSet::empty());

        // Locate .bqnt file via resolve_bqnt_path (single source of truth shared with
        // enable_distributed_moe — fixes auto-derived-path multi-GPU failure bd abuf).
        let resolved_bqnt_path: Option<PathBuf> = resolve_bqnt_path(model_dir);
        let is_explicit_bqnt = std::env::var("BQNT_PATH").is_ok();

        let bqnt = resolved_bqnt_path.as_deref().and_then(|path| {
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
        });

        // If no bqnt found and quantizing, create a writer to cache for next launch.
        // Only create writer when BQNT_PATH is not explicitly set (avoid overwriting user files).
        // auto_bqnt_path: only needed as the writer-destination when no bqnt exists yet.
        let auto_bqnt_path = if !is_explicit_bqnt {
            model_dir.file_name().map(|n| {
                model_dir
                    .parent()
                    .unwrap_or(model_dir)
                    .join(format!("{}.q4.bqnt", n.to_string_lossy()))
            })
        } else {
            None
        };
        let save_bqnt_path = if bqnt.is_none()
            && !is_explicit_bqnt
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
            // bd 4ayf A3.2.3: discover names from the bqnt name_table (no HF dir needed);
            // fall back to safetensors for old (pre-name_table) bqnts or no bqnt.
            let names: Vec<String> = bqnt
                .as_ref()
                .map(|b| b.tensor_names())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| st.tensor_names().iter().map(|s| s.to_string()).collect());
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

        // Discover model features from tensor names (bd 4ayf A3.2.3: bqnt name_table first).
        let names: Vec<String> = bqnt
            .as_ref()
            .map(|b| b.tensor_names())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| st.tensor_names().iter().map(|s| s.to_string()).collect());
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
            let gated_out = config.num_q_heads * config.head_dim * 2;
            // bd 4ayf A3.2.3b: prefer the bqnt entry's out_features — the q_proj is QUANTIZED
            // in the bqnt, so its raw byte length is NOT bf16*shape. Fall back to st bf16 length.
            if let Some(e) = bqnt.as_ref().and_then(|b| b.entry(&q_name)) {
                e.out_features as usize == gated_out
            } else if let Ok(raw) = st.tensor_data(&q_name) {
                raw.len() == gated_out * config.hidden_size * 2 // bf16
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

        // bd 4ayf B1 + 4ayf.12: build the bulk-load arena BEFORE the weight-load closures so that
        // EVERY weight — embed/norms/recurrent/latent_proj AND the Packed linears — becomes a
        // non-owning VIEW into it (no per-tensor copies, no duplication). One VRAM block = the
        // bqnt data section, filled by a single bulk hipMemcpy. Single-GPU + bqnt only (multi-GPU
        // + quantize-at-load keep per-tensor; arena_view = None). The arena is moved into Model
        // below (the VRAM pointer survives the struct move) and outlives the views (Drop no-op).
        // Gate: skip if the data section won't fit with ~2GB headroom (activations/KV + the few
        // Bf16->f32 widened tensors that still copy); per-tensor fallback then.
        // bd 4ayf.12: skip the arena for MoE models — experts are FUSED (gate_up from 2 tensors
        // into one buffer), so they can't be arena views; the arena would hold their bytes AND the
        // fused copies (huge duplication -> OOM, e.g. nemotron 30B). MoE loads per-tensor (big
        // ones run multi-GPU). Dense models view ALL weights (no fusion -> no duplication).
        let has_moe = config
            .layers
            .iter()
            .any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        let (weight_arena, arena_view): (Option<DeviceBuffer<u8>>, Option<(*const u8, u64)>) =
            if !multi_gpu && !has_moe {
                match bqnt.as_ref().and_then(|b| b.data_section()) {
                    Some((data_start, span)) => {
                        let free = crate::cli::vram_free_per_gpu()
                            .get(device.0 as usize)
                            .copied()
                            .unwrap_or(0);
                        let needed = span.len() as u64 + Self::ARENA_VRAM_HEADROOM_BYTES;
                        if free == 0 || needed <= free as u64 {
                            let mut arena = DeviceBuffer::<u8>::alloc(device, span.len())?;
                            arena.copy_from_host(span)?;
                            let ptr = arena.as_ptr();
                            eprintln!(
                                "bd 4ayf B1: weight arena {} MiB, single bulk copy (all weights are views)",
                                span.len() / (1024 * 1024)
                            );
                            (Some(arena), Some((ptr, data_start)))
                        } else {
                            eprintln!(
                                "bd 4ayf B1: skipping weight arena ({} MiB data section vs {} MiB free) \
                                 — per-tensor load",
                                span.len() / (1024 * 1024),
                                free / (1024 * 1024)
                            );
                            (None, None)
                        }
                    }
                    None => (None, None),
                }
            } else {
                (None, None)
            };

        // bd 4ayf A3.2: bqnt-first bf16 loader (st = legacy fallback) for the embeds / norms /
        // lm_head tensors A2 now stores in the bqnt. Arena views when present (bd 4ayf.12).
        let load_bf16 = |name: &str, len: usize| -> Result<DeviceBuffer<u16>, ModelError> {
            if let Some(ref b) = bqnt {
                if let Ok(w) = crate::weights::load_weight_bf16_bqnt(b, name, device, len, arena_view) {
                    return Ok(w);
                }
            }
            crate::weights::load_weight_bf16(&st, name, device, len)
        };
        // bd 4ayf A3.2: bqnt-first f32 loader (mirrors load_weight_f32's dtype-flexibility:
        // F32 storage direct, Bf16 storage widened). For the GDN/Mamba2 recurrent-state +
        // f32-read norm tensors A2 stores in the bqnt. st = legacy fallback.
        let load_f32 = |name: &str, len: usize| -> Result<DeviceBuffer<f32>, ModelError> {
            if let Some(ref b) = bqnt {
                if let Ok(w) = crate::weights::load_weight_f32_bqnt(b, name, device, len, arena_view) {
                    return Ok(w);
                }
            }
            crate::weights::load_weight_f32(&st, name, device, len)
        };
        let embed_weight = load_bf16(&embed_name, config.vocab_size * config.hidden_size)?;
        let lm_head_weight = if config.tie_word_embeddings {
            // Weight-tied: reuse embed_weight pointer (allocate a dummy — the megakernel uses embed_weight)
            DeviceBuffer::<u16>::alloc(device, 0)? // placeholder, megakernel will use embed_weight
        } else {
            let lm_head_name = names
                .iter()
                .find(|n| n.contains("lm_head.weight"))
                .ok_or_else(|| ModelError::MissingWeight("lm_head.weight not found".into()))?
                .to_string();
            load_bf16(&lm_head_name, config.vocab_size * config.hidden_size)?
        };
        let final_norm_weight = load_bf16(&norm_name, config.hidden_size)?;

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

        // (bd 4ayf.12: the weight arena is built earlier — before the load_bf16/load_f32
        // closures — so embed/norms/recurrent/latent_proj + Packed linears are ALL arena views.)

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
                            if let Ok(lw) = crate::weights::load_linear_weight_bqnt(b, name, device, arena_view)
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
                    // use_quant==false (WEIGHT_QUANT_LAYERS-excluded, wants bf16) or the bqnt
                    // lacked the tensor: load from st. bd 4ayf.6: if st can't provide it
                    // (empty-HF_HOME), fall back to the bqnt's stored (quantized) copy — an
                    // excluded layer then loads quantized instead of bf16, so warn.
                    match load_linear_weight(&st, name, device, out_dim, in_dim, wq) {
                        Ok(lw) => Ok(lw),
                        Err(e) => {
                            if let Some(ref b) = bqnt {
                                if let Ok(lw) =
                                    crate::weights::load_linear_weight_bqnt(b, name, device, arena_view)
                                {
                                    if !use_quant {
                                        eprintln!(
                                            "WARNING (bd 4ayf.6): {name} is WEIGHT_QUANT_LAYERS-\
                                             excluded (wanted bf16) but no HF dir is available — \
                                             loading the bqnt's quantized copy instead."
                                        );
                                    }
                                    return Ok(lw);
                                }
                            }
                            Err(e)
                        }
                    }
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
                    bqnt.as_ref(),
                    &[
                        format!("{p}norm.weight"),
                        format!("{p}input_layernorm.weight"),
                    ],
                )?;
                let w = Mamba2LayerWeights {
                    input_norm: load_bf16(&norm_name, hs)?,
                    in_proj: load_lw(&format!("{p}mixer.in_proj.weight"), in_proj_size, hs)?,
                    conv1d_weight: load_bf16(&format!("{p}mixer.conv1d.weight"), cd * ck)?,
                    conv1d_bias: load_f32(&format!("{p}mixer.conv1d.bias"), cd)?,
                    dt_bias: load_f32(&format!("{p}mixer.dt_bias"), nh)?,
                    a_log: load_f32(&format!("{p}mixer.A_log"), nh)?,
                    d: load_f32(&format!("{p}mixer.D"), nh)?,
                    norm_weight: load_f32(&format!("{p}mixer.norm.weight"), intermediate)?,
                    out_proj: load_lw(&format!("{p}mixer.out_proj.weight"), hs, intermediate)?,
                };
                layers.push(LayerWeights::Mamba2(w));
            } else if *layer_type == LayerType::MoeFfn {
                // Standalone MoE FFN layer (Nemotron-H 'E' layers)
                let hs = config.hidden_size;
                let norm_name = find_weight_name(
                    &st,
                    bqnt.as_ref(),
                    &[
                        format!("{p}norm.weight"),
                        format!("{p}input_layernorm.weight"),
                    ],
                )?;
                let w = MoeFfnLayerWeights {
                    input_norm: load_bf16(&norm_name, hs)?,
                };
                layers.push(LayerWeights::MoeFfn(w));
                // Load MoE weights — Nemotron uses mixer.gate/mixer.experts instead of mlp.gate/mlp.experts
                // Try Nemotron naming first by checking if mixer.gate.weight exists
                let gate_name = format!("{p}mixer.gate.weight");
                // bd 4ayf A3.2.3b: MoE-prefix probe — bqnt entry first, st fallback.
                let has_mixer_gate = bqnt
                    .as_ref()
                    .map(|b| b.entry(&gate_name).is_some())
                    .unwrap_or(false)
                    || st.tensor_data(&gate_name).is_ok();
                let moe_prefix = if has_mixer_gate {
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
                    input_norm: load_bf16(
                        &find_weight_name(
                            &st,
                            bqnt.as_ref(),
                            &[
                                format!("{p}input_layernorm.weight"),
                                format!("{p}norm.weight"),
                            ],
                        )?,
                        hs,
                    )?,
                    w_q_gate: load_lw(
                        &find_weight_name(
                            &st,
                            bqnt.as_ref(),
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
                            bqnt.as_ref(),
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
                            bqnt.as_ref(),
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
                            bqnt.as_ref(),
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
                        // bd 4ayf A3.2.3b: element count bqnt-first (entry out*in), st len/2 fallback.
                        let len = bqnt
                            .as_ref()
                            .and_then(|b| b.entry(&name))
                            .map(|e| e.out_features as usize * e.in_features as usize)
                            .or_else(|| st.tensor_data(&name).ok().map(|r| r.len() / 2))
                            .ok_or_else(|| ModelError::MissingWeight(name.clone()))?;
                        load_bf16(&name, len)?
                    } else {
                        DeviceBuffer::<u16>::alloc(device, 0)?
                    },
                    k_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.k_norm.weight");
                        // bd 4ayf A3.2.3b: element count bqnt-first (entry out*in), st len/2 fallback.
                        let len = bqnt
                            .as_ref()
                            .and_then(|b| b.entry(&name))
                            .map(|e| e.out_features as usize * e.in_features as usize)
                            .or_else(|| st.tensor_data(&name).ok().map(|r| r.len() / 2))
                            .ok_or_else(|| ModelError::MissingWeight(name.clone()))?;
                        load_bf16(&name, len)?
                    } else {
                        DeviceBuffer::<u16>::alloc(device, 0)?
                    },
                    post_norm: {
                        let name =
                            find_weight_name(&st, bqnt.as_ref(), &[format!("{p}post_attention_layernorm.weight")]);
                        if let Ok(n) = name {
                            load_bf16(&n, hs)?
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
                // bd 4ayf A3.2.3b (+v1 backward-compat fix): conv1d raw bytes bqnt-first ONLY if
                // present at the expected bf16 size; a v1 bqnt may store it quantized/differently,
                // so size-mismatch falls back to st (was: use any bqnt data -> assert/panic).
                let conv_raw_bytes = match bqnt
                    .as_ref()
                    .and_then(|b| b.tensor_data(&conv_name))
                    .filter(|d| d.len() == conv_total * 2)
                {
                    Some(d) => d,
                    None => st
                        .tensor_data(&conv_name)
                        .map_err(|_| ModelError::MissingWeight(conv_name.clone()))?,
                };
                assert_eq!(conv_raw_bytes.len(), conv_total * 2);
                let conv_raw: &[u16] = unsafe {
                    std::slice::from_raw_parts(conv_raw_bytes.as_ptr() as *const u16, conv_total)
                };
                let mut conv_w_q_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_k_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_v_buf = DeviceBuffer::<u16>::alloc(device, v_dim * ck)?;
                conv_w_q_buf.copy_from_host(&conv_raw[..q_dim * ck])?;
                conv_w_k_buf.copy_from_host(&conv_raw[q_dim * ck..2 * q_dim * ck])?;
                conv_w_v_buf.copy_from_host(&conv_raw[2 * q_dim * ck..])?;
                let hs = config.hidden_size;
                let w = GdnLayerWeights {
                    input_norm: load_bf16(&format!("{p}input_layernorm.weight"), hs)?,
                    w_qkv: load_lw(&format!("{p}linear_attn.in_proj_qkv.weight"), qkv_out, hs)?,
                    w_a: load_lw(&format!("{p}linear_attn.in_proj_a.weight"), nvh, hs)?,
                    w_b: load_lw(&format!("{p}linear_attn.in_proj_b.weight"), nvh, hs)?,
                    w_z: load_lw(&format!("{p}linear_attn.in_proj_z.weight"), z_out, hs)?,
                    conv1d_weight_q: conv_w_q_buf,
                    conv1d_weight_k: conv_w_k_buf,
                    conv1d_weight_v: conv_w_v_buf,
                    a_log: load_f32(&format!("{p}linear_attn.A_log"), nvh)?,
                    dt_bias: load_bf16(&format!("{p}linear_attn.dt_bias"), nvh)?,
                    output_norm: load_f32(&format!("{p}linear_attn.norm.weight"), vd)?, // normalizes [nvh, vd] output
                    w_out: load_lw(&format!("{p}linear_attn.out_proj.weight"), hs, z_out)?,
                    post_norm: load_bf16(&format!("{p}post_attention_layernorm.weight"), hs)?,
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
            // alloc_portable_coherent (not _portable): _portable may use MTYPE_NC
            // (L2-cached) — GPU 0's op_rmsnorm_wx write to normed_stage can sit
            // in GPU 0's L2 past ack; workers' peer-reads get partial/stale data,
            // producing non-deterministic NaN injection at layer 0 K projection
            // (snl decode-step2 NaN bisect 2026-05-17, mirror snapshot).
            // Coherent forces fine-grained UC on all sides → ack-time visibility.
            normed_stage: braidinfer_hip::memory::MappedHostBuffer::<f32>::alloc_portable_coherent(hs)?,
            // bd braidinfer-sm16 sentinel for producer/consumer ordering — see
            // weights.rs:193 comment. One slot per attention layer (indexed by
            // attn_layer_count from compile_attention).
            normed_seq: {
                let n_attn = config.layers.iter()
                    .filter(|l| l.layer_type == crate::config::LayerType::Attention)
                    .count()
                    .max(1);
                braidinfer_hip::memory::MappedHostBuffer::<u32>::alloc_portable_coherent(n_attn)?
            },
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
        let watchdog = std::sync::Arc::new(crate::watchdog::WatchdogThread::spawn());
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
            weight_arena, // bd 4ayf B1: owns the bulk-load arena; outlives the weight views
            activations,
            gdn_conv_states,
            prefill_bufs: None,
            gdn_states,
            mamba2_states,
            seq_len: 0,
            megakernel_paged: None,
            page_allocator: None,
            quant_allocator: None,
            host_page_allocator: None,
            paged_seq: None,
            prefill_paged_page_table: None,
            prefill_paged_position_table: None,
            checkpoint_pool: None,
            last_checkpoint_slot: None,
            debug_nan: std::env::var("DEBUG_NAN").is_ok(),
            has_moe,
            debug_p2p_hidden: std::env::var("DEBUG_P2P_HIDDEN").is_ok(),
            weight_prefix: prefix.clone(),
            bqnt_path: resolved_bqnt_path,
            multi_gpu: None,
            distributed_moe: Vec::new(),
            moe_p2p: None,
            watchdog,
            megakernel_multi_gpu_p2p: None,
            persistent_workers: None,
            tracer: crate::tracer::Tracer::disabled(),
        })
    }

    /// Initialize distributed MoE expert dispatch. Populates `distributed_moe`
    /// per layer. On multi-GPU, also distributes expert weights across GPUs
    /// (round-robin) and sets up head-parallel attention buffers. On single-GPU
    /// (bd 174k), populates a 1-entry layout that aliases moe.expert_gate_up
    /// on GPU 0 with no VRAM duplication — needed by the unified mailbox-routed
    /// prefill path. Must be called after load, before first decode_step.
    pub fn enable_distributed_moe(&mut self) -> Result<(), ModelError> {
        if !self.has_moe {
            eprintln!("enable_distributed_moe: model has no MoE layers, skipping");
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
        // bd 174k Phase A: when only 1 GPU is available we still populate
        // distributed_moe (with num_devices=1, all experts on GPU 0). The
        // unified MoE prefill path (moe_ffn_forward_prefill_batched) reads
        // per-GPU expert config from distributed_moe regardless of GPU count.
        // distribute_moe_weights_from_ref handles num_devices=1 cleanly: it
        // aliases the existing moe.expert_gate_up buffer on GPU 0 (no VRAM
        // duplication), uses identity slot_map, and skips memcpy on gpu==0.
        let mut ctx_opt: Option<crate::multi_gpu::MultiGpuContext> = match ctx {
            Some(c) => Some(c),
            None => {
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
                eprintln!("Single-GPU MoE: populating distributed_moe[layer] with GPU-0-only layout (no VRAM duplication)");
                None
            }
        };

        // DeviceGuard pins the rest of this function to GPU 0 and restores
        // the caller's device when this guard drops at function return.
        let _gpu0_guard = braidinfer_hip::device::DeviceGuard::switch_to(DeviceId(0))?;

        // Distribute MoE weights across GPUs (or onto GPU 0 only when single-GPU).
        let num_devices = ctx_opt.as_ref().map_or(1, |c| c.num_devices);
        let hs = self.config.hidden_size;

        // Check if expert weights were loaded (single-GPU) or skipped (multi-GPU lite load)
        let experts_on_gpu0 = self.moe_weights.iter().any(|m| {
            m.as_ref().map_or(false, |moe| {
                moe.expert_gate_up.raw_data_ptr() != std::ptr::null()
            })
        });

        // bd abuf: use self.bqnt_path (set at load time via resolve_bqnt_path, which
        // considers both BQNT_PATH env and the auto-derived sibling path) so that a model
        // loaded via auto-derived path works multi-GPU without requiring BQNT_PATH to be set.
        let bqnt = self.bqnt_path.as_deref()
            .and_then(|p| crate::bqnt::MmapBqnt::open(p).ok());

        // VRAM diagnostic helper: prints per-GPU free MB.
        let print_vram = |label: &str| {
            let free_per_gpu: Vec<String> = crate::cli::vram_free_per_gpu()
                .iter()
                .enumerate()
                .map(|(i, &b)| format!("GPU{}={:.0}MB", i, b as f64 / (1024.0 * 1024.0)))
                .collect();
            eprintln!("  VRAM after {}: [{}]", label, free_per_gpu.join(", "));
        };
        print_vram("init");

        // bd 4ayf.13: pin the bqnt data section (PORTABLE, so every GPU can DMA from it) so the
        // thousands of per-expert h2d copies in distribute_moe_weights_from_bqnt DMA directly
        // instead of bounce-buffering through a staging copy. The multi-GPU expert distribute is
        // ~82% of load (measured 33.8s on qwen35_35b_a3b -g2) and dominated by these copies from
        // the (otherwise-unpinned) mmap — pinning is the primary win. Unregistered before return.
        const HIP_HOST_REGISTER_PORTABLE: u32 = 0x1;
        let pinned_bqnt: Option<*mut std::ffi::c_void> = bqnt.as_ref().and_then(|b| {
            b.data_section().and_then(|(_, span)| {
                let ptr = span.as_ptr() as *mut std::ffi::c_void;
                let rc = unsafe { ffi::hipHostRegister(ptr, span.len(), HIP_HOST_REGISTER_PORTABLE) };
                if rc == 0 {
                    eprintln!(
                        "bd 4ayf.13: pinned bqnt data section ({} MiB) for direct h2d DMA",
                        span.len() / (1024 * 1024)
                    );
                    Some(ptr)
                } else {
                    eprintln!("bd 4ayf.13: hipHostRegister(bqnt) rc={rc} — unpinned (slower) DMA");
                    None
                }
            })
        });

        let mut distributed = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            if let Some(ref moe) = self.moe_weights[i] {
                // P3 (braidinfer-4n5.7): MULTI-GPU disaggregation distributes experts
                // starting at GPU 1 — GPU 0 is SequenceAttention-only (attention +
                // coordinator), workers 1..N hold experts; OP_MOE_DISPATCH_POST sums
                // output_slots[1..total_gpus]. SINGLE-GPU (num_devices==1) has nothing
                // to disaggregate: GPU 0 IS the only GPU and runs experts itself via
                // op_moe_ffn (start_gpu=0). start_gpu=1 with num_devices=1 is invalid
                // ("start_gpu must be < num_devices") — the P3 regression caught by -g1.
                let start_gpu = if num_devices > 1 { 1usize } else { 0usize };
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
        print_vram("MoE distribute");

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
            // bd 174k Phase A: head-parallel attention split only applies to
            // true multi-GPU. Single-GPU (ctx_opt == None) skips this block.
            if let Some(ctx) = ctx_opt.as_mut() {
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
                    &self.config,
                    crate::megakernel::CHUNK_TOKENS,
                )?;
                print_vram("init_attn_buffers");
                // Split Q/K/V projection weights onto each GPU
                self.init_split_attn_weights(ctx, local_nqh, local_nkh, q_mult)?;
                print_vram("init_split_attn_weights");
            }
        }

        // multi_gpu is set only when there's a real multi-GPU context.
        // Single-GPU MoE populates distributed_moe (above) but leaves
        // self.multi_gpu = None — the unified prefill path handles this.
        self.multi_gpu = ctx_opt;

        // bd 174k Phase C: initialize moe_p2p NOW (at model setup time, before
        // any persistent_worker is launched). MoeP2pContext::init calls
        // DeviceBuffer::copy_from_host which trips the persistent-worker
        // guard if deferred until first prefill (after warmup spawned the
        // GPU 0 worker). This is safe here because enable_multi_gpu runs
        // immediately after Model::load on the binary side.
        // bd 4ayf.13: unregister the pinned bqnt data section NOW — after the distribute +
        // attn-split copies (the last bqnt uses) but BEFORE the persistent worker launches. Once
        // the cooperative worker holds the GPU's CUs, ANY HIP call (incl hipHostUnregister)
        // deadlocks (CLAUDE.md "What Causes Hangs in Persistent Mode"); placing it after the
        // worker launch was exit 247 (SIGKILL of the deadlocked process).
        if let Some(ptr) = pinned_bqnt {
            unsafe { ffi::hipHostUnregister(ptr) };
        }
        if self.has_moe {
            self.ensure_moe_workers_started()?;
        }
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

                // β' probe (bd braidinfer-sm16, udi #2652): byte-compare 16
                // bytes at offset 0 from src slice vs worker copy of the first
                // attn layer's w_q_gate + w_k. Layer 0 is where workers'
                // attn_k = 512/512 NaN on clean attn_normed input. If bytes
                // mismatch, copy_weight_slice has a slice-arithmetic bug; if
                // they match, β' is falsified and bug is in dequant or kernel.
                if std::env::var("BRAIDINFER_BETA_PRIME").is_ok() && layer_idx == attn_layer_indices[0] {
                    use braidinfer_hip::ffi;
                    let probe_byte_offset = w.w_q_gate.row_byte_offset_dim(q_row_start, hs);
                    let mut src_bytes = [0u8; 16];
                    let mut dst_bytes = [0u8; 16];
                    let src_ptr = unsafe { w.w_q_gate.raw_data_ptr().add(probe_byte_offset) };
                    unsafe {
                        let _ = ffi::hipMemcpy(
                            src_bytes.as_mut_ptr() as *mut std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            16,
                            ffi::hipMemcpyDeviceToHost,
                        );
                    }
                    let dst_q = ctx.workers[gpu_i].attn_w_q_gate.last().unwrap();
                    let dst_ptr = dst_q.raw_data_ptr();
                    unsafe {
                        let _ = ffi::hipMemcpy(
                            dst_bytes.as_mut_ptr() as *mut std::ffi::c_void,
                            dst_ptr as *const std::ffi::c_void,
                            16,
                            ffi::hipMemcpyDeviceToHost,
                        );
                    }
                    eprintln!(
                        "[β' L{layer_idx} g{gpu_i} w_q_gate@row{q_row_start}] src16={:02x?} dst16={:02x?} match={}",
                        src_bytes,
                        dst_bytes,
                        src_bytes == dst_bytes,
                    );

                    let k_byte_offset = w.w_k.row_byte_offset_dim(0, hs);
                    let mut src_k = [0u8; 16];
                    let mut dst_k = [0u8; 16];
                    let src_kptr = unsafe { w.w_k.raw_data_ptr().add(k_byte_offset) };
                    unsafe {
                        let _ = ffi::hipMemcpy(
                            src_k.as_mut_ptr() as *mut std::ffi::c_void,
                            src_kptr as *const std::ffi::c_void,
                            16,
                            ffi::hipMemcpyDeviceToHost,
                        );
                    }
                    let dst_kw = ctx.workers[gpu_i].attn_w_k.last().unwrap();
                    let dst_kptr = dst_kw.raw_data_ptr();
                    unsafe {
                        let _ = ffi::hipMemcpy(
                            dst_k.as_mut_ptr() as *mut std::ffi::c_void,
                            dst_kptr as *const std::ffi::c_void,
                            16,
                            ffi::hipMemcpyDeviceToHost,
                        );
                    }
                    eprintln!(
                        "[β' L{layer_idx} g{gpu_i} w_k@row0]            src16={:02x?} dst16={:02x?} match={}",
                        src_k,
                        dst_k,
                        src_k == dst_k,
                    );
                }
            }
        }
        // copy_weight_slice does not mutate current-device context, so no
        // explicit restore is needed here. (Prior code defensively set
        // DeviceId(0); the calling site, distribute_multi_gpu, holds a
        // DeviceGuard for GPU 0 that restores on its own return.)
        eprintln!(
            "Multi-GPU: split QKV weights for {} attn layers across {} GPUs",
            attn_layer_indices.len(),
            num_gpus
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_bqnt_path_with_env;
    use std::path::Path;

    fn defer_rm(p: std::path::PathBuf) -> impl Drop {
        struct D(std::path::PathBuf);
        impl Drop for D { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
        D(p)
    }

    /// explicit BQNT_PATH env value wins over auto-derive.
    #[test]
    fn resolve_explicit_env_wins() {
        let explicit = std::env::temp_dir().join("braidinfer_abuf_explicit.q4.bqnt");
        std::fs::write(&explicit, b"").unwrap();
        let _rm = defer_rm(explicit.clone());
        let result = resolve_bqnt_path_with_env(
            Some(explicit.to_str().unwrap().to_string()),
            Path::new("/nonexistent/model_dir"),
        );
        assert_eq!(result.as_deref(), Some(explicit.as_path()));
    }

    /// no env → auto-derives {parent}/{model_dir_name}.q4.bqnt when it exists.
    #[test]
    fn resolve_auto_derive_when_no_env() {
        let parent = std::env::temp_dir();
        let model_dir = parent.join("mymodel_abuf_test");
        let expected = parent.join("mymodel_abuf_test.q4.bqnt");
        std::fs::write(&expected, b"").unwrap();
        let _rm = defer_rm(expected.clone());
        let result = resolve_bqnt_path_with_env(None, &model_dir);
        assert_eq!(result.as_deref(), Some(expected.as_path()));
    }

    /// returns None when neither env nor auto path resolves to an existing file.
    #[test]
    fn resolve_none_when_no_file() {
        let result = resolve_bqnt_path_with_env(
            None,
            Path::new("/nonexistent/no_bqnt_here"),
        );
        assert_eq!(result, None);
    }
}
