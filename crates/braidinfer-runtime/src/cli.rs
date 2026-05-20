//! Shared CLI helpers used by the `generate` and `chat` binaries.
//!
//! Resolves model paths from the HuggingFace cache, queries VRAM, and applies
//! the auto-detection rules for `MULTI_GPU` and `PERSISTENT`.

use std::path::{Path, PathBuf};

use crate::bqnt::MmapBqnt;
use crate::config::{FfnType, ModelConfig};

/// Default HF snapshot used when no `MODEL` / `BQNT_PATH` is supplied.
pub const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

/// Strip recognized CLI flags from `argv` in place and return the parsed
/// values. Pilot helper for the eventual env-var → CLI-args migration
/// (braidinfer-wuf.16). Today this handles only `--audit-mtypes` to take
/// MTYPE_AUDIT off the env-var surface; future flags get added here.
///
/// The flag is removed from `argv` so positional args (prompt, model
/// path) keep their existing index semantics.
#[derive(Default, Debug)]
pub struct CliFlags {
    pub audit_mtypes: bool,
}

pub fn extract_cli_flags(argv: &mut Vec<String>) -> CliFlags {
    let mut flags = CliFlags::default();
    argv.retain(|a| {
        if a == "--audit-mtypes" {
            flags.audit_mtypes = true;
            false
        } else {
            true
        }
    });
    // Back-compat: legacy BRAIDINFER_MTYPE_AUDIT env var still honored.
    if flags.audit_mtypes && std::env::var("BRAIDINFER_MTYPE_AUDIT").is_err() {
        unsafe { std::env::set_var("BRAIDINFER_MTYPE_AUDIT", "1") };
    }
    flags
}

/// Resolve the HF snapshot directory that contains the tokenizer + config for
/// a given `.bqnt`. The bqnt records the model_name in its metadata; this
/// either points directly at an absolute snapshot path or names a HF repo
/// whose snapshot we locate in `~/.cache/huggingface/hub`.
///
/// Snapshot selection prefers a directory containing `tokenizer.json` and
/// falls back to the lexicographically-first snapshot if none has one.
pub fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
    let bqnt = MmapBqnt::open(Path::new(bqnt_path)).ok()?;
    let model_name = bqnt.model_name()?;
    if model_name.starts_with('/') {
        let p = Path::new(&model_name);
        if p.is_dir() {
            return Some(model_name);
        }
        // Absolute path miss: the snapshot the bqnt was quantized from has
        // moved or been GC'd. Recover by extracting the `models--<repo>`
        // segment and picking any snapshot present under that repo.
        if let Some(repo_seg) = model_name
            .split('/')
            .find(|seg| seg.starts_with("models--"))
        {
            return pick_snapshot_from_repo_seg(repo_seg);
        }
        return None;
    }
    let hf_name = model_name.replace('/', "--");
    pick_snapshot_from_repo_seg(&format!("models--{hf_name}"))
}

fn pick_snapshot_from_repo_seg(repo_seg: &str) -> Option<String> {
    let cache_dir = dirs::home_dir()?
        .join(".cache/huggingface/hub")
        .join(repo_seg)
        .join("snapshots");
    let mut snapshots: Vec<_> = std::fs::read_dir(&cache_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    snapshots.sort_by_key(|e| e.file_name());
    snapshots
        .iter()
        .find(|e| e.path().join("tokenizer.json").exists())
        .or_else(|| snapshots.first())
        .map(|e| e.path().to_string_lossy().to_string())
}

/// Free VRAM (bytes) on every visible GPU, in device-ID order.
pub fn vram_free_per_gpu() -> Vec<usize> {
    let mut count: i32 = 0;
    unsafe { braidinfer_hip::ffi::hipGetDeviceCount(&mut count) };
    (0..count)
        .map(|i| {
            unsafe { braidinfer_hip::ffi::hipSetDevice(i) };
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { braidinfer_hip::ffi::hipMemGetInfo(&mut free, &mut total) };
            free
        })
        .collect()
}

/// (used_mb, total_mb) for the current device.
pub fn vram_usage_mb() -> (f64, f64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        braidinfer_hip::ffi::hipMemGetInfo(&mut free, &mut total);
    }
    let used = (total - free) as f64 / (1024.0 * 1024.0);
    let total_mb = total as f64 / (1024.0 * 1024.0);
    (used, total_mb)
}

/// Result of resolving the user's model argument: the HF snapshot directory
/// to load the tokenizer/config from, and an optional `.bqnt` path that
/// overrides `BQNT_PATH`.
pub struct ResolvedModel {
    pub model_dir: PathBuf,
    pub bqnt_override: Option<String>,
}

/// Resolve `model_arg` (typically `MODEL` env or positional arg). Exits the
/// process with a clear error if the `.bqnt` cannot be located in HF cache.
///
/// Also sets `BQNT_PATH` in the environment when a `.bqnt` was supplied, so
/// downstream code that reads `BQNT_PATH` sees the user's choice.
pub fn resolve_model_arg(model_arg: Option<String>) -> ResolvedModel {
    let (model_path, bqnt_override) = match model_arg {
        Some(ref p) if p.ends_with(".bqnt") => {
            let hf_dir = resolve_hf_dir(p).unwrap_or_else(|| {
                eprintln!("Could not resolve HF cache dir for {p}");
                std::process::exit(1);
            });
            (hf_dir, Some(p.clone()))
        }
        Some(p) => (p, None),
        None => {
            let from_bqnt = std::env::var("BQNT_PATH")
                .ok()
                .and_then(|p| resolve_hf_dir(&p));
            (
                from_bqnt.unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string()),
                None,
            )
        }
    };
    if let Some(ref bqnt_path) = bqnt_override {
        unsafe {
            std::env::set_var("BQNT_PATH", bqnt_path);
        }
    }
    let model_dir = PathBuf::from(model_path);
    if !model_dir.exists() {
        eprintln!("Model not found at {}", model_dir.display());
        std::process::exit(1);
    }
    ResolvedModel {
        model_dir,
        bqnt_override,
    }
}

/// True if any layer of the model at `model_dir` is MoE.
pub fn detect_moe(model_dir: &Path) -> bool {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return false;
    }
    ModelConfig::from_config_json(&config_path)
        .ok()
        .map_or(false, |c| {
            c.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }))
        })
}

/// Size in bytes of the `.bqnt` file associated with `model_dir`. Uses
/// `BQNT_PATH` if set, otherwise auto-derives `<parent>/<model_name>.q4.bqnt`.
/// Returns 0 when neither path resolves to a real file.
pub fn bqnt_size_bytes(model_dir: &Path) -> u64 {
    std::env::var("BQNT_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            model_dir.file_name().map(|n| {
                model_dir
                    .parent()
                    .unwrap_or(model_dir)
                    .join(format!("{}.q4.bqnt", n.to_string_lossy()))
            })
        })
        .and_then(|p| std::fs::metadata(&p).ok())
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Apply auto-detection for `MULTI_GPU` and `PERSISTENT` env vars based on
/// model size vs single-GPU VRAM and MoE presence. Idempotent — only sets a
/// var if the user did not. Emits a one-line stderr notice when it does.
///
/// Returns `(multi_gpu, persistent)` so callers can branch without re-reading
/// the env afterwards.
pub fn apply_auto_modes(model_dir: &Path) -> (bool, bool) {
    let bqnt_size = bqnt_size_bytes(model_dir);
    let free_per_gpu = vram_free_per_gpu();
    let single_gpu_vram = free_per_gpu.first().copied().unwrap_or(0);
    // 15% headroom — MULTI_GPU when model exceeds 85% of a single card.
    let multi_gpu = std::env::var("MULTI_GPU").is_ok()
        || (bqnt_size > 0
            && bqnt_size as usize > single_gpu_vram * 85 / 100
            && free_per_gpu.len() > 1);
    if multi_gpu && std::env::var("MULTI_GPU").is_err() {
        eprintln!(
            "Auto: MULTI_GPU enabled (model {:.1}GB > single-GPU {:.1}GB free)",
            bqnt_size as f64 / 1e9,
            single_gpu_vram as f64 / 1e9,
        );
        unsafe { std::env::set_var("MULTI_GPU", "1") };
    }

    let _has_moe = detect_moe(model_dir);
    // PERSISTENT: enabled by default for all model architectures.
    //   - Multi-GPU: required (only the cooperative megakernel path supports
    //     P2P worker dispatch).
    //   - Single-GPU non-MoE: 2.1× speedup vs the paged path.
    //   - Single-GPU MoE: validated 2026-05-20 (qwen35_35b_a3b.q4 -g 1 N=5
    //     PERSISTENT=1: 5/5 PASS @ 14-18 tok/s with coherent output).
    // Opt out with PERSISTENT=0 (still needed for KV_QUANT=1 and
    // WEIGHT_QUANT=rnf4/mixed configurations until those land on the
    // megakernel path).
    let persistent = std::env::var("PERSISTENT").as_deref() != Ok("0");
    if persistent && std::env::var("PERSISTENT").is_err() {
        let reason = if multi_gpu {
            "required for multi-GPU"
        } else {
            "default for all model architectures"
        };
        eprintln!("Auto: PERSISTENT enabled ({reason})");
        unsafe { std::env::set_var("PERSISTENT", "1") };
    }

    validate_env_combos(multi_gpu, persistent);
    (multi_gpu, persistent)
}

/// Reject combinations of env vars that are guaranteed to crash at runtime
/// or load time. Surfacing these at startup (after MULTI_GPU/PERSISTENT have
/// been resolved) gives the user an immediate, actionable error instead of
/// a panic or InvalidConfig after several seconds of model load.
fn validate_env_combos(multi_gpu: bool, persistent: bool) {
    // WEIGHT_QUANT=rnf4|mixed is incompatible with the megakernel path
    // (LinearWeight::as_bf16_ptr panics on Packed). The megakernel is the
    // backbone of persistent + multi-GPU paths. Fail fast.
    match std::env::var("WEIGHT_QUANT").as_deref() {
        Ok("rnf4") | Ok("mixed") if persistent => {
            eprintln!(
                "Error: WEIGHT_QUANT={} is not supported on the megakernel path \
                 (auto-enabled via PERSISTENT=1 / multi-GPU). Either unset \
                 WEIGHT_QUANT, or set PERSISTENT=0 and run a single-GPU MoE model.",
                std::env::var("WEIGHT_QUANT").unwrap_or_default()
            );
            std::process::exit(1);
        }
        _ => {}
    }
    // KV_QUANT is not yet wired through the persistent or multi-GPU paths
    // (paged-only). Both binaries can hit this; centralize the guard here
    // so generate doesn't also silently produce wrong output.
    let kv_quant = std::env::var("KV_QUANT").as_deref() == Ok("1");
    if kv_quant && multi_gpu {
        eprintln!("Error: KV_QUANT=1 is not supported with MULTI_GPU=1");
        eprintln!("  KV quantization only works in single-GPU mode.");
        std::process::exit(1);
    }
    if kv_quant && persistent {
        eprintln!("Error: KV_QUANT=1 is not yet supported with PERSISTENT=1");
        eprintln!("  Either unset KV_QUANT or set PERSISTENT=0.");
        std::process::exit(1);
    }
}
