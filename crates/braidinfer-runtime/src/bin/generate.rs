use braidinfer_core::types::DeviceId;
use braidinfer_runtime::config::FfnType;
use braidinfer_runtime::generate::{TokenConfig, chat_generate, greedy_generate, load_tokenizer};
use braidinfer_runtime::model::Model;
use std::path::Path;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn vram_usage_mb() -> (f64, f64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        braidinfer_hip::ffi::hipMemGetInfo(&mut free, &mut total);
    }
    let used = (total - free) as f64 / (1024.0 * 1024.0);
    let total_mb = total as f64 / (1024.0 * 1024.0);
    (used, total_mb)
}

/// Query free VRAM (bytes) across all available GPUs.
fn vram_free_per_gpu() -> Vec<usize> {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "Hello, world!".to_string()
    };

    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let raw_mode = std::env::var("RAW").is_ok();

    fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
        let bqnt =
            braidinfer_runtime::bqnt::MmapBqnt::open(std::path::Path::new(bqnt_path)).ok()?;
        let model_name = bqnt.model_name()?;
        // If model_name is an absolute path that exists as a directory, use it directly
        if model_name.starts_with('/') {
            let p = std::path::Path::new(&model_name);
            if p.is_dir() {
                return Some(model_name);
            }
        }
        let hf_name = model_name.replace('/', "--");
        let cache_dir = dirs::home_dir()?
            .join(".cache/huggingface/hub")
            .join(format!("models--{hf_name}"))
            .join("snapshots");
        // Prefer snapshot that has tokenizer.json; fall back to first dir found
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

    // If MODEL ends with .bqnt, use it as BQNT_PATH and resolve HF dir for tokenizer
    let (model_path, bqnt_override) = match std::env::var("MODEL").ok() {
        Some(ref p) if p.ends_with(".bqnt") => {
            let hf_dir = resolve_hf_dir(p).unwrap_or_else(|| {
                eprintln!("Could not resolve HF cache dir for {p}");
                std::process::exit(1);
            });
            (hf_dir, Some(p.clone()))
        }
        Some(p) => (p, None),
        None => {
            // Auto-resolve model dir from BQNT_PATH metadata
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
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("Model not found at {}", model_path);
        std::process::exit(1);
    }

    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok());

    // Parse model config to detect MoE (needed for PERSISTENT auto-detection).
    let config_path = model_dir.join("config.json");
    let has_moe = config_path
        .exists()
        .then(|| braidinfer_runtime::config::ModelConfig::from_config_json(&config_path).ok())
        .flatten()
        .map_or(false, |c| c.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. })));

    // Determine bqnt file size (MODEL env var bqnt or auto-derived path).
    let bqnt_size_bytes: u64 = std::env::var("BQNT_PATH")
        .ok()
        .map(std::path::PathBuf::from)
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
        .unwrap_or(0);

    // Auto-detect MULTI_GPU: enable when model exceeds single-GPU VRAM (with 15% headroom).
    let free_per_gpu = vram_free_per_gpu();
    let single_gpu_vram = free_per_gpu.first().copied().unwrap_or(0);
    let multi_gpu = std::env::var("MULTI_GPU").is_ok()
        || (bqnt_size_bytes > 0
            && bqnt_size_bytes as usize > single_gpu_vram * 85 / 100
            && free_per_gpu.len() > 1);
    if multi_gpu && std::env::var("MULTI_GPU").is_err() {
        eprintln!(
            "Auto: MULTI_GPU enabled (model {:.1}GB > single-GPU {:.1}GB free)",
            bqnt_size_bytes as f64 / 1e9,
            single_gpu_vram as f64 / 1e9,
        );
        unsafe { std::env::set_var("MULTI_GPU", "1") };
    }

    // Auto-detect PERSISTENT: enabled for non-MoE single-GPU (2.1x speedup) and all multi-GPU.
    // Single-GPU MoE (hybrid GDN/SSM models like nemotron_cascade) not yet validated in
    // persistent mode — keep them on the paged path until tested.
    let persistent = std::env::var("PERSISTENT").as_deref() == Ok("1")
        || multi_gpu
        || !has_moe;
    if persistent && std::env::var("PERSISTENT").is_err() {
        let reason = if multi_gpu { "required for multi-GPU" } else { "non-MoE model" };
        eprintln!("Auto: PERSISTENT enabled ({reason})");
        unsafe { std::env::set_var("PERSISTENT", "1") };
    }

    let mut model =
        Model::load_with_max_seq_len(model_dir, device, max_seq_len).expect("load model");

    if multi_gpu {
        model.enable_multi_gpu().expect("enable multi-GPU");
    }

    let (vram_used, vram_total) = vram_usage_mb();
    eprintln!("VRAM after load: {:.0}/{:.0} MB", vram_used, vram_total);
    eprintln!("max_seq_len: {}", model.config().max_seq_len);
    eprintln!("stop tokens: {:?}", token_config.eos_token_ids);

    let start = Instant::now();
    let result = if raw_mode {
        greedy_generate(&mut model, &tokenizer, &token_config, &prompt, max_tokens)
    } else {
        chat_generate(
            &mut model,
            &tokenizer,
            &token_config,
            &prompt,
            None,
            max_tokens,
        )
    }
    .expect("generate");

    let elapsed = start.elapsed().as_secs_f64();
    let n_tokens = result.tokens.len();

    for piece in &result.text_pieces {
        print!("{}", piece);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();

    let tok_per_sec = if elapsed > 0.0 {
        n_tokens as f64 / elapsed
    } else {
        0.0
    };
    eprintln!(
        "{} tokens in {:.3}s = {:.1} tok/s",
        n_tokens, elapsed, tok_per_sec
    );
}
