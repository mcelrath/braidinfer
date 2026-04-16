use braidinfer_core::types::DeviceId;
use braidinfer_runtime::config::FfnType;
use braidinfer_runtime::model::Model;
use std::path::Path;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

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

fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
    let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(std::path::Path::new(bqnt_path)).ok()?;
    let model_name = bqnt.model_name()?;
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

fn load_model(model_dir: &Path, multi_gpu: bool) -> Model {
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut model = Model::load_with_max_seq_len(model_dir, device, max_seq_len)
        .expect("load model");
    if multi_gpu {
        model.enable_multi_gpu().expect("enable multi-GPU");
    }
    model
}

fn bench_decode(model: &mut Model, warmup: usize, runs: usize) {
    println!("=== Decode benchmark ===");
    let token_id = 9906u32; // "Hello"

    // Warmup: advance position to warmup, let the model run without timing
    for p in 0..warmup as u32 {
        model.decode_step_token(token_id, p).expect("decode");
    }

    // Timed: run `runs` consecutive decode steps and time each
    let mut pos = warmup as u32;
    let mut times_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        model.decode_step_token(token_id, pos).expect("decode");
        times_ns.push(t0.elapsed().as_nanos());
        pos += 1;
    }

    let avg_ms = times_ns.iter().sum::<u128>() as f64 / runs as f64 / 1_000_000.0;
    let min_ms = times_ns.iter().min().unwrap().clone() as f64 / 1_000_000.0;
    let max_ms = times_ns.iter().max().unwrap().clone() as f64 / 1_000_000.0;
    println!(
        "  positions {}-{}  avg={avg_ms:.2}ms  min={min_ms:.2}ms  max={max_ms:.2}ms  {:.1} tok/s",
        warmup, warmup + runs - 1, 1000.0 / avg_ms,
    );
}

fn bench_prefill(model: &mut Model, token_counts: &[usize]) {
    // MUST be called before bench_decode (before persistent worker starts) so reset_state works.
    // Calls model.prefill() which uses batched megakernel compile_prefill path.
    println!("=== Prefill benchmark (batched megakernel) ===");
    let token_id = 9906u32;

    // Warmup
    let warmup_tokens: Vec<u32> = vec![token_id; 8];
    model.reset_state().expect("reset");
    model.prefill(&warmup_tokens).expect("prefill warmup");

    for &n in token_counts {
        let tokens: Vec<u32> = vec![token_id; n];
        model.reset_state().expect("reset");
        let t0 = Instant::now();
        model.prefill(&tokens).expect("prefill");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let toks_per_sec = n as f64 / (elapsed_ms / 1000.0);
        println!("  n={n:4}  {elapsed_ms:7.1}ms  {toks_per_sec:6.1} tok/s");
    }
    model.reset_state().expect("reset");
}

fn bench_coherence(model: &mut Model, prompt_len: usize) {
    // MUST be called before bench_decode (before persistent worker starts) so reset_state works.
    // Validates batched prefill logits == sequential decode logits at the last token position.
    println!("=== Coherence test (batched prefill vs sequential decode) ===");
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 9906 + (i % 100)).collect();

    let top10 = |logits: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(10);
        idx
    };

    // Sequential reference: use paged path directly so persistent worker is NOT launched,
    // allowing reset_state() afterward without deadlock.
    model.reset_state().expect("reset");
    let mut seq_logits = vec![];
    for (i, &tok) in prompt.iter().enumerate() {
        seq_logits = model.decode_step_paged(tok, i as u32).expect("seq decode");
    }
    let seq_top = top10(&seq_logits);

    // Batched prefill
    model.reset_state().expect("reset");
    let prefill_logits = model.prefill(&prompt).expect("batched prefill");
    let prefill_top = top10(&prefill_logits);

    let matches = seq_top.iter().zip(prefill_top.iter()).filter(|(a, b)| a == b).count();
    let pass = matches >= 8; // allow ≥8/10 match (floating-point ordering may differ slightly)
    println!("  top-10 match: {matches}/10  [{}]", if pass { "PASS" } else { "FAIL" });
    println!("  seq top-5:     {:?}", &seq_top[..5.min(seq_top.len())]);
    println!("  prefill top-5: {:?}", &prefill_top[..5.min(prefill_top.len())]);

    model.reset_state().expect("reset");
}

fn main() {
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
            let from_bqnt = std::env::var("BQNT_PATH")
                .ok()
                .and_then(|p| resolve_hf_dir(&p));
            (from_bqnt.unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string()), None)
        }
    };
    if let Some(ref bqnt_path) = bqnt_override {
        unsafe { std::env::set_var("BQNT_PATH", bqnt_path) };
    }

    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("Model not found at {model_path}");
        std::process::exit(1);
    }

    // Auto-detect MoE, MULTI_GPU, PERSISTENT (same logic as generate binary)
    let config_path = model_dir.join("config.json");
    let has_moe = config_path
        .exists()
        .then(|| braidinfer_runtime::config::ModelConfig::from_config_json(&config_path).ok())
        .flatten()
        .map_or(false, |c| c.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. })));

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

    let free_per_gpu = vram_free_per_gpu();
    let single_gpu_vram = free_per_gpu.first().copied().unwrap_or(0);
    let multi_gpu = std::env::var("MULTI_GPU").is_ok()
        || (bqnt_size_bytes > 0 && bqnt_size_bytes as usize > single_gpu_vram * 85 / 100);
    if multi_gpu && std::env::var("MULTI_GPU").is_err() {
        eprintln!(
            "Auto: MULTI_GPU (model {:.1}GB > single-GPU {:.1}GB free)",
            bqnt_size_bytes as f64 / 1e9,
            single_gpu_vram as f64 / 1e9,
        );
        unsafe { std::env::set_var("MULTI_GPU", "1") };
    }

    let persistent = std::env::var("PERSISTENT").as_deref() == Ok("1") || multi_gpu || !has_moe;
    if persistent && std::env::var("PERSISTENT").is_err() {
        let reason = if multi_gpu { "required for multi-GPU" } else { "non-MoE model" };
        eprintln!("Auto: PERSISTENT ({reason})");
        unsafe { std::env::set_var("PERSISTENT", "1") };
    }

    let mut model = load_model(model_dir, multi_gpu);
    eprintln!("Model loaded: {model_path}");

    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let runs: usize = std::env::var("BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);

    // Coherence and prefill FIRST: must run before persistent worker starts.
    bench_coherence(&mut model, 8);
    bench_prefill(&mut model, &[8, 32, 128, 512]);
    bench_decode(&mut model, warmup, runs);
}
