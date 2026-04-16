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

fn bench_decode(model: &mut Model, warmup: usize, runs: usize, positions: &[u32]) {
    println!("=== Decode benchmark ===");
    let token_id = 9906u32; // "Hello"

    for &pos in positions {
        // Warmup
        for _ in 0..warmup {
            model.reset_state().expect("reset");
            // prefill up to pos
            for p in 0..pos {
                model.decode_step_token(token_id, p).expect("decode");
            }
        }

        // Timed runs
        let mut total_ns = 0u128;
        for _ in 0..runs {
            let t0 = Instant::now();
            model.decode_step_token(token_id, pos).expect("decode");
            total_ns += t0.elapsed().as_nanos();
        }

        let avg_ms = total_ns as f64 / runs as f64 / 1_000_000.0;
        let toks_per_sec = 1000.0 / avg_ms;
        println!("  pos={pos:4}  avg={avg_ms:7.2}ms  {toks_per_sec:6.1} tok/s");
    }
}

fn bench_prefill(model: &mut Model, token_counts: &[usize]) {
    println!("=== Prefill benchmark ===");
    let token_id = 9906u32;

    for &n in token_counts {
        let tokens: Vec<u32> = vec![token_id; n];

        // Warmup
        model.reset_state().expect("reset");
        model.prefill(&tokens).expect("prefill");

        // Timed
        model.reset_state().expect("reset");
        let t0 = Instant::now();
        model.prefill(&tokens).expect("prefill");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let toks_per_sec = n as f64 / (elapsed_ms / 1000.0);
        println!("  n={n:4}  {elapsed_ms:7.2}ms  {toks_per_sec:7.1} tok/s");
    }
}

fn bench_coherence(model: &mut Model, prompt_len: usize, gen_len: usize) {
    println!("=== Coherence test (determinism) ===");
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 9906 + (i % 100)).collect();

    let generate = |m: &mut Model| -> Vec<u32> {
        m.reset_state().expect("reset");
        m.prefill(&prompt).expect("prefill");
        let mut tokens = Vec::with_capacity(gen_len);
        let mut tok = prompt[prompt_len - 1];
        for p in prompt_len..(prompt_len + gen_len) {
            tok = m.decode_step_token(tok, p as u32).expect("decode");
            tokens.push(tok);
        }
        tokens
    };

    let run1 = generate(model);
    let run2 = generate(model);

    let matches = run1.iter().zip(run2.iter()).filter(|(a, b)| a == b).count();
    let pass = matches == gen_len;
    println!("  runs match: {matches}/{gen_len}  [{}]", if pass { "PASS" } else { "FAIL" });
    print!("  tokens: ");
    for (i, &t) in run1.iter().take(20).enumerate() {
        if i > 0 { print!(", "); }
        print!("{t}");
    }
    if run1.len() > 20 { print!(", ..."); }
    println!();
    if !pass {
        print!("  run2:   ");
        for (i, &t) in run2.iter().take(20).enumerate() {
            if i > 0 { print!(", "); }
            print!("{t}");
        }
        if run2.len() > 20 { print!(", ..."); }
        println!();
    }
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

    bench_decode(&mut model, warmup, runs, &[0, 64, 256, 512]);
    bench_prefill(&mut model, &[8, 32, 128, 512]);
    bench_coherence(&mut model, 8, 32);
}
