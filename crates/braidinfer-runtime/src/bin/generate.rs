use std::path::Path;
use std::time::Instant;
use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{chat_generate, greedy_generate, load_tokenizer, TokenConfig};
use braidinfer_runtime::model::Model;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn vram_usage_mb() -> (f64, f64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe { braidinfer_hip::ffi::hipMemGetInfo(&mut free, &mut total); }
    let used = (total - free) as f64 / (1024.0 * 1024.0);
    let total_mb = total as f64 / (1024.0 * 1024.0);
    (used, total_mb)
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

    let model_path = std::env::var("MODEL").ok()
        .or_else(|| {
            // Auto-resolve model dir from BQNT_PATH metadata
            std::env::var("BQNT_PATH").ok().and_then(|bqnt_path| {
                let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(std::path::Path::new(&bqnt_path)).ok()?;
                let model_name = bqnt.model_name()?;
                let hf_name = model_name.replace('/', "--");
                let cache_dir = dirs::home_dir()?
                    .join(".cache/huggingface/hub")
                    .join(format!("models--{hf_name}"))
                    .join("snapshots");
                std::fs::read_dir(&cache_dir).ok()?
                    .filter_map(|e| e.ok())
                    .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .map(|e| e.path().to_string_lossy().to_string())
            })
        })
        .unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string());
    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("Model not found at {}", model_path);
        std::process::exit(1);
    }

    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN").ok().and_then(|v| v.parse().ok());
    let mut model = Model::load_with_max_seq_len(model_dir, device, max_seq_len).expect("load model");

    // Enable multi-GPU if NUM_GPUS > 1 or MULTI_GPU is set
    if std::env::var("MULTI_GPU").is_ok() {
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
        chat_generate(&mut model, &tokenizer, &token_config, &prompt, None, max_tokens)
    }.expect("generate");

    let elapsed = start.elapsed().as_secs_f64();
    let n_tokens = result.tokens.len();

    for piece in &result.text_pieces {
        print!("{}", piece);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();

    let tok_per_sec = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 };
    eprintln!("{} tokens in {:.3}s = {:.1} tok/s", n_tokens, elapsed, tok_per_sec);
}
