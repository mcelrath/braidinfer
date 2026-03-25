use std::path::Path;
use std::time::Instant;
use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{greedy_generate, load_tokenizer};
use braidinfer_runtime::model::Qwen35Model;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

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

    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found at {}", MODEL_DIR);
        std::process::exit(1);
    }

    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let device = DeviceId(0);
    let mut model = Qwen35Model::load(model_dir, device).expect("load model");

    let start = Instant::now();
    let result = greedy_generate(&mut model, &tokenizer, &prompt, max_tokens)
        .expect("greedy_generate");

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
