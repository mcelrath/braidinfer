//! braidinferd — long-running inference daemon (Phase 3 scaffolding,
//! braidinfer-wks epic).
//!
//! This binary establishes the daemon-shaped structure that Phase 4
//! (RPC protocol) and Phase 5 (multi-session client cutover) build on.
//! It owns the HIP context and the persistent_worker cooperative
//! kernels; client processes will eventually drive inference via Unix
//! socket RPC instead of via this binary's CLI.
//!
//! Phase 3 surface is intentionally minimal:
//!   - load one model
//!   - run one inference from CLI args
//!   - handle SIGTERM gracefully (Drop runs → workers shut down)
//!   - print tokens to stdout
//!
//! Indistinguishable from `bin/generate` from a user POV. The structural
//! difference is that it goes through `InProcessDispatch` (the trait
//! abstraction Phase 2 introduced) rather than calling `Model` directly,
//! so when Phase 4 adds the Unix socket the dispatch path is already
//! the same surface the daemon will expose to clients.
//!
//! Usage:
//!   MODEL=models/qwen35_2b.q4.bqnt MAX_TOKENS=50 \
//!     python3 scripts/launch-gpu.py --timeout 600 -- \
//!     target/release/braidinferd "Hello"

use std::path::Path;

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::dispatch_service::{
    DecodeRequest, DispatchService, InProcessDispatch, SessionId,
};
use braidinfer_runtime::generate::{TokenConfig, load_tokenizer};
use braidinfer_runtime::model::Model;

fn resolve_model_dir(model_path: &str) -> String {
    if !model_path.ends_with(".bqnt") {
        return model_path.to_string();
    }
    let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(Path::new(model_path)).expect("open bqnt");
    let model_name = bqnt.model_name().expect("bqnt missing model_name");
    if model_name.starts_with('/') && Path::new(&model_name).is_dir() {
        return model_name;
    }
    let hf_name = model_name.replace('/', "--");
    let cache_dir = dirs::home_dir()
        .expect("home dir")
        .join(".cache/huggingface/hub")
        .join(format!("models--{hf_name}"))
        .join("snapshots");
    std::fs::read_dir(&cache_dir)
        .expect("read snapshots")
        .filter_map(|e| e.ok())
        .find(|e| e.path().join("tokenizer.json").exists())
        .map(|e| e.path().to_string_lossy().to_string())
        .expect("no snapshot with tokenizer found")
}

fn main() {
    let model_path = std::env::var("MODEL").expect("set MODEL=<bqnt or hf-dir path>");
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The history of computing began long before".to_string());
    let max_tokens: u32 = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let model_dir_path = resolve_model_dir(&model_path);
    if model_path.ends_with(".bqnt") {
        unsafe {
            std::env::set_var("BQNT_PATH", &model_path);
        }
    }

    eprintln!("[braidinferd] loading model from {model_dir_path}");
    let device = DeviceId(0);
    let model_dir = Path::new(&model_dir_path);
    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let model = Model::load(model_dir, device).expect("load model");
    let mut dispatch = InProcessDispatch::new(model);

    eprintln!("[braidinferd] model loaded, starting inference");
    let session: SessionId = dispatch.create_session().expect("create session");

    // Tokenize prompt
    let encoding = tokenizer.encode(prompt.as_str(), false).expect("tokenize");
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[braidinferd] prompt: {} tokens", prompt_ids.len());

    // Prefill
    let mut next_token = if prompt_ids.is_empty() {
        eprintln!("[braidinferd] empty prompt; nothing to do");
        dispatch.drop_session(session);
        return;
    } else if prompt_ids.len() == 1 {
        // For single-token prompts the trait's prefill still runs Model::prefill,
        // which handles n=1 internally.
        dispatch.prefill(session, &prompt_ids).expect("prefill")
    } else {
        dispatch.prefill(session, &prompt_ids).expect("prefill")
    };

    let mut position = prompt_ids.len() as u32;
    let mut generated = 0u32;
    let start = std::time::Instant::now();

    while generated < max_tokens {
        if token_config.is_stop_token(next_token) {
            break;
        }
        let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{piece}");
        // Don't auto-flush per token; keep stdout buffered for now (Phase 4 will
        // use the RPC frame stream which has its own per-token framing).

        let next = dispatch
            .decode_step_batch(&[DecodeRequest {
                session,
                input_token: next_token,
                position,
            }])
            .expect("decode");
        next_token = next[0];
        position += 1;
        generated += 1;
    }
    println!();
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[braidinferd] {generated} tokens in {elapsed:.3}s = {:.1} tok/s",
        generated as f64 / elapsed
    );

    dispatch.drop_session(session);
    // dispatch (and the model it owns) drops here, shutting down the
    // persistent worker. SIGTERM during the loop unwinds via the
    // dispatch_batch panic path and Drop cleans up the same way.
}
