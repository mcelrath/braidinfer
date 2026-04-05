use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, BufReader};

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{
    apply_chat_template, load_tokenizer, ChatMessage, TokenConfig,
};
use braidinfer_runtime::model::Model;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[tokio::main]
async fn main() {
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    let system_prompt = std::env::var("SYSTEM").ok();

    fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
        let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(Path::new(bqnt_path)).ok()?;
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
    }

    let model_arg = std::env::var("MODEL").ok()
        .or_else(|| std::env::args().nth(1));

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
            let from_bqnt = std::env::var("BQNT_PATH").ok().and_then(|p| resolve_hf_dir(&p));
            (from_bqnt.unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string()), None)
        }
    };
    if let Some(ref bqnt_path) = bqnt_override {
        unsafe { std::env::set_var("BQNT_PATH", bqnt_path); }
    }

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

    if std::env::var("MULTI_GPU").is_ok() {
        model.enable_multi_gpu().expect("enable multi-GPU");
    }

    eprintln!("braidinfer chat (max_tokens={max_tokens}, ^D to quit)");
    if let Some(sys) = &system_prompt {
        eprintln!("system: {sys}");
    }

    let mut history: Vec<(String, String)> = Vec::new();
    let mut prev_ids: Vec<u32> = Vec::new(); // all tokens fed so far (for incremental prefill)
    let mut position: u32 = 0;
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        eprint!("> ");
        io::stderr().flush().unwrap();

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            _ => break,
        };
        let user_input = line.trim().to_string();
        if user_input.is_empty() {
            continue;
        }

        let mut messages: Vec<ChatMessage<'_>> = Vec::new();
        if let Some(sys) = &system_prompt {
            messages.push(ChatMessage { role: "system", content: sys });
        }
        for (u, a) in &history {
            messages.push(ChatMessage { role: "user", content: u });
            messages.push(ChatMessage { role: "assistant", content: a });
        }
        messages.push(ChatMessage { role: "user", content: &user_input });

        let full_ids = match apply_chat_template(&tokenizer, &token_config, &messages) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("template error: {e}");
                continue;
            }
        };

        // Incremental prefill: only feed tokens not already in KV cache
        let new_ids = &full_ids[prev_ids.len()..];

        let start = Instant::now();

        // Prefill new tokens only (KV cache retains previous turns)
        let last_logits = if new_ids.len() == 1 {
            match model.decode_step(new_ids[0], position) {
                Ok(l) => l,
                Err(e) => { eprintln!("prefill error: {e}"); continue; }
            }
        } else {
            // decode_step each new token (prefill expects position=0 for full sequence)
            let mut logits = Vec::new();
            for (i, &tok) in new_ids.iter().enumerate() {
                match model.decode_step(tok, position + i as u32) {
                    Ok(l) => logits = l,
                    Err(e) => { eprintln!("prefill error: {e}"); continue; }
                }
            }
            logits
        };
        position += new_ids.len() as u32;

        let prefill_elapsed = start.elapsed().as_secs_f64();
        eprintln!("[prefill {} new tokens ({} total) in {prefill_elapsed:.2}s]",
                  new_ids.len(), full_ids.len());

        // Streaming decode — print each token as it's generated
        let mut next_token = last_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32).unwrap_or(0);
        let mut response = String::new();
        let mut n_gen = 0u32;
        let decode_start = Instant::now();

        for _ in 0..max_tokens {
            if token_config.is_stop_token(next_token) { break; }

            let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
            print!("{piece}");
            io::stdout().flush().unwrap();
            response.push_str(&piece);
            n_gen += 1;

            next_token = match model.decode_step_token(next_token, position) {
                Ok(t) => t,
                Err(e) => { eprintln!("\ngenerate error: {e}"); break; }
            };
            position += 1;
        }
        println!();

        let decode_elapsed = decode_start.elapsed().as_secs_f64();
        eprintln!("[{n_gen} tokens in {decode_elapsed:.2}s = {:.1} tok/s]",
                  n_gen as f64 / decode_elapsed);

        history.push((user_input, response));

        // Re-template the full history to compute prev_ids for next turn's prefix match.
        // If the template adds end-of-turn tokens after the assistant response, feed them
        // to the model so the KV cache stays in sync.
        let mut full_messages: Vec<ChatMessage<'_>> = Vec::new();
        if let Some(sys) = &system_prompt {
            full_messages.push(ChatMessage { role: "system", content: sys });
        }
        for (u, a) in &history {
            full_messages.push(ChatMessage { role: "user", content: u });
            full_messages.push(ChatMessage { role: "assistant", content: a });
        }
        let expected = apply_chat_template(&tokenizer, &token_config, &full_messages)
            .unwrap_or_default();

        // Feed any end-of-turn template tokens the model hasn't seen yet
        if expected.len() > position as usize {
            for &tok in &expected[position as usize..] {
                let _ = model.decode_step(tok, position);
                position += 1;
            }
        }
        prev_ids = expected;
    }
}
