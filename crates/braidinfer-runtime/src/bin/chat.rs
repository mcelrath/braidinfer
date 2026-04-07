use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, BufReader};

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{
    ChatMessage, TokenConfig, apply_chat_template_thinking, load_tokenizer,
};
use braidinfer_runtime::model::Model;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[tokio::main]
async fn main() {
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let system_prompt = std::env::var("SYSTEM").ok();

    fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
        let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(Path::new(bqnt_path)).ok()?;
        let model_name = bqnt.model_name()?;
        let hf_name = model_name.replace('/', "--");
        let cache_dir = dirs::home_dir()?
            .join(".cache/huggingface/hub")
            .join(format!("models--{hf_name}"))
            .join("snapshots");
        std::fs::read_dir(&cache_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.path().to_string_lossy().to_string())
    }

    let model_arg = std::env::var("MODEL")
        .ok()
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
    let mut model =
        Model::load_with_max_seq_len(model_dir, device, max_seq_len).expect("load model");

    if std::env::var("MULTI_GPU").is_ok() {
        model.enable_multi_gpu().expect("enable multi-GPU");
    }

    eprintln!("braidinfer chat (max_tokens={max_tokens}, ^D to quit)");
    if let Some(sys) = &system_prompt {
        eprintln!("system: {sys}");
    }

    let enable_thinking = std::env::var("THINK").is_ok();
    let mut history: Vec<(String, String)> = Vec::new();
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
            messages.push(ChatMessage {
                role: "system",
                content: sys,
            });
        }
        for (u, a) in &history {
            messages.push(ChatMessage {
                role: "user",
                content: u,
            });
            messages.push(ChatMessage {
                role: "assistant",
                content: a,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: &user_input,
        });

        let prompt_ids = match apply_chat_template_thinking(
            &tokenizer,
            &token_config,
            &messages,
            enable_thinking,
        ) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("template error: {e}");
                continue;
            }
        };

        // Full re-prefill each turn (incremental requires exact token tracking
        // which breaks when re-encoding the assistant response tokenizes differently)
        model.reset_state().expect("reset");

        let start = Instant::now();
        let n_prompt = prompt_ids.len();
        let last_logits = if n_prompt <= 1 {
            match model.decode_step(prompt_ids[0], 0) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("prefill error: {e}");
                    continue;
                }
            }
        } else {
            match model.prefill(&prompt_ids) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("prefill error: {e}");
                    continue;
                }
            }
        };

        let prefill_elapsed = start.elapsed().as_secs_f64();
        eprintln!("[prefill {n_prompt} tokens in {prefill_elapsed:.2}s]");

        // Streaming decode — print each token as it's generated
        let mut next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let mut position = n_prompt as u32;
        let mut response = String::new();
        let mut n_gen = 0u32;
        let mut in_thinking = false;
        let decode_start = Instant::now();

        // Detect <think> and </think> tokens for thinking block display
        let think_start_id = tokenizer.token_to_id("<think>");
        let think_end_id = tokenizer.token_to_id("</think>");

        for _ in 0..max_tokens {
            if token_config.is_stop_token(next_token) {
                break;
            }

            // Handle thinking block display
            if Some(next_token) == think_start_id {
                in_thinking = true;
                eprint!("\x1b[2m<think>"); // dim text for thinking
                io::stderr().flush().unwrap();
            } else if Some(next_token) == think_end_id {
                in_thinking = false;
                eprintln!("</think>\x1b[0m"); // reset formatting
            } else {
                let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
                if in_thinking {
                    eprint!("{piece}"); // thinking goes to stderr (dim)
                    io::stderr().flush().unwrap();
                } else {
                    print!("{piece}");
                    io::stdout().flush().unwrap();
                    response.push_str(&piece);
                }
            }
            n_gen += 1;

            next_token = match model.decode_step_token(next_token, position) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("\ngenerate error: {e}");
                    break;
                }
            };
            position += 1;
        }
        println!();

        let decode_elapsed = decode_start.elapsed().as_secs_f64();
        eprintln!(
            "[{n_gen} tokens in {decode_elapsed:.2}s = {:.1} tok/s]",
            n_gen as f64 / decode_elapsed
        );

        history.push((user_input, response));
    }
}
