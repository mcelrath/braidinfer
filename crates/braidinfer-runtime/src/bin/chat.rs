use std::io::{self, Write};
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, BufReader};

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::cli::{apply_auto_modes, extract_cli_flags, resolve_model_arg};
use braidinfer_runtime::generate::{
    ChatMessage, apply_chat_template_thinking, load_tokenizer_and_config,
};
use braidinfer_runtime::model::Model;

#[tokio::main]
async fn main() {
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let system_prompt = std::env::var("SYSTEM").ok();

    let mut args: Vec<String> = std::env::args().collect();
    let _flags = extract_cli_flags(&mut args);
    let model_arg = std::env::var("MODEL").ok().or_else(|| args.into_iter().nth(1));
    let resolved = resolve_model_arg(model_arg);
    let model_dir = resolved.model_dir.as_path();

    let (tokenizer, token_config) =
        load_tokenizer_and_config(model_dir, resolved.bqnt_override.as_deref())
            .expect("load tokenizer/config");
    if token_config.chat_template().is_none() {
        eprintln!(
            "Error: {} has no chat_template — base model, not instruction-tuned.",
            model_dir.display()
        );
        eprintln!("  Use `cargo run --release --bin generate -- <prompt>` for raw prompting,");
        eprintln!("  or load an instruction-tuned variant (e.g. *-Instruct).");
        std::process::exit(1);
    }
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok());

    let multi_gpu = apply_auto_modes(model_dir);

    if std::env::var("KV_QUANT").as_deref() == Ok("1") {
        eprintln!("KV_QUANT enabled (residual_pc int4)");
    }

    let mut model =
        Model::load_with_max_seq_len(model_dir, device, max_seq_len).expect("load model");

    if multi_gpu || model.has_moe() {
        if let Err(e) = model.enable_distributed_moe() {
            let per_gpu_free_mb: Vec<f64> =
                braidinfer_runtime::cli::vram_free_per_gpu()
                    .iter()
                    .map(|&b| b as f64 / (1024.0 * 1024.0))
                    .collect();
            eprintln!("ERROR: enable_multi_gpu failed: {e:?}");
            eprintln!("  Per-GPU free VRAM (MB): {:?}", per_gpu_free_mb);
            eprintln!("  Hints:");
            eprintln!("    - increase GPU count (try -g 4 or -g 8)");
            eprintln!("    - reduce MAX_SEQ_LEN (e.g. MAX_SEQ_LEN=4096)");
            eprintln!("    - use a smaller quant (e.g. .q4.bqnt instead of .q8.bqnt)");
            eprintln!("    - HipError(2) = hipErrorOutOfMemory: not enough VRAM for distributed weights + KV caches + scratch");
            std::process::exit(1);
        }
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
