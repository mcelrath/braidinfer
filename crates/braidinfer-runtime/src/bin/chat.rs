use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, BufReader};

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{
    apply_chat_template, generate_from_ids, load_tokenizer, ChatMessage, TokenConfig,
};
use braidinfer_runtime::model::Qwen35Model;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[tokio::main]
async fn main() {
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    let system_prompt = std::env::var("SYSTEM").ok();

    let model_dir = Path::new(MODEL_DIR);
    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let device = DeviceId(0);
    let mut model = Qwen35Model::load(model_dir, device).expect("load model");

    eprintln!("braidinfer chat (max_tokens={max_tokens}, ^D to quit)");
    if let Some(sys) = &system_prompt {
        eprintln!("system: {sys}");
    }

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
            messages.push(ChatMessage { role: "system", content: sys });
        }
        for (u, a) in &history {
            messages.push(ChatMessage { role: "user", content: u });
            messages.push(ChatMessage { role: "assistant", content: a });
        }
        messages.push(ChatMessage { role: "user", content: &user_input });

        let prompt_ids = match apply_chat_template(&tokenizer, &token_config, &messages) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("template error: {e}");
                continue;
            }
        };

        model.reset_state().expect("reset");

        let start = Instant::now();
        let result = match generate_from_ids(&mut model, &tokenizer, &token_config, &prompt_ids, max_tokens) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("generate error: {e}");
                continue;
            }
        };
        let elapsed = start.elapsed().as_secs_f64();
        let n = result.tokens.len();

        let response: String = result.text_pieces.join("");
        println!("{response}");
        eprintln!("[{n} tokens in {elapsed:.2}s = {:.1} tok/s]", n as f64 / elapsed);

        history.push((user_input, response));
    }
}
