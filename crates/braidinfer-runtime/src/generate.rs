use std::path::Path;
use tokenizers::Tokenizer;

use crate::model::{ModelError, Qwen35Model};

const EOS_TOKEN_ID: u32 = 151643;

pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub text_pieces: Vec<String>,
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;
    Ok(tokenizer)
}

pub fn greedy_generate(
    model: &mut Qwen35Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<GenerateResult, ModelError> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| ModelError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut all_tokens: Vec<u32> = prompt_ids.clone();
    let mut text_pieces: Vec<String> = Vec::new();

    let n_prompt = prompt_ids.len();
    let last_logits = if n_prompt == 0 {
        return Ok(GenerateResult { tokens: vec![], text_pieces: vec![] });
    } else if n_prompt == 1 {
        model.decode_step(prompt_ids[0], 0)?
    } else {
        model.prefill(&prompt_ids)?
    };

    // Greedy decode from the last prompt token's logits
    let mut logits = last_logits;
    let mut position = n_prompt as u32;

    for _ in 0..max_tokens {
        let next_token = argmax(&logits);
        if next_token == EOS_TOKEN_ID {
            break;
        }
        all_tokens.push(next_token);

        let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
        text_pieces.push(piece);

        logits = model.decode_step(next_token, position)?;
        position += 1;
    }

    Ok(GenerateResult {
        tokens: all_tokens[n_prompt..].to_vec(),
        text_pieces,
    })
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
