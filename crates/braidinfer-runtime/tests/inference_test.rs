use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Qwen35Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[test]
fn test_model_loads_and_generates() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found at {MODEL_DIR}, skipping test");
        return;
    }

    let mut model = Qwen35Model::load(model_dir, device).expect("load model");

    // Feed a simple prompt: "Hello" = token 9707 in Qwen tokenizer
    // (This is approximate; exact token ID may differ)
    let prompt_token = 9707u32;

    let logits = model.decode_step(prompt_token, 0).expect("decode step 0");
    assert_eq!(logits.len(), 248320);

    let next_token = argmax(&logits);
    println!("Token 0 (input={prompt_token}) → next={next_token}, top logit={:.4}", logits[next_token as usize]);

    // Generate a few more tokens
    for pos in 1..5u32 {
        let prev = if pos == 1 { next_token } else { argmax(&model.decode_step(next_token, pos - 1).unwrap()) };
        let logits = model.decode_step(prev, pos).expect(&format!("decode step {pos}"));
        let tok = argmax(&logits);
        println!("Token {pos} (input={prev}) → next={tok}, top logit={:.4}", logits[tok as usize]);
    }

    println!("Inference test passed: model loaded and generated 5 tokens");
}
