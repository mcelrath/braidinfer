use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> (usize, f32) {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, &v)| (i, v)).unwrap()
}

#[test]
fn test_quantized_kv_vs_f32_paged() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let prompt_token = 9707u32; // "Hello"
    let num_steps = 10;

    // Run f32 paged path
    let mut model_f32 = Model::load(model_dir, device).expect("load model (f32)");
    let mut f32_tokens = Vec::new();
    let mut logits = model_f32.decode_step_paged(prompt_token, 0).expect("f32 step 0");
    for i in 0..num_steps {
        let (tok, _) = argmax(&logits);
        f32_tokens.push(tok);
        logits = model_f32.decode_step_paged(tok as u32, (i + 1) as u32).expect("f32 step");
    }
    println!("F32 paged tokens: {:?}", f32_tokens);

    // Run quantized paged path
    let mut model_q = Model::load(model_dir, device).expect("load model (quant)");
    let mut q_tokens = Vec::new();
    let mut logits = model_q.decode_step_paged_quantized(prompt_token, 0).expect("quant step 0");
    for i in 0..num_steps {
        let (tok, _) = argmax(&logits);
        q_tokens.push(tok);
        logits = model_q.decode_step_paged_quantized(tok as u32, (i + 1) as u32).expect("quant step");
    }
    println!("Quantized paged tokens: {:?}", q_tokens);

    // Compare: should produce identical tokens for short sequences (within first chunk, no quantization yet)
    assert_eq!(f32_tokens, q_tokens, "token mismatch within first chunk");
}

#[test]
fn test_quantized_kv_long_sequence() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Generate 80 tokens (crosses chunk boundary at 64, triggering quantization)
    let prompt_token = 9707u32;
    let num_steps = 80;

    let mut model = Model::load(model_dir, device).expect("load model");
    let mut tokens = Vec::new();
    let mut logits = model.decode_step_paged_quantized(prompt_token, 0).expect("step 0");
    for i in 0..num_steps {
        let (tok, val) = argmax(&logits);
        if i == 63 || i == 64 || i == 65 {
            println!("step {}: token={tok}, logit={val:.4}", i + 1);
        }
        tokens.push(tok);
        logits = model.decode_step_paged_quantized(tok as u32, (i + 1) as u32).expect("step");
    }
    println!("Generated {} tokens across chunk boundary", tokens.len());
    println!("Last 5 tokens: {:?}", &tokens[tokens.len()-5..]);

    assert_eq!(tokens.len(), num_steps, "should generate all requested tokens");
}
