use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_9B: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/";
const MODEL_4B: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/";

fn decode_one(model_dir: &str, label: &str) {
    let p = Path::new(model_dir);
    if !p.exists() { eprintln!("SKIP {label}"); return; }
    let device = DeviceId(0);
    let mut model = Model::load(p, device).expect("load");
    let cfg = model.config();
    println!("{label}: hidden={}, layers={}, nh={}, nvh={}, tie_embed={}",
        cfg.hidden_size, cfg.num_layers,
        cfg.linear_num_heads, cfg.linear_num_value_heads,
        cfg.tie_word_embeddings);
    let logits = model.decode_step(9707, 0).expect("decode");
    let (idx, val) = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let nonzero = logits.iter().filter(|v| v.abs() > 1e-6).count();
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_logit: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    println!("  argmax={idx}, logit={val:.4}, nonzero={nonzero}/{}", logits.len());
    println!("  range: [{min_logit:.4}, {max_logit:.4}], mean={mean_logit:.4}");
    println!("  logits[0..5]: {:.4?}", &logits[..5]);
    // Check for NaN/Inf
    let nans = logits.iter().filter(|v| v.is_nan()).count();
    let infs = logits.iter().filter(|v| v.is_infinite()).count();
    if nans > 0 || infs > 0 { println!("  WARNING: {nans} NaN, {infs} Inf"); }
}

#[test]
fn test_debug_4b_vs_9b() {
    decode_one(MODEL_4B, "4B");
    decode_one(MODEL_9B, "9B");
}

#[test]
fn test_9b_multi_step() {
    let p = Path::new(MODEL_9B);
    if !p.exists() { return; }
    let device = DeviceId(0);
    let mut model = Model::load(p, device).expect("load");
    let mut token = 9707u32;
    for step in 0..5 {
        let logits = model.decode_step(token, step).expect("step");
        let (idx, val) = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        println!("9B step {step}: token={token} -> argmax={idx}, logit={val:.4}");
        token = idx as u32;
    }
}
