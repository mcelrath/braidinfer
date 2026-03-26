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
    println!("{label}: hidden={}, layers={}, nh={}, nvh={}",
        model.config().hidden_size, model.config().num_layers,
        model.config().linear_num_heads, model.config().linear_num_value_heads);
    let logits = model.decode_step(9707, 0).expect("decode");
    let (idx, val) = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    let nonzero = logits.iter().filter(|v| v.abs() > 1e-6).count();
    let all_same = logits.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
    println!("  argmax={idx}, logit={val:.4}, nonzero={nonzero}/{}, all_same={all_same}",
        logits.len());
    println!("  logits[0..5]: {:.4?}", &logits[..5]);
}

#[test]
fn test_debug_4b_vs_9b() {
    decode_one(MODEL_4B, "4B");
    decode_one(MODEL_9B, "9B");
}
