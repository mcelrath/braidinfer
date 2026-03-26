use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn test_model_trace() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");
    let (logits, traces) = model.decode_step_traced(9707, 0).expect("decode");

    for (name, hidden) in &traces {
        println!(
            "{:>12}: first5=[{:.6}, {:.6}, {:.6}, {:.6}, {:.6}], norm={:.6}",
            name, hidden[0], hidden[1], hidden[2], hidden[3], hidden[4], norm(hidden)
        );
    }

    let argmax = logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    println!("argmax={}, logit={:.4}", argmax.0, argmax.1);
}
