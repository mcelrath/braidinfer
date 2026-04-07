use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn first5(v: &[f32]) -> String {
    format!(
        "[{:.8}, {:.8}, {:.8}, {:.8}, {:.8}]",
        v[0], v[1], v[2], v[3], v[4]
    )
}

#[test]
fn test_gdn_layer0_trace() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");
    let traces = model.gdn_layer0_trace(9707).expect("trace");
    for (name, vals) in &traces {
        println!(
            "{:>20}: first5={}, norm={:.6}",
            name,
            first5(vals),
            norm(vals)
        );
    }
}
