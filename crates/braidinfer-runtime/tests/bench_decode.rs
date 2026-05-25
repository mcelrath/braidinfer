use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;
use std::time::Instant;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn bench_decode_step() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");

    // Warmup
    let _ = model.decode_step(9707, 0).expect("warmup");

    let n = 100;
    let start = Instant::now();
    for i in 0..n {
        let _ = model.decode_step(9707, i).expect("decode");
    }
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!(
        "{n} decode steps in {:.3}s = {per_token_ms:.3} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64()
    );
}
