use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Qwen35Model;
use std::path::Path;
use std::time::Instant;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
}

#[test]
fn test_prefill_correctness() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens: Vec<u32> = vec![9707, 13, 220, 5120, 374, 1234, 5678, 42];

    // Sequential decode
    let mut model_seq = Qwen35Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step_paged(tok, i as u32).expect("decode");
    }
    let seq_argmax = argmax(&seq_logits);
    println!("Sequential: argmax={seq_argmax}");

    // Prefill
    let mut model_pre = Qwen35Model::load(model_dir, device).expect("load");
    let pre_logits = model_pre.prefill(&tokens).expect("prefill");
    let pre_argmax = argmax(&pre_logits);
    println!("Prefill: argmax={pre_argmax}");

    assert_eq!(seq_argmax, pre_argmax, "prefill and sequential should agree");
    let diff: f32 = seq_logits.iter().zip(pre_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("Max logit diff: {diff:.6}");
    assert!(diff < 0.001, "prefill vs sequential max_diff={diff}");
}

#[test]
fn test_prefill_benchmark() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let n = 128;
    let tokens: Vec<u32> = (0..n).map(|i| 9707 + (i % 10) as u32).collect();

    let mut model = Qwen35Model::load(model_dir, device).expect("load");

    // Warmup
    model.prefill(&tokens[..8]).expect("warmup");

    // Benchmark
    let mut model = Qwen35Model::load(model_dir, device).expect("reload");
    let start = Instant::now();
    let _logits = model.prefill(&tokens).expect("prefill");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!("Prefill {n} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());
}
