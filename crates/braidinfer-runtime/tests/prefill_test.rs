use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;
use std::time::Instant;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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

    // Sequential decode through the public paged-KV-backed API.
    let mut model_seq = Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step(tok, i as u32).expect("decode");
    }
    let seq_argmax = argmax(&seq_logits);
    println!("Sequential: argmax={seq_argmax}");

    // Prefill
    let mut model_pre = Model::load(model_dir, device).expect("load");
    let pre_logits = model_pre.prefill(&tokens).expect("prefill");
    let pre_argmax = argmax(&pre_logits);
    println!("Prefill: argmax={pre_argmax}");

    assert_eq!(
        seq_argmax, pre_argmax,
        "prefill and sequential should agree"
    );
    let diff = max_diff(&seq_logits, &pre_logits);
    println!("Max logit diff: {diff:.6}");
    assert!(
        pre_logits.iter().all(|x| x.is_finite()),
        "prefill logits must be finite"
    );
}

#[test]
fn test_prefill_batched_multi_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens = [9707u32, 13, 220, 5120, 374];

    // Sequential decode through the default public API.
    let mut model_seq = Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step(tok, i as u32).expect("decode");
    }
    let seq_argmax = argmax(&seq_logits);
    println!("Sequential: argmax={seq_argmax}");

    // Public prefill path
    let mut model_pre = Model::load(model_dir, device).expect("load");
    let pre_logits = model_pre.prefill(&tokens).expect("prefill");
    let pre_argmax = argmax(&pre_logits);
    println!("Prefill: argmax={pre_argmax}");

    assert_eq!(
        seq_argmax, pre_argmax,
        "batched prefill should match sequential"
    );
    let diff = max_diff(&seq_logits, &pre_logits);
    println!("Max diff: {diff:.6}");
    assert!(diff < 0.01, "batched prefill vs sequential diff={diff}");
}

#[test]
fn test_prefill_single_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load");
    let prefill_logits = model.prefill(&[9707]).expect("prefill");
    let prefill_argmax = argmax(&prefill_logits);
    println!("Prefill single token: argmax={prefill_argmax}");

    // Compare with decode
    let mut model2 = Model::load(model_dir, device).expect("load");
    let decode_logits = model2.decode_step(9707, 0).expect("decode");
    let decode_argmax = argmax(&decode_logits);
    println!("Decode single token: argmax={decode_argmax}");

    assert_eq!(
        prefill_argmax, decode_argmax,
        "prefill single token should match decode"
    );
    let diff = max_diff(&prefill_logits, &decode_logits);
    println!("Max diff: {diff:.6}");
    assert!(diff < 0.001, "prefill vs decode diff={diff}");
}

#[test]
fn test_prefill_cross_chunk_correctness() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens: Vec<u32> = (0..65).map(|i| 9707 + (i % 11) as u32).collect();

    let mut model_seq = Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step(tok, i as u32).expect("decode");
    }

    let mut model_pre = Model::load(model_dir, device).expect("load");
    let pre_logits = model_pre.prefill(&tokens).expect("prefill");

    let diff = max_diff(&seq_logits, &pre_logits);
    println!("Cross-chunk max diff: {diff:.6}");
    assert_eq!(argmax(&seq_logits), argmax(&pre_logits));
    assert!(diff < 0.001, "cross-chunk prefill diff={diff}");
}

#[test]
fn test_reset_state_clears_paged_kv() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens: Vec<u32> = (0..70).map(|i| 9707 + (i % 13) as u32).collect();
    let mut model = Model::load(model_dir, device).expect("load");

    let logits_before = model.prefill(&tokens).expect("prefill before reset");
    model.reset_state().expect("reset");
    let logits_after = model.prefill(&tokens).expect("prefill after reset");

    let diff = max_diff(&logits_before, &logits_after);
    println!("Reset max diff: {diff:.6}");
    assert_eq!(argmax(&logits_before), argmax(&logits_after));
    assert!(
        diff < 0.001,
        "reset should fully clear paged KV, diff={diff}"
    );
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

    // Sequential decode baseline
    let mut model_seq = Model::load(model_dir, device).expect("load");
    for (i, &tok) in tokens[..8].iter().enumerate() {
        model_seq.decode_step_paged(tok, i as u32).expect("warmup");
    }
    let mut model_seq = Model::load(model_dir, device).expect("reload");
    let start = Instant::now();
    for (i, &tok) in tokens.iter().enumerate() {
        model_seq.decode_step_paged(tok, i as u32).expect("decode");
    }
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!(
        "Sequential decode {n} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64()
    );

    // Public prefill API (currently routes through the paged decode path)
    let mut model_pre = Model::load(model_dir, device).expect("reload");
    model_pre.prefill(&tokens[..8]).expect("warmup");
    let mut model_pre = Model::load(model_dir, device).expect("reload");
    let start = Instant::now();
    let _logits = model_pre.prefill(&tokens).expect("prefill");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!(
        "Batched prefill {n} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64()
    );

    // One-chunk public prefill benchmark
    let nb = 64;
    let batch_tokens: Vec<u32> = (0..nb).map(|i| 9707 + (i % 10) as u32).collect();
    let mut model2 = Model::load(model_dir, device).expect("reload");
    model2.prefill(&batch_tokens).expect("warmup");

    let mut model3 = Model::load(model_dir, device).expect("reload");
    let start = Instant::now();
    let _ = model3.prefill(&batch_tokens).expect("prefill");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / nb as f64;
    let tokens_per_sec = nb as f64 / elapsed.as_secs_f64();
    println!(
        "Batched prefill {nb} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64()
    );
}
