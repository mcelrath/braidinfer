use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::megakernel::{MegakernelProgram, PrefillBuffers};
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
    let mut model_seq = Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step_paged(tok, i as u32).expect("decode");
    }
    let seq_argmax = argmax(&seq_logits);
    println!("Sequential: argmax={seq_argmax}");

    // Prefill
    let mut model_pre = Model::load(model_dir, device).expect("load");
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
fn test_prefill_batched_multi_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens = [9707u32, 13, 220, 5120, 374];

    // Sequential decode (using flat megakernel, not paged)
    let mut model_seq = Model::load(model_dir, device).expect("load");
    let mut seq_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        seq_logits = model_seq.decode_step(tok, i as u32).expect("decode");
    }
    let seq_argmax = argmax(&seq_logits);
    println!("Sequential: argmax={seq_argmax}");

    // Batched prefill
    let mut model_pre = Model::load(model_dir, device).expect("load");
    let mut prefill_bufs = PrefillBuffers::alloc(device, model_pre.config(), tokens.len()).expect("alloc");
    let program = MegakernelProgram::compile_prefill(&model_pre, &tokens, 0, &mut prefill_bufs).expect("compile");
    println!("Prefill program: {} instructions", program.instruction_count());
    program.execute(model_pre.stream()).expect("execute");
    model_pre.stream().synchronize().expect("sync");
    let pre_logits = model_pre.read_logits().expect("read logits");
    let pre_argmax = argmax(&pre_logits);
    println!("Batched prefill: argmax={pre_argmax}");

    assert_eq!(seq_argmax, pre_argmax, "batched prefill should match sequential");
    let diff: f32 = seq_logits.iter().zip(pre_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("Max diff: {diff:.6}");
    assert!(diff < 0.01, "batched prefill vs sequential diff={diff}");
}

#[test]
fn test_prefill_batched_single_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load");

    // Compile prefill program for single token (should match decode exactly)
    let mut prefill_bufs = PrefillBuffers::alloc(device, &model.config(), 1).expect("alloc prefill");
    let program = MegakernelProgram::compile_prefill(&model, &[9707], 0, &mut prefill_bufs).expect("compile prefill");
    println!("Prefill program: {} instructions", program.instruction_count());

    // Execute prefill
    program.execute(model.stream()).expect("execute");
    model.stream().synchronize().expect("sync");
    let prefill_logits = model.read_logits().expect("read logits");
    let prefill_argmax = argmax(&prefill_logits);
    println!("Prefill single token: argmax={prefill_argmax}");

    // Compare with decode
    let mut model2 = Model::load(model_dir, device).expect("load");
    let decode_logits = model2.decode_step(9707, 0).expect("decode");
    let decode_argmax = argmax(&decode_logits);
    println!("Decode single token: argmax={decode_argmax}");

    assert_eq!(prefill_argmax, decode_argmax, "prefill single token should match decode");
    let diff: f32 = prefill_logits.iter().zip(decode_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("Max diff: {diff:.6}");
    assert!(diff < 0.001, "prefill vs decode diff={diff}");
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
    println!("Sequential decode {n} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());

    // Batched prefill API (uses compile_prefill internally)
    let mut model_pre = Model::load(model_dir, device).expect("reload");
    model_pre.prefill(&tokens[..8]).expect("warmup");
    let mut model_pre = Model::load(model_dir, device).expect("reload");
    let start = Instant::now();
    let _logits = model_pre.prefill(&tokens).expect("prefill");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!("Batched prefill {n} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());

    // Batched prefill benchmark (64 tokens = 1 chunk)
    let nb = 64;
    let batch_tokens: Vec<u32> = (0..nb).map(|i| 9707 + (i % 10) as u32).collect();
    let mut model2 = Model::load(model_dir, device).expect("reload");
    let mut prefill_bufs = PrefillBuffers::alloc(device, model2.config(), nb).expect("alloc");
    let program = MegakernelProgram::compile_prefill(&model2, &batch_tokens, 0, &mut prefill_bufs).expect("compile");
    println!("Batched program: {} instructions", program.instruction_count());
    // Warmup
    program.execute(model2.stream()).expect("warmup");
    model2.stream().synchronize().expect("sync");

    let mut model3 = Model::load(model_dir, device).expect("reload");
    let mut prefill_bufs2 = PrefillBuffers::alloc(device, model3.config(), nb).expect("alloc");
    let program2 = MegakernelProgram::compile_prefill(&model3, &batch_tokens, 0, &mut prefill_bufs2).expect("compile");
    let start = Instant::now();
    program2.execute(model3.stream()).expect("execute");
    model3.stream().synchronize().expect("sync");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / nb as f64;
    let tokens_per_sec = nb as f64 / elapsed.as_secs_f64();
    println!("Batched prefill {nb} tokens: {:.3}s = {per_token_ms:.2} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());
}
