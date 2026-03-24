use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Qwen35Model;
use braidinfer_runtime::megakernel::MegakernelProgram;
use std::path::Path;
use std::time::Instant;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn test_megakernel_correctness() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Qwen35Model::load(model_dir, device).expect("load model");
    let ref_logits = model.decode_step(9707, 0).expect("naive decode");
    let ref_argmax = ref_logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    println!("Reference: argmax={}, logit={:.4}", ref_argmax.0, ref_argmax.1);
    assert_eq!(ref_argmax.0, 13);

    // Reload model (reset state) and test megakernel
    let mut model = Qwen35Model::load(model_dir, device).expect("reload model");
    let mut program = MegakernelProgram::compile(&model).expect("compile program");
    println!("Program: {} instructions, {} blocks", program.instruction_count(), program.block_count());

    model.set_position(0).expect("set pos");
    program.update_step(9707, 0).expect("update step");
    program.execute(model.stream()).expect("megakernel execute");
    model.stream().synchronize().expect("sync");

    let mega_logits = model.read_logits().expect("read logits");
    let mega_argmax = mega_logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    println!("Megakernel: argmax={}, logit={:.4}", mega_argmax.0, mega_argmax.1);
    assert_eq!(mega_argmax.0, 13, "megakernel should produce argmax=13");

    let max_diff = ref_logits.iter().zip(mega_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max logit diff: {:.6}", max_diff);
    assert!(max_diff < 0.01, "max_diff={}", max_diff);
}

#[test]
fn test_megakernel_benchmark() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Qwen35Model::load(model_dir, device).expect("load model");
    let mut program = MegakernelProgram::compile(&model).expect("compile");

    // Warmup
    model.set_position(0).expect("set pos");
    program.update_step(9707, 0).expect("update");
    program.execute(model.stream()).expect("execute");
    model.stream().synchronize().expect("sync");

    let n = 100;
    let start = Instant::now();
    for i in 0..n {
        model.set_position(i).expect("set pos");
        program.update_step(9707, i).expect("update");
        program.execute(model.stream()).expect("execute");
    }
    model.stream().synchronize().expect("sync");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!("Megakernel: {n} steps in {:.3}s = {per_token_ms:.3} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());
}
