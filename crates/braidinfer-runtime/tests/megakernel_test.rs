use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Qwen35Model;
use braidinfer_runtime::megakernel::MegakernelProgram;
use std::path::Path;
use std::time::Instant;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> (usize, f32) {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, &v)| (i, v)).unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

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
    let (ref_idx, ref_val) = argmax(&ref_logits);
    println!("Reference: argmax={ref_idx}, logit={ref_val:.4}");
    assert_eq!(ref_idx, 13);

    let mut model = Qwen35Model::load(model_dir, device).expect("reload model");
    let mut program = MegakernelProgram::compile(&model).expect("compile program");
    println!("Program: {} instructions, {} blocks", program.instruction_count(), program.block_count());

    model.set_position(0).expect("set pos");
    program.update_step(9707, 0, model.stream()).expect("update step");
    program.execute(model.stream()).expect("megakernel execute");
    model.stream().synchronize().expect("sync");

    let mega_logits = model.read_logits().expect("read logits");
    let (mega_idx, mega_val) = argmax(&mega_logits);
    println!("Megakernel: argmax={mega_idx}, logit={mega_val:.4}");
    assert_eq!(mega_idx, 13, "megakernel should produce argmax=13");

    let diff = max_diff(&ref_logits, &mega_logits);
    println!("Max logit diff: {diff:.6}");
    assert!(diff < 0.01, "max_diff={diff}");
}

#[test]
fn test_paged_single_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Get reference logits from flat megakernel
    let mut model = Qwen35Model::load(model_dir, device).expect("load model");
    let mut program = MegakernelProgram::compile(&model).expect("compile flat");
    model.set_position(0).expect("set pos");
    program.update_step(9707, 0, model.stream()).expect("flat update");
    program.execute(model.stream()).expect("flat execute");
    model.stream().synchronize().expect("sync");
    let flat_logits = model.read_logits().expect("read flat logits");
    let (flat_idx, flat_val) = argmax(&flat_logits);
    println!("Flat megakernel: argmax={flat_idx}, logit={flat_val:.4}");

    // Get paged logits
    let mut model = Qwen35Model::load(model_dir, device).expect("reload for paged");
    let paged_logits = model.decode_step_paged(9707, 0).expect("paged decode");
    let (paged_idx, paged_val) = argmax(&paged_logits);
    println!("Paged: argmax={paged_idx}, logit={paged_val:.4}");

    let diff = max_diff(&flat_logits, &paged_logits);
    println!("Flat vs Paged max diff: {diff:.6}");
    assert_eq!(paged_idx, flat_idx, "paged argmax={paged_idx} != flat argmax={flat_idx}");
    assert!(diff < 0.1, "flat vs paged max_diff={diff} (>0.1)");
}

#[test]
fn test_paged_multi_token() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let tokens = [9707u32, 13, 220, 5120, 374];

    // Flat path: 5 tokens
    let mut model = Qwen35Model::load(model_dir, device).expect("load model");
    let mut program = MegakernelProgram::compile(&model).expect("compile flat");
    for (i, &tok) in tokens.iter().enumerate() {
        model.set_position(i as u32).expect("set pos");
        program.update_step(tok, i as u32, model.stream()).expect("flat update");
        program.execute(model.stream()).expect("flat execute");
        model.stream().synchronize().expect("sync");
    }
    let flat_logits = model.read_logits().expect("flat logits");
    let (flat_idx, flat_val) = argmax(&flat_logits);
    println!("Flat 5-token: argmax={flat_idx}, logit={flat_val:.4}");

    // Paged path: same 5 tokens
    let mut model = Qwen35Model::load(model_dir, device).expect("reload for paged");
    let mut paged_logits = vec![];
    for (i, &tok) in tokens.iter().enumerate() {
        paged_logits = model.decode_step_paged(tok, i as u32).expect("paged decode");
    }
    let (paged_idx, paged_val) = argmax(&paged_logits);
    println!("Paged 5-token: argmax={paged_idx}, logit={paged_val:.4}");

    let diff = max_diff(&flat_logits, &paged_logits);
    println!("5-token flat vs paged max diff: {diff:.6}");
    assert_eq!(paged_idx, flat_idx, "paged argmax={paged_idx} != flat argmax={flat_idx}");
    assert!(diff < 0.1, "5-token max_diff={diff}");
}

#[test]
fn test_paged_chunk_boundary() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Run 70 tokens — crosses 64-token chunk boundary
    let mut model_flat = Qwen35Model::load(model_dir, device).expect("load flat");
    let mut program = MegakernelProgram::compile(&model_flat).expect("compile flat");
    for i in 0..70u32 {
        model_flat.set_position(i).expect("set pos");
        program.update_step(9707, i, model_flat.stream()).expect("flat update");
        program.execute(model_flat.stream()).expect("flat execute");
        model_flat.stream().synchronize().expect("sync");
    }
    let flat_logits = model_flat.read_logits().expect("flat logits");
    let (flat_idx, flat_val) = argmax(&flat_logits);
    println!("Flat 70-token: argmax={flat_idx}, logit={flat_val:.4}");

    let mut model_paged = Qwen35Model::load(model_dir, device).expect("load paged");
    let mut paged_logits = vec![];
    for i in 0..70u32 {
        paged_logits = model_paged.decode_step_paged(9707, i).expect("paged decode");
    }
    let (paged_idx, paged_val) = argmax(&paged_logits);
    println!("Paged 70-token: argmax={paged_idx}, logit={paged_val:.4}");

    let diff = max_diff(&flat_logits, &paged_logits);
    println!("70-token flat vs paged max diff: {diff:.6}");
    assert!(diff < 0.5, "70-token chunk boundary max_diff={diff}");
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

    model.set_position(0).expect("set pos");
    program.update_step(9707, 0, model.stream()).expect("update");
    program.execute(model.stream()).expect("execute");
    model.stream().synchronize().expect("sync");

    let n = 100;
    let start = Instant::now();
    for i in 0..n {
        model.set_position(i).expect("set pos");
        program.update_step(9707, i, model.stream()).expect("update");
        program.execute(model.stream()).expect("execute");
    }
    model.stream().synchronize().expect("sync");
    let elapsed = start.elapsed();
    let per_token_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let tokens_per_sec = n as f64 / elapsed.as_secs_f64();
    println!("Megakernel: {n} steps in {:.3}s = {per_token_ms:.3} ms/token = {tokens_per_sec:.1} tok/s",
        elapsed.as_secs_f64());
}
