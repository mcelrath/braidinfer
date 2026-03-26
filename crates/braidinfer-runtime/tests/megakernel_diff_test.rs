use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::megakernel::MegakernelProgram;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn test_megakernel_diff_investigation() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() { return; }

    // Run naive
    let mut model = Model::load(model_dir, device).expect("load");
    let logits_naive = model.decode_step(9707, 0).expect("decode");

    // Reload and run megakernel
    drop(model);
    let mut model = Model::load(model_dir, device).expect("load");
    let mut prog = MegakernelProgram::compile(&model).expect("compile");
    model.set_position(0).unwrap();
    prog.update_step(9707, 0).unwrap();
    prog.execute(model.stream()).unwrap();
    model.stream().synchronize().unwrap();
    let logits_mega = model.read_logits().expect("read");

    let cross_diff = logits_naive.iter().zip(logits_mega.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Naive vs mega max diff: {:.10}", cross_diff);

    // Find where the biggest diffs are
    let mut diffs: Vec<(usize, f32, f32, f32)> = logits_naive.iter().zip(logits_mega.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
        .filter(|(_, _, _, d)| *d > 0.000001)
        .collect();
    diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("Top 10 diffs:");
    for (i, naive, mega, diff) in diffs.iter().take(10) {
        println!("  idx={}: naive={:.8} mega={:.8} diff={:.10}", i, naive, mega, diff);
    }
    println!("Total entries with diff > 1e-6: {}", diffs.len());
}
