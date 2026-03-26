use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn max_diff_vecs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().zip(b.iter())
        .flat_map(|(va, vb)| va.iter().zip(vb.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max)
}

#[test]
fn test_checkpoint_roundtrip() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");

    // Run 16 tokens to build up recurrent state
    for pos in 0u32..16 {
        model.decode_step_paged(9707, pos).expect("decode step");
    }

    // Read GDN state, save checkpoint
    let state_before = model.read_gdn_state().expect("read gdn");
    let slot = model.save_recurrent_checkpoint().expect("save checkpoint");

    // Mutate GDN state by running more tokens
    for pos in 16u32..24 {
        model.decode_step_paged(9707, pos).expect("decode step");
    }
    let state_mutated = model.read_gdn_state().expect("read gdn mutated");
    let mutate_diff = max_diff_vecs(&state_before, &state_mutated);
    println!("GDN state drift after 8 more tokens: {mutate_diff:.6}");
    assert!(mutate_diff > 0.001, "GDN state should change after more tokens");

    // Restore checkpoint
    model.restore_recurrent_checkpoint(slot).expect("restore");
    let state_restored = model.read_gdn_state().expect("read gdn restored");

    let restore_diff = max_diff_vecs(&state_before, &state_restored);
    println!("GDN state diff after restore: {restore_diff:.6}");
    assert_eq!(restore_diff, 0.0, "checkpoint restore should be bitwise identical");
}
