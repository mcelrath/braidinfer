//! bd srg6.7 (braidinfer-utuy): integration smoke test for the batched
//! paged-prefill writer wired into `Model::prefill` for dense single-GPU
//! models.
//!
//! Exercises the new path:
//!   Model::prefill(tokens) → prefill_paged → compile_prefill_paged_persistent
//!   → dispatch_via_worker → act.logits → decode_step loop
//!
//! Asserts:
//!   - no panic
//!   - no NaN in any logits vector
//!   - decode produces ≥2 unique tokens (not stuck on a single token)
//!
//! Reference: persistent_paged_test.rs pattern.

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn argmax(logits: &[f32]) -> u32 {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    idx as u32
}

#[test]
fn test_paged_prefill_then_decode_smoke() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found at {}, skipping", MODEL_DIR);
        return;
    }

    unsafe { std::env::remove_var("KV_QUANT") };

    let mut model = Model::load(model_dir, device).expect("load model");

    // 16-token prompt (single CHUNK_TOKENS slab — qwen3.5 CHUNK_TOKENS=64).
    let prompt: Vec<u32> = (0..16u32).map(|i| 1000 + i).collect();

    let mut logits = model
        .prefill(&prompt)
        .expect("paged prefill via compile_prefill_paged_persistent");

    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "prefill produced {} NaN logits", nan_count);

    let start_pos = prompt.len() as u32;
    let mut tokens: Vec<u32> = Vec::new();
    for i in 0..20 {
        let tok = argmax(&logits);
        tokens.push(tok);
        logits = model
            .decode_step(tok, start_pos + i as u32)
            .expect("decode_step after paged prefill");
        let nan_count = logits.iter().filter(|v| v.is_nan()).count();
        assert_eq!(
            nan_count, 0,
            "decode step {} produced {} NaN logits",
            i, nan_count
        );
    }

    let unique: std::collections::HashSet<_> = tokens.iter().collect();
    println!(
        "Paged-prefill smoke tokens: {:?} ({} unique of {})",
        tokens,
        unique.len(),
        tokens.len()
    );
    assert!(
        unique.len() >= 2,
        "decode after paged prefill produced only one unique token: {:?}",
        tokens
    );
}
