//! Smoke test for the persistent paged decode path (braidinfer-8gz).
//!
//! Verifies:
//! 1. Model loads in PERSISTENT mode without panic.
//! 2. A short decode loop produces sensible (non-NaN, non-degenerate) tokens.
//!
//! Does NOT compare against flat-cache reference because:
//! - test_quantized_kv_vs_f32_paged (kv_quant_e2e_test.rs) already shows that path is flaky
//!   with bare-token input (no chat template). Tracked as exterior_algebra-cxf.
//! - This is a smoke test for the new code path, not a numerical equivalence test.
//!
//! bd b8iy 2026-05-21: removed obsolete test_persistent_kv_quant_returns_error.
//! KV_QUANT+PERSISTENT is now supported (bd 9gmh) via
//! quantize_sealed_chunk_via_worker; the InvalidConfig guard it asserted on
//! has been removed.

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
fn test_persistent_paged_decode_smoke() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    unsafe { std::env::remove_var("KV_QUANT") };

    let mut model = Model::load(model_dir, device).expect("load model in persistent mode");

    let prompt_token = 9707u32;
    let mut logits = model.decode_step(prompt_token, 0).expect("persistent decode step 0");

    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "persistent decode step 0 produced {} NaN logits", nan_count);

    let mut tokens: Vec<u32> = Vec::new();
    for i in 0..10 {
        let tok = argmax(&logits);
        tokens.push(tok);
        logits = model.decode_step(tok, (i + 1) as u32).expect("persistent decode step");
        let nan_count = logits.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "persistent decode step {} produced {} NaN logits", i + 1, nan_count);
    }

    let unique: std::collections::HashSet<_> = tokens.iter().collect();
    println!("Persistent paged tokens: {:?} ({} unique of {})", tokens, unique.len(), tokens.len());
    assert!(
        unique.len() >= 2,
        "persistent decode produced only one unique token across 10 steps: {:?}",
        tokens
    );
}

