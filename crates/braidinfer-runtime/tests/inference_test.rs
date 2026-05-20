use std::borrow::Cow;

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::tracer::{Probe, ProbeFilter, Tracer};
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn test_model_trace() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");
    let (logits, traces) = model.decode_step_traced(9707, 0).expect("decode");

    for (name, hidden) in &traces {
        println!(
            "{:>12}: first5=[{:.6}, {:.6}, {:.6}, {:.6}, {:.6}], norm={:.6}",
            name,
            hidden[0],
            hidden[1],
            hidden[2],
            hidden[3],
            hidden[4],
            norm(hidden)
        );
    }

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("argmax={}, logit={:.4}", argmax.0, argmax.1);
}

#[test]
fn test_model_trace_v2() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut model = Model::load(model_dir, device).expect("load model");

    // Reference: synchronous D2H path.
    let (logits_ref, traces_ref) = model.decode_step_traced(9707, 0).expect("decode (ref)");

    // Reset sequence state for v2 run.
    model.reset_state().expect("reset_state");

    // Build a Tracer using the model's compute stream as the SDMA stream for GPU 0.
    // decode_step_traced_v2 does not use persistent_workers, so the compute stream
    // is safe to reuse as the copy stream (model init path, no cooperative kernel).
    let streams = vec![model.stream().raw()];
    let mut tracer = Tracer::with_filter_and_streams(streams, ProbeFilter::All);

    let logits_v2 = model
        .decode_step_traced_v2(9707, 0, &mut tracer)
        .expect("decode (v2)");

    // Logits must match.
    assert_eq!(logits_ref.len(), logits_v2.len(), "logits length mismatch");
    for (i, (a, b)) in logits_ref.iter().zip(logits_v2.iter()).enumerate() {
        assert_eq!(a, b, "logits[{i}] mismatch");
    }

    // Check per-layer probes match the reference traces.
    let num_layers = traces_ref.len() - 2; // subtract embed + final_norm entries
    // embed
    let embed_v2 = tracer.read_f32(Probe::Embed).expect("embed probe missing");
    let (_, embed_ref) = traces_ref.iter().find(|(n, _)| n == "embed").unwrap();
    assert_eq!(embed_v2, embed_ref.as_slice(), "embed mismatch");

    // final_norm
    let fnorm_v2 = tracer.read_f32(Probe::FinalNorm).expect("final_norm probe missing");
    let (_, fnorm_ref) = traces_ref.iter().find(|(n, _)| n == "final_norm").unwrap();
    assert_eq!(fnorm_v2, fnorm_ref.as_slice(), "final_norm mismatch");

    // per-layer
    for i in 0..num_layers {
        let probe = Probe::Custom(Cow::Owned(format!("layer_{i}")));
        let layer_v2 = tracer.read_f32(probe).unwrap_or_else(|| panic!("layer_{i} probe missing"));
        let (_, layer_ref) = traces_ref
            .iter()
            .find(|(n, _)| n == &format!("layer_{i}"))
            .unwrap_or_else(|| panic!("layer_{i} missing from ref"));
        assert_eq!(layer_v2, layer_ref.as_slice(), "layer_{i} mismatch");
    }

    println!("test_model_trace_v2: all {} probes match reference", traces_ref.len() + 1);
}
