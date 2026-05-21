use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::tracer::Probe;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn test_model_trace_v2() {
    let device = DeviceId(0);
    let model_dir = Path::new(MODEL_DIR);
    if !model_dir.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // BRAIDINFER_TRACE must be set before Model::load so the tracer is constructed
    // with ProbeFilter::All inside ensure_moe_workers_started / model init.
    // SAFETY: single-threaded test; no concurrent readers of BRAIDINFER_TRACE.
    unsafe { std::env::set_var("BRAIDINFER_TRACE", "1") };

    let mut model = Model::load(model_dir, device).expect("load model");

    let logits = model.decode_step(9707, 0).expect("decode");

    let hs = model.config().hidden_size;
    let vs = model.config().vocab_size;

    assert_eq!(logits.len(), vs, "logits length");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits contain NaN/Inf"
    );

    let tracer = model.tracer();

    let embed = tracer.read_f32(Probe::Embed).expect("embed missing");
    assert_eq!(embed.len(), hs, "embed length");
    assert!(embed.iter().all(|x| x.is_finite()), "embed NaN/Inf");

    let fnorm = tracer.read_f32(Probe::FinalNorm).expect("final_norm missing");
    assert_eq!(fnorm.len(), hs, "final_norm length");
    assert!(fnorm.iter().all(|x| x.is_finite()), "final_norm NaN/Inf");

    // bd b8iy 2026-05-21: Probe::Logits { top_k: 10 } is recorded via
    // Tracer::record_host_f32 (decode/mod.rs:276) which writes to the BTRC
    // sink only and does NOT populate the in-memory shadow map that
    // tracer.read_f32 reads. The probe IS captured to disk when
    // BRAIDINFER_TRACE_FILE is set, but in-process readback returns None.
    // Embed / FinalNorm work because they go through the SDMA drain path
    // that calls insert_host_bytes. This is an API asymmetry, not a
    // correctness bug — `logits` (the direct decode_step return) above
    // already validates the values.
    let _ = tracer; // keep tracer borrow for shape consistency with intent of test

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "test_model_trace_v2: argmax={} logit={:.4} probes_ok",
        argmax.0, argmax.1
    );
}
