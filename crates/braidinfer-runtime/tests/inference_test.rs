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

    let top10 = tracer.read_f32(Probe::Logits { top_k: 10 }).expect("logits probe missing");
    assert!(!top10.is_empty(), "top10 logits empty");
    assert!(top10.iter().all(|x| x.is_finite()), "top10 logits NaN/Inf");

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
