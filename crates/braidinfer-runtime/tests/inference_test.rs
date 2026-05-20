use std::borrow::Cow;

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::tracer::{Probe, ProbeFilter, Tracer};
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

    let mut model = Model::load(model_dir, device).expect("load model");

    let streams = vec![model.stream().raw()];
    let mut tracer = Tracer::with_filter_and_streams(streams, ProbeFilter::All);

    let logits = model
        .decode_step_traced_v2(9707, 0, &mut tracer)
        .expect("decode (v2)");

    let hs = model.config().hidden_size;
    let vs = model.config().vocab_size;
    let num_layers = model.config().num_layers;

    assert_eq!(logits.len(), vs, "logits length");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits contain NaN/Inf"
    );

    let embed = tracer.read_f32(Probe::Embed).expect("embed missing");
    assert_eq!(embed.len(), hs, "embed length");
    assert!(embed.iter().all(|x| x.is_finite()), "embed NaN/Inf");

    let fnorm = tracer.read_f32(Probe::FinalNorm).expect("final_norm missing");
    assert_eq!(fnorm.len(), hs, "final_norm length");
    assert!(fnorm.iter().all(|x| x.is_finite()), "final_norm NaN/Inf");

    for i in 0..num_layers {
        let probe = Probe::Custom(Cow::Owned(format!("layer_{i}")));
        let layer = tracer
            .read_f32(probe)
            .unwrap_or_else(|| panic!("layer_{i} probe missing"));
        assert_eq!(layer.len(), hs, "layer_{i} length");
        assert!(
            layer.iter().all(|x| x.is_finite()),
            "layer_{i} NaN/Inf"
        );
    }

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
