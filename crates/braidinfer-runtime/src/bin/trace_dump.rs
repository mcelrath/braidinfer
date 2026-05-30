use braidinfer_core::types::DeviceId;
use braidinfer_runtime::cli::{apply_auto_modes, resolve_model_arg, vram_usage_mb};
use braidinfer_runtime::generate::load_tokenizer_and_config;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::tracer::Probe;
use std::borrow::Cow;

fn stat(name: &str, slice: &[f32]) {
    let n = slice.len();
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut finite_n = 0usize;
    for &x in slice {
        if x.is_nan() {
            nan += 1;
        } else if x.is_infinite() {
            inf += 1;
        } else {
            finite_n += 1;
            let a = x.abs();
            if a > max_abs {
                max_abs = a;
            }
            sum += x as f64;
            sum_sq += (x as f64) * (x as f64);
        }
    }
    let mean = if finite_n > 0 { sum / finite_n as f64 } else { 0.0 };
    let var = if finite_n > 0 {
        (sum_sq / finite_n as f64) - mean * mean
    } else {
        0.0
    };
    let std = var.max(0.0).sqrt();
    let first4: Vec<String> = slice
        .iter()
        .take(4)
        .map(|x| format!("{x:+.4e}"))
        .collect();
    println!(
        "  {name:<32} n={n:<8} nan={nan} inf={inf} max_abs={max_abs:.4e} mean={mean:+.4e} std={std:.4e} first4=[{}]",
        first4.join(", ")
    );
}

fn main() {
    // BRAIDINFER_TRACE must be set before Model::load so the tracer is built
    // with ProbeFilter::All (or whatever regex the operator chose).
    // SAFETY: single-threaded main; no concurrent env readers.
    unsafe {
        if std::env::var("BRAIDINFER_TRACE").is_err() {
            std::env::set_var("BRAIDINFER_TRACE", "1");
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let model_arg = std::env::var("MODEL").ok();
    let prompt = if args.len() > 1 { args[1].clone() } else { "The quick brown fox".to_string() };
    let num_steps: u32 = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let resolved = resolve_model_arg(model_arg);
    let _multi_gpu = apply_auto_modes(&resolved.model_dir);

    println!("trace_dump: model_dir={:?}", resolved.model_dir);
    println!("trace_dump: prompt={prompt:?} steps={num_steps}");

    let device = DeviceId(0);
    let mut model = Model::load(&resolved.model_dir, device).expect("load model");
    let (used, total) = vram_usage_mb();
    println!("VRAM after load: {used:.0}/{total:.0} MB");

    let tokenizer = load_tokenizer_and_config(&resolved.model_dir, resolved.bqnt_override.as_deref())
        .expect("tokenizer/config")
        .0;
    let tokens: Vec<u32> = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    println!("prompt tokens: {tokens:?}");

    let prefill_len = tokens.len();
    let last_logits = model.prefill(&tokens).expect("prefill");
    let mut next_tok = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as u32;

    println!();
    println!("=== after prefill ({prefill_len} tokens, next_tok={next_tok}) ===");
    print_diagnostic(&mut model);

    for step in 0..num_steps {
        let pos = prefill_len as u32 + step;
        let logits = model.decode_step(next_tok, pos).expect("decode_step");
        next_tok = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        println!();
        println!("=== after decode step {step} (pos={pos}, next_tok={next_tok}) ===");
        print_diagnostic(&mut model);
    }

    println!();
    println!("trace_dump: done");
}

fn print_diagnostic(model: &mut Model) {
    if let Err(e) = model.snapshot_gdn_states() {
        eprintln!("snapshot_gdn_states failed: {e}");
    }

    let cfg = model.config();
    let num_layers = cfg.num_layers;
    let tracer = model.tracer();

    println!("--- hidden states (megakernel dump pipeline + SDMA) ---");
    if let Some(buf) = tracer.read_f32(Probe::Embed) {
        stat("embed", buf);
    } else {
        println!("  embed: missing");
    }
    let sample_layers: Vec<usize> = if num_layers >= 3 {
        vec![0, num_layers / 2, num_layers - 1]
    } else {
        (0..num_layers).collect()
    };
    for &i in &sample_layers {
        if let Some(buf) = tracer.read_f32(Probe::PostMixer { layer: i }) {
            stat(&format!("L{i}.post_mixer"), buf);
        }
        if let Some(buf) = tracer.read_f32(Probe::PostFfn { layer: i }) {
            stat(&format!("L{i}.post_ffn"), buf);
        }
    }
    if let Some(buf) = tracer.read_f32(Probe::FinalNorm) {
        stat("final_norm", buf);
    }
    if let Some(buf) = tracer.read_f32(Probe::Logits { top_k: 10 }) {
        stat("top10_logits", buf);
    }

    println!("--- SSM/GDN recurrent states (Model::snapshot_gdn_states + SDMA) ---");
    let mut gdn_layers_found = 0;
    for i in 0..num_layers {
        let probe = Probe::Custom(Cow::Owned(format!("gdn_state_{i}")));
        if let Some(buf) = tracer.read_f32(probe) {
            if gdn_layers_found < 3 || i == num_layers - 1 {
                stat(&format!("gdn_state_{i}"), buf);
            }
            gdn_layers_found += 1;
        }
    }
    println!("  (total GDN/SSM layers captured: {gdn_layers_found})");
}
