//! Multi-GPU prefill trace dump: loads a model with enable_distributed_moe,
//! runs model.prefill(tokens), reads per-layer probes from the tracer,
//! and writes a BTRC file with L{i}_hidden + logits checkpoint names.
//!
//! Usage:
//!   BRAIDINFER_TRACE=1 \
//!   BRAIDINFER_TRACE_FILE=/path/out.btrc \
//!   MODEL=/path/to/model \
//!   MAX_TOKENS=1 \
//!   python3 scripts/launch-gpu.py -g4 --timeout 600 -- \
//!     target/release/trace_dump_mgpu "The quick brown fox"
//!
//! For multi-GPU (MULTI_GPU=1 or -gN via apply_auto_modes):
//!   MULTI_GPU=1 ... target/release/trace_dump_mgpu "prompt"
//!
//! The BTRC output uses checkpoint names "L{i}_hidden" for per-layer end-of-layer
//! hidden states and "logits" for the final logits vector. These match the HF
//! reference naming in exterior_algebra/results/nemotron_super_120b_btrc/.
//!
//! bd braidinfer-4ayf.21

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::cli::{apply_auto_modes, resolve_model_arg, vram_usage_mb};
use braidinfer_runtime::generate::load_tokenizer_and_config;
use braidinfer_runtime::model::Model;
use braidinfer_runtime::tracer::{Probe, TraceSink};

fn main() {
    // Set BRAIDINFER_TRACE=1 before Model::load so the tracer is initialized
    // with ProbeFilter::All (or whatever regex the operator chose).
    // SAFETY: single-threaded main, no concurrent env readers.
    unsafe {
        if std::env::var("BRAIDINFER_TRACE").is_err() {
            std::env::set_var("BRAIDINFER_TRACE", "1");
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let model_arg = std::env::var("MODEL").ok();
    let prompt = if args.len() > 1 { args[1].clone() } else { "The quick brown fox".to_string() };

    let resolved = resolve_model_arg(model_arg);
    let model_dir = resolved.model_dir.as_path();

    println!("trace_dump_mgpu: model_dir={:?}", model_dir);
    println!("trace_dump_mgpu: prompt={prompt:?}");

    let device = DeviceId(0);
    let multi_gpu = apply_auto_modes(model_dir);

    let mut model = Model::load(model_dir, device).expect("load model");
    let (used, total) = vram_usage_mb();
    println!("VRAM after load: {used:.0}/{total:.0} MB");

    // Enable distributed MoE (required even for single-GPU MoE models).
    if multi_gpu || model.has_moe() {
        if let Err(e) = model.enable_distributed_moe() {
            eprintln!("ERROR: enable_distributed_moe failed: {e:?}");
            std::process::exit(1);
        }
    }
    let (used, total) = vram_usage_mb();
    println!("VRAM after enable_distributed_moe: {used:.0}/{total:.0} MB");

    let (tokenizer, _token_config) =
        load_tokenizer_and_config(model_dir, resolved.bqnt_override.as_deref())
            .expect("tokenizer/config");
    let tokens: Vec<u32> = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    println!("prompt tokens ({} tokens): {:?}", tokens.len(), &tokens[..tokens.len().min(10)]);

    // Run prefill. Probes are captured into tracer shadows during execution.
    let logits = model.prefill(&tokens).expect("prefill");
    let next_tok = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    println!("prefill complete, next_tok={next_tok}");

    // Sync tracer SDMA streams before reading shadows.
    model.tracer_mut().drain().expect("tracer drain");

    let cfg = model.config();
    let num_layers = cfg.num_layers;

    // Collect probes into BTRC.
    // For each layer: prefer PostFfn (end-of-FFN/MoE residual), fall back to PostMixer
    // (end-of-attention/GDN/SSM residual). This gives "L{i}_hidden" = decoder-layer output.
    let trace_path = std::env::var("BRAIDINFER_TRACE_FILE").ok();
    let mut sink = trace_path.as_ref().map(|p| {
        TraceSink::open(p).unwrap_or_else(|e| {
            eprintln!("TraceSink::open({p}) failed: {e}");
            std::process::exit(1);
        })
    });

    println!();
    println!("=== Per-layer probe summary ===");
    let tracer = model.tracer();
    for layer_i in 0..num_layers {
        // PostFfn takes priority (captures after MoE residual or dense FFN residual).
        let hidden_data = tracer.read_f32(Probe::PostFfn { layer: layer_i })
            .or_else(|| tracer.read_f32(Probe::PostMixer { layer: layer_i }));

        let cp_name = format!("L{layer_i}_hidden");
        if let Some(data) = hidden_data {
            let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let nan_count = data.iter().filter(|x| x.is_nan()).count();
            println!("  {cp_name:<20} n={:<8} max_abs={:.4e} nan={nan_count}", data.len(), max_abs);
            if let Some(ref mut s) = sink {
                s.write_checkpoint(&cp_name, data).expect("write_checkpoint");
            }
        } else {
            println!("  {cp_name:<20} MISSING");
        }
    }

    // FinalNorm probe.
    if let Some(data) = tracer.read_f32(Probe::FinalNorm) {
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!("  final_norm            n={:<8} max_abs={:.4e}", data.len(), max_abs);
    }

    // Logits — write full logits vector (not just top-k) for compare_traces.py alignment.
    let cp_logits = "logits";
    let max_abs_l = logits.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let nan_l = logits.iter().filter(|x| x.is_nan()).count();
    println!("  {cp_logits:<20} n={:<8} max_abs={:.4e} nan={nan_l}", logits.len(), max_abs_l);
    if let Some(ref mut s) = sink {
        s.write_checkpoint(cp_logits, &logits).expect("write logits");
    }

    if let Some(s) = sink {
        s.close().expect("sink close");
        println!();
        println!("BTRC written to: {:?}", trace_path.unwrap());
    } else {
        println!();
        println!("No BRAIDINFER_TRACE_FILE set — probe data printed above only.");
    }
    println!("trace_dump_mgpu: done");
}
