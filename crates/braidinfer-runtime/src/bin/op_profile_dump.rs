// Profile a model decode by per-op cyc/call. Build with
// BRAIDINFER_OP_PROFILE=1 to enable; without the flag this binary still
// runs but the OP_PROFILE_BEGIN/END macros are no-ops in the kernels and
// every counter reads as zero.
//
// Usage:
//   MODEL=models/qwen35_2b.q4.bqnt MAX_TOKENS=100 \
//     python3 scripts/launch-gpu.py --timeout 600 -- \
//     cargo run --release -p braidinfer-runtime --bin op_profile_dump
//
// The CLI installs a process-global OpProfile, runs the model's standard
// generate path, drops the dispatcher (which shuts down the persistent
// worker), then reads the counters and prints a sorted table.

use std::path::Path;

use braidinfer_core::types::DeviceId;
use braidinfer_runtime::generate::{TokenConfig, greedy_generate, load_tokenizer};
use braidinfer_runtime::model::Model;
use braidinfer_runtime::op_profile;

fn main() {
    let model_path = std::env::var("MODEL").expect("set MODEL=<bqnt or hf-dir path>");
    let prompt = std::env::args().nth(1)
        .unwrap_or_else(|| "The history of computing began long before".to_string());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(100);

    // Resolve HF dir same as generate.rs does
    let model_dir_path = if model_path.ends_with(".bqnt") {
        let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(Path::new(&model_path))
            .expect("open bqnt");
        let model_name = bqnt.model_name().expect("bqnt missing model_name");
        if model_name.starts_with('/') && Path::new(&model_name).is_dir() {
            model_name
        } else {
            let hf_name = model_name.replace('/', "--");
            let cache_dir = dirs::home_dir().expect("home dir")
                .join(".cache/huggingface/hub")
                .join(format!("models--{hf_name}"))
                .join("snapshots");
            std::fs::read_dir(&cache_dir).expect("read snapshots")
                .filter_map(|e| e.ok())
                .find(|e| e.path().join("tokenizer.json").exists())
                .map(|e| e.path().to_string_lossy().to_string())
                .expect("no snapshot with tokenizer found")
        }
    } else {
        model_path.clone()
    };
    if model_path.ends_with(".bqnt") {
        unsafe { std::env::set_var("BQNT_PATH", &model_path); }
    }

    // Allocate the OpProfile BEFORE Model::load (which may lazy-create
    // the persistent worker on first decode call). install_global sets
    // the process-global pointer that PersistentDispatch::add_device reads.
    let device = DeviceId(0);
    let profile = op_profile::OpProfile::alloc(device).expect("alloc OpProfile");
    op_profile::install_global(&profile);
    eprintln!("[op_profile] installed counter buffer on GPU 0 ({} slots)", op_profile::NUM_SLOTS);

    let model_dir = Path::new(&model_dir_path);
    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let mut model = Model::load(model_dir, device).expect("load model");

    eprintln!("[op_profile] running greedy_generate, max_tokens={max_tokens}");
    let result = greedy_generate(&mut model, &tokenizer, &token_config, &prompt, max_tokens)
        .expect("generate");
    eprintln!("[op_profile] generated {} tokens", result.tokens.len());

    // Capture shape data before dropping Model.
    let vocab_size = model.config().vocab_size;
    let hidden_size = model.config().hidden_size;

    // Drop the model — this drops PersistentDispatch, shutting down the
    // persistent worker and flushing all atomic ops. Only then can we
    // hipMemcpy the counters out without deadlocking.
    drop(model);
    eprintln!("[op_profile] model dropped, persistent worker shut down");

    // SAFETY: persistent worker is gone (Model dropped). hipMemcpy is safe.
    let stats = unsafe { profile.dump_after_shutdown() }.expect("dump");

    // wallclock rate is constant per arch (gfx1100 = 100 MHz). Convert
    // ticks/call → us/call so columns are physically interpretable, and
    // tag known-shape ops with achieved memory bandwidth so the table
    // exposes whether they are BW- or compute-bound.
    let rate_khz = op_profile::wallclock_rate_khz(device).expect("wallclock rate");
    // OP_LINEAR_PROJ is only emitted for the lm_head (megakernel_compile.rs
    // flags it explicitly). Reads vocab_size × hidden_size of bf16 weight.
    // Opcode 2 = OP_LINEAR_PROJ per kernels/opcodes.h.
    let lm_head_bytes = (vocab_size * hidden_size * 2) as u64;
    let shapes = vec![op_profile::OpShape {
        opcode: 2,
        bytes_per_dispatch: lm_head_bytes,
        label: "lm_head bf16",
    }];
    eprintln!(
        "[op_profile] wallclock_rate={} kHz (1 tick = {:.2} ns); lm_head shape vocab={} hidden={} bytes/dispatch={} MB",
        rate_khz,
        1.0e6 / rate_khz as f64,
        vocab_size,
        hidden_size,
        lm_head_bytes / (1024 * 1024),
    );
    println!("\n{}", op_profile::format_table_with_bw(&stats, rate_khz, &shapes));
}
