//! bqnt-quantize: Convert safetensors model to BQNT quantized format.
//!
//! Usage:
//!   cargo run --bin bqnt_quantize -- --model Qwen/Qwen3.5-122B-A10B --format q4 --output model.bqnt

use std::path::{Path, PathBuf};
use std::time::Instant;

use safetensors::SafeTensors;
use serde_json::json;

use braidinfer_runtime::bqnt::BqntWriter;
use braidinfer_runtime::quant::{self, WeightFormat};

/// Patterns for tensors that must stay bf16 (router weights).
const BF16_PATTERNS: &[&str] = &[
    "gate.weight",               // MoE router gate
    "e_score_correction_bias",   // Nemotron router bias
];

/// Patterns for tensors to skip (not weight matrices).
const SKIP_PATTERNS: &[&str] = &[
    "layernorm.weight", "norm.weight",
    "A_log", "dt_bias",
    "in_proj_a.weight", "in_proj_b.weight",
];

fn should_skip(name: &str) -> bool {
    SKIP_PATTERNS.iter().any(|p| name.contains(p))
}

fn should_bf16(name: &str) -> bool {
    BF16_PATTERNS.iter().any(|p| name.contains(p))
}

fn resolve_model_path(model_name: &str) -> PathBuf {
    // Check if it's a local directory
    let local = PathBuf::from(model_name);
    if local.is_dir() {
        return local;
    }
    // Check HF cache
    let hf_name = model_name.replace('/', "--");
    let cache_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".cache/huggingface/hub")
        .join(format!("models--{hf_name}"));
    if cache_dir.is_dir() {
        // Find latest snapshot
        let snapshots = cache_dir.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    return entry.path();
                }
            }
        }
    }
    // Return as-is, let caller handle error
    PathBuf::from(model_name)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut format_str = "q4".to_string();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => { i += 1; model_name = args[i].clone(); }
            "--format" | "-f" => { i += 1; format_str = args[i].clone(); }
            "--output" | "-o" => { i += 1; output_path = args[i].clone(); }
            "--help" | "-h" => {
                eprintln!("Usage: bqnt_quantize --model <name> [--format q4|q8|mixed] [--output <path>]");
                std::process::exit(0);
            }
            _ => { eprintln!("Unknown arg: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    if model_name.is_empty() {
        eprintln!("Error: --model required");
        std::process::exit(1);
    }

    let default_format = match format_str.as_str() {
        "q4" => WeightFormat::PcG32Q4,
        "q8" => WeightFormat::Rnf4G128,
        "mixed" => WeightFormat::PcG32Q4, // MLP at Q4, rest at Q8
        _ => {
            eprintln!("Error: --format must be q4, q8, or mixed");
            std::process::exit(1);
        }
    };
    let is_mixed = format_str == "mixed";

    let model_dir = resolve_model_path(&model_name);
    if !model_dir.is_dir() {
        eprintln!("Error: model directory not found: {}", model_dir.display());
        std::process::exit(1);
    }

    if output_path.is_empty() {
        let short = model_name.split('/').last().unwrap_or(&model_name);
        output_path = format!("{short}.{format_str}.bqnt");
    }

    eprintln!("Model:  {model_name}");
    eprintln!("Dir:    {}", model_dir.display());
    eprintln!("Format: {format_str}");
    eprintln!("Output: {output_path}");

    let start = Instant::now();
    let mut writer = BqntWriter::create(Path::new(&output_path))
        .expect("Failed to create output file");

    // Find all safetensors shards
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&model_dir)
        .expect("Failed to read model directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.ends_with(".safetensors")
        })
        .map(|e| e.path())
        .collect();
    shards.sort();

    eprintln!("Found {} safetensors shard(s)", shards.len());

    let mut total_params: u64 = 0;
    let mut quantized_params: u64 = 0;
    let mut bf16_params: u64 = 0;
    let mut tensor_count: u32 = 0;

    for (si, shard_path) in shards.iter().enumerate() {
        let data = std::fs::read(shard_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", shard_path.display()));
        let safetensors = SafeTensors::deserialize(&data)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", shard_path.display()));

        let names: Vec<String> = safetensors.names().iter().map(|s| s.to_string()).collect();
        eprintln!("Shard {}/{}: {} tensors", si + 1, shards.len(), names.len());

        for name in &names {
            let tensor = safetensors.tensor(name).unwrap();
            let shape = tensor.shape();

            if shape.len() < 2 || should_skip(name) {
                continue;
            }

            let ndim = shape.len() as u32;
            let out_dim = shape[0];
            let in_dim: usize = shape[1..].iter().product();
            let n_elements = out_dim * in_dim;
            total_params += n_elements as u64;

            // Determine format for this tensor
            let fmt = if should_bf16(name) {
                WeightFormat::Bf16
            } else if is_mixed {
                // Mixed: MLP + attention at Q4, GDN at Q8
                if name.contains("mlp.") || name.contains("self_attn") {
                    WeightFormat::PcG32Q4
                } else {
                    WeightFormat::Rnf4G128
                }
            } else {
                default_format
            };

            // Get bf16 data
            let raw = tensor.data();
            let bf16_slice: &[u16] = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const u16, n_elements)
            };

            // Quantize
            let packed = match fmt {
                WeightFormat::Bf16 => {
                    bf16_params += n_elements as u64;
                    raw.to_vec()
                }
                WeightFormat::PcG32Q4 => {
                    quantized_params += n_elements as u64;
                    quant::quantize_pc_g32_q4(bf16_slice, out_dim, in_dim)
                }
                WeightFormat::Rnf4G128 => {
                    quantized_params += n_elements as u64;
                    quant::quantize_rnf4_g128(bf16_slice, out_dim, in_dim)
                }
            };

            writer.write_tensor(name, fmt, out_dim as u32, in_dim as u32, ndim, &packed)
                .unwrap_or_else(|e| panic!("Failed to write tensor {name}: {e}"));

            tensor_count += 1;
            if tensor_count % 50 == 0 {
                eprint!("  {tensor_count} tensors written...\r");
            }
        }
    }

    let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    let effective_bpw = if total_params > 0 {
        file_size as f64 * 8.0 / total_params as f64
    } else {
        0.0
    };

    let metadata = json!({
        "model_name": model_name,
        "quantizer_version": "braidinfer-bqnt-v1",
        "default_format": format_str,
        "quantization_stats": {
            "total_params": total_params,
            "quantized_params": quantized_params,
            "bf16_params": bf16_params,
            "tensor_count": tensor_count,
            "file_size_bytes": file_size,
            "effective_bpw": effective_bpw,
        }
    });

    writer.finish(&metadata.to_string())
        .expect("Failed to finalize BQNT file");

    let elapsed = start.elapsed();
    eprintln!("\nDone in {:.1}s", elapsed.as_secs_f64());
    eprintln!("Tensors: {tensor_count}");
    eprintln!("Params:  {total_params} total, {quantized_params} quantized, {bf16_params} bf16");
    eprintln!("Size:    {:.1} GB ({:.2} bits/param)", file_size as f64 / 1e9, effective_bpw);
    eprintln!("Output:  {output_path}");
}
