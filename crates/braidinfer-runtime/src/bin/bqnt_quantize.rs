//! bqnt-quantize: Convert safetensors model to BQNT quantized format.
//!
//! Usage:
//!   cargo run --bin bqnt_quantize -- --model Qwen/Qwen3.5-122B-A10B --format q4 --output model.bqnt

use std::path::{Path, PathBuf};
use std::time::Instant;

use safetensors::{Dtype, SafeTensors};
use serde_json::json;

use braidinfer_runtime::bqnt::BqntWriter;
use braidinfer_runtime::quant::{self, WeightFormat};

/// Convert FP8_E4M3 byte to f32.
/// E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa. Max ±448, min subnormal 2^-9.
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    let val = if exp == 0 {
        (mant as f32 / 8.0) * (1.0 / 64.0)
    } else {
        (1.0 + mant as f32 / 8.0) * f32::from_bits(((exp as u32 + 120) & 0xFF) << 23)
    };
    if sign == 1 { -val } else { val }
}

/// Convert f32 to bf16 (truncate lower 16 bits of mantissa, round-to-nearest-even).
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

/// Convert bf16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Patterns for tensors that must stay bf16 (router weights).
const BF16_PATTERNS: &[&str] = &[
    "gate.weight",             // MoE router gate
    "e_score_correction_bias", // Nemotron router bias
];

/// Patterns for tensors to skip (not weight matrices).
const SKIP_PATTERNS: &[&str] = &[
    "layernorm.weight",
    "norm.weight",
    "A_log",
    "dt_bias",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "conv1d.bias",
    "qscale_weight",
    "qscale_act", // FP8 per-tensor scales (scalar metadata)
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
            "--model" | "-m" => {
                i += 1;
                model_name = args[i].clone();
            }
            "--format" | "-f" => {
                i += 1;
                format_str = args[i].clone();
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bqnt_quantize --model <name> [--format q4|q8|mixed] [--output <path>]"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
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

    // First pass: count tensors to size the entry table reservation correctly.
    let max_tensors: usize = shards.iter().map(|shard_path| {
        let data = std::fs::read(shard_path).expect("Failed to read shard");
        SafeTensors::deserialize(&data).map(|st| st.names().len()).unwrap_or(0)
    }).sum();
    eprintln!("Total tensors (first pass): {max_tensors}");

    let mut writer =
        BqntWriter::create(Path::new(&output_path), max_tensors).expect("Failed to create output file");

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
            // For 3D fused expert tensors [ne, inner, hidden], reshape to [ne*inner, hidden]
            // so quantization groups align with per-row dequantization in GPU kernels.
            let (out_dim, in_dim) = if ndim == 3 {
                (shape[0] * shape[1], shape[2])
            } else {
                (shape[0], shape[1..].iter().product())
            };
            let n_elements = out_dim * in_dim;
            total_params += n_elements as u64;

            // Determine format for this tensor
            let fmt = if should_bf16(name) {
                WeightFormat::Bf16
            } else if is_mixed {
                // Mixed: MLP/experts at Q4, attention/GDN at Q8 (RNF4)
                // MLA attention (wq_a, wq_b, wkv_a, wkv_b, wo) → RNF4
                // Standard attention (self_attn.*_proj) → RNF4
                // Expert FFN (experts.*.w1/w2/w3, mlp.*) → Q4
                // Shared experts → Q4
                if name.contains("experts.")
                    || name.contains("shared_expert")
                    || name.contains("mlp.")
                {
                    WeightFormat::PcG32Q4
                } else {
                    WeightFormat::Rnf4G128
                }
            } else {
                default_format
            };

            // Get weight data as bf16, handling FP8 dequantization if needed
            let raw = tensor.data();
            let dtype = tensor.dtype();

            let (bf16_data, bf16_slice): (Option<Vec<u16>>, &[u16]) = match dtype {
                Dtype::BF16 => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const u16, n_elements)
                    };
                    (None, slice)
                }
                Dtype::F16 => {
                    // F16 -> BF16: convert via f32
                    let f16_slice = unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const u16, n_elements)
                    };
                    let converted: Vec<u16> = f16_slice
                        .iter()
                        .map(|&bits| {
                            // f16: 1 sign, 5 exp (bias=15), 10 mantissa
                            let sign = ((bits >> 15) & 1) as u32;
                            let exp = ((bits >> 10) & 0x1F) as u32;
                            let mant = (bits & 0x3FF) as u32;
                            let f32_bits = if exp == 0 {
                                if mant == 0 {
                                    sign << 31
                                } else {
                                    // subnormal f16 -> normalize for f32
                                    let f = f32::from_bits((mant as u32) << 13)
                                        * f32::from_bits(0x33800000);
                                    (sign << 31) | (f.to_bits() & 0x7FFFFFFF)
                                }
                            } else if exp == 0x1F {
                                (sign << 31) | 0x7F800000 | (mant << 13) // inf/nan
                            } else {
                                (sign << 31) | ((exp + 112) << 23) | (mant << 13)
                            };
                            f32_to_bf16(f32::from_bits(f32_bits))
                        })
                        .collect();
                    let ptr = converted.as_ptr();
                    let slice = unsafe { std::slice::from_raw_parts(ptr, n_elements) };
                    (Some(converted), slice)
                }
                Dtype::F8_E4M3 => {
                    // FP8 E4M3 -> BF16: dequantize each byte, apply qscale
                    let fp8_bytes: &[u8] = &raw[..n_elements];

                    // Look up per-tensor scale (qscale_weight)
                    let scale_name = format!(
                        "{}.qscale_weight",
                        name.rsplit_once('.')
                            .map(|(prefix, _)| prefix)
                            .unwrap_or(name)
                    );
                    let scale: f32 = safetensors
                        .tensor(&scale_name)
                        .ok()
                        .and_then(|st| {
                            let sd = st.data();
                            if st.dtype() == Dtype::BF16 && sd.len() >= 2 {
                                Some(bf16_to_f32(u16::from_le_bytes([sd[0], sd[1]])))
                            } else if st.dtype() == Dtype::F32 && sd.len() >= 4 {
                                Some(f32::from_le_bytes([sd[0], sd[1], sd[2], sd[3]]))
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1.0);

                    let converted: Vec<u16> = fp8_bytes
                        .iter()
                        .map(|&b| f32_to_bf16(fp8_e4m3_to_f32(b) * scale))
                        .collect();
                    let ptr = converted.as_ptr();
                    let slice = unsafe { std::slice::from_raw_parts(ptr, n_elements) };
                    (Some(converted), slice)
                }
                other => {
                    eprintln!("  Skipping {name}: unsupported dtype {other:?}");
                    total_params -= n_elements as u64;
                    continue;
                }
            };

            // Quantize
            let packed = match fmt {
                WeightFormat::Bf16 => {
                    bf16_params += n_elements as u64;
                    // Store as bf16 bytes
                    unsafe {
                        std::slice::from_raw_parts(bf16_slice.as_ptr() as *const u8, n_elements * 2)
                    }
                    .to_vec()
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
            drop(bf16_data); // free conversion buffer

            writer
                .write_tensor(name, fmt, out_dim as u32, in_dim as u32, ndim, &packed)
                .unwrap_or_else(|e| panic!("Failed to write tensor {name}: {e}"));

            tensor_count += 1;
            if tensor_count % 50 == 0 {
                eprint!("  {tensor_count} tensors written...\r");
            }
        }
    }

    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let effective_bpw = if total_params > 0 {
        file_size as f64 * 8.0 / total_params as f64
    } else {
        0.0
    };

    // Include model config for self-contained bqnt files.
    // Try config.json (HuggingFace), fall back to params.json (Mistral native).
    let config_json: serde_json::Value = std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .or_else(|| {
            std::fs::read_to_string(model_dir.join("params.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        })
        .unwrap_or(serde_json::Value::Null);

    let metadata = json!({
        "model_name": model_name,
        "quantizer_version": "braidinfer-bqnt-v1",
        "default_format": format_str,
        "model_config": config_json,
        "quantization_stats": {
            "total_params": total_params,
            "quantized_params": quantized_params,
            "bf16_params": bf16_params,
            "tensor_count": tensor_count,
            "file_size_bytes": file_size,
            "effective_bpw": effective_bpw,
        }
    });

    writer
        .finish(&metadata.to_string())
        .expect("Failed to finalize BQNT file");

    let elapsed = start.elapsed();
    eprintln!("\nDone in {:.1}s", elapsed.as_secs_f64());
    eprintln!("Tensors: {tensor_count}");
    eprintln!("Params:  {total_params} total, {quantized_params} quantized, {bf16_params} bf16");
    eprintln!(
        "Size:    {:.1} GB ({:.2} bits/param)",
        file_size as f64 / 1e9,
        effective_bpw
    );
    eprintln!("Output:  {output_path}");
}
