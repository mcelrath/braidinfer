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

/// Convert FP8_E5M2 byte to f32.
/// E5M2: 1 sign, 5 exponent (bias=15), 2 mantissa. Inf/NaN encodings exist.
fn fp8_e5m2_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 2) & 0x1F;
    let mant = b & 0x3;
    if exp == 0x1F {
        if mant != 0 {
            return 0.0; // NaN → 0
        } else {
            // Inf → clamp to bf16::MAX (0x7F7F = 3.3895313e38)
            let bf16_max = bf16_to_f32(0x7F7F);
            return if sign == 1 { -bf16_max } else { bf16_max };
        }
    }
    let val = if exp == 0 {
        // subnormal: (-1)^sign * 2^(-14) * (mant / 4)
        (mant as f32 / 4.0) * (1.0 / (1u32 << 14) as f32)
    } else {
        // normal: (-1)^sign * 2^(exp-15) * (1 + mant/4)
        (1.0 + mant as f32 / 4.0) * f32::from_bits(((exp as u32 + 112) & 0xFF) << 23)
    };
    if sign == 1 { -val } else { val }
}

/// Per-channel or scalar scale tensor.
enum ScaleTensor {
    Scalar(f32),
    PerChannel(Vec<f32>),
}

/// Read a scale tensor, handling scalar (1 element) or per-channel (out_dim elements).
fn read_scale_tensor(tensors: &SafeTensors, key: &str, out_dim: usize) -> Option<ScaleTensor> {
    let st = tensors.tensor(key).ok()?;
    let data = st.data();
    let dtype = st.dtype();
    let elem_size = match dtype {
        Dtype::F32 => 4,
        Dtype::BF16 => 2,
        _ => return None,
    };
    let n_elems = data.len() / elem_size;
    let read_f32 = |i: usize| -> f32 {
        match dtype {
            Dtype::F32 => {
                let off = i * 4;
                f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
            }
            Dtype::BF16 => {
                let off = i * 2;
                bf16_to_f32(u16::from_le_bytes([data[off], data[off+1]]))
            }
            _ => 1.0,
        }
    };
    if n_elems == 1 {
        Some(ScaleTensor::Scalar(read_f32(0)))
    } else if n_elems == out_dim {
        Some(ScaleTensor::PerChannel((0..out_dim).map(read_f32).collect()))
    } else if n_elems > 0 {
        // Unexpected size: use first element as scalar fallback
        Some(ScaleTensor::Scalar(read_f32(0)))
    } else {
        None
    }
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
    "qscale_act",         // FP8 per-tensor scales (scalar metadata)
    "weight_scale_inv",   // Mistral FP8 scale (inverse)
    "activation_scale",   // Mistral FP8 activation scale
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
    let max_tensors: usize = shards
        .iter()
        .map(|shard_path| {
            let data = std::fs::read(shard_path).expect("Failed to read shard");
            SafeTensors::deserialize(&data)
                .map(|st| st.names().len())
                .unwrap_or(0)
        })
        .sum();
    eprintln!("Total tensors (first pass): {max_tensors}");

    let mut writer = BqntWriter::create(Path::new(&output_path), max_tensors)
        .expect("Failed to create output file");

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
                Dtype::F8_E4M3 | Dtype::F8_E5M2 => {
                    // FP8 -> BF16: dequantize each byte, apply scale.
                    // Two naming conventions:
                    //   qscale_weight: bf16 = fp8 * qscale_weight  (braidinfer internal)
                    //   weight_scale_inv: bf16 = fp8 * weight_scale_inv  (Mistral FP8)
                    let fp8_bytes: &[u8] = &raw[..n_elements];
                    let tensor_prefix = name.rsplit_once('.')
                        .map(|(prefix, _)| prefix)
                        .unwrap_or(name);

                    let decode: fn(u8) -> f32 = if dtype == Dtype::F8_E4M3 {
                        fp8_e4m3_to_f32
                    } else {
                        fp8_e5m2_to_f32
                    };

                    let scale_tensor = read_scale_tensor(
                        &safetensors, &format!("{tensor_prefix}.qscale_weight"), out_dim)
                    .or_else(|| read_scale_tensor(
                        &safetensors, &format!("{tensor_prefix}.weight_scale_inv"), out_dim));

                    let converted: Vec<u16> = match scale_tensor {
                        None => {
                            eprintln!("  WARN: no scale for FP8 tensor {name}, using 1.0");
                            fp8_bytes.iter().map(|&b| f32_to_bf16(decode(b))).collect()
                        }
                        Some(ScaleTensor::Scalar(s)) => {
                            fp8_bytes.iter().map(|&b| f32_to_bf16(decode(b) * s)).collect()
                        }
                        Some(ScaleTensor::PerChannel(ref scales)) => {
                            fp8_bytes
                                .chunks(in_dim)
                                .enumerate()
                                .flat_map(|(r, row)| {
                                    let s = scales[r];
                                    row.iter().map(move |&b| f32_to_bf16(decode(b) * s))
                                })
                                .collect()
                        }
                    };
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
                .write_tensor(name, braidinfer_runtime::bqnt::StorageDtype::from_weight_format(fmt), out_dim as u32, in_dim as u32, ndim, &packed)
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

#[cfg(test)]
mod tests {
    use super::*;

    // E5M2: normal value: sign=0 exp=16 (bias=15 → 2^1) mant=1 → 1.0 * 2^1 * (1+0.25) = 2.5
    // byte: 0b0_10000_01 = 0x41
    #[test]
    fn test_fp8_e5m2_normal() {
        let b = 0b0_10000_01u8; // exp=16, mant=1
        let v = fp8_e5m2_to_f32(b);
        assert!((v - 2.5).abs() < 1e-6, "expected 2.5, got {v}");
    }

    // E5M2: subnormal: sign=0 exp=0 mant=2 → 2^-14 * (2/4) = 2^-14 * 0.5 = 2^-15
    // byte: 0b0_00000_10 = 0x02
    #[test]
    fn test_fp8_e5m2_subnormal() {
        let b = 0b0_00000_10u8;
        let v = fp8_e5m2_to_f32(b);
        let expected = 0.5 / (1u32 << 14) as f32;
        assert!((v - expected).abs() < 1e-10, "expected {expected}, got {v}");
    }

    // E5M2: NaN (exp=0x1F, mant!=0) → 0.0
    // byte: 0b0_11111_01 = 0x7D
    #[test]
    fn test_fp8_e5m2_nan_to_zero() {
        let b = 0b0_11111_01u8;
        let v = fp8_e5m2_to_f32(b);
        assert_eq!(v, 0.0, "NaN should map to 0.0, got {v}");
    }

    // E5M2: Inf (exp=0x1F, mant=0) → bf16::MAX
    // byte: 0b0_11111_00 = 0x7C
    #[test]
    fn test_fp8_e5m2_inf_to_bf16_max() {
        let b = 0b0_11111_00u8;
        let v = fp8_e5m2_to_f32(b);
        let bf16_max = bf16_to_f32(0x7F7F);
        assert_eq!(v, bf16_max, "Inf should map to bf16::MAX ({bf16_max}), got {v}");
    }

    // Per-channel scale: 2 rows with different scales
    #[test]
    fn test_per_channel_scale_application() {
        // 2 rows, 2 cols, fp8 E4M3 bytes all 0x3C (value 1.0 in E4M3)
        // verify by checking output bf16 values
        // 0x38: sign=0 exp=7 mant=0 → (1+0/8)*2^(7-7) = 1.0
        let b = fp8_e4m3_to_f32(0x38);
        assert!((b - 1.0).abs() < 1e-5, "fp8 0x38 should be ~1.0, got {b}");

        let scales = vec![2.0f32, 3.0f32];
        let fp8_bytes = [0x38u8; 4]; // 2 rows × 2 cols
        let in_dim = 2usize;
        let converted: Vec<u16> = fp8_bytes
            .chunks(in_dim)
            .enumerate()
            .flat_map(|(r, row)| {
                let s = scales[r];
                row.iter().map(move |&byte| f32_to_bf16(fp8_e4m3_to_f32(byte) * s))
            })
            .collect();

        let row0 = bf16_to_f32(converted[0]);
        let row0b = bf16_to_f32(converted[1]);
        let row1 = bf16_to_f32(converted[2]);
        let row1b = bf16_to_f32(converted[3]);
        assert!((row0 - 2.0).abs() < 0.01, "row0 col0: expected 2.0, got {row0}");
        assert!((row0b - 2.0).abs() < 0.01, "row0 col1: expected 2.0, got {row0b}");
        assert!((row1 - 3.0).abs() < 0.01, "row1 col0: expected 3.0, got {row1}");
        assert!((row1b - 3.0).abs() < 0.01, "row1 col1: expected 3.0, got {row1b}");
    }
}
