//! Weight quantization: rnf4_g128 (8.25-bit lossless) and PcG32Q4 (5-bit).
//!
//! Quantizes bf16 weights at load time with zero calibration data.
//! rnf4_g128 uses two rounds of NF4 (Normal Float 4) quantization with residual
//! correction, achieving lossless quality (18.61 PPL vs 18.67 bf16 on Qwen3.5-0.8B).

use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream, HipResult};
use crate::kernel::LinearProjKernel;

// --- Formats and types ---

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightFormat {
    Bf16,
    Rnf4G128,  // residual NF4, group_size=128, 8.25 bits/element
    PcG32Q4,   // per-channel-group asymmetric 4-bit, group_size=32, 5.0 bits/element
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightQuantMode {
    Bf16,    // No quantization (default)
    Rnf4,    // All linear weights at rnf4_g128 (8.25 bits, lossless)
    Mixed,   // MLP at PcG32Q4 (5 bits), rest at rnf4_g128
}

/// Packed quantized weight buffer on GPU.
pub struct PackedWeights {
    pub data: DeviceBuffer<u8>,
    pub format: WeightFormat,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// A linear projection weight: either bf16 or quantized.
pub enum LinearWeight {
    Bf16(DeviceBuffer<u16>),
    Packed(PackedWeights),
}

impl LinearWeight {
    /// Get raw bf16 pointer for megakernel instruction packing.
    /// Panics if weight is quantized — megakernel only supports bf16.
    pub fn as_bf16_ptr(&self) -> *const u16 {
        match self {
            LinearWeight::Bf16(buf) => buf.as_ptr(),
            LinearWeight::Packed(_) => panic!("Cannot use quantized weights with megakernel — use WEIGHT_QUANT=bf16 for dense models"),
        }
    }

    /// Get bf16 DeviceBuffer reference for megakernel / fused kernels.
    pub fn as_bf16(&self) -> &DeviceBuffer<u16> {
        match self {
            LinearWeight::Bf16(buf) => buf,
            LinearWeight::Packed(_) => panic!("Cannot use quantized weights with megakernel"),
        }
    }

    /// Get raw data pointer (u8) for any weight format.
    /// For bf16: cast u16* to u8*. For packed: direct data pointer.
    pub fn raw_data_ptr(&self) -> *const u8 {
        match self {
            LinearWeight::Bf16(buf) => buf.as_ptr() as *const u8,
            LinearWeight::Packed(pw) => pw.data.as_ptr(),
        }
    }

    /// Get weight format.
    pub fn weight_format(&self) -> WeightFormat {
        match self {
            LinearWeight::Bf16(_) => WeightFormat::Bf16,
            LinearWeight::Packed(pw) => pw.format,
        }
    }

    /// Number of logical elements (out_dim * in_dim).
    pub fn num_elements(&self) -> usize {
        match self {
            LinearWeight::Bf16(buf) => buf.len(),
            LinearWeight::Packed(pw) => pw.out_dim * pw.in_dim,
        }
    }

    /// Device this weight resides on.
    pub fn device(&self) -> DeviceId {
        match self {
            LinearWeight::Bf16(buf) => buf.device(),
            LinearWeight::Packed(pw) => pw.data.device(),
        }
    }

    /// Dispatch linear projection through the appropriate kernel.
    pub fn forward(
        &self,
        kernel: &LinearProjKernel,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        out_dim: u32,
        in_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        match self {
            LinearWeight::Bf16(buf) => kernel.forward(output, buf, input, out_dim, in_dim, stream),
            LinearWeight::Packed(pw) => kernel.forward_packed(output, pw, input, stream),
        }
    }

    /// Dispatch with raw output/input pointers for MoE expert sub-buffer access.
    /// `byte_offset`: offset in bytes into the underlying buffer.
    pub fn forward_sub(
        &self,
        kernel: &LinearProjKernel,
        output: *mut f32,
        input: *const f32,
        out_dim: u32,
        in_dim: u32,
        byte_offset: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        match self {
            LinearWeight::Bf16(buf) => {
                let w_ptr = unsafe { (buf.as_ptr() as *const u8).add(byte_offset) as *const u16 };
                kernel.forward_ptr(output, w_ptr, input, out_dim, in_dim, stream)
            }
            LinearWeight::Packed(pw) => {
                let func_name = match pw.format {
                    WeightFormat::Rnf4G128 => "linear_proj_rnf4_g128",
                    WeightFormat::PcG32Q4 => "linear_proj_pcg32_q4",
                    WeightFormat::Bf16 => "linear_proj_f32",
                };
                kernel.forward_packed_ptr(
                    output,
                    unsafe { pw.data.as_ptr().add(byte_offset) },
                    input, out_dim, in_dim, func_name, stream,
                )
            }
        }
    }

    /// Compute byte offset for `row_start` rows, with explicit in_dim.
    pub fn row_byte_offset_dim(&self, row_start: usize, in_dim: usize) -> usize {
        match self {
            LinearWeight::Bf16(_) => row_start * in_dim * 2,
            LinearWeight::Packed(pw) => match pw.format {
                WeightFormat::Bf16 => row_start * in_dim * 2,
                WeightFormat::Rnf4G128 => {
                    let groups_per_row = (in_dim + 127) / 128;
                    row_start * groups_per_row * 132
                }
                WeightFormat::PcG32Q4 => {
                    let groups_per_row = (in_dim + 31) / 32;
                    row_start * groups_per_row * 20
                }
            },
        }
    }
}

// --- NF4 constants ---

/// NF4 codebook: 16 quantile-matched levels for N(0,1), from QLoRA.
pub const NF4_TABLE: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500,  0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0,
];

/// NF4 decision boundaries: midpoints between adjacent codebook levels.
/// Used by `nf4_quantize` to map normalized values to 4-bit indices.
pub const NF4_BOUNDARIES: [f32; 15] = [
    (NF4_TABLE[0]  + NF4_TABLE[1])  / 2.0,
    (NF4_TABLE[1]  + NF4_TABLE[2])  / 2.0,
    (NF4_TABLE[2]  + NF4_TABLE[3])  / 2.0,
    (NF4_TABLE[3]  + NF4_TABLE[4])  / 2.0,
    (NF4_TABLE[4]  + NF4_TABLE[5])  / 2.0,
    (NF4_TABLE[5]  + NF4_TABLE[6])  / 2.0,
    (NF4_TABLE[6]  + NF4_TABLE[7])  / 2.0,
    (NF4_TABLE[7]  + NF4_TABLE[8])  / 2.0,
    (NF4_TABLE[8]  + NF4_TABLE[9])  / 2.0,
    (NF4_TABLE[9]  + NF4_TABLE[10]) / 2.0,
    (NF4_TABLE[10] + NF4_TABLE[11]) / 2.0,
    (NF4_TABLE[11] + NF4_TABLE[12]) / 2.0,
    (NF4_TABLE[12] + NF4_TABLE[13]) / 2.0,
    (NF4_TABLE[13] + NF4_TABLE[14]) / 2.0,
    (NF4_TABLE[14] + NF4_TABLE[15]) / 2.0,
];

// --- CPU-side format conversion ---

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + rounding) >> 16) as u16
}

fn bf16_to_f32_cpu(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn unpack_bf16_group(src: &[u16], count: usize, dst: &mut [f32]) {
    for i in 0..count {
        dst[i] = bf16_to_f32_cpu(src[i]);
    }
}

// --- NF4 quantization ---

/// Map a normalized value (in [-1, 1]) to a 4-bit NF4 index (0..15).
///
/// Uses a binary search tree over NF4_BOUNDARIES: 4 comparisons instead of 15.
/// The tree structure: start at midpoint (index 7), then refine by halving.
/// Each step adds a power-of-two offset if the value is >= the boundary at
/// the current probe position. After 4 steps, the accumulated index is exact.
#[inline(always)]
pub fn nf4_quantize(x: f32) -> u8 {
    let b = &NF4_BOUNDARIES;
    let mut i: usize = 0;
    i += if x >= b[7]     { 8 } else { 0 };
    i += if x >= b[i + 3] { 4 } else { 0 };
    i += if x >= b[i + 1] { 2 } else { 0 };
    i += if x >= b[i]     { 1 } else { 0 };
    i as u8
}

// --- Group quantization ---

/// Quantize one group of 128 bf16 elements to rnf4_g128 packed format (132 bytes).
/// Layout: [64B idx1_packed | 2B absmax1_bf16 | 64B idx2_packed | 2B absmax2_bf16]
fn quantize_rnf4_group(bf16_data: &[u16], count: usize, out: &mut [u8]) {
    let mut vals = [0.0f32; 128];
    unpack_bf16_group(bf16_data, count, &mut vals);

    // Round 1: NF4 with absmax
    let absmax1 = vals[..count].iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-10);
    let inv1 = 1.0 / absmax1;
    let mut idx1 = [0u8; 128];
    let mut dequant1 = [0.0f32; 128];
    for i in 0..count {
        idx1[i] = nf4_quantize(vals[i] * inv1);
        dequant1[i] = NF4_TABLE[idx1[i] as usize] * absmax1;
    }

    // Round 2: NF4 on residual
    let mut absmax2 = 0.0f32;
    for i in 0..count {
        absmax2 = absmax2.max((vals[i] - dequant1[i]).abs());
    }
    absmax2 = absmax2.max(1e-10);
    let inv2 = 1.0 / absmax2;
    let mut idx2 = [0u8; 128];
    for i in 0..count {
        idx2[i] = nf4_quantize((vals[i] - dequant1[i]) * inv2);
    }

    // Pack: 2 values per byte, low nibble first
    for i in (0..128).step_by(2) {
        out[i / 2] = (idx1[i] & 0xF) | ((idx1[i + 1] & 0xF) << 4);
    }
    let a1 = f32_to_bf16(absmax1);
    out[64] = (a1 & 0xFF) as u8;
    out[65] = (a1 >> 8) as u8;
    for i in (0..128).step_by(2) {
        out[66 + i / 2] = (idx2[i] & 0xF) | ((idx2[i + 1] & 0xF) << 4);
    }
    let a2 = f32_to_bf16(absmax2);
    out[130] = (a2 & 0xFF) as u8;
    out[131] = (a2 >> 8) as u8;
}

// --- Public quantization API ---

/// Quantize bf16 weights to rnf4_g128 format (parallel over rows).
pub fn quantize_rnf4_g128(bf16_data: &[u16], out_dim: usize, in_dim: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 128;
    let group_bytes = 132;
    let num_groups_per_row = (in_dim + group_size - 1) / group_size;
    let total_bytes = out_dim * num_groups_per_row * group_bytes;
    let mut packed = vec![0u8; total_bytes];

    packed.par_chunks_mut(num_groups_per_row * group_bytes)
        .enumerate()
        .for_each(|(row, row_out)| {
            for g in 0..num_groups_per_row {
                let base = row * in_dim + g * group_size;
                let count = std::cmp::min(group_size, in_dim - g * group_size);
                let dst = g * group_bytes;
                quantize_rnf4_group(&bf16_data[base..base + count], count, &mut row_out[dst..dst + group_bytes]);
            }
        });
    packed
}

/// Quantize bf16 weights to PcG32Q4 format (parallel over rows).
pub fn quantize_pc_g32_q4(bf16_data: &[u16], out_dim: usize, in_dim: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 32;
    let group_bytes = 20;
    let num_groups_per_row = (in_dim + group_size - 1) / group_size;
    let total_bytes = out_dim * num_groups_per_row * group_bytes;
    let mut packed = vec![0u8; total_bytes];

    packed.par_chunks_mut(num_groups_per_row * group_bytes)
        .enumerate()
        .for_each(|(row, row_out)| {
            for g in 0..num_groups_per_row {
                let base = row * in_dim + g * group_size;
                let count = std::cmp::min(group_size, in_dim - g * group_size);

                let mut vals = [0.0f32; 32];
                unpack_bf16_group(&bf16_data[base..base + count], count, &mut vals);

                let mn = vals[..count].iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = vals[..count].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((mx - mn) / 15.0).max(1e-10);
                let inv_scale = 1.0 / scale;

                let mut indices = [0u8; 32];
                for i in 0..count {
                    indices[i] = ((vals[i] - mn) * inv_scale).round().clamp(0.0, 15.0) as u8;
                }

                let dst = g * group_bytes;
                for i in (0..32).step_by(2) {
                    row_out[dst + i / 2] = (indices[i] & 0xF) | ((indices[i + 1] & 0xF) << 4);
                }
                let mn_bf16 = f32_to_bf16(mn);
                let sc_bf16 = f32_to_bf16(scale);
                row_out[dst + 16] = (mn_bf16 & 0xFF) as u8;
                row_out[dst + 17] = (mn_bf16 >> 8) as u8;
                row_out[dst + 18] = (sc_bf16 & 0xFF) as u8;
                row_out[dst + 19] = (sc_bf16 >> 8) as u8;
            }
        });
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_quantize_covers_all_bins() {
        for (i, &level) in NF4_TABLE.iter().enumerate() {
            assert_eq!(nf4_quantize(level), i as u8, "NF4_TABLE[{i}] = {level} should map to bin {i}");
        }
    }

    #[test]
    fn nf4_boundaries_are_midpoints() {
        for i in 0..15 {
            let expected = (NF4_TABLE[i] + NF4_TABLE[i + 1]) / 2.0;
            assert!((NF4_BOUNDARIES[i] - expected).abs() < 1e-7, "boundary {i}");
        }
    }
}
