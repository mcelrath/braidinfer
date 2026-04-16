//! Shared helpers and discriminant enum used by all compile_*.rs modules.

use super::instructions::*;
use super::Instruction;
use super::{OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_RMSNORM, OP_RMSNORM_WX};
use crate::model::{KvCache, LinearWeight, WeightFormat};

pub(super) fn emit_linear_proj(inst: &mut Instruction, weight: &LinearWeight, ptr_slot: usize) {
    match weight {
        LinearWeight::Bf16(buf) => {
            inst.words[ptr_slot] = buf.as_ptr() as u64;
        }
        LinearWeight::Packed(pw) => {
            let op = match pw.format {
                WeightFormat::Rnf4G128 => OP_LINEAR_PROJ_RNF4,
                WeightFormat::PcG32Q4 => OP_LINEAR_PROJ_PCG32,
                WeightFormat::Bf16 => OP_LINEAR_PROJ,
            };
            // Replace opcode (low 32 bits), preserve grid_x (high 32 bits)
            inst.words[0] = (inst.words[0] & 0xFFFF_FFFF_0000_0000u64) | op as u64;
            inst.words[ptr_slot] = pw.data.as_ptr() as u64;
        }
    }
}

/// Emit a batched linear projection. For bf16, uses single batched instruction.
/// For quantized (PCG32/RNF4), emits per-token loop (kernel batching TODO: braidinfer-xxy).
pub(super) fn emit_batched_linear_proj(
    weight: &LinearWeight,
    output: *mut f32,
    input: *const f32,
    out_dim: usize,
    in_dim: usize,
    n: usize,
    no_sync: bool,
    instructions: &mut Vec<Instruction>,
) {
    let (opcode, w_ptr) = linear_proj_opcode_ptr(weight);
    let inst = LinearProjInst::new(opcode, out_dim as u32, output, w_ptr, input, out_dim as i32, in_dim as i32, n as i32);
    let inst = if no_sync { inst.no_sync() } else { inst };
    instructions.push(inst.into_inst());
}

/// Return (opcode, weight_data_ptr) for a LinearWeight.
pub(super) fn linear_proj_opcode_ptr(weight: &LinearWeight) -> (u32, *const u8) {
    match weight {
        LinearWeight::Bf16(buf) => (OP_LINEAR_PROJ, buf.as_ptr() as *const u8),
        LinearWeight::Packed(pw) => {
            let op = match pw.format {
                WeightFormat::Rnf4G128 => OP_LINEAR_PROJ_RNF4,
                WeightFormat::PcG32Q4 => OP_LINEAR_PROJ_PCG32,
                WeightFormat::Bf16 => OP_LINEAR_PROJ,
            };
            (op, pw.data.as_ptr())
        }
    }
}

/// Choose RMSNorm opcode based on model config.
pub(super) fn rmsnorm_opcode(one_plus_w: bool) -> u32 {
    if one_plus_w { OP_RMSNORM } else { OP_RMSNORM_WX }
}

pub(super) fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Discriminates the variant parts (KV write + attention op) of an attention layer.
pub(super) enum AttentionVariant<'a> {
    /// Flat (non-paged) decode: GQA attention, KV written after mRoPE.
    FlatKv { kv_cache: &'a KvCache },
    /// Paged decode: OP_ATTN_PAGED, KV written BEFORE mRoPE.
    PagedKv { attn_layer_index: usize },
    /// Prefill (N tokens): OP_ATTN_PREFILL, bulk KV write after mRoPE.
    Prefill {
        kv_cache: &'a KvCache,
        start_pos: u32,
    },
}
