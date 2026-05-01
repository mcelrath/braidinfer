//! Shared helpers and discriminant enum used by all compile_*.rs modules.

use super::instructions::*;
use super::Instruction;
use super::{
    OP_FFN_GATE_UP, OP_FFN_GATE_UP_RNF4, OP_FFN_GATE_UP_RNF4_WX, OP_FFN_GATE_UP_WX,
    OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_RMSNORM, OP_RMSNORM_WX,
};
use crate::model::{KvCache, LinearWeight, WeightFormat};

/// Emit a batched linear projection. For bf16, uses single batched instruction.
/// For quantized (PCG32/RNF4), emits per-token loop (kernel batching TODO: braidinfer-xxy).
pub(super) fn emit_batched_linear_proj(
    weight: &LinearWeight,
    output: *mut f32,
    input: *const f32,
    out_dim: usize,
    in_dim: usize,
    n: usize,
    instructions: &mut Vec<Instruction>,
) {
    let (opcode, w_ptr) = linear_proj_opcode_ptr(weight);
    let inst = LinearProjInst::new(opcode, out_dim as u32, output, w_ptr, input, out_dim as i32, in_dim as i32, n as i32);
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

/// Choose fused FFN gate+up opcode based on weight format and RMSNorm convention.
pub(super) fn ffn_gate_up_opcode(rnf4: bool, one_plus_w: bool) -> u32 {
    match (rnf4, one_plus_w) {
        (false, true)  => OP_FFN_GATE_UP,
        (false, false) => OP_FFN_GATE_UP_WX,
        (true,  true)  => OP_FFN_GATE_UP_RNF4,
        (true,  false) => OP_FFN_GATE_UP_RNF4_WX,
    }
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
