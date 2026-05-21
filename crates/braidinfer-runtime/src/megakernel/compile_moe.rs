//! MoE FFN layer compilation: shared expert helpers and single/multi-GPU MoE dispatch.

use super::compile_common::{div_ceil, linear_proj_opcode_ptr, rmsnorm_opcode};
use super::instructions::*;
use super::{Instruction, INST_SIZE, MegakernelProgram};
use super::{OP_LINEAR_PROJ, OP_MOE_FFN};
use crate::config::ModelConfig;
use crate::weights::{ActivationBuffers, LayerWeights};

impl MegakernelProgram {
    /// Emit shared expert instructions into `instructions`.
    pub(super) fn emit_shared_expert(
        se: &crate::weights::DenseFfnWeights,
        moe: &crate::weights::MoeWeights,
        act: &ActivationBuffers,
        hs: usize,
        se_is: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        if moe.has_gate_proj {
            let (op, wp) = linear_proj_opcode_ptr(&se.gate_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_gate.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(SiluMulInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_gate.as_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        } else {
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(ReluSqInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        }
        let (op, wp) = linear_proj_opcode_ptr(&se.down_proj);
        instructions.push(LinearProjInst::new(op, hs as u32, act.moe_expert_out.as_write_ptr(), wp, act.moe_expert_act.as_ptr(), hs as i32, se_is as i32, 0).into_inst());

        if let Some(ref gate_buf) = moe.shared_expert_gate {
            instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, 1, act.moe_scores.as_write_ptr(), gate_buf.as_ptr() as *const u8, act.normed.as_ptr(), 1, hs as i32, 0).into_inst());
            instructions.push(SigmoidWeightedAddInst::new(div_ceil(hs as u32, 256), act.ffn_down.as_write_ptr(), act.moe_scores.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        } else {
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.ffn_down.as_write_ptr(), act.ffn_down.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        }
    }

    // Overload for ffn_down_stage output (multi-GPU path)
    pub(super) fn emit_shared_expert_stage(
        se: &crate::weights::DenseFfnWeights,
        moe: &crate::weights::MoeWeights,
        act: &ActivationBuffers,
        hs: usize,
        se_is: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        if moe.has_gate_proj {
            let (op, wp) = linear_proj_opcode_ptr(&se.gate_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_gate.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(SiluMulInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_gate.as_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        } else {
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(ReluSqInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        }
        let (op, wp) = linear_proj_opcode_ptr(&se.down_proj);
        instructions.push(LinearProjInst::new(op, hs as u32, act.moe_expert_out.as_write_ptr(), wp, act.moe_expert_act.as_ptr(), hs as i32, se_is as i32, 0).into_inst());

        if let Some(ref gate_buf) = moe.shared_expert_gate {
            instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, 1, act.moe_scores.as_write_ptr(), gate_buf.as_ptr() as *const u8, act.normed.as_ptr(), 1, hs as i32, 0).into_inst());
            instructions.push(SigmoidWeightedAddInst::new(div_ceil(hs as u32, 256), act.ffn_down_stage.as_write_ptr(), act.moe_scores.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        } else {
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.ffn_down_stage.as_write_ptr(), act.ffn_down_stage.as_ptr() as *const f32, act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        }
    }

    /// Compile MoE FFN for one layer: norm + gate + OP_MOE_GATE + OP_MOE_FFN + shared expert + residual.
    pub(super) fn compile_moe_ffn(
        cfg: &ModelConfig,
        layer_idx: usize,
        layer: &LayerWeights,
        moe: &crate::weights::MoeWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::config::{FfnType, GateType};
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let (k, gate_type, eis) = match &cfg.layers[layer_idx].ffn_type {
            FfnType::MoE {
                num_active,
                gate_type,
                expert_intermediate_size,
                ..
            } => (*num_active, gate_type.clone(), *expert_intermediate_size),
            _ => unreachable!(),
        };
        let ne = moe.num_experts;

        // Get norm weight pointer
        let norm_ptr = match layer {
            LayerWeights::Attention(w) => w.post_norm.as_ptr(),
            LayerWeights::Gdn(w) => w.post_norm.as_ptr(),
            LayerWeights::MoeFfn(w) => w.input_norm.as_ptr(),
            _ => panic!("no norm weight for MoE FFN layer"),
        };

        // D2D_COPY: hidden → residual (NO_SYNC: norm reads hidden)
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());

        // RMSNorm: hidden → normed
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), norm_ptr, hs as i32, eps).into_inst());

        // Gate projection: normed → moe_scores[num_experts]
        instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, ne as u32, act.moe_scores.as_write_ptr(), moe.gate.as_ptr() as *const u8, act.normed.as_ptr(), ne as i32, hs as i32, 0).into_inst());

        // OP_MOE_GATE: top-k selection on GPU
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref().map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());
        instructions.push(MoeGateInst::new(act.moe_scores.as_ptr(), act.moe_expert_ids.as_write_ptr(), act.moe_expert_weights.as_write_ptr(), ne as i32, k as i32, gate_mode, rsf, bias_ptr).into_inst());

        // OP_MOE_FFN: fused expert loop (internal grid.sync())
        // Currently only supports PcG32Q4 weights in the GPU kernel
        assert!(
            matches!(
                moe.expert_gate_up.weight_format(),
                crate::quant::WeightFormat::PcG32Q4
            ),
            "OP_MOE_FFN only supports PcG32Q4 expert weights (got {:?})",
            moe.expert_gate_up.weight_format()
        );
        let gate_up_expert_stride = if moe.has_gate_proj {
            moe.expert_gate_up.row_byte_offset_dim(2 * eis, hs)
        } else {
            moe.expert_gate_up.row_byte_offset_dim(eis, hs)
        };
        let down_expert_stride = moe.expert_down.row_byte_offset_dim(hs, eis);
        let gate_up_row_stride = moe.expert_gate_up.row_byte_offset_dim(1, hs);

        let flags =
            (if moe.has_gate_proj { 1u32 } else { 0 }) | (if !moe.has_gate_proj { 2 } else { 0 }); // bit1 = relu²

        let grid_x = std::cmp::max(eis, hs) as u32;

        instructions.push(Instruction {
            words: {
                let mut w = [0u64; INST_SIZE];
                w[0] = make_opcode_gridx(OP_MOE_FFN, grid_x);
                w[1] = act.moe_expert_ids.as_ptr() as u64;
                w[2] = act.moe_expert_weights.as_ptr() as u64;
                w[3] = act.normed.as_ptr() as u64;
                w[4] = act.ffn_down.as_write_ptr() as u64;
                w[5] = moe.expert_gate_up.raw_data_ptr() as u64;
                w[6] = gate_up_expert_stride as u64;
                w[7] = moe.expert_down.raw_data_ptr() as u64;
                w[8] = down_expert_stride as u64;
                w[9] = k as u64;
                w[10] = (hs | (eis << 16)) as u64;
                w[11] = flags as u64;
                w[12] = act.moe_expert_gate.as_ptr() as u64;
                w[13] = act.moe_expert_up.as_ptr() as u64;
                w[14] = act.moe_expert_act.as_ptr() as u64;
                w[15] = act.moe_expert_out.as_ptr() as u64;
                w[16] = gate_up_row_stride as u64;
                w
            }
        });

        // Shared expert (if present)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &cfg.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                }
                _ => eis,
            };

            Self::emit_shared_expert(se, moe, act, hs, se_is, instructions);
        }

        // Residual: hidden = residual + ffn_down
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.residual.as_ptr(), act.ffn_down.as_ptr(), hs as i32).into_inst());
    }

    /// Multi-GPU variant: emit norm + gate proj + OP_MOE_GATE + OP_BARRIER.
    /// CPU dispatch loop handles expert FFN; megakernel resumes for shared expert + residual.
    /// Returns the instruction index of the emitted OP_BARRIER.
    ///
    /// Note: barrier_flag_ptr and resume_flag_ptr are patched in after MoeBarrierState is
    /// allocated (in compile_multi_gpu). Initially zero; patched by execute_multi_gpu().
    /// `emit_post_barrier`: when false, skip shared-expert + residual_add after OP_BARRIER.
    /// Used by compile_inner_p2p for Nemotron-H fc2_latent_proj layers, where those
    /// instructions are inserted in the correct form after OP_MOE_DISPATCH instead.
    pub(super) fn compile_moe_ffn_multi_gpu(
        cfg: &ModelConfig,
        layer_idx: usize,
        layer: &LayerWeights,
        moe: &crate::weights::MoeWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
        emit_post_barrier: bool,
    ) -> usize {
        use crate::config::{FfnType, GateType};
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let (k, gate_type, ne, eis) = match &cfg.layers[layer_idx].ffn_type {
            FfnType::MoE {
                num_active,
                gate_type,
                num_experts,
                expert_intermediate_size,
                ..
            } => (
                *num_active,
                gate_type.clone(),
                *num_experts,
                *expert_intermediate_size,
            ),
            _ => unreachable!(),
        };

        let norm_ptr = match layer {
            LayerWeights::Attention(w) => w.post_norm.as_ptr(),
            LayerWeights::Gdn(w) => w.post_norm.as_ptr(),
            LayerWeights::MoeFfn(w) => w.input_norm.as_ptr(),
            _ => panic!("no norm weight for MoE FFN layer"),
        };

        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), norm_ptr, hs as i32, eps).into_inst());
        instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, ne as u32, act.moe_scores.as_write_ptr(), moe.gate.as_ptr() as *const u8, act.normed.as_ptr(), ne as i32, hs as i32, 0).into_inst());
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.normed_stage.as_write_ptr(), act.normed.as_ptr(), hs as i32).into_inst());

        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref().map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());
        instructions.push(MoeGateInst::new(act.moe_scores.as_ptr(), act.moe_expert_ids.as_write_ptr(), act.moe_expert_weights.as_write_ptr(), ne as i32, k as i32, gate_mode, rsf, bias_ptr).into_inst());

        // OP_BARRIER: grid_x=1: only block 0 runs op_barrier
        let barrier_inst_idx = instructions.len();
        instructions.push(BarrierInst::new(layer_idx as i32).into_inst());

        if emit_post_barrier {
            // After barrier: compute shared expert (if present) and add to ffn_down_stage.
            if let Some(ref se) = moe.shared_expert {
                let se_is = match &cfg.layers[layer_idx].ffn_type {
                    FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                        if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                    }
                    _ => eis,
                };
                Self::emit_shared_expert_stage(se, moe, act, hs, se_is, instructions);
            }

            // Final residual: hidden = residual + ffn_down_stage
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.residual.as_ptr(), act.ffn_down_stage.as_ptr() as *const f32, hs as i32).into_inst());
        }

        barrier_inst_idx
    }
}
