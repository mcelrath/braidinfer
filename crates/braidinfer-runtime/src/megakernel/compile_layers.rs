//! GDN, Mamba2, and FFN layer compilation.

use super::compile_common::{div_ceil, emit_linear_proj, linear_proj_opcode_ptr, rmsnorm_opcode};
use crate::quant::WeightFormat;
use super::instructions::*;
use super::{FLAG_NO_SYNC, Instruction, MegakernelProgram, PrefillBuffers};
use super::{OP_FFN_DOWN_RES, OP_FFN_DOWN_RES_RNF4, OP_FFN_GATE_UP, OP_FFN_GATE_UP_RNF4, OP_LINEAR_PROJ};
#[allow(unused_imports)]
use crate::model::{
    ActivationBuffers, GdnState, LayerWeights, Mamba2State, ModelConfig, RecurrentLayerKind,
};

impl MegakernelProgram {
    pub(super) fn compile_gdn_layer(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        conv_state: &braidinfer_hip::memory::DeviceBuffer<f32>,
        gdn_state: &GdnState,
        instructions: &mut Vec<Instruction>,
    ) {
        let w = match layer {
            LayerWeights::Gdn(w) => w,
            _ => panic!("expected GDN layer"),
        };
        let hs = cfg.hidden_size;
        let nh = cfg.linear_num_heads;
        let nvh = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let ck = cfg.linear_conv_kernel_dim;
        let qkv_dim = nh * kd * 2 + nvh * vd;
        let eps = cfg.rms_norm_eps;

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.normed.as_write_ptr(), act.hidden.as_ptr(), w.input_norm.as_ptr(), hs as i32, eps,
        ).into_inst());

        // 2. QKV projection — NO_SYNC: next 3 instructions (a/b/z proj) read normed, not qkv
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_qkv);
            instructions.push(LinearProjInst::new(op, qkv_dim as u32, act.qkv.as_write_ptr(), wp, act.normed.as_ptr(), qkv_dim as i32, hs as i32, 0).no_sync().into_inst());
        }

        // 3. Project a, b, z
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_a);
            instructions.push(LinearProjInst::new(op, nvh as u32, act.a_proj.as_write_ptr(), wp, act.normed.as_ptr(), nvh as i32, hs as i32, 0).no_sync().into_inst());
        }
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_b);
            instructions.push(LinearProjInst::new(op, nvh as u32, act.b_proj.as_write_ptr(), wp, act.normed.as_ptr(), nvh as i32, hs as i32, 0).no_sync().into_inst());
        }
        // z proj: SYNC ensures QKV+a+b+z all complete before conv1d reads qkv
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_z);
            instructions.push(LinearProjInst::new(op, (nvh * vd) as u32, act.z_proj.as_write_ptr(), wp, act.normed.as_ptr(), (nvh * vd) as i32, hs as i32, 0).into_inst());
        }

        // 4. Causal conv1d on QKV (3 separate calls for q, k, v slices)
        let q_dim = nh * kd;
        let k_dim = nh * kd;
        let v_dim = nvh * vd;

        // Conv on Q — NO_SYNC
        instructions.push(Conv1dInst::new(
            div_ceil(q_dim as u32, 256),
            conv_state.as_write_ptr(), act.qkv.as_ptr(), w.conv1d_weight_q.as_ptr(), act.q_gdn.as_write_ptr(),
            q_dim as i32, ck as i32,
        ).no_sync().into_inst());

        // Conv on K — NO_SYNC
        instructions.push(Conv1dInst::new(
            div_ceil(k_dim as u32, 256),
            unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) },
            unsafe { act.qkv.as_ptr().add(q_dim) },
            w.conv1d_weight_k.as_ptr(), act.k_gdn.as_write_ptr(),
            k_dim as i32, ck as i32,
        ).no_sync().into_inst());

        // Conv on V — NO_SYNC (GDN gate reads a_proj, not v_gdn; next sync covers v_gdn)
        instructions.push(Conv1dInst::new(
            div_ceil(v_dim as u32, 256),
            unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) },
            unsafe { act.qkv.as_ptr().add(q_dim + k_dim) },
            w.conv1d_weight_v.as_ptr(), act.v_gdn.as_write_ptr(),
            v_dim as i32, ck as i32,
        ).no_sync().into_inst());

        // 5. GDN gate
        let gqa_group = nvh / nh;
        instructions.push(GdnGateInst::new(
            div_ceil(nvh as u32, 256),
            act.gate_gdn.as_write_ptr(), act.a_proj.as_ptr(), w.a_log.as_ptr(), w.dt_bias.as_ptr(), nvh as i32,
        ).into_inst());

        // 6. GDN recurrent
        instructions.push(GdnRecurInst::new(
            nvh as u32,
            act.q_gdn.as_ptr(), act.k_gdn.as_ptr(), act.v_gdn.as_ptr(), act.gate_gdn.as_ptr(), act.b_proj.as_ptr(),
            gdn_state.recurrent.as_write_ptr(), act.recurrent_out.as_write_ptr(),
            kd as i32, vd as i32, gqa_group as i32,
        ).into_inst());

        // 7. RMSNorm gated
        instructions.push(RmsNormGateInst::new(
            nvh as u32,
            act.normed_gated.as_write_ptr(), act.recurrent_out.as_ptr(), act.z_proj.as_ptr(), w.output_norm.as_ptr(),
            nvh as i32, vd as i32, eps,
        ).into_inst());

        // 8. Output projection
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_out);
            instructions.push(LinearProjInst::new(op, hs as u32, act.out_proj.as_write_ptr(), wp, act.normed_gated.as_ptr(), hs as i32, (nvh * vd) as i32, 0).into_inst());
        }

        // 9. Residual
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.out_proj.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
    }

    pub(super) fn compile_mamba2_layer(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        state: &Mamba2State,
        instructions: &mut Vec<Instruction>,
    ) {
        let w = match layer {
            LayerWeights::Mamba2(w) => w,
            _ => panic!("expected Mamba2 layer"),
        };
        let (nh, hd, sd, ck, ng, cd) = match &cfg.recurrent_kind {
            RecurrentLayerKind::Mamba2 {
                num_heads,
                head_dim,
                state_dim,
                conv_kernel,
                n_groups,
                conv_dim,
                ..
            } => (*num_heads, *head_dim, *state_dim, *conv_kernel, *n_groups, *conv_dim),
            _ => panic!("compile_mamba2_layer but no Mamba2 config"),
        };
        let hs = cfg.hidden_size;
        let intermediate = nh * hd;         // gate size + ssm output size
        let in_proj_size = intermediate + cd + nh; // gate + xBC + dt
        let eps = cfg.rms_norm_eps;
        let group_size = intermediate / ng; // value_dim per norm group

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.normed.as_write_ptr(), act.hidden.as_ptr(), w.input_norm.as_ptr(), hs as i32, eps,
        ).into_inst());

        // 2. in_proj
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.in_proj);
            instructions.push(LinearProjInst::new(op, in_proj_size as u32, act.mamba2_in_proj.as_write_ptr(), wp, act.normed.as_ptr(), in_proj_size as i32, hs as i32, 0).into_inst());
        }

        // 3. conv1d
        instructions.push(Mamba2Conv1dInst::new(
            div_ceil(cd as u32, 256),
            state.conv.as_write_ptr(),
            unsafe { act.mamba2_in_proj.as_ptr().add(intermediate) },
            w.conv1d_weight.as_ptr(),
            w.conv1d_bias.as_ptr(),
            act.mamba2_conv_out.as_write_ptr(),
            cd as i32, ck as i32,
        ).into_inst());

        // 4. SSM update
        instructions.push(SsmUpdateInst::new(
            nh as u32,
            state.ssm.as_write_ptr(),
            act.mamba2_conv_out.as_ptr(),
            unsafe { act.mamba2_in_proj.as_ptr().add(intermediate + cd) },
            w.dt_bias.as_ptr(),
            w.a_log.as_ptr(),
            unsafe { act.mamba2_conv_out.as_ptr().add(intermediate) },
            unsafe { act.mamba2_conv_out.as_ptr().add(intermediate + ng * sd) },
            w.d.as_ptr(),
            act.mamba2_ssm_out.as_write_ptr(),
            nh as i32, hd as i32, sd as i32, ng as i32,
        ).into_inst());

        // 5. mamba2_norm_gated
        instructions.push(Mamba2NormGatedInst::new(
            ng as u32,
            act.mamba2_conv_out.as_write_ptr(),
            act.mamba2_ssm_out.as_ptr(),
            act.mamba2_in_proj.as_ptr(),
            w.norm_weight.as_ptr(),
            ng as i32, group_size as i32, eps,
        ).into_inst());

        // 6. out_proj
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.out_proj);
            instructions.push(LinearProjInst::new(op, hs as u32, act.out_proj.as_write_ptr(), wp, act.mamba2_conv_out.as_ptr(), hs as i32, intermediate as i32, 0).into_inst());
        }

        // 7. Residual
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.out_proj.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
    }

    pub(super) fn compile_ffn_batched(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        bufs: &PrefillBuffers,
        n: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::model::LinearWeight;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            _ => panic!("prefill FFN only for Gdn/Attention layers"),
        };

        let all_bf16 = matches!(w_gate, LinearWeight::Bf16(_))
            && matches!(w_up, LinearWeight::Bf16(_))
            && matches!(w_down, LinearWeight::Bf16(_));
        let all_rnf4 = matches!(w_gate, LinearWeight::Packed(pw) if pw.format == WeightFormat::Rnf4G128)
            && matches!(w_up, LinearWeight::Packed(pw) if pw.format == WeightFormat::Rnf4G128)
            && matches!(w_down, LinearWeight::Packed(pw) if pw.format == WeightFormat::Rnf4G128);

        if all_bf16 {
            // Fused path: OP_FFN_GATE_UP + OP_FFN_DOWN_RES (bf16, processes all N tokens)
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP, (is * n) as u32,
                bufs.ffn_act.as_write_ptr(), bufs.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate.as_bf16_ptr() as *const u8, w_up.as_bf16_ptr() as *const u8,
                hs as i32, is as i32, eps, n as i32,
            ).no_sync().into_inst());
            instructions.push(D2dCopyInst::new(
                div_ceil((n * hs) as u32, 256),
                bufs.residual.as_write_ptr(), bufs.hidden.as_ptr(), (n * hs) as i32,
            ).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES, (hs * n) as u32,
                bufs.hidden.as_write_ptr(), bufs.residual.as_ptr(),
                w_down.as_bf16_ptr() as *const u8, bufs.ffn_act.as_ptr(),
                hs as i32, is as i32, n as i32,
            ).into_inst());
        } else if all_rnf4 {
            // Fused path: OP_FFN_GATE_UP_RNF4 + OP_FFN_DOWN_RES_RNF4 (rnf4, processes all N tokens)
            let wg_ptr = match w_gate { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let wu_ptr = match w_up   { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let wd_ptr = match w_down { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP_RNF4, (is * n) as u32,
                bufs.ffn_act.as_write_ptr(), bufs.hidden.as_ptr(), post_norm.as_ptr(),
                wg_ptr, wu_ptr,
                hs as i32, is as i32, eps, n as i32,
            ).no_sync().into_inst());
            instructions.push(D2dCopyInst::new(
                div_ceil((n * hs) as u32, 256),
                bufs.residual.as_write_ptr(), bufs.hidden.as_ptr(), (n * hs) as i32,
            ).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES_RNF4, (hs * n) as u32,
                bufs.hidden.as_write_ptr(), bufs.residual.as_ptr(),
                wd_ptr, bufs.ffn_act.as_ptr(),
                hs as i32, is as i32, n as i32,
            ).into_inst());
        } else {
            // Unfused path for quantized weights: process one token at a time.
            // Uses ffn_gate_scratch/ffn_up_scratch/ffn_down_scratch as single-token intermediates.
            for t in 0..n {
                let hidden_t = unsafe { bufs.hidden.as_write_ptr().add(t * hs) };
                let normed_t = unsafe { bufs.normed.as_write_ptr().add(t * hs) };
                let residual_t = unsafe { bufs.residual.as_write_ptr().add(t * hs) };

                // D2D_COPY: hidden[t] → residual[t]  (no_sync: RMSNorm reads hidden, not residual)
                instructions.push(D2dCopyInst::new(
                    div_ceil(hs as u32, 256), residual_t, hidden_t, hs as i32,
                ).no_sync().into_inst());

                // RMSNorm: hidden[t] → normed[t]
                instructions.push(RmsNormInst::new(
                    rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
                    normed_t, hidden_t, post_norm.as_ptr(), hs as i32, eps,
                ).into_inst());

                // Gate: normed[t] → ffn_gate_scratch  (no_sync: up reads same normed)
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                    emit_linear_proj(&mut inst, w_gate, 2);
                    inst.words[1] = bufs.ffn_gate_scratch.as_write_ptr() as u64;
                    inst.words[3] = normed_t as u64;
                    inst.words[4] = is as u64;
                    inst.words[5] = hs as u64;
                    inst.words[0] |= FLAG_NO_SYNC as u64;
                    instructions.push(inst);
                }

                // Up: normed[t] → ffn_up_scratch
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                    emit_linear_proj(&mut inst, w_up, 2);
                    inst.words[1] = bufs.ffn_up_scratch.as_write_ptr() as u64;
                    inst.words[3] = normed_t as u64;
                    inst.words[4] = is as u64;
                    inst.words[5] = hs as u64;
                    instructions.push(inst);
                }

                // SiLU(gate) * up → ffn_act[t..t+is]
                let ffn_act_t = unsafe { bufs.ffn_act.as_write_ptr().add(t * is) };
                instructions.push(SiluMulInst::new(
                    div_ceil(is as u32, 256),
                    ffn_act_t, bufs.ffn_gate_scratch.as_ptr(), bufs.ffn_up_scratch.as_ptr(), is as i32,
                ).into_inst());

                // Down: ffn_act[t] → ffn_down_scratch
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                    emit_linear_proj(&mut inst, w_down, 2);
                    inst.words[1] = bufs.ffn_down_scratch.as_write_ptr() as u64;
                    inst.words[3] = ffn_act_t as u64;
                    inst.words[4] = hs as u64;
                    inst.words[5] = is as u64;
                    instructions.push(inst);
                }

                // Residual: ffn_down_scratch + residual[t] → hidden[t]
                instructions.push(ResidualAddInst::new(
                    div_ceil(hs as u32, 256),
                    hidden_t, bufs.ffn_down_scratch.as_ptr(), residual_t, hs as i32,
                ).into_inst());
            }
        }
    }

    pub(super) fn compile_ffn(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::model::LinearWeight;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            _ => panic!("prefill FFN only for Gdn/Attention layers"),
        };

        let all_bf16 = matches!(w_gate, LinearWeight::Bf16(_))
            && matches!(w_up, LinearWeight::Bf16(_))
            && matches!(w_down, LinearWeight::Bf16(_));

        let all_rnf4 = matches!(w_gate, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128)
            && matches!(w_up, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128)
            && matches!(w_down, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128);

        if all_bf16 {
            // Fused path: OP_FFN_GATE_UP + OP_FFN_DOWN_RES (bf16 only, batch=0=single token)
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP, is as u32,
                act.ffn_act.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate.as_bf16_ptr() as *const u8, w_up.as_bf16_ptr() as *const u8,
                hs as i32, is as i32, eps, 0,
            ).no_sync().into_inst());
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES, hs as u32,
                act.hidden.as_write_ptr(), act.residual.as_ptr(), w_down.as_bf16_ptr() as *const u8, act.ffn_act.as_ptr(),
                hs as i32, is as i32, 0,
            ).into_inst());
        } else if all_rnf4 {
            let w_gate_ptr = match w_gate { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_up_ptr   = match w_up   { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_down_ptr = match w_down { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP_RNF4, is as u32,
                act.ffn_act.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate_ptr, w_up_ptr,
                hs as i32, is as i32, eps, 0,
            ).no_sync().into_inst());
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES_RNF4, hs as u32,
                act.hidden.as_write_ptr(), act.residual.as_ptr(), w_down_ptr, act.ffn_act.as_ptr(),
                hs as i32, is as i32, 0,
            ).into_inst());
        } else {
            // Unfused path for quantized weights (decode n=1 only)
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).no_sync().into_inst());
            instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(), hs as i32, eps).into_inst());
            {
                let (op, wp) = linear_proj_opcode_ptr(w_gate);
                instructions.push(LinearProjInst::new(op, is as u32, act.ffn_gate.as_write_ptr(), wp, act.normed.as_ptr(), is as i32, hs as i32, 0).no_sync().into_inst());
            }
            {
                let (op, wp) = linear_proj_opcode_ptr(w_up);
                instructions.push(LinearProjInst::new(op, is as u32, act.ffn_up.as_write_ptr(), wp, act.normed.as_ptr(), is as i32, hs as i32, 0).into_inst());
            }
            instructions.push(SiluMulInst::new(div_ceil(is as u32, 256), act.ffn_act.as_write_ptr(), act.ffn_gate.as_ptr(), act.ffn_up.as_ptr(), is as i32).into_inst());
            {
                let (op, wp) = linear_proj_opcode_ptr(w_down);
                instructions.push(LinearProjInst::new(op, hs as u32, act.ffn_down.as_write_ptr(), wp, act.ffn_act.as_ptr(), hs as i32, is as i32, 0).into_inst());
            }
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.ffn_down.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
        }
    }
}
