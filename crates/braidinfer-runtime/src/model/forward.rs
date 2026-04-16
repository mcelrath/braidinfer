//! Layer forward passes: GDN, attention, Mamba2, FFN (dense + MoE).
//! These are `impl Model` methods extracted for maintainability.

use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;

use super::Model;
use crate::config::*;
use crate::weights::*;


impl Model {
    pub(crate) fn gdn_forward(&mut self, layer_idx: usize, gdn_idx: usize) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nh = cfg.linear_num_heads as u32;
        let nvh = cfg.linear_num_value_heads as u32;
        let kd = cfg.linear_key_head_dim as u32;
        let vd = cfg.linear_value_head_dim as u32;
        let ck = cfg.linear_conv_kernel_dim as u32;
        let eps = cfg.rms_norm_eps;

        let weights = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => panic!("expected GDN layer at {layer_idx}"),
        };

        // 1. RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &weights.input_norm,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;

        // 2. Project QKV [6144]
        weights.w_qkv.forward(
            &self.kernels.linear_proj,
            &mut self.activations.qkv,
            &self.activations.normed,
            nh * kd * 2 + nvh * vd,
            hs,
            &self.stream,
        )?;

        // 3. Project a [nvh], b [nvh], z [nvh*vd]
        weights.w_a.forward(
            &self.kernels.linear_proj,
            &mut self.activations.a_proj,
            &self.activations.normed,
            nvh,
            hs,
            &self.stream,
        )?;
        weights.w_b.forward(
            &self.kernels.linear_proj,
            &mut self.activations.b_proj,
            &self.activations.normed,
            nvh,
            hs,
            &self.stream,
        )?;
        weights.w_z.forward(
            &self.kernels.linear_proj,
            &mut self.activations.z_proj,
            &self.activations.normed,
            nvh * vd,
            hs,
            &self.stream,
        )?;

        // 4. Causal conv1d on Q, K, V — using raw-pointer variant to avoid staging copies.
        //
        // gdn_conv_states[gdn_idx] is packed [q_state | k_state | v_state]:
        //   q_state: [nh*kd, ck-1], k_state: [nh*kd, ck-1], v_state: [nvh*vd, ck-1]
        // Input for each is the corresponding slice of qkv: [q | k | v]
        // Output goes directly to q_gdn / k_gdn / v_gdn.
        //
        // SAFETY: raw pointers into DeviceBuffers that remain live and unmodified for the
        // duration of these async kernel launches (all on the same stream).
        let conv_q_len = nh as usize * kd as usize;
        let conv_k_len = nh as usize * kd as usize;
        let conv_v_len = nvh as usize * vd as usize;
        let ck_usize = ck as usize;
        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);

        let (conv_w_q_ptr, conv_w_k_ptr, conv_w_v_ptr) = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => (
                w.conv1d_weight_q.as_ptr(),
                w.conv1d_weight_k.as_ptr(),
                w.conv1d_weight_v.as_ptr(),
            ),
            _ => unreachable!(),
        };
        let state_base = self.gdn_conv_states[gdn_idx].as_mut_ptr();
        let qkv_base = self.activations.qkv.as_ptr();
        unsafe {
            self.kernels.causal_conv1d.forward_ptr(
                state_base,
                qkv_base,
                conv_w_q_ptr,
                self.activations.q_gdn.as_mut_ptr(),
                conv_q_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward_ptr(
                state_base.add(conv_state_q_len),
                qkv_base.add(conv_q_len),
                conv_w_k_ptr,
                self.activations.k_gdn.as_mut_ptr(),
                conv_k_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward_ptr(
                state_base.add(conv_state_q_len + conv_state_k_len),
                qkv_base.add(conv_q_len + conv_k_len),
                conv_w_v_ptr,
                self.activations.v_gdn.as_mut_ptr(),
                conv_v_len as u32,
                ck,
                &self.stream,
            )?;
        }

        // 5. Compute GDN gate
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &weights_gdn.a_log,
            &self.activations.a_proj,
            &weights_gdn.dt_bias,
            nvh,
            &self.stream,
        )?;

        // 6. GDN recurrent step v2 (nvh heads, GQA group = nvh/nh)
        let gqa_group = nvh / nh;
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[gdn_idx].recurrent,
            &mut self.activations.recurrent_out,
            nvh,
            kd,
            vd,
            gqa_group,
            &self.stream,
        )?;

        // 7. RMSNorm gated (recurrent_out with z gate)
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &weights_gdn.output_norm,
            nvh, // value heads, not key heads
            vd,
            eps,
            &self.stream,
        )?;

        // 8. Output projection [1024, 2048]
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        weights_gdn.w_out.forward(
            &self.kernels.linear_proj,
            &mut self.activations.out_proj,
            &self.activations.normed_gated,
            hs,
            nvh * vd, // value heads, not key heads
            &self.stream,
        )?;

        // 9. Residual add: hidden = out_proj + hidden
        // Copy hidden to residual first, then add
        unsafe {
            d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        // 10. FFN — extract raw pointers to avoid borrow conflict with &mut self
        Ok(())
    }

    pub(crate) fn ffn_forward(
        &mut self,
        post_norm: &DeviceBuffer<u16>,
        w_gate: &DeviceBuffer<u16>,
        w_up: &DeviceBuffer<u16>,
        w_down: &DeviceBuffer<u16>,
    ) -> HipResult<()> {
        let hs = self.config.hidden_size as u32;
        let is = self.config.intermediate_size as u32;
        let eps = self.config.rms_norm_eps;

        self.kernels.ffn_fused.forward_gate_up(
            &mut self.activations.ffn_act,
            &self.activations.hidden,
            post_norm,
            w_gate,
            w_up,
            hs,
            is,
            eps,
            &self.stream,
        )?;

        // Save residual
        unsafe {
            d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.ffn_fused.forward_down_residual(
            &mut self.activations.hidden,
            &self.activations.residual,
            w_down,
            &self.activations.ffn_act,
            hs,
            is,
            &self.stream,
        )
    }


    /// Mamba2 SSM layer forward pass (Nemotron-H 'M' layers).
    /// Steps: norm → in_proj → split → conv1d → split → ssm_update → norm_gated → out_proj → residual
    pub(crate) fn mamba2_forward(
        &mut self,
        layer_idx: usize,
        mamba2_idx: usize,
    ) -> Result<(), ModelError> {
        let w = match &self.layers[layer_idx] {
            LayerWeights::Mamba2(w) => w as *const Mamba2LayerWeights,
            _ => panic!("mamba2_forward called on non-Mamba2 layer"),
        };
        let (nh, hd, sd, _ck, ng, cd) = match &self.config.recurrent_kind {
            RecurrentLayerKind::Mamba2 {
                num_heads,
                head_dim,
                state_dim,
                conv_kernel,
                n_groups,
                conv_dim,
                ..
            } => (
                *num_heads,
                *head_dim,
                *state_dim,
                *conv_kernel,
                *n_groups,
                *conv_dim,
            ),
            _ => panic!("mamba2_forward but no Mamba2 config"),
        };
        let hs = self.config.hidden_size as u32;
        let intermediate = (nh * hd) as u32;
        let in_proj_size = (nh * hd + cd + nh) as u32;
        let eps = self.config.rms_norm_eps;

        // 1. RMSNorm
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &(*w).input_norm,
                1,
                hs,
                eps,
                self.config.rms_norm_one_plus_w,
                &self.stream,
            )?;
        }

        // Debug: per-step tracing for layer 0
        let dbg = self.debug_nan && layer_idx == 0;
        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.normed.len();
            let mut buf = vec![0.0f32; n];
            self.activations.normed.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "  M2.L0 after norm: max_abs={max_abs:.4e}, first5={:.4?}",
                &buf[..5]
            );
        }

        // 2. in_proj: normed → [gate(intermediate), xBC(conv_dim), dt(num_heads)]
        unsafe {
            (*w).in_proj.forward(
                &self.kernels.linear_proj,
                &mut self.activations.mamba2_in_proj,
                &self.activations.normed,
                in_proj_size,
                hs,
                &self.stream,
            )?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_in_proj.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_in_proj.copy_to_host(&mut buf)?;
            let gate_max = buf[..nh * hd]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let xbc_max = buf[nh * hd..nh * hd + cd]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let dt_max = buf[nh * hd + cd..]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "  M2.L0 in_proj: gate_max={gate_max:.4e}, xBC_max={xbc_max:.4e}, dt_max={dt_max:.4e}"
            );
        }

        // 3. Conv1d update on xBC with bias + silu activation
        // Input is mamba2_in_proj[intermediate..intermediate+cd], output to mamba2_conv_out
        {
            let state = &mut self.mamba2_states[mamba2_idx];
            unsafe {
                self.kernels.causal_conv1d.forward_with_bias_ptr(
                    state.conv.as_mut_ptr(),
                    self.activations.mamba2_in_proj.as_ptr().add(nh * hd),
                    (*w).conv1d_weight.as_ptr(),
                    (*w).conv1d_bias.as_ptr(),
                    self.activations.mamba2_conv_out.as_mut_ptr(),
                    cd as u32,
                    _ck as u32,
                    &self.stream,
                )?;
            }
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_conv_out.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_conv_out.copy_to_host(&mut buf)?;
            let x_max = buf[..nh * hd]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let b_max = buf[nh * hd..nh * hd + ng * sd]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let c_max = buf[nh * hd + ng * sd..]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            eprintln!("  M2.L0 conv1d: x_max={x_max:.4e}, B_max={b_max:.4e}, C_max={c_max:.4e}");
        }

        // 4. Split conv_out → x[intermediate], B[ng*sd], C[ng*sd]
        // x = conv_out[0..intermediate], B = conv_out[intermediate..intermediate+ng*sd], C = conv_out[intermediate+ng*sd..]
        // dt = mamba2_in_proj[intermediate+cd..intermediate+cd+nh]

        // 5. selective_state_update
        let state = &mut self.mamba2_states[mamba2_idx];
        unsafe {
            self.kernels.ssm_update.forward_ptr(
                state.ssm.as_mut_ptr(),
                self.activations.mamba2_conv_out.as_ptr(),
                self.activations.mamba2_in_proj.as_ptr().add(nh * hd + cd),
                (*w).dt_bias.as_ptr(),
                (*w).a_log.as_ptr(),
                self.activations.mamba2_conv_out.as_ptr().add(nh * hd),
                self.activations.mamba2_conv_out.as_ptr().add(nh * hd + ng * sd),
                (*w).d.as_ptr(),
                self.activations.mamba2_ssm_out.as_mut_ptr(),
                nh as u32,
                hd as u32,
                sd as u32,
                ng as u32,
                &self.stream,
            )?;
        }

        if dbg {
            self.stream.synchronize()?;
            let state = &self.mamba2_states[mamba2_idx];
            let ssm_n = state.ssm.len();
            let mut ssm_buf = vec![0.0f32; ssm_n];
            state.ssm.copy_to_host(&mut ssm_buf)?;
            let ssm_max = ssm_buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let ssm_mean = ssm_buf.iter().map(|x| x.abs()).sum::<f32>() / ssm_n as f32;
            eprintln!("  M2.L0 ssm_state: n={ssm_n} max_abs={ssm_max:.4e} mean_abs={ssm_mean:.4e}");

            let n = self.activations.mamba2_ssm_out.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_ssm_out.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "  M2.L0 ssm_out: max_abs={max_abs:.4e}, first5={:.4?}",
                &buf[..5]
            );
        }

        // 6. rmsnorm_gated: normed_out = rmsnorm(ssm_out * silu(gate)) * weight
        // Mamba2/Nemotron uses norm_before_gate=False: norm the gated product.
        // Per-group norm (group_size = intermediate / n_groups).
        let group_size = (nh * hd / ng) as u32;
        for g in 0..ng {
            let off = g * group_size as usize;
            unsafe {
                self.kernels.rmsnorm_gated.forward_post_ptr(
                    self.activations.mamba2_conv_out.as_mut_ptr().add(off),
                    self.activations.mamba2_ssm_out.as_ptr().add(off),
                    self.activations.mamba2_in_proj.as_ptr().add(off),
                    (*w).norm_weight.as_ptr().add(off),
                    1,
                    group_size,
                    eps,
                    &self.stream,
                )?;
            }
        }

        // 7. out_proj: normed_out → output[hidden_size]
        unsafe {
            (*w).out_proj.forward(
                &self.kernels.linear_proj,
                &mut self.activations.out_proj,
                &self.activations.mamba2_conv_out,
                hs,
                intermediate,
                &self.stream,
            )?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.out_proj.len();
            let mut buf = vec![0.0f32; n.min(self.config.hidden_size)];
            self.activations.out_proj.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "  M2.L0 out_proj: max_abs={max_abs:.4e}, first5={:.4?}",
                &buf[..5]
            );
        }

        // 8. Residual add
        unsafe {
            crate::model::d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                self.config.hidden_size,
                &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        Ok(())
    }
}
