//! Layer forward passes: GDN (used by decode_step_traced and decode_step_traced_v2).
//! These are `impl Model` methods extracted for maintainability.
//! ffn_forward and mamba2_forward deleted: no callers outside trace paths; megakernel handles both.

use braidinfer_hip::HipResult;

use super::Model;
use crate::weights::*;
use crate::gpu_utils::d2d_copy_f32;


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

        // 10. FFN — handled by megakernel in the persistent decode path.
        Ok(())
    }
}
