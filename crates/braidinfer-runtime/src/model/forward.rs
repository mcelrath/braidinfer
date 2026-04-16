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

        // 4. Causal conv1d: split qkv into q/k/v
        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &self.activations.qkv,
                0,
                nh as usize * kd as usize,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &self.activations.qkv,
                nh as usize * kd as usize,
                nh as usize * kd as usize,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &self.activations.qkv,
                nh as usize * kd as usize * 2,
                nvh as usize * vd as usize,
                &self.stream,
            )?;
        }

        let conv_q_out_len = nh as usize * kd as usize;
        let conv_k_out_len = nh as usize * kd as usize;
        let conv_v_out_len = nvh as usize * vd as usize;
        let ck_usize = ck as usize;

        // Split conv state into q/k/v sub-states
        // gdn_conv_states[gdn_idx] is [6144, ck-1] = [6144 * (ck-1)].
        // Split into 3 sub-states: q=[2048,ck-1], k=[2048,ck-1], v=[2048,ck-1].
        let conv_state_q_len = conv_q_out_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_out_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_out_len * (ck_usize - 1);

        unsafe {
            d2d_copy_f32(
                &mut self.activations.gdn_cs_q,
                0,
                &self.gdn_conv_states[gdn_idx],
                0,
                conv_state_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.gdn_cs_k,
                0,
                &self.gdn_conv_states[gdn_idx],
                conv_state_q_len,
                conv_state_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.gdn_cs_v,
                0,
                &self.gdn_conv_states[gdn_idx],
                conv_state_q_len + conv_state_k_len,
                conv_state_v_len,
                &self.stream,
            )?;
        }

        // Run 3 conv1d operations using pre-split weight buffers from the layer
        // SAFETY: Raw pointers break the borrow on self.layers so we can mutably access
        // self.activations. The pointers remain valid because layers[layer_idx] is not
        // modified or moved during this function call.
        let (conv_w_q_ptr, conv_w_k_ptr, conv_w_v_ptr) = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => (
                &w.conv1d_weight_q as *const DeviceBuffer<u16>,
                &w.conv1d_weight_k as *const DeviceBuffer<u16>,
                &w.conv1d_weight_v as *const DeviceBuffer<u16>,
            ),
            _ => unreachable!(),
        };
        unsafe {
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_q,
                &self.activations.q_gdn,
                &*conv_w_q_ptr,
                &mut self.activations.gdn_conv_out_q,
                conv_q_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_k,
                &self.activations.k_gdn,
                &*conv_w_k_ptr,
                &mut self.activations.gdn_conv_out_k,
                conv_k_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_v,
                &self.activations.v_gdn,
                &*conv_w_v_ptr,
                &mut self.activations.gdn_conv_out_v,
                conv_v_out_len as u32,
                ck,
                &self.stream,
            )?;
        }

        // Write back updated conv states
        unsafe {
            d2d_copy_f32(
                &mut self.gdn_conv_states[gdn_idx],
                0,
                &self.activations.gdn_cs_q,
                0,
                conv_state_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.gdn_conv_states[gdn_idx],
                conv_state_q_len,
                &self.activations.gdn_cs_k,
                0,
                conv_state_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.gdn_conv_states[gdn_idx],
                conv_state_q_len + conv_state_k_len,
                &self.activations.gdn_cs_v,
                0,
                conv_state_v_len,
                &self.stream,
            )?;
        }

        // conv_out_q/k/v now hold the post-conv Q,K,V (with SiLU applied inside the kernel)
        // Copy them back to q_gdn, k_gdn, v_gdn
        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &self.activations.gdn_conv_out_q,
                0,
                conv_q_out_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &self.activations.gdn_conv_out_k,
                0,
                conv_k_out_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &self.activations.gdn_conv_out_v,
                0,
                conv_v_out_len,
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
            let func = self
                .kernels
                .causal_conv1d
                .module
                .get_function("causal_conv1d_update_bias_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.conv.as_mut_ptr().cast();
            let mut in_ptr: *const std::ffi::c_void =
                unsafe { self.activations.mamba2_in_proj.as_ptr().add(nh * hd).cast() };
            let mut w_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_weight.as_ptr().cast() };
            let mut bias_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_bias.as_ptr().cast() };
            let mut out_ptr: *mut std::ffi::c_void =
                self.activations.mamba2_conv_out.as_mut_ptr().cast();
            let mut i_cd = cd as i32;
            let mut i_ck = _ck as i32;
            let mut args: [*mut std::ffi::c_void; 7] = [
                std::ptr::addr_of_mut!(state_ptr).cast(),
                std::ptr::addr_of_mut!(in_ptr).cast(),
                std::ptr::addr_of_mut!(w_ptr).cast(),
                std::ptr::addr_of_mut!(bias_ptr).cast(),
                std::ptr::addr_of_mut!(out_ptr).cast(),
                std::ptr::addr_of_mut!(i_cd).cast(),
                std::ptr::addr_of_mut!(i_ck).cast(),
            ];
            let block_size = 256u32;
            let grid_size = (cd as u32 + block_size - 1) / block_size;
            func.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                0,
                &self.stream,
                &mut args,
            )?;
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
            let x_ptr = self.activations.mamba2_conv_out.as_ptr();
            let b_ptr = self.activations.mamba2_conv_out.as_ptr().add(nh * hd);
            let c_ptr = self
                .activations
                .mamba2_conv_out
                .as_ptr()
                .add(nh * hd + ng * sd);
            let dt_ptr = self.activations.mamba2_in_proj.as_ptr().add(nh * hd + cd);

            // Create temporary DeviceBuffer wrappers pointing to sub-regions
            // We need to call the kernel with raw pointers
            let func = self
                .kernels
                .ssm_update
                .module
                .get_function("selective_state_update_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.ssm.as_mut_ptr().cast();
            let mut x_p: *const std::ffi::c_void = x_ptr.cast();
            let mut dt_p: *const std::ffi::c_void = dt_ptr.cast();
            let mut dt_bias_p: *const std::ffi::c_void = (*w).dt_bias.as_ptr().cast();
            let mut a_log_p: *const std::ffi::c_void = (*w).a_log.as_ptr().cast();
            let mut b_p: *const std::ffi::c_void = b_ptr.cast();
            let mut c_p: *const std::ffi::c_void = c_ptr.cast();
            let mut d_p: *const std::ffi::c_void = (*w).d.as_ptr().cast();
            let mut out_p: *mut std::ffi::c_void =
                self.activations.mamba2_ssm_out.as_mut_ptr().cast();
            let mut i_nh = nh as i32;
            let mut i_hd = hd as i32;
            let mut i_sd = sd as i32;
            let mut i_ng = ng as i32;

            let mut args: [*mut std::ffi::c_void; 13] = [
                std::ptr::addr_of_mut!(state_ptr).cast(),
                std::ptr::addr_of_mut!(x_p).cast(),
                std::ptr::addr_of_mut!(dt_p).cast(),
                std::ptr::addr_of_mut!(dt_bias_p).cast(),
                std::ptr::addr_of_mut!(a_log_p).cast(),
                std::ptr::addr_of_mut!(b_p).cast(),
                std::ptr::addr_of_mut!(c_p).cast(),
                std::ptr::addr_of_mut!(d_p).cast(),
                std::ptr::addr_of_mut!(out_p).cast(),
                std::ptr::addr_of_mut!(i_nh).cast(),
                std::ptr::addr_of_mut!(i_hd).cast(),
                std::ptr::addr_of_mut!(i_sd).cast(),
                std::ptr::addr_of_mut!(i_ng).cast(),
            ];

            func.launch((nh as u32, 1, 1), (256, 1, 1), 0, &self.stream, &mut args)?;
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
        {
            let func = self
                .kernels
                .rmsnorm_gated
                .module
                .get_function("rmsnorm_gated_post_f32")?;
            for g in 0..ng {
                let off = g * group_size as usize;
                let mut out_p: *mut std::ffi::c_void = unsafe {
                    self.activations
                        .mamba2_conv_out
                        .as_mut_ptr()
                        .add(off)
                        .cast()
                };
                let mut x_p: *const std::ffi::c_void =
                    unsafe { self.activations.mamba2_ssm_out.as_ptr().add(off).cast() };
                let mut z_p: *const std::ffi::c_void =
                    unsafe { self.activations.mamba2_in_proj.as_ptr().add(off).cast() };
                let mut w_p: *const std::ffi::c_void =
                    unsafe { (*w).norm_weight.as_ptr().add(off).cast() };
                let mut i_nh = 1i32;
                let mut i_vd = group_size as i32;
                let mut f_eps = eps;
                let mut args: [*mut std::ffi::c_void; 7] = [
                    std::ptr::addr_of_mut!(out_p).cast(),
                    std::ptr::addr_of_mut!(x_p).cast(),
                    std::ptr::addr_of_mut!(z_p).cast(),
                    std::ptr::addr_of_mut!(w_p).cast(),
                    std::ptr::addr_of_mut!(i_nh).cast(),
                    std::ptr::addr_of_mut!(i_vd).cast(),
                    std::ptr::addr_of_mut!(f_eps).cast(),
                ];
                func.launch((1, 1, 1), (256, 1, 1), 256 * 4, &self.stream, &mut args)?;
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
