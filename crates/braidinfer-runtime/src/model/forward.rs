//! Layer forward passes: GDN, attention, Mamba2, FFN (dense + MoE).
//! These are `impl Model` methods extracted for maintainability.

use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::HipResult;

use crate::config::*;
use crate::weights::*;
use super::Model;

impl Model {
    pub(crate) fn gdn_forward(
        &mut self,
        layer_idx: usize,
        gdn_idx: usize,
    ) -> HipResult<()> {
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
        weights.w_qkv.forward(&self.kernels.linear_proj,
            &mut self.activations.qkv, &self.activations.normed,
            nh * kd * 2 + nvh * vd, hs, &self.stream)?;

        // 3. Project a [nvh], b [nvh], z [nvh*vd]
        weights.w_a.forward(&self.kernels.linear_proj,
            &mut self.activations.a_proj, &self.activations.normed, nvh, hs, &self.stream)?;
        weights.w_b.forward(&self.kernels.linear_proj,
            &mut self.activations.b_proj, &self.activations.normed, nvh, hs, &self.stream)?;
        weights.w_z.forward(&self.kernels.linear_proj,
            &mut self.activations.z_proj, &self.activations.normed, nvh * vd, hs, &self.stream)?;

        // 4. Causal conv1d: split qkv into q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, nh as usize * kd as usize, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, nh as usize * kd as usize * 2, nvh as usize * vd as usize, &self.stream)?;
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
            d2d_copy_f32(&mut self.activations.gdn_cs_q, 0, &self.gdn_conv_states[gdn_idx], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_k, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_v, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
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
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], 0, &self.activations.gdn_cs_q, 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len, &self.activations.gdn_cs_k, 0, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, &self.activations.gdn_cs_v, 0, conv_state_v_len, &self.stream)?;
        }

        // conv_out_q/k/v now hold the post-conv Q,K,V (with SiLU applied inside the kernel)
        // Copy them back to q_gdn, k_gdn, v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.gdn_conv_out_q, 0, conv_q_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.gdn_conv_out_k, 0, conv_k_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.gdn_conv_out_v, 0, conv_v_out_len, &self.stream)?;
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
            nvh,  // value heads, not key heads
            vd,
            eps,
            &self.stream,
        )?;

        // 8. Output projection [1024, 2048]
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        weights_gdn.w_out.forward(&self.kernels.linear_proj,
            &mut self.activations.out_proj, &self.activations.normed_gated,
            hs, nvh * vd,  // value heads, not key heads
            &self.stream,
        )?;

        // 9. Residual add: hidden = out_proj + hidden
        // Copy hidden to residual first, then add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
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
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
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

    /// MoE FFN forward: route to top-k experts, run expert FFNs, combine.
    /// Uses individual kernel launches (no megakernel).
    pub(crate) fn moe_ffn_forward(&mut self, layer_idx: usize) -> HipResult<()> {
        let moe = self.moe_weights[layer_idx].as_ref()
            .expect("moe_ffn_forward called on non-MoE layer");
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let ne = moe.num_experts;
        let eps = self.config.rms_norm_eps;

        // SAFETY: Raw pointer breaks borrow on self.layers to allow mutable access to
        // self.activations. Pointer valid for duration of this function (layers not modified).
        let norm_weight = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::MoeFfn(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("no norm weight for this layer type in MoE FFN"),
        };

        // 1. RMSNorm(hidden) → normed
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*norm_weight,
                1, hs as u32, eps, self.config.rms_norm_one_plus_w, &self.stream,
            )?;
        }

        // Save residual
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
        }

        // 2. Gate projection: normed → scores[num_experts]
        self.kernels.linear_proj.forward(
            &mut self.activations.moe_scores,
            &moe.gate,
            &self.activations.normed,
            ne as u32, hs as u32, &self.stream,
        )?;

        let (k, gate_type) = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { num_active, gate_type, .. } => (*num_active, gate_type.clone()),
            _ => unreachable!(),
        };

        // 3. GPU-side top-k selection + weight computation (replaces CPU round-trip)
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        self.kernels.moe_gate.forward(
            &self.activations.moe_scores,
            self.activations.moe_expert_ids.as_mut_ptr(),
            self.activations.moe_expert_weights.as_mut_ptr(),
            bias_ptr,
            ne as u32, k as u32, gate_mode, rsf,
            &self.stream,
        )?;

        // Multi-GPU path: dispatch routed experts across GPUs (non-megakernel path)
        let used_multi_gpu = if self.multi_gpu.is_some() && !self.distributed_moe.is_empty() {
            if let Some(ref dist_moe) = self.distributed_moe[layer_idx] {
                // D2D copy normed → normed_stage (MappedHostBuffer) on GPU 0 stream.
                // normed_stage.host_ptr() is then CPU-visible without a sync D2H copy.
                self.stream.synchronize()?;
                braidinfer_hip::memory::memcpy_d2d(
                    self.activations.normed_stage.as_write_ptr() as *mut u8,
                    self.activations.normed.as_ptr() as *const u8,
                    hs * 4,
                )?;
                let normed_host: &[f32] = unsafe {
                    std::slice::from_raw_parts(self.activations.normed_stage.host_ptr(), hs)
                };
                let mgpu = self.multi_gpu.as_mut().unwrap();
                crate::moe_dispatch::dispatch_moe_layer_sync(
                    mgpu,
                    &self.worker_kernels,
                    dist_moe,
                    normed_host,
                    &mut self.activations.ffn_down_stage,
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_expert_weights,
                    k, hs, eis,
                    &self.stream,
                )?;
                // Copy ffn_down_stage (host-mapped) → ffn_down (GPU 0 VRAM)
                braidinfer_hip::memory::memcpy_h2d(
                    self.activations.ffn_down.as_write_ptr() as *mut u8,
                    unsafe { std::slice::from_raw_parts(self.activations.ffn_down_stage.host_ptr() as *const u8, hs * 4) },
                    hs * 4,
                )?;
                true
            } else { false }
        } else { false };

        // Trace: MoE routing (after gate, before expert FFN)
        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut normed_buf = vec![0.0f32; hs];
            self.activations.normed.copy_to_host(&mut normed_buf)?;
            self.trace.as_mut().unwrap().write_checkpoint(
                &format!("L{layer_idx}.moe_normed"), &normed_buf);

            let ids = unsafe { std::slice::from_raw_parts(self.activations.moe_expert_ids.host_ptr(), k) };
            let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
            self.trace.as_mut().unwrap().write_checkpoint(
                &format!("L{layer_idx}.moe_expert_ids"), &ids_f32);

            let weights = unsafe { std::slice::from_raw_parts(self.activations.moe_expert_weights.host_ptr(), k) };
            self.trace.as_mut().unwrap().write_checkpoint(
                &format!("L{layer_idx}.moe_expert_weights"), weights);
        }

        if !used_multi_gpu {
        // Single-GPU path: read back expert_ids + weights via host pointer (no hipMemcpy)
        self.stream.synchronize()?;
        let expert_ids: Vec<i32> = unsafe {
            std::slice::from_raw_parts(self.activations.moe_expert_ids.host_ptr() as *const i32, k).to_vec()
        };
        let expert_weights: Vec<f32> = unsafe {
            std::slice::from_raw_parts(self.activations.moe_expert_weights.host_ptr() as *const f32, k).to_vec()
        };

        // 4. Zero accumulation buffer on GPU
        unsafe {
            let rc = braidinfer_hip::ffi::hipMemsetAsync(
                self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
                0, hs * 4, self.stream.raw(),
            );
            if rc != 0 {
                return Err(braidinfer_hip::HipError(rc).into());
            }
        }

        // Debug MoE routing
        if self.debug_nan && layer_idx <= 3 {
            eprintln!("  MoE L{layer_idx}: topk={:?} weights={:?}",
                &expert_ids, &expert_weights);
        }

        // 5. For each selected expert: run FFN and GPU-accumulate
        for j in 0..k {
            let expert_id = expert_ids[j] as usize;
            let w = expert_weights[j];

            let down_offset = moe.expert_down.row_byte_offset_dim(expert_id * hs, eis);

            if moe.has_gate_proj {
                // SwiGLU: gate_proj → silu → * up_proj
                let gate_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * 2 * eis, hs);
                let up_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * 2 * eis + eis, hs);

                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_gate.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, gate_offset, &self.stream,
                )?;
                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_up.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, up_offset, &self.stream,
                )?;
                self.kernels.silu_mul.forward(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_gate,
                    &self.activations.moe_expert_up,
                    eis as u32, &self.stream,
                )?;
            } else {
                // relu²: up_proj → relu² (no gate_proj)
                let up_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * eis, hs);
                moe.expert_gate_up.forward_sub(
                    &self.kernels.linear_proj,
                    self.activations.moe_expert_up.as_mut_ptr(),
                    self.activations.normed.as_ptr(),
                    eis as u32, hs as u32, up_offset, &self.stream,
                )?;
                self.kernels.silu_mul.relu_squared(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_up,
                    eis as u32, &self.stream,
                )?;
            }

            // Debug: check intermediate values
            if self.debug_nan && layer_idx <= 1 && j == 0 {
                self.stream.synchronize()?;
                let up_len = self.activations.moe_expert_up.len();
                let act_len = self.activations.moe_expert_act.len();
                let mut up_buf = vec![0.0f32; up_len];
                let mut act_buf = vec![0.0f32; act_len];
                self.activations.moe_expert_up.copy_to_host(&mut up_buf)?;
                self.activations.moe_expert_act.copy_to_host(&mut act_buf)?;
                let up_max = up_buf[..eis].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let act_max = act_buf[..eis].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("  Expert {expert_id}: up_max={up_max:.4}, act_max(relu²)={act_max:.4}, w={w:.6}");
            }

            // Down projection (pre-allocated buffer)
            moe.expert_down.forward_sub(
                &self.kernels.linear_proj,
                self.activations.moe_expert_out.as_mut_ptr(),
                self.activations.moe_expert_act.as_ptr(),
                hs as u32, eis as u32, down_offset, &self.stream,
            )?;

            // GPU-side weighted accumulate: ffn_down += w * expert_out
            self.kernels.residual_add.weighted_accumulate(
                &mut self.activations.ffn_down,
                &self.activations.moe_expert_out,
                w,
                hs as u32,
                &self.stream,
            )?;
        }

        } // end if !used_multi_gpu

        // 6. Shared expert (always-on, added to output — runs on GPU 0 for both paths)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &self.config.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } =>
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size },
                _ => eis,
            };
            // Shared expert receives the same normed input as routed experts.
            // HF NemotronHMOE: residuals = hidden_states (already normed by block)
            se.up_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.moe_expert_up, &self.activations.normed,
                se_is as u32, hs as u32, &self.stream)?;

            if moe.has_gate_proj {
                // SwiGLU shared expert
                se.gate_proj.forward(&self.kernels.linear_proj,
                    &mut self.activations.moe_expert_gate, &self.activations.normed,
                    se_is as u32, hs as u32, &self.stream)?;
                self.kernels.silu_mul.forward(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_gate,
                    &self.activations.moe_expert_up,
                    se_is as u32, &self.stream)?;
            } else {
                // relu² shared expert
                self.kernels.silu_mul.relu_squared(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_up,
                    se_is as u32, &self.stream)?;
            }

            se.down_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.moe_expert_out, &self.activations.moe_expert_act,
                hs as u32, se_is as u32, &self.stream)?;

            // Apply shared expert gate: sigmoid(gate @ input) * shared_output
            let se_weight = if let Some(ref gate_buf) = moe.shared_expert_gate {
                // Compute sigmoid(gate @ normed_input) on CPU
                self.stream.synchronize()?;
                let mut normed = vec![0.0f32; hs];
                self.activations.normed.copy_to_host(&mut normed)?;
                let mut gate_w = vec![0u16; hs];
                gate_buf.copy_to_host(&mut gate_w)?;
                let dot: f32 = normed.iter().zip(gate_w.iter())
                    .map(|(&x, &w)| x * f32::from_bits((w as u32) << 16))
                    .sum();
                1.0 / (1.0 + (-dot).exp()) // sigmoid
            } else {
                1.0
            };
            self.kernels.residual_add.weighted_accumulate(
                &mut self.activations.ffn_down,
                &self.activations.moe_expert_out,
                se_weight,
                hs as u32, &self.stream)?;
        }

        // Debug: check ffn_down magnitude
        if self.debug_nan && layer_idx <= 3 {
            self.stream.synchronize()?;
            let ffn_len = self.activations.ffn_down.len();
            let mut buf = vec![0.0f32; ffn_len];
            self.activations.ffn_down.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  MoE L{layer_idx} ffn_down(len={ffn_len}) max_abs={max_abs:.4}");
        }

        // Trace: ffn_down after expert accumulation + shared expert (before residual)
        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; hs];
            self.activations.ffn_down.copy_to_host(&mut buf)?;
            self.trace.as_mut().unwrap().write_checkpoint(
                &format!("L{layer_idx}.moe_ffn_down"), &buf);
        }

        // 7. Residual add: hidden = residual + ffn_down
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.residual,
            &self.activations.ffn_down,
            hs as u32, &self.stream,
        )?;

        Ok(())
    }

    /// Mamba2 SSM layer forward pass (Nemotron-H 'M' layers).
    /// Steps: norm → in_proj → split → conv1d → split → ssm_update → norm_gated → out_proj → residual
    pub(crate) fn mamba2_forward(&mut self, layer_idx: usize, mamba2_idx: usize) -> Result<(), ModelError> {
        let w = match &self.layers[layer_idx] {
            LayerWeights::Mamba2(w) => w as *const Mamba2LayerWeights,
            _ => panic!("mamba2_forward called on non-Mamba2 layer"),
        };
        let (nh, hd, sd, _ck, ng, cd) = match &self.config.recurrent_kind {
            RecurrentLayerKind::Mamba2 { num_heads, head_dim, state_dim, conv_kernel, n_groups, conv_dim, .. } =>
                (*num_heads, *head_dim, *state_dim, *conv_kernel, *n_groups, *conv_dim),
            _ => panic!("mamba2_forward but no Mamba2 config"),
        };
        let hs = self.config.hidden_size as u32;
        let intermediate = (nh * hd) as u32;
        let in_proj_size = (nh * hd + cd + nh) as u32;
        let eps = self.config.rms_norm_eps;

        // 1. RMSNorm
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed, &self.activations.hidden,
                &(*w).input_norm, 1, hs, eps,
                self.config.rms_norm_one_plus_w, &self.stream,
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
            eprintln!("  M2.L0 after norm: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 2. in_proj: normed → [gate(intermediate), xBC(conv_dim), dt(num_heads)]
        unsafe {
            (*w).in_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.mamba2_in_proj, &self.activations.normed,
                in_proj_size, hs, &self.stream)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_in_proj.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_in_proj.copy_to_host(&mut buf)?;
            let gate_max = buf[..nh*hd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let xbc_max = buf[nh*hd..nh*hd+cd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let dt_max = buf[nh*hd+cd..].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 in_proj: gate_max={gate_max:.4e}, xBC_max={xbc_max:.4e}, dt_max={dt_max:.4e}");
        }

        // 3. Conv1d update on xBC with bias + silu activation
        // Input is mamba2_in_proj[intermediate..intermediate+cd], output to mamba2_conv_out
        {
            let state = &mut self.mamba2_states[mamba2_idx];
            let func = self.kernels.causal_conv1d.module.get_function("causal_conv1d_update_bias_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.conv.as_mut_ptr().cast();
            let mut in_ptr: *const std::ffi::c_void = unsafe {
                self.activations.mamba2_in_proj.as_ptr().add(nh * hd).cast()
            };
            let mut w_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_weight.as_ptr().cast() };
            let mut bias_ptr: *const std::ffi::c_void = unsafe { (*w).conv1d_bias.as_ptr().cast() };
            let mut out_ptr: *mut std::ffi::c_void = self.activations.mamba2_conv_out.as_mut_ptr().cast();
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
            func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, &self.stream, &mut args)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.mamba2_conv_out.len();
            let mut buf = vec![0.0f32; n];
            self.activations.mamba2_conv_out.copy_to_host(&mut buf)?;
            let x_max = buf[..nh*hd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let b_max = buf[nh*hd..nh*hd+ng*sd].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let c_max = buf[nh*hd+ng*sd..].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
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
            let c_ptr = self.activations.mamba2_conv_out.as_ptr().add(nh * hd + ng * sd);
            let dt_ptr = self.activations.mamba2_in_proj.as_ptr().add(nh * hd + cd);

            // Create temporary DeviceBuffer wrappers pointing to sub-regions
            // We need to call the kernel with raw pointers
            let func = self.kernels.ssm_update.module.get_function("selective_state_update_f32")?;
            let mut state_ptr: *mut std::ffi::c_void = state.ssm.as_mut_ptr().cast();
            let mut x_p: *const std::ffi::c_void = x_ptr.cast();
            let mut dt_p: *const std::ffi::c_void = dt_ptr.cast();
            let mut dt_bias_p: *const std::ffi::c_void = (*w).dt_bias.as_ptr().cast();
            let mut a_log_p: *const std::ffi::c_void = (*w).a_log.as_ptr().cast();
            let mut b_p: *const std::ffi::c_void = b_ptr.cast();
            let mut c_p: *const std::ffi::c_void = c_ptr.cast();
            let mut d_p: *const std::ffi::c_void = (*w).d.as_ptr().cast();
            let mut out_p: *mut std::ffi::c_void = self.activations.mamba2_ssm_out.as_mut_ptr().cast();
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

            func.launch(
                (nh as u32, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &mut args,
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
            eprintln!("  M2.L0 ssm_out: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 6. rmsnorm_gated: normed_out = rmsnorm(ssm_out * silu(gate)) * weight
        // Mamba2/Nemotron uses norm_before_gate=False: norm the gated product.
        // Per-group norm (group_size = intermediate / n_groups).
        let group_size = (nh * hd / ng) as u32;
        {
            let func = self.kernels.rmsnorm_gated.module.get_function("rmsnorm_gated_post_f32")?;
            for g in 0..ng {
                let off = g * group_size as usize;
                let mut out_p: *mut std::ffi::c_void = unsafe { self.activations.mamba2_conv_out.as_mut_ptr().add(off).cast() };
                let mut x_p: *const std::ffi::c_void = unsafe { self.activations.mamba2_ssm_out.as_ptr().add(off).cast() };
                let mut z_p: *const std::ffi::c_void = unsafe { self.activations.mamba2_in_proj.as_ptr().add(off).cast() };
                let mut w_p: *const std::ffi::c_void = unsafe { (*w).norm_weight.as_ptr().add(off).cast() };
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
            (*w).out_proj.forward(&self.kernels.linear_proj,
                &mut self.activations.out_proj, &self.activations.mamba2_conv_out,
                hs, intermediate, &self.stream)?;
        }

        if dbg {
            self.stream.synchronize()?;
            let n = self.activations.out_proj.len();
            let mut buf = vec![0.0f32; n.min(self.config.hidden_size)];
            self.activations.out_proj.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("  M2.L0 out_proj: max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
        }

        // 8. Residual add
        unsafe {
            crate::model::d2d_copy_f32(
                &mut self.activations.residual, 0,
                &self.activations.hidden, 0,
                self.config.hidden_size, &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs, &self.stream,
        )?;

        Ok(())
    }

    pub(crate) fn attention_forward(
        &mut self,
        layer_idx: usize,
        kv_cache_idx: usize,
        position: u32,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nqh = cfg.num_q_heads as u32;
        let nkh = cfg.num_kv_heads as u32;
        let hd = cfg.head_dim as u32;
        let rd = cfg.rope_dim as u32;
        let s0 = cfg.mrope_section[0] as u32;
        let s1 = cfg.mrope_section[1] as u32;
        let s2 = cfg.mrope_section[2] as u32;
        let eps = cfg.rms_norm_eps;
        let max_sl = cfg.max_seq_len as u32;
        let sync_debug = std::env::var("SYNC_DEBUG").is_ok();

        macro_rules! sync_check {
            ($label:expr) => {
                if sync_debug {
                    if let Err(e) = self.stream.synchronize() {
                        eprintln!("SYNC_DEBUG: crash at L{}.{}", layer_idx, $label);
                        return Err(e);
                    }
                    eprintln!("SYNC_DEBUG: L{}.{} OK", layer_idx, $label);
                }
            };
        }

        // 1. RMSNorm
        // SAFETY: Raw pointer breaks borrow on self.layers for mutable self.activations access.
        // Pointer valid: layers not modified during attention_forward.
        let input_norm = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("expected attention layer"),
        };
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*input_norm,
                1,
                hs,
                eps,
                cfg.rms_norm_one_plus_w,
                &self.stream,
            )?;
        }
        sync_check!("rmsnorm");

        // 2. Project Q+gate, K, V
        // Use raw pointers to LinearWeight to work around borrow checker
        // (self.layers borrows self, but we need &mut self.activations)
        let (w_q_gate_p, w_k_p, w_v_p, w_o_p, q_norm_w, k_norm_w) =
            match &self.layers[layer_idx] {
                LayerWeights::Attention(w) => (
                    &w.w_q_gate as *const LinearWeight,
                    &w.w_k as *const LinearWeight,
                    &w.w_v as *const LinearWeight,
                    &w.w_o as *const LinearWeight,
                    &w.q_norm as *const DeviceBuffer<u16>,
                    &w.k_norm as *const DeviceBuffer<u16>,
                ),
                _ => unreachable!(),
            };

        let q_mult = if cfg.has_output_gate { 2u32 } else { 1 };
        unsafe {
            (*w_q_gate_p).forward(&self.kernels.linear_proj,
                &mut self.activations.q_gate_attn, &self.activations.normed,
                nqh * hd * q_mult, hs, &self.stream)?;
            sync_check!("q_proj");
            (*w_k_p).forward(&self.kernels.linear_proj,
                &mut self.activations.k_attn, &self.activations.normed,
                nkh * hd, hs, &self.stream)?;
            sync_check!("k_proj");
            (*w_v_p).forward(&self.kernels.linear_proj,
                &mut self.activations.v_attn, &self.activations.normed,
                nkh * hd, hs, &self.stream)?;
            sync_check!("v_proj");
        }


        // 3. Split q_gate_attn → q, gate (gated) or just copy (non-gated)
        let hd_usize = hd as usize;
        if cfg.has_output_gate {
            unsafe {
                for h in 0..nqh as usize {
                    let src_q = h * hd_usize * 2;
                    let src_g = h * hd_usize * 2 + hd_usize;
                    let dst = h * hd_usize;
                    d2d_copy_f32(&mut self.activations.q_attn, dst, &self.activations.q_gate_attn, src_q, hd_usize, &self.stream)?;
                    d2d_copy_f32(&mut self.activations.gate_attn, dst, &self.activations.q_gate_attn, src_g, hd_usize, &self.stream)?;
                }
            }
        } else {
            // Non-gated: q_gate_attn IS q_attn, just copy
            let total = nqh as usize * hd_usize;
            unsafe {
                d2d_copy_f32(&mut self.activations.q_attn, 0, &self.activations.q_gate_attn, 0, total, &self.stream)?;
            }
        }
        sync_check!("q_copy");

        // 4a. Write K,V to cache BEFORE QK-norm (pre-norm K has full dynamic range
        //     for quantization; post-norm K is bounded ±0.06 which destroys Q4 quantization).
        //     See exterior_algebra kb-20260328-115542-e172dd.
        {
            let max_sl = self.config.max_seq_len;
            for h in 0..nkh as usize {
                let src_off = h * hd as usize;
                let dst_off = h * max_sl * hd as usize + position as usize * hd as usize;
                unsafe {
                    d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].k, dst_off, &self.activations.k_attn, src_off, hd as usize, &self.stream)?;
                    d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].v, dst_off, &self.activations.v_attn, src_off, hd as usize, &self.stream)?;
                }
            }
        }

        // 4b. QK norm (in-place on q_attn, k_attn — for current token's attention computation)
        if cfg.has_qk_norm {
            let q_norm_len = unsafe { (*q_norm_w).len() };
            if q_norm_len == hd as usize {
                // Per-head QK norm (Qwen3.5 style): weight is [head_dim]
                unsafe {
                    self.kernels.qk_norm.forward(
                        &mut self.activations.q_attn,
                        &mut self.activations.k_attn,
                        &*q_norm_w,
                        &*k_norm_w,
                        nqh,
                        nkh,
                        hd,
                        eps,
                        &self.stream,
                    )?;
                }
            } else {
                // Full-hidden QK norm (OLMoE style): weight is [hidden_size], apply as RMSNorm
                // Use normed buffer as temp to avoid aliasing
                unsafe {
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.q_attn,
                        &*q_norm_w,
                        1, nqh * hd, eps, cfg.rms_norm_one_plus_w, &self.stream,
                    )?;
                    d2d_copy_f32(&mut self.activations.q_attn, 0, &self.activations.normed, 0, (nqh * hd) as usize, &self.stream)?;
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.k_attn,
                        &*k_norm_w,
                        1, nkh * hd, eps, cfg.rms_norm_one_plus_w, &self.stream,
                    )?;
                    d2d_copy_f32(&mut self.activations.k_attn, 0, &self.activations.normed, 0, (nkh * hd) as usize, &self.stream)?;
                }
            }
        }


        // 5. Apply RoPE (skip for Nemotron-H which has no rotary embeddings)
        if cfg.use_rope {
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe { std::ptr::copy_nonoverlapping(pos_data.as_ptr(), self.activations.position_ids.host_ptr(), pos_data.len()) };

        self.kernels.mrope.forward(
            &mut self.activations.q_attn,
            &mut self.activations.k_attn,
            &self.activations.inv_freq,
            self.activations.position_ids.as_ptr(),
            nqh,
            nkh,
            hd,
            rd,
            s0,
            s1,
            s2,
            &self.stream,
        )?;
        sync_check!("mrope");
        } // end if cfg.use_rope

        // 6. KV write already done at step 4a (before QK-norm) for quantization quality.


        // 7. GQA attention
        let seq_len = position + 1;
        self.kernels.gqa_attention.forward(
            &mut self.activations.attn_out,
            &self.activations.q_attn,
            &self.kv_caches[kv_cache_idx].k,
            &self.kv_caches[kv_cache_idx].v,
            nqh,
            nkh,
            hd,
            seq_len,
            max_sl as u32,
            &self.stream,
        )?;
        sync_check!("gqa_attention");


        // 8. Output gate (Qwen3.5 only) or pass-through
        let final_attn = if cfg.has_output_gate {
            self.kernels.output_gate.forward(
                &mut self.activations.gated_out,
                &self.activations.attn_out,
                &self.activations.gate_attn,
                nqh * hd,
                &self.stream,
            )?;
            &self.activations.gated_out as *const DeviceBuffer<f32>
        } else {
            &self.activations.attn_out as *const DeviceBuffer<f32>
        };


        // 9. Output projection
        unsafe {
            (*w_o_p).forward(&self.kernels.linear_proj,
                &mut self.activations.out_proj, &*final_attn,
                hs, nqh * hd, &self.stream)?;
        }


        // 10. Residual add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
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
