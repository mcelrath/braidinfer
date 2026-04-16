use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;

use super::Model;
use crate::config::*;
use crate::weights::*;

impl Model {
    /// Uses individual kernel launches (no megakernel).
    pub(crate) fn moe_ffn_forward(&mut self, layer_idx: usize) -> HipResult<()> {
        let moe = self.moe_weights[layer_idx]
            .as_ref()
            .expect("moe_ffn_forward called on non-MoE layer");
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let ne = moe.num_experts;
        let eps = self.config.rms_norm_eps;
        // Expert input/output dimension: moe_latent_size (1024) for Nemotron-H, hs for standard models.
        let latent_size = moe.gate_up_in_dim;
        // SAFETY: Raw pointers break borrow on self.moe_weights to allow mutable access to
        // self.activations. Pointers valid for duration of this function (moe_weights not modified).
        let fc1_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc1_latent_proj.as_ref().map(|w| w as *const _);
        let fc2_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc2_latent_proj.as_ref().map(|w| w as *const _);
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
                1,
                hs as u32,
                eps,
                self.config.rms_norm_one_plus_w,
                &self.stream,
            )?;
        }

        // Save residual
        unsafe {
            d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                hs,
                &self.stream,
            )?;
        }

        // 2. Gate projection: normed → scores[num_experts]
        self.kernels.linear_proj.forward(
            &mut self.activations.moe_scores,
            &moe.gate,
            &self.activations.normed,
            ne as u32,
            hs as u32,
            &self.stream,
        )?;

        let (k, gate_type) = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE {
                num_active,
                gate_type,
                ..
            } => (*num_active, gate_type.clone()),
            _ => unreachable!(),
        };

        // 3. GPU-side top-k selection + weight computation (replaces CPU round-trip)
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK {
                routed_scaling_factor,
            } => (1, *routed_scaling_factor),
            GateType::Sigmoid {
                routed_scaling_factor,
            } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe
            .score_correction_bias_gpu
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null());
        self.kernels.moe_gate.forward(
            &self.activations.moe_scores,
            self.activations.moe_expert_ids.as_mut_ptr(),
            self.activations.moe_expert_weights.as_mut_ptr(),
            bias_ptr,
            ne as u32,
            k as u32,
            gate_mode,
            rsf,
            &self.stream,
        )?;

        let used_multi_gpu = false; // multi-GPU dispatch via OP_MOE_DISPATCH megakernel (moe_p2p.rs)

        // Trace: MoE routing (after gate, before expert FFN)
        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut normed_buf = vec![0.0f32; hs];
            self.activations.normed.copy_to_host(&mut normed_buf)?;
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_normed"), &normed_buf);

            let ids = unsafe {
                std::slice::from_raw_parts(self.activations.moe_expert_ids.host_ptr(), k)
            };
            let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_expert_ids"), &ids_f32);

            let weights = unsafe {
                std::slice::from_raw_parts(self.activations.moe_expert_weights.host_ptr(), k)
            };
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_expert_weights"), weights);
        }

        if !used_multi_gpu {
            // Single-GPU path: read back expert_ids + weights via host pointer (no hipMemcpy)
            self.stream.synchronize()?;
            let expert_ids: Vec<i32> = unsafe {
                std::slice::from_raw_parts(
                    self.activations.moe_expert_ids.host_ptr() as *const i32,
                    k,
                )
                .to_vec()
            };
            let expert_weights: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    self.activations.moe_expert_weights.host_ptr() as *const f32,
                    k,
                )
                .to_vec()
            };

            // 4a. Apply fc1_latent_proj if present: normed(hs) → moe_latent(latent_size).
            // moe_latent is the expert input and is READ-ONLY during the expert loop.
            if let Some(fc1) = fc1_ptr {
                unsafe { &*fc1 }.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.moe_latent,
                    &self.activations.normed,
                    latent_size as u32,
                    hs as u32,
                    &self.stream,
                )?;
            }

            // 4b. Zero accumulation buffer: moe_expert_out[0..latent_size] (if fc2) else ffn_down.
            // When fc2 is present, accumulate latent_size floats into moe_expert_out (reused as
            // latent accumulator), then fc2 projects moe_expert_out → ffn_down after the loop.
            let (accum_ptr, accum_size) = if fc2_ptr.is_some() {
                (self.activations.moe_expert_out.as_mut_ptr() as *mut std::ffi::c_void, latent_size)
            } else {
                (self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void, hs)
            };
            unsafe {
                let rc = braidinfer_hip::ffi::hipMemsetAsync(accum_ptr, 0, accum_size * 4, self.stream.raw());
                if rc != 0 {
                    return Err(braidinfer_hip::HipError(rc).into());
                }
            }

            // Debug MoE routing
            if self.debug_nan && layer_idx <= 3 {
                eprintln!(
                    "  MoE L{layer_idx}: topk={:?} weights={:?}",
                    &expert_ids, &expert_weights
                );
            }

            // 5. For each selected expert: run FFN and GPU-accumulate
            // Input: moe_latent (if fc1 present) else normed.
            // Per-expert temp: moe_expert_gate, moe_expert_up, moe_expert_act, then moe_expert_out.
            // Accumulate: moe_expert_out (reused as latent accumulator if fc2) else ffn_down.
            let expert_in_dim = moe.gate_up_in_dim;
            let expert_input_ptr: *const f32 = if fc1_ptr.is_some() {
                self.activations.moe_latent.as_ptr()
            } else {
                self.activations.normed.as_ptr()
            };
            for j in 0..k {
                let expert_id = expert_ids[j] as usize;
                let w = expert_weights[j];

                let down_offset = moe.expert_down.row_byte_offset_dim(expert_id * expert_in_dim, eis);

                if moe.has_gate_proj {
                    // SwiGLU: gate_proj → silu → * up_proj
                    let gate_offset = moe
                        .expert_gate_up
                        .row_byte_offset_dim(expert_id * 2 * eis, expert_in_dim);
                    let up_offset = moe
                        .expert_gate_up
                        .row_byte_offset_dim(expert_id * 2 * eis + eis, expert_in_dim);

                    moe.expert_gate_up.forward_sub(
                        &self.kernels.linear_proj,
                        self.activations.moe_expert_gate.as_mut_ptr(),
                        expert_input_ptr,
                        eis as u32,
                        expert_in_dim as u32,
                        gate_offset,
                        &self.stream,
                    )?;
                    moe.expert_gate_up.forward_sub(
                        &self.kernels.linear_proj,
                        self.activations.moe_expert_up.as_mut_ptr(),
                        expert_input_ptr,
                        eis as u32,
                        expert_in_dim as u32,
                        up_offset,
                        &self.stream,
                    )?;
                    self.kernels.silu_mul.forward(
                        &mut self.activations.moe_expert_act,
                        &self.activations.moe_expert_gate,
                        &self.activations.moe_expert_up,
                        eis as u32,
                        &self.stream,
                    )?;
                } else {
                    // relu²: up_proj → relu² (no gate_proj)
                    let up_offset = moe.expert_gate_up.row_byte_offset_dim(expert_id * eis, expert_in_dim);
                    moe.expert_gate_up.forward_sub(
                        &self.kernels.linear_proj,
                        self.activations.moe_expert_up.as_mut_ptr(),
                        expert_input_ptr,
                        eis as u32,
                        expert_in_dim as u32,
                        up_offset,
                        &self.stream,
                    )?;
                    self.kernels.silu_mul.relu_squared(
                        &mut self.activations.moe_expert_act,
                        &self.activations.moe_expert_up,
                        eis as u32,
                        &self.stream,
                    )?;
                }

                if fc2_ptr.is_some() {
                    // With fc2: use ffn_down as per-expert temp, accumulate into moe_expert_out.
                    // After loop: fc2(moe_expert_out) → ffn_down.
                    moe.expert_down.forward_sub(
                        &self.kernels.linear_proj,
                        self.activations.ffn_down.as_mut_ptr(),
                        self.activations.moe_expert_act.as_ptr(),
                        expert_in_dim as u32,
                        eis as u32,
                        down_offset,
                        &self.stream,
                    )?;
                    self.kernels.residual_add.weighted_accumulate(
                        &mut self.activations.moe_expert_out,
                        &self.activations.ffn_down,
                        w,
                        expert_in_dim as u32,
                        &self.stream,
                    )?;
                } else {
                    // Without fc2: use moe_expert_out as per-expert temp, accumulate into ffn_down.
                    moe.expert_down.forward_sub(
                        &self.kernels.linear_proj,
                        self.activations.moe_expert_out.as_mut_ptr(),
                        self.activations.moe_expert_act.as_ptr(),
                        expert_in_dim as u32,
                        eis as u32,
                        down_offset,
                        &self.stream,
                    )?;
                    self.kernels.residual_add.weighted_accumulate(
                        &mut self.activations.ffn_down,
                        &self.activations.moe_expert_out,
                        w,
                        expert_in_dim as u32,
                        &self.stream,
                    )?;
                }
            }

            // 5b. Apply fc2_latent_proj if present: accumulated moe_expert_out → ffn_down.
            if let Some(fc2) = fc2_ptr {
                unsafe { &*fc2 }.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.ffn_down,
                    &self.activations.moe_expert_out,
                    hs as u32,
                    latent_size as u32,
                    &self.stream,
                )?;
            }
        } // end if !used_multi_gpu

        // 6. Shared expert (always-on, added to output — runs on GPU 0 for both paths)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &self.config.layers[layer_idx].ffn_type {
                FfnType::MoE {
                    shared_intermediate_size,
                    expert_intermediate_size,
                    ..
                } => {
                    if *shared_intermediate_size > 0 {
                        *shared_intermediate_size
                    } else {
                        *expert_intermediate_size
                    }
                }
                _ => eis,
            };
            // Shared expert receives the same normed input as routed experts.
            // HF NemotronHMOE: residuals = hidden_states (already normed by block)
            se.up_proj.forward(
                &self.kernels.linear_proj,
                &mut self.activations.moe_expert_up,
                &self.activations.normed,
                se_is as u32,
                hs as u32,
                &self.stream,
            )?;

            if moe.has_gate_proj {
                // SwiGLU shared expert
                se.gate_proj.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.moe_expert_gate,
                    &self.activations.normed,
                    se_is as u32,
                    hs as u32,
                    &self.stream,
                )?;
                self.kernels.silu_mul.forward(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_gate,
                    &self.activations.moe_expert_up,
                    se_is as u32,
                    &self.stream,
                )?;
            } else {
                // relu² shared expert
                self.kernels.silu_mul.relu_squared(
                    &mut self.activations.moe_expert_act,
                    &self.activations.moe_expert_up,
                    se_is as u32,
                    &self.stream,
                )?;
            }

            se.down_proj.forward(
                &self.kernels.linear_proj,
                &mut self.activations.moe_expert_out,
                &self.activations.moe_expert_act,
                hs as u32,
                se_is as u32,
                &self.stream,
            )?;

            // Apply shared expert gate: ffn_down[i] += sigmoid(dot(gate_w, normed)) * expert_out[i]
            if let Some(ref gate_buf) = moe.shared_expert_gate {
                self.kernels.dot_sigmoid_scale_add.forward(
                    &mut self.activations.ffn_down,
                    &self.activations.moe_expert_out,
                    &self.activations.normed,
                    gate_buf,
                    hs as u32,
                    &self.stream,
                )?;
            } else {
                self.kernels.residual_add.weighted_accumulate(
                    &mut self.activations.ffn_down,
                    &self.activations.moe_expert_out,
                    1.0,
                    hs as u32,
                    &self.stream,
                )?;
            }
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
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_ffn_down"), &buf);
        }

        // 7. Residual add: hidden = residual + ffn_down
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.residual,
            &self.activations.ffn_down,
            hs as u32,
            &self.stream,
        )?;

        Ok(())
    }
}
