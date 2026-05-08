use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;

use super::Model;
use crate::config::*;
use crate::weights::*;
use crate::gpu_utils::d2d_copy_f32;
use crate::moe_p2p::{MAX_ACTIVE_EXPERTS, MAX_PREFILL_BATCH};

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

    /// Process n tokens through one MoE layer using batched P2P dispatch.
    /// Input: prefill_hidden[0..n*hs] — current layer's input.
    /// Output: prefill_hidden[0..n*hs] updated with MoE FFN output + residual.
    ///
    /// Dispatches all n tokens to worker GPUs in a single round-trip, while GPU 0
    /// computes its own local experts in parallel. Replaces n sequential round-trips.
    pub(crate) fn moe_ffn_forward_prefill_batched(
        &mut self,
        layer_idx: usize,
        prefill_hidden: &mut DeviceBuffer<f32>, // [n × hs]
        n: usize,
    ) -> HipResult<()> {
        assert!(n <= MAX_PREFILL_BATCH, "n={n} exceeds MAX_PREFILL_BATCH={MAX_PREFILL_BATCH}");
        // SAFETY: raw pointer breaks borrow on moe_weights to allow mutable self.activations access.
        let moe_ptr = self.moe_weights[layer_idx]
            .as_ref()
            .expect("moe_ffn_forward_prefill_batched called on non-MoE layer")
            as *const crate::weights::MoeWeights;
        let moe = unsafe { &*moe_ptr };
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let latent_size = moe.gate_up_in_dim;
        let has_gate = moe.has_gate_proj;
        let fc1_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc1_latent_proj.as_ref().map(|w| w as *const _);
        let fc2_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc2_latent_proj.as_ref().map(|w| w as *const _);
        let norm_weight = match &self.layers[layer_idx] {
            LayerWeights::MoeFfn(w) => &w.input_norm as *const DeviceBuffer<u16>,
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
            _ => panic!("layer {} is not MoeFfn, Attention, or Gdn", layer_idx),
        };
        let (k, gate_type) = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { num_active, gate_type, .. } => (*num_active, gate_type.clone()),
            _ => unreachable!(),
        };
        assert!(k <= MAX_ACTIVE_EXPERTS, "k={k} > MAX_ACTIVE_EXPERTS={MAX_ACTIVE_EXPERTS}");
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

        let mut all_activations = vec![0.0f32; n * latent_size];
        let mut all_expert_ids = vec![0i32; n * k];
        let mut all_expert_weights = vec![0.0f32; n * k];

        // Step 1: Per-token norm + gate + topk + collect activation to host staging
        for t in 0..n {
            unsafe {
                d2d_copy_f32(&mut self.activations.hidden, 0, prefill_hidden, t * hs, hs, &self.stream)?;
            }
            unsafe {
                self.kernels.rmsnorm.forward(
                    &mut self.activations.normed,
                    &self.activations.hidden,
                    &*norm_weight,
                    1, hs as u32, self.config.rms_norm_eps, self.config.rms_norm_one_plus_w,
                    &self.stream,
                )?;
            }
            if let Some(fc1) = fc1_ptr {
                unsafe { &*fc1 }.forward(
                    &self.kernels.linear_proj, &mut self.activations.moe_latent,
                    &self.activations.normed, latent_size as u32, hs as u32, &self.stream,
                )?;
            }
            self.kernels.linear_proj.forward(
                &mut self.activations.moe_scores, &moe.gate, &self.activations.normed,
                moe.num_experts as u32, hs as u32, &self.stream,
            )?;
            self.kernels.moe_gate.forward(
                &self.activations.moe_scores,
                self.activations.moe_expert_ids.as_mut_ptr(),
                self.activations.moe_expert_weights.as_mut_ptr(),
                bias_ptr, moe.num_experts as u32, k as u32, gate_mode, rsf, &self.stream,
            )?;
            self.stream.synchronize()?;

            // Trace: capture normed (pre-fc1) for token 0 — safe here, persistent worker not started.
            if t == 0 && self.trace.is_some() {
                let mut normed_buf = vec![0.0f32; hs];
                self.activations.normed.copy_to_host(&mut normed_buf)?;
                self.trace.as_mut().unwrap()
                    .write_checkpoint(&format!("L{layer_idx}.moe_normed"), &normed_buf);
            }

            let act_src = if fc1_ptr.is_some() {
                self.activations.moe_latent.as_ptr()
            } else {
                self.activations.normed.as_ptr()
            };
            let dst_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    all_activations[t * latent_size..].as_mut_ptr() as *mut u8,
                    latent_size * 4,
                )
            };
            braidinfer_hip::memory::memcpy_d2h(dst_bytes, act_src as *const u8, latent_size * 4)?;

            let ids_src = unsafe {
                std::slice::from_raw_parts(self.activations.moe_expert_ids.host_ptr() as *const i32, k)
            };
            let wts_src = unsafe {
                std::slice::from_raw_parts(self.activations.moe_expert_weights.host_ptr() as *const f32, k)
            };
            all_expert_ids[t * k..(t + 1) * k].copy_from_slice(ids_src);
            all_expert_weights[t * k..(t + 1) * k].copy_from_slice(wts_src);
        }

        // Trace: write expert IDs and weights for token 0
        if self.trace.is_some() {
            let ids_f32: Vec<f32> = all_expert_ids[0..k].iter().map(|&x| x as f32).collect();
            self.trace.as_mut().unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_expert_ids"), &ids_f32);
            self.trace.as_mut().unwrap()
                .write_checkpoint(&format!("L{layer_idx}.moe_expert_weights"), &all_expert_weights[0..k]);
        }

        // Step 2: Stage activations + per-token routing into GPU 0 VRAM (UC) so
        // workers can P2P-read them. Workers will be dispatched per-token via
        // OP_MOE_FFN_REMOTE on their persistent_worker mailbox.
        // We need the routing on the GPU side (P2P-readable). The activation
        // staging buffer is already UC; expert ids/weights live in
        // host-mapped GPU 0 buffers (act.moe_expert_ids/_weights) but those
        // hold only one token. For prefill we keep them on the host side and
        // pass the host pointer via hipHostGetDevicePointer (single-GPU model
        // path uses host-mapped expert_ids/weights too — see model_load).
        // Simpler approach: write per-token id/weight into act.moe_expert_ids
        // and dispatch immediately, since the worker reads them once.
        let (output_slots_raw, num_gpus, num_workers, worker_act_bases) = {
            let p2p = self.moe_p2p.as_mut().expect("moe_p2p not initialized for prefill batched");
            // Per-worker device pointers to the portable host-mapped
            // activation_staging. Even with `hipHostMallocPortable`, ROCm
            // requires the device pointer to be retrieved from each GPU's
            // context (done at init time, see moe_p2p::MoeP2pContext::init).
            // Pilot site for typed-pointer epic braidinfer-77r.5.
            let bases: Vec<*mut f32> = (0..p2p.workers.len())
                .map(|w| p2p.activation_staging_dev_ptr_for(w))
                .collect();
            (
                p2p.output_slots.as_mut_ptr(),
                p2p.num_gpus,
                p2p.workers.len(),
                bases,
            )
        };
        // Write all_activations directly to the host-mapped staging buffer.
        // No DMA, no kernel launch — CPU stores land in pinned host RAM and
        // are immediately visible to worker GPUs via the portable device_ptr.
        // Replaces a former copy_from_host() that deadlocked under GPU 0's
        // persistent_worker (eh2). Empirical bandwidth: ~10 GB/s per worker
        // × 4 workers concurrent = ~40 GB/s aggregate (vs ~6 GB/s for UC P2P).
        {
            let p2p = self.moe_p2p.as_mut().unwrap();
            let host_ptr = p2p.activation_staging.host_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    all_activations.as_ptr(),
                    host_ptr,
                    n * latent_size,
                );
            }
        }

        // Step 2.5: Dispatch OP_MOE_FFN_REMOTE for all (token, worker) pairs.
        // Fire all workers concurrently per token; sequencing across tokens is
        // handled by per-worker FIFO. We collect (gpu_idx, seq) and wait at end.
        let mut all_seqs: Vec<(usize, u32)> = Vec::with_capacity(n * num_workers);
        for t in 0..n {
            // Routing pointers for token t: host-mapped expert_ids/_weights are
            // single-token buffers; copy this token's slice to them so the
            // worker can read after we trigger the dispatch.
            unsafe {
                let ids_dst = self.activations.moe_expert_ids.host_ptr() as *mut i32;
                let wts_dst = self.activations.moe_expert_weights.host_ptr() as *mut f32;
                for j in 0..k {
                    std::ptr::write_volatile(ids_dst.add(j), all_expert_ids[t * k + j]);
                    std::ptr::write_volatile(wts_dst.add(j), all_expert_weights[t * k + j]);
                }
            }
            // Build instruction per worker.
            let p2p = self.moe_p2p.as_ref().unwrap();
            let dispatch = self.persistent_workers.as_mut().unwrap();
            for w in 0..num_workers {
                let gpu_id = w + 1;
                let out_slot = unsafe { output_slots_raw.add((t * num_gpus + gpu_id) * hs) };
                // Per-worker device pointer (portable host-mapped requires
                // per-context device pointers on AMD ROCm).
                let act_p2p = unsafe {
                    worker_act_bases[w].add(t * latent_size) as *const f32
                };
                let inst = p2p.build_ffn_remote_inst(
                    w, layer_idx, act_p2p, out_slot,
                    self.activations.moe_expert_ids.as_ptr() as *const i32,
                    self.activations.moe_expert_weights.as_ptr() as *const f32,
                    k, eis, hs, latent_size, has_gate, !has_gate,
                );
                let single = std::slice::from_ref(&inst);
                let seq = dispatch.dispatch_batch_fire(gpu_id, single);
                all_seqs.push((gpu_id, seq));
            }
            // Wait for THIS token's worker dispatches before reusing the
            // single-slot moe_expert_ids/_weights buffers for the next token.
            for (g, s) in all_seqs.drain(..) {
                dispatch.wait_ack(g, s);
            }
        }

        // Step 3: GPU 0 computes its local expert subset for all tokens.
        // Accumulate into ffn_down, then D2D copy to output_slots[(t * num_gpus + 0) * hs].
        for t in 0..n {
            // Load token t's activation into GPU buffer for expert computation
            let act_dst_ptr = if fc1_ptr.is_some() {
                self.activations.moe_latent.as_mut_ptr() as *mut u8
            } else {
                self.activations.normed.as_mut_ptr() as *mut u8
            };
            let src_bytes = unsafe {
                std::slice::from_raw_parts(
                    all_activations[t * latent_size..].as_ptr() as *const u8,
                    latent_size * 4,
                )
            };
            braidinfer_hip::memory::memcpy_h2d(act_dst_ptr, src_bytes, latent_size * 4)?;

            // Zero ffn_down for accumulation
            unsafe {
                let rc = braidinfer_hip::ffi::hipMemsetAsync(
                    self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
                    0, hs * 4, self.stream.raw(),
                );
                if rc != 0 { return Err(braidinfer_hip::HipError(rc)); }
            }

            let expert_in: *const f32 = if fc1_ptr.is_some() {
                self.activations.moe_latent.as_ptr()
            } else {
                self.activations.normed.as_ptr()
            };

            // Run experts assigned to GPU 0 (slot_map[eid] is Some iff on GPU 0).
            // Use dist.gpu0_gate_up_base/gpu0_down_base with slot-based strides:
            // moe.expert_gate_up is lite-loaded (hipMalloc(0)) for MULTI_GPU models and
            // must not be dereferenced.
            if let Some(dist) = &self.distributed_moe[layer_idx] {
                let buf0 = &dist.expert_buffers[0];
                let func_name = match dist.weight_format {
                    crate::model::WeightFormat::Bf16 => "linear_proj_f32",
                    crate::model::WeightFormat::Rnf4G128 => "linear_proj_rnf4_g128",
                    crate::model::WeightFormat::PcG32Q4 => "linear_proj_pcg32_q4",
                };
                for j in 0..k {
                    let eid = all_expert_ids[t * k + j] as usize;
                    let ew = all_expert_weights[t * k + j];
                    let slot = match buf0.slot_map[eid] { Some(s) => s, None => continue };
                    let gu_ptr = unsafe { dist.gpu0_gate_up_base.add(slot * dist.gate_up_expert_stride) };
                    let dn_ptr = unsafe { dist.gpu0_down_base.add(slot * dist.down_expert_stride) };
                    if has_gate {
                        // gate rows [0..eis), up rows [eis..2*eis) packed consecutively
                        let up_ptr = unsafe { gu_ptr.add(eis * dist.gate_up_row_stride) };
                        self.kernels.linear_proj.forward_packed_ptr(
                            self.activations.moe_expert_gate.as_mut_ptr(),
                            gu_ptr, expert_in, eis as u32, latent_size as u32, func_name, &self.stream,
                        )?;
                        self.kernels.linear_proj.forward_packed_ptr(
                            self.activations.moe_expert_up.as_mut_ptr(),
                            up_ptr, expert_in, eis as u32, latent_size as u32, func_name, &self.stream,
                        )?;
                        self.kernels.silu_mul.forward(
                            &mut self.activations.moe_expert_act,
                            &self.activations.moe_expert_gate,
                            &self.activations.moe_expert_up,
                            eis as u32, &self.stream,
                        )?;
                    } else {
                        self.kernels.linear_proj.forward_packed_ptr(
                            self.activations.moe_expert_up.as_mut_ptr(),
                            gu_ptr, expert_in, eis as u32, latent_size as u32, func_name, &self.stream,
                        )?;
                        self.kernels.silu_mul.relu_squared(
                            &mut self.activations.moe_expert_act,
                            &self.activations.moe_expert_up,
                            eis as u32, &self.stream,
                        )?;
                    }
                    self.kernels.linear_proj.forward_packed_ptr(
                        self.activations.moe_expert_out.as_mut_ptr(),
                        dn_ptr, self.activations.moe_expert_act.as_ptr(),
                        latent_size as u32, eis as u32, func_name, &self.stream,
                    )?;
                    self.kernels.residual_add.weighted_accumulate(
                        &mut self.activations.ffn_down,
                        &self.activations.moe_expert_out,
                        ew, latent_size as u32, &self.stream,
                    )?;
                }
            }

            // D2D copy ffn_down → output_slots[(t * num_gpus + 0) * hs]
            unsafe {
                let rc = braidinfer_hip::ffi::hipMemcpyAsync(
                    output_slots_raw.add(t * num_gpus * hs) as *mut std::ffi::c_void,
                    self.activations.ffn_down.as_ptr() as *const std::ffi::c_void,
                    latent_size * 4,
                    braidinfer_hip::ffi::hipMemcpyDeviceToDevice,
                    self.stream.raw(),
                );
                if rc != 0 { return Err(braidinfer_hip::HipError(rc)); }
            }
        }

        // Step 4: Wait for GPU 0's stream to finish (its expert compute already
        // wrote to output_slots[t * num_gpus * hs]). Worker dispatches are
        // already ack'd above (see Step 2.5).
        self.stream.synchronize()?;

        // Step 5: Sum output_slots + shared expert + residual per token
        let output_slots_raw = self.moe_p2p.as_ref().unwrap().output_slots.as_ptr();
        for t in 0..n {
            unsafe {
                d2d_copy_f32(&mut self.activations.hidden, 0, prefill_hidden, t * hs, hs, &self.stream)?;
                d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
            }
            // Zero ffn_down; sum all GPU slots into it via moe_expert_out as temp
            unsafe {
                let rc = braidinfer_hip::ffi::hipMemsetAsync(
                    self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
                    0, hs * 4, self.stream.raw(),
                );
                if rc != 0 { return Err(braidinfer_hip::HipError(rc)); }
            }
            for g in 0..num_gpus {
                let slot_offset = (t * num_gpus + g) * hs;
                unsafe {
                    let rc = braidinfer_hip::ffi::hipMemcpyAsync(
                        self.activations.moe_expert_out.as_mut_ptr() as *mut std::ffi::c_void,
                        output_slots_raw.add(slot_offset) as *const std::ffi::c_void,
                        latent_size * 4,
                        braidinfer_hip::ffi::hipMemcpyDeviceToDevice,
                        self.stream.raw(),
                    );
                    if rc != 0 { return Err(braidinfer_hip::HipError(rc)); }
                }
                self.kernels.residual_add.weighted_accumulate(
                    &mut self.activations.ffn_down,
                    &self.activations.moe_expert_out,
                    1.0, latent_size as u32, &self.stream,
                )?;
            }

            // fc2_latent_proj if present: ffn_down has latent-space sum, project → hs
            if let Some(fc2) = fc2_ptr {
                // Copy ffn_down (latent) → moe_expert_out as input to fc2
                unsafe {
                    let rc = braidinfer_hip::ffi::hipMemcpyAsync(
                        self.activations.moe_expert_out.as_mut_ptr() as *mut std::ffi::c_void,
                        self.activations.ffn_down.as_ptr() as *const std::ffi::c_void,
                        latent_size * 4,
                        braidinfer_hip::ffi::hipMemcpyDeviceToDevice,
                        self.stream.raw(),
                    );
                    if rc != 0 { return Err(braidinfer_hip::HipError(rc)); }
                }
                unsafe { &*fc2 }.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.ffn_down,
                    &self.activations.moe_expert_out,
                    hs as u32, latent_size as u32, &self.stream,
                )?;
            }

            // Shared expert (re-compute normed for this token)
            let moe = self.moe_weights[layer_idx].as_ref().unwrap();
            if let Some(ref se) = moe.shared_expert {
                let se_is = match &self.config.layers[layer_idx].ffn_type {
                    FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                        if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                    }
                    _ => eis,
                };
                unsafe {
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed, &self.activations.hidden, &*norm_weight,
                        1, hs as u32, self.config.rms_norm_eps, self.config.rms_norm_one_plus_w,
                        &self.stream,
                    )?;
                }
                se.up_proj.forward(
                    &self.kernels.linear_proj, &mut self.activations.moe_expert_up,
                    &self.activations.normed, se_is as u32, hs as u32, &self.stream,
                )?;
                if moe.has_gate_proj {
                    se.gate_proj.forward(
                        &self.kernels.linear_proj, &mut self.activations.moe_expert_gate,
                        &self.activations.normed, se_is as u32, hs as u32, &self.stream,
                    )?;
                    self.kernels.silu_mul.forward(
                        &mut self.activations.moe_expert_act,
                        &self.activations.moe_expert_gate, &self.activations.moe_expert_up,
                        se_is as u32, &self.stream,
                    )?;
                } else {
                    self.kernels.silu_mul.relu_squared(
                        &mut self.activations.moe_expert_act, &self.activations.moe_expert_up,
                        se_is as u32, &self.stream,
                    )?;
                }
                se.down_proj.forward(
                    &self.kernels.linear_proj, &mut self.activations.moe_expert_out,
                    &self.activations.moe_expert_act, hs as u32, se_is as u32, &self.stream,
                )?;
                if let Some(ref gate_buf) = moe.shared_expert_gate {
                    self.kernels.dot_sigmoid_scale_add.forward(
                        &mut self.activations.ffn_down, &self.activations.moe_expert_out,
                        &self.activations.normed, gate_buf, hs as u32, &self.stream,
                    )?;
                } else {
                    self.kernels.residual_add.weighted_accumulate(
                        &mut self.activations.ffn_down, &self.activations.moe_expert_out,
                        1.0, hs as u32, &self.stream,
                    )?;
                }
            }

            // Trace: ffn_down after all expert accumulation + shared expert (token 0 only)
            if t == 0 && self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; hs];
                self.activations.ffn_down.copy_to_host(&mut buf)?;
                self.trace.as_mut().unwrap()
                    .write_checkpoint(&format!("L{layer_idx}.moe_ffn_down"), &buf);
            }

            // Residual add → write back to prefill_hidden[t * hs]
            self.kernels.residual_add.forward(
                &mut self.activations.hidden, &self.activations.residual,
                &self.activations.ffn_down, hs as u32, &self.stream,
            )?;
            unsafe {
                d2d_copy_f32(prefill_hidden, t * hs, &self.activations.hidden, 0, hs, &self.stream)?;
            }
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Single-GPU batched MoE prefill: processes n tokens with one sync instead of n.
    /// For PCG32Q4 weights: uses batched GEMM per expert (loads weights once for all routed tokens).
    /// For other formats or models with fc2_latent_proj: falls back to per-token.
    pub(crate) fn moe_ffn_forward_prefill_single_gpu_batched(
        &mut self,
        layer_idx: usize,
        prefill_hidden: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> HipResult<()> {
        let hs = self.config.hidden_size;

        // Fall back to per-token for Nemotron models with fc2_latent_proj (rare path)
        let has_fc2 = self.moe_weights[layer_idx].as_ref().unwrap().fc2_latent_proj.is_some();
        // Fall back for non-PCG32Q4 expert weights
        let use_batched_gemm = match &self.moe_weights[layer_idx].as_ref().unwrap().expert_gate_up {
            LinearWeight::Packed(pw) => pw.format == WeightFormat::PcG32Q4,
            _ => false,
        };
        // Fall back if expert weights are lite-loaded stubs (multi-GPU path where weights
        // were not transferred to GPU 0, e.g. MULTI_GPU auto-enabled but only 1 device available)
        let experts_loaded = self.moe_weights[layer_idx].as_ref().unwrap()
            .expert_gate_up.raw_data_ptr() != std::ptr::null();
        if has_fc2 || !use_batched_gemm || !experts_loaded {
            for t in 0..n {
                unsafe {
                    d2d_copy_f32(&mut self.activations.hidden, 0, prefill_hidden, t * hs, hs, &self.stream)?;
                }
                self.moe_ffn_forward(layer_idx)?;
                unsafe {
                    d2d_copy_f32(prefill_hidden, t * hs, &self.activations.hidden, 0, hs, &self.stream)?;
                }
            }
            return Ok(());
        }

        // SAFETY: raw ptr breaks borrow on moe_weights to allow mutable access to self.activations.
        let moe_ptr = self.moe_weights[layer_idx].as_ref().unwrap() as *const MoeWeights;
        let moe = unsafe { &*moe_ptr };
        let eis = moe.expert_intermediate_size;
        let ne = moe.num_experts;
        let eps = self.config.rms_norm_eps;
        let latent_size = moe.gate_up_in_dim;
        let has_gate = moe.has_gate_proj;

        let (k, gate_type) = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { num_active, gate_type, .. } => (*num_active, gate_type.clone()),
            _ => unreachable!(),
        };
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        let norm_weight_ptr = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::MoeFfn(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("no norm weight for layer {layer_idx}"),
        };

        // Raw pointers for weight buffers (avoid reborrowing moe_weights in expert loop)
        let gu_base: *const u8 = match &moe.expert_gate_up {
            LinearWeight::Packed(pw) => pw.data.as_ptr(),
            _ => unreachable!(),
        };
        let dn_base: *const u8 = match &moe.expert_down {
            LinearWeight::Packed(pw) => pw.data.as_ptr(),
            LinearWeight::Bf16(buf) => buf.as_ptr() as *const u8,
        };

        // PHASE 1: Queue all gate computations asynchronously (no per-token sync).
        // Saves residual + normed activations per token for use in expert phase.
        for t in 0..n {
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    self.activations.hidden.as_mut_ptr() as *mut std::ffi::c_void,
                    prefill_hidden.as_ptr().add(t * hs) as *const std::ffi::c_void,
                    hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
                // Save residual
                braidinfer_hip::ffi::hipMemcpyAsync(
                    self.activations.prefill_moe_residual.as_mut_ptr().add(t * hs) as *mut std::ffi::c_void,
                    self.activations.hidden.as_ptr() as *const std::ffi::c_void,
                    hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
            }
            unsafe {
                self.kernels.rmsnorm.forward(
                    &mut self.activations.normed,
                    &self.activations.hidden,
                    &*norm_weight_ptr,
                    1, hs as u32, eps,
                    self.config.rms_norm_one_plus_w,
                    &self.stream,
                )?;
            }
            // Save normed for expert phase (latent_size == hs when no fc1)
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    self.activations.prefill_moe_normed.as_mut_ptr().add(t * latent_size) as *mut std::ffi::c_void,
                    self.activations.normed.as_ptr() as *const std::ffi::c_void,
                    latent_size * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
            }
            // Gate projection: normed → moe_scores
            self.kernels.linear_proj.forward(
                &mut self.activations.moe_scores,
                &moe.gate,
                &self.activations.normed,
                ne as u32, hs as u32,
                &self.stream,
            )?;
            // Top-k selection: write to per-token slots in device buffer
            self.kernels.moe_gate.forward(
                &self.activations.moe_scores,
                unsafe { self.activations.prefill_moe_ids_dev.as_mut_ptr().add(t * k) },
                unsafe { self.activations.prefill_moe_weights_dev.as_mut_ptr().add(t * k) },
                bias_ptr,
                ne as u32, k as u32, gate_mode, rsf,
                &self.stream,
            )?;
        }

        // ONE sync for all gate computations
        self.stream.synchronize()?;

        // D2H: read all expert IDs and routing weights
        let mut all_ids = vec![0i32; n * k];
        let mut all_weights_host = vec![0.0f32; n * k];
        braidinfer_hip::memory::memcpy_d2h(
            unsafe { std::slice::from_raw_parts_mut(all_ids.as_mut_ptr() as *mut u8, n * k * 4) },
            self.activations.prefill_moe_ids_dev.as_ptr() as *const u8,
            n * k * 4,
        )?;
        braidinfer_hip::memory::memcpy_d2h(
            unsafe { std::slice::from_raw_parts_mut(all_weights_host.as_mut_ptr() as *mut u8, n * k * 4) },
            self.activations.prefill_moe_weights_dev.as_ptr() as *const u8,
            n * k * 4,
        )?;

        // PHASE 2: Batched expert computation — load each expert's weights once for all routed tokens.
        // Zero the output accumulator [n × hs]
        self.kernels.moe_prefill.zero_batch(
            self.activations.prefill_moe_ffn_out.as_mut_ptr(),
            (n * hs) as i32,
            &self.stream,
        )?;

        for eid in 0..ne {
            // Collect tokens routed to expert eid
            let mut token_indices: Vec<i32> = Vec::new();
            let mut token_weights_vec: Vec<f32> = Vec::new();
            for t in 0..n {
                for j in 0..k {
                    if all_ids[t * k + j] == eid as i32 {
                        token_indices.push(t as i32);
                        token_weights_vec.push(all_weights_host[t * k + j]);
                    }
                }
            }
            let count = token_indices.len();
            if count == 0 { continue; }

            // H2D: upload indices and per-token routing weights for scatter
            braidinfer_hip::memory::memcpy_h2d(
                self.activations.prefill_moe_token_indices.as_mut_ptr() as *mut u8,
                unsafe { std::slice::from_raw_parts(token_indices.as_ptr() as *const u8, count * 4) },
                count * 4,
            )?;
            braidinfer_hip::memory::memcpy_h2d(
                self.activations.prefill_moe_token_weights.as_mut_ptr() as *mut u8,
                unsafe { std::slice::from_raw_parts(token_weights_vec.as_ptr() as *const u8, count * 4) },
                count * 4,
            )?;

            // Gather token activations: prefill_moe_expert_input[0..count×latent_size]
            self.kernels.moe_prefill.gather(
                self.activations.prefill_moe_expert_input.as_mut_ptr(),
                self.activations.prefill_moe_normed.as_ptr(),
                self.activations.prefill_moe_token_indices.as_ptr(),
                count as i32, latent_size as i32,
                &self.stream,
            )?;

            // Byte offsets into packed weight buffers for expert eid
            let (gate_off, up_off) = if has_gate {
                (
                    moe.expert_gate_up.row_byte_offset_dim(eid * 2 * eis, latent_size),
                    moe.expert_gate_up.row_byte_offset_dim(eid * 2 * eis + eis, latent_size),
                )
            } else {
                (0, moe.expert_gate_up.row_byte_offset_dim(eid * eis, latent_size))
            };
            let dn_off = moe.expert_down.row_byte_offset_dim(eid * latent_size, eis);

            if has_gate {
                self.kernels.moe_prefill.linear_proj_pcg32_batched(
                    self.activations.prefill_moe_gate_out.as_mut_ptr(),
                    unsafe { gu_base.add(gate_off) },
                    self.activations.prefill_moe_expert_input.as_ptr(),
                    eis as i32, latent_size as i32, count as i32,
                    &self.stream,
                )?;
                self.kernels.moe_prefill.linear_proj_pcg32_batched(
                    self.activations.prefill_moe_up_out.as_mut_ptr(),
                    unsafe { gu_base.add(up_off) },
                    self.activations.prefill_moe_expert_input.as_ptr(),
                    eis as i32, latent_size as i32, count as i32,
                    &self.stream,
                )?;
                self.kernels.moe_prefill.silu_mul_batched(
                    self.activations.prefill_moe_act_out.as_mut_ptr(),
                    self.activations.prefill_moe_gate_out.as_ptr(),
                    self.activations.prefill_moe_up_out.as_ptr(),
                    (count * eis) as i32,
                    &self.stream,
                )?;
            } else {
                self.kernels.moe_prefill.linear_proj_pcg32_batched(
                    self.activations.prefill_moe_up_out.as_mut_ptr(),
                    unsafe { gu_base.add(up_off) },
                    self.activations.prefill_moe_expert_input.as_ptr(),
                    eis as i32, latent_size as i32, count as i32,
                    &self.stream,
                )?;
                self.kernels.moe_prefill.relu_sq_batched(
                    self.activations.prefill_moe_act_out.as_mut_ptr(),
                    self.activations.prefill_moe_up_out.as_ptr(),
                    (count * eis) as i32,
                    &self.stream,
                )?;
            }

            // Down projection: [count × latent_size] (expert_down out_dim = latent_size = hs here)
            self.kernels.moe_prefill.linear_proj_pcg32_batched(
                self.activations.prefill_moe_down_out.as_mut_ptr(),
                unsafe { dn_base.add(dn_off) },
                self.activations.prefill_moe_act_out.as_ptr(),
                latent_size as i32, eis as i32, count as i32,
                &self.stream,
            )?;

            // Scatter-add: prefill_moe_ffn_out[token_indices[j]*hs..] += token_weights[j] * down[j*hs..]
            self.kernels.moe_prefill.scatter_add_weighted(
                self.activations.prefill_moe_ffn_out.as_mut_ptr(),
                self.activations.prefill_moe_down_out.as_ptr(),
                self.activations.prefill_moe_token_indices.as_ptr(),
                self.activations.prefill_moe_token_weights.as_ptr(),
                count as i32, latent_size as i32,
                &self.stream,
            )?;
        }

        // PHASE 3: Per-token shared expert + residual add.
        let has_shared = moe.shared_expert.is_some();
        let se_is = if has_shared {
            match &self.config.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                }
                _ => eis,
            }
        } else { 0 };
        // Raw pointers for shared expert weights to avoid borrow issues
        let se_up_ptr = moe.shared_expert.as_ref().map(|se| &se.up_proj as *const LinearWeight);
        let se_gate_proj_ptr = if has_gate {
            moe.shared_expert.as_ref().map(|se| &se.gate_proj as *const LinearWeight)
        } else { None };
        let se_down_ptr = moe.shared_expert.as_ref().map(|se| &se.down_proj as *const LinearWeight);
        let se_gate_buf_ptr = moe.shared_expert_gate.as_ref().map(|g| g as *const DeviceBuffer<u16>);

        for t in 0..n {
            // Load batched expert output into ffn_down
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    self.activations.ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
                    self.activations.prefill_moe_ffn_out.as_ptr().add(t * hs) as *const std::ffi::c_void,
                    hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
            }

            if has_shared {
                // Restore normed for shared expert input
                unsafe {
                    braidinfer_hip::ffi::hipMemcpyAsync(
                        self.activations.normed.as_mut_ptr() as *mut std::ffi::c_void,
                        self.activations.prefill_moe_normed.as_ptr().add(t * hs) as *const std::ffi::c_void,
                        hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                    );
                }
                let se_up = unsafe { &*se_up_ptr.unwrap() };
                let se_down = unsafe { &*se_down_ptr.unwrap() };
                se_up.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.moe_expert_up,
                    &self.activations.normed,
                    se_is as u32, hs as u32,
                    &self.stream,
                )?;
                if let Some(gp) = se_gate_proj_ptr {
                    unsafe { &*gp }.forward(
                        &self.kernels.linear_proj,
                        &mut self.activations.moe_expert_gate,
                        &self.activations.normed,
                        se_is as u32, hs as u32,
                        &self.stream,
                    )?;
                    self.kernels.silu_mul.forward(
                        &mut self.activations.moe_expert_act,
                        &self.activations.moe_expert_gate,
                        &self.activations.moe_expert_up,
                        se_is as u32, &self.stream,
                    )?;
                } else {
                    self.kernels.silu_mul.relu_squared(
                        &mut self.activations.moe_expert_act,
                        &self.activations.moe_expert_up,
                        se_is as u32, &self.stream,
                    )?;
                }
                se_down.forward(
                    &self.kernels.linear_proj,
                    &mut self.activations.moe_expert_out,
                    &self.activations.moe_expert_act,
                    hs as u32, se_is as u32,
                    &self.stream,
                )?;
                if let Some(gb) = se_gate_buf_ptr {
                    self.kernels.dot_sigmoid_scale_add.forward(
                        &mut self.activations.ffn_down,
                        &self.activations.moe_expert_out,
                        &self.activations.normed,
                        unsafe { &*gb },
                        hs as u32, &self.stream,
                    )?;
                } else {
                    self.kernels.residual_add.weighted_accumulate(
                        &mut self.activations.ffn_down,
                        &self.activations.moe_expert_out,
                        1.0, hs as u32, &self.stream,
                    )?;
                }
            }

            // Restore residual and add FFN output
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    self.activations.residual.as_mut_ptr() as *mut std::ffi::c_void,
                    self.activations.prefill_moe_residual.as_ptr().add(t * hs) as *const std::ffi::c_void,
                    hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
            }
            self.kernels.residual_add.forward(
                &mut self.activations.hidden,
                &self.activations.residual,
                &self.activations.ffn_down,
                hs as u32, &self.stream,
            )?;
            // Write back to prefill_hidden
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    prefill_hidden.as_mut_ptr().add(t * hs) as *mut std::ffi::c_void,
                    self.activations.hidden.as_ptr() as *const std::ffi::c_void,
                    hs * 4, braidinfer_hip::ffi::hipMemcpyDeviceToDevice, self.stream.raw(),
                );
            }
        }

        self.stream.synchronize()?;
        Ok(())
    }
}
