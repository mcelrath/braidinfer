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
    /// bd 9gmh Phase 1 F+G+H: migrated from host-launched .forward kernels +
    /// memcpy_d2h to mailbox-dispatched megakernel programs. Safe to run
    /// while GPU 0's persistent_worker holds all CUs (the mailbox path uses
    /// only host-mapped WorkerQueue::inst[] writes — no compute on
    /// self.stream, no synchronous hipMemcpy). Per-token sequential dispatch
    /// (rmsnorm + fc1 + gate + moe_gate) — each token waits for the previous
    /// before reading its expert IDs. Step 3 (GPU 0 local expert compute)
    /// and Step 5 (sum + shared expert + residual) likewise emit per-token
    /// megakernel programs.
    pub(crate) fn moe_ffn_forward_prefill_batched(
        &mut self,
        layer_idx: usize,
        prefill_hidden: &mut DeviceBuffer<f32>, // [n × hs]
        n: usize,
    ) -> HipResult<()> {
        use crate::megakernel::compile_common::{linear_proj_opcode_ptr, rmsnorm_opcode, div_ceil};
        use crate::megakernel::instructions::{
            D2dCopyInst, DotSigmoidScaleAddInst, LinearProjInst, MoeGateInst, NopInst,
            ReluSqInst, ResidualAddInst, RmsNormInst, ScaleAddInst, SiluMulInst,
        };
        use crate::megakernel::{Instruction, OP_LINEAR_PROJ, OP_NOP};
        #[allow(unused_imports)]
        use crate::persistent_dispatch::BatchDispatcher;

        assert!(n <= MAX_PREFILL_BATCH, "n={n} exceeds MAX_PREFILL_BATCH={MAX_PREFILL_BATCH}");
        // SAFETY: raw pointers break borrow on moe_weights / layers / distributed_moe
        // to allow mutable self.activations + persistent_workers access during the
        // per-token instruction-stream construction.
        let moe_ptr = self.moe_weights[layer_idx]
            .as_ref()
            .expect("moe_ffn_forward_prefill_batched called on non-MoE layer")
            as *const crate::weights::MoeWeights;
        let moe = unsafe { &*moe_ptr };
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let latent_size = moe.gate_up_in_dim;
        let has_gate = moe.has_gate_proj;
        let eps = self.config.rms_norm_eps;
        let rms_one_plus_w = self.config.rms_norm_one_plus_w;
        let ne = moe.num_experts;

        let fc1_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc1_latent_proj.as_ref().map(|w| w as *const _);
        let fc2_ptr: Option<*const crate::quant::LinearWeight> =
            moe.fc2_latent_proj.as_ref().map(|w| w as *const _);
        let norm_weight_buf_ptr = match &self.layers[layer_idx] {
            LayerWeights::MoeFfn(w) => &w.input_norm as *const DeviceBuffer<u16>,
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
            _ => panic!("layer {} is not MoeFfn, Attention, or Gdn", layer_idx),
        };
        let norm_weight = unsafe { &*norm_weight_buf_ptr }.as_ptr();
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
        let bias_ptr: *const u8 = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());

        let rmsnorm_op = rmsnorm_opcode(rms_one_plus_w);
        // moe.gate is DeviceBuffer<u16> (always bf16, see weights.rs:85).
        let gate_proj_op = OP_LINEAR_PROJ;
        let gate_proj_data: *const u8 = moe.gate.as_ptr() as *const u8;
        let (fc1_op, fc1_data) = fc1_ptr
            .map(|p| linear_proj_opcode_ptr(unsafe { &*p }))
            .unwrap_or((0, std::ptr::null()));
        let (fc2_op, fc2_data) = fc2_ptr
            .map(|p| linear_proj_opcode_ptr(unsafe { &*p }))
            .unwrap_or((0, std::ptr::null()));

        // GPU 0 local expert routing (distributed MoE weights, per-format opcode).
        let (
            dist_weight_format,
            gate_up_expert_stride,
            down_expert_stride,
            gate_up_row_stride,
            gpu0_gate_up_base,
            gpu0_down_base,
            slot_map_owned,
        ) = {
            let dist = self.distributed_moe[layer_idx]
                .as_ref()
                .expect("distributed_moe not initialized for MoE prefill");
            let slot_map: Vec<Option<usize>> = dist.expert_buffers[0].slot_map.clone();
            (
                dist.weight_format,
                dist.gate_up_expert_stride,
                dist.down_expert_stride,
                dist.gate_up_row_stride,
                dist.gpu0_gate_up_base,
                dist.gpu0_down_base,
                slot_map,
            )
        };
        let expert_proj_op = match dist_weight_format {
            crate::weights::WeightFormat::Rnf4G128 => crate::megakernel::OP_LINEAR_PROJ_RNF4,
            crate::weights::WeightFormat::PcG32Q4 => crate::megakernel::OP_LINEAR_PROJ_PCG32,
            crate::weights::WeightFormat::Bf16 => OP_LINEAR_PROJ,
        };

        // Shared expert + shared-expert gate.
        let shared_expert_ptr: Option<*const crate::weights::DenseFfnWeights> =
            moe.shared_expert.as_ref().map(|se| se as *const _);
        let shared_expert_gate_ptr: Option<*const DeviceBuffer<u16>> =
            moe.shared_expert_gate.as_ref().map(|g| g as *const _);
        let se_is = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
            }
            _ => eis,
        };

        // Snapshot all p2p raw pointers up front to release the borrow before
        // re-borrowing persistent_workers (mailbox dispatcher) below.
        let (
            act_staging_dev,
            per_token_ids_host,
            per_token_wts_host,
            per_token_ids_dev,
            per_token_wts_dev,
            gpu0_zero_dev,
            gpu0_out_dev,
            num_workers,
            num_gpus,
            worker_act_bases,
        ) = {
            let p2p = self.moe_p2p.as_mut().expect("moe_p2p not initialized for prefill batched");
            // Host-mapped portable_coherent staging. Workers read via per-worker
            // per-context device_ptr (activation_staging_dev_ptr_for(w)).
            let act_staging_dev = p2p.activation_staging.as_ptr() as *mut f32;
            let per_token_ids_host = p2p.per_token_expert_ids.host_ptr();
            let per_token_wts_host = p2p.per_token_expert_weights.host_ptr();
            let per_token_ids_dev = p2p.per_token_expert_ids.as_mut_ptr();
            let per_token_wts_dev = p2p.per_token_expert_weights.as_mut_ptr();
            let gpu0_zero_dev = p2p.gpu0_zero_buffer.as_ptr();
            let gpu0_out_dev = p2p.output_slots_dev_ptrs[0];
            let num_workers = p2p.workers.len();
            let num_gpus = p2p.num_gpus;
            let worker_act_bases: Vec<*mut f32> = (0..num_workers)
                .map(|w| p2p.activation_staging_dev_ptr_for(w)).collect();
            (
                act_staging_dev,
                per_token_ids_host, per_token_wts_host,
                per_token_ids_dev, per_token_wts_dev,
                gpu0_zero_dev, gpu0_out_dev,
                num_workers, num_gpus,
                worker_act_bases,
            )
        };

        // VRAM scratch + activation pointers (single-token slots, reused per token).
        let hidden_dev = self.activations.hidden.as_mut_ptr();
        let normed_dev = self.activations.normed.as_mut_ptr();
        let scores_dev = self.activations.moe_scores.as_mut_ptr();
        let expert_gate_dev = self.activations.moe_expert_gate.as_mut_ptr();
        let expert_up_dev = self.activations.moe_expert_up.as_mut_ptr();
        let expert_act_dev = self.activations.moe_expert_act.as_mut_ptr();
        let expert_out_dev = self.activations.moe_expert_out.as_mut_ptr();
        let ffn_down_dev = self.activations.ffn_down.as_mut_ptr();
        let residual_dev = self.activations.residual.as_mut_ptr();

        let device_idx = self.device.0 as usize;

        // === STEP 1: ALL tokens' rmsnorm + (fc1?) + gate_proj + moe_gate in ONE batch ===
        // bd 9gmh Phase 1 (udi msg #3414): writes go to VRAM scratch (prefill_moe_normed),
        // then a final multi-block D2D-copy stages to host-mapped UC activation_staging
        // with signal_ptr set so op_d2d_copy's producer-readback (megakernel_ops.hip:1747-1755)
        // drains GPU 0's PCIe-posted writes before workers P2P-read in Step 2.5.
        // Matches decode_step's working pattern: single D2D-copy of moe_act_uc_handoff
        // across all 96 blocks with implicit drain. The earlier rmsnorm-direct-to-staging
        // path (grid_x=1, single-block) didn't drain reliably for peer GPUs.
        let prefill_normed_dev = self.activations.prefill_moe_normed.as_mut_ptr();
        let mut insts: Vec<Instruction> = Vec::with_capacity(n * 5 + 1);
        for t in 0..n {
            let token_input = unsafe { prefill_hidden.as_ptr().add(t * hs) };
            let normed_slot = unsafe { prefill_normed_dev.add(t * latent_size) };
            let ids_slot = unsafe { per_token_ids_dev.add(t * MAX_ACTIVE_EXPERTS) };
            let wts_slot = unsafe { per_token_wts_dev.add(t * MAX_ACTIVE_EXPERTS) };

            if fc1_ptr.is_some() {
                insts.push(RmsNormInst::new(
                    rmsnorm_op, 1, normed_dev, token_input, norm_weight, hs as i32, eps,
                ).into_inst());
                insts.push(LinearProjInst::new(
                    fc1_op, latent_size as u32, normed_slot, fc1_data, normed_dev,
                    latent_size as i32, hs as i32, 0,
                ).into_inst());
            } else {
                insts.push(RmsNormInst::new(
                    rmsnorm_op, 1, normed_slot, token_input, norm_weight, hs as i32, eps,
                ).into_inst());
            }
            // gate_proj reads from VRAM scratch (same-GPU cached, fast).
            insts.push(LinearProjInst::new(
                gate_proj_op, ne as u32, scores_dev, gate_proj_data, normed_slot as *const f32,
                ne as i32, hs as i32, 0,
            ).into_inst());
            insts.push(MoeGateInst::new(
                scores_dev as *const f32, ids_slot, wts_slot,
                ne as i32, k as i32, gate_mode, rsf, bias_ptr,
            ).into_inst());
        }
        // Multi-block D2D-copy with .with_signal() — producer-readback drain forces
        // GPU 0's writes to host DRAM before workers fire (udi msg #3451). Sentinel
        // is VRAM DeviceBuffer<u32> (kernel patch 0001 + VRAM atomic works on gfx11);
        // signal_seq=1 constant (matches decode's compile_attention.rs:486 pattern).
        let drain_signal_ptr = self.moe_p2p.as_ref().unwrap().step1_drain_sentinel.as_ptr() as *mut u32;
        insts.push(
            D2dCopyInst::new(
                div_ceil((n * latent_size) as u32, 256),
                act_staging_dev,
                prefill_normed_dev as *const f32,
                (n * latent_size) as i32,
            )
            .with_signal(drain_signal_ptr, 1)
            .into_inst(),
        );
        // bd 9gmh Phase 1 (udi msg #3453): trailing NOP. Without an instruction after
        // the D2D-copy, dispatch_batch_slice's ack only proves the kernel returned, NOT
        // that the D2D's PCIe-bound writes have drained. The MES per-block barrier
        // between D2D and NOP forces the producer-readback drain to complete before
        // the kernel exits the batch.
        insts.push(NopInst {
            opcode_gridx: ((OP_NOP as u64) | (1u64 << 32)),
            dump_buf: std::ptr::null(),
            max_slots: 0,
            dump_counter: std::ptr::null(),
            _pad: [0; 14],
        }.into_inst());
        {
            let dispatch = self.persistent_workers.as_mut()
                .expect("persistent_workers not initialized");
            dispatch.dispatch_batch_slice(device_idx, &insts);
        }

        // bd 9gmh Phase 1: investigation point — multi-token MoE produces NaN logits
        // 5/5 with several drain-barrier attempts (volatile-read host barrier; SDMA D2H
        // sync; SYSTEM-scope ack at megakernel.hip:309). The DIAG eprintln "fix" worked
        // by accident of timing delay. Next session: re-examine whether the race is
        // really at Step 1/2.5 boundary or elsewhere; consider adding GPU-side diagnostic
        // (write a known constant to staging+output_slots tail end and check kernel-side
        // for corruption) and a single giant dispatch (vs per-token batches).

        // === STEP 2.5: per-token worker dispatch via OP_MOE_FFN_REMOTE ===
        // Workers P2P-read from activation_staging via their per-context device pointers.
        // Routing IDs/weights are passed via the per-token activations.moe_expert_ids/_weights
        // single-slot buffers (worker-readable; we copy this token's slice from the
        // per_token_* host-mapped buffers before each dispatch).
        let mut all_seqs: Vec<(usize, u32)> = Vec::with_capacity(num_workers);
        for t in 0..n {
            unsafe {
                let ids_dst = self.activations.moe_expert_ids.host_ptr() as *mut i32;
                let wts_dst = self.activations.moe_expert_weights.host_ptr() as *mut f32;
                let ids_src = per_token_ids_host.add(t * MAX_ACTIVE_EXPERTS);
                let wts_src = per_token_wts_host.add(t * MAX_ACTIVE_EXPERTS);
                for j in 0..k {
                    std::ptr::write_volatile(ids_dst.add(j), *ids_src.add(j));
                    std::ptr::write_volatile(wts_dst.add(j), *wts_src.add(j));
                }
            }
            let p2p = self.moe_p2p.as_ref().unwrap();
            let pd = self.persistent_workers.as_mut().unwrap();
            // bd 9gmh fix (2026-05-24): workers write to their own local_output
            // VRAM, NOT cross-fabric to GPU 0's output_slots. Worker kernel-
            // issued PCIe writes to GPU 0 trigger a gfx11 MMHUB/HDP hazard that
            // corrupts unrelated GPU 0 L2 lines (prefill_normed_dev,
            // expert_*_dev), causing NaN cascades even when GPU 0 never reads
            // the worker slots (per mes-researcher diagnosis + bd 9gmh probes
            // SUM_G0_ONLY=NaN, NO_STEP5+workers=NaN, NO_P2P_WRITE=coherent).
            // Host SDMA-async D2H below moves the data via a different traffic
            // class (SDMA, not compute-engine PCIe), avoiding the hazard.
            for w in 0..num_workers {
                let gpu_id = w + 1;
                let out_target = p2p.local_output_ptr_for(w);
                let act_p2p = unsafe { worker_act_bases[w].add(t * latent_size) as *const f32 };
                let inst = p2p.build_ffn_remote_inst(
                    w, layer_idx, act_p2p, out_target,
                    self.activations.moe_expert_ids.as_ptr() as *const i32,
                    self.activations.moe_expert_weights.as_ptr() as *const f32,
                    k, eis, hs, latent_size, has_gate, !has_gate,
                );
                let single = std::slice::from_ref(&inst);
                let seq = pd.dispatch_batch_fire(gpu_id, single);
                all_seqs.push((gpu_id, seq));
            }
            if let Err(e) = pd.try_wait_acks_many(&all_seqs) {
                panic!("{e}");
            }
            all_seqs.clear();

            // SDMA-async D2H: worker local_output (worker VRAM) → host-mapped
            // output_slots[t, w+1, ..]. Uses worker's SDMA stream (independent
            // of compute CUs held by worker's persistent_worker). Per-worker
            // stream sync blocks CPU only; doesn't need GPU 0 CUs.
            unsafe {
                let host_outs = p2p.output_slots.host_ptr();
                let bytes = hs * std::mem::size_of::<f32>();
                let mut streams: Vec<braidinfer_hip::ffi::hipStream_t> =
                    Vec::with_capacity(num_workers);
                for w in 0..num_workers {
                    let gpu_id = w + 1;
                    let device = p2p.workers[w].device;
                    pd.ensure_sdma_stream(device).expect("ensure_sdma_stream");
                    let stream = pd.sdma_stream(gpu_id);
                    let off = (t * num_gpus + gpu_id) * hs;
                    let dst = host_outs.add(off) as *mut std::ffi::c_void;
                    let src = p2p.local_output_ptr_for(w) as *const std::ffi::c_void;
                    let _guard = braidinfer_hip::device::DeviceGuard::switch_to(device)
                        .expect("DeviceGuard");
                    braidinfer_hip::error::check(braidinfer_hip::ffi::hipMemcpyAsync(
                        dst, src, bytes,
                        braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                        stream,
                    )).expect("hipMemcpyAsync D2H");
                    streams.push(stream);
                }
                for s in streams {
                    braidinfer_hip::error::check(braidinfer_hip::ffi::hipStreamSynchronize(s))
                        .expect("hipStreamSynchronize");
                }
            }

        }

        // === STEP 3: GPU 0 local expert compute via mailbox ===
        // Per-token program: zero ffn_down (via D2D copy from pre-zeroed buffer),
        // for each routed expert on GPU 0: gate/up projections + activation + down
        // projection + weighted accumulate, then D2D ffn_down → output_slots[(t*num_gpus+0)*hs].
        for t in 0..n {
            let mut insts: Vec<Instruction> = Vec::new();
            // Zero ffn_down (latent_size floats) via D2D copy from gpu0_zero_buffer.
            insts.push(D2dCopyInst::new(
                div_ceil(latent_size as u32, 256),
                ffn_down_dev, gpu0_zero_dev, latent_size as i32,
            ).into_inst());
            // bd 9gmh Phase 1: Step 3 reads from VRAM prefill_moe_normed (same-GPU,
            // local cached) instead of host-mapped activation_staging. Reading via
            // host-mapped UC went through GART/PCIe → potential L0 staleness across
            // layers; reading local VRAM uses standard same-GPU cache coherence which
            // works correctly. The host-mapped staging is still needed for workers
            // (Step 2.5 P2P read).
            let expert_input = unsafe { prefill_normed_dev.add(t * latent_size) as *const f32 };
            for j in 0..k {
                let eid = unsafe { *per_token_ids_host.add(t * MAX_ACTIVE_EXPERTS + j) } as usize;
                let ew = unsafe { *per_token_wts_host.add(t * MAX_ACTIVE_EXPERTS + j) };
                let slot = match slot_map_owned[eid] { Some(s) => s, None => continue };
                let gu_ptr = unsafe { gpu0_gate_up_base.add(slot * gate_up_expert_stride) };
                let dn_ptr = unsafe { gpu0_down_base.add(slot * down_expert_stride) };
                if has_gate {
                    let up_ptr = unsafe { gu_ptr.add(eis * gate_up_row_stride) };
                    insts.push(LinearProjInst::new(
                        expert_proj_op, eis as u32, expert_gate_dev, gu_ptr, expert_input,
                        eis as i32, latent_size as i32, 0,
                    ).into_inst());
                    insts.push(LinearProjInst::new(
                        expert_proj_op, eis as u32, expert_up_dev, up_ptr, expert_input,
                        eis as i32, latent_size as i32, 0,
                    ).into_inst());
                    insts.push(SiluMulInst::new(
                        div_ceil(eis as u32, 256),
                        expert_act_dev, expert_gate_dev as *const f32,
                        expert_up_dev as *const f32, eis as i32,
                    ).into_inst());
                } else {
                    insts.push(LinearProjInst::new(
                        expert_proj_op, eis as u32, expert_up_dev, gu_ptr, expert_input,
                        eis as i32, latent_size as i32, 0,
                    ).into_inst());
                    insts.push(ReluSqInst::new(
                        div_ceil(eis as u32, 256),
                        expert_act_dev, expert_up_dev as *const f32, eis as i32,
                    ).into_inst());
                }
                insts.push(LinearProjInst::new(
                    expert_proj_op, latent_size as u32, expert_out_dev, dn_ptr,
                    expert_act_dev as *const f32,
                    latent_size as i32, eis as i32, 0,
                ).into_inst());
                insts.push(ScaleAddInst::new(
                    div_ceil(latent_size as u32, 256),
                    ffn_down_dev, expert_out_dev as *const f32, ew, latent_size as i32,
                ).into_inst());
            }
            // D2D ffn_down → output_slots[(t * num_gpus + 0) * hs].
            let gpu0_out_slot = unsafe { gpu0_out_dev.add(t * num_gpus * hs) };
            insts.push(D2dCopyInst::new(
                div_ceil(latent_size as u32, 256),
                gpu0_out_slot, ffn_down_dev as *const f32, latent_size as i32,
            ).into_inst());

            let dispatch = self.persistent_workers.as_mut().unwrap();
            dispatch.dispatch_batch_slice(device_idx, &insts);
        }

        // === STEP 5: per-token sum + fc2 + shared expert + residual ===
        for t in 0..n {
            let mut insts: Vec<Instruction> = Vec::new();
            let token_input = unsafe { prefill_hidden.as_ptr().add(t * hs) };
            let token_output = unsafe { prefill_hidden.as_mut_ptr().add(t * hs) };

            // D2D hidden ← prefill_hidden[t*hs..]; D2D residual ← hidden.
            insts.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                hidden_dev, token_input, hs as i32,
            ).into_inst());
            insts.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                residual_dev, hidden_dev as *const f32, hs as i32,
            ).into_inst());
            // Zero ffn_down (hs floats).
            insts.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                ffn_down_dev, gpu0_zero_dev, hs as i32,
            ).into_inst());
            // Sum num_gpus output slots via expert_out as scratch + scale_add.
            for g in 0..num_gpus {
                let slot_offset = (t * num_gpus + g) * hs;
                let slot_src = unsafe { gpu0_out_dev.add(slot_offset) as *const f32 };
                insts.push(D2dCopyInst::new(
                    div_ceil(latent_size as u32, 256),
                    expert_out_dev, slot_src, latent_size as i32,
                ).into_inst());
                insts.push(ScaleAddInst::new(
                    div_ceil(latent_size as u32, 256),
                    ffn_down_dev, expert_out_dev as *const f32, 1.0, latent_size as i32,
                ).into_inst());
            }
            // fc2_latent_proj if present: ffn_down latent-sum → expert_out (D2D) → fc2 → ffn_down.
            if fc2_ptr.is_some() {
                insts.push(D2dCopyInst::new(
                    div_ceil(latent_size as u32, 256),
                    expert_out_dev, ffn_down_dev as *const f32, latent_size as i32,
                ).into_inst());
                insts.push(LinearProjInst::new(
                    fc2_op, hs as u32, ffn_down_dev, fc2_data, expert_out_dev as *const f32,
                    hs as i32, latent_size as i32, 0,
                ).into_inst());
            }
            // Shared expert (recompute normed from hidden).
            if let Some(se_ptr) = shared_expert_ptr {
                let se = unsafe { &*se_ptr };
                let (se_up_op, se_up_data) = linear_proj_opcode_ptr(&se.up_proj);
                let (se_down_op, se_down_data) = linear_proj_opcode_ptr(&se.down_proj);

                insts.push(RmsNormInst::new(
                    rmsnorm_op, 1, normed_dev, hidden_dev as *const f32, norm_weight, hs as i32, eps,
                ).into_inst());
                insts.push(LinearProjInst::new(
                    se_up_op, se_is as u32, expert_up_dev, se_up_data, normed_dev as *const f32,
                    se_is as i32, hs as i32, 0,
                ).into_inst());
                if has_gate {
                    let (se_gate_op, se_gate_data) = linear_proj_opcode_ptr(&se.gate_proj);
                    insts.push(LinearProjInst::new(
                        se_gate_op, se_is as u32, expert_gate_dev, se_gate_data,
                        normed_dev as *const f32,
                        se_is as i32, hs as i32, 0,
                    ).into_inst());
                    insts.push(SiluMulInst::new(
                        div_ceil(se_is as u32, 256),
                        expert_act_dev, expert_gate_dev as *const f32,
                        expert_up_dev as *const f32, se_is as i32,
                    ).into_inst());
                } else {
                    insts.push(ReluSqInst::new(
                        div_ceil(se_is as u32, 256),
                        expert_act_dev, expert_up_dev as *const f32, se_is as i32,
                    ).into_inst());
                }
                insts.push(LinearProjInst::new(
                    se_down_op, hs as u32, expert_out_dev, se_down_data,
                    expert_act_dev as *const f32,
                    hs as i32, se_is as i32, 0,
                ).into_inst());

                if let Some(seg_ptr) = shared_expert_gate_ptr {
                    // Fused OP_DOT_SIGMOID_SCALE_ADD: single-block kernel does
                    // dot + sigmoid + scale_add in one instruction with LDS broadcast
                    // of the sigmoid scale to all threads in the block. Replaces
                    // OP_LINEAR_PROJ(1×hs)+OP_SIGMOID_WEIGHTED_ADD pair which had
                    // cross-block L0 staleness on intermediate scratch[0].
                    let seg_data = unsafe { &*seg_ptr }.as_ptr();
                    insts.push(DotSigmoidScaleAddInst::new(
                        ffn_down_dev,
                        expert_out_dev as *const f32,
                        normed_dev as *const f32,
                        seg_data,
                        hs as i32,
                    ).into_inst());
                } else {
                    insts.push(ScaleAddInst::new(
                        div_ceil(hs as u32, 256),
                        ffn_down_dev, expert_out_dev as *const f32, 1.0, hs as i32,
                    ).into_inst());
                }
            }
            // Final residual add: hidden = residual + ffn_down; D2D back to prefill_hidden.
            insts.push(ResidualAddInst::new(
                div_ceil(hs as u32, 256),
                hidden_dev, residual_dev as *const f32, ffn_down_dev as *const f32, hs as i32,
            ).into_inst());
            insts.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                token_output, hidden_dev as *const f32, hs as i32,
            ).into_inst());

            let dispatch = self.persistent_workers.as_mut().unwrap();
            dispatch.dispatch_batch_slice(device_idx, &insts);
        }

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
