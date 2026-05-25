use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;

use super::Model;
use crate::config::*;
use crate::weights::*;
use crate::gpu_utils::d2d_copy_f32;
use crate::moe_p2p::{MAX_ACTIVE_EXPERTS, MAX_PREFILL_BATCH};

impl Model {
    /// Uses individual kernel launches (no megakernel).
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

        // bd i7gl: GPU 0 expert iteration (gate_up_base, down_base, slot_map,
        // expert_proj_op, weight_format, strides) is no longer destructured here
        // — Step 3 dispatches OP_MOE_FFN_REMOTE which reads the per-layer
        // MoeWorkerConfig from VRAM (built at MoeP2pContext::init from the same
        // DistributedMoeWeights). Single source of truth, no Rust-side mirror.

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
            let act_staging_dev = p2p.activation_staging.dev_ptr(0);
            let per_token_ids_host = p2p.per_token_expert_ids.host_ptr();
            let per_token_wts_host = p2p.per_token_expert_weights.host_ptr();
            let per_token_ids_dev = p2p.per_token_expert_ids.as_mut_ptr();
            let per_token_wts_dev = p2p.per_token_expert_weights.as_mut_ptr();
            let gpu0_zero_dev = p2p.gpu0_zero_buffer.as_ptr();
            let gpu0_out_dev = p2p.output_slots.dev_ptr(0);
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
        // GPU 0's writes to host DRAM before workers fire. Sentinel is VRAM
        // DeviceBuffer<u32> (kernel patch 0001 + VRAM atomic works on gfx11).
        // bd el1f: monotonic signal_seq = layer_idx + 1. Single sentinel is
        // overwritten in-place each layer; if the seq were constant=1 and layer 0
        // wrote 1, layer 1's workers would acquire-spin on seq=1 and skip the wait
        // (sentinel still 1 from prior layer) — reading stale activation_staging.
        // Monotonic + zero-init guarantees workers wait for their layer's Step 1.
        let drain_signal_ptr = self.moe_p2p.as_ref().unwrap().step1_drain_sentinel.as_ptr() as *mut u32;
        let drain_signal_seq = (layer_idx as u32) + 1;
        insts.push(
            D2dCopyInst::new(
                div_ceil((n * latent_size) as u32, 256),
                act_staging_dev,
                prefill_normed_dev as *const f32,
                (n * latent_size) as i32,
            )
            .with_signal(drain_signal_ptr, drain_signal_seq)
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

        // bd el1f: Step 1 → Step 2.5 drain mechanism empirically validated
        // by a 200x statistical regression gate (100x -g 1 + 100x -g 4 on
        // qwen35_35b_a3b multi-token prefill, text-identical within each
        // config, zero NaN). The producer-readback drain in op_d2d_copy
        // (megakernel_ops.hip:1772-1793) — single-thread volatile load of
        // host-mapped UC dst[0] forcing PCIe drain, then SYSTEM-scope ack
        // — combined with the CPU-dispatch latency between GPU 0's ack and
        // workers' kernel entry is sufficient on gfx11 + linux-p2p kernel
        // patches 0001/0012/0013/0016/0017/0018 + hsa-rocr-p2p-mtype-uc.
        //
        // The active-acquire approach (op_moe_ffn_remote.wait_ptr/seq) was
        // explored under el1f Phase A and found incompatible with gfx11's
        // peer-VRAM L2 coherence model: GPU 0's atomic_store to its own
        // VRAM sentinel sits in GPU 0's L2; peers' MTYPE_UC reads pull
        // through PCIe to memory controller and miss the L2-cached write.
        // Host-mapped UC sentinels wedge the SYSTEM-scope atomic
        // (moe_p2p.rs:341 — prior bd 9gmh finding). The wait_ptr/wait_seq
        // fields on MoeFfnRemoteInst are wired through but pass null in
        // production. Step 2.5 dispatch (below) reaffirms this with a
        // _ = drain_signal_{ptr,seq} bind.

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
                // bd el1f: wait_ptr infrastructure is in place but PASS NULL.
                // The acquire-on-VRAM-sentinel pattern (Phase A as designed)
                // deadlocks: GPU 0's L2 holds the producer atomic_store; peer
                // GPUs' MTYPE_UC reads pull through PCIe to memory controller,
                // missing the L2-cached write. moe_p2p.rs:341 documents that
                // host-mapped UC sentinels wedge the SYSTEM-scope atomic, so
                // moving the sentinel doesn't help either. The drain therefore
                // relies on implicit CPU-dispatch-latency timing (current
                // production approach). Phase B will VERIFY empirically that
                // this timing is sufficient; if VERIFY fails, the fix is a
                // new sentinel medium — separate epic.
                let _ = drain_signal_ptr;
                let _ = drain_signal_seq;
                let inst = p2p.build_ffn_remote_inst(
                    w, layer_idx, act_p2p, out_target,
                    self.activations.moe_expert_ids.as_ptr() as *const i32,
                    self.activations.moe_expert_weights.as_ptr() as *const f32,
                    k, eis, hs, latent_size, has_gate, !has_gate,
                    std::ptr::null(),
                    0,
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

        // === STEP 3: GPU 0 local expert compute via OP_MOE_FFN_REMOTE ===
        // bd i7gl: unified with the peer-worker compute path. GPU 0 dispatches
        // OP_MOE_FFN_REMOTE on its own persistent_worker with self-pointing
        // activation_p2p (prefill_normed_dev) and output_slot_p2p (host-mapped
        // output_slots[t,0]). The kernel (op_moe_ffn_remote, megakernel_moe.hip)
        // iterates the k routed experts internally and skips any whose
        // MoeWorkerConfig.entries[eid].gate_up_ptr is null (= not local to this
        // GPU). On GPU 0 the layer's MoeWorkerConfig has all experts marked
        // local (populated by bd 174k's distribute_moe_weights_from_ref with
        // num_devices=1), so the in-kernel skip is a no-op.
        for t in 0..n {
            let expert_input = unsafe { prefill_normed_dev.add(t * latent_size) as *const f32 };
            let gpu0_out_slot = unsafe { gpu0_out_dev.add(t * num_gpus * hs) };
            let ids = unsafe { per_token_ids_dev.add(t * MAX_ACTIVE_EXPERTS) as *const i32 };
            let wts = unsafe { per_token_wts_dev.add(t * MAX_ACTIVE_EXPERTS) as *const f32 };
            let p2p = self.moe_p2p.as_ref().unwrap();
            let inst = p2p.build_ffn_remote_inst_gpu0(
                layer_idx, expert_input, gpu0_out_slot, ids, wts,
                k, eis, hs, latent_size, has_gate, !has_gate,
                std::ptr::null(),
                0,
            );
            let insts = std::slice::from_ref(&inst);
            let dispatch = self.persistent_workers.as_mut().unwrap();
            dispatch.dispatch_batch_slice(device_idx, insts);
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

}
