use braidinfer_hip::memory::DeviceBuffer;

use crate::megakernel::{MegakernelProgram, OP_HALT, SHARED_LPROJ_TOTAL};

use super::Model;
use super::ModelError;
use crate::config::*;
use crate::weights::*;
use crate::gpu_utils::d2d_copy_f32;

impl Model {
    /// Persistent worker decode using paged KV cache.
    /// On first call: compiles paged megakernel, initializes page allocator + sequence,
    /// then launches persistent worker.
    pub(super) fn decode_step_persistent(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile PAGED megakernel FIRST (needs GPU queries),
        // then launch persistent worker (occupies all SMs).
        if self.persistent_workers.is_none() {
            let max_chunks = self.max_paged_chunks();

            if self.megakernel_paged.is_none() {
                let mut mk = MegakernelProgram::compile_paged(self)?;
                mk.init_paged_buffers(max_chunks).map_err(ModelError::Hip)?;
                self.megakernel_paged = Some(mk);
            }

            // Patch LM head instruction to write to logits_mapped (host-mapped)
            // so CPU can read without hipMemcpy (which deadlocks the cooperative kernel).
            // This must be done whether megakernel_paged was just compiled or
            // pre-compiled by prefill_paged (which doesn't patch logits_mapped).
            {
                let mk = self.megakernel_paged.as_mut().unwrap();
                let n_inst = mk.instructions.len();
                let lm_head_idx = n_inst - 2; // second-to-last (before HALT)
                mk.instructions[lm_head_idx].words[1] =
                    self.activations.logits_mapped.as_write_ptr() as u64;
            }

            // Ensure paged decode state (page_allocator + paged_seq) is initialized.
            self.ensure_paged_decode_state(false)?;

            // PCG32 full kernel requires SHARED_LPROJ_TOTAL (31776B) for its LDS tile.
            let shared_mem = SHARED_LPROJ_TOTAL as u32;
            let dispatch =
                PersistentDispatch::init(&[self.device], shared_mem, self.config.hidden_size).map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
        }

        // Write position_ids directly to host-mapped memory (no hipMemcpy)
        self.set_position(position).map_err(ModelError::Hip)?;

        // Append token to paged sequence state (allocates chunk slot if needed).
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(position as i32, alloc_mut).map_err(ModelError::Hip)?;
        }

        // Host-side patching only: update instructions without hipMemcpyAsync.
        // Persistent caller will dispatch via dispatch_batch_slice instead.
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq = self.paged_seq.as_ref().unwrap();
            let allocator = self.page_allocator.as_ref().unwrap();
            mk.update_step_paged_no_upload(token_id, position, seq, allocator)
                .map_err(ModelError::Hip)?;
        }

        // Dispatch: send all instructions (excluding HALT) via persistent worker mailbox.
        // HALT EXCLUSION (CRITICAL): the persistent cooperative kernel loops forever
        // waiting for the next batch; HALT would cause it to exit. We must never
        // send HALT over the mailbox — only send to halt_idx (exclusive).
        let mk = self.megakernel_paged.as_ref().unwrap();
        let halt_idx = mk
            .instructions
            .iter()
            .position(|inst| (inst.words[0] as u32 as u64) == OP_HALT as u64)
            .unwrap_or(mk.instructions.len());
        let dispatch = self.persistent_workers.as_mut().unwrap();
        dispatch.dispatch_batch_slice(0, &mk.instructions[..halt_idx]);

        // Read logits directly from host-mapped memory (no hipMemcpy needed)
        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Multi-GPU persistent worker decode for MoE models.
    /// Initializes P2P workers on first call, then delegates to decode_step_p2p.
    pub(super) fn decode_step_persistent_multi_gpu(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        // Lazy-init: compile P2P megakernel + launch workers on ALL GPUs
        if self.persistent_workers.is_none() {
            self.init_multi_gpu_persistent()?;
        }

        // P2P megakernel is always initialized above when has_moe && num_gpus > 1.
        // For non-MoE multi-GPU models, fall through to decode_step_paged.
        if self.megakernel_multi_gpu_p2p.is_some() {
            return self.decode_step_p2p(token_id, position);
        }
        self.decode_step_paged(token_id, position)
    }

    /// Lazily start MoE expert workers (GPUs 1-3) without launching the GPU 0 decode
    /// persistent cooperative kernel. Safe to call during prefill (no cooperative kernel
    /// running on GPU 0 yet, so hipMalloc is allowed).
    pub(super) fn ensure_moe_workers_started(&mut self) -> Result<(), ModelError> {
        if self.moe_p2p.is_some() {
            return Ok(());
        }
        if !self.has_moe {
            return Ok(());
        }
        let num_gpus = match self.multi_gpu.as_ref() {
            Some(m) => m.num_devices,
            None => return Ok(()),
        };
        if num_gpus <= 1 {
            return Ok(());
        }
        let moe_worker_shared_mem = 1024u32 * 4 + 256;
        let hs = self.config.hidden_size;
        let max_eis = self
            .config
            .layers
            .iter()
            .filter_map(|l| match &l.ffn_type {
                crate::model::FfnType::MoE { expert_intermediate_size, .. } => {
                    Some(*expert_intermediate_size)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let worker_devices: Vec<_> = (1..num_gpus)
            .map(|i| braidinfer_core::types::DeviceId(i as u32))
            .collect();
        let num_total_layers = self.config.layers.len();
        let dist_refs: Vec<Option<&crate::weights::DistributedMoeWeights>> =
            self.distributed_moe.iter().map(|d| d.as_ref()).collect();
        let gate_up_in_dim = self.config.moe_latent_size.unwrap_or(hs);
        let p2p = crate::moe_p2p::MoeP2pContext::init(
            self.device,
            &worker_devices,
            hs,
            gate_up_in_dim,
            max_eis,
            num_total_layers,
            &dist_refs,
            moe_worker_shared_mem,
        )
        .map_err(ModelError::Hip)?;
        let mk_p2p = MegakernelProgram::compile_multi_gpu_p2p(self, &p2p)
            .map_err(ModelError::Hip)?;
        self.moe_p2p = Some(p2p);
        self.megakernel_multi_gpu_p2p = Some(mk_p2p);
        eprintln!("  MoE P2P dispatch initialized: {} worker GPUs (prefill path)", num_gpus - 1);
        Ok(())
    }

    fn init_multi_gpu_persistent(&mut self) -> Result<(), ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;

        let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
        let moe_worker_shared_mem = 1024u32 * 4 + 256;
        let shared_mem = moe_worker_shared_mem.max(SHARED_LPROJ_TOTAL);
        let hs = self.config.hidden_size;
        let max_eis = self
            .config
            .layers
            .iter()
            .filter_map(|l| match &l.ffn_type {
                crate::model::FfnType::MoE { expert_intermediate_size, .. } => {
                    Some(*expert_intermediate_size)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        // Init GPU-native P2P MoE dispatch BEFORE launching the persistent cooperative
        // worker on GPU 0. hipMalloc on GPU 0 deadlocks if the cooperative kernel is
        // already running. Launch persistent_worker LAST.
        self.ensure_moe_workers_started()?;

        // Launch persistent cooperative worker on GPU 0 LAST.
        let all_devices: Vec<_> = (0..num_gpus)
            .map(|i| braidinfer_core::types::DeviceId(i as u32))
            .collect();
        let dispatch = PersistentDispatch::init_multi_gpu(
            self.device,
            &all_devices,
            shared_mem,
            hs,
            max_eis,
        )
        .map_err(ModelError::Hip)?;
        self.persistent_workers = Some(dispatch);
        Ok(())
    }

    /// GPU-native P2P MoE decode: OP_MOE_DISPATCH handled entirely by megakernel.
    /// No CPU-side expert dispatching. Attention is still head-parallel (same as before).
    pub(super) fn decode_step_p2p(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.set_position(position).map_err(ModelError::Hip)?;

        // Update per-step state in p2p megakernel (embedding ptr, mRoPE positions, etc.)
        let mk = self.megakernel_multi_gpu_p2p.as_mut().unwrap();
        mk.update_step_host_only(token_id, position)?;
        let n_inst = mk.instructions.len();
        mk.instructions[n_inst - 2].words[1] = self.activations.logits_mapped.as_write_ptr() as u64;

        let _hs = self.config.hidden_size;
        let has_head_parallel = self
            .multi_gpu
            .as_ref()
            .map(|m| !m.workers[0].attn_kv_caches.is_empty())
            .unwrap_or(false);
        let use_distributed_qkv = has_head_parallel && {
            !self
                .megakernel_multi_gpu_p2p
                .as_ref()
                .unwrap()
                .multi_gpu_attn_boundaries
                .is_empty()
        };
        let attn_boundaries: Vec<(usize, usize)> = if has_head_parallel {
            let mk_ref = self.megakernel_multi_gpu_p2p.as_ref().unwrap();
            if use_distributed_qkv {
                mk_ref.multi_gpu_attn_boundaries.clone()
            } else {
                mk_ref
                    ._mrope_inst_indices
                    .iter()
                    .zip(mk_ref.gqa_attn_inst_indices.iter())
                    .map(|(&m, &g)| (m, g))
                    .collect()
            }
        } else {
            Vec::new()
        };
        let n_inst = self
            .megakernel_multi_gpu_p2p
            .as_ref()
            .unwrap()
            .instructions
            .len();

        let mut attn_i = 0usize;
        let mut i = 0usize;
        let mut seg_start = 0usize; // start of current segment in mk.instructions
        // Gather instructions returned from dispatch_head_parallel_attention, prepended
        // to the next segment to save one dispatch round-trip per attention layer.
        let mut pending_gather: Vec<crate::megakernel::Instruction> = Vec::new();

        while i < n_inst {
            let opcode = self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[i].words[0] as u32 as u64;
            if opcode == OP_HALT as u64 {
                break;
            }

            // Head-parallel attention boundary: flush segment, dispatch parallel QKV+GQA
            if has_head_parallel && attn_i < attn_boundaries.len() {
                let (flush_idx, resume_idx) = attn_boundaries[attn_i];
                if i == flush_idx {
                    // Include this instruction in the segment, then flush.
                    // Prepend any pending gather from previous attention layer.
                    let seg_end = i + 1;
                    {
                        let mk_insts = &self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[seg_start..seg_end];
                        if pending_gather.is_empty() {
                            self.persistent_workers.as_mut().unwrap().dispatch_batch_slice(0, mk_insts);
                        } else {
                            // Fuse pending gather with this segment: one combined dispatch
                            let mut combined = std::mem::take(&mut pending_gather);
                            combined.extend_from_slice(mk_insts);
                            self.persistent_workers.as_mut().unwrap().dispatch_batch_slice(0, &combined);
                        }
                    }
                    pending_gather = self.dispatch_head_parallel_attention(attn_i, position)?;
                    attn_i += 1;
                    i = if use_distributed_qkv { resume_idx } else { resume_idx + 1 };
                    seg_start = i;
                    continue;
                }
            }

            i += 1;
        }

        // Dispatch remaining segment, prepending any pending gather
        if seg_start < i || !pending_gather.is_empty() {
            let debug_hidden = self.debug_p2p_hidden;
            let mk_insts = &self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[seg_start..i];
            let mut batch_idx = 0usize;
            if pending_gather.is_empty() {
                for chunk in mk_insts.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                    self.persistent_workers.as_mut().unwrap().dispatch_batch(0, chunk);
                    if debug_hidden {
                        let src = self.activations.hidden.as_ptr() as *const u8;
                        let mut buf = [0u8; 8];
                        braidinfer_hip::memory::memcpy_d2h(&mut buf, src, 8)?;
                        let v0 = f32::from_ne_bytes([buf[0],buf[1],buf[2],buf[3]]);
                        let v1 = f32::from_ne_bytes([buf[4],buf[5],buf[6],buf[7]]);
                        eprintln!("DBG p2p batch {batch_idx}: h[0]={v0:.6} h[1]={v1:.6}");
                        batch_idx += 1;
                    }
                }
            } else {
                // Fuse pending gather with remaining segment
                let mut combined = pending_gather;
                combined.extend_from_slice(mk_insts);
                for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                    self.persistent_workers.as_mut().unwrap().dispatch_batch(0, chunk);
                    if debug_hidden {
                        let src = self.activations.hidden.as_ptr() as *const u8;
                        let mut buf = [0u8; 8];
                        braidinfer_hip::memory::memcpy_d2h(&mut buf, src, 8)?;
                        let v0 = f32::from_ne_bytes([buf[0],buf[1],buf[2],buf[3]]);
                        let v1 = f32::from_ne_bytes([buf[4],buf[5],buf[6],buf[7]]);
                        eprintln!("DBG p2p batch {batch_idx}: h[0]={v0:.6} h[1]={v1:.6}");
                        batch_idx += 1;
                    }
                }
            }
        }

        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();

        if self.trace.is_some() {
            // Multi-GPU trace: only top10_logits available. hidden/normed are in GPU VRAM
            // and inaccessible while the persistent cooperative worker holds all CUs.
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed
                .iter()
                .take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace.as_mut().unwrap().write_checkpoint("top10_logits", &top10);
        }

        // After first token: print worker timing report if MOE_TIMING env var is set.
        if self.seq_len == 0 && self.moe_timing {
            if let Some(ref p2p) = self.moe_p2p {
                p2p.print_timing_report(2500.0); // 7900XTX ~2500 MHz
            }
        }

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Head-parallel GQA attention. Two modes:
    ///
    /// **Distributed QKV** (use_distributed_qkv=true, triggered from RMSNorm boundary):
    ///   Each GPU receives normed (P2P broadcast from GPU 0) and projects its own Q/K/V slices.
    ///   After GQA, gate slices are collected to GPU 0 for output-gate (run later in megakernel).
    ///
    /// **Legacy** (use_distributed_qkv=false, triggered from mRoPE boundary):
    ///   GPU 0 has already projected Q/K/V. Only KV write + GQA are distributed.
    ///
    /// After this returns, activations.attn_out[0..nqh*hd] contains the concatenated GQA outputs,
    /// and (for distributed mode) activations.gate_attn[0..nqh*hd] contains the full gate.
    /// Head-parallel attention dispatch.
    /// Returns gather instructions (OP_D2D_COPY from GPUs 1+ to GPU 0) to be prepended
    /// to the next GPU 0 segment, saving one dispatch round-trip per attention layer.
    /// GPU 1+ streams are synchronized before return, so gather is safe to fuse.
    fn dispatch_head_parallel_attention(
        &mut self,
        attn_i: usize,
        position: u32,
    ) -> Result<Vec<crate::megakernel::Instruction>, ModelError> {
        use crate::megakernel::instructions::{
            D2dCopyInst, DeinterleaveInst, GqaAttnInst as GqaAttnInstLocal, LinearProjInst,
            MropeInst, QkNormInst,
        };
        use crate::megakernel::{
            Instruction, OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4,
        };
        use crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS;
        use crate::quant::{LinearWeight, WeightFormat};

        fn emit_linear_proj_inst(
            batch: &mut Vec<Instruction>,
            w: &LinearWeight,
            out_ptr: *mut f32,
            in_ptr: *const f32,
            out_dim: usize,
            in_dim: usize,
        ) {
            let (opcode, w_ptr) = match w.weight_format() {
                WeightFormat::PcG32Q4 => (OP_LINEAR_PROJ_PCG32, w.raw_data_ptr()),
                WeightFormat::Rnf4G128 => (OP_LINEAR_PROJ_RNF4, w.raw_data_ptr()),
                WeightFormat::Bf16 => (OP_LINEAR_PROJ, w.raw_data_ptr()),
            };
            batch.push(LinearProjInst::new(opcode, out_dim as u32, out_ptr, w_ptr, in_ptr, out_dim as i32, in_dim as i32, 0).into_inst());
        }

        let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
        let nqh = self.config.num_q_heads;
        let nkh = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let hs = self.config.hidden_size;
        let max_sl = self.config.max_seq_len;
        let local_nqh = nqh / num_gpus;
        let local_nkh = nkh; // GQA: KV heads replicated on every GPU, not split
        let head_stride = max_sl * hd;
        let q_mult = if self.config.has_output_gate { 2 } else { 1 };
        let has_gate = self.config.has_output_gate;
        let use_distributed_qkv = !self
            .megakernel_multi_gpu_p2p
            .as_ref()
            .unwrap()
            .multi_gpu_attn_boundaries
            .is_empty();

        // GPU 0 base pointers (P2P-accessible)
        // Use normed_stage (GART/MappedHostBuffer) for broadcast source, NOT normed (device VRAM).
        // On RDNA3 PCIe, P2P reads bypass GPU 0's L2 and hit VRAM — which may be stale since
        // op_rmsnorm_wx writes go through L2. normed_stage is write-through to system RAM,
        // so GPU 1-3's peer_copy_kernel reads the correct value.
        let normed_base = self.activations.normed_stage.device_ptr() as u64;
        let k_attn_base = self.activations.k_attn.as_ptr() as u64;
        let v_attn_base = self.activations.v_attn.as_ptr() as u64;
        let q_attn_base = self.activations.q_attn.as_ptr() as u64;
        let attn_out_base = self.activations.attn_out.as_write_ptr() as u64;
        let gate_attn_base = self.activations.gate_attn.as_write_ptr() as u64;

        let mut seq_nums: Vec<(usize, u32)> = Vec::with_capacity(num_gpus);

        for gpu_i in 0..num_gpus {
            let mut batch: Vec<Instruction> = Vec::new();

            // Resolve per-GPU buffer pointers
            let (kv_k_base, kv_v_base, q_ptr, out_ptr) = {
                let mgpu = self.multi_gpu.as_ref().unwrap();
                let kc = &mgpu.workers[gpu_i].attn_kv_caches[attn_i];
                let q = if gpu_i == 0 {
                    q_attn_base
                } else {
                    mgpu.workers[gpu_i].attn_q.as_ref().unwrap().as_ptr() as u64
                };
                let out = if gpu_i == 0 {
                    attn_out_base
                } else {
                    mgpu.workers[gpu_i]
                        .attn_out
                        .as_ref()
                        .unwrap()
                        .as_write_ptr() as u64
                };
                (
                    kc.k.as_write_ptr() as u64,
                    kc.v.as_write_ptr() as u64,
                    q,
                    out,
                )
            };

            if use_distributed_qkv {
                // ── Distributed QKV mode ──────────────────────────────────────────────────
                // GPU 0 has normed in act.normed (from megakernel RMSNorm).
                // GPUs 1..n need normed via P2P broadcast.
                // Get this attention layer's weights (GPU 0 VRAM, P2P-accessible from all GPUs).
                let layer_idx_for_attn = self
                    .config
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.layer_type == crate::config::LayerType::Attention)
                    .nth(attn_i)
                    .map(|(i, _)| i)
                    .unwrap();

                let (normed_local, q_gate_ptr, k_local_ptr, v_local_ptr, gate_ptr) = {
                    let mgpu = self.multi_gpu.as_ref().unwrap();
                    if gpu_i == 0 {
                        let q_gate = self.activations.q_gate_attn.as_write_ptr() as u64;
                        let k = k_attn_base;
                        let v = v_attn_base;
                        let gate = gate_attn_base;
                        (normed_base, q_gate, k, v, gate)
                    } else {
                        let w = &mgpu.workers[gpu_i];
                        let normed = w.attn_normed.as_ref().unwrap().as_write_ptr() as u64;
                        let q_gate = w.attn_q_gate.as_ref().unwrap().as_write_ptr() as u64;
                        let k = w.attn_k.as_ref().unwrap().as_write_ptr() as u64;
                        let v = w.attn_v.as_ref().unwrap().as_write_ptr() as u64;
                        let gate = w.attn_gate.as_ref()
                            .map(|b| b.as_write_ptr() as u64)
                            .unwrap_or(0);
                        (normed, q_gate, k, v, gate)
                    }
                };

                // 0. Broadcast normed to GPUs 1..n
                if gpu_i > 0 {
                    batch.push(D2dCopyInst::new((hs as u32 + 255) / 256, normed_local as *mut f32, normed_base as *const f32, hs as i32).into_inst());
                }

                // 1-3. QKV projections.
                // GPU 0: use original layer weights (no copy, row_start=0).
                // GPUs 1+: use pre-copied slice in ctx.workers[gpu_i].attn_w_*.
                if gpu_i == 0 {
                    let aw = match &self.layers[layer_idx_for_attn] {
                        LayerWeights::Attention(w) => w,
                        _ => panic!("expected attention layer"),
                    };
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_q_gate,
                        q_gate_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nqh * hd * q_mult,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_k,
                        k_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_v,
                        v_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                } else {
                    let w_q = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_q_gate[attn_i]
                            as *const LinearWeight)
                    };
                    let w_k = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_k[attn_i]
                            as *const LinearWeight)
                    };
                    let w_v = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_v[attn_i]
                            as *const LinearWeight)
                    };
                    emit_linear_proj_inst(
                        &mut batch,
                        w_q,
                        q_gate_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nqh * hd * q_mult,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        w_k,
                        k_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        w_v,
                        v_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                }

                // 4. Deinterleave Q+gate → q_attn, gate_attn (only for gated Q)
                if has_gate {
                    let total = local_nqh * hd;
                    batch.push(DeinterleaveInst::new((total as u32 + 255) / 256, q_ptr as *mut f32, gate_ptr as *mut f32, q_gate_ptr as *const f32, local_nqh as i32, hd as i32, 1).into_inst());
                } else {
                    // Non-gated: q_gate IS q, just copy
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, q_gate_ptr as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // 5. QK-norm on local k (only for models with QK-norm weights — e.g. Qwen3, not Nemotron-H)
                if self.config.has_qk_norm {
                    let (q_norm_ptr, k_norm_ptr, qk_norm_eps) = {
                        match &self.layers[layer_idx_for_attn] {
                            LayerWeights::Attention(w) => (
                                w.q_norm.as_ptr(),
                                w.k_norm.as_ptr(),
                                self.config.rms_norm_eps,
                            ),
                            _ => panic!("expected attention layer"),
                        }
                    };
                    batch.push(QkNormInst::new((local_nqh + local_nkh) as u32, q_ptr as *mut f32, k_local_ptr as *mut f32, q_norm_ptr, k_norm_ptr, local_nqh as i32, local_nkh as i32, hd as i32, qk_norm_eps, 0).into_inst());
                }

                // 6. KV write (local — from local k/v to local KV cache)
                for h_local in 0..local_nkh {
                    let src_k = k_local_ptr + (h_local * hd * 4) as u64;
                    let src_v = v_local_ptr + (h_local * hd * 4) as u64;
                    let dst_k =
                        kv_k_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let dst_v =
                        kv_v_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_k as *mut f32, src_k as *const f32, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_v as *mut f32, src_v as *const f32, hd as i32).into_inst());
                }

                // 7. mRoPE on local Q+K — only for models that use RoPE
                if self.config.use_rope {
                    let rd = self.config.rope_dim;
                    let ms = self.config.mrope_sections();
                    batch.push(MropeInst::new((local_nqh + local_nkh) as u32, q_ptr as *mut f32, k_local_ptr as *mut f32, self.activations.inv_freq.as_ptr(), self.activations.position_ids.as_ptr(), local_nqh as i32, local_nkh as i32, hd as i32, rd as i32, ms[0] as i32, ms[1] as i32, ms[2] as i32, 0).into_inst());
                }

                // 8. GQA (same as legacy path)
                let seq_len = (position + 1) as i32;
                {
                    let mut inst = GqaAttnInstLocal::new(local_nqh as u32, out_ptr as *mut f32, q_ptr as *const f32, kv_k_base as *const f32, kv_v_base as *const f32, nqh as i32, nkh as i32, hd as i32, seq_len, max_sl as i32);
                    inst.q_head_start = (gpu_i * local_nqh) as u64;
                    batch.push(inst.into_inst());
                }
            } else {
                // ── Legacy mode (QKV already projected on GPU 0) ─────────────────────────
                // KV write: per KV head, from GPU 0's k/v_attn to this GPU's KV cache
                for h_local in 0..local_nkh {
                    let h_global = gpu_i * local_nkh + h_local;
                    let src_k = k_attn_base + (h_global * hd * 4) as u64;
                    let src_v = v_attn_base + (h_global * hd * 4) as u64;
                    let dst_k =
                        kv_k_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let dst_v =
                        kv_v_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_k as *mut f32, src_k as *const f32, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_v as *mut f32, src_v as *const f32, hd as i32).into_inst());
                }

                // For GPU i > 0: copy Q slice from GPU 0's q_attn to local attn_q
                if gpu_i > 0 {
                    let src_q = q_attn_base + (gpu_i * local_nqh * hd * 4) as u64;
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, src_q as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // GQA attention
                let seq_len = (position + 1) as i32;
                let q_src = if gpu_i == 0 { q_attn_base } else { q_ptr };
                {
                    let mut inst = GqaAttnInstLocal::new(local_nqh as u32, out_ptr as *mut f32, q_src as *const f32, kv_k_base as *const f32, kv_v_base as *const f32, nqh as i32, nkh as i32, hd as i32, seq_len, max_sl as i32);
                    inst.q_head_start = (gpu_i * local_nqh) as u64;
                    batch.push(inst.into_inst());
                }
            }

            // GPU 0: dispatch via persistent worker. GPUs 1+: kbk on compute_stream.
            if gpu_i == 0 {
                assert!(
                    batch.len() <= MAX_BATCH_INSTRUCTIONS,
                    "attn batch overflow gpu=0 len={}",
                    batch.len()
                );
                let seq = self
                    .persistent_workers
                    .as_mut()
                    .unwrap()
                    .dispatch_batch_fire(0, &batch);
                seq_nums.push((0, seq));
            } else {
                self.dispatch_attn_kbk(gpu_i, attn_i, position, &batch)
                    .map_err(ModelError::Hip)?;
            }
        }

        // Wait for GPU 0 persistent worker to complete
        for &(gpu_i, seq) in &seq_nums {
            self.persistent_workers
                .as_ref()
                .unwrap()
                .wait_ack(gpu_i, seq);
        }
        // Wait for GPUs 1+ compute_streams to complete
        for gpu_i in 1..num_gpus {
            braidinfer_hip::device::Device::set_current(braidinfer_core::types::DeviceId(
                gpu_i as u32,
            ))
            .map_err(ModelError::Hip)?;
            self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                .compute_stream
                .synchronize()
                .map_err(ModelError::Hip)?;
        }
        // Reset to GPU 0 for gather
        braidinfer_hip::device::Device::set_current(braidinfer_core::types::DeviceId(0))
            .map_err(ModelError::Hip)?;

        // Gather GPU 1..num_gpus attn_out + gate_attn via persistent worker OP_D2D_COPY.
        // MUST NOT use peer_copy_async (kernel launch on GPU 0) while persistent cooperative
        // worker holds all CUs. Route all GPU-0 copies through persistent worker protocol.
        // These instructions are returned to the caller to be prepended to the next segment,
        // fusing them into one dispatch round-trip (safe: GPU 1+ streams already synchronized).
        let mut gather_batch: Vec<Instruction> = Vec::new();
        let n_elems = local_nqh * hd;
        let grid_x = ((n_elems as u32) + 255) / 256;

        // attn_out gather: GPU i → act.attn_out[i*n_elems..]
        for gpu_i in 1..num_gpus {
            let src = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                .attn_out
                .as_ref()
                .unwrap()
                .as_ptr() as *const f32;
            let dst =
                unsafe { (self.activations.attn_out.as_write_ptr()).add(gpu_i * n_elems) };
            gather_batch.push(D2dCopyInst::new(grid_x, dst, src, n_elems as i32).into_inst());
        }

        // gate_attn gather: GPU i → act.gate_attn[i*n_elems..]
        // GPU 0's gate was written directly to act.gate_attn[0..n_elems] by deinterleave.
        if use_distributed_qkv && has_gate {
            for gpu_i in 1..num_gpus {
                let src = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                    .attn_gate
                    .as_ref()
                    .unwrap()
                    .as_ptr() as *const f32;
                let dst = unsafe {
                    self.activations
                        .gate_attn
                        .as_write_ptr()
                        .add(gpu_i * n_elems)
                };
                gather_batch.push(D2dCopyInst::new(grid_x, dst, src, n_elems as i32).into_inst());
            }
        }

        assert!(
            gather_batch.len() <= MAX_BATCH_INSTRUCTIONS,
            "gather batch overflow len={}",
            gather_batch.len()
        );

        Ok(gather_batch)
    }

    /// Dispatch a batch of attention instructions via kbk on GPU i's compute_stream.
    /// Used for GPUs 1+ where persistent cooperative workers cannot coexist with MoE kbk.
    fn dispatch_attn_kbk(
        &mut self,
        gpu_i: usize,
        _attn_i: usize,
        _position: u32,
        batch: &[crate::megakernel::Instruction],
    ) -> braidinfer_hip::HipResult<()> {
        use crate::megakernel::{OP_D2D_COPY, OP_DEINTERLEAVE, OP_GQA_ATTN, OP_MROPE, OP_QK_NORM};
        use crate::megakernel::{OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4};
        use crate::moe_dispatch::dispatch_proj;
        use crate::quant::WeightFormat;
        use braidinfer_core::types::DeviceId;
        use braidinfer_hip::device::Device;

        Device::set_current(DeviceId(gpu_i as u32))?;

        let stream = unsafe {
            &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].compute_stream
                as *const braidinfer_hip::stream::Stream)
        };

        for inst in batch {
            let opcode = inst.words[0] as u32;
            match opcode {
                OP_D2D_COPY => {
                    let dst = inst.words[1] as *mut u8;
                    let src = inst.words[2] as *const u8;
                    // Recover original element count from inst word 3 (set_int(3, n))
                    let n_elems = inst.words[3] as usize;
                    let size = n_elems * 4; // f32 bytes
                    crate::multi_gpu::MultiGpuContext::peer_copy_async(
                        dst,
                        src,
                        size,
                        &self.multi_gpu.as_ref().unwrap().workers[gpu_i].peer_copy_module,
                        stream,
                    )?;
                }
                OP_LINEAR_PROJ | OP_LINEAR_PROJ_PCG32 | OP_LINEAR_PROJ_RNF4 => {
                    let out = inst.words[1] as *mut f32;
                    let w_bytes = inst.words[2] as *const u8;
                    let inp = inst.words[3] as *const f32;
                    let out_dim = inst.words[4] as u32;
                    let in_dim = inst.words[5] as u32;
                    let fmt = match opcode {
                        OP_LINEAR_PROJ_PCG32 => WeightFormat::PcG32Q4,
                        OP_LINEAR_PROJ_RNF4 => WeightFormat::Rnf4G128,
                        _ => WeightFormat::Bf16,
                    };
                    let kernel = &self.worker_kernels[gpu_i].linear_proj;
                    dispatch_proj(kernel, out, w_bytes, inp, out_dim, in_dim, fmt, stream)?;
                }
                OP_DEINTERLEAVE => {
                    let dst_q = inst.words[1] as *mut f32;
                    let dst_gate = inst.words[2] as *mut f32;
                    let src = inst.words[3] as *const f32;
                    let num_heads = inst.words[4] as u32;
                    let head_dim = inst.words[5] as u32;
                    self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                        .deinterleave_kernel
                        .forward_ptr(dst_q, dst_gate, src, num_heads, head_dim, stream)?;
                }
                OP_QK_NORM => {
                    let q = inst.words[1] as *mut f32;
                    let k = inst.words[2] as *mut f32;
                    let q_norm = inst.words[3] as *const u16;
                    let k_norm = inst.words[4] as *const u16;
                    let nqh = inst.words[5] as u32;
                    let nkh = inst.words[6] as u32;
                    let hd = inst.words[7] as u32;
                    let eps = f32::from_bits(inst.words[8] as u32);
                    self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                        .qk_norm_kernel
                        .forward_ptr(q, k, q_norm, k_norm, nqh, nkh, hd, eps, stream)?;
                }
                OP_MROPE => {
                    let q = inst.words[1] as *mut f32;
                    let k = inst.words[2] as *mut f32;
                    let inv_freq = inst.words[3] as *const f32;
                    let pos_ids = inst.words[4] as *const i32;
                    let nqh = inst.words[5] as u32;
                    let nkh = inst.words[6] as u32;
                    let hd = inst.words[7] as u32;
                    let rd = inst.words[8] as u32;
                    let s0 = inst.words[9] as u32;
                    let s1 = inst.words[10] as u32;
                    let s2 = inst.words[11] as u32;
                    self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                        .mrope_kernel
                        .forward_ptr(
                            q, k, inv_freq, pos_ids, nqh, nkh, hd, rd, s0, s1, s2, stream,
                        )?;
                }
                OP_GQA_ATTN => {
                    let out = inst.words[1] as *mut f32;
                    let q = inst.words[2] as *const f32;
                    let k_cache = inst.words[3] as *const f32;
                    let v_cache = inst.words[4] as *const f32;
                    let nqh = inst.words[5] as u32;
                    let nkh = inst.words[6] as u32;
                    let hd = inst.words[7] as u32;
                    let sl = inst.words[8] as u32;
                    let msl = inst.words[9] as u32;
                    let q_head_start = inst.words[10] as u32;
                    let local_nqh = (inst.words[0] >> 32) as u32; // gupd = block count
                    self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                        .gqa_kernel
                        .forward_ptr(out, q, k_cache, v_cache, nqh, nkh, hd, sl, msl, local_nqh, q_head_start, stream)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Single-GPU per-layer decode with activation trace checkpoints.
    /// Only reachable when trace.is_some() && multi_gpu.is_none().
    pub(super) fn decode_step_trace(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;
        let sync_debug = self.sync_debug;
        if self
            .config
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Attention)
        {
            self.append_paged_decode_token(position)?;
        }

        macro_rules! sync_check_moe {
            ($label:expr) => {
                if sync_debug {
                    if let Err(e) = self.stream.synchronize() {
                        eprintln!("SYNC_DEBUG: crash at pos={}.{}", position, $label);
                        return Err(e.into());
                    }
                    eprintln!("SYNC_DEBUG: pos={}.{} OK", position, $label);
                }
            };
        }

        // Set position_ids for mRoPE/RoPE
        self.set_position(position).map_err(ModelError::Hip)?;

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        sync_check_moe!("embed");

        if self.debug_nan {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "embed(tok={token_id}): max_abs={max_abs:.4e}, first5={:.4?}",
                &buf[..5]
            );
        }

        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("embed", &buf);
        }

        // Process each layer
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        let mut mamba2_idx = 0usize;
        for layer_i in 0..self.config.num_layers {
            match self.config.layers[layer_i].layer_type {
                LayerType::Attention => {
                    self.attention_forward(layer_i, kv_idx, position)?;
                    sync_check_moe!(format!("L{layer_i}.attn"));
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    self.gdn_forward(layer_i, gdn_idx)?;
                    sync_check_moe!(format!("L{layer_i}.gdn"));
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    self.mamba2_forward(layer_i, mamba2_idx)?;
                    sync_check_moe!(format!("L{layer_i}.mamba2"));
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Standalone MoE FFN layer — just norm + MoE dispatch + residual
                    // The norm is applied inside moe_ffn_forward, skip to FFN below
                }
                LayerType::LfmConv => panic!("LfmConv not yet implemented"),
            }

            // Debug: check for NaN in hidden state after each layer
            if self.debug_nan {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                let nan_count = buf.iter().filter(|x| x.is_nan()).count();
                let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "L{layer_i} ({:?}): {nan_count} NaN, max_abs={max_abs:.2e}",
                    self.config.layers[layer_i].layer_type
                );
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace
                    .as_mut()
                    .unwrap()
                    .write_checkpoint(&format!("L{layer_i}.post_mixer"), &buf);
            }

            // FFN: dense, MoE, or None (standalone layers like Nemotron M/*)
            if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. }) {
                self.moe_ffn_forward(layer_i)?;
            } else if matches!(self.config.layers[layer_i].ffn_type, FfnType::None) {
                // No FFN for this layer (Nemotron M and * layers)
            } else {
                // Dense FFN: fused (bf16) or unfused (quantized)
                let hs = self.config.hidden_size;
                let is = self.config.intermediate_size;
                let eps = self.config.rms_norm_eps;

                // SAFETY: Raw pointers break borrow on self.layers for mutable self.activations.
                let (post_norm_p, w_gate_p, w_up_p, w_down_p) = match &self.layers[layer_i] {
                    LayerWeights::Attention(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    LayerWeights::Gdn(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    _ => panic!("dense FFN only for Attention/Gdn layers"),
                };

                let all_bf16 = unsafe {
                    matches!(&*w_gate_p, LinearWeight::Bf16(_))
                        && matches!(&*w_up_p, LinearWeight::Bf16(_))
                        && matches!(&*w_down_p, LinearWeight::Bf16(_))
                };

                if all_bf16 {
                    unsafe {
                        self.ffn_forward(
                            &*post_norm_p,
                            (*w_gate_p).as_bf16(),
                            (*w_up_p).as_bf16(),
                            (*w_down_p).as_bf16(),
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_bf16"));
                } else {
                    // Unfused path for quantized weights
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
                    unsafe {
                        self.kernels.rmsnorm.forward(
                            &mut self.activations.normed,
                            &self.activations.hidden,
                            &*post_norm_p,
                            1,
                            hs as u32,
                            eps,
                            self.config.rms_norm_one_plus_w,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_norm"));
                    unsafe {
                        (*w_gate_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_gate,
                            &self.activations.normed,
                            is as u32,
                            hs as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_gate"));
                    unsafe {
                        (*w_up_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_up,
                            &self.activations.normed,
                            is as u32,
                            hs as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_up"));
                    self.kernels.silu_mul.forward(
                        &mut self.activations.ffn_act,
                        &self.activations.ffn_gate,
                        &self.activations.ffn_up,
                        is as u32,
                        &self.stream,
                    )?;
                    sync_check_moe!(format!("L{layer_i}.ffn_silu"));
                    unsafe {
                        (*w_down_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_down,
                            &self.activations.ffn_act,
                            hs as u32,
                            is as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_down"));
                    self.kernels.residual_add.forward(
                        &mut self.activations.hidden,
                        &self.activations.ffn_down,
                        &self.activations.residual,
                        hs as u32,
                        &self.stream,
                    )?;
                }
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace
                    .as_mut()
                    .unwrap()
                    .write_checkpoint(&format!("L{layer_i}.post_ffn"), &buf);
            }
        }

        // Final RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &self.final_norm_weight,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;

        // LM head
        let lm_head_w = if self.config.tie_word_embeddings {
            &self.embed_weight
        } else {
            &self.lm_head_weight
        };
        self.kernels.linear_proj.forward(
            &mut self.activations.logits,
            lm_head_w,
            &self.activations.normed,
            self.config.vocab_size as u32,
            hs,
            &self.stream,
        )?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        if self.trace.is_some() {
            let mut hid_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut hid_buf)?;
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("final_hidden", &hid_buf);

            let mut norm_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.normed.copy_to_host(&mut norm_buf)?;
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("final_norm", &norm_buf);

            // Capture top-10 logits (token_id + value pairs as f32)
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed
                .iter()
                .take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("top10_logits", &top10);
        }

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, false)
    }

    /// Run a single decode step with quantized KV cache (4-bit residual_pc).
    /// Sealed chunks are quantized to int4; active chunk stays f32.
    pub fn decode_step_paged_quantized(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, true)
    }

    fn decode_step_paged_inner(
        &mut self,
        token_id: u32,
        position: u32,
        quantized: bool,
    ) -> Result<Vec<f32>, ModelError> {
        let max_chunks = self.max_paged_chunks();

        // Lazy-init: compile paged megakernel
        if self.megakernel_paged.is_none() {
            let mut mk = MegakernelProgram::compile_paged(self)?;
            mk.init_paged_buffers(max_chunks)?;
            if quantized {
                mk.enable_quantized_kv(max_chunks, &self.config)?;
            }
            self.megakernel_paged = Some(mk);
        } else {
            let mk = self.megakernel_paged.as_ref().unwrap();
            assert_eq!(
                mk.quantized_kv, quantized,
                "cannot mix decode_step_paged and decode_step_paged_quantized on the same model"
            );
        }

        // Lazy-init: f32 PageAllocator (staging) and SequenceState
        self.ensure_paged_decode_state(quantized)?;

        // append_token
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(position as i32, alloc_mut)?;
        }

        // Write position_ids to host-mapped memory before paged step (no hipMemcpy).
        self.set_position(position).map_err(ModelError::Hip)?;

        let stream = &self.stream;
        let mk = self.megakernel_paged.as_mut().unwrap();
        let seq = self.paged_seq.as_ref().unwrap();
        let allocator = self.page_allocator.as_ref().unwrap();

        mk.update_step_paged(token_id, position, seq, allocator, stream)?;
        mk.execute(stream)?;
        stream.synchronize()?;

        // Post-step: handle chunk seal + quantization
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let q_alloc = self.quant_allocator.as_mut();
            mk.post_step_paged(
                position,
                seq_mut,
                alloc_mut,
                q_alloc,
                &self.config,
                &self.stream,
            )?;
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }
}
