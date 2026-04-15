use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;

use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram};
use crate::paged_kv::{self, PageAllocator, RecurrentCheckpointPool, SequenceState};

// Re-export weight types and config for backward compatibility
pub use crate::config::*;
pub use crate::weights::*;

mod forward;
mod model_load; // Weight loading and initialization // Layer forward passes (GDN, attention, Mamba2, FFN, MoE)

// ---- Main model struct ----

pub struct Model {
    // MUST be declared first: dropped first, shuts down cooperative kernels before any hipFree.
    // DeviceBuffer::drop → hipFree → SyncAllStreams blocks if cooperative kernel is still running.
    pub(crate) persistent_workers: Option<crate::persistent_dispatch::PersistentDispatch>,
    // GPU-native P2P MoE dispatch: cooperative kernels on GPUs 1-3. Drop before other GPU 1-3 resources.
    pub(crate) moe_p2p: Option<crate::moe_p2p::MoeP2pContext>,
    pub(crate) config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    pub(crate) kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) lm_head_weight: DeviceBuffer<u16>, // separate from embed when tie_word_embeddings=false
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) moe_weights: Vec<Option<MoeWeights>>, // per-layer MoE FFN (None for dense FFN layers)
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) legacy_kv_caches: Option<Vec<KvCache>>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) mamba2_states: Vec<Mamba2State>,
    pub(crate) seq_len: u32,
    pub(crate) megakernel: Option<MegakernelProgram>,
    // Paged KV path (lazy-init)
    pub(crate) megakernel_paged: Option<MegakernelProgram>,
    pub(crate) page_allocator: Option<PageAllocator>,
    pub(crate) quant_allocator: Option<PageAllocator>,
    pub(crate) paged_seq: Option<SequenceState>,
    pub(crate) paged_page_table: Option<DeviceBuffer<u64>>,
    pub(crate) paged_position_table: Option<DeviceBuffer<i32>>,
    pub(crate) checkpoint_pool: Option<RecurrentCheckpointPool>,
    pub(crate) last_checkpoint_slot: Option<u32>,
    pub(crate) trace: Option<crate::trace::TraceWriter>,
    pub(crate) debug_nan: bool,
    pub(crate) weight_prefix: String, // tensor name prefix (e.g. "model.language_model.")
    // Multi-GPU expert parallel (None for single-GPU)
    pub(crate) multi_gpu: Option<crate::multi_gpu::MultiGpuContext>,
    pub(crate) distributed_moe: Vec<Option<crate::weights::DistributedMoeWeights>>,
    pub(crate) worker_kernels: Vec<crate::moe_dispatch::WorkerKernels>,
    // Multi-GPU megakernel programs
    pub(crate) megakernel_multi_gpu: Option<MegakernelProgram>,
    pub(crate) megakernel_multi_gpu_p2p: Option<MegakernelProgram>,
}

// ---- Model impl ----

impl Model {
    fn max_paged_chunks(&self) -> usize {
        (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS
    }

    fn ensure_paged_decode_state(&mut self, quantized: bool) -> Result<(), ModelError> {
        let max_chunks = self.max_paged_chunks();
        if self.page_allocator.is_none() {
            self.page_allocator = Some(PageAllocator::new(
                self.device,
                &self.config,
                CHUNK_TOKENS,
                max_chunks as u32,
            )?);
            self.paged_seq = Some(SequenceState::new(CHUNK_TOKENS as u32));
        }

        if self.paged_page_table.is_none() {
            self.paged_page_table = Some(DeviceBuffer::alloc(self.device, max_chunks)?);
        }
        if self.paged_position_table.is_none() {
            self.paged_position_table =
                Some(DeviceBuffer::alloc(self.device, self.config.max_seq_len)?);
        }

        if quantized && self.quant_allocator.is_none() {
            self.quant_allocator = Some(PageAllocator::new_quantized(
                self.device,
                &self.config,
                CHUNK_TOKENS,
                max_chunks as u32,
            )?);
        }
        Ok(())
    }

    fn append_paged_decode_token(&mut self, position: u32) -> Result<(), ModelError> {
        self.ensure_paged_decode_state(false)?;
        let seq_mut = self.paged_seq.as_mut().unwrap();
        if seq_mut.seq_len == position {
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(position as i32, alloc_mut)?;
        }
        Ok(())
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn set_position(&mut self, position: u32) -> HipResult<()> {
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                pos_data.as_ptr(),
                self.activations.position_ids.host_ptr(),
                3,
            );
        }
        Ok(())
    }

    pub fn read_logits(&self) -> Result<Vec<f32>, ModelError> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// GPU-resident argmax: run decode step and return token ID without transferring logits.
    pub fn decode_step_token(&mut self, token_id: u32, position: u32) -> Result<u32, ModelError> {
        let logits = self.decode_step(token_id, position)?;
        if self.persistent_workers.is_some() {
            let nan_count = logits.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("WARN: {nan_count}/{} NaN in logits", logits.len());
            }
            let (idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .unwrap();
            Ok(idx as u32)
        } else {
            let result = self.kernels.argmax.forward(
                &self.activations.logits,
                &mut self.activations.argmax_result,
                self.config.vocab_size as u32,
                &self.stream,
            )?;
            Ok(result)
        }
    }

    /// Run a single decode step. Returns logits [vocab_size].
    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let is_multi_gpu = self.multi_gpu.is_some();
        if self.trace.is_some() {
            return self.decode_step_moe(token_id, position);
        }
        if is_multi_gpu {
            if std::env::var("PERSISTENT").as_deref() == Ok("1") {
                return self.decode_step_persistent_multi_gpu(token_id, position);
            }
            return self.decode_step_moe(token_id, position);
        }
        if std::env::var("PERSISTENT").as_deref() == Ok("1") {
            // Single-GPU persistent path cannot handle MoE: compile() emits OP_MOE_FFN,
            // but persistent_worker.hip has an intentional empty first branch for OP_MOE_FFN
            // (VGPR optimization trick) so expert FFN is silently skipped → garbled output.
            // MoE models need multi-GPU path or paged decode.
            let has_moe = self
                .config
                .layers
                .iter()
                .any(|l| matches!(l.ffn_type, crate::model::FfnType::MoE { .. }));
            if !has_moe {
                return self.decode_step_persistent(token_id, position);
            }
        }
        self.decode_step_paged(token_id, position)
    }

    /// Persistent worker decode: compile megakernel program, replay via CPU-scheduled dispatch.
    fn decode_step_persistent(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile megakernel program FIRST (needs GPU queries),
        // then launch persistent worker (occupies all SMs).
        if self.persistent_workers.is_none() {
            if self.megakernel.is_none() {
                let mk = MegakernelProgram::compile(self)?;
                self.megakernel = Some(mk);
            }

            // Use same shared_mem as megakernel (4096 for MoE, 2048 for dense) + 256 for local_inst
            let has_moe = self
                .config
                .layers
                .iter()
                .any(|l| matches!(l.ffn_type, crate::model::FfnType::MoE { .. }));
            let shared_mem = if has_moe {
                1024u32 * 4 + 256
            } else {
                256u32 * 4 * 2 + 256
            };
            let dispatch =
                PersistentDispatch::init(&[self.device], shared_mem, 0).map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
        }

        // Write position_ids directly to host-mapped memory (no hipMemcpy)
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                pos_data.as_ptr(),
                self.activations.position_ids.host_ptr(),
                3,
            );
        }

        let mk = self.megakernel.as_mut().unwrap();
        mk.instructions[mk.embedding_inst_idx].set_int(3, token_id as i32);
        let hd = mk.head_dim_attn;
        let max_sl = mk.max_seq_len as usize;
        let head_stride = max_sl * hd;
        for (layer_i, head_indices) in mk.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = mk.kv_base_ptrs[layer_i];
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let offset =
                    (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
                mk.instructions[k_idx].words[1] = k_base + offset as u64;
                mk.instructions[v_idx].words[1] = v_base + offset as u64;
            }
        }
        let seq_len = position + 1;
        for &idx in &mk.gqa_attn_inst_indices {
            mk.instructions[idx].set_int(8, seq_len as i32);
        }
        // dispatch instructions via persistent worker

        // Patch LM head instruction to write to logits_mapped (host-mapped)
        // so CPU can read without hipMemcpy (which deadlocks cooperative kernel).
        let n_inst = mk.instructions.len();
        let lm_head_idx = n_inst - 2; // second-to-last (before HALT)
        mk.instructions[lm_head_idx].words[1] =
            self.activations.logits_mapped.as_write_ptr() as u64;

        // Batch dispatch: send all instructions as batches of up to 64.
        // Worker processes all with grid.sync() between them, acks once per batch.
        let batch: Vec<_> = mk
            .instructions
            .iter()
            .take_while(|inst| (inst.words[0] & 0x7FFFFFFF) != 16)
            .cloned()
            .collect();
        let dispatch = self.persistent_workers.as_mut().unwrap();
        for chunk in batch.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
            dispatch.dispatch_batch(0, chunk);
        }

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
    /// Persistent worker on GPU 0 handles dense layers (60% faster than kbk).
    /// At MoE layers: worker paused, kbk dispatch across all GPUs, worker resumed.
    fn decode_step_persistent_multi_gpu(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::Instruction;
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile multi-GPU megakernel + launch workers on ALL GPUs
        if self.persistent_workers.is_none() {
            if self.megakernel_multi_gpu.is_none() {
                let mk = MegakernelProgram::compile_multi_gpu(self)?;
                self.megakernel_multi_gpu = Some(mk);
            }
            let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
            // OP_LINEAR_PROJ_PCG32/RNF4 tiled-LDS needs (8+7680+256)*4 = 31776 bytes per block.
            // moe_worker_kernel (GPUs 1-3) only needs 4352B; persistent_worker (GPU 0) needs 31776B.
            // Pass the larger value; moe_worker has its own calculation in moe_p2p.rs.
            let moe_worker_shared_mem = 1024u32 * 4 + 256; // 4352B for moe_worker_kernel
            let shared_mem = moe_worker_shared_mem.max(31776u32); // 31776B for persistent_worker
            let hs = self.config.hidden_size;
            let max_eis = self
                .config
                .layers
                .iter()
                .filter_map(|l| match &l.ffn_type {
                    crate::model::FfnType::MoE {
                        expert_intermediate_size,
                        ..
                    } => Some(*expert_intermediate_size),
                    _ => None,
                })
                .max()
                .unwrap_or(0);

            // Init GPU-native P2P MoE dispatch (moe_worker_kernel on GPUs 1-3) BEFORE
            // launching the persistent cooperative worker on GPU 0. hipMalloc on GPU 0
            // deadlocks if the cooperative kernel is already running (ROCm synchronizes
            // all GPU work before allocating). Launch persistent_worker LAST.
            let has_moe = self
                .config
                .layers
                .iter()
                .any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
            if has_moe && num_gpus > 1 {
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
                eprintln!(
                    "  MoE P2P dispatch initialized: {} worker GPUs",
                    num_gpus - 1
                );
            }

            // Launch persistent cooperative worker on GPU 0 LAST — after all GPU memory
            // operations are complete (hipMalloc deadlocks on running cooperative kernels).
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
        }

        // Use P2P megakernel if available (GPU-native MoE dispatch, no OP_BARRIER)
        if self.megakernel_multi_gpu_p2p.is_some() {
            return self.decode_step_p2p(token_id, position);
        }

        // Update host-side instructions
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                pos_data.as_ptr(),
                self.activations.position_ids.host_ptr(),
                3,
            );
        }
        let mk = self.megakernel_multi_gpu.as_mut().unwrap();
        mk.update_step_host_only(token_id, position)?;

        // Patch LM head to write to logits_mapped
        let n_inst = mk.instructions.len();
        mk.instructions[n_inst - 2].words[1] = self.activations.logits_mapped.as_write_ptr() as u64;

        let hs = self.config.hidden_size;
        // Precompute head-parallel attention boundaries.
        // For each attn layer: (mrope_idx, gqa_idx) — we flush at mrope, skip up to gqa.
        let has_head_parallel = self
            .multi_gpu
            .as_ref()
            .map(|m| !m.workers[0].attn_kv_caches.is_empty())
            .unwrap_or(false);
        // Use distributed QKV boundaries if available (multi_gpu path with split projections),
        // otherwise fall back to mRoPE/GQA boundaries (legacy head-parallel path).
        let use_distributed_qkv = has_head_parallel && {
            let mk_ref = self.megakernel_multi_gpu.as_ref().unwrap();
            !mk_ref.multi_gpu_attn_boundaries.is_empty()
        };
        let attn_boundaries: Vec<(usize, usize)> = if has_head_parallel {
            let mk_ref = self.megakernel_multi_gpu.as_ref().unwrap();
            if use_distributed_qkv {
                mk_ref.multi_gpu_attn_boundaries.clone()
            } else {
                mk_ref
                    ._mrope_inst_indices
                    .iter()
                    .zip(mk_ref.gqa_attn_inst_indices.iter())
                    .map(|(&mrope_idx, &gqa_idx)| (mrope_idx, gqa_idx))
                    .collect()
            }
        } else {
            Vec::new()
        };
        let n_inst = self
            .megakernel_multi_gpu
            .as_ref()
            .unwrap()
            .instructions
            .len();

        // Split instruction stream at OP_BARRIER markers into segments.
        // Dense segments batched to GPU 0. At OP_BARRIER: build per-GPU expert
        // FFN batches, dispatch to all GPUs in parallel via persistent workers.
        // At attention mRoPE boundary: flush and dispatch head-parallel attention.
        let mut segment: Vec<Instruction> = Vec::new();
        let mut attn_i = 0usize;
        let mut i = 0usize;
        let layer_timing = std::env::var("LAYER_TIMING").is_ok();
        let step_start = std::time::Instant::now();
        let mut layer_t = std::time::Instant::now();
        let mut moe_total_us = 0u64;
        let mut attn_total_us = 0u64;
        let mut dense_total_us = 0u64;
        let mut n_moe = 0u32;
        let mut n_attn = 0u32;
        // Pending reduce instructions: prepended to next dense segment dispatch to
        // merge two dispatch_batch round-trips into one per MoE layer.
        let mut pending_reduce: Vec<Instruction> = Vec::new();

        while i < n_inst {
            let inst = self.megakernel_multi_gpu.as_ref().unwrap().instructions[i].clone();
            i += 1;
            let opcode = inst.words[0] & 0x7FFFFFFF;
            if opcode == 16 {
                break;
            }

            if opcode == 33 {
                // OP_BARRIER — MoE dispatch point
                let layer_idx = inst.words[3] as usize;
                let (k, eis) = match &self.config.layers[layer_idx].ffn_type {
                    crate::model::FfnType::MoE {
                        num_active,
                        expert_intermediate_size,
                        ..
                    } => (*num_active, *expert_intermediate_size),
                    _ => panic!("OP_BARRIER on non-MoE layer"),
                };

                let dist_moe = self.distributed_moe[layer_idx]
                    .as_ref()
                    .expect("missing distributed MoE weights");

                // Append normed→normed_stage copy to the tail of the dense segment so it
                // executes in the same dispatch_batch, saving one round-trip per MoE layer.
                // (H2D broadcast to GPUs 1-3 happens via parallel hipMemcpyAsync in dispatch_moe_layer_kbk.)
                {
                    let mut copy_inst = Instruction::new(17, (hs as u32 + 255) / 256);
                    copy_inst.words[1] = self.activations.normed_stage.as_write_ptr() as u64;
                    copy_inst.words[2] = self.activations.normed.as_ptr() as u64;
                    copy_inst.words[3] = hs as u64;
                    segment.push(copy_inst);
                }

                // Flush dense segment (now includes the normed→normed_stage copy).
                // Prepend any pending reduce instructions from the previous MoE layer so they
                // execute in the same dispatch_batch, saving one round-trip per MoE layer.
                if !segment.is_empty() || !pending_reduce.is_empty() {
                    if layer_timing {
                        dense_total_us += layer_t.elapsed().as_micros() as u64;
                        layer_t = std::time::Instant::now();
                    }
                    let mut combined: Vec<Instruction> = std::mem::take(&mut pending_reduce);
                    combined.extend_from_slice(&segment);
                    segment.clear();
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS)
                    {
                        dispatch.dispatch_batch(0, chunk);
                    }
                }

                // Build OP_EXPERT_FFN batch for GPU 0 (persistent worker handles its experts).
                // GPUs 1+ run via kbk in parallel with GPU 0.
                let expert_ids_snap: Vec<i32> = unsafe {
                    std::slice::from_raw_parts(
                        self.activations.moe_expert_ids.host_ptr() as *const i32,
                        k,
                    )
                    .to_vec()
                };
                let expert_wts_snap: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(
                        self.activations.moe_expert_weights.host_ptr() as *const f32,
                        k,
                    )
                    .to_vec()
                };
                let mut gpu0_batch: Vec<Instruction> = Vec::new();
                {
                    let mgpu = self.multi_gpu.as_ref().unwrap();
                    let buf = &dist_moe.expert_buffers[0];
                    let sg_ptr = mgpu.workers[0].scratch_gate.as_ptr() as u64;
                    let su_ptr = mgpu.workers[0].scratch_up.as_ptr() as u64;
                    let sa_ptr = mgpu.workers[0].scratch_act.as_ptr() as u64;
                    let sd_ptr = mgpu.workers[0].scratch_down.as_ptr() as u64;
                    let act_ptr = self.activations.normed_stage.device_ptr() as u64;
                    let out_ptr = self
                        .persistent_workers
                        .as_ref()
                        .unwrap()
                        .moe_output_slot
                        .device_ptr() as u64;
                    let gu_row_stride = dist_moe.gate_up_row_stride as u64;
                    for j in 0..k {
                        let eid = expert_ids_snap[j] as usize;
                        if dist_moe.expert_device[eid] != 0 {
                            continue;
                        }
                        let local_slot = buf.slot_map[eid].expect("GPU 0 expert missing slot");
                        let gu_ptr = unsafe {
                            dist_moe
                                .gpu0_gate_up_base
                                .add(local_slot * dist_moe.gate_up_expert_stride)
                        } as u64;
                        let dn_ptr = unsafe {
                            dist_moe
                                .gpu0_down_base
                                .add(local_slot * dist_moe.down_expert_stride)
                        } as u64;
                        let ew_bits = expert_wts_snap[j].to_bits() as u64;

                        if dist_moe.has_gate_proj {
                            // Gate proj → scratch_gate
                            let mut g = Instruction::new(
                                crate::megakernel::OP_LINEAR_PROJ_PCG32,
                                eis as u32,
                            );
                            g.words[1] = sg_ptr;
                            g.words[2] = gu_ptr;
                            g.words[3] = act_ptr;
                            g.words[4] = eis as u64;
                            g.words[5] = hs as u64;
                            g.words[6] = 1;
                            gpu0_batch.push(g);
                            // Up proj → scratch_up (rows offset by eis * row_stride)
                            let gu_up = gu_ptr + (eis as u64) * gu_row_stride;
                            let mut u = Instruction::new(
                                crate::megakernel::OP_LINEAR_PROJ_PCG32,
                                eis as u32,
                            );
                            u.words[1] = su_ptr;
                            u.words[2] = gu_up;
                            u.words[3] = act_ptr;
                            u.words[4] = eis as u64;
                            u.words[5] = hs as u64;
                            u.words[6] = 1;
                            gpu0_batch.push(u);
                            // SiLU: silu(scratch_gate) * scratch_up → scratch_act
                            let mut silu = Instruction::new(
                                crate::megakernel::OP_SILU_MUL,
                                (eis as u32 + 255) / 256,
                            );
                            silu.words[1] = sa_ptr;
                            silu.words[2] = sg_ptr;
                            silu.words[3] = su_ptr;
                            silu.words[4] = eis as u64;
                            gpu0_batch.push(silu);
                        } else {
                            // Up-only proj → scratch_up
                            let mut u = Instruction::new(
                                crate::megakernel::OP_LINEAR_PROJ_PCG32,
                                eis as u32,
                            );
                            u.words[1] = su_ptr;
                            u.words[2] = gu_ptr;
                            u.words[3] = act_ptr;
                            u.words[4] = eis as u64;
                            u.words[5] = hs as u64;
                            u.words[6] = 1;
                            gpu0_batch.push(u);
                            // ReLU²: relu(scratch_up)² → scratch_act
                            let mut rsq = Instruction::new(
                                crate::megakernel::OP_RELU_SQ,
                                (eis as u32 + 255) / 256,
                            );
                            rsq.words[1] = sa_ptr;
                            rsq.words[2] = su_ptr;
                            rsq.words[3] = eis as u64;
                            gpu0_batch.push(rsq);
                        }
                        // Down proj: scratch_act → scratch_down
                        let mut d =
                            Instruction::new(crate::megakernel::OP_LINEAR_PROJ_PCG32, hs as u32);
                        d.words[1] = sd_ptr;
                        d.words[2] = dn_ptr;
                        d.words[3] = sa_ptr;
                        d.words[4] = hs as u64;
                        d.words[5] = eis as u64;
                        d.words[6] = 1;
                        gpu0_batch.push(d);
                        // Scale add: moe_output_slot += ew * scratch_down
                        let mut sa_inst = Instruction::new(
                            crate::megakernel::OP_SCALE_ADD,
                            (hs as u32 + 255) / 256,
                        );
                        sa_inst.words[1] = out_ptr;
                        sa_inst.words[2] = sd_ptr;
                        sa_inst.words[3] = ew_bits;
                        sa_inst.words[4] = hs as u64;
                        gpu0_batch.push(sa_inst);
                    }
                }

                // Zero moe_output_slot so GPU 0 experts accumulate into a clean buffer.
                // Always zeroed: when no GPU 0 experts run, the D2D_COPY init will write zeros.
                unsafe {
                    std::ptr::write_bytes(
                        self.persistent_workers
                            .as_ref()
                            .unwrap()
                            .moe_output_slot
                            .host_ptr(),
                        0,
                        hs,
                    );
                }

                // Fire GPU 0 expert batch non-blocking (fat worker computes while CPU dispatches GPUs 1+).
                let seq0 = if !gpu0_batch.is_empty() {
                    Some(
                        self.persistent_workers
                            .as_mut()
                            .unwrap()
                            .dispatch_batch_fire(0, &gpu0_batch),
                    )
                } else {
                    None
                };

                // Dispatch GPUs 1+ via kbk — runs in parallel with GPU 0's fat worker.
                // dispatch_moe_layer_kbk dispatches compute only; no D2H/CPU gather.
                let active_mask = {
                    let mgpu =
                        self.multi_gpu.as_mut().unwrap() as *mut crate::multi_gpu::MultiGpuContext;
                    let wk = self.worker_kernels.as_slice()
                        as *const [crate::moe_dispatch::WorkerKernels];
                    let dm = dist_moe as *const crate::weights::DistributedMoeWeights;
                    let ns = &self.activations.normed_stage
                        as *const braidinfer_hip::memory::MappedHostBuffer<f32>;
                    let ids = &self.activations.moe_expert_ids
                        as *const braidinfer_hip::memory::MappedHostBuffer<i32>;
                    let wts = &self.activations.moe_expert_weights
                        as *const braidinfer_hip::memory::MappedHostBuffer<f32>;
                    unsafe {
                        crate::moe_dispatch::dispatch_moe_layer_kbk(
                            &mut *mgpu, &*wk, &*dm, &*ns, &*ids, &*wts, k, hs, eis,
                        )
                        .map_err(ModelError::Hip)?
                    }
                };

                // Sync GPUs 1+ first (CPU-side stream sync, overlaps with GPU 0 persistent worker).
                // Then wait_ack GPU 0. Both may already be done by the time we check.
                let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
                for gpu_i in 1..num_gpus {
                    if active_mask & (1 << gpu_i) == 0 {
                        continue;
                    }
                    let worker = &self.multi_gpu.as_ref().unwrap().workers[gpu_i];
                    braidinfer_hip::Device::set_current(worker.device).map_err(ModelError::Hip)?;
                    worker
                        .compute_stream
                        .synchronize()
                        .map_err(ModelError::Hip)?;
                }
                braidinfer_hip::Device::set_current(braidinfer_core::types::DeviceId(0))
                    .map_err(ModelError::Hip)?;
                if let Some(seq) = seq0 {
                    self.persistent_workers.as_ref().unwrap().wait_ack(0, seq);
                }

                // On-GPU reduction: build a persistent worker batch that:
                // 1. Copies GPU 0's moe_output_slot → ffn_down (init with GPU 0 result, or zero if no GPU 0 experts)
                // 2. For each GPU 1-3 with experts: D2D_COPY expert_out → scratch_down, SCALE_ADD into ffn_down
                {
                    let ffn_down_ptr = self.activations.ffn_down.as_write_ptr() as u64;
                    let moe_slot_ptr = self
                        .persistent_workers
                        .as_ref()
                        .unwrap()
                        .moe_output_slot
                        .device_ptr() as u64;
                    let scratch_down_ptr = self.multi_gpu.as_ref().unwrap().workers[0]
                        .scratch_down
                        .as_ptr() as u64;
                    let grid_hs = (hs as u32 + 255) / 256;
                    let mut reduce_batch: Vec<Instruction> = Vec::new();

                    // Step 1: copy GPU 0 result (or zeros) into ffn_down
                    let mut init = Instruction::new(17, grid_hs); // OP_D2D_COPY
                    init.words[1] = ffn_down_ptr;
                    init.words[2] = moe_slot_ptr;
                    init.words[3] = hs as u64;
                    reduce_batch.push(init);

                    // Step 2: accumulate each GPU 1+ expert_out
                    for gpu_i in 1..num_gpus {
                        if active_mask & (1 << gpu_i) == 0 {
                            continue;
                        }
                        let expert_out_ptr = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                            .expert_out
                            .as_ptr() as u64;
                        // D2D_COPY: GPU i expert_out → GPU 0 scratch_down (P2P)
                        let mut copy = Instruction::new(17, grid_hs);
                        copy.words[1] = scratch_down_ptr;
                        copy.words[2] = expert_out_ptr;
                        copy.words[3] = hs as u64;
                        reduce_batch.push(copy);
                        // SCALE_ADD: ffn_down += 1.0 * scratch_down
                        let mut add = Instruction::new(crate::megakernel::OP_SCALE_ADD, grid_hs);
                        add.words[1] = ffn_down_ptr;
                        add.words[2] = scratch_down_ptr;
                        add.words[3] = 1.0f32.to_bits() as u64;
                        add.words[4] = hs as u64;
                        reduce_batch.push(add);
                    }

                    // Don't dispatch yet — defer to next dense segment flush to merge into one batch.
                    pending_reduce = reduce_batch;
                }

                if layer_timing {
                    let us = layer_t.elapsed().as_micros() as u64;
                    moe_total_us += us;
                    n_moe += 1;
                    layer_t = std::time::Instant::now();
                }
                continue;
            }

            // Head-parallel attention: at boundary, flush segment, dispatch parallel QKV+GQA
            if has_head_parallel && attn_i < attn_boundaries.len() {
                let (flush_idx, resume_idx) = attn_boundaries[attn_i];
                if i - 1 == flush_idx {
                    // Include boundary instruction (RMSNorm or mRoPE) in segment, flush to GPU 0.
                    // Prepend any pending reduce from previous MoE layer.
                    segment.push(inst);
                    {
                        let mut combined: Vec<Instruction> = std::mem::take(&mut pending_reduce);
                        combined.extend_from_slice(&segment);
                        segment.clear();
                        let dispatch = self.persistent_workers.as_mut().unwrap();
                        for chunk in
                            combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS)
                        {
                            dispatch.dispatch_batch(0, chunk);
                        }
                    }
                    if layer_timing {
                        layer_t = std::time::Instant::now();
                    }
                    self.dispatch_head_parallel_attention(attn_i, position)?;
                    if layer_timing {
                        let us = layer_t.elapsed().as_micros() as u64;
                        attn_total_us += us;
                        n_attn += 1;
                        layer_t = std::time::Instant::now();
                    }
                    attn_i += 1;
                    // For distributed QKV (new): resume_idx = output_gate_idx (don't skip it)
                    // For legacy mRoPE boundary: resume_idx = gqa_idx (skip the GQA itself: +1)
                    i = if use_distributed_qkv {
                        resume_idx
                    } else {
                        resume_idx + 1
                    };
                    continue;
                }
            }

            segment.push(inst);
        }

        // Flush final segment (with any pending reduce prefix)
        if !segment.is_empty() || !pending_reduce.is_empty() {
            let mut combined: Vec<Instruction> = std::mem::take(&mut pending_reduce);
            combined.extend_from_slice(&segment);
            let dispatch = self.persistent_workers.as_mut().unwrap();
            for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                dispatch.dispatch_batch(0, chunk);
            }
        }

        if layer_timing && position > 0 {
            let total = moe_total_us + attn_total_us + dense_total_us;
            let wall_us = step_start.elapsed().as_micros() as u64;
            eprintln!(
                "LAYER_TIMING pos={position}: moe={:.1}ms×{n_moe}  attn={:.1}ms×{n_attn}  dense={:.1}ms  tracked={:.1}ms  wall={:.1}ms",
                moe_total_us as f64 / 1000.,
                attn_total_us as f64 / 1000.,
                dense_total_us as f64 / 1000.,
                total as f64 / 1000.,
                wall_us as f64 / 1000.
            );
        }

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

    /// GPU-native P2P MoE decode: OP_MOE_DISPATCH handled entirely by megakernel.
    /// No CPU-side expert dispatching. Attention is still head-parallel (same as before).
    fn decode_step_p2p(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::Instruction;

        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                pos_data.as_ptr(),
                self.activations.position_ids.host_ptr(),
                3,
            );
        }

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

        let mut segment: Vec<Instruction> = Vec::new();
        let mut attn_i = 0usize;
        let mut i = 0usize;

        while i < n_inst {
            let inst = self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[i].clone();
            i += 1;
            let opcode = inst.words[0] & 0x7FFFFFFF;
            if opcode == 16 {
                break;
            } // OP_HALT

            // Head-parallel attention boundary: flush segment, dispatch parallel QKV+GQA
            if has_head_parallel && attn_i < attn_boundaries.len() {
                let (flush_idx, resume_idx) = attn_boundaries[attn_i];
                if i - 1 == flush_idx {
                    segment.push(inst);
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    for chunk in segment.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS)
                    {
                        dispatch.dispatch_batch(0, chunk);
                    }
                    segment.clear();
                    self.dispatch_head_parallel_attention(attn_i, position)?;
                    attn_i += 1;
                    i = if use_distributed_qkv {
                        resume_idx
                    } else {
                        resume_idx + 1
                    };
                    continue;
                }
            }

            segment.push(inst);
        }

        if !segment.is_empty() {
            let dispatch = self.persistent_workers.as_mut().unwrap();
            let debug_hidden = std::env::var("DEBUG_P2P_HIDDEN").is_ok();
            let mut batch_idx = 0usize;
            for chunk in segment.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                dispatch.dispatch_batch(0, chunk);
                if debug_hidden {
                    // Sync + read hidden[0:2] after each batch
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

        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();
        // DBG: print top token and logit distribution
        {
            let nan_count = logits.iter().filter(|v| v.is_nan()).count();
            let mut top5: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top5.truncate(5);
            eprintln!("DBG p2p logits: nan={nan_count} top5={top5:?} logits[11]={:.4}", logits[11]);
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
    fn dispatch_head_parallel_attention(
        &mut self,
        attn_i: usize,
        position: u32,
    ) -> Result<(), ModelError> {
        use crate::megakernel::{
            Instruction, OP_D2D_COPY, OP_DEINTERLEAVE, OP_GQA_ATTN, OP_LINEAR_PROJ,
            OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_MROPE, OP_QK_NORM,
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
            let mut inst = Instruction::new(opcode, out_dim as u32);
            inst.set_output_ptr(1, out_ptr);
            inst.set_ptr(2, w_ptr);
            inst.set_ptr(3, in_ptr);
            inst.set_int(4, out_dim as i32);
            inst.set_int(5, in_dim as i32);
            batch.push(inst);
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
            .megakernel_multi_gpu
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
                    let mut inst = Instruction::new(OP_D2D_COPY, (hs as u32 + 255) / 256);
                    inst.set_output_ptr(1, normed_local as *mut f32);
                    inst.set_ptr(2, normed_base as *const f32);
                    inst.set_int(3, hs as i32);
                    inst.set_no_sync();
                    batch.push(inst);
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
                    let mut inst = Instruction::new(OP_DEINTERLEAVE, (total as u32 + 255) / 256);
                    inst.set_output_ptr(1, q_ptr as *mut f32);
                    inst.set_output_ptr(2, gate_ptr as *mut f32);
                    inst.set_ptr(3, q_gate_ptr as *const f32);
                    inst.set_int(4, local_nqh as i32);
                    inst.set_int(5, hd as i32);
                    inst.set_int(6, 1); // batch=1
                    batch.push(inst);
                } else {
                    // Non-gated: q_gate IS q, just copy
                    let mut inst =
                        Instruction::new(OP_D2D_COPY, ((local_nqh * hd) as u32 + 255) / 256);
                    inst.set_output_ptr(1, q_ptr as *mut f32);
                    inst.set_ptr(2, q_gate_ptr as *const f32);
                    inst.set_int(3, (local_nqh * hd) as i32);
                    batch.push(inst);
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
                    let mut inst = Instruction::new(OP_QK_NORM, (local_nqh + local_nkh) as u32);
                    inst.set_output_ptr(1, q_ptr as *mut f32);
                    inst.set_output_ptr(2, k_local_ptr as *mut f32);
                    inst.set_ptr(3, q_norm_ptr);
                    inst.set_ptr(4, k_norm_ptr);
                    inst.set_int(5, local_nqh as i32);
                    inst.set_int(6, local_nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_float(8, qk_norm_eps);
                    batch.push(inst);
                }

                // 6. KV write (local — from local k/v to local KV cache)
                for h_local in 0..local_nkh {
                    let src_k = k_local_ptr + (h_local * hd * 4) as u64;
                    let src_v = v_local_ptr + (h_local * hd * 4) as u64;
                    let dst_k =
                        kv_k_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let dst_v =
                        kv_v_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let mut inst = Instruction::new(OP_D2D_COPY, ((hd as u32) + 255) / 256);
                    inst.set_output_ptr(1, dst_k as *mut f32);
                    inst.set_ptr(2, src_k as *const f32);
                    inst.set_int(3, hd as i32);
                    inst.set_no_sync();
                    batch.push(inst);
                    let mut inst = Instruction::new(OP_D2D_COPY, ((hd as u32) + 255) / 256);
                    inst.set_output_ptr(1, dst_v as *mut f32);
                    inst.set_ptr(2, src_v as *const f32);
                    inst.set_int(3, hd as i32);
                    inst.set_no_sync();
                    batch.push(inst);
                }

                // 7. mRoPE on local Q+K — only for models that use RoPE
                if self.config.use_rope {
                    let rd = self.config.rope_dim;
                    let ms = &self.config.mrope_section;
                    let mut inst = Instruction::new(OP_MROPE, (local_nqh + local_nkh) as u32);
                    inst.set_output_ptr(1, q_ptr as *mut f32);
                    inst.set_output_ptr(2, k_local_ptr as *mut f32);
                    inst.set_ptr(3, self.activations.inv_freq.as_ptr());
                    inst.set_ptr(4, self.activations.position_ids.as_ptr());
                    inst.set_int(5, local_nqh as i32);
                    inst.set_int(6, local_nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, rd as i32);
                    inst.set_int(9, ms[0] as i32);
                    inst.set_int(10, ms[1] as i32);
                    inst.set_int(11, ms[2] as i32);
                    batch.push(inst);
                }

                // 8. GQA (same as legacy path)
                let seq_len = (position + 1) as i32;
                {
                    let mut inst = Instruction::new(OP_GQA_ATTN, local_nqh as u32);
                    inst.set_output_ptr(1, out_ptr as *mut f32);
                    inst.set_ptr(2, q_ptr as *const f32);
                    inst.set_ptr(3, kv_k_base as *const f32);
                    inst.set_ptr(4, kv_v_base as *const f32);
                    inst.set_int(5, nqh as i32);          // global nqh for gqa_group
                    inst.set_int(6, nkh as i32);          // global nkh
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, seq_len);
                    inst.set_int(9, max_sl as i32);
                    inst.set_int(10, (gpu_i * local_nqh) as i32); // q_head_start
                    batch.push(inst);
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
                    let mut inst = Instruction::new(OP_D2D_COPY, ((hd as u32) + 255) / 256);
                    inst.set_output_ptr(1, dst_k as *mut f32);
                    inst.set_ptr(2, src_k as *const f32);
                    inst.set_int(3, hd as i32);
                    inst.set_no_sync();
                    batch.push(inst);
                    let mut inst = Instruction::new(OP_D2D_COPY, ((hd as u32) + 255) / 256);
                    inst.set_output_ptr(1, dst_v as *mut f32);
                    inst.set_ptr(2, src_v as *const f32);
                    inst.set_int(3, hd as i32);
                    inst.set_no_sync();
                    batch.push(inst);
                }

                // For GPU i > 0: copy Q slice from GPU 0's q_attn to local attn_q
                if gpu_i > 0 {
                    let src_q = q_attn_base + (gpu_i * local_nqh * hd * 4) as u64;
                    let mut inst =
                        Instruction::new(OP_D2D_COPY, ((local_nqh * hd) as u32 + 255) / 256);
                    inst.set_output_ptr(1, q_ptr as *mut f32);
                    inst.set_ptr(2, src_q as *const f32);
                    inst.set_int(3, (local_nqh * hd) as i32);
                    batch.push(inst);
                }

                // GQA attention
                let seq_len = (position + 1) as i32;
                let q_src = if gpu_i == 0 { q_attn_base } else { q_ptr };
                {
                    let mut inst = Instruction::new(OP_GQA_ATTN, local_nqh as u32);
                    inst.set_output_ptr(1, out_ptr as *mut f32);
                    inst.set_ptr(2, q_src as *const f32);
                    inst.set_ptr(3, kv_k_base as *const f32);
                    inst.set_ptr(4, kv_v_base as *const f32);
                    inst.set_int(5, nqh as i32);          // global nqh for gqa_group
                    inst.set_int(6, nkh as i32);          // global nkh
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, seq_len);
                    inst.set_int(9, max_sl as i32);
                    inst.set_int(10, (gpu_i * local_nqh) as i32); // q_head_start
                    batch.push(inst);
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
        {
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
                let mut inst = Instruction::new(OP_D2D_COPY, grid_x);
                inst.set_output_ptr(1, dst);
                inst.set_ptr(2, src);
                inst.set_int(3, n_elems as i32);
                inst.set_no_sync();
                gather_batch.push(inst);
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
                    let mut inst = Instruction::new(OP_D2D_COPY, grid_x);
                    inst.set_output_ptr(1, dst);
                    inst.set_ptr(2, src);
                    inst.set_int(3, n_elems as i32);
                    inst.set_no_sync();
                    gather_batch.push(inst);
                }
            }

            if !gather_batch.is_empty() {
                assert!(
                    gather_batch.len() <= MAX_BATCH_INSTRUCTIONS,
                    "gather batch overflow len={}",
                    gather_batch.len()
                );
                self.persistent_workers
                    .as_mut()
                    .unwrap()
                    .dispatch_batch(0, &gather_batch);
            }
        }

        Ok(())
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
            let opcode = (inst.words[0] & 0x7FFFFFFF) as u32;
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

    fn decode_step_moe(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;
        let sync_debug = std::env::var("SYNC_DEBUG").is_ok();
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
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(
                pos_data.as_ptr(),
                self.activations.position_ids.host_ptr(),
                pos_data.len(),
            )
        };

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
                    if layer_i >= 5 && layer_i <= 10 {
                        self.stream.synchronize()?;
                        let src = self.activations.hidden.as_ptr() as *const u8;
                        let mut buf = [0u8; 8];
                        braidinfer_hip::memory::memcpy_d2h(&mut buf, src, 8)?;
                        let v0 = f32::from_ne_bytes([buf[0],buf[1],buf[2],buf[3]]);
                        let v1 = f32::from_ne_bytes([buf[4],buf[5],buf[6],buf[7]]);
                        eprintln!("DBG ref attn L{layer_i}: h[0]={v0:.6} h[1]={v1:.6}");
                    }
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
                    // DEBUG: print hidden[0:2] after Mamba2 layers near divergence point
                    if layer_i >= 5 && layer_i <= 8 {
                        self.stream.synchronize()?;
                        let src = self.activations.hidden.as_ptr() as *const u8;
                        let mut buf = [0u8; 8];
                        braidinfer_hip::memory::memcpy_d2h(&mut buf, src, 8)?;
                        let v0 = f32::from_ne_bytes([buf[0],buf[1],buf[2],buf[3]]);
                        let v1 = f32::from_ne_bytes([buf[4],buf[5],buf[6],buf[7]]);
                        eprintln!("DBG ref mamba L{layer_i}: h[0]={v0:.6} h[1]={v1:.6}");
                    }
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

        // DBG: print top5 logits from reference moe path
        {
            let nan_count = logits.iter().filter(|v| v.is_nan()).count();
            let mut top5: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top5.truncate(5);
            eprintln!("DBG moe logits pos={position}: nan={nan_count} top5={top5:?} logits[11]={:.4}", logits[11]);
        }

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

    /// Save the current GDN recurrent states into a checkpoint pool slot.
    /// Lazy-initializes the pool on first call. Returns the slot index.
    pub fn save_recurrent_checkpoint(&mut self) -> Result<u32, ModelError> {
        if self.checkpoint_pool.is_none() {
            // Pool capacity 1: prefill uses ring-buffer overwrite (only most-recent needed).
            // Speculative decode (future) may increase this.
            self.checkpoint_pool =
                Some(RecurrentCheckpointPool::new(self.device, &self.config, 1)?);
        }
        // Free previous slot before allocating new one (ring buffer with capacity 1)
        if let Some(prev) = self.last_checkpoint_slot.take() {
            self.checkpoint_pool.as_mut().unwrap().free(prev);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> =
            self.gdn_states.iter().map(|s| &s.recurrent).collect();
        let pool = self.checkpoint_pool.as_mut().unwrap();
        let slot = paged_kv::save_checkpoint(pool, &recurrent_bufs, self.stream.raw())?;
        self.last_checkpoint_slot = Some(slot);
        Ok(slot)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    /// Saves GDN checkpoints at each 64-token chunk boundary.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }
        let mut logits = vec![];
        for (i, &tok) in tokens.iter().enumerate() {
            logits = self.decode_step(tok, i as u32)?;
        }
        Ok(logits)
    }

    /// Read all GDN recurrent state to host (for testing).
    pub fn read_gdn_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_states.len());
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            state.recurrent.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Restore GDN recurrent states from a previously saved checkpoint slot.
    pub fn restore_recurrent_checkpoint(&mut self, slot: u32) -> Result<(), ModelError> {
        let pool = self
            .checkpoint_pool
            .as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self
            .gdn_states
            .iter_mut()
            .map(|s| &mut s.recurrent)
            .collect();
        let stream_raw = self.stream.raw();
        paged_kv::restore_checkpoint(pool, slot, &mut recurrent_bufs, stream_raw)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn read_hidden(&self) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut buf = vec![0.0f32; self.config.hidden_size];
        self.activations.hidden.copy_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn decode_step_traced(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();
        if self
            .config
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Attention)
        {
            self.append_paged_decode_token(position)?;
        }

        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..self.config.num_layers {
            if self.config.layers[i].layer_type == LayerType::Attention {
                self.attention_forward(i, kv_idx, position)?;
                kv_idx += 1;
            } else {
                self.gdn_forward(i, gdn_idx)?;
                gdn_idx += 1;
            }
            traces.push((format!("layer_{i}"), self.read_hidden()?));
        }

        unsafe {
            d2d_copy_f32(
                &mut self.activations.normed,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden,
            &self.activations.normed,
            &self.final_norm_weight,
            1,
            hs,
            self.config.rms_norm_eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits,
            &self.embed_weight,
            &self.activations.hidden,
            vs,
            hs,
            &self.stream,
        )?;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok((logits, traces))
    }

    fn read_buf(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut v = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut v)?;
        Ok(v)
    }

    pub fn gdn_layer0_trace(
        &mut self,
        token_id: u32,
    ) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &w.input_norm,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        w.w_qkv.forward(
            &self.kernels.linear_proj,
            &mut self.activations.qkv,
            &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        w.w_a.forward(
            &self.kernels.linear_proj,
            &mut self.activations.a_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_b.forward(
            &self.kernels.linear_proj,
            &mut self.activations.b_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_z.forward(
            &self.kernels.linear_proj,
            &mut self.activations.z_proj,
            &self.activations.normed,
            nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nvh_traced * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &self.activations.qkv,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &self.activations.qkv,
                conv_q_len,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &self.activations.qkv,
                conv_q_len + conv_k_len,
                conv_v_len,
                &self.stream,
            )?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            d2d_copy_u16(
                &mut conv_w_q,
                0,
                &w.conv1d_weight,
                0,
                conv_q_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_k,
                0,
                &w.conv1d_weight,
                conv_q_len * ck_usize,
                conv_k_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_v,
                0,
                &w.conv1d_weight,
                (conv_q_len + conv_k_len) * ck_usize,
                conv_v_len * ck_usize,
                &self.stream,
            )?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(
                &mut cs_q,
                0,
                &self.gdn_conv_states[0],
                0,
                conv_state_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_k,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len,
                conv_state_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_v,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len + conv_state_k_len,
                conv_state_v_len,
                &self.stream,
            )?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(
            &mut cs_q,
            &self.activations.q_gdn,
            &conv_w_q,
            &mut conv_out_q,
            conv_q_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_k,
            &self.activations.k_gdn,
            &conv_w_k,
            &mut conv_out_k,
            conv_k_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_v,
            &self.activations.v_gdn,
            &conv_w_v,
            &mut conv_out_v,
            conv_v_len as u32,
            ck,
            &self.stream,
        )?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &conv_out_q,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &conv_out_k,
                0,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &conv_out_v,
                0,
                conv_v_len,
                &self.stream,
            )?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &w.a_log,
            &self.activations.a_proj,
            &w.dt_bias,
            nh,
            &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent,
            &mut self.activations.recurrent_out,
            nvh_traced,
            kd,
            vd,
            gqa_traced,
            &self.stream,
        )?;
        traces.push((
            "recurrent_out".into(),
            self.read_buf(&self.activations.recurrent_out)?,
        ));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &w.output_norm,
            nvh_traced,
            vd,
            eps,
            &self.stream,
        )?;
        traces.push((
            "normed_gated".into(),
            self.read_buf(&self.activations.normed_gated)?,
        ));

        // out_proj
        w.w_out.forward(
            &self.kernels.linear_proj,
            &mut self.activations.out_proj,
            &self.activations.normed_gated,
            hs,
            nvh_traced * vd,
            &self.stream,
        )?;
        traces.push((
            "out_proj".into(),
            self.read_buf(&self.activations.out_proj)?,
        ));

        // Residual
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
        traces.push(("after_residual".into(), self.read_hidden()?));

        Ok(traces)
    }

    pub fn reset_state(&mut self) -> Result<(), ModelError> {
        let nh = self.config.linear_num_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let ck = self.config.linear_conv_kernel_dim;
        let nvh_r = self.config.linear_num_value_heads;
        let qkv_out = nh * kd * 2 + nvh_r * vd;

        for state in &mut self.gdn_states {
            let zeros = vec![0.0f32; nvh_r * kd * vd];
            state.recurrent.copy_from_host(&zeros)?;
        }
        for conv_state in &mut self.gdn_conv_states {
            let zeros = vec![0.0f32; qkv_out * (ck - 1)];
            conv_state.copy_from_host(&zeros)?;
        }
        if let Some(caches) = self.legacy_kv_caches.as_mut() {
            let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
            let zeros_kv = vec![0.0f32; kv_size];
            for cache in caches {
                cache.k.copy_from_host(&zeros_kv)?;
                cache.v.copy_from_host(&zeros_kv)?;
            }
        }
        self.seq_len = 0;
        if let Some(seq) = self.paged_seq.as_mut() {
            if let Some(q_alloc) = self.quant_allocator.as_mut() {
                seq.free_quant_slots(q_alloc);
            }
            if let Some(alloc) = self.page_allocator.as_mut() {
                seq.reset(alloc);
            }
        }
        Ok(())
    }
}
