use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;

use crate::megakernel::{MegakernelProgram, PrefillBuffers, CHUNK_TOKENS};
use crate::paged_kv::{self, PageAllocator, RecurrentCheckpointPool, SequenceState};

// Re-export weight types and config for backward compatibility
pub use crate::config::*;
pub use crate::weights::*;

mod model_load;  // Weight loading and initialization
mod forward;     // Layer forward passes (GDN, attention, Mamba2, FFN, MoE)

// ---- Main model struct ----

pub struct Model {
    // MUST be declared first: dropped first, shuts down cooperative kernels before any hipFree.
    // DeviceBuffer::drop → hipFree → SyncAllStreams blocks if cooperative kernel is still running.
    pub(crate) persistent_workers: Option<crate::persistent_dispatch::PersistentDispatch>,
    pub(crate) config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    pub(crate) kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) lm_head_weight: DeviceBuffer<u16>,  // separate from embed when tie_word_embeddings=false
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) moe_weights: Vec<Option<MoeWeights>>,  // per-layer MoE FFN (None for dense FFN layers)
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) kv_caches: Vec<KvCache>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) mamba2_states: Vec<Mamba2State>,
    pub(crate) seq_len: u32,
    pub(crate) megakernel: Option<MegakernelProgram>,
    // Paged KV path (lazy-init)
    pub(crate) megakernel_paged: Option<MegakernelProgram>,
    pub(crate) page_allocator: Option<PageAllocator>,
    pub(crate) quant_allocator: Option<PageAllocator>,
    pub(crate) paged_seq: Option<SequenceState>,
    pub(crate) checkpoint_pool: Option<RecurrentCheckpointPool>,
    pub(crate) last_checkpoint_slot: Option<u32>,
    pub(crate) trace: Option<crate::trace::TraceWriter>,
    pub(crate) debug_nan: bool,
    pub(crate) weight_prefix: String,  // tensor name prefix (e.g. "model.language_model.")
    // Multi-GPU expert parallel (None for single-GPU)
    pub(crate) multi_gpu: Option<crate::multi_gpu::MultiGpuContext>,
    pub(crate) distributed_moe: Vec<Option<crate::weights::DistributedMoeWeights>>,
    pub(crate) worker_kernels: Vec<crate::moe_dispatch::WorkerKernels>,
    // Multi-GPU megakernel: dense layers run in megakernel; MoE layers use CPU-dispatch via OP_BARRIER
    pub(crate) megakernel_multi_gpu: Option<MegakernelProgram>,
}

// ---- Model impl ----

impl Model {

    pub fn config(&self) -> &ModelConfig { &self.config }
    pub fn stream(&self) -> &Stream { &self.stream }
    pub fn vocab_size(&self) -> usize { self.config.vocab_size }

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
    /// Eliminates vocab_size×4 bytes PCIe transfer per token (e.g., 512KB for Nemotron).
    pub fn decode_step_token(&mut self, token_id: u32, position: u32) -> Result<u32, ModelError> {
        let logits = self.decode_step(token_id, position)?;
        if self.persistent_workers.is_some() {
            // CPU argmax: persistent worker occupies all SMs, can't launch GPU argmax
            let (idx, _) = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            Ok(idx as u32)
        } else {
            // GPU argmax: transfers only 4 bytes back
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
        let has_mamba2 = self.config.layers.iter().any(|l| l.layer_type == crate::config::LayerType::Mamba2);
        let is_multi_gpu = self.multi_gpu.is_some();
        // Mamba2 and trace mode always use kernel-by-kernel path.
        if has_mamba2 || self.trace.is_some() {
            return self.decode_step_moe(token_id, position);
        }
        // Multi-GPU MoE: use persistent dispatch when PERSISTENT=1.
        // braidinfer-sg4: grid.sync() cross-GPU works fine on RDNA3 — the prior deadlock
        // was a software bug (all 96 blocks computed ALL rows redundantly, timing out CPU).
        // Fixed by virtual block loops + single shared scratch + grid.sync() between phases.
        if is_multi_gpu {
            if std::env::var("PERSISTENT").as_deref() == Ok("1") {
                return self.decode_step_persistent_multi_gpu(token_id, position);
            }
            return self.decode_step_moe(token_id, position);
        }

        // Persistent worker path: CPU-scheduled dispatch via host-mapped work queue.
        // Gated by PERSISTENT=1 env var. Replaces megakernel for single-GPU.
        if std::env::var("PERSISTENT").is_ok() {
            return self.decode_step_persistent(token_id, position);
        }

        // Dense models: use megakernel (handles bf16 + quantized weights, both RMSNorm variants)
        if self.megakernel.is_none() {
            let mut mk = MegakernelProgram::compile(self)?;
            if let Ok(dump_path) = std::env::var("MEGAKERNEL_DUMP") {
                let max_slots: i32 = std::env::var("MEGAKERNEL_DUMP_SLOTS")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(500);
                mk.enable_dump(max_slots)?;
                eprintln!("Megakernel dump enabled: {} slots, output={}", max_slots, dump_path);
            }
            self.megakernel = Some(mk);
        }
        let mk = self.megakernel.as_mut().unwrap();
        mk.update_step(token_id, position, &self.stream)?;
        mk.execute(&self.stream)?;

        self.stream.synchronize()?;

        // Write dump after first decode token if MEGAKERNEL_DUMP is set
        if let Ok(dump_path) = std::env::var("MEGAKERNEL_DUMP") {
            if mk.dump_active() {
                mk.write_dump_btrc(&self.stream, &dump_path)?;
                mk.disable_dump()?;
            }
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Persistent worker decode: compile megakernel program, replay via CPU-scheduled dispatch.
    fn decode_step_persistent(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile megakernel program FIRST (needs GPU queries),
        // then launch persistent worker (occupies all SMs).
        if self.persistent_workers.is_none() {
            if self.megakernel.is_none() {
                let mk = MegakernelProgram::compile(self)?;
                self.megakernel = Some(mk);
            }

            // Use same shared_mem as megakernel (4096 for MoE, 2048 for dense) + 256 for local_inst
            let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, crate::model::FfnType::MoE { .. }));
            let shared_mem = if has_moe { 1024u32 * 4 + 256 } else { 256u32 * 4 * 2 + 256 };
            let dispatch = PersistentDispatch::init(&[self.device], shared_mem, 0)
                .map_err(ModelError::Hip)?;
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
                let offset = (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
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
        mk.instructions[lm_head_idx].words[1] = self.activations.logits_mapped.as_write_ptr() as u64;

        // Batch dispatch: send all instructions as batches of up to 64.
        // Worker processes all with grid.sync() between them, acks once per batch.
        let batch: Vec<_> = mk.instructions.iter()
            .take_while(|inst| (inst.words[0] & 0x7FFFFFFF) != 16)
            .cloned()
            .collect();
        let dispatch = self.persistent_workers.as_mut().unwrap();
        for chunk in batch.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
            dispatch.dispatch_batch(0, chunk);
        }

        // Read logits directly from host-mapped memory (no hipMemcpy needed)
        let logits = unsafe {
            std::slice::from_raw_parts(self.activations.logits_mapped.host_ptr(), self.config.vocab_size)
        }.to_vec();

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Multi-GPU persistent worker decode for MoE models.
    /// Persistent worker on GPU 0 handles dense layers (60% faster than kbk).
    /// At MoE layers: worker paused, kbk dispatch across all GPUs, worker resumed.
    fn decode_step_persistent_multi_gpu(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::Instruction;
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile multi-GPU megakernel + launch workers on ALL GPUs
        if self.persistent_workers.is_none() {
            if self.megakernel_multi_gpu.is_none() {
                let mk = MegakernelProgram::compile_multi_gpu(self)?;
                self.megakernel_multi_gpu = Some(mk);
            }
            let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
            let shared_mem = 1024u32 * 4 + 256;
            let hs = self.config.hidden_size;
            let max_eis = self.config.layers.iter().filter_map(|l| match &l.ffn_type {
                crate::model::FfnType::MoE { expert_intermediate_size, .. } => Some(*expert_intermediate_size),
                _ => None,
            }).max().unwrap_or(0);
            let all_devices: Vec<_> = (0..num_gpus).map(|i| braidinfer_core::types::DeviceId(i as u32)).collect();
            let dispatch = PersistentDispatch::init_multi_gpu(self.device, &all_devices, shared_mem, hs, max_eis)
                .map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);

        }

        // Update host-side instructions
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe {
            std::ptr::copy_nonoverlapping(pos_data.as_ptr(), self.activations.position_ids.host_ptr(), 3);
        }
        let mk = self.megakernel_multi_gpu.as_mut().unwrap();
        mk.update_step_host_only(token_id, position)?;

        // Patch LM head to write to logits_mapped
        let n_inst = mk.instructions.len();
        mk.instructions[n_inst - 2].words[1] = self.activations.logits_mapped.as_write_ptr() as u64;

        let hs = self.config.hidden_size;
        let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
        #[allow(non_snake_case)]
        let OP_EXPERT_FFN: u32 = 35;

        // Collect per-GPU buffer pointers once
        let gpu_ptrs: Vec<_> = (0..num_gpus).map(|gpu| {
            let w = &self.multi_gpu.as_ref().unwrap().workers[gpu];
            (w.scratch_gate.as_write_ptr() as u64,
             w.scratch_up.as_write_ptr() as u64,
             w.scratch_act.as_write_ptr() as u64)
        }).collect();
        let ffn_down_stage_ptr = self.activations.ffn_down_stage.as_write_ptr() as u64;

        // Split instruction stream at OP_BARRIER markers into segments.
        // Dense segments batched to GPU 0. At OP_BARRIER: build per-GPU expert
        // FFN batches, dispatch to all GPUs in parallel via persistent workers.
        let mut segment: Vec<Instruction> = Vec::new();

        for inst in mk.instructions.iter() {
            let opcode = inst.words[0] & 0x7FFFFFFF;
            if opcode == 16 { break; }

            if opcode == 33 { // OP_BARRIER — MoE dispatch point
                // Flush pending dense segment to GPU 0
                if !segment.is_empty() {
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    for chunk in segment.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                        dispatch.dispatch_batch(0, chunk);
                    }
                    segment.clear();
                }

                let layer_idx = inst.words[3] as usize;
                let (k, eis) = match &self.config.layers[layer_idx].ffn_type {
                    crate::model::FfnType::MoE { num_active, expert_intermediate_size, .. } =>
                        (*num_active, *expert_intermediate_size),
                    _ => panic!("OP_BARRIER on non-MoE layer"),
                };

                // Read expert routing from host-mapped memory
                let expert_ids: &[i32] = unsafe {
                    std::slice::from_raw_parts(self.activations.moe_expert_ids.host_ptr() as *const i32, k)
                };
                let expert_weights: &[f32] = unsafe {
                    std::slice::from_raw_parts(self.activations.moe_expert_weights.host_ptr() as *const f32, k)
                };
                let dist_moe = self.distributed_moe[layer_idx].as_ref()
                    .expect("missing distributed MoE weights");

                // D2D_COPY normed → normed_stage on GPU 0 (for all GPUs to read)
                {
                    let mut copy_inst = Instruction::new(17, (hs as u32 + 255) / 256);
                    copy_inst.words[1] = self.activations.normed_stage.as_write_ptr() as u64;
                    copy_inst.words[2] = self.activations.normed.as_ptr() as u64;
                    copy_inst.words[3] = hs as u64;
                    self.persistent_workers.as_mut().unwrap().dispatch_batch(0, &[copy_inst]);
                }
                let act_ptr = self.activations.normed_stage.as_ptr() as u64;

                // Group experts by GPU
                let mut per_gpu: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_gpus];
                for j in 0..k {
                    let eid = expert_ids[j] as usize;
                    per_gpu[dist_moe.expert_device[eid]].push((eid, expert_weights[j]));
                }

                // Each GPU accumulates into its OWN output slot (no race condition).
                // Per-GPU output slots are in moe_output_slots (host-mapped).
                let output_slot_ptrs: Vec<u64> = (0..num_gpus).map(|gpu| {
                    self.persistent_workers.as_ref().unwrap().moe_output_slots[gpu].as_write_ptr() as u64
                }).collect();

                // Zero per-GPU output slots
                for gpu in 0..num_gpus {
                    if per_gpu[gpu].is_empty() { continue; }
                    let slot = &self.persistent_workers.as_ref().unwrap().moe_output_slots[gpu];
                    unsafe { std::ptr::write_bytes(slot.host_ptr(), 0, hs); }
                }

                // Build per-GPU OP_EXPERT_FFN instruction batches
                let mut gpu_batches: Vec<(usize, Vec<Instruction>)> = Vec::new();
                for gpu in 0..num_gpus {
                    if per_gpu[gpu].is_empty() { continue; }
                    let (sg, su, sa) = gpu_ptrs[gpu];
                    let mut batch = Vec::new();

                    for &(eid, ew) in &per_gpu[gpu] {
                        let buf = &dist_moe.expert_buffers[gpu];
                        let slot = buf.slot_map[eid].expect("expert not on GPU");
                        let gu_offset = slot * dist_moe.gate_up_expert_stride;
                        let d_offset = slot * dist_moe.down_expert_stride;
                        let gate_up_ptr = unsafe { buf.gate_up.as_ptr().add(gu_offset) } as u64;
                        let down_ptr = unsafe { buf.down.as_ptr().add(d_offset) } as u64;

                        let mut inst = Instruction::new(OP_EXPERT_FFN, 0);
                        inst.words[1] = sg;
                        inst.words[2] = su;
                        inst.words[3] = sa;
                        inst.words[4] = output_slot_ptrs[gpu]; // per-GPU slot (no race)
                        inst.words[5] = gate_up_ptr;
                        inst.words[6] = down_ptr;
                        inst.words[7] = act_ptr;
                        inst.words[8] = ((eis as u64) << 32) | (hs as u64);
                        inst.words[9] = dist_moe.gate_up_row_stride as u64;
                        inst.words[10] = if dist_moe.has_gate_proj { 1 } else { 0 };
                        inst.words[11] = ew.to_bits() as u64;
                        batch.push(inst);
                    }
                    gpu_batches.push((gpu, batch));
                }

                // Dispatch expert FFN:
                // - GPU 0: use fat dense worker (OP_EXPERT_FFN via dispatch_batch)
                // - GPUs 1+: use lean MoE workers (high occupancy, moe_dispatch_fire)
                let dispatch = self.persistent_workers.as_mut().unwrap();
                let mut moe_pending: Vec<(usize, u32)> = Vec::new();

                // Dispatch expert FFN:
                // - GPU 0: fat dense worker (OP_EXPERT_FFN)
                // - GPUs 1+: lean MoE workers with per-block scratch (no grid.sync)
                for (gpu, batch) in &gpu_batches {
                    if *gpu == 0 {
                        dispatch.dispatch_batch(0, batch);
                    } else {
                        let moe_idx = gpu - 1;
                        let seq = dispatch.moe_dispatch_fire(moe_idx, batch);
                        moe_pending.push((moe_idx, seq));
                    }
                }
                for &(idx, seq) in &moe_pending {
                    dispatch.moe_wait_ack(idx, seq);
                }

                // CPU sums per-GPU output slots into ffn_down_stage (no race — sequential)
                unsafe { std::ptr::write_bytes(self.activations.ffn_down_stage.host_ptr(), 0, hs); }
                let out: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(self.activations.ffn_down_stage.host_ptr(), hs)
                };
                for gpu in 0..num_gpus {
                    if per_gpu[gpu].is_empty() { continue; }
                    let slot_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(dispatch.moe_output_slots[gpu].host_ptr(), hs)
                    };
                    for i in 0..hs { out[i] += slot_data[i]; }
                }

                // Copy ffn_down_stage → ffn_down for post-barrier instructions
                {
                    let mut copy_inst = Instruction::new(17, (hs as u32 + 255) / 256);
                    copy_inst.words[1] = self.activations.ffn_down.as_write_ptr() as u64;
                    copy_inst.words[2] = ffn_down_stage_ptr;
                    copy_inst.words[3] = hs as u64;
                    self.persistent_workers.as_mut().unwrap().dispatch_batch(0, &[copy_inst]);
                }

                continue;
            }

            segment.push(inst.clone());
        }

        // Flush final segment
        if !segment.is_empty() {
            let dispatch = self.persistent_workers.as_mut().unwrap();
            for chunk in segment.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                dispatch.dispatch_batch(0, chunk);
            }
        }

        let logits = unsafe {
            std::slice::from_raw_parts(self.activations.logits_mapped.host_ptr(), self.config.vocab_size)
        }.to_vec();

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// MoE decode step: kernel-by-kernel execution with MoE FFN dispatch.
    fn decode_step_moe(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;

        // Set position_ids for mRoPE/RoPE
        let pos_data = [position as i32, position as i32, position as i32];
        unsafe { std::ptr::copy_nonoverlapping(pos_data.as_ptr(), self.activations.position_ids.host_ptr(), pos_data.len()) };

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;

        if self.debug_nan {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("embed(tok={token_id}): max_abs={max_abs:.4e}, first5={:.4?}", &buf[..5]);
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
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    self.gdn_forward(layer_i, gdn_idx)?;
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    self.mamba2_forward(layer_i, mamba2_idx)?;
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
                eprintln!("L{layer_i} ({:?}): {nan_count} NaN, max_abs={max_abs:.2e}", self.config.layers[layer_i].layer_type);
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace.as_mut().unwrap().write_checkpoint(&format!("L{layer_i}.post_mixer"), &buf);
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
                    unsafe { self.ffn_forward(&*post_norm_p, (*w_gate_p).as_bf16(), (*w_up_p).as_bf16(), (*w_down_p).as_bf16())?; }
                } else {
                    // Unfused path for quantized weights
                    unsafe {
                        d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
                    }
                    unsafe {
                        self.kernels.rmsnorm.forward(
                            &mut self.activations.normed, &self.activations.hidden, &*post_norm_p,
                            1, hs as u32, eps, self.config.rms_norm_one_plus_w, &self.stream)?;
                    }
                    unsafe {
                        (*w_gate_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_gate, &self.activations.normed,
                            is as u32, hs as u32, &self.stream)?;
                        (*w_up_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_up, &self.activations.normed,
                            is as u32, hs as u32, &self.stream)?;
                    }
                    self.kernels.silu_mul.forward(
                        &mut self.activations.ffn_act, &self.activations.ffn_gate, &self.activations.ffn_up,
                        is as u32, &self.stream)?;
                    unsafe {
                        (*w_down_p).forward(&self.kernels.linear_proj,
                            &mut self.activations.ffn_down, &self.activations.ffn_act,
                            hs as u32, is as u32, &self.stream)?;
                    }
                    self.kernels.residual_add.forward(
                        &mut self.activations.hidden, &self.activations.ffn_down, &self.activations.residual,
                        hs as u32, &self.stream)?;
                }
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace.as_mut().unwrap().write_checkpoint(&format!("L{layer_i}.post_ffn"), &buf);
            }
        }

        // Final RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &self.final_norm_weight,
            1, hs, eps, self.config.rms_norm_one_plus_w, &self.stream,
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
            self.config.vocab_size as u32, hs, &self.stream,
        )?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        if self.trace.is_some() {
            let mut hid_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut hid_buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("final_hidden", &hid_buf);

            let mut norm_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.normed.copy_to_host(&mut norm_buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("final_norm", &norm_buf);

            // Capture top-10 logits (token_id + value pairs as f32)
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed.iter().take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace.as_mut().unwrap().write_checkpoint("top10_logits", &top10);
        }

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, false)
    }

    /// Run a single decode step with quantized KV cache (4-bit residual_pc).
    /// Sealed chunks are quantized to int4; active chunk stays f32.
    pub fn decode_step_paged_quantized(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, true)
    }

    fn decode_step_paged_inner(&mut self, token_id: u32, position: u32, quantized: bool) -> Result<Vec<f32>, ModelError> {
        let max_chunks = (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS;

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
            assert_eq!(mk.quantized_kv, quantized,
                "cannot mix decode_step_paged and decode_step_paged_quantized on the same model");
        }

        // Lazy-init: f32 PageAllocator (staging) and SequenceState
        if self.page_allocator.is_none() {
            self.page_allocator = Some(PageAllocator::new(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
            self.paged_seq = Some(SequenceState::new(CHUNK_TOKENS as u32));
        }

        // Lazy-init: quantized PageAllocator
        if quantized && self.quant_allocator.is_none() {
            self.quant_allocator = Some(PageAllocator::new_quantized(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
        }

        // append_token
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(alloc_mut)?;
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
            mk.post_step_paged(position, seq_mut, alloc_mut, q_alloc, &self.config, &self.stream)?;
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
            self.checkpoint_pool = Some(RecurrentCheckpointPool::new(
                self.device,
                &self.config,
                1,
            )?);
        }
        // Free previous slot before allocating new one (ring buffer with capacity 1)
        if let Some(prev) = self.last_checkpoint_slot.take() {
            self.checkpoint_pool.as_mut().unwrap().free(prev);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> = self.gdn_states.iter().map(|s| &s.recurrent).collect();
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

        // MoE and quantized-weight models can't use megakernel prefill
        // (batched FFN fused kernel only handles bf16).
        // Fall back to sequential decode.
        let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        let has_quant = self.config.weight_quant != WeightQuantMode::Bf16;
        if has_moe || has_quant {
            let mut logits = vec![];
            for (i, &tok) in tokens.iter().enumerate() {
                logits = self.decode_step_moe(tok, i as u32)?;
            }
            return Ok(logits);
        }

        let mut pos = 0u32;
        for chunk in tokens.chunks(CHUNK_TOKENS) {
            let mut bufs = PrefillBuffers::alloc(self.device, &self.config, chunk.len())?;
            let program = MegakernelProgram::compile_prefill(self, chunk, pos, &mut bufs)?;
            program.execute(&self.stream)?;
            self.stream.synchronize()?;
            pos += chunk.len() as u32;
            if pos < tokens.len() as u32 {
                let _slot = self.save_recurrent_checkpoint()?;
            }
        }
        self.read_logits()
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
        let pool = self.checkpoint_pool.as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self.gdn_states.iter_mut().map(|s| &mut s.recurrent).collect();
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

    pub fn decode_step_traced(&mut self, token_id: u32, position: u32) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
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

        unsafe { d2d_copy_f32(&mut self.activations.normed, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden, &self.activations.normed,
            &self.final_norm_weight, 1, hs, self.config.rms_norm_eps, self.config.rms_norm_one_plus_w, &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits, &self.embed_weight,
            &self.activations.hidden, vs, hs, &self.stream,
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

    pub fn gdn_layer0_trace(&mut self, token_id: u32) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed, &self.activations.hidden,
            &w.input_norm, 1, hs, eps, self.config.rms_norm_one_plus_w, &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        w.w_qkv.forward(&self.kernels.linear_proj,
            &mut self.activations.qkv, &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd, hs, &self.stream)?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        w.w_a.forward(&self.kernels.linear_proj,
            &mut self.activations.a_proj, &self.activations.normed, nvh_traced, hs, &self.stream)?;
        w.w_b.forward(&self.kernels.linear_proj,
            &mut self.activations.b_proj, &self.activations.normed, nvh_traced, hs, &self.stream)?;
        w.w_z.forward(&self.kernels.linear_proj,
            &mut self.activations.z_proj, &self.activations.normed, nvh_traced * vd, hs, &self.stream)?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nvh_traced * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, conv_q_len, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, conv_q_len + conv_k_len, conv_v_len, &self.stream)?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            d2d_copy_u16(&mut conv_w_q, 0, &w.conv1d_weight, 0, conv_q_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_k, 0, &w.conv1d_weight, conv_q_len * ck_usize, conv_k_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_v, 0, &w.conv1d_weight, (conv_q_len + conv_k_len) * ck_usize, conv_v_len * ck_usize, &self.stream)?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(&mut cs_q, 0, &self.gdn_conv_states[0], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut cs_k, 0, &self.gdn_conv_states[0], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut cs_v, 0, &self.gdn_conv_states[0], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(&mut cs_q, &self.activations.q_gdn, &conv_w_q, &mut conv_out_q, conv_q_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_k, &self.activations.k_gdn, &conv_w_k, &mut conv_out_k, conv_k_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_v, &self.activations.v_gdn, &conv_w_v, &mut conv_out_v, conv_v_len as u32, ck, &self.stream)?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &conv_out_q, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &conv_out_k, 0, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &conv_out_v, 0, conv_v_len, &self.stream)?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn, &w.a_log, &self.activations.a_proj,
            &w.dt_bias, nh, &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn, &self.activations.k_gdn, &self.activations.v_gdn,
            &self.activations.gate_gdn, &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent, &mut self.activations.recurrent_out,
            nvh_traced, kd, vd, gqa_traced, &self.stream,
        )?;
        traces.push(("recurrent_out".into(), self.read_buf(&self.activations.recurrent_out)?));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated, &self.activations.recurrent_out,
            &self.activations.z_proj, &w.output_norm, nvh_traced, vd, eps, &self.stream,
        )?;
        traces.push(("normed_gated".into(), self.read_buf(&self.activations.normed_gated)?));

        // out_proj
        w.w_out.forward(&self.kernels.linear_proj,
            &mut self.activations.out_proj, &self.activations.normed_gated,
            hs, nvh_traced * vd, &self.stream)?;
        traces.push(("out_proj".into(), self.read_buf(&self.activations.out_proj)?));

        // Residual
        unsafe { d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden, &self.activations.out_proj,
            &self.activations.residual, hs, &self.stream,
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
        let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
        let zeros_kv = vec![0.0f32; kv_size];
        for cache in &mut self.kv_caches {
            cache.k.copy_from_host(&zeros_kv)?;
            cache.v.copy_from_host(&zeros_kv)?;
        }
        self.seq_len = 0;
        // Free quantized KV slots back to pool
        if let (Some(seq), Some(q_alloc)) = (self.paged_seq.as_mut(), self.quant_allocator.as_mut()) {
            seq.free_quant_slots(q_alloc);
        }
        Ok(())
    }
}
