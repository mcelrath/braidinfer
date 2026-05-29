use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};
use braidinfer_hip::stream::Stream;

use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram};
use crate::paged_kv::{HostPageAllocator, PageAllocator, RecurrentCheckpointPool, SequenceState};

pub use crate::kernel::AllKernels;

use crate::config::ModelConfig;
use crate::weights::{
    ActivationBuffers, GdnState, LayerWeights, Mamba2State, ModelError, MoeWeights,
};

mod decode; // decode_step_* implementations and dispatch helpers
mod model_load; // weight loading and initialization
mod moe_forward; // moe_ffn_forward
mod state; // save/restore checkpoint, prefill, read_hidden, reset_state

// ---- Main model struct ----

pub struct Model {
    // MUST be declared first: dropped first, shuts down cooperative kernels before any hipFree.
    // DeviceBuffer::drop → hipFree → SyncAllStreams blocks if cooperative kernel is still running.
    pub(crate) persistent_workers: Option<crate::persistent_dispatch::PersistentDispatch>,
    // GPU-native P2P MoE dispatch: cooperative kernels on GPUs 1-3. Drop before other GPU 1-3 resources.
    pub(crate) moe_p2p: Option<crate::moe_p2p::MoeP2pContext>,
    // Singleton watchdog thread shared by all MegakernelProgram + PersistentDispatch instances.
    // Declared after persistent_workers/moe_p2p: drops AFTER them, so cooperative kernels signal
    // exit before the watchdog thread stops polling.
    // Underscore-prefixed to suppress dead_code warning — field is never read, only its Drop matters.
    #[allow(dead_code)]
    pub(crate) watchdog: std::sync::Arc<crate::watchdog::WatchdogThread>,
    pub config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    pub(crate) kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) lm_head_weight: DeviceBuffer<u16>, // separate from embed when tie_word_embeddings=false
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    // SAFETY: distributed_moe MUST be declared before moe_weights so it drops first.
    // DistributedMoeWeights::gpu0_gate_up_base may point into moe_weights[i].expert_gate_up
    // (the non-bqnt path in distribute_moe_weights_from_ref). If moe_weights dropped first,
    // gpu0_gate_up_base would dangle. Drop order = declaration order in Rust structs.
    pub(crate) distributed_moe: Vec<Option<crate::weights::DistributedMoeWeights>>,
    pub(crate) moe_weights: Vec<Option<MoeWeights>>, // per-layer MoE FFN (None for dense FFN layers)
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) mamba2_states: Vec<Mamba2State>,
    pub(crate) seq_len: u32,
    pub(crate) prefill_bufs: Option<crate::megakernel::PrefillBuffers>,
    // Paged KV path (lazy-init)
    pub(crate) megakernel_paged: Option<MegakernelProgram>,
    pub(crate) page_allocator: Option<PageAllocator>,
    pub(crate) quant_allocator: Option<PageAllocator>,
    /// Host-RAM KV tier (Phase C, braidinfer-4n5). None when
    /// `BRAIDINFER_HOST_KV_CHUNKS` is unset or zero (default: host tier OFF,
    /// behavior byte-identical to pre-Phase-C). Constructed lazily in
    /// `ensure_paged_decode_state` on the first decode/prefill call that
    /// enables paged KV, so hipHostMalloc fires before the persistent worker
    /// is launched and the cooperative kernel holds GPU CUs.
    ///
    /// # Drop ordering
    ///
    /// `HostPageAllocator` holds a `ManuallyDrop<PinnedBuffer<u8>>` pool.
    /// `hipHostFree` (called from `PinnedBuffer::drop`) must NOT fire while
    /// the persistent cooperative worker is running.  The pool is explicitly
    /// freed in `reset_state()` AFTER `drop(self.persistent_workers.take())`
    /// — mirrors the `ManuallyDrop<DeviceBuffer>` pattern in
    /// `PersistentDispatch::drop`.  Rust's automatic field-drop order would
    /// run `host_page_allocator` BEFORE `persistent_workers` (fields drop
    /// in reverse declaration order), so we rely on `reset_state` to tear
    /// down in the correct sequence rather than declaration order.
    pub(crate) host_page_allocator: Option<HostPageAllocator>,
    pub(crate) paged_seq: Option<SequenceState>,
    // bd srg6.7: host-mapped page/position tables for paged-prefill writer.
    // MappedHostBuffer (not DeviceBuffer) because writes happen while persistent
    // worker holds GPU CUs — copy_from_host would panic; host_ptr.write_volatile
    // is safe. Sized for max prompt: max_chunks u64s + 3*max_seq_len i32s.
    pub(crate) prefill_paged_page_table: Option<MappedHostBuffer<u64>>,
    pub(crate) prefill_paged_position_table: Option<MappedHostBuffer<i32>>,
    pub(crate) checkpoint_pool: Option<RecurrentCheckpointPool>,
    pub(crate) last_checkpoint_slot: Option<u32>,
    pub(crate) debug_nan: bool,
    pub(crate) has_moe: bool,          // cached at load time: any layer has FfnType::MoE
    pub(crate) debug_p2p_hidden: bool, // cached from DEBUG_P2P_HIDDEN env var at load time
    pub(crate) weight_prefix: String, // tensor name prefix (e.g. "model.language_model.")
    // Multi-GPU expert parallel (None for single-GPU)
    pub(crate) multi_gpu: Option<crate::multi_gpu::MultiGpuContext>,
    // Multi-GPU megakernel programs
    pub(crate) megakernel_multi_gpu_p2p: Option<MegakernelProgram>,
    /// SDMA-based unified tracer (Phase 2b). Replaces DecodeMirror.
    /// Constructed in ensure_moe_workers_started (for multi-GPU MoE) via
    /// Tracer::from_env, or with ProbeFilter::All when BRAIDINFER_DECODE_MIRROR=1
    /// and BRAIDINFER_TRACE is unset (deprecated compat shim — Phase 5 will
    /// consolidate env vars). Field is declared AFTER persistent_workers in source
    /// order; because Rust drops fields in REVERSE declaration order, tracer
    /// drops FIRST, releasing its PinnedBuffer shadows before PersistentDispatch
    /// releases the SDMA streams they borrowed.
    pub(crate) tracer: crate::tracer::Tracer,
}

// ---- Model impl ----

impl Model {
    /// True if any layer's FfnType is MoE. Cached at load time.
    pub fn has_moe(&self) -> bool {
        self.has_moe
    }

    fn max_paged_chunks(&self) -> usize {
        (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS
    }

    fn ensure_paged_decode_state(&mut self, quantized: bool) -> Result<(), ModelError> {
        let max_chunks = self.max_paged_chunks();
        // bd 4n5 Phase D: BRAIDINFER_VRAM_KV_CHUNKS caps the f32 VRAM chunk pool
        // below max_paged_chunks, forcing append_token to spill to the host tier
        // (HostPageAllocator) once exhausted. Test/bench override; unset = full pool
        // (current behavior). Only caps the VRAM (page_allocator) pool, NOT the quant
        // pool. Capping below max_chunks without the host tier enabled would just OOM,
        // so this is meaningful only with BRAIDINFER_HOST_KV_CHUNKS also set.
        let vram_chunks = std::env::var("BRAIDINFER_VRAM_KV_CHUNKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map(|n| n.min(max_chunks))
            .unwrap_or(max_chunks);
        if self.page_allocator.is_none() {
            self.page_allocator = Some(PageAllocator::new(
                self.device,
                &self.config,
                CHUNK_TOKENS,
                vram_chunks as u32,
            )?);
            self.paged_seq = Some(SequenceState::new(CHUNK_TOKENS as u32));
        }

        if quantized && self.quant_allocator.is_none() {
            self.quant_allocator = Some(PageAllocator::new_quantized(
                self.device,
                &self.config,
                CHUNK_TOKENS,
                max_chunks as u32,
            )?);
        }

        // Phase C (braidinfer-4n5): host-RAM KV tier.
        // Construct HostPageAllocator lazily here — before the persistent worker
        // is spawned — so hipHostMalloc fires with GPU CUs free.
        // Default: host tier OFF (env unset or zero → current behavior unchanged).
        if self.host_page_allocator.is_none() {
            if let Ok(s) = std::env::var("BRAIDINFER_HOST_KV_CHUNKS") {
                if let Ok(n) = s.parse::<u32>() {
                    if n > 0 {
                        let chunk_bytes = self
                            .page_allocator
                            .as_ref()
                            .expect("page_allocator initialized above")
                            .chunk_bytes();
                        // HostPageAllocator::new returns None on hipHostMalloc
                        // failure (ENOMEM) and logs a clear warning — graceful
                        // disable per OQ-2.  None here means host tier stays
                        // disabled; no hard error.
                        self.host_page_allocator =
                            HostPageAllocator::new(chunk_bytes, n);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn tracer(&self) -> &crate::tracer::Tracer {
        &self.tracer
    }

    /// SDMA-capture all GDN/Mamba2 recurrent SSM state buffers into the tracer.
    /// Safe under the persistent cooperative worker (SDMA engine is independent
    /// of CUs). Probes are named `gdn_state_{layer_idx}`. No-op if tracer is
    /// disabled. Call after a decode_step to inspect the recurrent matrices.
    pub fn snapshot_gdn_states(&mut self) -> HipResult<()> {
        if !self.tracer.enabled() {
            return Ok(());
        }
        use std::borrow::Cow;
        for (i, state) in self.gdn_states.iter().enumerate() {
            self.tracer.capture_f32(
                0,
                crate::tracer::Probe::Custom(Cow::Owned(format!("gdn_state_{i}"))),
                &state.recurrent,
            )?;
        }
        self.tracer.drain()
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
        // Mirror to each worker's per-GPU position_ids buffer. activations
        // .position_ids is non-portable host-mapped (only GPU 0's device_ptr
        // is valid); workers' MROPE in dispatch_head_parallel_attention needs
        // a pointer valid on its own device.
        if let Some(mgpu) = self.multi_gpu.as_ref() {
            for worker in &mgpu.workers {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pos_data.as_ptr(),
                        worker.position_ids_local.host_ptr(),
                        3,
                    );
                }
            }
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
        // Persistent path: logits already copied to host-mapped buffer by
        // decode_step_persistent / decode_step_p2p — do CPU argmax.
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
    }

    /// Run a single decode step. Returns logits [vocab_size].
    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        if self.multi_gpu.is_some() {
            return self.decode_step_persistent_multi_gpu(token_id, position);
        }
        // bd 9gmh Phase 3: PERSISTENT is always true — always route through the
        // cooperative megakernel worker path.
        self.decode_step_persistent(token_id, position)
    }
}
