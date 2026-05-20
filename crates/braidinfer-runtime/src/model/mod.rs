use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;

use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram};
use crate::paged_kv::{PageAllocator, RecurrentCheckpointPool, SequenceState};

// Re-export weight types and config for backward compatibility
pub use crate::config::*;
pub use crate::kernel::AllKernels;
pub use crate::weights::*;

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
    pub(crate) legacy_kv_caches: Option<Vec<KvCache>>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) mamba2_states: Vec<Mamba2State>,
    pub(crate) seq_len: u32,
    pub(crate) megakernel_prefill: Option<MegakernelProgram>,
    pub(crate) megakernel_prefill_partial: Option<MegakernelProgram>,
    pub(crate) megakernel_prefill_partial_n: usize,
    /// Cache of segment megakernel programs keyed by (layer_start, layer_end, chunk_len, start_pos).
    pub(crate) megakernel_prefill_segments: std::collections::HashMap<(usize, usize, usize, usize), MegakernelProgram>,
    pub(crate) prefill_bufs: Option<crate::megakernel::PrefillBuffers>,
    // Paged KV path (lazy-init)
    pub(crate) megakernel_paged: Option<MegakernelProgram>,
    pub(crate) page_allocator: Option<PageAllocator>,
    pub(crate) quant_allocator: Option<PageAllocator>,
    pub(crate) paged_seq: Option<SequenceState>,
    pub(crate) paged_page_table: Option<DeviceBuffer<u64>>,
    pub(crate) paged_position_table: Option<DeviceBuffer<i32>>,
    pub(crate) checkpoint_pool: Option<RecurrentCheckpointPool>,
    pub(crate) last_checkpoint_slot: Option<u32>,
    pub(crate) debug_nan: bool,
    pub(crate) has_moe: bool,          // cached at load time: any layer has FfnType::MoE
    pub(crate) persistent: bool,       // cached from PERSISTENT env var at load time
    pub(crate) kv_quant: bool,         // cached from KV_QUANT env var at load time
    pub(crate) sync_debug: bool,       // cached from SYNC_DEBUG env var at load time
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
    pub fn tracer(&self) -> &crate::tracer::Tracer {
        &self.tracer
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
        if is_multi_gpu {
            if self.persistent {
                return self.decode_step_persistent_multi_gpu(token_id, position);
            }
            return Err(ModelError::InvalidConfig(
                "Multi-GPU inference requires persistent mode (set PERSISTENT=1)".to_string(),
            ));
        }
        if self.persistent {
            // Critical guard: KV_QUANT under persistent is not yet wired through
            // (post_step_paged is not invoked in the persistent path, so chunks
            // never seal/quantize). Return InvalidConfig rather than silently
            // running unquantized. Tracked as a follow-up under braidinfer-8gz.
            if self.kv_quant {
                return Err(ModelError::InvalidConfig(
                    "KV_QUANT=1 with PERSISTENT=1 is not yet supported. \
                     Either unset KV_QUANT or unset PERSISTENT.".into(),
                ));
            }
            return self.decode_step_persistent(token_id, position);
        }
        if self.kv_quant {
            return self.decode_step_paged_quantized(token_id, position);
        }
        self.decode_step_paged(token_id, position)
    }
}
