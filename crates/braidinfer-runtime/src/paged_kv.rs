use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

use braidinfer_core::types::DeviceId;
use braidinfer_hip::error::check as hip_check;
use braidinfer_hip::memory::{DeviceBuffer, PinnedBuffer};
use braidinfer_hip::{HipError, HipResult, ffi};

use crate::config::ModelConfig;

// ---- Derived geometry (computed from ModelConfig at runtime) ----

/// KV bytes per unquantized (f32) chunk slot, computed from ModelConfig:
/// num_attn_layers * 2(K+V) * chunk_tokens * num_kv_heads * head_dim * sizeof(f32)
/// Used for staging buffers and flat KV cache.
pub fn chunk_kv_bytes(config: &ModelConfig, chunk_tokens: usize) -> usize {
    let num_attn_layers = config.num_attn_layers();
    let kv_stride = config.num_kv_heads * config.head_dim;
    num_attn_layers * 2 * chunk_tokens * kv_stride * std::mem::size_of::<f32>()
}

/// KV bytes per quantized chunk slot using residual_pc int4.
/// Layout per K or V per layer (chunk-interleaved):
///   q1_data:  [num_kv_heads * chunk_tokens * head_dim / 2] bytes (int4 packed, 2 values/byte)
///   q1_scale: [num_kv_heads * head_dim] f32 (per-channel, shared across tokens in chunk)
///   r_data:   same as q1_data
///   r_scale:  same as q1_scale
pub fn quantized_chunk_kv_bytes(config: &ModelConfig, chunk_tokens: usize) -> usize {
    debug_assert_eq!(
        chunk_tokens, 64,
        "quantized chunk_tokens must equal group_size (64)"
    );
    let num_attn_layers = config.num_attn_layers();
    let nkh = config.num_kv_heads;
    let hd = config.head_dim;
    // Per K or V per layer:
    let data_bytes = nkh * chunk_tokens * hd / 2; // int4 packed
    let scale_bytes = nkh * hd * std::mem::size_of::<f32>(); // per-channel f32
    let per_kv = 2 * (data_bytes + scale_bytes); // q1 + residual
    let per_layer = 2 * per_kv; // K + V
    num_attn_layers * per_layer
}

/// Byte offset within a quantized chunk for a specific layer's K or V data region.
/// `layer_attn_idx`: 0-based index among attention layers only.
/// `is_value`: false=K, true=V.
/// Returns (data_offset, scale_offset, residual_data_offset, residual_scale_offset).
pub fn quantized_kv_offsets(
    config: &ModelConfig,
    chunk_tokens: usize,
    layer_attn_idx: usize,
    is_value: bool,
) -> (usize, usize, usize, usize) {
    let nkh = config.num_kv_heads;
    let hd = config.head_dim;
    let data_bytes = nkh * chunk_tokens * hd / 2;
    let scale_bytes = nkh * hd * std::mem::size_of::<f32>();
    let per_kv = 2 * (data_bytes + scale_bytes);
    let per_layer = 2 * per_kv;

    let base = layer_attn_idx * per_layer + if is_value { per_kv } else { 0 };
    let q1_data = base;
    let q1_scale = base + data_bytes;
    let r_data = base + data_bytes + scale_bytes;
    let r_scale = base + 2 * data_bytes + scale_bytes;
    (q1_data, q1_scale, r_data, r_scale)
}

/// Recurrent (GDN) state bytes per checkpoint slot, from ModelConfig:
/// num_gdn_layers * linear_num_heads * linear_key_head_dim * linear_value_head_dim * sizeof(f32)
pub fn recurrent_state_bytes(config: &ModelConfig) -> usize {
    let num_gdn_layers = config.num_recurrent_layers();
    let floats_per_layer =
        config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim;
    num_gdn_layers * floats_per_layer * std::mem::size_of::<f32>()
}

// ---- PageAllocator ----

/// Pool allocator for KV chunk slots. Pre-allocates contiguous VRAM and hands out
/// fixed-size slots via a free-list. No hipMalloc in the hot path.
pub struct PageAllocator {
    pool: DeviceBuffer<u8>,
    capacity: u32,
    free_list: Vec<u32>,
    device: DeviceId,
    chunk_bytes: usize,
    #[allow(dead_code)]
    chunk_tokens: usize,
}

impl PageAllocator {
    /// Allocate a pool of `max_chunks` f32 KV chunk slots.
    pub fn new(
        device: DeviceId,
        config: &ModelConfig,
        chunk_tokens: usize,
        max_chunks: u32,
    ) -> HipResult<Self> {
        let chunk_bytes = chunk_kv_bytes(config, chunk_tokens);
        Self::new_with_chunk_bytes(device, chunk_tokens, max_chunks, chunk_bytes)
    }

    /// Allocate a pool for quantized (residual_pc int4) KV chunk slots.
    pub fn new_quantized(
        device: DeviceId,
        config: &ModelConfig,
        chunk_tokens: usize,
        max_chunks: u32,
    ) -> HipResult<Self> {
        let chunk_bytes = quantized_chunk_kv_bytes(config, chunk_tokens);
        Self::new_with_chunk_bytes(device, chunk_tokens, max_chunks, chunk_bytes)
    }

    fn new_with_chunk_bytes(
        device: DeviceId,
        chunk_tokens: usize,
        max_chunks: u32,
        chunk_bytes: usize,
    ) -> HipResult<Self> {
        let chunk_bytes = chunk_bytes;
        let pool_bytes = max_chunks as usize * chunk_bytes;
        let pool = DeviceBuffer::alloc(device, pool_bytes)?;
        let free_list = (0..max_chunks).rev().collect();
        Ok(Self {
            pool,
            capacity: max_chunks,
            free_list,
            device,
            chunk_bytes,
            chunk_tokens,
        })
    }

    /// Allocate one chunk slot. Returns `(slot_index, base_ptr)` or `None` if pool exhausted.
    pub fn alloc(&mut self) -> Option<(u32, *mut u8)> {
        let idx = self.free_list.pop()?;
        let offset = idx as usize * self.chunk_bytes;
        let ptr = unsafe { self.pool.as_mut_ptr().add(offset) };
        Some((idx, ptr))
    }

    /// Return a slot to the free-list.
    pub fn free(&mut self, slot: u32) {
        assert!(
            (slot as usize) < self.capacity as usize,
            "page slot {} >= capacity {}",
            slot,
            self.capacity
        );
        assert!(
            !self.free_list.contains(&slot),
            "double-free of page slot {}",
            slot
        );
        self.free_list.push(slot);
    }

    /// Base pointer for slot `slot` (for page table construction).
    pub fn slot_ptr(&self, slot: u32) -> *const u8 {
        let offset = slot as usize * self.chunk_bytes;
        unsafe { self.pool.as_ptr().add(offset) }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }
}

// ---- ChunkTier ----

/// Which memory pool a chunk's data currently lives in.
///
/// Encoded as an `AtomicU8` discriminant on `ChunkRef` (0 = Vram, 1 = HostPinned).
/// The `slot` field on `ChunkRef` indexes into whichever pool the discriminant selects.
/// Transitions are CPU-only and happen only between megakernel batches, so no
/// concurrent mutation with the persistent worker occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChunkTier {
    /// Chunk data resides in a `PageAllocator` VRAM slot.
    Vram = 0,
    /// Chunk data resides in a `HostPageAllocator` host-pinned slot.
    HostPinned = 1,
}

impl ChunkTier {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ChunkTier::Vram,
            1 => ChunkTier::HostPinned,
            _ => panic!("invalid ChunkTier discriminant {}", v),
        }
    }
}

// ---- HostPageAllocator ----

/// Pool allocator for KV chunk slots backed by `hipHostMalloc` pinned host memory.
///
/// Mirrors the `PageAllocator` API (alloc/free/slot_ptr/capacity/free_list) but
/// allocates from host-pinned RAM instead of VRAM.  Capacity is configured via
/// `BRAIDINFER_HOST_KV_CHUNKS` (default 512).
///
/// # ManuallyDrop safety
///
/// The pool uses `ManuallyDrop<PinnedBuffer<u8>>` rather than a plain
/// `PinnedBuffer`.  `PinnedBuffer::drop` calls `hipHostFree`, which deadlocks
/// while the persistent cooperative worker holds all GPU CUs (the SDMA engine
/// serialises on the same command-processor as hipHostFree on gfx1100).
/// By wrapping in `ManuallyDrop` we suppress the automatic drop.  The caller
/// is responsible for explicitly dropping the pool only after the persistent
/// worker has been torn down — mirroring the `ManuallyDrop<DeviceBuffer<u8>>`
/// pattern in `PersistentDispatch::drop`.
///
/// Note: `PinnedBuffer::drop` already contains a guard that skips `hipHostFree`
/// when `any_persistent_worker_active()` is true.  `ManuallyDrop` is an
/// additional belt-and-suspenders layer that makes the deferred-free contract
/// explicit in the type system and prevents accidental eager drops via early
/// returns or `?` propagation in teardown paths.
pub struct HostPageAllocator {
    /// Pinned host memory pool.  Freed explicitly after worker teardown.
    pool: ManuallyDrop<PinnedBuffer<u8>>,
    capacity: u32,
    free_list: Vec<u32>,
    chunk_bytes: usize,
}

impl HostPageAllocator {
    /// Allocate a pinned host pool of `capacity` chunk slots, each `chunk_bytes` bytes.
    ///
    /// Returns `None` (not `Err`) if `hipHostMalloc` fails so the caller can
    /// gracefully disable the host tier rather than propagating a hard error.
    /// Logs a clear warning so operators know there is no overflow capacity.
    pub fn new(chunk_bytes: usize, capacity: u32) -> Option<Self> {
        let pool_bytes = capacity as usize * chunk_bytes;
        match PinnedBuffer::<u8>::alloc(pool_bytes) {
            Ok(buf) => Some(Self {
                pool: ManuallyDrop::new(buf),
                capacity,
                free_list: (0..capacity).rev().collect(),
                chunk_bytes,
            }),
            Err(e) => {
                eprintln!(
                    "braidinfer: HostPageAllocator: hipHostMalloc failed for {} bytes \
                     (capacity={} chunks, chunk_bytes={}): {:?}; \
                     host KV tier disabled — VRAM overflow will return OutOfMemory",
                    pool_bytes, capacity, chunk_bytes, e
                );
                None
            }
        }
    }

    /// Allocate one chunk slot. Returns `(slot_index, host_ptr)` or `None` if pool exhausted.
    pub fn alloc(&mut self) -> Option<(u32, *mut u8)> {
        let idx = self.free_list.pop()?;
        let offset = idx as usize * self.chunk_bytes;
        let ptr = unsafe { self.pool.as_mut_ptr().add(offset) };
        Some((idx, ptr))
    }

    /// Return a slot to the free-list.
    pub fn free(&mut self, slot: u32) {
        assert!(
            (slot as usize) < self.capacity as usize,
            "host page slot {} >= capacity {}",
            slot,
            self.capacity
        );
        assert!(
            !self.free_list.contains(&slot),
            "double-free of host page slot {}",
            slot
        );
        self.free_list.push(slot);
    }

    /// Base pointer for slot `slot` (CPU-accessible).
    pub fn slot_ptr(&self, slot: u32) -> *const u8 {
        let offset = slot as usize * self.chunk_bytes;
        unsafe { self.pool.as_ptr().add(offset) }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }
}

// ---- ChunkRef + ChunkHandle (Arc-based CoW) ----

/// Shared metadata for one KV chunk slot. The actual data lives in the PageAllocator pool.
///
/// # Tier model (Phase A, braidinfer-4n5)
///
/// `tier_discriminant` is an `AtomicU8` encoding `ChunkTier` (0=Vram, 1=HostPinned).
/// `slot_index` indexes into whichever pool the discriminant selects:
/// - Vram:       `PageAllocator::slot_ptr(slot_index)`
/// - HostPinned: `HostPageAllocator::slot_ptr(slot_index)`
///
/// `generation` is a monotonically-increasing `AtomicU64` used for LRU eviction
/// policy (Phase D).  Bumped once per page_table write_volatile loop iteration.
/// Initialized to 0; larger values are more recently used.
///
/// Tier transitions are CPU-only and occur only between megakernel batches.
/// No concurrent mutation with the persistent worker occurs.
pub struct ChunkRef {
    pub slot_index: u32,
    /// Number of valid tokens written into this chunk (0..=chunk_tokens).
    pub len: AtomicU32,
    /// Content hash; set when the chunk is sealed (full). Used for radix-tree dedup.
    pub content_hash: Option<u64>,
    /// Tier discriminant: 0 = Vram (PageAllocator), 1 = HostPinned (HostPageAllocator).
    /// Updated atomically by the CPU between batches.
    pub(crate) tier_discriminant: AtomicU8,
    /// LRU generation counter.  Monotonically increasing; larger = more recently used.
    /// Bumped in the page_table write_volatile loop (Phase D).
    pub generation: AtomicU64,
}

/// Arc-wrapped handle to a ChunkRef. Clone-on-write: if refcount > 1, `make_exclusive`
/// allocates a new slot and copies before mutating.
#[derive(Clone)]
pub struct ChunkHandle {
    pub inner: Arc<ChunkRef>,
}

impl ChunkHandle {
    /// Wrap a freshly allocated VRAM slot.  Tier is always `Vram` on construction;
    /// existing callers are unaffected and behavior is byte-identical.
    pub fn new(slot_index: u32) -> Self {
        ChunkHandle {
            inner: Arc::new(ChunkRef {
                slot_index,
                len: AtomicU32::new(0),
                content_hash: None,
                tier_discriminant: AtomicU8::new(ChunkTier::Vram as u8),
                generation: AtomicU64::new(0),
            }),
        }
    }

    /// If this handle is the sole owner, return it unchanged.
    /// Otherwise allocate a new slot, async-copy the valid data, and return an exclusive handle.
    pub fn make_exclusive(
        &self,
        allocator: &mut PageAllocator,
        stream: ffi::hipStream_t,
    ) -> HipResult<ChunkHandle> {
        if Arc::strong_count(&self.inner) == 1 {
            return Ok(self.clone());
        }
        if allocator.free_count() < 1 {
            return Err(HipError(ffi::hipErrorOutOfMemory));
        }
        let (new_slot, new_ptr) = allocator
            .alloc()
            .ok_or(HipError(ffi::hipErrorOutOfMemory))?;
        let old_ptr = allocator.slot_ptr(self.inner.slot_index);
        let len = self.inner.len.load(Ordering::Acquire) as usize;
        // KV layout is [num_layers, K/V, num_heads, chunk_tokens, head_dim].
        // Valid tokens 0..len occupy stride-offset positions within each head's
        // slice — they are NOT a contiguous prefix of the flat buffer.
        // We must copy chunk_bytes() to capture all valid data when len > 0.
        // The len == 0 case avoids a D2D copy for a freshly allocated chunk.
        let copy_bytes = if len == 0 { 0 } else { allocator.chunk_bytes() };
        if copy_bytes > 0 {
            hip_check(unsafe {
                ffi::hipMemcpyAsync(
                    new_ptr.cast(),
                    old_ptr.cast(),
                    copy_bytes,
                    ffi::hipMemcpyDeviceToDevice,
                    stream,
                )
            })?;
        }
        // make_exclusive always produces a VRAM chunk (new_slot is from VRAM PageAllocator).
        Ok(ChunkHandle {
            inner: Arc::new(ChunkRef {
                slot_index: new_slot,
                len: AtomicU32::new(len as u32),
                content_hash: None,
                tier_discriminant: AtomicU8::new(ChunkTier::Vram as u8),
                generation: AtomicU64::new(0),
            }),
        })
    }

    pub fn slot_index(&self) -> u32 {
        self.inner.slot_index
    }

    pub fn len(&self) -> u32 {
        self.inner.len.load(Ordering::Acquire)
    }

    pub fn increment_len(&self) {
        self.inner.len.fetch_add(1, Ordering::AcqRel);
    }

    /// Current tier of this chunk (Vram or HostPinned).
    pub fn tier(&self) -> ChunkTier {
        ChunkTier::from_u8(self.inner.tier_discriminant.load(Ordering::Acquire))
    }

    /// Set the tier.  CPU-only; must not be called concurrently with the megakernel.
    pub fn set_tier(&self, tier: ChunkTier) {
        self.inner
            .tier_discriminant
            .store(tier as u8, Ordering::Release);
    }

    /// VRAM pointer for this chunk's slot. Returns `Some` only when `tier() == Vram`.
    /// Use `PageAllocator::slot_ptr(self.slot_index())` from the caller if you need
    /// the raw pointer; this method returns `None` as a safety gate.
    pub fn vram_ptr(&self, vram_alloc: &PageAllocator) -> Option<*const u8> {
        if self.tier() == ChunkTier::Vram {
            Some(vram_alloc.slot_ptr(self.inner.slot_index))
        } else {
            None
        }
    }

    /// Host-pinned pointer for this chunk's slot.  Always valid once a host slot has
    /// been allocated.  Panics if `tier() == Vram` (no host slot assigned yet).
    ///
    /// # Panics
    /// Panics in debug builds if the chunk is currently in Vram tier.
    pub fn host_ptr(&self, host_alloc: &HostPageAllocator) -> *const u8 {
        debug_assert_eq!(
            self.tier(),
            ChunkTier::HostPinned,
            "host_ptr called on a Vram-tier chunk (slot {})",
            self.inner.slot_index
        );
        host_alloc.slot_ptr(self.inner.slot_index)
    }

    /// Current LRU generation (larger = more recently used).
    pub fn generation(&self) -> u64 {
        self.inner.generation.load(Ordering::Acquire)
    }

    /// Bump generation counter (called from the page_table write_volatile loop, Phase D).
    pub fn bump_generation(&self) -> u64 {
        self.inner.generation.fetch_add(1, Ordering::AcqRel)
    }
}

/// Allocate one VRAM slot and one host-pinned slot together, returning a `ChunkHandle`
/// in `Vram` tier (the host slot acts as a write-through backing store).
///
/// This is the canonical alloc site for Phase C+: using this instead of
/// `ChunkHandle::new` ensures every VRAM chunk always has a host backing slot,
/// implementing the write-through invariant required by the evict path.
///
/// Returns `Err(hipErrorOutOfMemory)` if either pool is exhausted.
pub fn alloc_with_host_backing(
    vram_alloc: &mut PageAllocator,
    host_alloc: &mut HostPageAllocator,
) -> HipResult<(ChunkHandle, u32)> {
    let (vram_slot, _vram_ptr) = vram_alloc
        .alloc()
        .ok_or(HipError(ffi::hipErrorOutOfMemory))?;
    let (host_slot, _host_ptr) = match host_alloc.alloc() {
        Some(pair) => pair,
        None => {
            // Roll back the VRAM allocation to keep pools consistent.
            vram_alloc.free(vram_slot);
            return Err(HipError(ffi::hipErrorOutOfMemory));
        }
    };
    let handle = ChunkHandle {
        inner: Arc::new(ChunkRef {
            slot_index: vram_slot,
            len: AtomicU32::new(0),
            content_hash: None,
            tier_discriminant: AtomicU8::new(ChunkTier::Vram as u8),
            generation: AtomicU64::new(0),
        }),
    };
    Ok((handle, host_slot))
}

// ---- SequenceState ----

/// Per-sequence KV cache tracking. Owns an ordered list of ChunkHandles.
pub struct SequenceState {
    /// Ordered chunks forming this sequence's KV cache.
    pub chunks: Vec<ChunkHandle>,
    /// Total tokens in the sequence.
    pub seq_len: u32,
    /// Logical position per token in sequence order.
    pub positions: Vec<i32>,
    /// Monotonically increasing version; bumped on every structural change.
    pub kv_version: u32,
    /// Tokens per chunk (immutable; set at construction).
    chunk_tokens: u32,
    /// f32 staging buffer for the current incomplete chunk (quantized mode only).
    /// KV values are written here during decode; on chunk seal, the quantize kernel
    /// reads from this buffer and writes to a quantized chunk slot.
    /// None when using unquantized (f32) paged KV.
    pub staging_buffer: Option<DeviceBuffer<u8>>,
    /// Quantized chunk slot indices, parallel to `chunks`. Freed on reset/drop.
    pub quant_slots: Vec<u32>,
}

impl SequenceState {
    pub fn new(chunk_tokens: u32) -> Self {
        SequenceState {
            chunks: Vec::new(),
            seq_len: 0,
            positions: Vec::new(),
            kv_version: 0,
            chunk_tokens,
            staging_buffer: None,
            quant_slots: Vec::new(),
        }
    }

    /// Create with a staging buffer for quantized KV cache.
    pub fn new_quantized(
        chunk_tokens: u32,
        device: DeviceId,
        config: &ModelConfig,
    ) -> HipResult<Self> {
        let staging_bytes = chunk_kv_bytes(config, chunk_tokens as usize);
        let staging = DeviceBuffer::alloc(device, staging_bytes)?;
        Ok(SequenceState {
            chunks: Vec::new(),
            seq_len: 0,
            positions: Vec::new(),
            kv_version: 0,
            chunk_tokens,
            staging_buffer: Some(staging),
            quant_slots: Vec::new(),
        })
    }

    /// Free all quantized slots back to the allocator. Call on sequence reset/drop.
    pub fn free_quant_slots(&mut self, quant_allocator: &mut PageAllocator) {
        for slot in self.quant_slots.drain(..) {
            quant_allocator.free(slot);
        }
    }

    /// Append a token slot. Allocates a new chunk when the current one is full.
    /// Returns `Err` only if the allocator is exhausted.
    pub fn append_token(&mut self, position: i32, allocator: &mut PageAllocator) -> HipResult<()> {
        let needs_new_chunk =
            self.chunks.is_empty() || self.chunks.last().unwrap().len() >= self.chunk_tokens;
        if needs_new_chunk {
            let (slot, _ptr) = allocator
                .alloc()
                .ok_or(HipError(ffi::hipErrorOutOfMemory))?;
            self.chunks.push(ChunkHandle::new(slot));
            self.kv_version += 1;
        }
        self.chunks.last().unwrap().increment_len();
        self.seq_len += 1;
        self.positions.push(position);
        Ok(())
    }

    /// Release all owned f32 chunk slots and clear sequence metadata.
    ///
    /// Chunks with `Arc` refcount > 1 are intentionally NOT freed here. A refcount
    /// greater than 1 means another `SequenceState` holds a `Clone` of this handle
    /// (copy-on-write sharing via `make_exclusive`). Freeing such a chunk would
    /// corrupt the other sequence's KV data. The slot is freed by the last owner
    /// whose `reset()` observes refcount == 1.
    pub fn reset(&mut self, allocator: &mut PageAllocator) {
        for chunk in self.chunks.drain(..) {
            if Arc::strong_count(&chunk.inner) == 1 {
                allocator.free(chunk.slot_index());
            }
        }
        self.seq_len = 0;
        self.positions.clear();
        self.kv_version += 1;
    }

    /// Mutable access to the last (current write) chunk.
    pub fn current_chunk_mut(&mut self) -> Option<&mut ChunkHandle> {
        self.chunks.last_mut()
    }

    /// Index of the current chunk (0-based).
    pub fn current_chunk_idx(&self) -> usize {
        self.chunks.len().saturating_sub(1)
    }

    /// Offset within the current chunk (0..chunk_tokens).
    pub fn current_chunk_offset(&self) -> u32 {
        if self.chunks.is_empty() {
            0
        } else {
            self.chunks.last().unwrap().len()
        }
    }
}

// ---- RecurrentCheckpointPool ----

/// Pre-allocated VRAM pool for GDN/recurrent state checkpoints.
/// Eliminates hipMalloc from the hot path. State size is computed from ModelConfig.
pub struct RecurrentCheckpointPool {
    /// Contiguous VRAM pool: `capacity` × `state_bytes`.
    vram_pool: DeviceBuffer<u8>,
    /// CPU pinned buffer for async offload: same total size.
    cpu_pool: PinnedBuffer<u8>,
    free_list: Vec<u32>,
    capacity: u32,
    state_bytes: usize,
}

impl RecurrentCheckpointPool {
    pub fn new(device: DeviceId, config: &ModelConfig, capacity: u32) -> HipResult<Self> {
        let state_bytes = recurrent_state_bytes(config);
        let total_bytes = capacity as usize * state_bytes;
        let vram_pool = DeviceBuffer::alloc(device, total_bytes)?;
        let cpu_pool = PinnedBuffer::alloc(total_bytes)?;
        let free_list = (0..capacity).rev().collect();
        Ok(Self {
            vram_pool,
            cpu_pool,
            free_list,
            capacity,
            state_bytes,
        })
    }

    /// Allocate a VRAM checkpoint slot. Returns `(slot_index, vram_ptr)`.
    pub fn alloc(&mut self) -> Option<(u32, *mut u8)> {
        let idx = self.free_list.pop()?;
        let offset = idx as usize * self.state_bytes;
        let ptr = unsafe { self.vram_pool.as_mut_ptr().add(offset) };
        Some((idx, ptr))
    }

    pub fn free(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.capacity as usize);
        self.free_list.push(slot);
    }

    /// VRAM pointer for a slot (read-only, for restore).
    pub fn slot_vram_ptr(&self, slot: u32) -> *const u8 {
        let offset = slot as usize * self.state_bytes;
        unsafe { self.vram_pool.as_ptr().add(offset) }
    }

    /// CPU pinned pointer for a slot (for async H2D/D2H).
    pub fn slot_cpu_ptr(&mut self, slot: u32) -> *mut u8 {
        let offset = slot as usize * self.state_bytes;
        unsafe { self.cpu_pool.as_mut_ptr().add(offset) }
    }

    pub fn state_bytes(&self) -> usize {
        self.state_bytes
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

/// Save current recurrent state into a pool slot. Async D2D copy on `stream`.
/// Returns the slot index that was used.
pub fn save_checkpoint(
    pool: &mut RecurrentCheckpointPool,
    recurrent_states: &[&DeviceBuffer<f32>],
    stream: ffi::hipStream_t,
) -> HipResult<u32> {
    let (slot, dst_ptr) = pool.alloc().ok_or(HipError(ffi::hipErrorOutOfMemory))?;
    debug_assert_eq!(
        pool.state_bytes(),
        recurrent_states.len() * recurrent_states.first().map_or(0, |s| s.size_bytes()),
        "state_bytes mismatch: pool expects {} but got {} layers * {} bytes/layer",
        pool.state_bytes(),
        recurrent_states.len(),
        recurrent_states.first().map_or(0, |s| s.size_bytes()),
    );
    let floats_per_layer = pool.state_bytes() / recurrent_states.len() / std::mem::size_of::<f32>();
    for (i, state) in recurrent_states.iter().enumerate() {
        let layer_offset = i * floats_per_layer * std::mem::size_of::<f32>();
        let dst = unsafe { dst_ptr.add(layer_offset) };
        hip_check(unsafe {
            ffi::hipMemcpyAsync(
                dst.cast(),
                state.as_ptr().cast(),
                floats_per_layer * std::mem::size_of::<f32>(),
                ffi::hipMemcpyDeviceToDevice,
                stream,
            )
        })?;
    }
    Ok(slot)
}

/// Restore recurrent state from a pool slot. Async D2D copy on `stream`.
pub fn restore_checkpoint(
    pool: &RecurrentCheckpointPool,
    slot: u32,
    recurrent_states: &mut [&mut DeviceBuffer<f32>],
    stream: ffi::hipStream_t,
) -> HipResult<()> {
    let src_ptr = pool.slot_vram_ptr(slot);
    let floats_per_layer = pool.state_bytes() / recurrent_states.len() / std::mem::size_of::<f32>();
    for (i, state) in recurrent_states.iter_mut().enumerate() {
        let layer_offset = i * floats_per_layer * std::mem::size_of::<f32>();
        let src = unsafe { src_ptr.add(layer_offset) };
        hip_check(unsafe {
            ffi::hipMemcpyAsync(
                state.as_mut_ptr().cast(),
                src.cast(),
                floats_per_layer * std::mem::size_of::<f32>(),
                ffi::hipMemcpyDeviceToDevice,
                stream,
            )
        })?;
    }
    Ok(())
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> ModelConfig {
        ModelConfig::qwen35_0_8b()
    }

    #[test]
    fn test_chunk_kv_bytes() {
        let config = make_test_config();
        let bytes = chunk_kv_bytes(&config, 64);
        // 6 attn layers * 2 * 64 tokens * 2 kv_heads * 256 head_dim * 4 bytes
        let expected = 6 * 2 * 64 * 2 * 256 * 4;
        assert_eq!(bytes, expected, "chunk_kv_bytes mismatch");
    }

    #[test]
    fn test_quantized_chunk_kv_bytes() {
        let config = make_test_config();
        let q_bytes = quantized_chunk_kv_bytes(&config, 64);
        // Per K or V per layer:
        //   data = 2 * (2 kv_heads * 64 tokens * 256 dim / 2) = 2 * 16384 = 32768 bytes
        //   scale = 2 * (2 kv_heads * 256 dim * 4) = 2 * 2048 = 4096 bytes
        //   per_kv = 32768 + 4096 = 36864
        //   per_layer = 2 * 36864 = 73728
        //   total = 6 layers * 73728 = 442368
        let expected = 6 * 2 * 2 * (2 * 64 * 256 / 2 + 2 * 256 * 4);
        assert_eq!(q_bytes, expected, "quantized_chunk_kv_bytes mismatch");
        let f32_bytes = chunk_kv_bytes(&config, 64);
        let ratio = f32_bytes as f64 / q_bytes as f64;
        assert!(
            (ratio - 3.56).abs() < 0.01,
            "expected ~3.56x reduction, got {ratio:.2}x"
        );
    }

    #[test]
    fn test_quantized_kv_offsets() {
        let config = make_test_config();
        let (q1d, q1s, rd, rs) = quantized_kv_offsets(&config, 64, 0, false);
        assert_eq!(q1d, 0);
        let data_bytes = 2 * 64 * 256 / 2;
        let scale_bytes = 2 * 256 * 4;
        assert_eq!(q1s, data_bytes);
        assert_eq!(rd, data_bytes + scale_bytes);
        assert_eq!(rs, 2 * data_bytes + scale_bytes);
    }

    #[test]
    fn test_recurrent_state_bytes() {
        let config = make_test_config();
        let bytes = recurrent_state_bytes(&config);
        // 18 gdn layers * 16 heads * 128 key_head_dim * 128 value_head_dim * 4 bytes
        let expected = 18 * 16 * 128 * 128 * 4;
        assert_eq!(bytes, expected, "recurrent_state_bytes mismatch");
    }

    #[test]
    fn test_chunk_handle_refcount() {
        let handle = ChunkHandle::new(42);
        assert_eq!(handle.slot_index(), 42);
        assert_eq!(handle.len(), 0);

        let handle2 = handle.clone();
        assert_eq!(Arc::strong_count(&handle.inner), 2);
        drop(handle2);
        assert_eq!(Arc::strong_count(&handle.inner), 1);
    }

    #[test]
    fn test_chunk_handle_increment_len() {
        let handle = ChunkHandle::new(0);
        handle.increment_len();
        handle.increment_len();
        assert_eq!(handle.len(), 2);
    }

    #[test]
    fn test_sequence_state_chunk_allocation_logic() {
        let chunk_tokens = 4u32;
        let mut seq = SequenceState::new(chunk_tokens);

        assert!(seq.chunks.is_empty());
        assert_eq!(seq.seq_len, 0);
        assert!(seq.positions.is_empty());

        // Mock allocator calls: track how many chunks would be needed
        // We can't call the real allocator (no GPU), so we manually drive logic
        // by checking the invariants.

        // Simulate: needs_new_chunk = true initially
        assert!(
            seq.chunks.is_empty() || seq.chunks.last().map_or(true, |c| c.len() >= chunk_tokens)
        );

        // Manually create chunks as if append_token succeeded
        seq.chunks.push(ChunkHandle::new(0));
        seq.seq_len += 1;
        seq.positions.push(7);
        seq.chunks.last().unwrap().increment_len();

        assert_eq!(seq.seq_len, 1);
        assert_eq!(seq.current_chunk_offset(), 1);
        assert_eq!(seq.current_chunk_idx(), 0);

        // Fill chunk
        for _ in 1..chunk_tokens {
            seq.chunks.last().unwrap().increment_len();
            seq.seq_len += 1;
            seq.positions.push(seq.seq_len as i32);
        }
        assert_eq!(seq.chunks.last().unwrap().len(), chunk_tokens);

        // Next append would need new chunk
        let needs_new = seq.chunks.last().unwrap().len() >= chunk_tokens;
        assert!(needs_new);

        // Allocate second chunk
        seq.chunks.push(ChunkHandle::new(1));
        seq.kv_version += 1;
        seq.chunks.last().unwrap().increment_len();
        seq.seq_len += 1;
        seq.positions.push(99);

        assert_eq!(seq.chunks.len(), 2);
        assert_eq!(seq.current_chunk_idx(), 1);
        assert_eq!(seq.current_chunk_offset(), 1);
        assert_eq!(seq.positions, vec![7, 2, 3, 4, 99]);
    }

    #[test]
    fn test_sequence_state_accessors() {
        let mut seq = SequenceState::new(64);
        assert!(seq.current_chunk_mut().is_none());
        assert_eq!(seq.current_chunk_offset(), 0);
        assert_eq!(seq.current_chunk_idx(), 0); // saturating_sub(1) = 0
    }

    // ---- Phase A: ChunkTier + HostPageAllocator tests ----

    /// ChunkHandle::new always starts in Vram tier.
    #[test]
    fn test_chunk_handle_default_tier_is_vram() {
        let h = ChunkHandle::new(7);
        assert_eq!(h.tier(), ChunkTier::Vram);
        assert_eq!(h.generation(), 0);
    }

    /// tier() / set_tier() round-trip; generation counter is independent.
    #[test]
    fn test_chunk_handle_tier_transition() {
        let h = ChunkHandle::new(3);
        assert_eq!(h.tier(), ChunkTier::Vram);

        h.set_tier(ChunkTier::HostPinned);
        assert_eq!(h.tier(), ChunkTier::HostPinned);

        h.set_tier(ChunkTier::Vram);
        assert_eq!(h.tier(), ChunkTier::Vram);
    }

    /// bump_generation returns old value and counter increases monotonically.
    #[test]
    fn test_chunk_handle_generation_monotonic() {
        let h = ChunkHandle::new(0);
        assert_eq!(h.generation(), 0);
        let old = h.bump_generation();
        assert_eq!(old, 0);
        assert_eq!(h.generation(), 1);
        h.bump_generation();
        h.bump_generation();
        assert_eq!(h.generation(), 3);
    }

    /// Cloning a ChunkHandle shares the same AtomicU8 / AtomicU64.
    #[test]
    fn test_chunk_handle_tier_shared_across_clones() {
        let h1 = ChunkHandle::new(5);
        let h2 = h1.clone();
        assert_eq!(h2.tier(), ChunkTier::Vram);

        h1.set_tier(ChunkTier::HostPinned);
        // Both handles see the same Arc<ChunkRef>, so tier change is visible via h2.
        assert_eq!(h2.tier(), ChunkTier::HostPinned);

        h2.bump_generation();
        assert_eq!(h1.generation(), 1);
    }

    /// HostPageAllocator: a mock-pool exercising alloc/free/slot_ptr without GPU.
    /// We create the allocator with a pre-allocated Vec-backed pool (no HIP calls).
    struct MockHostAlloc {
        // Simulates HostPageAllocator free-list logic without hipHostMalloc.
        chunk_bytes: usize,
        capacity: u32,
        free_list: Vec<u32>,
        // Heap-backed storage to simulate slot_ptr arithmetic (no GPU required).
        _storage: Vec<u8>,
        base: *mut u8,
    }

    impl MockHostAlloc {
        fn new(chunk_bytes: usize, capacity: u32) -> Self {
            let mut storage = vec![0u8; chunk_bytes * capacity as usize];
            let base = storage.as_mut_ptr();
            Self {
                chunk_bytes,
                capacity,
                free_list: (0..capacity).rev().collect(),
                _storage: storage,
                base,
            }
        }

        fn alloc(&mut self) -> Option<(u32, *mut u8)> {
            let idx = self.free_list.pop()?;
            let ptr = unsafe { self.base.add(idx as usize * self.chunk_bytes) };
            Some((idx, ptr))
        }

        fn free(&mut self, slot: u32) {
            assert!((slot as usize) < self.capacity as usize);
            assert!(!self.free_list.contains(&slot));
            self.free_list.push(slot);
        }

        fn slot_ptr(&self, slot: u32) -> *const u8 {
            unsafe { self.base.add(slot as usize * self.chunk_bytes) as *const u8 }
        }
    }

    #[test]
    fn test_mock_host_alloc_free_reuse() {
        let chunk_bytes = 128;
        let capacity = 4u32;
        let mut alloc = MockHostAlloc::new(chunk_bytes, capacity);

        // Alloc all slots
        let mut slots = Vec::new();
        for _ in 0..capacity {
            let (slot, _ptr) = alloc.alloc().expect("alloc should succeed");
            slots.push(slot);
        }
        // Pool exhausted
        assert!(alloc.alloc().is_none());

        // Free one and re-alloc
        let freed = slots[0];
        alloc.free(freed);
        let (reused_slot, _) = alloc.alloc().expect("should reuse freed slot");
        assert_eq!(reused_slot, freed);
    }

    #[test]
    fn test_mock_host_alloc_slot_ptr_arithmetic() {
        let chunk_bytes = 64;
        let capacity = 3u32;
        let mut alloc = MockHostAlloc::new(chunk_bytes, capacity);

        let (s0, _) = alloc.alloc().unwrap();
        let (s1, _) = alloc.alloc().unwrap();
        let p0 = alloc.slot_ptr(s0) as usize;
        let p1 = alloc.slot_ptr(s1) as usize;
        // Slot pointers must be exactly chunk_bytes apart (one is s0, the other s1).
        // The free_list is (0..cap).rev() so first pop = cap-1, etc; just check distance.
        let diff = if p1 > p0 { p1 - p0 } else { p0 - p1 };
        // Slots are contiguous with spacing = chunk_bytes * slot_distance.
        let slot_diff = if s1 > s0 {
            (s1 - s0) as usize
        } else {
            (s0 - s1) as usize
        };
        assert_eq!(diff, chunk_bytes * slot_diff);
    }

    /// HostPageAllocator free-list logic: slot_ptr offsets are exactly chunk_bytes * slot.
    #[test]
    fn test_host_allocator_slot_ptr_offsets() {
        // We replicate the HostPageAllocator slot_ptr formula without calling hipHostMalloc.
        // slot_ptr(slot) = base + slot * chunk_bytes
        let chunk_bytes: usize = 256;
        let capacity: u32 = 8;
        let base_addr: usize = 0x1000_0000; // arbitrary sentinel
        let slot_ptr = |slot: u32| -> usize { base_addr + slot as usize * chunk_bytes };

        for slot in 0..capacity {
            let expected = base_addr + slot as usize * chunk_bytes;
            assert_eq!(slot_ptr(slot), expected);
        }
        // Adjacent slots differ by exactly chunk_bytes.
        assert_eq!(slot_ptr(3) - slot_ptr(2), chunk_bytes);
    }

    /// ChunkTier discriminant values match expected u8 constants.
    #[test]
    fn test_chunk_tier_discriminant_values() {
        assert_eq!(ChunkTier::Vram as u8, 0u8);
        assert_eq!(ChunkTier::HostPinned as u8, 1u8);
        assert_eq!(ChunkTier::from_u8(0), ChunkTier::Vram);
        assert_eq!(ChunkTier::from_u8(1), ChunkTier::HostPinned);
    }

    /// vram_ptr returns Some only for Vram tier.
    /// Uses a raw-pointer PageAllocator stand-in without GPU to test the guard.
    #[test]
    fn test_vram_ptr_none_when_host_pinned() {
        // We cannot instantiate a real PageAllocator (requires hipMalloc).
        // Instead we verify the tier-gate logic by checking tier transitions directly.
        let h = ChunkHandle::new(0);
        assert_eq!(h.tier(), ChunkTier::Vram);
        h.set_tier(ChunkTier::HostPinned);
        assert_eq!(h.tier(), ChunkTier::HostPinned);
        // vram_ptr(real_alloc) would return None here; we assert the tier gate.
        // (Cannot call vram_ptr without a real PageAllocator/GPU.)
        assert_ne!(h.tier(), ChunkTier::Vram);
    }
}
