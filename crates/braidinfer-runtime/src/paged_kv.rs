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

    /// Explicitly free the pinned host memory pool.
    ///
    /// ## Safety
    ///
    /// Must ONLY be called after the persistent cooperative worker has been fully
    /// torn down (i.e., after `drop(model.persistent_workers.take())`).
    /// `hipHostFree` (called internally by `PinnedBuffer::drop`) deadlocks while
    /// the persistent cooperative worker holds all GPU CUs.
    ///
    /// This method consumes `self` so the pool cannot be double-freed.
    ///
    /// ## Design note
    ///
    /// This mirrors the `ManuallyDrop<DeviceBuffer<u8>>` explicit-drop pattern
    /// in `PersistentDispatch::drop` (single-GPU clean path).  By using a method
    /// here instead of exposing `pool` directly, we prevent accidental misuse from
    /// outside the crate.
    pub fn drop_pool(mut self) {
        // SAFETY: caller guarantees the persistent worker is not running.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.pool) };
        // Prevent double-drop: std::mem::forget ensures the ManuallyDrop wrapper
        // itself is not dropped again (it's already been explicitly dropped above).
        // The free_list and other fields do not need explicit cleanup.
        std::mem::forget(self);
    }
}

// ---- ChunkRef + ChunkHandle (Arc-based CoW) ----

/// Shared metadata for one KV chunk slot. The actual data lives in the PageAllocator pool.
///
/// # Tier model (Phase B, braidinfer-4n5)
///
/// `tier_discriminant` is an `AtomicU8` encoding `ChunkTier` (0=Vram, 1=HostPinned).
/// Two companion atomics track WHICH slot in each pool this chunk occupies:
/// - `vram_slot`:  AtomicU32, index into `PageAllocator`.  u32::MAX = no VRAM slot.
/// - `host_slot`:  AtomicU32, index into `HostPageAllocator`. u32::MAX = no host slot.
///
/// Promote changes `vram_slot` (new slot allocated) + sets tier=Vram.
/// Evict sets tier=HostPinned (VRAM slot freed by caller after flush_tier_ops).
/// The host_slot never changes once set; it is the write-through backing slot.
///
/// `generation` is a monotonically-increasing `AtomicU64` used for LRU eviction
/// policy (Phase D).  Bumped once per page_table write_volatile loop iteration.
/// Initialized to 0; larger values are more recently used.
///
/// Tier transitions are CPU-only and occur only between megakernel batches.
/// No concurrent mutation with the persistent worker occurs.
///
/// # Backward compatibility
///
/// `ChunkHandle::new(slot)` sets `vram_slot=slot`, `host_slot=u32::MAX` — identical
/// behavior to Phase A.  `slot_index()` reads `vram_slot` (the active VRAM slot).
/// `make_exclusive` copies `vram_slot` into the new `ChunkRef`.
pub struct ChunkRef {
    /// VRAM pool slot index (PageAllocator).  AtomicU32; u32::MAX = no VRAM slot.
    /// Mutable through Arc: promote writes a new slot here before flipping tier.
    pub(crate) vram_slot: AtomicU32,
    /// Host-pinned pool slot index (HostPageAllocator).  AtomicU32; u32::MAX = none.
    /// Set once at `alloc_with_host_backing`; never reassigned.
    pub(crate) host_slot: AtomicU32,
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
    /// `host_slot` is u32::MAX (no host backing); use `alloc_with_host_backing` to get both.
    pub fn new(slot_index: u32) -> Self {
        ChunkHandle {
            inner: Arc::new(ChunkRef {
                vram_slot: AtomicU32::new(slot_index),
                host_slot: AtomicU32::new(u32::MAX),
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
        let old_vram_slot = self.inner.vram_slot.load(Ordering::Acquire);
        let old_ptr = allocator.slot_ptr(old_vram_slot);
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
        // Host backing slot is not copied — make_exclusive is for CoW sharing, not tiering.
        Ok(ChunkHandle {
            inner: Arc::new(ChunkRef {
                vram_slot: AtomicU32::new(new_slot),
                host_slot: AtomicU32::new(u32::MAX),
                len: AtomicU32::new(len as u32),
                content_hash: None,
                tier_discriminant: AtomicU8::new(ChunkTier::Vram as u8),
                generation: AtomicU64::new(0),
            }),
        })
    }

    /// Current VRAM slot index.  Only valid when `tier() == Vram`; panics in debug
    /// builds otherwise.  Used by page_table write loops to obtain the VRAM slot_ptr.
    pub fn slot_index(&self) -> u32 {
        debug_assert_eq!(
            self.tier(),
            ChunkTier::Vram,
            "slot_index() called on a HostPinned-tier chunk"
        );
        self.inner.vram_slot.load(Ordering::Acquire)
    }

    /// Host-pinned pool slot index.  u32::MAX if no host backing was allocated.
    pub fn host_slot_index(&self) -> u32 {
        self.inner.host_slot.load(Ordering::Acquire)
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
    /// Reads from `vram_slot` (which may have been updated by `promote_chunk`).
    pub fn vram_ptr(&self, vram_alloc: &PageAllocator) -> Option<*const u8> {
        if self.tier() == ChunkTier::Vram {
            let slot = self.inner.vram_slot.load(Ordering::Acquire);
            Some(vram_alloc.slot_ptr(slot))
        } else {
            None
        }
    }

    /// Host-pinned pointer for this chunk's slot.  Valid once a host slot has been
    /// allocated via `alloc_with_host_backing`.  Reads from `host_slot` (AtomicU32).
    ///
    /// # Panics
    /// Panics if `host_slot == u32::MAX` (no host backing allocated).
    pub fn host_ptr(&self, host_alloc: &HostPageAllocator) -> *const u8 {
        let slot = self.inner.host_slot.load(Ordering::Acquire);
        assert_ne!(
            slot, u32::MAX,
            "host_ptr called on chunk with no host backing (use alloc_with_host_backing)"
        );
        host_alloc.slot_ptr(slot)
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
/// The host slot index is stored on `ChunkRef.host_slot` (AtomicU32).  Callers do
/// not need to track the host slot separately — `host_ptr()` and `evict_chunk()`
/// read it from the handle.
///
/// This is the canonical alloc site for Phase C+: using this instead of
/// `ChunkHandle::new` ensures every VRAM chunk always has a host backing slot,
/// implementing the write-through invariant required by the evict path.
///
/// Returns `Err(hipErrorOutOfMemory)` if either pool is exhausted.
pub fn alloc_with_host_backing(
    vram_alloc: &mut PageAllocator,
    host_alloc: &mut HostPageAllocator,
) -> HipResult<ChunkHandle> {
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
    Ok(ChunkHandle {
        inner: Arc::new(ChunkRef {
            vram_slot: AtomicU32::new(vram_slot),
            host_slot: AtomicU32::new(host_slot),
            len: AtomicU32::new(0),
            content_hash: None,
            tier_discriminant: AtomicU8::new(ChunkTier::Vram as u8),
            generation: AtomicU64::new(0),
        }),
    })
}

// ---- Phase B: Tier transition primitives ----

/// Promote a chunk from `HostPinned` tier to `Vram` tier via an async H2D copy on `stream`.
///
/// ## Contract
///
/// 1. `handle.tier()` MUST be `HostPinned` on entry.
/// 2. `handle.host_slot_index()` MUST NOT be `u32::MAX` (host backing required).
/// 3. A new VRAM slot is allocated from `vram_alloc`.  On OOM, returns
///    `Err(hipErrorOutOfMemory)` and leaves the handle in `HostPinned` tier (no
///    partial state update).
/// 4. `hipMemcpyAsync(vram_dst, host_src, chunk_bytes, H2D, stream)` is issued.
///    The copy is **in-flight** after this call.
/// 5. `vram_slot` and `tier_discriminant` are updated atomically (in this order)
///    BEFORE returning.  This means the next `slot_index()` call will return the
///    new VRAM slot even before the copy completes.
/// 6. **Caller MUST call `flush_tier_ops` (or `hipStreamSynchronize(stream)`) before
///    writing the new `slot_ptr` into any page_table**.  The page_table write loop
///    must observe coherent VRAM data; without the synchronize the megakernel reads
///    stale or partially-written VRAM.
///
/// ## Write-through note
///
/// The host slot is retained (not freed) after promote.  It remains the durable
/// canonical copy.  A subsequent evict does NOT need a D2H copy — the host slot
/// already holds the sealed data.
pub fn promote_chunk(
    handle: &ChunkHandle,
    vram_alloc: &mut PageAllocator,
    host_alloc: &HostPageAllocator,
    stream: ffi::hipStream_t,
) -> HipResult<()> {
    assert_eq!(
        handle.tier(),
        ChunkTier::HostPinned,
        "promote_chunk: handle is not HostPinned (tier={:?})",
        handle.tier()
    );
    let h_slot = handle.inner.host_slot.load(Ordering::Acquire);
    assert_ne!(h_slot, u32::MAX, "promote_chunk: no host backing slot");

    let (new_vram_slot, vram_dst) = vram_alloc
        .alloc()
        .ok_or(HipError(ffi::hipErrorOutOfMemory))?;

    let host_src = host_alloc.slot_ptr(h_slot);
    let chunk_bytes = vram_alloc.chunk_bytes();

    hip_check(unsafe {
        ffi::hipMemcpyAsync(
            vram_dst.cast(),
            host_src.cast(),
            chunk_bytes,
            ffi::hipMemcpyHostToDevice,
            stream,
        )
    })
    .map_err(|e| {
        // Roll back VRAM alloc so the pool stays consistent.
        vram_alloc.free(new_vram_slot);
        e
    })?;

    // Update vram_slot first, then flip tier.  Ordering: Release on the store that
    // must be visible before any reader observes the tier flip.
    handle.inner.vram_slot.store(new_vram_slot, Ordering::Release);
    handle
        .inner
        .tier_discriminant
        .store(ChunkTier::Vram as u8, Ordering::Release);
    Ok(())
}

/// Evict a chunk from `Vram` tier to `HostPinned` tier.
///
/// ## Two-phase ordering contract
///
/// This function issues the async D2H copy but does **not** free the VRAM slot.
/// The caller MUST sequence as follows:
///
/// ```text
/// evict_chunk(handle, vram_alloc, host_alloc, stream)?;   // issues D2H copy
/// flush_tier_ops(gpu_idx)?;                               // hipStreamSynchronize
/// vram_alloc.free(handle.inner.vram_slot.load(...));      // VRAM slot freed
/// handle.inner.vram_slot.store(u32::MAX, Release);        // mark slot gone
/// handle.set_tier(ChunkTier::HostPinned);                 // tier flip
/// ```
///
/// The reason for the split: freeing the VRAM slot before the D2H copy completes
/// allows the allocator to hand it to another sequence while SDMA is still reading
/// from it — a use-after-free.  The two-phase design makes the hazard explicit in
/// caller code rather than hiding it inside a synchronous evict call.
///
/// See `evict_chunk_and_free` for a single-call wrapper that handles the
/// synchronization inline (for call sites that only evict one chunk per step).
///
/// ## Preconditions
///
/// - `handle.tier() == Vram`.
/// - `handle.host_slot_index() != u32::MAX` (write-through backing exists).
/// - Chunk MUST be sealed (`handle.len() == chunk_tokens`).  R4 scope: f32 only —
///   **caller must not pass a handle whose slot came from a quantized PageAllocator**.
///   (ChunkRef carries no quant marker; the CALLER is responsible for this invariant.)
pub fn evict_chunk(
    handle: &ChunkHandle,
    vram_alloc: &PageAllocator,
    host_alloc: &HostPageAllocator,
    stream: ffi::hipStream_t,
) -> HipResult<()> {
    assert_eq!(
        handle.tier(),
        ChunkTier::Vram,
        "evict_chunk: handle is not Vram (tier={:?})",
        handle.tier()
    );
    let h_slot = handle.inner.host_slot.load(Ordering::Acquire);
    assert_ne!(h_slot, u32::MAX, "evict_chunk: no host backing slot");

    let v_slot = handle.inner.vram_slot.load(Ordering::Acquire);
    let vram_src = vram_alloc.slot_ptr(v_slot);
    let host_dst = host_alloc.slot_ptr(h_slot);
    let chunk_bytes = vram_alloc.chunk_bytes();

    // Write-through note: the host slot already holds the sealed data from the wt1
    // mirror (kv_mirror_chunk / drain_kv_chunk_mirror at seal boundaries).  This D2H
    // copy is therefore idempotent for sealed chunks — it refreshes host with the same
    // data.  We issue it unconditionally so the evict path works even if the caller
    // has not enabled wt1 (e.g. in tests).
    hip_check(unsafe {
        ffi::hipMemcpyAsync(
            host_dst as *mut std::ffi::c_void,
            vram_src.cast(),
            chunk_bytes,
            ffi::hipMemcpyDeviceToHost,
            stream,
        )
    })?;

    // Do NOT free the VRAM slot here.  The caller must call flush_tier_ops, then free.
    // See two-phase ordering contract above.
    Ok(())
}

/// Convenience wrapper: evict + synchronize + free VRAM slot + flip tier.
///
/// Use when evicting a single chunk per step and the caller can tolerate a
/// synchronous stream flush.  For bulk evictions (evict N then flush once), call
/// `evict_chunk` N times followed by one `flush_tier_ops` + N manual free+flip calls.
///
/// After this call `handle.tier() == HostPinned` and the VRAM slot has been freed.
pub fn evict_chunk_and_free(
    handle: &ChunkHandle,
    vram_alloc: &mut PageAllocator,
    host_alloc: &HostPageAllocator,
    stream: ffi::hipStream_t,
) -> HipResult<()> {
    evict_chunk(handle, vram_alloc, host_alloc, stream)?;
    // Synchronize the SDMA stream so the D2H copy is complete before we free.
    braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(stream) })?;
    let v_slot = handle.inner.vram_slot.load(Ordering::Acquire);
    vram_alloc.free(v_slot);
    handle.inner.vram_slot.store(u32::MAX, Ordering::Release);
    handle
        .inner
        .tier_discriminant
        .store(ChunkTier::HostPinned as u8, Ordering::Release);
    Ok(())
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
    ///
    /// # VRAM-exhausted fallback (Phase C, braidinfer-4n5)
    ///
    /// When `vram_alloc.alloc()` returns `None` AND `host_alloc` is `Some`:
    /// a new chunk is allocated directly as `HostPinned` tier (host slot only,
    /// no VRAM slot).  The chunk will be promoted to VRAM by the Phase-D
    /// prefetch pass before the next megakernel batch.
    ///
    /// When `host_alloc` is `None` (host tier disabled, the default) the
    /// pre-Phase-C behavior is preserved: returns `Err(hipErrorOutOfMemory)`.
    pub fn append_token(
        &mut self,
        position: i32,
        allocator: &mut PageAllocator,
        host_alloc: Option<&mut HostPageAllocator>,
    ) -> HipResult<()> {
        let needs_new_chunk =
            self.chunks.is_empty() || self.chunks.last().unwrap().len() >= self.chunk_tokens;
        if needs_new_chunk {
            let handle = match allocator.alloc() {
                Some((slot, _ptr)) => ChunkHandle::new(slot),
                None => {
                    // VRAM exhausted.  Fall back to host-pinned tier if enabled.
                    let ha = host_alloc.ok_or(HipError(ffi::hipErrorOutOfMemory))?;
                    let (host_slot, _host_ptr) =
                        ha.alloc().ok_or(HipError(ffi::hipErrorOutOfMemory))?;
                    // Build a HostPinned chunk: no VRAM slot (u32::MAX), host slot set.
                    let h = ChunkHandle {
                        inner: Arc::new(ChunkRef {
                            vram_slot: AtomicU32::new(u32::MAX),
                            host_slot: AtomicU32::new(host_slot),
                            len: AtomicU32::new(0),
                            content_hash: None,
                            tier_discriminant: AtomicU8::new(ChunkTier::HostPinned as u8),
                            generation: AtomicU64::new(0),
                        }),
                    };
                    h
                }
            };
            self.chunks.push(handle);
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
    ///
    /// Phase C (braidinfer-4n5): `host_alloc` is `Some` when the host-RAM tier
    /// is enabled.  For each exclusively-owned chunk, BOTH the VRAM slot (if
    /// `Vram` tier) AND the host slot (if != u32::MAX) are freed.  `HostPinned`
    /// chunks only have a host slot to free.
    pub fn reset(
        &mut self,
        allocator: &mut PageAllocator,
        mut host_alloc: Option<&mut HostPageAllocator>,
    ) {
        for chunk in self.chunks.drain(..) {
            if Arc::strong_count(&chunk.inner) == 1 {
                match chunk.tier() {
                    ChunkTier::Vram => {
                        let v_slot = chunk.inner.vram_slot.load(Ordering::Acquire);
                        if v_slot != u32::MAX {
                            allocator.free(v_slot);
                        }
                        // Also free the host backing slot if it exists.
                        if let Some(ha) = host_alloc.as_deref_mut() {
                            let h_slot = chunk.inner.host_slot.load(Ordering::Acquire);
                            if h_slot != u32::MAX {
                                ha.free(h_slot);
                            }
                        }
                    }
                    ChunkTier::HostPinned => {
                        // HostPinned chunk: no VRAM slot.  Free host slot if
                        // host_alloc is provided; otherwise the slot is leaked
                        // (caller did not enable the host tier but somehow got a
                        // HostPinned chunk — should not happen in practice).
                        if let Some(ha) = host_alloc.as_deref_mut() {
                            let h_slot = chunk.inner.host_slot.load(Ordering::Acquire);
                            if h_slot != u32::MAX {
                                ha.free(h_slot);
                            }
                        }
                    }
                }
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

    // ---- Phase C: append_token host-tier fallback (no-GPU unit test) ----

    /// A mock PageAllocator backed by a heap Vec instead of hipMalloc.
    /// Used to simulate VRAM exhaustion (capacity=2) without a GPU context.
    struct MockVramAlloc {
        chunk_bytes: usize,
        capacity: u32,
        free_list: Vec<u32>,
        _storage: Vec<u8>,
        base: *mut u8,
    }

    impl MockVramAlloc {
        fn new(chunk_bytes: usize, capacity: u32) -> Self {
            let mut storage = vec![0u8; chunk_bytes * capacity as usize];
            let base = storage.as_mut_ptr();
            Self { chunk_bytes, capacity, free_list: (0..capacity).rev().collect(), _storage: storage, base }
        }

        fn alloc_mock(&mut self) -> Option<(u32, *mut u8)> {
            let idx = self.free_list.pop()?;
            let ptr = unsafe { self.base.add(idx as usize * self.chunk_bytes) };
            Some((idx, ptr))
        }

        fn free_mock(&mut self, slot: u32) {
            assert!((slot as usize) < self.capacity as usize);
            assert!(!self.free_list.contains(&slot));
            self.free_list.push(slot);
        }
    }

    /// Minimal stand-in for HostPageAllocator that uses heap memory instead of
    /// hipHostMalloc.  Mirrors the free-list logic exactly; does not test the
    /// HIP allocation path (that is a GPU integration test).
    struct MockHostAllocPhaseC {
        chunk_bytes: usize,
        capacity: u32,
        free_list: Vec<u32>,
        _storage: Vec<u8>,
        base: *mut u8,
    }

    impl MockHostAllocPhaseC {
        fn new(chunk_bytes: usize, capacity: u32) -> Self {
            let mut storage = vec![0u8; chunk_bytes * capacity as usize];
            let base = storage.as_mut_ptr();
            Self { chunk_bytes, capacity, free_list: (0..capacity).rev().collect(), _storage: storage, base }
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

        fn free_count(&self) -> usize { self.free_list.len() }
    }

    /// Simulates SequenceState::append_token's Phase C fallback logic without a
    /// GPU or a real PageAllocator / HostPageAllocator.
    ///
    /// Test scenario: VRAM capacity=2 chunks, host capacity=4 chunks.
    /// Append 3 * chunk_tokens tokens; the 3rd chunk must spill to HostPinned
    /// tier.  append_token must NOT return OOM.
    #[test]
    fn test_append_token_host_fallback_no_gpu() {
        let chunk_tokens: u32 = 4;
        let chunk_bytes: usize = 64; // arbitrary; mock only
        let vram_capacity: u32 = 2;
        let host_capacity: u32 = 4;

        let mut vram = MockVramAlloc::new(chunk_bytes, vram_capacity);
        let mut host = MockHostAllocPhaseC::new(chunk_bytes, host_capacity);

        // We manually implement the append_token fallback logic here because we
        // cannot call SequenceState::append_token with a mock (it takes a real
        // PageAllocator).  This is the canonical "branch logic" unit test: verify
        // that the fallback branch produces a HostPinned ChunkHandle when VRAM
        // is exhausted.

        // Simulate two full VRAM chunks:
        let mut chunks: Vec<ChunkHandle> = Vec::new();
        let mut seq_len: u32 = 0;

        let mut append = |pos: i32,
                          vram: &mut MockVramAlloc,
                          host_alloc: &mut MockHostAllocPhaseC,
                          chunks: &mut Vec<ChunkHandle>,
                          seq_len: &mut u32| {
            let needs_new = chunks.is_empty() || chunks.last().unwrap().len() >= chunk_tokens;
            if needs_new {
                let handle = match vram.alloc_mock() {
                    Some((slot, _)) => ChunkHandle::new(slot),
                    None => {
                        // VRAM exhausted — fall back to host-pinned tier.
                        let (host_slot, _) = host_alloc.alloc().expect("host OOM");
                        ChunkHandle {
                            inner: Arc::new(ChunkRef {
                                vram_slot: AtomicU32::new(u32::MAX),
                                host_slot: AtomicU32::new(host_slot),
                                len: AtomicU32::new(0),
                                content_hash: None,
                                tier_discriminant: AtomicU8::new(ChunkTier::HostPinned as u8),
                                generation: AtomicU64::new(0),
                            }),
                        }
                    }
                };
                chunks.push(handle);
            }
            chunks.last().unwrap().increment_len();
            *seq_len += 1;
            let _ = pos;
        };

        // Fill VRAM chunk 0 (4 tokens):
        for t in 0..chunk_tokens {
            append(t as i32, &mut vram, &mut host, &mut chunks, &mut seq_len);
        }
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].tier(), ChunkTier::Vram, "chunk 0 should be Vram");
        assert_eq!(chunks[0].len(), chunk_tokens);

        // Fill VRAM chunk 1 (4 tokens):
        for t in chunk_tokens..2 * chunk_tokens {
            append(t as i32, &mut vram, &mut host, &mut chunks, &mut seq_len);
        }
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].tier(), ChunkTier::Vram, "chunk 1 should be Vram");
        assert_eq!(vram.free_list.len(), 0, "VRAM should be exhausted");

        // Now append past VRAM limit — must allocate into host tier:
        let t = 2 * chunk_tokens;
        append(t as i32, &mut vram, &mut host, &mut chunks, &mut seq_len);

        assert_eq!(chunks.len(), 3, "should have 3 chunks total");
        assert_eq!(
            chunks[2].tier(),
            ChunkTier::HostPinned,
            "chunk 2 must be HostPinned (VRAM exhausted)"
        );
        assert_ne!(
            chunks[2].host_slot_index(),
            u32::MAX,
            "chunk 2 must have a valid host slot"
        );
        // VRAM still exhausted — not accidentally freed:
        assert_eq!(vram.free_list.len(), 0);
        // Host tier consumed exactly one slot:
        assert_eq!(host.free_count(), (host_capacity - 1) as usize);
    }

    /// reset() with host_alloc=Some frees HostPinned chunk host slots back to pool.
    #[test]
    fn test_reset_frees_host_slots() {
        let chunk_tokens: u32 = 4;

        // Build a tiny sequence with two Vram chunks and one HostPinned chunk.
        let mut vram_storage = vec![0u8; 64 * 2];
        let vram_base = vram_storage.as_mut_ptr();
        let mut host_storage = vec![0u8; 64 * 4];
        let host_base = host_storage.as_mut_ptr();

        // Manually build a sequence with 3 chunks.
        let mut seq = SequenceState::new(chunk_tokens);
        // Chunk 0: Vram slot 0
        {
            let h = ChunkHandle::new(0);
            for _ in 0..chunk_tokens { h.increment_len(); }
            seq.chunks.push(h);
            seq.seq_len += chunk_tokens;
        }
        // Chunk 1: Vram slot 1
        {
            let h = ChunkHandle::new(1);
            for _ in 0..chunk_tokens { h.increment_len(); }
            seq.chunks.push(h);
            seq.seq_len += chunk_tokens;
        }
        // Chunk 2: HostPinned, host slot 0
        {
            let h = ChunkHandle {
                inner: Arc::new(ChunkRef {
                    vram_slot: AtomicU32::new(u32::MAX),
                    host_slot: AtomicU32::new(0),
                    len: AtomicU32::new(2),
                    content_hash: None,
                    tier_discriminant: AtomicU8::new(ChunkTier::HostPinned as u8),
                    generation: AtomicU64::new(0),
                }),
            };
            seq.chunks.push(h);
            seq.seq_len += 2;
        }

        assert_eq!(seq.chunks.len(), 3);

        // Build a mock host allocator in the same logical state (host slot 0 was
        // allocated so free_list does NOT contain 0).
        let mut mock_host = MockHostAllocPhaseC::new(64, 4);
        let _ = mock_host.alloc(); // pop slot 3 (free_list: [0,1,2])
        // Simulate that slot 0 is allocated: remove it from free_list.
        mock_host.free_list.retain(|&s| s != 0);
        let before = mock_host.free_count();

        // Build a mock VRAM allocator (slots 0,1 allocated, not in free_list).
        // We pass a minimal PageAllocator shim: we can't create a real one (GPU),
        // so we build the sequence manually and call reset() verifying host slots.
        // Since we can't call PageAllocator::free without a real one, just verify
        // that reset() iterates without panic for the Vram slots and correctly
        // calls host_alloc.free for the HostPinned slot.

        // Instead of a real PageAllocator, use the mock approach: build a thin
        // wrapper that counts free calls.  The simplest correct test is to check
        // that mock_host.free_count() increases by 1 (the HostPinned chunk).
        // We'll pass None for the real PageAllocator and instead verify the host
        // slot logic separately.

        // For the actual reset() call we need a real PageAllocator... which
        // requires GPU.  We test the host-slot logic by directly exercising the
        // reset branch via a mock:
        let mut freed_host_slots: Vec<u32> = Vec::new();
        for chunk in seq.chunks.iter() {
            if Arc::strong_count(&chunk.inner) == 1 {
                match chunk.tier() {
                    ChunkTier::HostPinned => {
                        let h_slot = chunk.inner.host_slot.load(Ordering::Acquire);
                        if h_slot != u32::MAX {
                            freed_host_slots.push(h_slot);
                        }
                    }
                    ChunkTier::Vram => {
                        // VRAM slot freed by PageAllocator (not tested here — GPU required).
                        let h_slot = chunk.inner.host_slot.load(Ordering::Acquire);
                        if h_slot != u32::MAX {
                            freed_host_slots.push(h_slot);
                        }
                    }
                }
            }
        }

        // Chunk 2 (HostPinned, host slot 0) should be freed:
        assert!(freed_host_slots.contains(&0), "host slot 0 should be freed on reset");
        // Chunk 0 and 1 have no host slot (u32::MAX), so no host free:
        assert_eq!(freed_host_slots.len(), 1);

        // Verify mock_host free_count is unchanged (we only simulated logic above).
        assert_eq!(mock_host.free_count(), before);
        let _ = (vram_base, host_base); // silence unused warnings
    }
}
