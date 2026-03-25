use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::{DeviceBuffer, PinnedBuffer};
use braidinfer_hip::{ffi, HipError, HipResult};
use braidinfer_hip::error::check as hip_check;

use crate::model::ModelConfig;

// ---- Derived geometry (computed from ModelConfig at runtime) ----

/// KV bytes per unquantized (f32) chunk slot, computed from ModelConfig:
/// num_attn_layers * 2(K+V) * chunk_tokens * num_kv_heads * head_dim * sizeof(f32)
/// Used for staging buffers and flat KV cache.
pub fn chunk_kv_bytes(config: &ModelConfig, chunk_tokens: usize) -> usize {
    let num_attn_layers = config.layer_is_attention.iter().filter(|&&a| a).count();
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
    debug_assert_eq!(chunk_tokens, 64, "quantized chunk_tokens must equal group_size (64)");
    let num_attn_layers = config.layer_is_attention.iter().filter(|&&a| a).count();
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
    let num_gdn_layers = config.layer_is_attention.iter().filter(|&&a| !a).count();
    let floats_per_layer =
        config.linear_num_heads * config.linear_key_head_dim * config.linear_value_head_dim;
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
        debug_assert!((slot as usize) < self.capacity as usize);
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

// ---- ChunkRef + ChunkHandle (Arc-based CoW) ----

/// Shared metadata for one KV chunk slot. The actual data lives in the PageAllocator pool.
pub struct ChunkRef {
    pub slot_index: u32,
    /// Number of valid tokens written into this chunk (0..=chunk_tokens).
    pub len: AtomicU32,
    /// Content hash; set when the chunk is sealed (full). Used for radix-tree dedup.
    pub content_hash: Option<u64>,
}

/// Arc-wrapped handle to a ChunkRef. Clone-on-write: if refcount > 1, `make_exclusive`
/// allocates a new slot and copies before mutating.
#[derive(Clone)]
pub struct ChunkHandle {
    pub inner: Arc<ChunkRef>,
}

impl ChunkHandle {
    /// Wrap a freshly allocated slot.
    pub fn new(slot_index: u32) -> Self {
        ChunkHandle {
            inner: Arc::new(ChunkRef {
                slot_index,
                len: AtomicU32::new(0),
                content_hash: None,
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
        // Copy entire chunk regardless of fill level. With [H,T,D] layout, valid
        // tokens for each head are at stride offsets, not a contiguous prefix.
        // Copying only a prefix would miss valid data for heads > 0.
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
        Ok(ChunkHandle {
            inner: Arc::new(ChunkRef {
                slot_index: new_slot,
                len: AtomicU32::new(len as u32),
                content_hash: None,
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
}

// ---- SequenceState ----

/// Per-sequence KV cache tracking. Owns an ordered list of ChunkHandles.
pub struct SequenceState {
    /// Ordered chunks forming this sequence's KV cache.
    pub chunks: Vec<ChunkHandle>,
    /// Total tokens in the sequence.
    pub seq_len: u32,
    /// Monotonically increasing version; bumped on every structural change.
    pub kv_version: u32,
    /// Tokens per chunk (immutable; set at construction).
    chunk_tokens: u32,
    /// f32 staging buffer for the current incomplete chunk (quantized mode only).
    /// KV values are written here during decode; on chunk seal, the quantize kernel
    /// reads from this buffer and writes to a quantized chunk slot.
    /// None when using unquantized (f32) paged KV.
    pub staging_buffer: Option<DeviceBuffer<u8>>,
}

impl SequenceState {
    pub fn new(chunk_tokens: u32) -> Self {
        SequenceState {
            chunks: Vec::new(),
            seq_len: 0,
            kv_version: 0,
            chunk_tokens,
            staging_buffer: None,
        }
    }

    /// Create with a staging buffer for quantized KV cache.
    pub fn new_quantized(chunk_tokens: u32, device: DeviceId, config: &ModelConfig) -> HipResult<Self> {
        let staging_bytes = chunk_kv_bytes(config, chunk_tokens as usize);
        let staging = DeviceBuffer::alloc(device, staging_bytes)?;
        Ok(SequenceState {
            chunks: Vec::new(),
            seq_len: 0,
            kv_version: 0,
            chunk_tokens,
            staging_buffer: Some(staging),
        })
    }

    /// Append a token slot. Allocates a new chunk when the current one is full.
    /// Returns `Err` only if the allocator is exhausted.
    pub fn append_token(&mut self, allocator: &mut PageAllocator) -> HipResult<()> {
        let needs_new_chunk = self.chunks.is_empty()
            || self.chunks.last().unwrap().len() >= self.chunk_tokens;
        if needs_new_chunk {
            let (slot, _ptr) = allocator
                .alloc()
                .ok_or(HipError(ffi::hipErrorOutOfMemory))?;
            self.chunks.push(ChunkHandle::new(slot));
            self.kv_version += 1;
        }
        self.chunks.last().unwrap().increment_len();
        self.seq_len += 1;
        Ok(())
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
        assert!((ratio - 3.56).abs() < 0.01, "expected ~3.56x reduction, got {ratio:.2}x");
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

        // Mock allocator calls: track how many chunks would be needed
        // We can't call the real allocator (no GPU), so we manually drive logic
        // by checking the invariants.

        // Simulate: needs_new_chunk = true initially
        assert!(seq.chunks.is_empty() || seq.chunks.last().map_or(true, |c| c.len() >= chunk_tokens));

        // Manually create chunks as if append_token succeeded
        seq.chunks.push(ChunkHandle::new(0));
        seq.seq_len += 1;
        seq.chunks.last().unwrap().increment_len();

        assert_eq!(seq.seq_len, 1);
        assert_eq!(seq.current_chunk_offset(), 1);
        assert_eq!(seq.current_chunk_idx(), 0);

        // Fill chunk
        for _ in 1..chunk_tokens {
            seq.chunks.last().unwrap().increment_len();
            seq.seq_len += 1;
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

        assert_eq!(seq.chunks.len(), 2);
        assert_eq!(seq.current_chunk_idx(), 1);
        assert_eq!(seq.current_chunk_offset(), 1);
    }

    #[test]
    fn test_sequence_state_accessors() {
        let mut seq = SequenceState::new(64);
        assert!(seq.current_chunk_mut().is_none());
        assert_eq!(seq.current_chunk_offset(), 0);
        assert_eq!(seq.current_chunk_idx(), 0); // saturating_sub(1) = 0
    }
}
