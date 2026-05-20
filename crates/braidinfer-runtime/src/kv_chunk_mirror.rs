//! Write-through KV chunk mirror (wt1 P2-c).
//!
//! Moved from mirror.rs during Phase 2b. Independent of DecodeMirror.

use braidinfer_hip::HipResult;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::PinnedBuffer;

/// Write-through KV mirror: one pinned-host copy per sealed chunk, flushed via
/// SDMA at chunk-seal boundaries. Provides a host-visible snapshot for
/// debugging/testing with a bounded mirror lag of at most 1 chunk (≤ CHUNK_TOKENS
/// tokens after the seal fires).
///
/// `snapshot()` returns (data_ptr, seq_pos_of_last_drain) so callers cannot
/// treat the mirror as the live VRAM truth — the seq_pos stamp shows exactly
/// how many tokens were visible when the last flush completed.
pub struct KvChunkMirror {
    /// One pinned buffer per sealed chunk, in seal order.
    /// Each buffer holds chunk_bytes bytes copied from VRAM via SDMA async.
    pub chunks: Vec<PinnedBuffer<u8>>,
    /// Sequence position of the last token in the most recently drained
    /// (hipStreamSynchronize-completed) chunk. u32::MAX = no drain yet.
    pub seq_pos_of_last_drain: u32,
    /// Byte size of one chunk (all layers K+V interleaved, same layout as VRAM).
    pub chunk_bytes: usize,
}

impl KvChunkMirror {
    pub fn new(chunk_bytes: usize) -> Self {
        KvChunkMirror {
            chunks: Vec::new(),
            seq_pos_of_last_drain: u32::MAX,
            chunk_bytes,
        }
    }

    /// Enqueue an async VRAM→host copy of the just-sealed chunk.
    /// `vram_ptr` is the base of the sealed chunk slot (GPU VRAM, device pointer).
    /// `stream` is the SDMA stream for the owning GPU.
    /// The copy is in-flight after this call; call `drain()` to synchronize.
    ///
    /// # Safety
    /// `vram_ptr` must remain valid until `drain()` completes for this chunk.
    pub fn enqueue_chunk(
        &mut self,
        vram_ptr: *const u8,
        stream: ffi::hipStream_t,
    ) -> HipResult<()> {
        let mut host_buf = PinnedBuffer::<u8>::alloc(self.chunk_bytes)?;
        braidinfer_hip::error::check(unsafe {
            ffi::hipMemcpyAsync(
                host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                vram_ptr as *const std::ffi::c_void,
                self.chunk_bytes,
                ffi::hipMemcpyDeviceToHost,
                stream,
            )
        })?;
        self.chunks.push(host_buf);
        Ok(())
    }

    /// Synchronize the SDMA stream and record the sequence position of the last
    /// drained chunk. After this call, `chunks.last()` contains coherent data.
    /// `sealed_chunk_last_pos` is the sequence position of the last token in
    /// the chunk just enqueued (= chunk_end_position = (chunk_idx+1)*CHUNK_TOKENS - 1).
    pub fn drain(&mut self, sealed_chunk_last_pos: u32, stream: ffi::hipStream_t) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(stream) })?;
        self.seq_pos_of_last_drain = sealed_chunk_last_pos;
        Ok(())
    }

    /// Return a reference to the most recently drained chunk data and the
    /// sequence position stamp. Callers must not treat this as live VRAM state —
    /// up to 1 chunk of lag is possible between drain and next token.
    pub fn snapshot(&self) -> Option<(&[u8], u32)> {
        self.chunks.last().map(|b| (b.as_slice(), self.seq_pos_of_last_drain))
    }
}
