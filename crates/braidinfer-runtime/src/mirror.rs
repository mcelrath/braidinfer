//! SDMA-based VRAM→host mirror for decode-step debugging (snl wt1 minimal).
//!
//! Allocates pinned-host shadows of cross-GPU debug-relevant tensors:
//!   - worker.attn_kv_caches[layer].k/v per (gpu, attn_layer)
//!   - GPU 0 activations.hidden, activations.attn_out
//!   - MoE output_slots (host-mapped, directly CPU-readable — no DMA)
//!   - Per-worker OP_MOE_FFN_REMOTE local_output (VRAM → SDMA shadow)
//!
//! Per-GPU SDMA streams (raw hipStream_t, created BEFORE persistent_workers
//! launch — afterwards hipStreamCreate may deadlock against the cooperative
//! kernel). `snapshot()` issues hipMemcpyAsync(DeviceToHost) per buffer on
//! its owner-GPU stream, then hipStreamSynchronize all. Per-buffer stats
//! (finite/NaN/Inf counts, max abs) printed to stderr.
//!
//! Probe basis: sdma_under_coop_fork T8-T11 (2026-05-16) showed
//! hipMemcpyPeerAsync + StreamSynchronize completes cleanly at 96-block
//! cooperative-launch occupancy (our production). H2D via the same SDMA
//! engine has even less surface than peer-pulls.

use crate::persistent_dispatch::PersistentDispatch;
use crate::weights::ActivationBuffers;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::{Device, DeviceGuard};
use braidinfer_hip::ffi;
use braidinfer_hip::memory::PinnedBuffer;

pub struct DecodeMirror {
    /// One SDMA stream per GPU, BORROWED from PersistentDispatch::sdma_streams
    /// (wt1 P2-a). DecodeMirror does NOT own these — destruction is the
    /// responsibility of PersistentDispatch::Drop. Storing raw hipStream_t
    /// here (no lifetime) is acceptable because PersistentDispatch outlives
    /// DecodeMirror within Model: persistent_workers is dropped AFTER
    /// decode_mirror in struct-field declaration order.
    streams: Vec<ffi::hipStream_t>,
    /// attn_kv[gpu_i][attn_layer] = (k_mirror, v_mirror).
    attn_kv: Vec<Vec<(PinnedBuffer<f32>, PinnedBuffer<f32>)>>,
    /// Per-worker attn_normed mirror (workers 1..N — GPU 0 reads
    /// normed_stage directly). For locating where bad data first enters
    /// the per-worker pipeline.
    worker_normed: Vec<PinnedBuffer<f32>>,
    /// Per-worker attn_q_gate (Q+gate interleaved) mirror — sized
    /// local_nqh * head_dim * q_mult.
    worker_q_gate: Vec<PinnedBuffer<f32>>,
    /// Per-worker attn_k mirror — sized local_nkh * head_dim.
    worker_k: Vec<PinnedBuffer<f32>>,
    /// GPU 0 act.hidden mirror.
    act_hidden: PinnedBuffer<f32>,
    /// GPU 0 act.attn_out mirror (full [num_gpus × local_nqh × hd]).
    act_attn_out: PinnedBuffer<f32>,
    /// Hidden size for slicing stats.
    hidden_size: usize,
    /// attn_out total floats (= num_gpus * local_nqh * head_dim).
    attn_out_floats: usize,
    /// KV head count + max_seq_len + head_dim — for layout-aware stats.
    local_nkh: usize,
    max_seq_len: usize,
    head_dim: usize,
    /// Per-worker pinned-host shadow of OP_MOE_FFN_REMOTE local_output (VRAM).
    /// Indexed by worker_idx (0 = GPU 1). Allocated by `alloc_moe_workers`.
    /// Empty if `alloc_moe_workers` was never called.
    worker_ffn_output: Vec<PinnedBuffer<f32>>,
    /// Number of worker GPUs (GPUs 1..N). Set by `alloc_moe_workers`.
    num_workers: usize,
}

unsafe impl Send for DecodeMirror {}
unsafe impl Sync for DecodeMirror {}

impl DecodeMirror {
    /// Allocate pinned-host mirrors. The caller supplies a borrowed SDMA stream
    /// per GPU (`streams[0]` = gpu0, `streams[1..]` = worker_devices, in order).
    /// Streams are owned by `PersistentDispatch::sdma_streams` (wt1 P2-a) and
    /// MUST already exist before the persistent_worker cooperative kernels
    /// launch on their GPUs. DecodeMirror does NOT destroy these streams.
    pub fn alloc(
        gpu0: DeviceId,
        worker_devices: &[DeviceId],
        streams: Vec<ffi::hipStream_t>,
        local_nkh: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_attn_layers: usize,
        hidden_size: usize,
        nqh_total: usize,
    ) -> HipResult<Self> {
        let num_gpus = 1 + worker_devices.len();
        assert_eq!(streams.len(), num_gpus, "DecodeMirror::alloc: streams.len() must equal 1 + worker_devices.len()");
        for (i, &s) in streams.iter().enumerate() {
            assert!(!s.is_null(), "DecodeMirror::alloc: streams[{i}] is null — PersistentDispatch::ensure_sdma_stream not called for this device");
        }
        // Per-GPU attn_kv mirrors. All pinned-host allocations are device-context
        // independent (hipHostMalloc).
        Device::set_current(gpu0)?;
        let kv_elems = local_nkh * max_seq_len * head_dim;
        let mut attn_kv: Vec<Vec<(PinnedBuffer<f32>, PinnedBuffer<f32>)>> =
            Vec::with_capacity(num_gpus);
        for _ in 0..num_gpus {
            let mut layer_vec = Vec::with_capacity(num_attn_layers);
            for _ in 0..num_attn_layers {
                let k = PinnedBuffer::<f32>::alloc(kv_elems)?;
                let v = PinnedBuffer::<f32>::alloc(kv_elems)?;
                layer_vec.push((k, v));
            }
            attn_kv.push(layer_vec);
        }
        let act_hidden = PinnedBuffer::<f32>::alloc(hidden_size)?;
        let attn_out_floats = nqh_total * head_dim;
        let act_attn_out = PinnedBuffer::<f32>::alloc(attn_out_floats)?;
        let mut worker_normed: Vec<PinnedBuffer<f32>> = Vec::with_capacity(worker_devices.len());
        let mut worker_q_gate: Vec<PinnedBuffer<f32>> = Vec::with_capacity(worker_devices.len());
        let mut worker_k: Vec<PinnedBuffer<f32>> = Vec::with_capacity(worker_devices.len());
        let local_nqh = nqh_total / (1 + worker_devices.len());
        for _ in worker_devices {
            worker_normed.push(PinnedBuffer::<f32>::alloc(hidden_size)?);
            // q_mult is unknown here; allocate for max (q_mult=2 = output-gate)
            worker_q_gate.push(PinnedBuffer::<f32>::alloc(local_nqh * head_dim * 2)?);
            worker_k.push(PinnedBuffer::<f32>::alloc(local_nkh * head_dim)?);
        }
        Device::set_current(gpu0)?;
        Ok(DecodeMirror {
            streams,
            attn_kv,
            worker_normed,
            worker_q_gate,
            worker_k,
            act_hidden,
            act_attn_out,
            hidden_size,
            attn_out_floats,
            local_nkh,
            max_seq_len,
            head_dim,
            worker_ffn_output: Vec::new(),
            num_workers: worker_devices.len(),
        })
    }

    /// Allocate per-worker pinned-host shadows for OP_MOE_FFN_REMOTE local_output.
    /// `worker_devices`: GPUs 1..N in order. `hidden_size`: floats per worker output.
    /// Must be called BEFORE the first MoE decode step. Safe to call multiple times
    /// (re-alloc is idempotent if dimensions match).
    pub fn alloc_moe_workers(
        &mut self,
        worker_devices: &[DeviceId],
        hidden_size: usize,
    ) -> HipResult<()> {
        if self.worker_ffn_output.len() == worker_devices.len() {
            // Already allocated.
            return Ok(());
        }
        self.worker_ffn_output.clear();
        for _ in worker_devices {
            self.worker_ffn_output.push(PinnedBuffer::<f32>::alloc(hidden_size)?);
        }
        Ok(())
    }

    /// Snapshot MoE output_slots (host-mapped, no DMA) and per-worker FFN_REMOTE
    /// local_output (SDMA copy) into their respective shadow buffers.
    ///
    /// `output_slots_host`: CPU-readable host pointer to MoeP2pContext::output_slots.
    ///   Layout: [MAX_PREFILL_BATCH × num_gpus × hidden_size]; decode uses
    ///   token 0 slots: offsets [gpu_id * hidden_size] for gpu_id in 0..num_gpus.
    /// `worker_local_outputs`: per-worker (worker_idx) VRAM pointer to local_output.
    /// `worker_devices`: GPU device IDs for workers 1..N.
    /// `num_gpus`: total GPU count (1 + num_workers).
    pub fn snapshot_moe(
        &mut self,
        output_slots_host: *const f32,
        worker_local_outputs: &[*const f32],
        worker_devices: &[DeviceId],
        num_gpus: usize,
        hidden_size: usize,
    ) -> HipResult<()> {
        // DeviceGuard saves the caller's current device and restores it on drop.
        // The inner loop uses Device::set_current for per-worker switching; the
        // guard ensures restoration to the caller's device regardless of exit path.
        let _guard = if let Some(&first_dev) = worker_devices.first() {
            Some(DeviceGuard::switch_to(first_dev)?)
        } else {
            None
        };
        // Copy MoE output_slots token-0 slots to a local snapshot (host-mapped UC,
        // directly readable — no DMA needed).
        // Layout: slot for (token=0, gpu_id) = output_slots_host + gpu_id * hidden_size.
        // We'll store them in worker_ffn_output[w] for w_idx=gpu_id-1 below, along with
        // SDMA copies — but output_slots is printed inline in print_moe_stats via the
        // host pointer, so no extra allocation is needed here.
        let _ = output_slots_host;
        let _ = num_gpus;

        // SDMA copy of each worker's local_output VRAM → pinned shadow.
        let copy_bytes = hidden_size * std::mem::size_of::<f32>();
        for (w_idx, &dev) in worker_devices.iter().enumerate() {
            if w_idx >= self.worker_ffn_output.len() || w_idx >= worker_local_outputs.len() {
                break;
            }
            Device::set_current(dev)?;
            let src = worker_local_outputs[w_idx];
            let dst = self.worker_ffn_output[w_idx].as_mut_ptr() as *mut std::ffi::c_void;
            unsafe {
                braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                    dst,
                    src as *const std::ffi::c_void,
                    copy_bytes,
                    ffi::hipMemcpyDeviceToHost,
                    self.streams[w_idx + 1],
                ))?;
            }
        }
        // Sync worker SDMA streams only (streams[1..]).
        for w_idx in 0..worker_devices.len().min(self.streams.len().saturating_sub(1)) {
            braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(self.streams[w_idx + 1]) })?;
        }
        // _guard drops here, restoring the caller's device.
        Ok(())
    }

    /// Print MoE output_slots and per-worker FFN_REMOTE local_output stats.
    /// `output_slots_host`: host pointer (portable coherent) to MoeP2pContext::output_slots.
    /// `num_gpus`: total GPU count.
    /// `hidden_size`: floats per slot.
    pub fn print_moe_stats(
        &self,
        label: &str,
        output_slots_host: *const f32,
        num_gpus: usize,
        hidden_size: usize,
    ) {
        let stat = |name: &str, slice: &[f32]| {
            let mut n_nan = 0usize;
            let mut n_inf = 0usize;
            let mut max_abs = 0.0f32;
            for &x in slice {
                if x.is_nan() { n_nan += 1; }
                else if x.is_infinite() { n_inf += 1; }
                else if x.abs() > max_abs { max_abs = x.abs(); }
            }
            eprintln!(
                "[moe {label}] {name}: n={} nan={} inf={} max_abs={:.4} first4={:?}",
                slice.len(), n_nan, n_inf, max_abs,
                &slice[..slice.len().min(4)]
            );
        };
        // output_slots: print slot for each gpu_id (token=0).
        for gpu_id in 0..num_gpus {
            let slice = unsafe {
                std::slice::from_raw_parts(output_slots_host.add(gpu_id * hidden_size), hidden_size)
            };
            stat(&format!("output_slots[gpu{}]", gpu_id), slice);
        }
        // Per-worker FFN_REMOTE local_output (post-SDMA snapshot).
        for (w_idx, buf) in self.worker_ffn_output.iter().enumerate() {
            stat(&format!("worker{}_ffn_out(gpu{})", w_idx, w_idx + 1), buf.as_slice());
        }
    }

    /// Issue DMA copies VRAM→host for all mirrored buffers, then sync all
    /// SDMA streams. Returns nothing — printable stats are dumped via
    /// `print_stats`. Caller controls current device on exit.
    pub fn snapshot(
        &mut self,
        gpu0: DeviceId,
        worker_devices: &[DeviceId],
        activations: &ActivationBuffers,
        workers_attn_kv: &[&[crate::weights::KvCache]],
        workers_attn_normed: &[Option<*const f32>],
        workers_attn_q_gate: &[Option<(*const f32, usize)>],
        workers_attn_k: &[Option<*const f32>],
        dispatch: &PersistentDispatch,
    ) -> HipResult<()> {
        // DeviceGuard saves the caller's current device and restores it on drop
        // (same fix class as ensure_sdma_stream: snapshot iterates GPUs via
        // set_current and must not leave a stale Device::current for the decode
        // hot path).
        let _guard = DeviceGuard::switch_to(gpu0)?;
        let hs_bytes = self.hidden_size * std::mem::size_of::<f32>();
        let attn_out_bytes = self.attn_out_floats * std::mem::size_of::<f32>();
        unsafe {
            braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                self.act_hidden.as_mut_ptr() as *mut std::ffi::c_void,
                activations.hidden.as_ptr() as *const std::ffi::c_void,
                hs_bytes,
                ffi::hipMemcpyDeviceToHost,
                self.streams[0],
            ))?;
            braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                self.act_attn_out.as_mut_ptr() as *mut std::ffi::c_void,
                activations.attn_out.as_ptr() as *const std::ffi::c_void,
                attn_out_bytes,
                ffi::hipMemcpyDeviceToHost,
                self.streams[0],
            ))?;
        }
        // Per-GPU attn_kv (GPU 0 first, then workers).
        // For worker GPUs (gpu_i > 0) insert L2-coherency fence per udi #567:
        //   hipEventRecord on compute stream → hipStreamWaitEvent on SDMA stream.
        // This ensures GPU N's L2 KV writes are flushed to VRAM before SDMA
        // reads (SDMA bypasses L2 on RDNA3).
        let kv_bytes = self.local_nkh * self.max_seq_len * self.head_dim * 4;
        for (gpu_i, kv_layers) in workers_attn_kv.iter().enumerate() {
            let dev = if gpu_i == 0 {
                gpu0
            } else {
                worker_devices[gpu_i - 1]
            };
            Device::set_current(dev)?;
            if gpu_i > 0 {
                // Record event on the compute stream (after ack) then make
                // the SDMA stream wait — cross-stream L2 flush fence.
                dispatch.record_kv_event(dev.0 as usize)?;
                dispatch.wait_kv_event_on_sdma(dev.0 as usize)?;
            }
            for (layer_i, kv) in kv_layers.iter().enumerate() {
                let (k_mirror, v_mirror) = &mut self.attn_kv[gpu_i][layer_i];
                unsafe {
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        k_mirror.as_mut_ptr() as *mut std::ffi::c_void,
                        kv.k.as_ptr() as *const std::ffi::c_void,
                        kv_bytes,
                        ffi::hipMemcpyDeviceToHost,
                        self.streams[gpu_i],
                    ))?;
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        v_mirror.as_mut_ptr() as *mut std::ffi::c_void,
                        kv.v.as_ptr() as *const std::ffi::c_void,
                        kv_bytes,
                        ffi::hipMemcpyDeviceToHost,
                        self.streams[gpu_i],
                    ))?;
                }
            }
        }
        // Per-worker attn_normed / attn_q_gate / attn_k (workers 1..N).
        let hs_bytes = self.hidden_size * std::mem::size_of::<f32>();
        for (w_idx, &dev) in worker_devices.iter().enumerate() {
            Device::set_current(dev)?;
            if let Some(Some(p)) = workers_attn_normed.get(w_idx + 1).copied() {
                unsafe {
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        self.worker_normed[w_idx].as_mut_ptr() as *mut std::ffi::c_void,
                        p as *const std::ffi::c_void,
                        hs_bytes,
                        ffi::hipMemcpyDeviceToHost,
                        self.streams[w_idx + 1],
                    ))?;
                }
            }
            if let Some(Some((p, n))) = workers_attn_q_gate.get(w_idx + 1).copied() {
                let copy_bytes = n.min(self.worker_q_gate[w_idx].len()) * 4;
                unsafe {
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        self.worker_q_gate[w_idx].as_mut_ptr() as *mut std::ffi::c_void,
                        p as *const std::ffi::c_void,
                        copy_bytes,
                        ffi::hipMemcpyDeviceToHost,
                        self.streams[w_idx + 1],
                    ))?;
                }
            }
            if let Some(Some(p)) = workers_attn_k.get(w_idx + 1).copied() {
                let copy_bytes = self.worker_k[w_idx].len() * 4;
                unsafe {
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        self.worker_k[w_idx].as_mut_ptr() as *mut std::ffi::c_void,
                        p as *const std::ffi::c_void,
                        copy_bytes,
                        ffi::hipMemcpyDeviceToHost,
                        self.streams[w_idx + 1],
                    ))?;
                }
            }
        }
        // Sync all streams.
        for &s in &self.streams {
            braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(s) })?;
        }
        // _guard drops here, restoring the caller's device.
        Ok(())
    }

    /// Print per-buffer stats (count, NaN, Inf, max abs) to stderr.
    /// Also receives a direct host-readable slice for normed_stage (no GPU
    /// memcpy needed since it's host-mapped) — comparing CPU-view of
    /// normed_stage against workers' attn_normed isolates whether the
    /// broadcast read garbage.
    pub fn print_stats_with_normed_stage(
        &self,
        label: &str,
        position: u32,
        normed_stage_host: &[f32],
    ) {
        self.print_stats(label, position);
        // β probe (bd braidinfer-sm16): count denormals + NaN + Inf bit-patterns
        // on GPU0's host-mapped normed_stage. If denormals appear AT GPU0 before
        // any cross-GPU transfer, the bug is producer-side (op_rmsnorm_wx), not
        // coherence. Single observation falsifies γ/δ/α per scope agent #ab6e129.
        let stat = |name: &str, slice: &[f32]| {
            let mut n_nan = 0usize;
            let mut n_inf = 0usize;
            let mut n_denorm = 0usize;
            let mut max_abs = 0.0f32;
            for &x in slice {
                let bits = x.to_bits();
                let exp = (bits >> 23) & 0xFF;
                let mant = bits & 0x7F_FFFF;
                if x.is_nan() {
                    n_nan += 1;
                } else if x.is_infinite() {
                    n_inf += 1;
                } else if exp == 0 && mant != 0 {
                    n_denorm += 1;
                } else if x.abs() > max_abs {
                    max_abs = x.abs();
                }
            }
            eprintln!(
                "[snap {label}] {name}: n={} nan={} inf={} denorm={} max_abs={:.4} first4={:?}",
                slice.len(),
                n_nan,
                n_inf,
                n_denorm,
                max_abs,
                &slice[..slice.len().min(4)],
            );
        };
        stat("normed_stage(CPU-view)", normed_stage_host);
    }

    /// Copy first 16 floats of `hidden_ptr` (VRAM, GPU 0) into the `act_hidden` pinned
    /// mirror via the GPU 0 SDMA stream, then synchronize. Prints a one-line diagnostic
    /// to stderr. Safe under the persistent cooperative kernel: uses the SDMA engine,
    /// not the CUs held by `persistent_worker`.
    ///
    /// Returns HipResult<()> so callers can propagate errors; diagnostic is always printed
    /// on success.
    pub fn snapshot_hidden_head16(
        &mut self,
        gpu0: DeviceId,
        hidden_ptr: *const f32,
        label: &str,
    ) -> HipResult<()> {
        let _guard = DeviceGuard::switch_to(gpu0)?;
        let n = 16usize.min(self.hidden_size);
        let copy_bytes = n * std::mem::size_of::<f32>();
        unsafe {
            braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                self.act_hidden.as_mut_ptr() as *mut std::ffi::c_void,
                hidden_ptr as *const std::ffi::c_void,
                copy_bytes,
                ffi::hipMemcpyDeviceToHost,
                self.streams[0],
            ))?;
            braidinfer_hip::error::check(ffi::hipStreamSynchronize(self.streams[0]))?;
        }
        let buf = &self.act_hidden.as_slice()[..n];
        let mut nan = false;
        let mut inf = false;
        let mut max_abs = 0.0f32;
        for &v in buf {
            if v.is_nan() { nan = true; }
            if v.is_infinite() { inf = true; }
            let a = v.abs();
            if a > max_abs { max_abs = a; }
        }
        eprintln!(
            "DBG hidden[{label}] nan={nan} inf={inf} max_abs={max_abs:.3e} h[0..4]={:.3e},{:.3e},{:.3e},{:.3e}",
            buf[0], buf[1], buf[2], buf[3]
        );
        // _guard drops here, restoring the caller's device.
        Ok(())
    }

    pub fn print_stats(&self, label: &str, position: u32) {
        let stat = |name: &str, slice: &[f32]| {
            let mut n_nan = 0usize;
            let mut n_inf = 0usize;
            let mut max_abs = 0.0f32;
            for &x in slice {
                if x.is_nan() {
                    n_nan += 1;
                } else if x.is_infinite() {
                    n_inf += 1;
                } else if x.abs() > max_abs {
                    max_abs = x.abs();
                }
            }
            eprintln!(
                "[snap {label} pos={position}] {name}: n={} nan={} inf={} max_abs={:.4} first4={:?}",
                slice.len(),
                n_nan,
                n_inf,
                max_abs,
                &slice[..slice.len().min(4)]
            );
        };
        stat("act.hidden", self.act_hidden.as_slice());
        stat("act.attn_out", self.act_attn_out.as_slice());
        for (w_idx, normed) in self.worker_normed.iter().enumerate() {
            stat(&format!("g{}.attn_normed", w_idx + 1), normed.as_slice());
        }
        for (w_idx, qg) in self.worker_q_gate.iter().enumerate() {
            stat(&format!("g{}.attn_q_gate", w_idx + 1), qg.as_slice());
        }
        for (w_idx, k) in self.worker_k.iter().enumerate() {
            stat(&format!("g{}.attn_k", w_idx + 1), k.as_slice());
        }
        // Per-(gpu, layer) KV stats — restrict to position+1 elements per head
        // to keep output focused on the live range.
        let used = (position as usize + 1).min(self.max_seq_len);
        for (gpu_i, kv_layers) in self.attn_kv.iter().enumerate() {
            // Only print first 2 layers per GPU to keep output digestible.
            for layer_i in 0..kv_layers.len().min(2) {
                let (k, v) = &kv_layers[layer_i];
                // Slice head 0's first `used` positions × head_dim.
                let slice_len = used * self.head_dim;
                stat(
                    &format!("g{gpu_i}.kv[{layer_i}].k(h0,p0..{used})"),
                    &k.as_slice()[..slice_len],
                );
                stat(
                    &format!("g{gpu_i}.kv[{layer_i}].v(h0,p0..{used})"),
                    &v.as_slice()[..slice_len],
                );
            }
        }
    }
}

// wt1 P2-a: no Drop impl — streams are borrowed from
// PersistentDispatch::sdma_streams (destroyed in that type's Drop) and the
// PinnedBuffers handle their own hipHostFree on drop. DecodeMirror has no
// owned HIP resources to release.

// ---- KvChunkMirror (wt1 P2-c) ----

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
