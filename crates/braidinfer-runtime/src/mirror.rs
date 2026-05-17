//! SDMA-based VRAM→host mirror for decode-step debugging (snl wt1 minimal).
//!
//! Allocates pinned-host shadows of cross-GPU debug-relevant tensors:
//!   - worker.attn_kv_caches[layer].k/v per (gpu, attn_layer)
//!   - GPU 0 activations.hidden, activations.attn_out
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

use crate::weights::ActivationBuffers;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::Device;
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
        })
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
    ) -> HipResult<()> {
        // GPU 0 activations.
        Device::set_current(gpu0)?;
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
        let kv_bytes = self.local_nkh * self.max_seq_len * self.head_dim * 4;
        for (gpu_i, kv_layers) in workers_attn_kv.iter().enumerate() {
            let dev = if gpu_i == 0 {
                gpu0
            } else {
                worker_devices[gpu_i - 1]
            };
            Device::set_current(dev)?;
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
        Device::set_current(gpu0)?;
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
        let stat = |name: &str, slice: &[f32]| {
            let mut n_nan = 0usize;
            let mut max_abs = 0.0f32;
            for &x in slice {
                if x.is_nan() {
                    n_nan += 1;
                } else if x.abs() > max_abs {
                    max_abs = x.abs();
                }
            }
            eprintln!(
                "[snap {label}] {name}: n={} nan={} max_abs={:.4} first4={:?}",
                slice.len(),
                n_nan,
                max_abs,
                &slice[..slice.len().min(4)],
            );
        };
        stat("normed_stage(CPU-view)", normed_stage_host);
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
