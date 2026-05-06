//! Multi-GPU context: P2P setup, per-device streams and events for expert parallel dispatch.

use crate::kernel::{DeinterleaveKernel, GqaAttentionKernel, MRoPEKernel, QkNormKernel};
use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{HipResult, ffi};

/// Opaque HIP event wrapper. NOT Send — pinned to creation device.
pub struct HipEvent {
    raw: ffi::hipEvent_t,
}

impl HipEvent {
    pub fn new() -> HipResult<Self> {
        let mut raw = std::ptr::null_mut();
        braidinfer_hip::error::check(unsafe { ffi::hipEventCreate(&mut raw) })?;
        Ok(HipEvent { raw })
    }

    pub fn raw(&self) -> ffi::hipEvent_t {
        self.raw
    }

    pub fn record(&self, stream: &Stream) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe { ffi::hipEventRecord(self.raw, stream.raw()) })
    }

    pub fn synchronize(&self) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe { ffi::hipEventSynchronize(self.raw) })
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::hipEventDestroy(self.raw) };
        }
    }
}

/// Per-GPU resources for expert parallel dispatch.
pub struct GpuWorker {
    pub device: DeviceId,
    pub compute_stream: Stream,
    // Compute-path P2P copy kernel (avoids SDMA PERMISSION_FAULT on RDNA3 PCIe)
    pub peer_copy_module: Module,
    // sync_flag module: <<<1,1>>> set_flag_kernel + wait_flag_kernel for
    // CPU-poll-based stream waits without HIP API calls (avoids deadlock with
    // cooperative kernels on the same device).
    pub sync_flag_module: Module,
    // Host-mapped flag written by set_flag_kernel; CPU spins on this in lieu
    // of compute_stream.synchronize(). Monotonic seq counter — each stream
    // chain ends with set_flag_kernel writing the next seq, CPU polls for it.
    pub compute_done_flag: braidinfer_hip::memory::MappedHostBuffer<u32>,
    pub compute_done_seq: std::sync::atomic::AtomicU32,
    // Per-worker position_ids buffer. activations.position_ids on GPU 0 is a
    // NON-PORTABLE MappedHostBuffer — its device pointer is only valid on
    // GPU 0. Workers reading via that pointer get invalid memory → MROPE
    // computes wrong rotation → attention is wrong. Each worker gets its own
    // host-mapped i32[3] buffer that the host writes per decode step.
    pub position_ids_local: braidinfer_hip::memory::MappedHostBuffer<i32>,
    // Head-parallel attention buffers (allocated by init_attn_buffers after construction)
    pub attn_kv_caches: Vec<crate::weights::KvCache>, // [num_attn_layers], each [local_nkh, max_seq_len, hd]
    pub attn_q: Option<DeviceBuffer<f32>>,            // [local_nqh * head_dim]
    pub attn_out: Option<DeviceBuffer<f32>>,          // [local_nqh * head_dim]
    // Distributed QKV projection buffers (allocated by init_split_attn_weights)
    pub attn_normed: Option<DeviceBuffer<f32>>, // [hidden_size] — P2P copy of GPU 0's normed
    pub attn_q_gate: Option<DeviceBuffer<f32>>, // [local_nqh*hd*q_mult] — Q+gate interleaved
    pub attn_k: Option<DeviceBuffer<f32>>,      // [local_nkh*hd]
    pub attn_v: Option<DeviceBuffer<f32>>,      // [local_nkh*hd]
    pub attn_gate: Option<DeviceBuffer<f32>>,   // [local_nqh*hd] — gate (split from q_gate)
    // Split attention projection weights (per attention layer), stored on this GPU
    pub attn_w_q_gate: Vec<crate::quant::LinearWeight>, // [local_nqh*hd*q_mult, hs] per attn layer
    pub attn_w_k: Vec<crate::quant::LinearWeight>,      // [local_nkh*hd, hs] per attn layer
    pub attn_w_v: Vec<crate::quant::LinearWeight>,      // [local_nkh*hd, hs] per attn layer
    // Kernels for kbk attention dispatch on GPUs 1+ (GPU 0 uses persistent worker)
    pub qk_norm_kernel: QkNormKernel,
    pub mrope_kernel: MRoPEKernel,
    pub gqa_kernel: GqaAttentionKernel,
    pub deinterleave_kernel: DeinterleaveKernel,
}

/// Multi-GPU context for expert parallel dispatch.
pub struct MultiGpuContext {
    pub num_devices: usize,
    pub workers: Vec<GpuWorker>,
}

impl MultiGpuContext {
    /// Initialize multi-GPU context with P2P access.
    /// Returns None if only 1 GPU available.
    pub fn init(hidden_size: usize, max_expert_is: usize) -> HipResult<Option<Self>> {
        let num_devices = Device::count()? as usize;
        if num_devices <= 1 {
            return Ok(None);
        }

        // Enable P2P access between all device pairs
        for i in 0..num_devices {
            for j in 0..num_devices {
                if i == j {
                    continue;
                }
                Device::set_current(DeviceId(i as u32))?;
                let mut can_access = 0i32;
                unsafe {
                    ffi::hipDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);
                }
                if can_access != 0 {
                    let rc = unsafe { ffi::hipDeviceEnablePeerAccess(j as i32, 0) };
                    // Ignore "already enabled" and "invalid device" (P2P not supported on this topology)
                    if rc != 0
                        && rc != ffi::hipErrorInvalidDevice
                        && rc != ffi::hipErrorPeerAccessAlreadyEnabled
                    {
                        eprintln!("Warning: hipDeviceEnablePeerAccess({i}→{j}) failed: rc={rc}");
                    } else {
                        eprintln!("P2P: GPU {i}→{j} enabled (can_access={can_access})");
                    }
                } else {
                    eprintln!("Warning: P2P not available between GPU {i} and GPU {j}");
                }
            }
        }

        // Create workers for each device
        let mut workers = Vec::with_capacity(num_devices);
        for i in 0..num_devices {
            let device = DeviceId(i as u32);
            Device::set_current(device)?;
            eprintln!(
                "Multi-GPU: allocating buffers on GPU {i} (hs={hidden_size}, eis={max_expert_is})"
            );
            workers.push(GpuWorker {
                device,
                compute_stream: Stream::new(device)?,
                peer_copy_module: Module::load(
                    device,
                    &crate::kernel::kernel_dir().join("peer_copy.hsaco"),
                )?,
                sync_flag_module: Module::load(
                    device,
                    &crate::kernel::kernel_dir().join("sync_flag.hsaco"),
                )?,
                compute_done_flag: braidinfer_hip::memory::MappedHostBuffer::<u32>::alloc(1)?,
                compute_done_seq: std::sync::atomic::AtomicU32::new(0),
                position_ids_local: braidinfer_hip::memory::MappedHostBuffer::<i32>::alloc(3)?,
                attn_kv_caches: Vec::new(),
                attn_q: None,
                attn_out: None,
                attn_normed: None,
                attn_q_gate: None,
                attn_k: None,
                attn_v: None,
                attn_gate: None,
                attn_w_q_gate: Vec::new(),
                attn_w_k: Vec::new(),
                attn_w_v: Vec::new(),
                qk_norm_kernel: QkNormKernel::load(device)?,
                mrope_kernel: MRoPEKernel::load(device)?,
                gqa_kernel: GqaAttentionKernel::load(device)?,
                deinterleave_kernel: DeinterleaveKernel::load(device)?,
            });
        }

        Device::set_current(DeviceId(0))?;
        eprintln!("Multi-GPU: {num_devices} devices, P2P enabled");

        Ok(Some(MultiGpuContext {
            num_devices,
            workers,
        }))
    }

    /// Allocate head-parallel attention buffers for all workers.
    /// Must be called after init(), before compile_multi_gpu.
    pub fn init_attn_buffers(
        &mut self,
        num_attn_layers: usize,
        local_nqh: usize,
        local_nkh: usize,
        head_dim: usize,
        max_seq_len: usize,
        hidden_size: usize,
        q_mult: usize,
    ) -> HipResult<()> {
        for worker in self.workers.iter_mut() {
            Device::set_current(worker.device)?;
            worker.attn_q = Some(DeviceBuffer::<f32>::alloc(
                worker.device,
                local_nqh * head_dim,
            )?);
            // attn_out is peer-read by GPU 0's persistent worker via D2D_COPY
            // gather. Without UC, GPU 0's L2 may serve stale entries from prior
            // decode steps (no KMD L2 invalidation between CPU-spin and the
            // gather kernel — see GFX1100_ARCH.md §5.1).
            worker.attn_out = Some(DeviceBuffer::<f32>::alloc_uncached(
                worker.device,
                local_nqh * head_dim,
            )?);
            // Distributed QKV projection activation buffers
            worker.attn_normed = Some(DeviceBuffer::<f32>::alloc(worker.device, hidden_size)?);
            worker.attn_q_gate = Some(DeviceBuffer::<f32>::alloc(
                worker.device,
                local_nqh * head_dim * q_mult,
            )?);
            worker.attn_k = Some(DeviceBuffer::<f32>::alloc(
                worker.device,
                local_nkh * head_dim,
            )?);
            worker.attn_v = Some(DeviceBuffer::<f32>::alloc(
                worker.device,
                local_nkh * head_dim,
            )?);
            if q_mult > 1 {
                // attn_gate is also peer-read by GPU 0's gather (alongside attn_out).
                worker.attn_gate = Some(DeviceBuffer::<f32>::alloc_uncached(
                    worker.device,
                    local_nqh * head_dim,
                )?);
            }
            for _ in 0..num_attn_layers {
                worker.attn_kv_caches.push(crate::weights::KvCache {
                    k: DeviceBuffer::<f32>::alloc(
                        worker.device,
                        local_nkh * max_seq_len * head_dim,
                    )?,
                    v: DeviceBuffer::<f32>::alloc(
                        worker.device,
                        local_nkh * max_seq_len * head_dim,
                    )?,
                });
            }
        }
        Device::set_current(DeviceId(0))?;
        eprintln!(
            "Multi-GPU attn: {} layers × {} workers, local_nqh={local_nqh} local_nkh={local_nkh} q_mult={q_mult}",
            num_attn_layers, self.num_devices
        );
        Ok(())
    }

    /// Copy a contiguous slice of rows from `src` (on GPU 0) to a new DeviceBuffer on `dst_device`.
    /// `row_start`, `num_rows`, `in_dim` are logical (pre-quantization layout).
    /// Returns a LinearWeight with the correct format, out_dim=num_rows, in_dim=in_dim.
    pub fn copy_weight_slice(
        src: &crate::quant::LinearWeight,
        dst_device: DeviceId,
        row_start: usize,
        num_rows: usize,
        in_dim: usize,
    ) -> HipResult<crate::quant::LinearWeight> {
        use braidinfer_hip::memory::DeviceBuffer;
        let byte_offset = src.row_byte_offset_dim(row_start, in_dim);
        let byte_len = src.row_byte_offset_dim(num_rows, in_dim);
        let dst_buf = DeviceBuffer::<u8>::alloc(dst_device, byte_len)?;
        let src_ptr = unsafe { src.raw_data_ptr().add(byte_offset) };
        braidinfer_hip::memory::memcpy_d2d(dst_buf.as_write_ptr(), src_ptr, byte_len)?;
        // Always return Packed — WeightFormat::Bf16 in Packed is valid and used by forward_sub.
        Ok(crate::quant::LinearWeight::Packed(
            crate::quant::PackedWeights {
                data: dst_buf,
                format: src.weight_format(),
                out_dim: num_rows,
                in_dim,
            },
        ))
    }

    /// Async P2P copy using compute kernel (avoids SDMA PERMISSION_FAULT on RDNA3 PCIe).
    /// Launched on `stream` from the source device. `dst` must be peer-accessible from src_device.
    pub fn peer_copy_async(
        dst: *mut u8,
        src: *const u8,
        size: usize,
        peer_copy_module: &Module,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = peer_copy_module.get_function("peer_copy_kernel")?;
        let threads = 256usize;
        let blocks = (size + threads - 1) / threads;
        let n = size as u64;
        let mut args: [*mut std::ffi::c_void; 3] = [
            &dst as *const _ as *mut std::ffi::c_void,
            &src as *const _ as *mut std::ffi::c_void,
            &n as *const _ as *mut std::ffi::c_void,
        ];
        func.launch(
            (blocks as u32, 1, 1),
            (threads as u32, 1, 1),
            0,
            stream,
            &mut args,
        )
    }

    /// Make a stream wait for an event (cross-stream synchronization).
    pub fn stream_wait_event(stream: &Stream, event: &HipEvent) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe {
            ffi::hipStreamWaitEvent(stream.raw(), event.raw(), 0)
        })
    }

    /// Stream-side mailbox-set: enqueue a `<<<1,1>>>` kernel that writes
    /// `value` to the host-mapped `flag` after a `__threadfence_system()`.
    /// Used to signal end-of-stream-work to the host without
    /// `hipStreamSynchronize`, which deadlocks while a cooperative kernel
    /// is running on the same device. The CPU should poll the host pointer
    /// of the same MappedHostBuffer with `read_volatile`.
    pub fn launch_set_flag(
        sync_flag_module: &Module,
        flag_dev_ptr: *mut u32,
        value: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = sync_flag_module.get_function("set_flag_kernel")?;
        let mut args: [*mut std::ffi::c_void; 2] = [
            &flag_dev_ptr as *const _ as *mut std::ffi::c_void,
            &value as *const _ as *mut std::ffi::c_void,
        ];
        func.launch((1, 1, 1), (1, 1, 1), 0, stream, &mut args)
    }

    /// Broadcast prefill K/V from GPU 0's `legacy_kv_caches` to each worker's
    /// `attn_kv_caches` (fixes braidinfer-sew). Multi-GPU prefill writes K/V
    /// only to GPU 0's flat-KV cache; the head-parallel decode path reads
    /// per-GPU local `attn_kv_caches`, leaving positions 0..prefill_len-1
    /// uninitialized → garbage attention output. This copies the prefill
    /// slice to every worker after prefill completes (cheap: nkh×prefill_len×hd
    /// floats per (layer, worker)).
    ///
    /// Uses hipMemcpyPeerAsync (SDMA — doesn't require GPU compute, so it
    /// runs concurrently with the cooperative moe_worker_kernel on workers).
    /// Synchronizes via the sync_flag mailbox to avoid hipStreamSynchronize
    /// deadlocking against the cooperative moe_worker.
    pub fn broadcast_prefill_kv_to_workers(
        &self,
        legacy_kv_caches: &[crate::weights::KvCache],
        attn_to_kv_idx: &[usize],
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        prefill_len: usize,
    ) -> HipResult<()> {
        if prefill_len == 0 || self.num_devices <= 1 {
            return Ok(());
        }
        let head_elems = max_seq_len * head_dim;
        let copy_bytes = prefill_len * head_dim * std::mem::size_of::<f32>();

        // DIAGNOSTIC: skip GPU 0 — the previous "first 4 tokens coherent"
        // result came from this configuration, suggesting GPU 0's
        // attn_kv_caches might already be valid from another path.
        for (attn_i, &kv_i) in attn_to_kv_idx.iter().enumerate() {
            let src_kv = &legacy_kv_caches[kv_i];
            for gpu_i in 1..self.num_devices {
                let worker = &self.workers[gpu_i];
                let dst_kv = &worker.attn_kv_caches[attn_i];
                for h in 0..num_kv_heads {
                    let src_k = unsafe { src_kv.k.as_ptr().add(h * head_elems) };
                    let dst_k = unsafe { dst_kv.k.as_write_ptr().add(h * head_elems) };
                    let src_v = unsafe { src_kv.v.as_ptr().add(h * head_elems) };
                    let dst_v = unsafe { dst_kv.v.as_write_ptr().add(h * head_elems) };
                    braidinfer_hip::error::check(unsafe {
                        ffi::hipMemcpyPeerAsync(
                            dst_k as *mut std::ffi::c_void,
                            gpu_i as i32,
                            src_k as *const std::ffi::c_void,
                            0,
                            copy_bytes,
                            worker.compute_stream.raw(),
                        )
                    })?;
                    braidinfer_hip::error::check(unsafe {
                        ffi::hipMemcpyPeerAsync(
                            dst_v as *mut std::ffi::c_void,
                            gpu_i as i32,
                            src_v as *const std::ffi::c_void,
                            0,
                            copy_bytes,
                            worker.compute_stream.raw(),
                        )
                    })?;
                }
            }
        }

        // Synchronize via mailbox: launch set_flag on each worker stream and
        // CPU-poll. Avoids hipStreamSynchronize, which would deadlock against
        // the cooperative moe_worker_kernel running on the worker GPU.
        use std::sync::atomic::Ordering;
        for gpu_i in 1..self.num_devices {
            Device::set_current(DeviceId(gpu_i as u32))?;
            let worker = &self.workers[gpu_i];
            let next_seq = worker.compute_done_seq.fetch_add(1, Ordering::Relaxed) + 1;
            Self::launch_set_flag(
                &worker.sync_flag_module,
                worker.compute_done_flag.as_write_ptr(),
                next_seq,
                &worker.compute_stream,
            )?;
            let host_ptr = worker.compute_done_flag.host_ptr();
            let start = std::time::Instant::now();
            loop {
                let v = unsafe { std::ptr::read_volatile(host_ptr) };
                if v >= next_seq { break; }
                if start.elapsed().as_secs() > 30 {
                    panic!(
                        "broadcast_prefill_kv_to_workers timeout gpu={gpu_i} \
                         seq={next_seq} flag_value={v}"
                    );
                }
                std::hint::spin_loop();
            }
        }
        Device::set_current(DeviceId(0))?;
        Ok(())
    }
}
