//! Multi-GPU context: P2P setup, per-device streams and events for expert parallel dispatch.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::{Device, DeviceGuard};
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};
use braidinfer_hip::module::Module;
use braidinfer_hip::staging::CrossGpuStaging;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{HipResult, ffi};

use crate::config::ModelConfig;
use crate::paged_kv::{PageAllocator, SequenceState};

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
    pub attn_q: Option<DeviceBuffer<f32>>,            // [local_nqh * head_dim]
    // attn_out/attn_gate: host-mapped UC (snl 2026-05-15 follow-up to
    // output_slots/moe_act_uc_handoff). Same §11.4-class fix: workers wrote
    // these into local VRAM UC and GPU 0 peer-read via D2D_COPY, which on
    // gfx1100 under PCIe pressure at 4+ GPU wedges MES. Routing through
    // host-mapped portable+coherent eliminates the cross-GPU peer-read.
    /// Owned as a [`CrossGpuStaging<f32>`]: per-iteration `devices` slice is
    /// `[worker.device, DeviceId(0)]`, so `.dev_ptr(0)` = worker self-view,
    /// `.dev_ptr(1)` = GPU 0 gather-side view. The cached `attn_out_dev_self`
    /// / `attn_out_dev_gpu0` raw-pointer fields are kept as hot-path
    /// pre-resolved aliases so downstream consumers don't need to revisit the
    /// staging object per decode step.
    pub attn_out: Option<CrossGpuStaging<f32>>, // [local_nqh * head_dim]
    pub attn_out_dev_self: Option<*mut f32>,
    pub attn_out_dev_gpu0: Option<*mut f32>,
    // Distributed QKV projection buffers (allocated by init_split_attn_weights)
    pub attn_normed: Option<DeviceBuffer<f32>>, // [hidden_size] — P2P copy of GPU 0's normed
    pub attn_q_gate: Option<DeviceBuffer<f32>>, // [local_nqh*hd*q_mult] — Q+gate interleaved
    pub attn_k: Option<DeviceBuffer<f32>>,      // [local_nkh*hd]
    pub attn_v: Option<DeviceBuffer<f32>>,      // [local_nkh*hd]
    // attn_gate stays as worker-VRAM UC: the DeinterleaveInst that writes it
    // emits cached vector stores, and on host-mapped pages the L2 dirty lines
    // are not flushed by the agent-scope ack fence before GPU 0's gather reads.
    // Worker-VRAM UC physical pages still serve correct values via peer-read.
    // Bisection (2026-05-15): out-host+gate-VRAM passes 3/3 with real content;
    // out-host+gate-host produced gibberish "write" tokens at 2-GPU.
    pub attn_gate: Option<DeviceBuffer<f32>>,
    // Split attention projection weights (per attention layer), stored on this GPU
    pub attn_w_q_gate: Vec<crate::quant::LinearWeight>, // [local_nqh*hd*q_mult, hs] per attn layer
    pub attn_w_k: Vec<crate::quant::LinearWeight>,      // [local_nkh*hd, hs] per attn layer
    pub attn_w_v: Vec<crate::quant::LinearWeight>,      // [local_nkh*hd, hs] per attn layer
    // Per-worker paged KV state (srg6.11/srg6.15). GQA replicates KV heads
    // across workers (local_nkh = nkh), so each worker's chunk holds the FULL
    // nkh × CHUNK_TOKENS × hd × 4 bytes. Allocated by init_attn_buffers;
    // the LIVE per-worker page/position tables read by the paged decode dispatch.
    pub page_allocator: Option<PageAllocator>,
    pub paged_seq: Option<SequenceState>,
    pub paged_page_table: Option<MappedHostBuffer<u64>>,
    pub paged_position_table: Option<MappedHostBuffer<i32>>,
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

        // bd braidinfer-sm16 / udi #3012 (IV) topology probe: dump each HIP
        // device's PCI BDF so cold-start logs map HIP-index → physical card.
        // Per-process latched-worker failures need this to discriminate
        // faulty-card vs init-sequence-race causes.
        for i in 0..num_devices {
            let mut buf = [0i8; 32];
            unsafe {
                let _ = ffi::hipDeviceGetPCIBusId(buf.as_mut_ptr(), 32, i as i32);
            }
            let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
            eprintln!("Multi-GPU: HIP {i} = PCI {}", cstr.to_string_lossy());
        }

        // Enable P2P access between all device pairs.
        // DeviceGuard saves the caller's current device and restores it on
        // drop, so the loop never leaves a stale current-device behind.
        for i in 0..num_devices {
            for j in 0..num_devices {
                if i == j {
                    continue;
                }
                let _guard = DeviceGuard::switch_to(DeviceId(i as u32))?;
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

        // Warm HWQ on each device. hipStreamCreate is the KFD HWQ first-touch
        // primitive that triggers kfd_wait_on_events on cold devices (bd
        // braidinfer-4e2m). Surfacing the first-touch here, in the bounded
        // P2P-enable phase, keeps any wedge contained to a single early step
        // instead of striking inside the per-worker push loop below where
        // partial state is harder to clean up.
        //
        // ud #3174: env BRAIDINFER_REVERSE_INIT swaps the warm-up iteration
        // order to discriminate KFD per-PASID first-queue init vs HIP queue
        // index as the source of the first-worker NaN latch. Workers vec
        // indexing remains 0..N (HIP-id-aligned); only the KFD touch order
        // is swapped.
        let warm_iter: Vec<usize> = if std::env::var("BRAIDINFER_REVERSE_INIT").is_ok() {
            (0..num_devices).rev().collect()
        } else {
            (0..num_devices).collect()
        };
        for &i in &warm_iter {
            let _guard = DeviceGuard::switch_to(DeviceId(i as u32))?;
            let warm = Stream::new(DeviceId(i as u32))?;
            warm.synchronize()?;
            // warm drops here -> hipStreamDestroy
        }
        eprintln!("Multi-GPU: HWQ warm-up complete on {num_devices} devices (order={warm_iter:?})");

        // Create workers for each device
        let mut workers = Vec::with_capacity(num_devices);
        for i in 0..num_devices {
            let device = DeviceId(i as u32);
            let _guard = DeviceGuard::switch_to(device)?;
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
                attn_q: None,
                attn_out: None,
                attn_out_dev_self: None,
                attn_out_dev_gpu0: None,
                attn_normed: None,
                attn_q_gate: None,
                attn_k: None,
                attn_v: None,
                attn_gate: None,
                attn_w_q_gate: Vec::new(),
                attn_w_k: Vec::new(),
                attn_w_v: Vec::new(),
                page_allocator: None,
                paged_seq: None,
                paged_page_table: None,
                paged_position_table: None,
            });
        }

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
        config: &ModelConfig,
        chunk_tokens: usize,
    ) -> HipResult<()> {
        let max_paged_chunks = ((max_seq_len + chunk_tokens - 1) / chunk_tokens) as u32;
        for worker in self.workers.iter_mut() {
            // DeviceGuard saves the caller's current device once per iteration
            // and restores it when the guard drops at end of iteration.
            let _worker_guard = DeviceGuard::switch_to(worker.device)?;
            worker.attn_q = Some(DeviceBuffer::<f32>::alloc(
                worker.device,
                local_nqh * head_dim,
            )?);
            // attn_out: host-mapped portable+coherent. Worker writes via its
            // own dev_ptr; GPU 0 gather reads via GPU 0's dev_ptr. Replaces
            // the prior worker-VRAM UC peer-read which triggered the §11.4
            // class wedge under PCIe pressure at 4+ GPU.
            {
                // Per-iteration staging: `[worker.device, GPU 0]`. Indexing is
                // local to this slice — `.dev_ptr(0)` is the worker self-view,
                // `.dev_ptr(1)` is the GPU 0 gather-side view. The hot-path
                // pre-resolved aliases below mirror the original API.
                let buf = CrossGpuStaging::<f32>::alloc(
                    local_nqh * head_dim,
                    &[worker.device, DeviceId(0)],
                )?;
                let self_ptr = buf.dev_ptr(0);
                let gpu0_ptr = buf.dev_ptr(1);
                worker.attn_out = Some(buf);
                worker.attn_out_dev_self = Some(self_ptr);
                worker.attn_out_dev_gpu0 = Some(gpu0_ptr);
            }
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
                // attn_gate: worker-VRAM UC. DeinterleaveInst writes via cached
                // vector stores; on host-mapped pages those L2 dirty lines are
                // not flushed by the ack fence and GPU 0's gather reads stale
                // values. Worker-VRAM UC physical pages still serve correct
                // values via P2P peer-read.
                worker.attn_gate = Some(DeviceBuffer::<f32>::alloc_uncached(
                    worker.device,
                    local_nqh * head_dim,
                )?);
            }
            // bd srg6.11 Phase A: per-worker paged KV state. Sized for FULL
            // nkh × chunk_tokens × hd × 4 bytes per chunk (KV heads replicated
            // under GQA — local_nkh = nkh). PageAllocator + position table are
            // GPU-resident; MappedHostBuffer alloc binds to the current device
            // (worker.device — DeviceGuard already switched above). page_table
            // is host-mapped (GART) so the host writes slot pointers per
            // dispatch without round-tripping through the persistent worker.
            worker.page_allocator = Some(PageAllocator::new(
                worker.device,
                config,
                chunk_tokens,
                max_paged_chunks,
            )?);
            worker.paged_seq = Some(SequenceState::new(chunk_tokens as u32));
            worker.paged_page_table =
                Some(MappedHostBuffer::<u64>::alloc(max_paged_chunks as usize)?);
            worker.paged_position_table =
                Some(MappedHostBuffer::<i32>::alloc(3 * max_seq_len)?);
        }
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
        // bd 4e2m audit candidate #1: DeviceBuffer::alloc(dst_device) silently
        // sets current device; the subsequent memcpy_d2d would then run under
        // dst_device context with NO guarantee about restoration on early-return.
        // Wrap in DeviceGuard for explicit save/restore. See bd 4e2m NOTES.
        let _guard = DeviceGuard::switch_to(dst_device)?;
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

    /// bd srg6.11 Phase B: broadcast GPU 0's paged KV chunks to each worker's
    /// per-worker paged state (Strategy A). Mirrors `broadcast_prefill_kv_to_workers`
    /// but copies WHOLE chunks instead of per-(layer, head) slabs — under GQA
    /// the per-worker chunk holds the FULL nkh × CHUNK_TOKENS × hd × 4 bytes
    /// (KV heads replicated, not sliced), so a single hipMemcpyPeerAsync per
    /// chunk per worker suffices.
    ///
    /// Pre: GPU 0's paged_seq is populated by prefill_mixed_chunk_paged (or
    /// equivalent); each worker's `page_allocator`/`paged_seq` are
    /// initialized (init_attn_buffers).
    ///
    /// Post: each worker's `paged_seq` has chunks at the same indices as
    /// GPU 0, each holding a peer-copy of the corresponding GPU 0 chunk's
    /// bytes; chunks' valid `len` matches GPU 0.
    ///
    /// bd srg6.13: wired into `prefill_batched` as a transitional no-op
    /// (multi-GPU MoE arm still uses the flat path and does NOT populate
    /// GPU 0's paged_seq, so this is effectively a no-op until srg6.12b
    /// switches the decode side and the prefill path populates paged_seq).
    pub fn broadcast_paged_chunks_to_workers(
        &mut self,
        gpu0_paged_seq: &SequenceState,
        gpu0_allocator: &PageAllocator,
    ) -> HipResult<()> {
        if self.num_devices <= 1 {
            return Ok(());
        }
        let chunk_bytes = gpu0_allocator.chunk_bytes();
        let gpu0_device = gpu0_allocator.device();

        for (chunk_idx, gpu0_chunk) in gpu0_paged_seq.chunks.iter().enumerate() {
            let gpu0_slot = gpu0_chunk.slot_index();
            let gpu0_len = gpu0_chunk.len();
            let gpu0_ptr = gpu0_allocator.slot_ptr(gpu0_slot);

            for gpu_i in 0..self.num_devices {
                if DeviceId(gpu_i as u32) == gpu0_device {
                    continue;
                }
                let worker = &mut self.workers[gpu_i];
                let worker_allocator = worker
                    .page_allocator
                    .as_mut()
                    .expect("worker.page_allocator must be initialized via init_attn_buffers");
                let worker_seq = worker
                    .paged_seq
                    .as_mut()
                    .expect("worker.paged_seq must be initialized via init_attn_buffers");

                // Ensure worker has a chunk at this index. paged_seq tracks
                // its own positions via append_token; we keep it in lockstep
                // with GPU 0 by appending positions until chunks.len() exceeds
                // chunk_idx. The position values mirror GPU 0's positions
                // vector. Allocator hands out a fresh slot whenever the
                // current chunk fills; that slot is what we peer-copy into.
                while worker_seq.chunks.len() <= chunk_idx {
                    let next_pos_idx = worker_seq.positions.len();
                    let pos = gpu0_paged_seq.positions[next_pos_idx];
                    worker_seq.append_token(pos, worker_allocator)?;
                }
                // Catch up `len` on the (possibly current) chunk so it
                // matches GPU 0's len. append_token incremented for one
                // position; we need additional bumps until the worker's
                // chunk len equals gpu0_len.
                while worker_seq.chunks[chunk_idx].len() < gpu0_len {
                    let next_pos_idx = worker_seq.positions.len();
                    let pos = gpu0_paged_seq.positions[next_pos_idx];
                    worker_seq.append_token(pos, worker_allocator)?;
                }

                let worker_slot = worker_seq.chunks[chunk_idx].slot_index();
                let worker_ptr =
                    worker_allocator.slot_ptr(worker_slot) as *mut std::ffi::c_void;
                braidinfer_hip::error::check(unsafe {
                    ffi::hipMemcpyPeerAsync(
                        worker_ptr,
                        gpu_i as i32,
                        gpu0_ptr as *const std::ffi::c_void,
                        gpu0_device.0 as i32,
                        chunk_bytes,
                        worker.compute_stream.raw(),
                    )
                })?;
            }
        }

        // Synchronize via mailbox: launch set_flag on each worker stream and
        // CPU-poll. Avoids hipStreamSynchronize (would deadlock against the
        // cooperative worker on the same device).
        use std::sync::atomic::Ordering;
        for gpu_i in 0..self.num_devices {
            if DeviceId(gpu_i as u32) == gpu0_device {
                continue;
            }
            let _guard = DeviceGuard::switch_to(DeviceId(gpu_i as u32))?;
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
                if v >= next_seq {
                    break;
                }
                if start.elapsed().as_secs() > 30 {
                    panic!(
                        "broadcast_paged_chunks_to_workers timeout gpu={gpu_i} \
                         seq={next_seq} flag_value={v}"
                    );
                }
                std::hint::spin_loop();
            }
        }
        Ok(())
    }
}
