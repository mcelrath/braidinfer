//! CPU-scheduled persistent worker dispatch (braidinfer-czl).
//! Each GPU runs a persistent cooperative kernel polling a host-mapped work queue.
//! CPU sequences operations via memory writes — no HIP API calls in the hot path.
//!
//! # HIP API PROHIBITION (root cause of every shutdown/cleanup hang)
//!
//! Calling ANY HIP runtime API on a device with a running cooperative kernel deadlocks.
//! HIP internally calls SyncAllStreams → releaseGpuMemoryFence → signal_wait_scacquire,
//! which waits for the cooperative kernel to complete. Persistent kernels never complete.
//!
//! PROHIBITED while persistent kernels run (includes implicit calls from Drop impls):
//!   hipFree (DeviceBuffer::drop), hipHostFree (MappedHostBuffer::drop),
//!   hipMalloc, hipHostMalloc, hipMemcpy, hipStreamSynchronize (stream.synchronize()),
//!   hipStreamQuery (stream.is_idle()), hipDeviceSynchronize, hipModuleLaunchKernel
//!
//! ALLOWED: read_volatile/write_volatile on MappedHostBuffer::host_ptr() (CPU ↔ system RAM)
//!
//! SHUTDOWN SEQUENCE:
//!   1. write_volatile(shutdown_flag, 1) on each worker's host-mapped queue
//!   2. Kernel writes completion_flag=1 to host-mapped memory before returning
//!   3. CPU polls completion_flag via read_volatile (NOT hipStreamQuery)
//!   4. After ALL completion_flags set → safe to call hipFree/hipHostFree
//!   5. Timeout 10s: leak resources rather than deadlock on hipFree
//!
//! IMPLEMENTATION: Wrap GPU resources in ManuallyDrop. Only free after
//! confirming kernel exit via host-mapped completion_flag.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::{Device, DeviceGuard};
use braidinfer_hip::ffi;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;

use crate::megakernel::{INST_SIZE, Instruction};
use crate::watchdog::WatchdogThread;

/// Max instructions per batch dispatch (dense worker).
pub const MAX_BATCH_INSTRUCTIONS: usize = 256;

/// Process-global flag set by SIGINT/SIGTERM handler. Polled by dispatch_batch's
/// ack-spin-loop so an interrupt can break out and trigger Drop-time shutdown of
/// the persistent worker. Without this, default Rust signal disposition kills the
/// process immediately, leaving cooperative kernels orphaned on the GPU.
static SHUTDOWN_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static SIGNAL_HANDLERS_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

extern "C" fn shutdown_signal_handler(_sig: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Install SIGINT/SIGTERM handlers that set SHUTDOWN_REQUESTED. Idempotent.
/// Must be called before launching the persistent worker so a signal between
/// launch and the first dispatch is observed.
fn install_signal_handlers_once() {
    use std::sync::atomic::Ordering;
    if SIGNAL_HANDLERS_INSTALLED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    unsafe {
        libc::signal(libc::SIGINT, shutdown_signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, shutdown_signal_handler as libc::sighandler_t);
    }
}

/// True if a SIGINT/SIGTERM has been received since process start (or last reset).
/// dispatch_batch checks this to abort an ack-spin and trigger an orderly shutdown.
pub(crate) fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(std::sync::atomic::Ordering::SeqCst)
}

/// Idempotent promotion guard for [`try_promote_dispatch_thread`]: ensures
/// the SCHED_FIFO + affinity calls fire only once even when callers
/// invoke from multiple decode entry points.
static DISPATCH_PROMOTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Promote the calling thread to SCHED_FIFO + pin it to a single CPU,
/// gated by the `BRAIDINFER_DISPATCH_RT=1` env var. Idempotent across
/// multiple call sites.
///
/// When enabled:
/// - Pins the calling thread to `BRAIDINFER_DISPATCH_CPU` (default 55,
///   matching the 8-GPU box recipe in `README.md` — `isolcpus=55-63`,
///   amdgpu IRQs on 56-63).
/// - Sets `SCHED_FIFO` priority to `BRAIDINFER_DISPATCH_PRIO` (default
///   50, mid-range so future IRQ-thread promotion can sit above us).
///
/// Prerequisites for SCHED_FIFO to succeed:
/// - `/etc/security/limits.conf` includes `<user> - rtprio 99` and the
///   shell has been re-logged or `sudo prlimit --rtprio=99 --pid=$$`
///   has been applied.
/// - `/proc/sys/kernel/sched_rt_runtime_us = -1` (otherwise default
///   95% throttle injects 50 ms stalls per second on a 100%-CPU
///   spin-poll).
///
/// Returns `Ok(true)` if promotion happened on this call, `Ok(false)`
/// if already promoted or the env flag is unset, and `Err` describing
/// the syscall failure (most commonly `EPERM` from rtprio limit not
/// being raised).
///
/// See PLAN-dispatch-daemon-phase4.md decision D-P4-CPU.
pub fn try_promote_dispatch_thread() -> Result<bool, String> {
    use std::sync::atomic::Ordering;

    let want_rt = std::env::var("BRAIDINFER_DISPATCH_RT")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !want_rt {
        return Ok(false);
    }
    if DISPATCH_PROMOTED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(false);
    }

    let cpu: usize = std::env::var("BRAIDINFER_DISPATCH_CPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(55);
    let prio: libc::c_int = std::env::var("BRAIDINFER_DISPATCH_PRIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    // sched_setaffinity first so the SCHED_FIFO promotion lands on the
    // intended CPU (otherwise the kernel would migrate us under FIFO,
    // briefly defeating the affinity intent).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let r = libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        if r != 0 {
            return Err(format!(
                "sched_setaffinity(cpu={cpu}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
    }

    unsafe {
        let param = libc::sched_param {
            sched_priority: prio,
        };
        let r = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
        if r != 0 {
            return Err(format!(
                "sched_setscheduler(SCHED_FIFO, prio={prio}) failed: {}. \
                 Verify /etc/security/limits.conf has 'rtprio 99' for this user \
                 and ulimit -r reports 99 (re-login or `sudo prlimit --rtprio=99 \
                 --pid=$$`). See README.md 'Host system tuning'.",
                std::io::Error::last_os_error()
            ));
        }
    }

    eprintln!(
        "[braidinfer] dispatch thread promoted: SCHED_FIFO prio={prio} cpu={cpu}"
    );
    Ok(true)
}

/// Recoverable error from the dispatch ack-spin path. The legacy
/// in-process call sites translate this into `panic!()` (preserving
/// existing semantics), but a long-running daemon (braidinfer-wks)
/// converts it into per-session RPC errors instead. See PLAN-dispatch-daemon.md.
#[derive(Debug)]
pub enum DispatchError {
    /// SIGINT/SIGTERM received during the ack-spin. Caller should drain
    /// in-flight work and return an error to its client.
    ShutdownRequested { gpu: usize, seq: u32 },
    /// 30s timeout without ack progress. Indicates a wedged kernel;
    /// supervisor should SIGKILL the daemon (kb braidinfer-wks Phase 3).
    Timeout {
        gpu: usize,
        seq: u32,
        ack: u32,
        progress_pc: u32,
        /// Phase 0b wedge diagnostic 2026-05-13: count of blocks that
        /// reached the kernel's first instruction (atomicAdd at entry).
        /// Compare to launched num_blocks: smaller = cooperative grid
        /// never fully scheduled (#5 hypothesis); equal = wedge downstream.
        block_alive_count: u32,
    },
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::ShutdownRequested { gpu, seq } => write!(
                f,
                "dispatch interrupted by SIGINT/SIGTERM (gpu={gpu}, seq={seq})"
            ),
            DispatchError::Timeout {
                gpu,
                seq,
                ack,
                progress_pc,
                block_alive_count,
            } => write!(
                f,
                "dispatch timeout gpu={gpu} seq={seq} ack={ack} progress_pc={progress_pc:#010x} block_alive_count={block_alive_count}"
            ),
        }
    }
}

impl std::error::Error for DispatchError {}

/// Dispatch surface for [`PersistentDispatch`]. Kept as a trait (rather
/// than inherent methods on PersistentDispatch) so call sites can hold
/// `&mut dyn BatchDispatcher` — historically there was a second impl
/// (`PerBatchDispatch`, deleted by braidinfer-wuf.2) and the indirection
/// kept dispatch sites uniform. The single-impl form is retained for
/// testability and future alternate dispatch strategies.
pub(crate) trait BatchDispatcher {
    /// Fire a single batch asynchronously and return a per-GPU seq token.
    /// Caller must `wait_ack(gpu, seq)` before reading any buffer the
    /// batch writes to.
    fn dispatch_batch_fire(&mut self, gpu_idx: usize, instructions: &[Instruction]) -> u32;
    /// Fire + wait in one call. Equivalent to dispatch_batch_fire +
    /// wait_ack(returned seq).
    fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]);
    /// Chunk into MAX_BATCH_INSTRUCTIONS slices, dispatch each via
    /// dispatch_batch (synchronous chunks). Used for combined-segment
    /// dispatches that exceed MAX_BATCH (rare).
    fn dispatch_batch_slice(&mut self, gpu_idx: usize, instructions: &[Instruction]);
    /// Block until the given seq has acked on `gpu_idx`. Per-GPU FIFO,
    /// so any seq drains everything queued ≤ it.
    fn wait_ack(&self, gpu_idx: usize, seq: u32);
    /// Wait for multiple (gpu, seq) targets in a single pass. Implementers
    /// may parallel-poll (PersistentDispatch) or deduplicate by GPU then
    /// Multi-target wait used by hybrid layer dispatch.
    fn try_wait_acks_many(&self, targets: &[(usize, u32)]) -> Result<(), DispatchError>;
}

/// Rust mirror of WorkerQueue from persistent_worker.hip.
/// Layout must match exactly (repr(C)).
#[repr(C)]
pub struct WorkerQueueLayout {
    pub seq_num: u32,
    pub shutdown: u32,
    pub num_instructions: u32, // how many instructions in this batch (1..MAX_BATCH)
    /// Diagnostic counter: each block's thread 0 atomicAdds 1 at kernel
    /// entry. Tells the host whether the cooperative grid is fully
    /// scheduled (count == num_blocks) or only some blocks landed
    /// (count < num_blocks). Read in DispatchError::Timeout message.
    pub block_alive_count: u32,
    pub inst: [u64; MAX_BATCH_INSTRUCTIONS * INST_SIZE], // instruction batch buffer
    pub ack: u32,
    pub done: u32, // kernel writes 1 when exiting after shutdown (for Drop polling)
    pub progress_pc: u32, // kernel writes pc before each instruction (for timeout diagnosis)
    pub _pad2: u32,
    // op_profile: GPU-resident u64 buffer for per-op cycle profiling.
    // Null when BRAIDINFER_OP_PROFILE build flag is unset. See
    // crates/braidinfer-runtime/src/op_profile.rs and kernels/op_profile.h.
    pub op_profile: *mut u64,
    // Trace-dump infrastructure (zqw): set by add_device when Model::trace
    // is active. The unified dispatch_opcode in kernels/megakernel_dispatch.hip
    // reads these to drive dump_instruction_output. Null base = trace disabled.
    // Field order matches C side (worker_queue.h) — pointers first to avoid
    // internal padding.
    pub dump_base: *mut std::ffi::c_void, // char* in C
    pub dump_count: *mut i32,
    pub dump_capacity: i32,
    pub _pad3: u32,
}

// Static check that WorkerQueueLayout matches the C struct size.
// 4*4 (head) + MAX_BATCH_INSTRUCTIONS*INST_SIZE*8 (inst) + 4*4 (tail) + 8 (op_profile)
//   + 8 (dump_base) + 4 (dump_capacity) + 8 (dump_count) + 4 (_pad3) = +24
const _: () = assert!(
    std::mem::size_of::<WorkerQueueLayout>()
        == 16 + MAX_BATCH_INSTRUCTIONS * INST_SIZE * 8 + 16 + 8 + 24,
    "WorkerQueueLayout size mismatch — verify C struct in kernels/worker_queue.h matches"
);

/// Descriptor for a single VRAM→host async copy issued via `mirror_dump`.
pub struct MirrorRegion {
    /// Source VRAM address on the target GPU.
    pub device_ptr: *const u8,
    /// Number of bytes to copy.
    pub byte_len: usize,
    /// Pinned-host destination (at least `byte_len` bytes).
    pub host_dst: *mut u8,
}
unsafe impl Send for MirrorRegion {}
unsafe impl Sync for MirrorRegion {}

/// Per-GPU worker state.
pub struct GpuWorker {
    pub device: DeviceId,
    pub queue: MappedHostBuffer<u8>, // WorkerQueueLayout, host-mapped
    pub stream: Stream,
    pub module: Module,
    pub seq_counter: u32,
    /// Event recorded on the compute stream after each ack to mark that
    /// all GPU L2 KV writes have been issued. SDMA stream waits on this
    /// event before reading KV from VRAM to prevent stale-L2 coherency bugs
    /// (udi #567 / forensic option 2).
    pub event_kv_written: ffi::hipEvent_t,
}

/// Persistent dispatch context: manages persistent cooperative workers on every
/// GPU running the model. Workers are launched incrementally — the unified-worker
/// design starts workers (GPUs 1..N-1) during prefill, then adds GPU 0 on first
/// decode call. `workers` is indexed by `DeviceId.0` so GPU N's worker is at slot N
/// (or `None` if not yet launched). This keeps the `dispatch_batch_fire(gpu_idx)`
/// API stable across the prefill→decode boundary.
///
/// `workers` slots are wrapped in `ManuallyDrop` so HIP resources (DeviceBuffer →
/// hipFree, MappedHostBuffer → hipHostFree, Stream → hipStreamDestroy, Module →
/// hipModuleUnload) are only freed after the cooperative kernel has confirmed exit
/// via the `done` flag. Auto-drop would free while the kernel is still running.
pub struct PersistentDispatch {
    /// Per-device workers, indexed by DeviceId.0. `None` means no worker on that GPU yet.
    pub workers: Vec<Option<std::mem::ManuallyDrop<GpuWorker>>>,
    /// Host-mapped buffer for GPU 0 expert FFN output (hidden_size f32 elements).
    /// Allocated on GPU 0 (MTYPE_UC). Valid device_ptr only from GPU 0.
    pub moe_output_slot: MappedHostBuffer<f32>,
    /// Host-side watchdog thread polling all persistent kernel WatchdogState pages.
    /// Dropped after workers to ensure the thread stops before the state pages are freed.
    pub watchdog: std::sync::Arc<WatchdogThread>,
    /// Optional per-op profile counter device pointer. Null when
    /// BRAIDINFER_OP_PROFILE build flag is unset (or set but no OpProfile
    /// passed in). Written into each WorkerQueue::op_profile field at
    /// worker launch in `add_device`. See crates/.../op_profile.rs.
    pub op_profile_dev_ptr: *mut u64,
    /// Per-device SDMA streams for out-of-band VRAM→host copies (decode-mirror,
    /// trace dump, etc.). Indexed by `DeviceId.0`. A null entry means no SDMA
    /// stream has been allocated for that GPU. MUST be allocated BEFORE the
    /// corresponding persistent_worker launches on that device — hipStreamCreate
    /// on a GPU running a cooperative kernel can deadlock (sdma_under_coop_fork
    /// probe). Streams are destroyed in `Drop` AFTER the persistent workers
    /// have exited (see drop ordering comments below).
    pub sdma_streams: Vec<ffi::hipStream_t>,
    /// Write-through KV chunk mirror (wt1 P2-c). `None` until
    /// `init_kv_chunk_mirror` is called. One mirror covers all GPUs in the
    /// model — chunks from each GPU are written through in seal order.
    /// `None` in non-debug / production paths to avoid pinned-memory overhead.
    pub kv_chunk_mirror: Option<crate::mirror::KvChunkMirror>,
}

// Raw pointer in PersistentDispatch — caller must keep the underlying
// DeviceBuffer alive until after this dispatch is dropped.
unsafe impl Send for PersistentDispatch {}
unsafe impl Sync for PersistentDispatch {}

fn multiprocessor_count(device: DeviceId) -> HipResult<u32> {
    // hipDeviceAttributeMultiprocessorCount = 63 in the HIP enum (verified via hipcc)
    let mut val = 0i32;
    braidinfer_hip::error::check(unsafe {
        ffi::hipDeviceGetAttribute(&mut val, 63, device.0 as i32)
    })?;
    Ok(val as u32)
}

impl PersistentDispatch {
    fn worker(&self, gpu_idx: usize) -> &GpuWorker {
        self.workers[gpu_idx].as_ref().expect("no persistent worker on this GPU")
    }
    fn worker_mut(&mut self, gpu_idx: usize) -> &mut GpuWorker {
        self.workers[gpu_idx].as_mut().expect("no persistent worker on this GPU")
    }

    fn request_shutdown(&self) {
        for slot in &self.workers {
            if let Some(worker) = slot.as_ref() {
                let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
                unsafe {
                    std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1);
                }
            }
        }
    }

    /// Initialize the dispatcher with `total_gpus` slots and launch workers on
    /// `devices`. Slot index = `DeviceId.0`, so e.g. an init with `total_gpus=4`
    /// and `devices=[GPU1, GPU2, GPU3]` yields workers populated at slots 1-3
    /// and slot 0 empty (GPU 0 added later via `add_device`).
    /// `hidden_size`: for MoE output slot allocation (0 if no MoE).
    pub fn init_with_total(total_gpus: usize, devices: &[DeviceId], shared_mem: u32, _hidden_size: usize, watchdog: std::sync::Arc<WatchdogThread>) -> HipResult<Self> {
        install_signal_handlers_once();
        let mut workers: Vec<Option<std::mem::ManuallyDrop<GpuWorker>>> = (0..total_gpus).map(|_| None).collect();
        let mut dispatch = PersistentDispatch {
            workers: std::mem::take(&mut workers),
            moe_output_slot: MappedHostBuffer::<f32>::alloc(1)?,
            watchdog,
            op_profile_dev_ptr: std::ptr::null_mut(),
            sdma_streams: vec![std::ptr::null_mut(); total_gpus],
            kv_chunk_mirror: None,
        };
        for &device in devices {
            dispatch.add_device(device, shared_mem)?;
        }
        Ok(dispatch)
    }

    /// Set the per-op profile counter device pointer (op_profile.rs). Must be
    /// called BEFORE init_with_total / add_device — the pointer is written
    /// into each WorkerQueue's op_profile field at launch time, and the
    /// kernel reads it via `queue->op_profile`.
    pub fn set_op_profile_ptr(&mut self, ptr: *mut u64) {
        self.op_profile_dev_ptr = ptr;
    }

    /// Lazily allocate a per-device SDMA stream for out-of-band copies
    /// (decode-mirror, trace dump). MUST be called BEFORE the persistent_worker
    /// launches on `device` — afterwards hipStreamCreate on the same GPU can
    /// deadlock against the running cooperative kernel.
    ///
    /// For worker GPUs (1..N-1) this is called internally by `add_device`
    /// immediately before `launch_cooperative`. For GPU 0 (whose persistent
    /// worker is added lazily on first decode call) callers must invoke this
    /// during model init while GPU 0 is still in the kbk-launch phase.
    ///
    /// Idempotent: returns Ok(()) if the stream slot is already populated.
    pub fn ensure_sdma_stream(&mut self, device: DeviceId) -> HipResult<()> {
        let slot = device.0 as usize;
        if slot >= self.sdma_streams.len() {
            self.sdma_streams.resize(slot + 1, std::ptr::null_mut());
        }
        if !self.sdma_streams[slot].is_null() {
            return Ok(());
        }
        // DeviceGuard saves the caller's current device and restores it on
        // drop, so callers iterating over worker GPUs never observe a stale
        // current-device after this call.
        let _guard = DeviceGuard::switch_to(device)?;
        let mut s: ffi::hipStream_t = std::ptr::null_mut();
        braidinfer_hip::error::check(unsafe { ffi::hipStreamCreate(&mut s) })?;
        self.sdma_streams[slot] = s;
        Ok(())
    }

    /// Accessor for the SDMA stream of a given GPU. Returns a raw hipStream_t
    /// (or null if `ensure_sdma_stream` was not called for that device).
    /// Caller must NOT call hipStreamDestroy — ownership remains with this
    /// PersistentDispatch and is released in Drop after worker teardown.
    pub fn sdma_stream(&self, gpu_idx: usize) -> ffi::hipStream_t {
        self.sdma_streams
            .get(gpu_idx)
            .copied()
            .unwrap_or(std::ptr::null_mut())
    }

    /// Enable write-through KV chunk mirroring (wt1 P2-c). Must be called
    /// before the first decode step. `chunk_bytes` is the byte size of one
    /// full chunk (all attention layers K+V, same layout as PageAllocator slots).
    /// Allocates `KvChunkMirror` into `self.kv_chunk_mirror`.
    pub fn init_kv_chunk_mirror(&mut self, chunk_bytes: usize) {
        self.kv_chunk_mirror = Some(crate::mirror::KvChunkMirror::new(chunk_bytes));
    }

    /// Enqueue an SDMA copy of a just-sealed VRAM chunk to pinned host memory.
    /// Call immediately after chunk seal is detected (at `post_step_paged`
    /// boundaries). The copy is async — it completes on `sdma_stream(gpu_idx)`.
    /// A subsequent `drain_kv_chunk_mirror` call synchronizes the stream.
    ///
    /// No-op if `kv_chunk_mirror` is None (mirror not enabled) or if the SDMA
    /// stream for `gpu_idx` is null (stream not yet allocated).
    ///
    /// # Safety
    /// `vram_ptr` must remain valid (chunk slot not freed/reused) until
    /// `drain_kv_chunk_mirror` has been called for this chunk.
    pub fn kv_mirror_chunk(
        &mut self,
        gpu_idx: usize,
        vram_ptr: *const u8,
    ) -> HipResult<()> {
        let stream = self.sdma_stream(gpu_idx);
        if stream.is_null() {
            return Ok(());
        }
        if let Some(mirror) = self.kv_chunk_mirror.as_mut() {
            mirror.enqueue_chunk(vram_ptr, stream)?;
        }
        Ok(())
    }

    /// Synchronize the SDMA stream for `gpu_idx` and record `sealed_chunk_last_pos`
    /// as the sequence position of the last drained chunk in the mirror.
    /// No-op if mirror is disabled or stream is null.
    pub fn drain_kv_chunk_mirror(
        &mut self,
        gpu_idx: usize,
        sealed_chunk_last_pos: u32,
    ) -> HipResult<()> {
        let stream = self.sdma_stream(gpu_idx);
        if stream.is_null() {
            return Ok(());
        }
        if let Some(mirror) = self.kv_chunk_mirror.as_mut() {
            mirror.drain(sealed_chunk_last_pos, stream)?;
        }
        Ok(())
    }

    /// Single-GPU init helper (unchanged signature for non-MoE callers).
    /// Allocates a single-slot dispatcher launched on `devices[0]`.
    pub fn init(devices: &[DeviceId], shared_mem: u32, hidden_size: usize, watchdog: std::sync::Arc<WatchdogThread>) -> HipResult<Self> {
        // Old signature: assume `devices` is a single-GPU list and allocate
        // exactly that many slots. Callers that need slot-by-DeviceId.0 layout
        // should call `init_with_total` directly.
        let total = devices.iter().map(|d| d.0 as usize + 1).max().unwrap_or(1);
        Self::init_with_total(total, devices, shared_mem, hidden_size, watchdog)
    }

    /// Append a GPU to the persistent worker pool. Used by the unified-worker
    /// design so workers (GPUs 1..N-1) can launch during prefill (when GPU 0
    /// is still running kbk kernels) and GPU 0 is added on first decode call.
    pub fn add_device(&mut self, device: DeviceId, shared_mem: u32) -> HipResult<()> {
        // wt1 P2-a: allocate SDMA stream BEFORE the persistent_worker
        // cooperative kernel launches on this device. hipStreamCreate after
        // launch can deadlock against the running coop kernel.
        self.ensure_sdma_stream(device)?;
        let kernel_dir = crate::kernel::kernel_dir();
        let queue_size = std::mem::size_of::<WorkerQueueLayout>();
        Device::set_current(device)?;
        let queue = MappedHostBuffer::<u8>::alloc(queue_size)?;
        // Write the op_profile counter pointer into the queue BEFORE launch.
        // Per-instance pointer (set_op_profile_ptr) takes priority; falls
        // back to the process-global (op_profile::install_global) if unset.
        // Null disables profiling on this worker. See PLAN-op-profile.md.
        // Also zero the trace-dump fields (zqw) — populated by Model when
        // trace is active via set_trace_dump_ptrs(...). Until then the
        // unified dispatch_opcode treats null dump_base as "trace disabled".
        unsafe {
            let profile_ptr = if !self.op_profile_dev_ptr.is_null() {
                self.op_profile_dev_ptr
            } else {
                crate::op_profile::get_global()
            };
            let q = queue.host_ptr() as *mut WorkerQueueLayout;
            std::ptr::addr_of_mut!((*q).op_profile).write(profile_ptr);
            std::ptr::addr_of_mut!((*q).dump_base).write(std::ptr::null_mut());
            std::ptr::addr_of_mut!((*q).dump_count).write(std::ptr::null_mut());
            std::ptr::addr_of_mut!((*q).dump_capacity).write(0);
        }
        let stream = Stream::new(device)?;
        // Allocate per-worker event for L2-coherency fence (udi #567).
        // Created here (before cooperative launch) — hipEventCreate after a
        // cooperative kernel is running can deadlock on some ROCm versions.
        let mut event_kv_written: ffi::hipEvent_t = std::ptr::null_mut();
        braidinfer_hip::error::check(unsafe {
            ffi::hipEventCreate(&mut event_kv_written)
        })?;
        // zqw merge: persistent_worker entry now lives in megakernel.hsaco
        // alongside megakernel_f32. One module load, get the right function.
        let module = Module::load(device, &kernel_dir.join("megakernel.hsaco"))?;
        let func = module.get_function("persistent_worker")?;
        let mut queue_ptr = queue.device_ptr() as *mut std::ffi::c_void;
        let wd_state_dev = self.watchdog.register(device)?;
        let mut wd_ptr: *mut std::ffi::c_void = wd_state_dev as *mut std::ffi::c_void;
        let mut args: [*mut std::ffi::c_void; 2] = [
            std::ptr::addr_of_mut!(queue_ptr).cast(),
            std::ptr::addr_of_mut!(wd_ptr).cast(),
        ];
        let bpsm_raw = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        let bpsm_max = bpsm_raw.min(2);
        let bpsm = bpsm_max;
        let num_cus = multiprocessor_count(device)?;
        let num_blocks = (bpsm as u32 * num_cus).max(num_cus);
        func.launch_cooperative(
            (num_blocks, 1, 1),
            (256, 1, 1),
            shared_mem,
            &stream,
            &mut args,
        )?;
        eprintln!(
            "  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B shared, bpsm_raw={bpsm_raw} bpsm={bpsm} num_cus={num_cus})",
            device.0
        );
        braidinfer_hip::set_persistent_worker_active(device, true);
        let slot = device.0 as usize;
        if slot >= self.workers.len() {
            self.workers.resize_with(slot + 1, || None);
        }
        assert!(self.workers[slot].is_none(), "double-init persistent worker on GPU {}", slot);
        self.workers[slot] = Some(std::mem::ManuallyDrop::new(GpuWorker {
            device,
            queue,
            stream,
            module,
            seq_counter: 0,
            event_kv_written,
        }));
        Ok(())
    }

    /// True if persistent worker on this GPU has been launched.
    pub fn has_worker(&self, gpu_idx: usize) -> bool {
        gpu_idx < self.workers.len() && self.workers[gpu_idx].is_some()
    }

    /// Wait for a GPU to ack a specific seq number, returning a recoverable
    /// error on shutdown / timeout instead of panicking. This is the new
    /// daemon-friendly path (braidinfer-wks Phase 1). For the in-process
    /// binary that wants the legacy panic-on-shutdown semantics, use
    /// [`Self::wait_ack`] (a thin wrapper that unwraps).
    pub(crate) fn try_wait_ack(&self, gpu_idx: usize, seq: u32) -> Result<(), DispatchError> {
        let q_ptr = self.worker(gpu_idx).queue.host_ptr() as *const WorkerQueueLayout;
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq {
                return Ok(());
            }
            if shutdown_requested() {
                return Err(DispatchError::ShutdownRequested { gpu: gpu_idx, seq });
            }
            if start.elapsed().as_secs() > 30 {
                let progress_pc = unsafe {
                    std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).progress_pc))
                };
                let block_alive_count = unsafe {
                    std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).block_alive_count))
                };
                return Err(DispatchError::Timeout {
                    gpu: gpu_idx,
                    seq,
                    ack,
                    progress_pc,
                    block_alive_count,
                });
            }
            std::hint::spin_loop();
        }
    }

    /// Wait for multiple (gpu, seq) targets to ack in a single shared poll
    /// loop. Round-robins across all targets, returning when every one has
    /// acked. This is the foundation of the single-thread multi-GPU
    /// dispatcher (PLAN-dispatch-daemon.md Phase 1) — one polling thread
    /// can service all 8 GPUs concurrently because each iteration costs
    /// ~100 ns × N targets, vs ~10 ms of GPU compute per dispatch.
    ///
    /// Memory ordering: the `ack` fields live in independent host-DRAM
    /// cache lines (one per GPU's host-mapped queue). Round-robin reads
    /// across them require no fences — each line's value is independent
    /// state.
    ///
    /// Unlike sequential `try_wait_ack` per target, this also reports
    /// per-iteration polling overhead when `DISPATCH_RTT=1` is set
    /// (success-criterion measurement for Phase 1).
    pub(crate) fn try_wait_acks_many(
        &self,
        targets: &[(usize, u32)],
    ) -> Result<(), DispatchError> {
        if targets.is_empty() {
            return Ok(());
        }
        // Cache the queue pointers up front; one indirection per iteration
        // is wasteful when the inner loop is hot.
        let q_ptrs: Vec<*const WorkerQueueLayout> = targets
            .iter()
            .map(|(g, _)| self.worker(*g).queue.host_ptr() as *const WorkerQueueLayout)
            .collect();
        let mut done = vec![false; targets.len()];
        let mut remaining = targets.len();
        let start = std::time::Instant::now();

        let report_overhead = std::env::var("DISPATCH_RTT").is_ok();
        let mut iter_count: u64 = 0;

        while remaining > 0 {
            for i in 0..targets.len() {
                if done[i] {
                    continue;
                }
                let (_gpu, seq) = targets[i];
                let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptrs[i]).ack)) };
                if ack == seq {
                    done[i] = true;
                    remaining -= 1;
                }
            }
            if remaining == 0 {
                break;
            }
            if shutdown_requested() {
                // Find the first not-yet-acked target and report it.
                for i in 0..targets.len() {
                    if !done[i] {
                        let (gpu, seq) = targets[i];
                        return Err(DispatchError::ShutdownRequested { gpu, seq });
                    }
                }
                unreachable!("remaining > 0 but all done");
            }
            if start.elapsed().as_secs() > 30 {
                for i in 0..targets.len() {
                    if !done[i] {
                        let (gpu, seq) = targets[i];
                        let ack = unsafe {
                            std::ptr::read_volatile(std::ptr::addr_of!((*q_ptrs[i]).ack))
                        };
                        let progress_pc = unsafe {
                            std::ptr::read_volatile(std::ptr::addr_of!((*q_ptrs[i]).progress_pc))
                        };
                        let block_alive_count = unsafe {
                            std::ptr::read_volatile(std::ptr::addr_of!((*q_ptrs[i]).block_alive_count))
                        };
                        return Err(DispatchError::Timeout {
                            gpu,
                            seq,
                            ack,
                            progress_pc,
                            block_alive_count,
                        });
                    }
                }
                unreachable!("remaining > 0 but all done");
            }
            iter_count += 1;
            std::hint::spin_loop();
        }
        if report_overhead {
            let elapsed_us = start.elapsed().as_micros() as u64;
            let per_iter_ns = if iter_count > 0 {
                (elapsed_us * 1000) / iter_count
            } else {
                0
            };
            eprintln!(
                "wait_acks_many n_targets={} iters={iter_count} total={elapsed_us}us per_iter={per_iter_ns}ns",
                targets.len()
            );
        }
        Ok(())
    }

    /// Legacy wait-ack that panics on shutdown / timeout to preserve the
    /// existing in-process binary semantics (Drop runs during unwind, the
    /// panic message is already part of the user-visible behavior). Daemon
    /// code paths must use [`Self::try_wait_ack`] / [`Self::try_wait_acks_many`]
    /// directly so they can convert errors to per-session RPC failures.
    pub(crate) fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        if let Err(e) = self.try_wait_ack(gpu_idx, seq) {
            panic!("{e}");
        }
    }

    /// Dispatch a batch of instructions to a GPU. Worker executes all with grid.sync()
    /// between them, acks once at the end. One signal round-trip per batch.
    pub(crate) fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        assert!(instructions.len() <= MAX_BATCH_INSTRUCTIONS);
        let w = self.worker_mut(gpu_idx);
        let q_ptr = w.queue.host_ptr() as *mut WorkerQueueLayout;

        // Copy all instructions to the batch buffer
        for (i, inst) in instructions.iter().enumerate() {
            let offset = i * INST_SIZE;
            for j in 0..INST_SIZE {
                unsafe {
                    std::ptr::write_volatile(
                        std::ptr::addr_of_mut!((*q_ptr).inst[offset + j]),
                        inst.words[j],
                    );
                }
            }
        }

        // Write num_instructions BEFORE seq_num (worker reads num_instructions after seeing seq_num)
        unsafe {
            std::ptr::write_volatile(
                std::ptr::addr_of_mut!((*q_ptr).num_instructions),
                instructions.len() as u32,
            );
        }

        // Trigger worker
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe {
            std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq);
        }

        // Wait for ack via the shared try_wait_ack path, panicking to
        // preserve legacy in-process semantics. Daemon should use
        // dispatch_batch_fire + try_wait_acks_many directly.
        let start = std::time::Instant::now();
        if let Err(e) = self.try_wait_ack(gpu_idx, seq) {
            // Augment timeout messages with the original instruction-level
            // diagnostics that the old inline path produced.
            if let DispatchError::Timeout {
                progress_pc, ack, block_alive_count, ..
            } = e
            {
                let opcode0 = instructions[0].words[0] as u32;
                let stuck_op = instructions
                    .get(progress_pc as usize)
                    .map(|i| i.words[0] as u32 as u64)
                    .unwrap_or(0);
                let stuck_grid_x = instructions
                    .get(progress_pc as usize)
                    .map(|i| (i.words[0] >> 32) as u32)
                    .unwrap_or(0);
                panic!(
                    "dispatch_batch timeout gpu={gpu_idx} seq={seq} n={} ack={ack} \
                     opcode0={opcode0} stuck_pc={progress_pc} stuck_op={stuck_op} \
                     stuck_grid_x={stuck_grid_x} \
                     probe_s_full=0x{block_alive_count:08x}",
                    instructions.len()
                );
            }
            panic!("{e}");
        }
        if std::env::var("DISPATCH_RTT").is_ok() {
            let us = start.elapsed().as_micros();
            let op0 = instructions[0].words[0] as u32;
            eprintln!(
                "dispatch_batch gpu={gpu_idx} n={} op0={op0:#x} rtt={us}us",
                instructions.len()
            );
        }
    }

    /// Dispatch a slice of instructions to a GPU, sending in chunks of MAX_BATCH_INSTRUCTIONS.
    /// Avoids the caller needing to collect into an owned Vec first.
    pub(crate) fn dispatch_batch_slice(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        for chunk in instructions.chunks(MAX_BATCH_INSTRUCTIONS) {
            self.dispatch_batch(gpu_idx, chunk);
        }
    }

    /// Fire a batch of instructions to a GPU WITHOUT waiting for ack. Returns seq for wait_ack.
    /// Caller must call wait_ack(gpu_idx, seq) before reading GPU 0 output.
    /// Used to overlap GPU 0 OP_EXPERT_FFN with kbk dispatch on GPUs 1+.
    pub(crate) fn dispatch_batch_fire(
        &mut self,
        gpu_idx: usize,
        instructions: &[Instruction],
    ) -> u32 {
        assert!(instructions.len() <= MAX_BATCH_INSTRUCTIONS);
        let w = self.worker_mut(gpu_idx);
        let q_ptr = w.queue.host_ptr() as *mut WorkerQueueLayout;
        for (i, inst) in instructions.iter().enumerate() {
            let offset = i * INST_SIZE;
            for j in 0..INST_SIZE {
                unsafe {
                    std::ptr::write_volatile(
                        std::ptr::addr_of_mut!((*q_ptr).inst[offset + j]),
                        inst.words[j],
                    );
                }
            }
        }
        // Write num_instructions BEFORE seq_num (worker reads num_instructions after seeing seq_num)
        unsafe {
            std::ptr::write_volatile(
                std::ptr::addr_of_mut!((*q_ptr).num_instructions),
                instructions.len() as u32,
            );
        }
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe {
            std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq);
        }
        seq
    }

    /// Request worker shutdown via host-mapped flags only.
    ///
    /// This intentionally does not call any HIP APIs. Cooperative kernels must
    /// exit and signal `done` before stream or memory cleanup becomes safe.
    pub fn shutdown(&mut self) {
        self.request_shutdown();
    }

    /// Queue a single VRAM→host async copy on the SDMA stream for `gpu_idx`.
    ///
    /// Caller MUST call `Device::set_current(device)` for `gpu_idx` before
    /// this call. The copy is issued asynchronously; call `mirror_sync` to
    /// wait for completion before reading `region.host_dst`.
    ///
    /// Returns `Err` if `sdma_stream` for `gpu_idx` is null (caller forgot
    /// `ensure_sdma_stream`) or if `hipMemcpyAsync` fails.
    pub fn mirror_dump(&self, gpu_idx: usize, region: &MirrorRegion) -> HipResult<()> {
        let stream = self.sdma_stream(gpu_idx);
        if stream.is_null() {
            return Err(braidinfer_hip::error::HipError(ffi::hipErrorInvalidValue));
        }
        braidinfer_hip::error::check(unsafe {
            ffi::hipMemcpyAsync(
                region.host_dst as *mut std::ffi::c_void,
                region.device_ptr as *const std::ffi::c_void,
                region.byte_len,
                ffi::hipMemcpyDeviceToHost,
                stream,
            )
        })
    }

    /// Wait for all in-flight copies on the SDMA stream for `gpu_idx` to land.
    ///
    /// Equivalent to `hipStreamSynchronize(sdma_streams[gpu_idx])`. Returns
    /// `Err` if the stream is null or synchronize fails.
    pub fn mirror_sync(&self, gpu_idx: usize) -> HipResult<()> {
        let stream = self.sdma_stream(gpu_idx);
        if stream.is_null() {
            return Err(braidinfer_hip::error::HipError(ffi::hipErrorInvalidValue));
        }
        braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(stream) })
    }

    /// Record `event_kv_written` on the NULL stream for worker `gpu_idx`.
    ///
    /// IMPORTANT: Must NOT record on `worker.stream` (the cooperative kernel's
    /// launch stream) — that stream is permanently busy (persistent_worker runs
    /// forever), so hipEventRecord on it would never fire, deadlocking any
    /// hipStreamWaitEvent caller.
    ///
    /// Recording on NULL stream provides an ordering fence relative to all GPU
    /// memory operations that precede it (NULL stream synchronizes with all
    /// non-blocking streams on ROCm). Call this AFTER ack is received from the
    /// worker to create a happens-before edge between the worker's L2 writes
    /// and the SDMA copy (udi #567 / forensic option 2).
    ///
    /// No-op if `gpu_idx` has no worker slot.
    pub fn record_kv_event(&self, gpu_idx: usize) -> HipResult<()> {
        let Some(Some(worker)) = self.workers.get(gpu_idx) else { return Ok(()); };
        // Record on NULL stream (not worker.stream): the cooperative kernel's
        // launch stream is permanently busy, so recording there would stall the
        // event forever. NULL stream on ROCm synchronizes with the default
        // per-device context and fires promptly.
        braidinfer_hip::error::check(unsafe {
            ffi::hipEventRecord(worker.event_kv_written, std::ptr::null_mut())
        })
    }

    /// Make the SDMA stream of `gpu_idx` wait for `event_kv_written` on that
    /// worker's compute stream before proceeding. This creates the
    /// cross-stream ordering point that ensures L2 stores land in VRAM before
    /// SDMA reads (udi #567 / forensic option 2).
    ///
    /// No-op if the SDMA stream for `gpu_idx` is null.
    pub fn wait_kv_event_on_sdma(&self, gpu_idx: usize) -> HipResult<()> {
        let sdma = self.sdma_stream(gpu_idx);
        if sdma.is_null() { return Ok(()); }
        let Some(Some(worker)) = self.workers.get(gpu_idx) else { return Ok(()); };
        braidinfer_hip::error::check(unsafe {
            ffi::hipStreamWaitEvent(sdma, worker.event_kv_written, 0)
        })
    }


    /// braidinfer-4fg.4: in-band shutdown via OP_HALT instruction batches.
    /// Dispatches a single-instruction OP_HALT batch to each worker queue.
    /// The kernel processes OP_HALT in its normal pc loop and exits naturally
    /// — same code path as every other op dispatch, which is proven to work
    /// during decode. Replaces the OUT-OF-BAND queue->shutdown flag path
    /// that wedges intermittently at the post-poll atomic_block_barrier.
    pub fn send_halt_all(&mut self) {
        // 4fg.5: switch from OP_HALT-instruction dispatch to the legacy
        // queue->shutdown=1 path. The worker's inner-poll detects shutdown
        // first, sets g_shutdown_seen via AGENT-scope (no PCIe), post-poll
        // barrier delivers it, and ALL blocks early-return — skipping
        // watchdog_poll_and_check's internal atomic_block_barrier. The
        // OP_HALT path required reading queue->inst and going through
        // back-to-back barriers (post-poll + watchdog), which kb 4fg-3
        // identified as wedge-prone.
        for slot in self.workers.iter_mut() {
            let Some(worker) = slot.as_mut() else { continue };
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe {
                std::ptr::write_volatile(
                    std::ptr::addr_of_mut!((*q_ptr).shutdown),
                    1u32,
                );
            }
        }
    }

}

impl BatchDispatcher for PersistentDispatch {
    fn dispatch_batch_fire(&mut self, gpu_idx: usize, instructions: &[Instruction]) -> u32 {
        PersistentDispatch::dispatch_batch_fire(self, gpu_idx, instructions)
    }
    fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        PersistentDispatch::dispatch_batch(self, gpu_idx, instructions)
    }
    fn dispatch_batch_slice(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        PersistentDispatch::dispatch_batch_slice(self, gpu_idx, instructions)
    }
    fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        PersistentDispatch::wait_ack(self, gpu_idx, seq)
    }
    fn try_wait_acks_many(&self, targets: &[(usize, u32)]) -> Result<(), DispatchError> {
        PersistentDispatch::try_wait_acks_many(self, targets)
    }
}

impl Drop for PersistentDispatch {
    /// Shutdown all workers and wait for them to exit via kernel-written `done` flag.
    ///
    /// INVARIANT: No HIP API calls (hipFree, hipMalloc, hipMemcpy, hipStreamSynchronize,
    /// hipStreamQuery) after cooperative kernels are launched. These all trigger
    /// SyncAllStreams → releaseGpuMemoryFence which deadlocks with a running cooperative kernel.
    ///
    /// The ONLY way to communicate after launch is via host-mapped memory (read_volatile /
    /// write_volatile on host_ptr). The kernel writes `done=1` before returning so we can
    /// poll without any HIP API.
    fn drop(&mut self) {
        // Two-phase shutdown to handle multi-GPU MoE q8 (and similar slow-batch
        // workloads) where the worker may not see `shutdown` for several hundred
        // ms because it's mid-batch. The previous single-phase 5s deadline was
        // (a) absolute (cascading leaks if any one worker took >5s) and (b) too
        // short for q8 4-GPU MoE.
        //
        // Phase 1 (1s clean / 200ms panic): cooperative shutdown via the
        // shutdown flag. Worker checks at outer-loop boundaries, so this is
        // sufficient when workers are between batches.
        //
        // Phase 2 (up to total_timeout): if any worker hasn't acked by the
        // phase-1 deadline, fire watchdog `force_exit` on every registered
        // WatchdogState. Workers honor force_exit at their next
        // watchdog_poll_and_check call (at every instruction boundary inside
        // every kernel that includes watchdog.h), which is much more frequent
        // than the outer-loop shutdown check. Continue polling until the total
        // timeout. After that, leak (don't free) to avoid hipFree deadlocking
        // against a still-running cooperative kernel.
        //
        // All polling is PARALLEL across workers (single shared spin loop) so
        // one slow worker can't cascade-starve the others' deadline.
        let panicking = std::thread::panicking();
        let phase1_timeout = if panicking {
            std::time::Duration::from_millis(200)
        } else {
            std::time::Duration::from_secs(1)
        };
        let total_timeout = if panicking {
            std::time::Duration::from_secs(2)
        } else {
            std::time::Duration::from_secs(30)
        };

        // braidinfer-4fg.4: in-band shutdown via OP_HALT instead of
        // out-of-band queue->shutdown flag. The kernel processes OP_HALT in
        // its normal pc loop and exits naturally — same code path as every
        // other op dispatch, which is proven to work during decode. The
        // previous shutdown=1 flag path hit a wedge at the first
        // atomic_block_barrier after the inner-poll exit. Crucially we do
        // NOT call request_shutdown here — that triggers the wedge-prone
        // path. OP_HALT alone is sufficient.
        self.send_halt_all();
        let start = std::time::Instant::now();
        let phase1_deadline = start + phase1_timeout;
        let total_deadline = start + total_timeout;

        let mut worker_done = vec![false; self.workers.len()];
        // Empty slots count as already-done.
        for (idx, slot) in self.workers.iter().enumerate() {
            if slot.is_none() {
                worker_done[idx] = true;
            }
        }

        let poll_round = |worker_done: &mut [bool], workers: &[Option<std::mem::ManuallyDrop<GpuWorker>>]| -> bool {
            let mut all = true;
            for (idx, slot) in workers.iter().enumerate() {
                if worker_done[idx] {
                    continue;
                }
                if let Some(worker) = slot.as_ref() {
                    let q_ptr = worker.queue.host_ptr() as *const WorkerQueueLayout;
                    let done = unsafe {
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).done))
                    };
                    if done != 0 {
                        worker_done[idx] = true;
                    } else {
                        all = false;
                    }
                } else {
                    worker_done[idx] = true;
                }
            }
            all
        };

        // Phase 1: cooperative shutdown polling.
        while !poll_round(&mut worker_done, &self.workers)
            && std::time::Instant::now() < phase1_deadline
        {
            std::hint::spin_loop();
        }

        // Phase 2: if any worker hasn't acked, escalate to watchdog force_exit.
        if !worker_done.iter().all(|&d| d) {
            let stuck: Vec<u32> = self
                .workers
                .iter()
                .enumerate()
                .filter(|(idx, _)| !worker_done[*idx])
                .filter_map(|(_, slot)| slot.as_ref().map(|w| w.device.0))
                .collect();
            eprintln!(
                "braidinfer: shutdown polling slow on GPUs {:?} after {:?}; firing watchdog force_exit fallback (total deadline {:?})",
                stuck, phase1_timeout, total_timeout
            );
            self.watchdog.force_exit_all();

            while !poll_round(&mut worker_done, &self.workers)
                && std::time::Instant::now() < total_deadline
            {
                std::hint::spin_loop();
            }
        }

        // Log any workers that still didn't exit (will be leaked).
        for (idx, slot) in self.workers.iter().enumerate() {
            if !worker_done[idx] {
                if let Some(worker) = slot.as_ref() {
                    let q_ptr = worker.queue.host_ptr() as *const WorkerQueueLayout;
                    let (seq, ack, progress_pc, num_inst, shutdown, done) = unsafe {(
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).seq_num)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).progress_pc)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).num_instructions)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).shutdown)),
                        std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).done)),
                    )};
                    eprintln!(
                        "braidinfer: persistent worker shutdown timeout on GPU {} (leaking) {} \
                         [seq={seq} ack={ack} progress_pc={progress_pc} num_inst={num_inst} \
                         shutdown={shutdown} done={done}]",
                        worker.device.0,
                        if panicking { "[panic]" } else { "" }
                    );
                }
            }
        }

        // On panic, terminate the process AFTER the best-effort shutdown above. This
        // matches the original behavior (panic = abort) but ensures we attempted the
        // shutdown signal first so the kernel has a chance to release the GPU.
        if panicking {
            std::process::exit(1);
        }

        // braidinfer-4fg.3: for multi-GPU, abort BEFORE Stream/Module/Buffer
        // drops. Even though the kernel reorder (sentinel check before
        // watchdog_poll_and_check) eliminates the common back-to-back-barrier
        // wedge, the outer barrier (line 148 of megakernel.hip) can still
        // wedge intermittently (~1/5 runs observed). When that happens,
        // hipStreamDestroy blocks waiting for the kernel to release CUs,
        // libc::exit never completes. We can't reliably detect the wedge
        // from host. Multi-GPU = always fast-exit via _exit. The OS reclaims
        // memory and amdgpu force-releases GPUs.
        // braidinfer-4fg.3: for multi-GPU, abort BEFORE Stream/Module/Buffer
        // drops. The kernel-internal atomic_block_barrier wedge can leave the
        // cooperative kernel running even after the worker thread set done=1
        // (kb rdna3-atomic-block-barrier-multi-gpu-fundamental-issue). When
        // that happens, hipStreamDestroy blocks waiting for the kernel to
        // release CUs. We can't reliably detect the wedge from host. Multi-GPU
        // = always fast-exit via _exit. Verified bounded ~4s exit (vs 600s+
        // hang). Investigation in braidinfer-4fg.3/awj/setprio didn't isolate
        // the root cause — possibly HW-level scheduler/cache interaction
        // under cross-GPU PCIe pressure that's not addressable in software.
        // braidinfer-4fg.4: multi-GPU always fast-aborts. The host's done=1
        // signal turns out to be NECESSARY but NOT SUFFICIENT — observed
        // 20% of runs where worker writes done=1 but the cooperative
        // kernel still isn't fully terminated (some blocks lingering),
        // and hipStreamDestroy blocks waiting for the kernel. Without a
        // reliable kernel-termination signal, the only safe choice is to
        // skip the Stream/Module cleanup entirely on multi-GPU and let the
        // OS reclaim on process exit. OP_HALT still fires (cleaner GPU
        // release when the kernel actually does terminate cleanly) but
        // we don't depend on it for bounded exit.
        if self.workers.len() > 1 {
            self.watchdog.force_exit_all();
            std::thread::sleep(std::time::Duration::from_millis(200));
            unsafe { libc::_exit(134); }
        }

        // Single-GPU clean path: drop workers normally. No multi-GPU wedge
        // observed on single-GPU per kb.
        for (idx, slot) in self.workers.iter_mut().enumerate() {
            if worker_done[idx] {
                if let Some(mut worker) = slot.take() {
                    let device = worker.device;
                    braidinfer_hip::set_persistent_worker_active(device, false);
                    unsafe {
                        std::mem::ManuallyDrop::drop(&mut worker);
                    }
                }
            }
        }

        // wt1 P2-a: destroy SDMA streams AFTER the persistent workers have
        // exited (single-GPU clean path). On the multi-GPU `_exit(134)` path
        // above we skip this — the OS reclaims the streams as part of process
        // teardown, matching the existing Stream/Module skip rationale.
        for s in self.sdma_streams.iter_mut() {
            if !s.is_null() {
                unsafe { let _ = ffi::hipStreamDestroy(*s); }
                *s = std::ptr::null_mut();
            }
        }

        // Single-GPU leak (rare): fast-exit.
        if braidinfer_hip::any_persistent_worker_active() {
            eprintln!(
                "braidinfer: single-GPU worker leaked; fast-exiting via _exit(134)."
            );
            self.watchdog.force_exit_all();
            std::thread::sleep(std::time::Duration::from_millis(200));
            unsafe { libc::_exit(134); }
        }
    }
}
