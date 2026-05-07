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
use braidinfer_hip::device::Device;
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

/// Rust mirror of WorkerQueue from persistent_worker.hip.
/// Layout must match exactly (repr(C)).
#[repr(C)]
pub struct WorkerQueueLayout {
    pub seq_num: u32,
    pub shutdown: u32,
    pub num_instructions: u32, // how many instructions in this batch (1..MAX_BATCH)
    pub _pad: u32,
    pub inst: [u64; MAX_BATCH_INSTRUCTIONS * INST_SIZE], // instruction batch buffer
    pub ack: u32,
    pub done: u32, // kernel writes 1 when exiting after shutdown (for Drop polling)
    pub progress_pc: u32, // kernel writes pc before each instruction (for timeout diagnosis)
    pub _pad2: u32,
}

/// Per-GPU worker state.
pub struct GpuWorker {
    pub device: DeviceId,
    pub queue: MappedHostBuffer<u8>, // WorkerQueueLayout, host-mapped
    pub stream: Stream,
    pub module: Module,
    pub seq_counter: u32,
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
    pub watchdog: WatchdogThread,
}

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
    pub fn init_with_total(total_gpus: usize, devices: &[DeviceId], shared_mem: u32, _hidden_size: usize) -> HipResult<Self> {
        install_signal_handlers_once();
        let watchdog = WatchdogThread::spawn();
        let mut workers: Vec<Option<std::mem::ManuallyDrop<GpuWorker>>> = (0..total_gpus).map(|_| None).collect();
        let mut dispatch = PersistentDispatch {
            workers: std::mem::take(&mut workers),
            moe_output_slot: MappedHostBuffer::<f32>::alloc(1)?,
            watchdog,
        };
        for &device in devices {
            dispatch.add_device(device, shared_mem)?;
        }
        Ok(dispatch)
    }

    /// Single-GPU init helper (unchanged signature for non-MoE callers).
    /// Allocates a single-slot dispatcher launched on `devices[0]`.
    pub fn init(devices: &[DeviceId], shared_mem: u32, hidden_size: usize) -> HipResult<Self> {
        // Old signature: assume `devices` is a single-GPU list and allocate
        // exactly that many slots. Callers that need slot-by-DeviceId.0 layout
        // should call `init_with_total` directly.
        let total = devices.iter().map(|d| d.0 as usize + 1).max().unwrap_or(1);
        Self::init_with_total(total, devices, shared_mem, hidden_size)
    }

    /// Append a GPU to the persistent worker pool. Used by the unified-worker
    /// design so workers (GPUs 1..N-1) can launch during prefill (when GPU 0
    /// is still running kbk kernels) and GPU 0 is added on first decode call.
    pub fn add_device(&mut self, device: DeviceId, shared_mem: u32) -> HipResult<()> {
        let kernel_dir = crate::kernel::kernel_dir();
        let queue_size = std::mem::size_of::<WorkerQueueLayout>();
        Device::set_current(device)?;
        let queue = MappedHostBuffer::<u8>::alloc(queue_size)?;
        let stream = Stream::new(device)?;
        let module = Module::load(device, &kernel_dir.join("persistent_worker.hsaco"))?;
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
        let bpsm = std::env::var("BRAIDINFER_BPSM")
            .ok().and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(1, bpsm_max as u32) as usize)
            .unwrap_or(bpsm_max as usize);
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
        braidinfer_hip::set_persistent_worker_active(true);
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
        }));
        Ok(())
    }

    /// True if persistent worker on this GPU has been launched.
    pub fn has_worker(&self, gpu_idx: usize) -> bool {
        gpu_idx < self.workers.len() && self.workers[gpu_idx].is_some()
    }

    /// Wait for a GPU to ack a specific seq number.
    pub(crate) fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        let q_ptr = self.worker(gpu_idx).queue.host_ptr() as *const WorkerQueueLayout;
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq {
                break;
            }
            if shutdown_requested() {
                panic!("wait_ack interrupted: SIGINT/SIGTERM (gpu={gpu_idx}, seq={seq})");
            }
            if start.elapsed().as_secs() > 30 {
                let progress_pc = unsafe {
                    std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).progress_pc))
                };
                panic!(
                    "wait_ack timeout gpu={gpu_idx} seq={seq} ack={ack} progress_pc={progress_pc}"
                );
            }
            std::hint::spin_loop();
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

        // Wait for ack
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq {
                break;
            }
            if shutdown_requested() {
                panic!(
                    "dispatch_batch interrupted: SIGINT/SIGTERM (gpu={gpu_idx}, seq={seq})"
                );
            }
            if start.elapsed().as_secs() > 30 {
                let opcode0 = instructions[0].words[0] as u32;
                let progress_pc = unsafe {
                    std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).progress_pc))
                };
                let stuck_op = instructions
                    .get(progress_pc as usize)
                    .map(|i| i.words[0] as u32 as u64)
                    .unwrap_or(0);
                let stuck_grid_x = instructions
                    .get(progress_pc as usize)
                    .map(|i| (i.words[0] >> 32) as u32)
                    .unwrap_or(0);
                panic!(
                    "dispatch_batch timeout gpu={gpu_idx} seq={seq} n={} opcode0={opcode0} \
                     stuck_pc={progress_pc} stuck_op={stuck_op} stuck_grid_x={stuck_grid_x}",
                    instructions.len()
                );
            }
            std::hint::spin_loop();
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

    /// Number of GPUs with launched workers.
    pub fn num_gpus(&self) -> usize {
        self.workers.iter().filter(|s| s.is_some()).count()
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
        // ALWAYS request shutdown — even on panic. The kernel polls the shutdown flag at
        // the top of its outer instruction loop. If at least one block reaches the poll
        // (and writes seq_num=0xFFFFFFFFu), surviving blocks see the sentinel and exit
        // via the next grid.sync. If blocks are split between "done" and "stuck inside
        // an instruction", grid.sync deadlocks — that's the case we time out on.
        //
        // Use a short timeout on panic (we're already aborting; the user is waiting) and
        // a longer one on clean exit. Either way, leak (don't free) on timeout to avoid
        // hipFree deadlocks against a still-running kernel.
        let panicking = std::thread::panicking();
        let shutdown_timeout = if panicking {
            std::time::Duration::from_secs(2)
        } else {
            std::time::Duration::from_secs(5)
        };

        self.request_shutdown();
        let shutdown_deadline = std::time::Instant::now() + shutdown_timeout;
        let mut worker_done = vec![false; self.workers.len()];
        for (idx, slot) in self.workers.iter().enumerate() {
            let Some(worker) = slot.as_ref() else { worker_done[idx] = true; continue; };
            let q_ptr = worker.queue.host_ptr() as *const WorkerQueueLayout;
            loop {
                let done = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).done)) };
                if done != 0 {
                    worker_done[idx] = true;
                    break;
                }
                if std::time::Instant::now() > shutdown_deadline {
                    eprintln!(
                        "braidinfer: persistent worker shutdown timeout on GPU {} (leaking) {}",
                        worker.device.0,
                        if panicking { "[panic]" } else { "" }
                    );
                    break;
                }
                std::hint::spin_loop();
            }
        }
        // Free HIP resources only for workers that confirmed exit. Timed-out
        // workers are intentionally leaked to avoid deadlocking on HIP cleanup.
        for (idx, slot) in self.workers.iter_mut().enumerate() {
            if worker_done[idx] {
                if let Some(mut worker) = slot.take() {
                    unsafe {
                        std::mem::ManuallyDrop::drop(&mut worker);
                    }
                }
            }
        }
        braidinfer_hip::set_persistent_worker_active(false);

        // On panic, terminate the process AFTER the best-effort shutdown above. This
        // matches the original behavior (panic = abort) but ensures we attempted the
        // shutdown signal first so the kernel has a chance to release the GPU.
        if panicking {
            std::process::exit(1);
        }
    }
}
