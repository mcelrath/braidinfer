//! Per-batch cooperative dispatch (braidinfer-pky Phase 0b).
//!
//! Each GPU runs a one-shot `megakernel_f32` cooperative kernel per
//! sub-batch instead of a persistent worker polling a host-mapped queue.
//! The kernel exits at `OP_HALT` (or when `pc == num_instructions`),
//! returning control to the host. The next sub-batch launches a fresh
//! kernel — kernel boundary drains L2 and resets MES scheduling state,
//! which is the architectural hypothesis under test by Phase 0b.
//!
//! API mirrors [`crate::persistent_dispatch::PersistentDispatch`] so the
//! `decode_step_p2p` / `dispatch_head_parallel_attention` /
//! `dispatch_moe_workers_decode_async` call sites can route to either
//! via a thin selector on `Model::per_batch_coop`.
//!
//! HIP-API safety: unlike persistent dispatch, every coop kernel exits
//! before the next host call. `hipMemcpyAsync` (instruction upload),
//! `hipStreamSynchronize`, `hipFree`, etc. are all safe between launches.
//! The persistent path's prohibition list does not apply here.
//!
//! Phase 0b is HIP-mediated (`hipModuleLaunchCooperativeKernel`,
//! ~44-224µs/launch on gfx1100 per kb gfx1100-cooperative-launch-overhead).
//! If Phase 0b passes the determinism gate, Phase 1 replaces the launch
//! with direct amdkfd doorbells (~5µs/launch).

use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::Device;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;

use crate::megakernel::{INST_SIZE, Instruction, NUM_CUS};
use crate::persistent_dispatch::{BatchDispatcher, DispatchError};
use crate::watchdog::WatchdogThread;

/// Same upper bound as [`crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS`].
/// Callers that already chunk against the persistent ceiling do not need
/// to re-tune for per-batch coop.
pub const MAX_BATCH_INSTRUCTIONS: usize = 256;

/// Per-GPU state for one-shot cooperative dispatch.
pub struct PerBatchWorker {
    pub device: DeviceId,
    pub stream: Stream,
    /// Module handle kept alive for the lifetime of the worker. The
    /// `Function` lookup is cached internally by `Module::get_function`
    /// (HashMap by name), so re-resolving on each launch is free after
    /// the first call.
    pub module: Module,
    /// VRAM scratch program buffer sized for MAX_BATCH_INSTRUCTIONS.
    /// hipMemcpyAsync target per launch.
    pub program_buf: DeviceBuffer<u64>,
    /// Host-side flat scratch reused across launches; avoids per-launch alloc.
    pub flat_scratch: Vec<u64>,
    pub num_blocks: u32,
    pub shared_mem: u32,
    /// Device pointer to this GPU's WatchdogState page (registered via
    /// the shared `WatchdogThread`). Stable for the worker's lifetime.
    pub wd_dev_ptr: *mut crate::watchdog::WatchdogState,
    /// Monotonic launch counter — returned by `dispatch_batch_fire` so
    /// caller pattern (seq round-trip via `wait_ack`) maps cleanly. Each
    /// per-GPU stream is FIFO, so `wait_ack(gpu, seq)` reduces to
    /// `stream.synchronize()` after at least `seq` launches issued.
    pub launch_counter: u32,
}

pub struct PerBatchDispatch {
    /// Per-device workers, indexed by `DeviceId.0`. `None` = no worker on
    /// that GPU yet (matches `PersistentDispatch` layout for slot-by-id
    /// addressing across the prefill→decode boundary).
    pub workers: Vec<Option<PerBatchWorker>>,
    /// Shared host-side watchdog thread; per-GPU `WatchdogState` pages
    /// are registered on `add_device`.
    pub watchdog: WatchdogThread,
    /// op_profile counter device pointer (null when build flag unset).
    /// Written into kernel arg 3 (`op_profile`) on every launch.
    pub op_profile_dev_ptr: *mut u64,
}

// PerBatchWorker / PerBatchDispatch own raw pointers but the underlying
// host-mapped pages outlive the workers (kept alive by WatchdogThread
// and op_profile's global state).
unsafe impl Send for PerBatchDispatch {}
unsafe impl Sync for PerBatchDispatch {}

impl PerBatchDispatch {
    /// Allocate dispatcher with `total_gpus` slots and launch nothing
    /// yet — callers `add_device` per GPU. The total/slot model matches
    /// `PersistentDispatch::init_with_total` so the choice is just
    /// swapping the type at construction.
    pub fn init_with_total(total_gpus: usize, devices: &[DeviceId], shared_mem: u32) -> HipResult<Self> {
        let watchdog = WatchdogThread::spawn();
        let workers: Vec<Option<PerBatchWorker>> = (0..total_gpus).map(|_| None).collect();
        let mut dispatch = PerBatchDispatch {
            workers,
            watchdog,
            op_profile_dev_ptr: std::ptr::null_mut(),
        };
        for &device in devices {
            dispatch.add_device(device, shared_mem)?;
        }
        Ok(dispatch)
    }

    /// Set the op_profile counter pointer. Must be set BEFORE `add_device`
    /// (or before each subsequent `add_device`) — workers cache the
    /// pointer at launch-arg construction time.
    pub fn set_op_profile_ptr(&mut self, ptr: *mut u64) {
        self.op_profile_dev_ptr = ptr;
    }

    /// Append a GPU to the worker pool. Loads megakernel.hsaco's
    /// `megakernel_f32` entry, allocates VRAM scratch program buffer,
    /// registers WatchdogState. No kernel is launched — coop launches
    /// happen per `dispatch_batch*` call.
    pub fn add_device(&mut self, device: DeviceId, shared_mem: u32) -> HipResult<()> {
        let kernel_dir = crate::kernel::kernel_dir();
        Device::set_current(device)?;
        let stream = Stream::new(device)?;
        let module = Module::load(device, &kernel_dir.join("megakernel.hsaco"))?;
        // Prime the function cache so subsequent get_function calls are
        // HashMap-hit fast paths and so any HSACO symbol error fires here,
        // not at first launch.
        let blocks_per_sm = {
            let func = module.get_function("megakernel_f32")?;
            func.max_active_blocks_per_sm(256, shared_mem as usize)?
        };
        let blocks_per_sm = blocks_per_sm.max(1) as u32;
        let num_blocks = blocks_per_sm * NUM_CUS;

        let program_buf = DeviceBuffer::<u64>::alloc(device, MAX_BATCH_INSTRUCTIONS * INST_SIZE)?;
        let wd_dev_ptr = self.watchdog.register(device)?;

        let slot = device.0 as usize;
        if slot >= self.workers.len() {
            self.workers.resize_with(slot + 1, || None);
        }
        assert!(
            self.workers[slot].is_none(),
            "double-init per-batch worker on GPU {}",
            slot
        );
        self.workers[slot] = Some(PerBatchWorker {
            device,
            stream,
            module,
            program_buf,
            flat_scratch: Vec::with_capacity(MAX_BATCH_INSTRUCTIONS * INST_SIZE),
            num_blocks,
            shared_mem,
            wd_dev_ptr,
            launch_counter: 0,
        });
        eprintln!(
            "  GPU {}: per-batch coop dispatcher armed ({num_blocks} blocks, {shared_mem}B shared)",
            device.0
        );
        Ok(())
    }

    fn worker_mut(&mut self, gpu_idx: usize) -> &mut PerBatchWorker {
        self.workers[gpu_idx]
            .as_mut()
            .expect("no per-batch worker on this GPU")
    }
    fn worker(&self, gpu_idx: usize) -> &PerBatchWorker {
        self.workers[gpu_idx]
            .as_ref()
            .expect("no per-batch worker on this GPU")
    }

    fn dispatch_batch_fire_inner(
        &mut self,
        gpu_idx: usize,
        instructions: &[Instruction],
    ) -> u32 {
        assert!(
            instructions.len() <= MAX_BATCH_INSTRUCTIONS,
            "instruction batch {} exceeds MAX_BATCH_INSTRUCTIONS={}",
            instructions.len(),
            MAX_BATCH_INSTRUCTIONS
        );
        let p0b_diag = std::env::var("BRAIDINFER_P0B_DIAG").is_ok();
        if p0b_diag {
            let op0 = instructions
                .first()
                .map(|i| i.words[0] as u32)
                .unwrap_or(0);
            let op_last = instructions
                .last()
                .map(|i| i.words[0] as u32)
                .unwrap_or(0);
            eprintln!(
                "[p0b]     dispatch_batch_fire gpu={gpu_idx} n={} op0={op0} op_last={op_last}",
                instructions.len()
            );
        }
        let op_profile_ptr = self.op_profile_dev_ptr;
        // Save current device so we can restore it after the launch. Callers
        // outside the dispatcher (e.g. moe_ffn_forward_prefill_batched's GPU
        // 0 kbk follow-on) rely on the current device staying as it was
        // before the dispatch; otherwise their HIP API calls hit the wrong
        // context and fail with hipErrorInvalidValue (400).
        let prior_device = Device::current().expect("Device::current failed");
        let w = self.worker_mut(gpu_idx);
        Device::set_current(w.device).expect("Device::set_current failed");

        // Repack into flat u64 scratch and upload to VRAM scratch program.
        w.flat_scratch.clear();
        for inst in instructions {
            w.flat_scratch.extend_from_slice(&inst.words);
        }
        let upload_bytes = w.flat_scratch.len() * std::mem::size_of::<u64>();
        braidinfer_hip::error::check(unsafe {
            ffi::hipMemcpyAsync(
                w.program_buf.as_mut_ptr().cast(),
                w.flat_scratch.as_ptr().cast(),
                upload_bytes,
                ffi::hipMemcpyHostToDevice,
                w.stream.raw(),
            )
        })
        .expect("hipMemcpyAsync instruction upload failed");

        let mut prog_ptr: *const std::ffi::c_void = w.program_buf.as_ptr().cast();
        let mut num_inst: i32 = instructions.len() as i32;
        let mut wd_ptr: *mut std::ffi::c_void = w.wd_dev_ptr as *mut std::ffi::c_void;
        let mut op_profile_arg: *mut std::ffi::c_void = op_profile_ptr as *mut std::ffi::c_void;
        let mut args: [*mut std::ffi::c_void; 4] = [
            std::ptr::addr_of_mut!(prog_ptr).cast(),
            std::ptr::addr_of_mut!(num_inst).cast(),
            std::ptr::addr_of_mut!(wd_ptr).cast(),
            std::ptr::addr_of_mut!(op_profile_arg).cast(),
        ];
        let func = w
            .module
            .get_function("megakernel_f32")
            .expect("megakernel_f32 not in hsaco");
        func.launch_cooperative(
            (w.num_blocks, 1, 1),
            (256, 1, 1),
            w.shared_mem,
            &w.stream,
            &mut args,
        )
        .expect("launch_cooperative failed");

        w.launch_counter += 1;
        let seq = w.launch_counter;
        if p0b_diag {
            eprintln!("[p0b]     dispatch_batch_fire gpu={gpu_idx} launched seq={seq}");
        }
        // Restore caller's device. `w` borrow ends here naturally; the
        // Device::set_current call doesn't touch `self`.
        Device::set_current(prior_device).expect("Device::set_current(prior) failed");
        seq
    }

}

impl BatchDispatcher for PerBatchDispatch {
    /// Upload an instruction batch and launch `megakernel_f32` on the
    /// per-GPU stream. Returns immediately — the launch is queued; the
    /// caller waits via `wait_ack` (which is `stream.synchronize()`).
    /// Per-GPU streams are FIFO, so the returned seq token only needs to
    /// indicate "drain everything queued so far on this GPU."
    fn dispatch_batch_fire(
        &mut self,
        gpu_idx: usize,
        instructions: &[Instruction],
    ) -> u32 {
        self.dispatch_batch_fire_inner(gpu_idx, instructions)
    }

    /// Dispatch + wait. Equivalent to `dispatch_batch_fire` + `wait_ack`.
    fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        let seq = self.dispatch_batch_fire_inner(gpu_idx, instructions);
        self.wait_ack(gpu_idx, seq);
    }

    /// Slice into MAX_BATCH-sized chunks and dispatch each synchronously.
    fn dispatch_batch_slice(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        for chunk in instructions.chunks(MAX_BATCH_INSTRUCTIONS) {
            self.dispatch_batch(gpu_idx, chunk);
        }
    }

    /// Wait for any launches queued on this GPU's stream to retire.
    /// `seq` is informational only — per-GPU streams are FIFO, so
    /// `stream.synchronize()` drains everything ≤ `launch_counter`.
    fn wait_ack(&self, gpu_idx: usize, _seq: u32) {
        let p0b_diag = std::env::var("BRAIDINFER_P0B_DIAG").is_ok();
        if p0b_diag {
            eprintln!("[p0b]     wait_ack gpu={gpu_idx} seq={_seq} (stream.synchronize start)");
        }
        self.worker(gpu_idx)
            .stream
            .synchronize()
            .expect("stream.synchronize failed");
        if p0b_diag {
            eprintln!("[p0b]     wait_ack gpu={gpu_idx} seq={_seq} done");
        }
    }

    /// Wait for multiple per-GPU sequence targets in one pass. Each per-GPU
    /// stream is independent, so this is just per-GPU `stream.synchronize`
    /// (deduplicated by GPU index).
    fn try_wait_acks_many(&self, targets: &[(usize, u32)]) -> Result<(), DispatchError> {
        let p0b_diag = std::env::var("BRAIDINFER_P0B_DIAG").is_ok();
        if p0b_diag {
            eprintln!("[p0b]     try_wait_acks_many targets={targets:?}");
        }
        let mut seen = vec![false; self.workers.len()];
        for &(gpu, seq) in targets {
            if gpu >= seen.len() {
                continue;
            }
            if seen[gpu] {
                continue;
            }
            seen[gpu] = true;
            self.worker(gpu)
                .stream
                .synchronize()
                .map_err(|_| DispatchError::Timeout {
                    gpu,
                    seq,
                    ack: 0,
                    progress_pc: 0,
                    block_alive_count: 0,
                })?;
        }
        Ok(())
    }

    fn has_worker(&self, gpu_idx: usize) -> bool {
        gpu_idx < self.workers.len() && self.workers[gpu_idx].is_some()
    }

    fn num_gpus(&self) -> usize {
        self.workers.iter().filter(|s| s.is_some()).count()
    }
}
