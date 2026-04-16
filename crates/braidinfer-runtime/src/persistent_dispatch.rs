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

/// Max instructions per batch dispatch (dense worker).
pub const MAX_BATCH_INSTRUCTIONS: usize = 64;
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

/// Persistent dispatch context: manages the fat cooperative worker on GPU 0.
///
/// `workers` is wrapped in `ManuallyDrop` so HIP resources (DeviceBuffer → hipFree,
/// MappedHostBuffer → hipHostFree, Stream → hipStreamDestroy, Module → hipModuleUnload)
/// are only freed after the cooperative kernel has confirmed exit via the `done` flag.
/// Auto-drop would free while the kernel is still running.
///
/// MoE dispatch: GPU 0 gets experts via OP_EXPERT_FFN (fat worker); GPUs 1+ via kbk.
/// moe_output_slot holds GPU 0's expert accumulation result (host-mapped, MTYPE_UC).
/// CPU zeros it before firing GPU 0's batch, then adds it into ffn_down_stage after.
pub struct PersistentDispatch {
    pub workers: Vec<std::mem::ManuallyDrop<GpuWorker>>,
    /// Host-mapped buffer for GPU 0 expert FFN output (hidden_size f32 elements).
    /// Allocated on GPU 0 (MTYPE_UC). Valid device_ptr only from GPU 0.
    pub moe_output_slot: MappedHostBuffer<f32>,
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
    fn request_shutdown(&self) {
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe {
                std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1);
            }
        }
    }

    /// Launch persistent workers on specified GPUs.
    /// `hidden_size`: for MoE output slot allocation (0 if no MoE).
    pub fn init(devices: &[DeviceId], shared_mem: u32, _hidden_size: usize) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let queue_size = std::mem::size_of::<WorkerQueueLayout>();
        let mut workers = Vec::with_capacity(devices.len());

        for &device in devices {
            Device::set_current(device)?;
            let queue = MappedHostBuffer::<u8>::alloc(queue_size)?;
            let stream = Stream::new(device)?;
            let module = Module::load(device, &kernel_dir.join("persistent_worker.hsaco"))?;
            let func = module.get_function("persistent_worker")?;
            let mut queue_ptr = queue.device_ptr() as *mut std::ffi::c_void;
            let mut args: [*mut std::ffi::c_void; 1] = [std::ptr::addr_of_mut!(queue_ptr).cast()];
            let bpsm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
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
                "  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B shared)",
                device.0
            );
            braidinfer_hip::set_persistent_worker_active(true);
            workers.push(std::mem::ManuallyDrop::new(GpuWorker {
                device,
                queue,
                stream,
                module,
                seq_counter: 0,
            }));
        }

        let moe_output_slot = MappedHostBuffer::<f32>::alloc(1)?; // placeholder for init() path
        Ok(PersistentDispatch {
            workers,
            moe_output_slot,
        })
    }

    /// Wait for a GPU to ack a specific seq number.
    pub(crate) fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        let q_ptr = self.workers[gpu_idx].queue.host_ptr() as *const WorkerQueueLayout;
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq {
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// Dispatch a batch of instructions to a GPU. Worker executes all with grid.sync()
    /// between them, acks once at the end. One signal round-trip per batch.
    pub(crate) fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        assert!(instructions.len() <= MAX_BATCH_INSTRUCTIONS);
        let w = &mut self.workers[gpu_idx];
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
            if start.elapsed().as_secs() > 30 {
                let opcode0 = instructions[0].words[0] & 0x7FFFFFFF;
                let progress_pc = unsafe {
                    std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).progress_pc))
                };
                let stuck_op = instructions
                    .get(progress_pc as usize)
                    .map(|i| i.words[0] & 0x7FFFFFFF)
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
            let op0 = instructions[0].words[0] & 0x7FFFFFFF;
            eprintln!(
                "dispatch_batch gpu={gpu_idx} n={} op0={op0:#x} rtt={us}us",
                instructions.len()
            );
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
        let w = &mut self.workers[gpu_idx];
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

    /// Multi-GPU init: launch persistent workers on ALL devices.
    /// GPU 0 handles MoE + its head slice; GPUs 1+ handle their head slices.
    /// Launch persistent cooperative worker on GPU 0 only.
    /// GPUs 1+ use kbk (hipLaunchKernel) for MoE expert dispatch — cooperative kernels
    /// hold all SMs and deadlock with kbk on the same device.
    pub fn init_multi_gpu(
        gpu0: DeviceId,
        _all_devices: &[DeviceId],
        shared_mem: u32,
        hidden_size: usize,
        _max_eis: usize,
    ) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        Device::set_current(gpu0)?;
        let queue = MappedHostBuffer::<u8>::alloc(std::mem::size_of::<WorkerQueueLayout>())?;
        let stream = Stream::new(gpu0)?;
        let module = Module::load(gpu0, &kernel_dir.join("persistent_worker.hsaco"))?;
        let func = module.get_function("persistent_worker")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        let num_cus = multiprocessor_count(gpu0)?;
        let num_blocks = (blocks_per_sm as u32 * num_cus).max(num_cus);
        let mut q = queue.device_ptr() as *mut std::ffi::c_void;
        let mut args = [std::ptr::addr_of_mut!(q).cast::<std::ffi::c_void>()];
        func.launch_cooperative(
            (num_blocks, 1, 1),
            (256, 1, 1),
            shared_mem,
            &stream,
            &mut args,
        )?;
        eprintln!(
            "  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B); GPUs 1+ use kbk",
            gpu0.0
        );
        braidinfer_hip::set_persistent_worker_active(true);
        let moe_output_slot = MappedHostBuffer::<f32>::alloc(hidden_size.max(1))?;
        Ok(PersistentDispatch {
            workers: vec![std::mem::ManuallyDrop::new(GpuWorker {
                device: gpu0,
                queue,
                stream,
                module,
                seq_counter: 0,
            })],
            moe_output_slot,
        })
    }

    /// Request worker shutdown via host-mapped flags only.
    ///
    /// This intentionally does not call any HIP APIs. Cooperative kernels must
    /// exit and signal `done` before stream or memory cleanup becomes safe.
    pub fn shutdown(&mut self) {
        self.request_shutdown();
    }

    /// Number of GPUs with workers.
    pub fn num_gpus(&self) -> usize {
        self.workers.len()
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
        // If we're already panicking (e.g. dispatch_batch timeout), the kernel is stuck
        // inside grid.sync() and will never see shutdown. Exit immediately.
        if std::thread::panicking() {
            std::process::exit(1);
        }
        self.request_shutdown();
        // Poll kernel-written `done` flag. Kernel writes done=1 before returning on shutdown.
        let shutdown_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut worker_done = vec![false; self.workers.len()];
        for (idx, worker) in self.workers.iter().enumerate() {
            let q_ptr = worker.queue.host_ptr() as *const WorkerQueueLayout;
            loop {
                let done = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).done)) };
                if done != 0 {
                    worker_done[idx] = true;
                    break;
                }
                if std::time::Instant::now() > shutdown_deadline {
                    eprintln!(
                        "braidinfer: persistent worker shutdown timeout on GPU {}",
                        worker.device.0
                    );
                    break;
                }
                std::hint::spin_loop();
            }
        }
        // Free HIP resources only for workers that confirmed exit. Timed-out
        // workers are intentionally leaked to avoid deadlocking on HIP cleanup.
        for (idx, worker) in self.workers.iter_mut().enumerate() {
            if worker_done[idx] {
                unsafe {
                    std::mem::ManuallyDrop::drop(worker);
                }
            }
        }
    }
}
