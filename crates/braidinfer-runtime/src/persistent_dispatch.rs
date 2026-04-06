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
use braidinfer_hip::device::Device;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use braidinfer_hip::ffi;

use crate::megakernel::{Instruction, INST_SIZE};

/// Max instructions per batch dispatch (dense worker).
pub const MAX_BATCH_INSTRUCTIONS: usize = 64;
/// Rust mirror of WorkerQueue from persistent_worker.hip.
/// Layout must match exactly (repr(C)).
#[repr(C)]
pub struct WorkerQueueLayout {
    pub seq_num: u32,
    pub shutdown: u32,
    pub num_instructions: u32,  // how many instructions in this batch (1..MAX_BATCH)
    pub _pad: u32,
    pub inst: [u64; MAX_BATCH_INSTRUCTIONS * INST_SIZE],  // instruction batch buffer
    pub ack: u32,
    pub done: u32,  // kernel writes 1 when exiting after shutdown (for Drop polling)
    pub _pad2: [u32; 2],
}

/// Per-GPU worker state.
pub struct GpuWorker {
    pub device: DeviceId,
    pub queue: MappedHostBuffer<u8>,  // WorkerQueueLayout, host-mapped
    pub stream: Stream,
    pub module: Module,
    pub seq_counter: u32,
}

impl GpuWorker {
    /// Dispatch a single instruction and wait for completion.
    pub(crate) fn dispatch_and_wait(&mut self, inst: &Instruction) {
        let q_ptr = self.queue.host_ptr() as *mut WorkerQueueLayout;

        // Copy instruction words to work queue
        for i in 0..INST_SIZE {
            unsafe {
                std::ptr::write_volatile(
                    std::ptr::addr_of_mut!((*q_ptr).inst[i]),
                    inst.words[i],
                );
            }
        }

        // Increment and write seq_num (triggers worker)
        self.seq_counter += 1;
        let seq = self.seq_counter;
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }

        // Poll ack with timeout
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq { break; }
            if start.elapsed().as_secs() > 10 {
                let opcode = inst.words[0] & 0x7FFFFFFF;
                let grid_x = (inst.words[0] >> 32) as u32;
                panic!("PERSISTENT: dispatch timeout seq={seq} opcode={opcode} grid_x={grid_x} ack={ack}");
            }
            std::hint::spin_loop();
        }
    }
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
            func.launch_cooperative((num_blocks, 1, 1), (256, 1, 1), shared_mem, &stream, &mut args)?;
            eprintln!("  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B shared)", device.0);
            workers.push(std::mem::ManuallyDrop::new(GpuWorker { device, queue, stream, module, seq_counter: 0 }));
        }

        let moe_output_slot = MappedHostBuffer::<f32>::alloc(1)?; // placeholder for init() path
        Ok(PersistentDispatch { workers, moe_output_slot })
    }

    /// Wait for a GPU to ack a specific seq number.
    pub(crate) fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        let q_ptr = self.workers[gpu_idx].queue.host_ptr() as *const WorkerQueueLayout;
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq { break; }
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
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).num_instructions), instructions.len() as u32); }

        // Trigger worker
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }

        // Wait for ack
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq { break; }
            if start.elapsed().as_secs() > 2 {
                let opcode0 = instructions[0].words[0] & 0x7FFFFFFF;
                panic!("dispatch_batch timeout gpu={gpu_idx} seq={seq} n={} opcode0={opcode0}", instructions.len());
            }
            std::hint::spin_loop();
        }
    }

    /// Fire a batch of instructions to a GPU WITHOUT waiting for ack. Returns seq for wait_ack.
    /// Caller must call wait_ack(gpu_idx, seq) before reading GPU 0 output.
    /// Used to overlap GPU 0 OP_EXPERT_FFN with kbk dispatch on GPUs 1+.
    pub(crate) fn dispatch_batch_fire(&mut self, gpu_idx: usize, instructions: &[Instruction]) -> u32 {
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
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).num_instructions), instructions.len() as u32); }
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }
        seq
    }

    /// Multi-GPU init: fat dense worker on GPU 0 only.
    /// MoE dispatch uses kbk (hipLaunchKernel) on GPUs 1+ — no lean workers launched.
    pub fn init_multi_gpu(gpu0: DeviceId, _all_devices: &[DeviceId], shared_mem: u32, hidden_size: usize, _max_eis: usize) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        Device::set_current(gpu0)?;
        let gpu0_queue = MappedHostBuffer::<u8>::alloc(std::mem::size_of::<WorkerQueueLayout>())?;
        let gpu0_module = Module::load(gpu0, &kernel_dir.join("persistent_worker.hsaco"))?;
        let gpu0_func = gpu0_module.get_function("persistent_worker")?;
        let gpu0_stream = Stream::new(gpu0)?;
        let blocks_per_sm = gpu0_func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        let num_cus = multiprocessor_count(gpu0)?;
        let num_blocks = (blocks_per_sm as u32 * num_cus).max(num_cus);
        {
            let mut q = gpu0_queue.device_ptr() as *mut std::ffi::c_void;
            let mut args = [std::ptr::addr_of_mut!(q).cast::<std::ffi::c_void>()];
            gpu0_func.launch_cooperative((num_blocks, 1, 1), (256, 1, 1), shared_mem, &gpu0_stream, &mut args)?;
        }
        // Allocate GPU 0 MoE output slot (MTYPE_UC): expert FFN results accumulate here.
        let moe_output_slot = MappedHostBuffer::<f32>::alloc(hidden_size.max(1))?;
        eprintln!("  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B); MoE GPU 0 via batch ops, GPUs 1+ via kbk", gpu0.0);
        Ok(PersistentDispatch {
            workers: vec![std::mem::ManuallyDrop::new(GpuWorker {
                device: gpu0, queue: gpu0_queue, stream: gpu0_stream,
                module: gpu0_module, seq_counter: 0,
            })],
            moe_output_slot,
        })
    }

    /// Shut down workers (write shutdown flag, sync streams).
    pub fn shutdown(&mut self) {
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1); }
        }
        for worker in &self.workers {
            let _ = Device::set_current(worker.device);
            let _ = worker.stream.synchronize();
        }
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
        // Write shutdown flags (host-mapped write, no HIP API)
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1); }
        }
        // Poll kernel-written `done` flag. Kernel writes done=1 before returning on shutdown.
        let shutdown_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *const WorkerQueueLayout;
            loop {
                let done = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).done)) };
                if done != 0 { break; }
                if std::time::Instant::now() > shutdown_deadline {
                    eprintln!("braidinfer: persistent worker shutdown timeout on GPU {}", worker.device.0);
                    break;
                }
                std::hint::spin_loop();
            }
        }
        // Cooperative kernel has exited (or timed out). Safe to call HIP API now.
        for worker in &mut self.workers {
            unsafe { std::mem::ManuallyDrop::drop(worker); }
        }
    }
}
