//! CPU-scheduled persistent worker dispatch (braidinfer-czl).
//! Each GPU runs a persistent cooperative kernel polling a host-mapped work queue.
//! CPU sequences operations via memory writes — no HIP API calls in the hot path.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;

use crate::megakernel::{Instruction, INST_SIZE};

/// Rust mirror of WorkerQueue from persistent_worker.hip.
/// Layout must match exactly (repr(C)).
#[repr(C)]
pub struct WorkerQueueLayout {
    pub seq_num: u32,
    pub shutdown: u32,
    pub _pad: [u32; 2],
    pub inst: [u64; INST_SIZE],  // 17 u64s = 136 bytes
    pub ack: u32,
    pub _pad2: [u32; 3],
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
    /// Get mutable reference to queue layout.
    fn queue_mut(&self) -> &mut WorkerQueueLayout {
        unsafe { &mut *(self.queue.host_ptr() as *mut WorkerQueueLayout) }
    }

    /// Get reference to queue layout.
    fn queue_ref(&self) -> &WorkerQueueLayout {
        unsafe { &*(self.queue.host_ptr() as *const WorkerQueueLayout) }
    }

    /// Dispatch a single instruction and wait for completion.
    pub fn dispatch_and_wait(&mut self, inst: &Instruction) {
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

/// Persistent dispatch context: manages worker kernels on all GPUs.
pub struct PersistentDispatch {
    pub workers: Vec<GpuWorker>,
}

impl PersistentDispatch {
    /// Launch persistent workers on specified GPUs.
    pub fn init(devices: &[DeviceId], shared_mem: u32) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let queue_size = std::mem::size_of::<WorkerQueueLayout>();
        let mut workers = Vec::with_capacity(devices.len());

        for &device in devices {
            Device::set_current(device)?;

            let queue = MappedHostBuffer::<u8>::alloc(queue_size)?;
            // Zero-init already done by MappedHostBuffer::alloc

            let stream = Stream::new(device)?;
            let module = Module::load(device, &kernel_dir.join("persistent_worker.hsaco"))?;
            let func = module.get_function("persistent_worker")?;

            // Kernel arg: pointer to WorkerQueue (device-mapped address)
            let mut queue_ptr = queue.device_ptr() as *mut std::ffi::c_void;
            let mut args: [*mut std::ffi::c_void; 1] = [
                std::ptr::addr_of_mut!(queue_ptr).cast(),
            ];

            // Query max cooperative blocks for this kernel/shared_mem combination.
            let num_blocks = 96u32;

            func.launch_cooperative(
                (num_blocks, 1, 1),
                (256, 1, 1),
                shared_mem,
                &stream,
                &mut args,
            )?;

            eprintln!("  GPU {}: persistent worker launched ({num_blocks} blocks, {shared_mem}B shared)",
                      device.0);

            workers.push(GpuWorker {
                device,
                queue,
                stream,
                module,
                seq_counter: 0,
            });
        }

        Ok(PersistentDispatch { workers })
    }

    /// Dispatch an instruction to a specific GPU and wait for completion.
    pub fn dispatch(&mut self, gpu_idx: usize, inst: &Instruction) {
        self.workers[gpu_idx].dispatch_and_wait(inst);
    }

    /// Dispatch an instruction to GPU 0 (primary).
    pub fn dispatch_gpu0(&mut self, inst: &Instruction) {
        self.workers[0].dispatch_and_wait(inst);
    }

    /// Execute a full program (sequence of instructions) on GPU 0.
    pub fn execute_program(&mut self, instructions: &[Instruction]) {
        for inst in instructions {
            self.workers[0].dispatch_and_wait(inst);
        }
    }

    /// Dispatch different instructions to each GPU in parallel, wait for all.
    /// `per_gpu[i]` is the instruction for GPU i (None = skip that GPU).
    pub fn dispatch_parallel(&mut self, per_gpu: &[Option<Instruction>]) {
        // Fire all instructions (non-blocking writes)
        let mut pending = Vec::new();
        for (i, maybe_inst) in per_gpu.iter().enumerate() {
            if let Some(inst) = maybe_inst {
                if i < self.workers.len() {
                    let w = &mut self.workers[i];
                    let q_ptr = w.queue.host_ptr() as *mut WorkerQueueLayout;
                    for j in 0..INST_SIZE {
                        unsafe {
                            std::ptr::write_volatile(
                                std::ptr::addr_of_mut!((*q_ptr).inst[j]),
                                inst.words[j],
                            );
                        }
                    }
                    w.seq_counter += 1;
                    let seq = w.seq_counter;
                    unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }
                    pending.push((i, seq));
                }
            }
        }
        // Wait for all acks
        for (i, seq) in pending {
            let q_ptr = self.workers[i].queue.host_ptr() as *const WorkerQueueLayout;
            loop {
                let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
                if ack == seq { break; }
                std::hint::spin_loop();
            }
        }
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

    /// Re-launch workers after shutdown (reuses existing modules and queues).
    pub fn relaunch(&mut self, shared_mem: u32) -> HipResult<()> {
        for worker in &mut self.workers {
            Device::set_current(worker.device)?;
            // Reset queue state
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe {
                std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), 0);
                std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 0);
                std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).ack), 0);
            }
            worker.seq_counter = 0;

            let func = worker.module.get_function("persistent_worker")?;
            let mut queue_ptr = worker.queue.device_ptr() as *mut std::ffi::c_void;
            let mut args: [*mut std::ffi::c_void; 1] = [
                std::ptr::addr_of_mut!(queue_ptr).cast(),
            ];
            let num_blocks = 96u32;
            func.launch_cooperative(
                (num_blocks, 1, 1), (256, 1, 1), shared_mem, &worker.stream, &mut args,
            )?;
        }
        Ok(())
    }

    /// Number of GPUs with workers.
    pub fn num_gpus(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for PersistentDispatch {
    fn drop(&mut self) {
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1); }
        }
        for worker in &self.workers {
            let _ = Device::set_current(worker.device);
            let _ = worker.stream.synchronize();
        }
    }
}
