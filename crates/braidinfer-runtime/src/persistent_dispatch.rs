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

/// Max instructions per batch dispatch (dense worker).
pub const MAX_BATCH_INSTRUCTIONS: usize = 64;
/// Max instructions per MoE batch (lean MoE worker).
pub const MAX_MOE_BATCH: usize = 16;

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
    pub _pad2: [u32; 3],
}

/// Rust mirror of MoeWorkerQueue from moe_gemv_worker.hip.
#[repr(C)]
pub struct MoeWorkerQueueLayout {
    pub seq_num: u32,
    pub shutdown: u32,
    pub num_instructions: u32,
    pub _pad: u32,
    pub inst: [u64; MAX_MOE_BATCH * INST_SIZE],
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

/// Per-GPU lean MoE worker (separate from the fat dense worker).
pub struct MoeGpuWorker {
    pub device: DeviceId,
    pub queue: MappedHostBuffer<u8>,
    pub stream: Stream,
    pub module: Module,
    pub seq_counter: u32,
    // Per-block scratch buffers: [num_blocks * max_eis] for gate/up/act
    pub scratch_gate: braidinfer_hip::DeviceBuffer<f32>,
    pub scratch_up:   braidinfer_hip::DeviceBuffer<f32>,
    pub scratch_act:  braidinfer_hip::DeviceBuffer<f32>,
    pub num_blocks: u32,
}

/// Persistent dispatch context: manages worker kernels on all GPUs.
pub struct PersistentDispatch {
    pub workers: Vec<GpuWorker>,
    /// Lean MoE GEMV workers — one per GPU, high occupancy (~6 blocks/CU).
    pub moe_workers: Vec<MoeGpuWorker>,
    /// Per-GPU host-mapped output slots for MoE result gathering [hidden_size each].
    pub moe_output_slots: Vec<MappedHostBuffer<f32>>,
}

impl PersistentDispatch {
    /// Launch persistent workers on specified GPUs.
    /// `hidden_size`: for MoE output slot allocation (0 if no MoE).
    pub fn init(devices: &[DeviceId], shared_mem: u32, hidden_size: usize) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let queue_size = std::mem::size_of::<WorkerQueueLayout>();
        let mut workers = Vec::with_capacity(devices.len());
        let mut moe_output_slots = Vec::with_capacity(devices.len());

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

            // 96 blocks: limited by 251 VGPRs per wavefront (2016 VGPRs/block).
            // Total device VGPRs: 294K. 192 blocks would need 387K → doesn't fit.
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

        // Allocate per-GPU host-mapped output slots for MoE result gathering
        for _ in 0..devices.len() {
            let slot = if hidden_size > 0 {
                MappedHostBuffer::<f32>::alloc(hidden_size)?
            } else {
                MappedHostBuffer::<f32>::alloc(1)?
            };
            moe_output_slots.push(slot);
        }

        // Launch lean MoE GEMV workers on GPUs 1+ (GPU 0 uses fat worker for its experts).
        // Can't coexist two cooperative kernels on the same GPU.
        let moe_queue_size = std::mem::size_of::<MoeWorkerQueueLayout>();
        let mut moe_workers = Vec::with_capacity(devices.len());
        // GPU 0 placeholder (no MoE worker)
        if hidden_size > 0 && devices.len() > 1 {
            let moe_shared_mem = 256u32 * 4;
            for &device in &devices[1..] { // skip GPU 0
                Device::set_current(device)?;
                let queue = MappedHostBuffer::<u8>::alloc(moe_queue_size)?;
                let stream = Stream::new(device)?;
                let module = Module::load(device, &kernel_dir.join("moe_gemv_worker.hsaco"))?;
                let func = module.get_function("moe_gemv_worker")?;

                let blocks_per_sm = func.max_active_blocks_per_sm(256, moe_shared_mem as usize)?;
                let num_blocks = (blocks_per_sm as u32 * 96).min(576); // 96 CUs, target 6/CU
                eprintln!("  GPU {}: MoE GEMV worker launched ({num_blocks} blocks, {blocks_per_sm}/CU)",
                          device.0);

                let mut queue_ptr = queue.device_ptr() as *mut std::ffi::c_void;
                let mut args: [*mut std::ffi::c_void; 1] = [
                    std::ptr::addr_of_mut!(queue_ptr).cast(),
                ];
                func.launch_cooperative(
                    (num_blocks, 1, 1), (256, 1, 1), moe_shared_mem, &stream, &mut args,
                )?;

                moe_workers.push(MoeGpuWorker {
                    device, queue, stream, module, seq_counter: 0,
                    scratch_gate: braidinfer_hip::DeviceBuffer::<f32>::alloc(device, 1)?,
                    scratch_up:   braidinfer_hip::DeviceBuffer::<f32>::alloc(device, 1)?,
                    scratch_act:  braidinfer_hip::DeviceBuffer::<f32>::alloc(device, 1)?,
                    num_blocks: num_blocks,
                });
            }
        }

        Ok(PersistentDispatch { workers, moe_workers, moe_output_slots })
    }

    /// Dispatch an instruction to a specific GPU and wait for completion.
    pub fn dispatch(&mut self, gpu_idx: usize, inst: &Instruction) {
        self.workers[gpu_idx].dispatch_and_wait(inst);
    }

    /// Dispatch an instruction to a GPU WITHOUT waiting for ack. Returns seq for wait_ack.
    pub fn dispatch_fire(&mut self, gpu_idx: usize, inst: &Instruction) -> u32 {
        let w = &mut self.workers[gpu_idx];
        let q_ptr = w.queue.host_ptr() as *mut WorkerQueueLayout;
        for j in 0..INST_SIZE {
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).inst[j]), inst.words[j]); }
        }
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }
        seq
    }

    /// Wait for a GPU to ack a specific seq number.
    pub fn wait_ack(&self, gpu_idx: usize, seq: u32) {
        let q_ptr = self.workers[gpu_idx].queue.host_ptr() as *const WorkerQueueLayout;
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq { break; }
            std::hint::spin_loop();
        }
    }

    /// Dispatch an instruction to GPU 0 (primary).
    pub fn dispatch_gpu0(&mut self, inst: &Instruction) {
        self.workers[0].dispatch_and_wait(inst);
    }

    /// Dispatch a batch of instructions to a GPU. Worker executes all with grid.sync()
    /// between them, acks once at the end. One signal round-trip per batch.
    pub fn dispatch_batch(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
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

    /// Multi-GPU init: fat dense worker on gpu0 only, lean MoE workers on all GPUs except gpu0.
    /// Multi-GPU init: fat dense worker on gpu0, lean MoE workers on GPUs 1+.
    /// CRITICAL ordering: ALL hipHostMalloc allocations BEFORE any cooperative kernel launch.
    /// hipHostGetDevicePointer stalls with running cooperative kernels.
    pub fn init_multi_gpu(gpu0: DeviceId, all_devices: &[DeviceId], shared_mem: u32, hidden_size: usize, max_eis: usize) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let moe_shared_mem = 256u32 * 4;
        let moe_queue_size = std::mem::size_of::<MoeWorkerQueueLayout>();
        let dense_queue_size = std::mem::size_of::<WorkerQueueLayout>();

        // PHASE 1: Allocate ALL host-mapped buffers (before any kernel launch)
        Device::set_current(gpu0)?;
        let gpu0_queue = MappedHostBuffer::<u8>::alloc(dense_queue_size)?;

        let mut moe_queues: Vec<MappedHostBuffer<u8>> = Vec::new();
        for &device in &all_devices[1..] {
            Device::set_current(device)?;
            moe_queues.push(MappedHostBuffer::<u8>::alloc(moe_queue_size)?);
        }

        let mut moe_output_slots: Vec<MappedHostBuffer<f32>> = Vec::new();
        for _ in 0..all_devices.len() {
            moe_output_slots.push(MappedHostBuffer::<f32>::alloc(hidden_size.max(1))?);
        }

        // PHASE 2: Load modules and query occupancy (no kernel launches yet)
        Device::set_current(gpu0)?;
        let gpu0_module = Module::load(gpu0, &kernel_dir.join("persistent_worker.hsaco"))?;
        let gpu0_func = gpu0_module.get_function("persistent_worker")?;
        let _ = gpu0_func.max_active_blocks_per_sm(256, shared_mem as usize)?; // warm up

        let mut moe_modules: Vec<Module> = Vec::new();
        for &device in &all_devices[1..] {
            Device::set_current(device)?;
            let module = Module::load(device, &kernel_dir.join("moe_gemv_worker.hsaco"))?;
            let func = module.get_function("moe_gemv_worker")?;
            let bps = func.max_active_blocks_per_sm(256, moe_shared_mem as usize)?;
            eprintln!("  GPU {}: lean MoE worker (384 blocks, {bps}/CU)", device.0);
            moe_modules.push(module);
        }

        // PHASE 3: Launch all cooperative kernels (all host-mapped memory already allocated)
        Device::set_current(gpu0)?;
        let gpu0_stream = Stream::new(gpu0)?;
        {
            let mut q = gpu0_queue.device_ptr() as *mut std::ffi::c_void;
            let mut args = [std::ptr::addr_of_mut!(q).cast::<std::ffi::c_void>()];
            gpu0_func.launch_cooperative((96, 1, 1), (256, 1, 1), shared_mem, &gpu0_stream, &mut args)?;
        }
        eprintln!("  GPU {}: persistent worker launched (96 blocks, {shared_mem}B)", gpu0.0);

        let mut moe_workers: Vec<MoeGpuWorker> = Vec::new();
        for (i, &device) in all_devices[1..].iter().enumerate() {
            Device::set_current(device)?;
            let stream = Stream::new(device)?;
            let func = moe_modules[i].get_function("moe_gemv_worker")?;
            // Per-block scratch: 96 blocks × max_eis floats each
            let num_blocks = 384u32; // test: 4 args at 384 blocks (same as 1-arg that worked)
            let scratch_sz = num_blocks as usize * max_eis.max(1);
            let scratch_gate = braidinfer_hip::DeviceBuffer::<f32>::alloc(device, scratch_sz)?;
            let scratch_up   = braidinfer_hip::DeviceBuffer::<f32>::alloc(device, scratch_sz)?;
            let scratch_act  = braidinfer_hip::DeviceBuffer::<f32>::alloc(device, scratch_sz)?;

            let mut q  = moe_queues[i].device_ptr() as *mut std::ffi::c_void;
            let mut sg = scratch_gate.as_write_ptr() as *mut std::ffi::c_void;
            let mut su = scratch_up.as_write_ptr()   as *mut std::ffi::c_void;
            let mut sa = scratch_act.as_write_ptr()  as *mut std::ffi::c_void;
            let mut args = [
                std::ptr::addr_of_mut!(q).cast::<std::ffi::c_void>(),
                std::ptr::addr_of_mut!(sg).cast::<std::ffi::c_void>(),
                std::ptr::addr_of_mut!(su).cast::<std::ffi::c_void>(),
                std::ptr::addr_of_mut!(sa).cast::<std::ffi::c_void>(),
            ];
            func.launch_cooperative((num_blocks, 1, 1), (256, 1, 1), moe_shared_mem, &stream, &mut args)
                .map_err(|e| { eprintln!("  GPU {}: lean MoE launch FAILED: {:?}", device.0, e); e })?;
            moe_workers.push(MoeGpuWorker {
                device,
                queue: moe_queues.remove(0),
                stream,
                module: moe_modules.remove(0),
                seq_counter: 0,
                scratch_gate,
                scratch_up,
                scratch_act,
                num_blocks,
            });
        }

        Device::set_current(gpu0)?;
        Ok(PersistentDispatch {
            workers: vec![GpuWorker {
                device: gpu0, queue: gpu0_queue, stream: gpu0_stream,
                module: gpu0_module, seq_counter: 0,
            }],
            moe_workers,
            moe_output_slots,
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
            let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
            // 251 VGPRs × 8 waves/block = 2016 VGPRs/block. 96 CUs × 1536 VGPRs/SIMD × 2 SIMDs
            // = 294K total. Max blocks = 294912/2016 = 146. Cooperative launch needs clean fit → 96.
            let num_blocks = 96u32;
            func.launch_cooperative(
                (num_blocks, 1, 1), (256, 1, 1), shared_mem, &worker.stream, &mut args,
            )?;
        }
        Ok(())
    }

    /// Dispatch a sequence of instructions to a GPU, waiting only after the last one.
    pub fn dispatch_sequence(&mut self, gpu_idx: usize, instructions: &[Instruction]) {
        for (i, inst) in instructions.iter().enumerate() {
            let w = &mut self.workers[gpu_idx];
            let q_ptr = w.queue.host_ptr() as *mut WorkerQueueLayout;
            for j in 0..INST_SIZE {
                unsafe {
                    std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).inst[j]), inst.words[j]);
                }
            }
            w.seq_counter += 1;
            let seq = w.seq_counter;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }

            // Only wait for ack on the last instruction
            if i == instructions.len() - 1 {
                let start = std::time::Instant::now();
                loop {
                    let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
                    if ack == seq { break; }
                    if start.elapsed().as_secs() > 10 {
                        let opcode = inst.words[0] & 0x7FFFFFFF;
                        panic!("dispatch_sequence timeout gpu={gpu_idx} seq={seq} opcode={opcode}");
                    }
                    std::hint::spin_loop();
                }
            } else {
                // Wait briefly for worker to consume before writing next instruction
                let q_ptr = q_ptr as *const WorkerQueueLayout;
                loop {
                    let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
                    if ack == seq { break; }
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Dispatch a batch of OP_EXPERT_FFN instructions to a GPU's lean MoE worker.
    /// Fire without waiting — call moe_wait_ack after dispatching to all GPUs.
    pub fn moe_dispatch_fire(&mut self, gpu_idx: usize, instructions: &[Instruction]) -> u32 {
        assert!(instructions.len() <= MAX_MOE_BATCH);
        let w = &mut self.moe_workers[gpu_idx];
        let q_ptr = w.queue.host_ptr() as *mut MoeWorkerQueueLayout;

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
        unsafe {
            std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).num_instructions), instructions.len() as u32);
        }
        w.seq_counter += 1;
        let seq = w.seq_counter;
        unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).seq_num), seq); }
        seq
    }

    /// Wait for a MoE worker to ack.
    pub fn moe_wait_ack(&self, gpu_idx: usize, seq: u32) {
        let q_ptr = self.moe_workers[gpu_idx].queue.host_ptr() as *const MoeWorkerQueueLayout;
        let start = std::time::Instant::now();
        loop {
            let ack = unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*q_ptr).ack)) };
            if ack == seq { break; }
            if start.elapsed().as_secs() > 2 {
                panic!("moe_wait_ack timeout gpu={gpu_idx} seq={seq}");
            }
            std::hint::spin_loop();
        }
    }

    /// Number of GPUs with workers.
    pub fn num_gpus(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for PersistentDispatch {
    fn drop(&mut self) {
        // Shut down fat dense workers
        for worker in &self.workers {
            let q_ptr = worker.queue.host_ptr() as *mut WorkerQueueLayout;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1); }
        }
        // Shut down lean MoE workers
        for worker in &self.moe_workers {
            let q_ptr = worker.queue.host_ptr() as *mut MoeWorkerQueueLayout;
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*q_ptr).shutdown), 1); }
        }
        for worker in &self.workers {
            let _ = Device::set_current(worker.device);
            let _ = worker.stream.synchronize();
        }
        for worker in &self.moe_workers {
            let _ = Device::set_current(worker.device);
            let _ = worker.stream.synchronize();
        }
    }
}
