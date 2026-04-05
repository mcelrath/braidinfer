//! Multi-GPU context: P2P setup, per-device streams and events for expert parallel dispatch.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer, PinnedBuffer};

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

    pub fn raw(&self) -> ffi::hipEvent_t { self.raw }

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
    pub transfer_stream: Stream,
    pub broadcast_done: HipEvent,  // signaled after activation arrives
    pub compute_done: HipEvent,    // signaled after expert FFN completes
    // Per-worker activation and scratch buffers
    pub activation_in: DeviceBuffer<f32>,   // [hidden_size] — receives broadcast
    pub expert_out: DeviceBuffer<f32>,      // [hidden_size] — accumulated output
    pub scratch_gate: DeviceBuffer<f32>,    // [max_expert_intermediate_size]
    pub scratch_up: DeviceBuffer<f32>,      // [max_expert_intermediate_size]
    pub scratch_act: DeviceBuffer<f32>,     // [max_expert_intermediate_size]
    pub scratch_down: DeviceBuffer<f32>,    // [hidden_size] — down proj output before scale_add
    pub transfer_done: HipEvent,   // signaled after D2H gather completes
    // Compute-path P2P copy kernel (avoids SDMA PERMISSION_FAULT on RDNA3 PCIe)
    pub peer_copy_module: Module,
    // Pre-allocated pinned host buffer for async gather (hipHostMalloc for true async DMA)
    pub gather_host: PinnedBuffer<f32>,
}

/// Multi-GPU context for expert parallel dispatch.
pub struct MultiGpuContext {
    pub num_devices: usize,
    pub workers: Vec<GpuWorker>,
    pub gather_stream: Stream,     // on GPU 0, used to gather results from workers
    pub gather_done: HipEvent,     // signaled after all results gathered
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
                if i == j { continue; }
                Device::set_current(DeviceId(i as u32))?;
                let mut can_access = 0i32;
                unsafe {
                    ffi::hipDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);
                }
                if can_access != 0 {
                    let rc = unsafe { ffi::hipDeviceEnablePeerAccess(j as i32, 0) };
                    // Ignore "already enabled" and "invalid device" (P2P not supported on this topology)
                    if rc != 0 && rc != ffi::hipErrorInvalidDevice && rc != ffi::hipErrorPeerAccessAlreadyEnabled {
                        eprintln!("Warning: hipDeviceEnablePeerAccess({i}→{j}) failed: rc={rc}");
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
            eprintln!("Multi-GPU: allocating buffers on GPU {i} (hs={hidden_size}, eis={max_expert_is})");
            workers.push(GpuWorker {
                device,
                compute_stream: Stream::new(device)?,
                transfer_stream: Stream::new(device)?,
                broadcast_done: HipEvent::new()?,
                compute_done: HipEvent::new()?,
                activation_in: DeviceBuffer::<f32>::alloc(device, hidden_size)?,
                expert_out: DeviceBuffer::<f32>::alloc(device, hidden_size)?,
                scratch_gate: DeviceBuffer::<f32>::alloc(device, max_expert_is)?,
                scratch_up: DeviceBuffer::<f32>::alloc(device, max_expert_is)?,
                scratch_act: DeviceBuffer::<f32>::alloc(device, max_expert_is)?,
                scratch_down: DeviceBuffer::<f32>::alloc(device, hidden_size)?,
                transfer_done: HipEvent::new()?,
                peer_copy_module: Module::load(device, &crate::kernel::kernel_dir().join("peer_copy.hsaco"))?,
                gather_host: PinnedBuffer::<f32>::alloc(hidden_size)?,
            });
        }

        // Gather stream + event on GPU 0
        Device::set_current(DeviceId(0))?;
        let gather_stream = Stream::new(DeviceId(0))?;
        let gather_done = HipEvent::new()?;

        eprintln!("Multi-GPU: {num_devices} devices, P2P enabled");

        Ok(Some(MultiGpuContext {
            num_devices,
            workers,
            gather_stream,
            gather_done,
        }))
    }

    /// Async P2P copy using compute kernel (avoids SDMA PERMISSION_FAULT on RDNA3 PCIe).
    /// Launched on `stream` from the source device. `dst` must be peer-accessible from src_device.
    pub fn peer_copy_async(
        dst: *mut u8, src: *const u8,
        size: usize, peer_copy_module: &Module, stream: &Stream,
    ) -> HipResult<()> {
        let func = peer_copy_module.get_function("peer_copy_kernel")?;
        let threads = 256usize;
        let blocks = (size + threads - 1) / threads;
        let n = size as u64;
        let mut args: [*mut std::ffi::c_void; 3] = [
            &dst as *const _ as *mut std::ffi::c_void,
            &src as *const _ as *mut std::ffi::c_void,
            &n   as *const _ as *mut std::ffi::c_void,
        ];
        func.launch((blocks as u32, 1, 1), (threads as u32, 1, 1), 0, stream, &mut args)
    }

    /// Make a stream wait for an event (cross-stream synchronization).
    pub fn stream_wait_event(stream: &Stream, event: &HipEvent) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe {
            ffi::hipStreamWaitEvent(stream.raw(), event.raw(), 0)
        })
    }
}

/// GPU-initiated MoE work queue: host-mapped memory shared between GPU 0 megakernel
/// and persistent worker kernels on GPUs 1..N-1.
pub struct MoeWorkQueue {
    /// Host-mapped work item (512 bytes). GPU 0 megakernel writes, workers poll.
    pub work_item: MappedHostBuffer<u8>,
    /// Per-GPU output slots (host-mapped for cross-GPU accessibility): [total_gpus * hidden_size].
    pub output_slots: MappedHostBuffer<f32>,
    /// Per-worker shutdown flags (host-mapped).
    pub shutdown_flags: Vec<MappedHostBuffer<u32>>,
    pub seq_counter: u32,
    pub worker_streams: Vec<Stream>,
    pub worker_configs: Vec<DeviceBuffer<u8>>,
    pub worker_modules: Vec<Module>,
    /// GPU 0 worker config (device memory on GPU 0) for megakernel OP_MOE_DISPATCH.
    pub gpu0_config: DeviceBuffer<u8>,
    pub num_workers: usize,
    pub hidden_size: usize,
}

impl MoeWorkQueue {
    /// Initialize and launch persistent worker kernels.
    pub fn init(
        ctx: &MultiGpuContext,
        dist_moe_layers: &[Option<crate::weights::DistributedMoeWeights>],
        hidden_size: usize,
        expert_intermediate_size: usize,
    ) -> HipResult<Self> {
        // Workers run on GPUs 1..N-1. GPU 0 runs the cooperative megakernel
        // and computes its own experts inline via OP_MOE_DISPATCH.
        let num_workers = ctx.num_devices - 1;
        let total_gpus = ctx.num_devices;
        if num_workers == 0 {
            return Err(braidinfer_hip::HipError(1).into());
        }

        // Allocate host-mapped work item (512 bytes)
        Device::set_current(DeviceId(0))?;
        let work_item = MappedHostBuffer::<u8>::alloc(512)?;

        // Per-GPU output slots (host-mapped — accessible from all GPUs via GART)
        let output_slots = MappedHostBuffer::<f32>::alloc(total_gpus * hidden_size)?;

        // Per-worker shutdown flags
        let mut shutdown_flags = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let flag = MappedHostBuffer::<u32>::alloc(1)?;
            shutdown_flags.push(flag);
        }

        // Build worker configs and launch persistent kernels
        let mut worker_streams = Vec::with_capacity(num_workers);
        let mut worker_configs = Vec::with_capacity(num_workers);
        let mut worker_modules = Vec::with_capacity(num_workers);

        let kernel_dir = crate::kernel::kernel_dir();

        // Find the first MoE layer to get expert layout
        let first_moe = dist_moe_layers.iter()
            .find_map(|x| x.as_ref())
            .expect("MoeWorkQueue::init called without MoE layers");

        // Workers are indexed 0..num_workers, mapping to GPUs 1..num_devices-1.
        // Worker w runs on GPU w+1. Worker's my_gpu_id = w (used to index output_slots).
        for w in 0..num_workers {
            let gpu = w + 1; // actual GPU index
            let device = DeviceId(gpu as u32);
            Device::set_current(device)?;

            let gate_up_row_stride = first_moe.gate_up_row_stride;
            let down_groups_per_row = (expert_intermediate_size + 31) / 32;
            let down_row_stride = down_groups_per_row * 20;
            let mut config_bytes = vec![0u8; std::mem::size_of::<MoeWorkerConfigRust>()];
            let config = unsafe { &mut *(config_bytes.as_mut_ptr() as *mut MoeWorkerConfigRust) };
            config.my_gpu_id = gpu as u32; // actual GPU index — indexes into output_slots
            config.num_experts_local = first_moe.expert_buffers[gpu].local_expert_count as u32;
            config.gate_up_row_stride = gate_up_row_stride as u32;
            config.down_row_stride = down_row_stride as u32;
            config.hidden_size = hidden_size as u32;
            config.expert_intermediate_size = expert_intermediate_size as u32;

            // Populate expert entries from all MoE layers (use first layer's layout)
            let buf = &first_moe.expert_buffers[gpu];
            for eid in 0..first_moe.num_experts {
                if first_moe.expert_device[eid] != gpu { continue; }
                let slot = buf.slot_map[eid].expect("expert slot missing");
                let gu_offset = slot * first_moe.gate_up_expert_stride;
                let d_offset = slot * first_moe.down_expert_stride;
                config.entries[eid].global_expert_id = eid as u32;
                config.entries[eid].gate_up_ptr = unsafe { buf.gate_up.as_ptr().add(gu_offset) } as u64;
                config.entries[eid].down_ptr = unsafe { buf.down.as_ptr().add(d_offset) } as u64;
            }

            // Upload config to device memory
            let mut config_buf = DeviceBuffer::<u8>::alloc(device, config_bytes.len())?;
            config_buf.copy_from_host(&config_bytes)?;
            worker_configs.push(config_buf);

            // Create stream and load module
            let stream = Stream::new(device)?;
            let module = Module::load(device, &kernel_dir.join("moe_worker.hsaco"))?;
            worker_streams.push(stream);
            worker_modules.push(module);
        }

        // Launch persistent worker kernels on GPUs 1..N-1
        for w in 0..num_workers {
            let gpu = w + 1;
            let device = DeviceId(gpu as u32);
            Device::set_current(device)?;

            let func = worker_modules[w].get_function("moe_worker_kernel")?;
            let worker = &ctx.workers[gpu];

            let mut wq_ptr = work_item.as_ptr() as *mut std::ffi::c_void;
            let mut sf_ptr = shutdown_flags[w].as_ptr() as *mut std::ffi::c_void;
            let mut cfg_ptr = worker_configs[w].as_ptr() as *mut std::ffi::c_void;
            let mut act_ptr = worker.activation_in.as_ptr() as *mut std::ffi::c_void;
            let mut sg_ptr = worker.scratch_gate.as_ptr() as *mut std::ffi::c_void;
            let mut su_ptr = worker.scratch_up.as_ptr() as *mut std::ffi::c_void;
            let mut sa_ptr = worker.scratch_act.as_ptr() as *mut std::ffi::c_void;
            let mut eo_ptr = worker.expert_out.as_ptr() as *mut std::ffi::c_void;

            let mut args: [*mut std::ffi::c_void; 8] = [
                std::ptr::addr_of_mut!(wq_ptr).cast(),
                std::ptr::addr_of_mut!(sf_ptr).cast(),
                std::ptr::addr_of_mut!(cfg_ptr).cast(),
                std::ptr::addr_of_mut!(act_ptr).cast(),
                std::ptr::addr_of_mut!(sg_ptr).cast(),
                std::ptr::addr_of_mut!(su_ptr).cast(),
                std::ptr::addr_of_mut!(sa_ptr).cast(),
                std::ptr::addr_of_mut!(eo_ptr).cast(),
            ];

            // Use max occupancy blocks for cooperative kernel
            let num_blocks = 192u32; // Same as megakernel
            let shared_mem = 256 * 4; // 256 floats for reduction

            func.launch_cooperative(
                (num_blocks, 1, 1),
                (256, 1, 1),
                shared_mem,
                &worker_streams[w],
                &mut args,
            )?;

            eprintln!("  GPU {gpu}: persistent worker kernel launched ({num_blocks} blocks)");
        }

        // Build GPU 0 worker config (for megakernel's OP_MOE_DISPATCH inline expert compute)
        Device::set_current(DeviceId(0))?;
        let gate_up_row_stride = first_moe.gate_up_row_stride;
        let down_groups_per_row = (expert_intermediate_size + 31) / 32;
        let down_row_stride = down_groups_per_row * 20;
        let mut gpu0_config_bytes = vec![0u8; std::mem::size_of::<MoeWorkerConfigRust>()];
        {
            let config = unsafe { &mut *(gpu0_config_bytes.as_mut_ptr() as *mut MoeWorkerConfigRust) };
            config.my_gpu_id = 0;
            config.num_experts_local = first_moe.expert_buffers[0].local_expert_count as u32;
            config.gate_up_row_stride = gate_up_row_stride as u32;
            config.down_row_stride = down_row_stride as u32;
            config.hidden_size = hidden_size as u32;
            config.expert_intermediate_size = expert_intermediate_size as u32;

            let buf = &first_moe.expert_buffers[0];
            for eid in 0..first_moe.num_experts {
                if first_moe.expert_device[eid] != 0 { continue; }
                let slot = buf.slot_map[eid].expect("GPU 0 expert slot missing");
                let gu_offset = slot * first_moe.gate_up_expert_stride;
                let d_offset = slot * first_moe.down_expert_stride;
                config.entries[eid].global_expert_id = eid as u32;
                config.entries[eid].gate_up_ptr = unsafe { buf.gate_up.as_ptr().add(gu_offset) } as u64;
                config.entries[eid].down_ptr = unsafe { buf.down.as_ptr().add(d_offset) } as u64;
            }
        }
        let mut gpu0_config = DeviceBuffer::<u8>::alloc(DeviceId(0), gpu0_config_bytes.len())?;
        gpu0_config.copy_from_host(&gpu0_config_bytes)?;

        Ok(MoeWorkQueue {
            work_item,
            output_slots,
            shutdown_flags,
            seq_counter: 0,
            worker_streams,
            worker_configs,
            worker_modules,
            gpu0_config,
            num_workers,
            hidden_size,
        })
    }

    /// Get host pointer to work item (for megakernel instruction packing).
    pub fn work_item_ptr(&self) -> *mut u8 {
        self.work_item.host_ptr() as *mut u8
    }

    /// Get GPU 0 worker config device pointer (for OP_MOE_DISPATCH instruction packing).
    pub fn gpu0_config_ptr(&self) -> *const u8 {
        self.gpu0_config.as_ptr()
    }

    /// Get device pointer to output slots (host-mapped, accessible from all GPUs).
    pub fn output_slots_ptr(&self) -> *mut f32 {
        self.output_slots.device_ptr() as *mut f32
    }

    /// Increment and return next sequence number.
    pub fn next_seq(&mut self) -> u32 {
        self.seq_counter += 1;
        self.seq_counter
    }
}

impl Drop for MoeWorkQueue {
    fn drop(&mut self) {
        // Signal all workers to shut down
        for flag in &self.shutdown_flags {
            unsafe { std::ptr::write_volatile(flag.host_ptr(), 1u32); }
        }
        // Wait for workers to exit (worker w runs on GPU w+1)
        for (w, stream) in self.worker_streams.iter().enumerate() {
            let _ = Device::set_current(DeviceId((w + 1) as u32));
            let _ = stream.synchronize();
        }
    }
}

/// Rust-side mirror of MoeWorkerConfig from moe_work_queue.h.
#[repr(C)]
struct MoeWorkerConfigRust {
    my_gpu_id: u32,
    num_experts_local: u32,
    gate_up_row_stride: u32,
    down_row_stride: u32,
    hidden_size: u32,
    expert_intermediate_size: u32,
    _pad: [u32; 2],
    entries: [MoeExpertEntryRust; 256],
}

/// Rust-side mirror of MoeWorkItem from moe_work_queue.h.
#[repr(C)]
pub struct MoeWorkItemRust {
    pub seq_num: u32,
    pub layer_idx: u32,
    pub num_active: u32,
    pub hidden_size: u32,
    pub expert_intermediate_size: u32,
    pub has_gate_proj: u32,
    pub num_workers: u32,
    pub _pad0: u32,
    pub expert_ids: [i32; 32],
    pub expert_weights: [f32; 32],
    pub activation_ptr: u64,
    pub output_slots_ptr: u64,
    pub ack_flags: [u32; 8],
}

#[repr(C)]
struct MoeExpertEntryRust {
    global_expert_id: u32,
    _pad: u32,
    gate_up_ptr: u64,
    down_ptr: u64,
}
