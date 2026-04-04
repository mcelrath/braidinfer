//! Multi-GPU context: P2P setup, per-device streams and events for expert parallel dispatch.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_hip::memory::{DeviceBuffer, PinnedBuffer};

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
