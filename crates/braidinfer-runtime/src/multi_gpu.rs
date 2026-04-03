//! Multi-GPU context: P2P setup, per-device streams and events for expert parallel dispatch.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_hip::memory::DeviceBuffer;

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
                    // Ignore "already enabled" errors
                    if rc != 0 && rc != ffi::hipErrorInvalidDevice {
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

    /// Async P2P copy: src buffer on src_device → dst buffer on dst_device.
    pub fn memcpy_peer_async(
        dst: *mut std::ffi::c_void, dst_device: DeviceId,
        src: *const std::ffi::c_void, src_device: DeviceId,
        size: usize, stream: &Stream,
    ) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe {
            ffi::hipMemcpyPeerAsync(
                dst, dst_device.0 as i32,
                src, src_device.0 as i32,
                size, stream.raw(),
            )
        })
    }

    /// Make a stream wait for an event (cross-stream synchronization).
    pub fn stream_wait_event(stream: &Stream, event: &HipEvent) -> HipResult<()> {
        braidinfer_hip::error::check(unsafe {
            ffi::hipStreamWaitEvent(stream.raw(), event.raw(), 0)
        })
    }
}
