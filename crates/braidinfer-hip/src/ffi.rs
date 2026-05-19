#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type hipDevice_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;
pub type hipEvent_t = *mut c_void;

/// Mirror of HIP's hipPointerAttribute_t struct (per hip_runtime_api.h:257).
/// type: hipMemoryType (0=Unregistered, 1=Host, 2=Device, 3=Managed)
/// allocationFlags: the flags passed at allocation time (e.g.
/// hipDeviceMallocUncached=0x3 for DeviceBuffer::alloc_uncached).
#[repr(C)]
#[derive(Default)]
pub struct HipPointerAttribute {
    pub mem_type: c_uint,    // hipMemoryType enum value
    pub device: c_int,
    pub device_pointer: *mut c_void,
    pub host_pointer: *mut c_void,
    pub is_managed: c_int,
    pub allocation_flags: c_uint,
}

pub type hipError_t = c_uint;

pub const hipSuccess: hipError_t = 0;
pub const hipErrorInvalidValue: hipError_t = 1;
pub const hipErrorOutOfMemory: hipError_t = 2;
pub const hipErrorNotInitialized: hipError_t = 3;
pub const hipErrorInvalidDevice: hipError_t = 101;
pub const hipErrorPeerAccessAlreadyEnabled: hipError_t = 704;

pub const hipMemcpyHostToDevice: c_uint = 1;
pub const hipMemcpyDeviceToHost: c_uint = 2;
pub const hipMemcpyDeviceToDevice: c_uint = 3;

pub const hipHostMallocDefault: c_uint = 0;
pub const hipHostMallocPortable: c_uint = 1;
pub const hipHostMallocMapped: c_uint = 2;
/// Forces fine-grained coherent allocation (CPU writes immediately visible
/// to GPU and vice versa, bypassing GPU L2 caching). Used to test the
/// L2-staleness hypothesis for the multi-GPU worker-poll wedge (kb
/// phase-5-prime-zuk-q9z-2026-05-12): default `Mapped` may be MTYPE_NC
/// on the worker side under some ROCm/firmware configurations, in which
/// case the worker's volatile poll of `queue->seq_num` reads a stale
/// L2-cached line indefinitely.
pub const hipHostMallocCoherent: c_uint = 0x40000000;

unsafe extern "C" {
    // Device management
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device: c_int) -> hipError_t;
    pub fn hipGetDevice(device: *mut c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipDeviceReset() -> hipError_t;
    pub fn hipDeviceGetAttribute(pi: *mut c_int, attr: c_int, device: c_int) -> hipError_t;

    // Memory management
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipExtMallocWithFlags(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_uint)
    -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemset(ptr: *mut c_void, value: c_int, size: usize) -> hipError_t;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipHostRegister(host_ptr: *mut c_void, size_bytes: usize, flags: c_uint) -> hipError_t;
    pub fn hipHostUnregister(host_ptr: *mut c_void) -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipPointerGetAttributes(
        attributes: *mut HipPointerAttribute,
        ptr: *const c_void,
    ) -> hipError_t;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    // Stream management
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: u32) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    /// Non-blocking check: hipSuccess if all work complete, hipErrorNotReady if still running.
    pub fn hipStreamQuery(stream: hipStream_t) -> hipError_t;

    // Mapped host memory
    pub fn hipHostGetDevicePointer(
        device_ptr: *mut *mut c_void,
        host_ptr: *mut c_void,
        flags: c_uint,
    ) -> hipError_t;

    // Module / kernel management
    pub fn hipModuleLoad(module: *mut hipModule_t, fname: *const c_char) -> hipError_t;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        num_blocks: *mut c_int,
        func: hipFunction_t,
        block_size: c_int,
        dyn_shared_mem_per_blk: usize,
    ) -> hipError_t;
    // Cooperative kernel launch (module-loaded kernels)
    pub fn hipModuleLaunchCooperativeKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
    ) -> hipError_t;

    // Event management
    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, stop: hipEvent_t) -> hipError_t;

    // Device info
    pub fn hipDeviceGetPCIBusId(
        pci_bus_id: *mut std::os::raw::c_char,
        len: c_int,
        device: c_int,
    ) -> hipError_t;

    // Peer access
    pub fn hipDeviceEnablePeerAccess(peer_device: c_int, flags: c_uint) -> hipError_t;
    pub fn hipDeviceCanAccessPeer(
        can_access: *mut c_int,
        device: c_int,
        peer_device: c_int,
    ) -> hipError_t;
    pub fn hipMemcpyPeer(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: usize,
    ) -> hipError_t;
    pub fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    // Stream-event synchronization
    pub fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> hipError_t;
}
