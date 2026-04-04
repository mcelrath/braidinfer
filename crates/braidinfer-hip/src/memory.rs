use crate::{error, ffi, HipResult};
use braidinfer_core::types::DeviceId;
use std::marker::PhantomData;
use std::ptr;

/// GPU device memory buffer. Encodes device ID to prevent cross-device misuse.
/// Deliberately NOT Send/Sync — GPU pointers are device-local.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    device: DeviceId,
    _marker: PhantomData<*mut T>, // !Send, !Sync
}

impl<T> DeviceBuffer<T> {
    pub fn alloc(device: DeviceId, len: usize) -> HipResult<Self> {
        crate::device::Device::set_current(device)?;
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe { ffi::hipMalloc(&mut ptr, size) })?;
        Ok(DeviceBuffer {
            ptr: ptr.cast(),
            len,
            device,
            _marker: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> HipResult<()> {
        assert!(data.len() <= self.len, "source larger than buffer");
        let size = data.len() * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpy(
                self.ptr.cast(),
                data.as_ptr().cast(),
                size,
                ffi::hipMemcpyHostToDevice,
            )
        })
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> HipResult<()> {
        assert!(data.len() >= self.len, "destination smaller than buffer");
        let size = self.len * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr().cast(),
                self.ptr.cast(),
                size,
                ffi::hipMemcpyDeviceToHost,
            )
        })
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if crate::device::Device::set_current(self.device).is_err() {
                // Accept leak rather than free on wrong device
                eprintln!(
                    "braidinfer: leaked {}B on {:?} (set_current failed)",
                    self.size_bytes(),
                    self.device
                );
                return;
            }
            let err = unsafe { ffi::hipFree(self.ptr.cast()) };
            if err != 0 {
                eprintln!(
                    "braidinfer: hipFree failed (error {}) for {}B on {:?}",
                    err,
                    self.size_bytes(),
                    self.device
                );
            }
        }
    }
}

/// Pinned host memory buffer. Required for hipMemcpyAsync.
/// Safe to access from any thread (IS Send+Sync).
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// Pinned memory is host memory, safe to share across threads
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    pub fn alloc(len: usize) -> HipResult<Self> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe {
            ffi::hipHostMalloc(&mut ptr, size, ffi::hipHostMallocDefault)
        })?;
        Ok(PinnedBuffer {
            ptr: ptr.cast(),
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// # Safety
    /// Caller must ensure no other references to this buffer exist.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Mapped host buffer: accessible from both CPU (host_ptr) and GPU (device_ptr).
/// Uses hipHostMallocMapped which maps the allocation into the GPU address space.
/// Reads/writes are coherent: GPU accesses go through GART (MTYPE_UC, no L2 caching).
/// Use for barrier flags and small shared state between CPU and a running GPU kernel.
pub struct MappedHostBuffer<T> {
    host_ptr: *mut T,
    device_ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// Mapped memory is host memory, safe to share across threads
unsafe impl<T: Send> Send for MappedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for MappedHostBuffer<T> {}

impl<T> MappedHostBuffer<T> {
    pub fn alloc(len: usize) -> HipResult<Self> {
        let size = len * std::mem::size_of::<T>();
        let mut host_ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe {
            ffi::hipHostMalloc(&mut host_ptr, size, ffi::hipHostMallocMapped)
        })?;
        let mut device_ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe {
            ffi::hipHostGetDevicePointer(&mut device_ptr, host_ptr, 0)
        })?;
        // Zero-initialize: hipHostMalloc does not guarantee zeroed memory.
        let typed_ptr = host_ptr.cast::<T>();
        unsafe { ptr::write_bytes(typed_ptr, 0, len); }
        Ok(MappedHostBuffer {
            host_ptr: typed_ptr,
            device_ptr: device_ptr.cast(),
            len,
            _marker: PhantomData,
        })
    }

    /// CPU-side pointer. Use for direct CPU reads/writes.
    pub fn host_ptr(&self) -> *mut T {
        self.host_ptr
    }

    /// GPU-side pointer. Pass this to kernels via instruction slots.
    pub fn device_ptr(&self) -> *const T {
        self.device_ptr as *const T
    }

    /// GPU-side pointer alias (matches DeviceBuffer::as_ptr for uniform kernel code).
    pub fn as_ptr(&self) -> *const T {
        self.device_ptr as *const T
    }

    /// Mutable GPU-side pointer alias (matches DeviceBuffer::as_mut_ptr).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.device_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for MappedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            let err = unsafe { ffi::hipHostFree(self.host_ptr.cast()) };
            if err != 0 {
                eprintln!(
                    "braidinfer: hipHostFree failed (error {}) for mapped buffer",
                    err,
                );
            }
        }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let err = unsafe { ffi::hipHostFree(self.ptr.cast()) };
            if err != 0 {
                eprintln!(
                    "braidinfer: hipHostFree failed (error {}) for {}B pinned buffer",
                    err,
                    self.len * std::mem::size_of::<T>()
                );
            }
        }
    }
}
