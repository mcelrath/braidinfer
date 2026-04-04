use crate::{error, ffi, HipResult};
use braidinfer_core::types::DeviceId;

/// HIP stream. Pinned to the device it was created on.
/// NOT Send — must stay on the OS thread that created it.
pub struct Stream {
    raw: ffi::hipStream_t,
    device: DeviceId,
}

impl Stream {
    pub fn new(device: DeviceId) -> HipResult<Self> {
        crate::device::Device::set_current(device)?;
        let mut raw = std::ptr::null_mut();
        error::check(unsafe { ffi::hipStreamCreate(&mut raw) })?;
        Ok(Stream { raw, device })
    }

    pub fn raw(&self) -> ffi::hipStream_t {
        self.raw
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn synchronize(&self) -> HipResult<()> {
        error::check(unsafe { ffi::hipStreamSynchronize(self.raw) })
    }

    /// Non-blocking check: returns true if all work in the stream is complete.
    pub fn is_idle(&self) -> bool {
        let err = unsafe { ffi::hipStreamQuery(self.raw) };
        err == ffi::hipSuccess
    }

    /// Async H2D copy from pinned host memory. Requires PinnedBuffer to guarantee
    /// true async transfer — regular heap memory silently degrades to synchronous.
    pub fn copy_to_device_async<T>(
        &self,
        dst: &mut crate::memory::DeviceBuffer<T>,
        src: &crate::memory::PinnedBuffer<T>,
    ) -> HipResult<()> {
        assert_eq!(dst.device(), self.device, "stream/buffer device mismatch");
        assert!(src.len() <= dst.len(), "source larger than destination");
        let size = src.len() * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size,
                ffi::hipMemcpyHostToDevice,
                self.raw,
            )
        })
    }

    /// Async D2H copy to pinned host memory.
    pub fn copy_to_host_async<T>(
        &self,
        dst: &mut crate::memory::PinnedBuffer<T>,
        src: &crate::memory::DeviceBuffer<T>,
    ) -> HipResult<()> {
        assert_eq!(src.device(), self.device, "stream/buffer device mismatch");
        assert!(src.len() <= dst.len(), "source larger than destination");
        let size = src.len() * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size,
                ffi::hipMemcpyDeviceToHost,
                self.raw,
            )
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = crate::device::Device::set_current(self.device);
            unsafe { ffi::hipStreamDestroy(self.raw) };
        }
    }
}
