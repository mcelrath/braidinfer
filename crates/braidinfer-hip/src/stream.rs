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

    pub fn copy_async<T>(
        &self,
        dst: &mut crate::memory::DeviceBuffer<T>,
        src: &[T],
    ) -> HipResult<()> {
        assert_eq!(dst.device(), self.device, "stream/buffer device mismatch");
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
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = crate::device::Device::set_current(self.device);
            unsafe { ffi::hipStreamDestroy(self.raw) };
        }
    }
}
