use crate::{HipResult, error, ffi};
use braidinfer_core::types::DeviceId;

/// RAII guard that saves the current HIP device on construction, switches to
/// a target device, and restores the saved device on drop.
///
/// `DeviceGuard` is `!Send` because HIP device context is thread-local; moving
/// the guard across threads would restore the wrong thread's context.
///
/// Drop never panics: errors from `hipSetDevice` on drop are silently ignored
/// (the saved device may have been removed in tests, and panicking in drop is
/// UB during stack unwinding).
pub struct DeviceGuard {
    saved: ffi::hipDevice_t,
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl DeviceGuard {
    /// Save current device and switch to `target`. Returns an error if either
    /// `hipGetDevice` or `hipSetDevice(target)` fails.
    pub fn switch_to(target: DeviceId) -> HipResult<Self> {
        let mut current = 0i32;
        error::check(unsafe { ffi::hipGetDevice(&mut current) })?;
        error::check(unsafe { ffi::hipSetDevice(target.0 as i32) })?;
        Ok(Self {
            saved: current as ffi::hipDevice_t,
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        let _ = unsafe { ffi::hipSetDevice(self.saved as i32) };
    }
}

pub struct Device {
    pub id: DeviceId,
}

impl Device {
    pub fn count() -> HipResult<u32> {
        let mut count = 0i32;
        error::check(unsafe { ffi::hipGetDeviceCount(&mut count) })?;
        Ok(count as u32)
    }

    pub fn set_current(id: DeviceId) -> HipResult<()> {
        error::check(unsafe { ffi::hipSetDevice(id.0 as i32) })
    }

    pub fn current() -> HipResult<DeviceId> {
        let mut id = 0i32;
        error::check(unsafe { ffi::hipGetDevice(&mut id) })?;
        Ok(DeviceId(id as u32))
    }

    pub fn synchronize() -> HipResult<()> {
        error::check(unsafe { ffi::hipDeviceSynchronize() })
    }

    pub fn new(id: DeviceId) -> HipResult<Self> {
        let count = Self::count()?;
        if id.0 >= count {
            return Err(crate::HipError(ffi::hipErrorInvalidDevice));
        }
        Ok(Device { id })
    }
}
