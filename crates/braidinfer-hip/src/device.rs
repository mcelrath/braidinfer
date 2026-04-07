use crate::{HipResult, error, ffi};
use braidinfer_core::types::DeviceId;

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
