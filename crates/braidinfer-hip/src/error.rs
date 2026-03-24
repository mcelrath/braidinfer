use crate::ffi;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError(pub u32);

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self.0 {
            ffi::hipErrorInvalidValue => "invalid value",
            ffi::hipErrorOutOfMemory => "out of memory",
            ffi::hipErrorNotInitialized => "not initialized",
            ffi::hipErrorInvalidDevice => "invalid device",
            other => return write!(f, "HIP error {other}"),
        };
        write!(f, "HIP error: {msg}")
    }
}

impl std::error::Error for HipError {}

pub type HipResult<T> = Result<T, HipError>;

#[inline]
pub fn check(code: u32) -> HipResult<()> {
    if code == ffi::hipSuccess {
        Ok(())
    } else {
        Err(HipError(code))
    }
}
