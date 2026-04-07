use crate::{HipError, HipResult, error, ffi};
use braidinfer_core::types::DeviceId;
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::path::Path;

/// A loaded HIP module (compiled .co/.hsaco binary). Tied to the device it was loaded on.
pub struct Module {
    raw: ffi::hipModule_t,
    device: DeviceId,
}

impl Module {
    pub fn load(device: DeviceId, path: &Path) -> HipResult<Self> {
        crate::device::Device::set_current(device)?;
        let path_str = path.to_str().ok_or(HipError(ffi::hipErrorInvalidValue))?;
        let path_c = CString::new(path_str).map_err(|_| HipError(ffi::hipErrorInvalidValue))?;
        let mut raw = std::ptr::null_mut();
        error::check(unsafe { ffi::hipModuleLoad(&mut raw, path_c.as_ptr()) })?;
        Ok(Module { raw, device })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn get_function(&self, name: &str) -> HipResult<Function<'_>> {
        let name_c = CString::new(name).map_err(|_| HipError(ffi::hipErrorInvalidValue))?;
        let mut func = std::ptr::null_mut();
        error::check(unsafe { ffi::hipModuleGetFunction(&mut func, self.raw, name_c.as_ptr()) })?;
        Ok(Function {
            raw: func,
            _module: PhantomData,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = crate::device::Device::set_current(self.device);
            unsafe { ffi::hipModuleUnload(self.raw) };
        }
    }
}

/// A kernel function handle. Borrows the Module it was loaded from —
/// cannot outlive it.
pub struct Function<'module> {
    raw: ffi::hipFunction_t,
    _module: PhantomData<&'module Module>,
}

impl Function<'_> {
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &crate::stream::Stream,
        args: &mut [*mut c_void],
    ) -> HipResult<()> {
        error::check(unsafe {
            ffi::hipModuleLaunchKernel(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem,
                stream.raw(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }

    pub fn launch_cooperative(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &crate::stream::Stream,
        args: &mut [*mut c_void],
    ) -> HipResult<()> {
        error::check(unsafe {
            ffi::hipModuleLaunchCooperativeKernel(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem,
                stream.raw(),
                args.as_mut_ptr(),
            )
        })
    }

    pub fn max_active_blocks_per_sm(
        &self,
        block_size: u32,
        dynamic_shared_mem: usize,
    ) -> HipResult<i32> {
        let mut num_blocks: std::ffi::c_int = 0;
        error::check(unsafe {
            ffi::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut num_blocks,
                self.raw,
                block_size as std::ffi::c_int,
                dynamic_shared_mem,
            )
        })?;
        Ok(num_blocks)
    }
}
