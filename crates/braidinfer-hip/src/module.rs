use crate::{error, ffi, HipResult};
use std::ffi::{c_void, CString};
use std::path::Path;

/// A loaded HIP module (compiled .hsaco or .co binary).
pub struct Module {
    raw: ffi::hipModule_t,
}

impl Module {
    pub fn load(path: &Path) -> HipResult<Self> {
        let path_str = CString::new(path.to_str().expect("invalid path")).unwrap();
        let mut raw = std::ptr::null_mut();
        error::check(unsafe { ffi::hipModuleLoad(&mut raw, path_str.as_ptr()) })?;
        Ok(Module { raw })
    }

    pub fn get_function(&self, name: &str) -> HipResult<Function> {
        let name_c = CString::new(name).unwrap();
        let mut func = std::ptr::null_mut();
        error::check(unsafe {
            ffi::hipModuleGetFunction(&mut func, self.raw, name_c.as_ptr())
        })?;
        Ok(Function { raw: func })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::hipModuleUnload(self.raw) };
        }
    }
}

pub struct Function {
    raw: ffi::hipFunction_t,
}

impl Function {
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
}
