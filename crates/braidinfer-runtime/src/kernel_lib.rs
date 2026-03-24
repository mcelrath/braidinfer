use braidinfer_core::types::DeviceId;
use braidinfer_hip::module::Module;
use braidinfer_hip::HipResult;
use std::collections::HashMap;
use std::path::PathBuf;

fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

/// Owns all loaded kernel modules for a single device.
/// Prevents re-loading the same .hsaco and provides a single point
/// for kernel lookup. Prepares for the megakernel (one module, many functions).
pub struct KernelLibrary {
    device: DeviceId,
    modules: HashMap<String, Module>,
}

impl KernelLibrary {
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            modules: HashMap::new(),
        }
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn load_module(&mut self, name: &str) -> HipResult<()> {
        if self.modules.contains_key(name) {
            return Ok(());
        }
        let path = kernel_dir().join(format!("{name}.hsaco"));
        let module = Module::load(self.device, &path)?;
        self.modules.insert(name.to_string(), module);
        Ok(())
    }

    pub fn module(&self, name: &str) -> Option<&Module> {
        self.modules.get(name)
    }

    pub fn load_all_kernels(&mut self) -> HipResult<()> {
        let kernel_names = ["rmsnorm", "linear_proj", "silu_mul", "residual_add", "embedding", "lm_head", "gdn_recurrent_step", "mrope", "gqa_attention"];
        for name in &kernel_names {
            self.load_module(name)?;
        }
        Ok(())
    }
}
