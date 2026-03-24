use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use std::ffi::c_void;
use std::path::PathBuf;

fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

pub struct RmsNormKernel {
    module: Module,
    device: DeviceId,
}

impl RmsNormKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        braidinfer_hip::Device::set_current(device)?;
        let path = kernel_dir().join("rmsnorm.hsaco");
        let module = Module::load(&path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        num_rows: u32,
        hidden_size: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("rmsnorm_f32")?;

        let mut out_ptr = output.as_mut_ptr() as *mut c_void;
        let mut in_ptr = input.as_ptr() as *mut c_void;
        let mut w_ptr = weight.as_ptr() as *mut c_void;
        let mut hs = hidden_size as i32;
        let mut ep = eps;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
        ];

        let block_size = 256u32.min(hidden_size);
        func.launch(
            (num_rows, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }
}
