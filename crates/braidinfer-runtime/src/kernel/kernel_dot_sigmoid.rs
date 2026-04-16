use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::ffi::c_void;

use super::kernel_dir;

pub struct DotSigmoidScaleAddKernel {
    module: Module,
    device: DeviceId,
}

impl DotSigmoidScaleAddKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("dot_sigmoid_scale_add.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// output[i] += sigmoid(dot(bf16_weight, f32_input)) * src[i]
    /// Single-block fused kernel: eliminates CPU round-trip for shared expert gate.
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        src: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(src.device(), self.device);
        assert_eq!(input.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("dot_sigmoid_scale_add_bf16_f32")?;
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut src_ptr: *const c_void = src.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(src_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        const BLOCK_SIZE: u32 = 256;
        let num_warps = (BLOCK_SIZE + 31) / 32;
        let shmem = num_warps * std::mem::size_of::<f32>() as u32;
        func.launch((1, 1, 1), (BLOCK_SIZE, 1, 1), shmem, stream, &mut args)
    }
}
