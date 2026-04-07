use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::ffi::c_void;

use super::kernel_dir;

pub struct LinearProjKernel {
    module: Module,
    device: DeviceId,
}

impl LinearProjKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("linear_proj.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        out_dim: u32,
        in_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("linear_proj_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];

        let block_size = 256u32.min(in_dim);
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }

    /// Raw pointer version for MoE expert sub-buffer access.
    pub fn forward_ptr(
        &self,
        output: *mut f32,
        weight: *const u16,
        input: *const f32,
        out_dim: u32,
        in_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("linear_proj_f32")?;
        let mut out_ptr: *mut c_void = output.cast();
        let mut w_ptr: *const c_void = weight.cast();
        let mut in_ptr: *const c_void = input.cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;
        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];
        let block_size = 256u32.min(in_dim);
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }

    /// Forward pass with PackedWeights (dispatches by format).
    pub fn forward_packed(
        &self,
        output: &mut DeviceBuffer<f32>,
        weight: &crate::model::PackedWeights,
        input: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> HipResult<()> {
        let out_dim = weight.out_dim as u32;
        let in_dim = weight.in_dim as u32;

        let func_name = match weight.format {
            crate::model::WeightFormat::Bf16 => "linear_proj_f32",
            crate::model::WeightFormat::Rnf4G128 => "linear_proj_rnf4_g128",
            crate::model::WeightFormat::PcG32Q4 => "linear_proj_pcg32_q4",
        };
        let func = self.module.get_function(func_name)?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut w_ptr: *const c_void = weight.data.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];

        let block_size = 256u32;
        let rows_per_block = if func_name == "linear_proj_rnf4_g128" {
            4u32
        } else {
            1
        };
        let grid = (out_dim + rows_per_block - 1) / rows_per_block;
        // Shared memory: input cache (in_dim * 4 bytes) + wave reduction (rows_per_block * 2 * 4 bytes)
        let shared_bytes = (in_dim as u32) * 4 + rows_per_block * 2 * 4;
        func.launch(
            (grid, 1, 1),
            (block_size, 1, 1),
            shared_bytes,
            stream,
            &mut args,
        )
    }

    /// Raw pointer version for quantized weights with byte offset (MoE experts).
    pub fn forward_packed_ptr(
        &self,
        output: *mut f32,
        weight: *const u8,
        input: *const f32,
        out_dim: u32,
        in_dim: u32,
        func_name: &str,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function(func_name)?;
        let mut out_ptr: *mut c_void = output.cast();
        let mut w_ptr: *const c_void = weight.cast();
        let mut in_ptr: *const c_void = input.cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;
        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];
        let block_size = 256u32;
        let rows_per_block = if func_name == "linear_proj_rnf4_g128" {
            4u32
        } else {
            1
        };
        let grid = (out_dim + rows_per_block - 1) / rows_per_block;
        let shared_bytes = in_dim * 4 + rows_per_block * 2 * 4;
        func.launch(
            (grid, 1, 1),
            (block_size, 1, 1),
            shared_bytes,
            stream,
            &mut args,
        )
    }
}

pub struct MoeGateKernel {
    module: Module,
    _device: DeviceId,
}

impl MoeGateKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("moe_gate.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self {
            module,
            _device: device,
        })
    }

    /// Run GPU-side top-k selection + weight computation.
    /// gate_mode: 0=softmax, 1=norm_topk, 2=sigmoid
    /// correction_bias_ptr: raw device pointer, or null if no bias.
    pub fn forward(
        &self,
        scores: &DeviceBuffer<f32>,
        expert_ids_ptr: *mut i32,
        expert_weights_ptr: *mut f32,
        correction_bias_ptr: *const f32,
        num_experts: u32,
        k: u32,
        gate_mode: u32,
        routed_scaling_factor: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("moe_gate_topk")?;

        let mut s_ptr: *const c_void = scores.as_ptr().cast();
        let mut id_ptr: *mut c_void = expert_ids_ptr.cast();
        let mut w_ptr: *mut c_void = expert_weights_ptr.cast();
        let mut bias_ptr: *const c_void = correction_bias_ptr.cast();
        let mut ne = num_experts as i32;
        let mut kk = k as i32;
        let mut gm = gate_mode as i32;
        let mut rsf = routed_scaling_factor;

        let mut args: [*mut c_void; 8] = [
            std::ptr::addr_of_mut!(s_ptr).cast(),
            std::ptr::addr_of_mut!(id_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(bias_ptr).cast(),
            std::ptr::addr_of_mut!(ne).cast(),
            std::ptr::addr_of_mut!(kk).cast(),
            std::ptr::addr_of_mut!(gm).cast(),
            std::ptr::addr_of_mut!(rsf).cast(),
        ];

        // shared mem: 1024 floats (512 for selection + 512 for raw scores)
        let shared_mem = 1024 * 4;
        func.launch((1, 1, 1), (256, 1, 1), shared_mem, stream, &mut args)
    }
}
