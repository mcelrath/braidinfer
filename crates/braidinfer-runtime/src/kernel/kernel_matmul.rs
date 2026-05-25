use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::ffi::c_void;

use super::kernel_dir;

/// WMMA-accelerated batched GEMM for gfx1100 RDNA3 (wave32).
/// Supports bf16 activations × bf16 weights → f32 output,
/// and f32 activations × RNF4G128 weights → f32 output (fused dequant).
pub struct WmmaGemmKernel {
    module_bf16: Module,
    module_rnf4: Module,
    device: DeviceId,
}

impl WmmaGemmKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let bf16_path = kernel_dir().join("wmma_gemm_bf16.hsaco");
        let rnf4_path = kernel_dir().join("wmma_gemm_rnf4g128.hsaco");
        let module_bf16 = Module::load(device, &bf16_path)?;
        let module_rnf4 = Module::load(device, &rnf4_path)?;
        Ok(Self { module_bf16, module_rnf4, device })
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// C = A @ B^T, A[M,K] bf16, B[N,K] bf16, C[M,N] f32.
    /// K must be a multiple of 16.
    pub fn gemm_bf16(
        &self,
        c: &mut DeviceBuffer<f32>,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        m: u32,
        n: u32,
        k: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module_bf16.get_function("wmma_gemm_bf16")?;
        let mut c_ptr: *mut c_void = c.as_mut_ptr().cast();
        let mut a_ptr: *const c_void = a.as_ptr().cast();
        let mut b_ptr: *const c_void = b.as_ptr().cast();
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut args: [*mut c_void; 6] = [
            std::ptr::addr_of_mut!(c_ptr).cast(),
            std::ptr::addr_of_mut!(a_ptr).cast(),
            std::ptr::addr_of_mut!(b_ptr).cast(),
            std::ptr::addr_of_mut!(mm).cast(),
            std::ptr::addr_of_mut!(nn).cast(),
            std::ptr::addr_of_mut!(kk).cast(),
        ];
        let grid_x = (m + 15) / 16;
        let grid_y = (n + 15) / 16;
        func.launch((grid_x, grid_y, 1), (32, 1, 1), 0, stream, &mut args)
    }

    /// C = A @ B^T, A[M,K] f32, B[N,K] RNF4G128, C[M,N] f32.
    /// K must be a multiple of 128 (group size).
    pub fn gemm_rnf4g128(
        &self,
        c: &mut DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<u8>,
        m: u32,
        n: u32,
        k: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module_rnf4.get_function("wmma_gemm_rnf4g128")?;
        let mut c_ptr: *mut c_void = c.as_mut_ptr().cast();
        let mut a_ptr: *const c_void = a.as_ptr().cast();
        let mut b_ptr: *const c_void = b.as_ptr().cast();
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut args: [*mut c_void; 6] = [
            std::ptr::addr_of_mut!(c_ptr).cast(),
            std::ptr::addr_of_mut!(a_ptr).cast(),
            std::ptr::addr_of_mut!(b_ptr).cast(),
            std::ptr::addr_of_mut!(mm).cast(),
            std::ptr::addr_of_mut!(nn).cast(),
            std::ptr::addr_of_mut!(kk).cast(),
        ];
        let grid_x = (m + 15) / 16;
        let grid_y = (n + 15) / 16;
        func.launch((grid_x, grid_y, 1), (32, 1, 1), 0, stream, &mut args)
    }
}

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
        weight: &crate::weights::PackedWeights,
        input: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> HipResult<()> {
        let out_dim = weight.out_dim as u32;
        let in_dim = weight.in_dim as u32;

        let func_name = match weight.format {
            crate::weights::WeightFormat::Bf16 => "linear_proj_f32",
            crate::weights::WeightFormat::Rnf4G128 => "linear_proj_rnf4_g128",
            crate::weights::WeightFormat::PcG32Q4 => "linear_proj_pcg32_q4",
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
