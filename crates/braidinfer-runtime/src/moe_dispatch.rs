//! MoE dispatch utilities: per-GPU kernel modules and single-projection dispatch.
//! Used by the P2P megakernel path (decode_step_p2p) for worker GPU expert execution.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;

use crate::kernel::{LinearProjKernel, ResidualAddKernel, SiluMulKernel};
use crate::quant::WeightFormat;

/// Dispatch a single linear projection for a packed (non-bf16) or bf16 expert weight slice.
///
/// For quantized formats (PcG32Q4, Rnf4G128) uses `forward_packed_ptr`.
/// For bf16 casts the raw byte pointer to u16 and uses `forward_ptr`.
pub(crate) fn dispatch_proj(
    kernel: &crate::kernel::LinearProjKernel,
    output: *mut f32,
    weight_bytes: *const u8,
    input: *const f32,
    out_dim: u32,
    in_dim: u32,
    fmt: WeightFormat,
    stream: &Stream,
) -> HipResult<()> {
    match fmt {
        WeightFormat::PcG32Q4 => kernel.forward_packed_ptr(
            output,
            weight_bytes,
            input,
            out_dim,
            in_dim,
            "linear_proj_pcg32_q4",
            stream,
        ),
        WeightFormat::Rnf4G128 => kernel.forward_packed_ptr(
            output,
            weight_bytes,
            input,
            out_dim,
            in_dim,
            "linear_proj_rnf4_g128",
            stream,
        ),
        WeightFormat::Bf16 => kernel.forward_ptr(
            output,
            weight_bytes as *const u16,
            input,
            out_dim,
            in_dim,
            stream,
        ),
    }
}

/// Per-GPU kernel modules for expert FFN execution.
pub struct WorkerKernels {
    pub device: DeviceId,
    pub linear_proj: LinearProjKernel,
    pub silu_mul: SiluMulKernel,
    pub residual_add: ResidualAddKernel,
}

impl WorkerKernels {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        Device::set_current(device)?;
        Ok(WorkerKernels {
            device,
            linear_proj: LinearProjKernel::load(device)?,
            silu_mul: SiluMulKernel::load(device)?,
            residual_add: ResidualAddKernel::load(device)?,
        })
    }
}
