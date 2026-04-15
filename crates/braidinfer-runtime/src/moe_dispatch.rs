//! Event-based multi-GPU MoE expert dispatch.
//! CPU reads expert_ids from MappedHostBuffer (no hipMemcpy on GPU 0), broadcasts
//! activation to worker GPUs, launches expert FFN in parallel, gathers results back.
//!
//! KEY CONSTRAINT: During OP_BARRIER, the cooperative megakernel occupies ALL GPU 0 SMs.
//! ANY hipMemcpy on GPU 0 or any GPU 0 kernel launch will deadlock.
//! Solution: all CPU↔GPU communication uses MappedHostBuffer (GART/pinned memory).
//! Worker GPU (GPU 1..N-1) compute is fine; only GPU 0 is off-limits.
//! Experts are distributed only to GPUs 1..N-1 (see distribute_moe_weights_from_*).

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{HipResult, ffi};

use crate::kernel::{LinearProjKernel, ResidualAddKernel, SiluMulKernel};
use crate::multi_gpu::MultiGpuContext;
use crate::quant::WeightFormat;
use crate::weights::DistributedMoeWeights;

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

/// Compute-only dispatch for GPUs 1..N-1: launches expert FFNs and records compute_done.
/// Does NOT D2H gather or CPU-accumulate. Caller is responsible for syncing and on-GPU reduction.
/// Returns bitmask of GPUs with active experts (bit i set ↔ GPU i has experts).
pub fn dispatch_moe_layer_kbk(
    ctx: &mut MultiGpuContext,
    worker_kernels: &[WorkerKernels],
    moe: &DistributedMoeWeights,
    _normed_stage: &MappedHostBuffer<f32>,
    moe_expert_ids: &MappedHostBuffer<i32>,
    moe_expert_weights: &MappedHostBuffer<f32>,
    k: usize,
    hs: usize,
    eis: usize,
) -> HipResult<u32> {
    let expert_ids: &[i32] =
        unsafe { std::slice::from_raw_parts(moe_expert_ids.host_ptr() as *const i32, k) };
    let expert_weights: &[f32] =
        unsafe { std::slice::from_raw_parts(moe_expert_weights.host_ptr() as *const f32, k) };
    let num_devices = ctx.num_devices;
    let mut per_gpu: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_devices];
    for j in 0..k {
        let eid = expert_ids[j] as usize;
        let gpu = moe.expert_device[eid];
        per_gpu[gpu].push((eid, expert_weights[j]));
    }

    // Zero per-worker output buffers and H2D broadcast activation (parallel via SDMA engines)
    let act_host: *const std::ffi::c_void = _normed_stage.host_ptr() as *const std::ffi::c_void;
    for gpu in 1..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        let worker = &ctx.workers[gpu];
        Device::set_current(worker.device)?;
        unsafe {
            ffi::hipMemsetAsync(
                worker.expert_out.as_ptr() as *mut std::ffi::c_void,
                0,
                hs * 4,
                worker.compute_stream.raw(),
            );
            let rc = ffi::hipMemcpyAsync(
                worker.activation_in.as_ptr() as *mut std::ffi::c_void,
                act_host,
                hs * 4,
                ffi::hipMemcpyHostToDevice,
                worker.transfer_stream.raw(),
            );
            if rc != 0 {
                return Err(braidinfer_hip::HipError(rc));
            }
        }
        worker.broadcast_done.record(&worker.transfer_stream)?;
    }

    // Launch expert FFN on each GPU 1..N-1
    for gpu in 1..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        let worker = &mut ctx.workers[gpu];
        let kernels = &worker_kernels[gpu];
        let buf = &moe.expert_buffers[gpu];
        MultiGpuContext::stream_wait_event(&worker.compute_stream, &worker.broadcast_done)?;
        Device::set_current(worker.device)?;
        let input_ptr: *const f32 = worker.activation_in.as_ptr();
        let gate_up_base: *const u8 = buf.gate_up.as_ptr();
        let down_base: *const u8 = buf.down.as_ptr();
        let fmt = moe.weight_format;
        for &(expert_id, weight) in &per_gpu[gpu] {
            let local_slot = buf.slot_map[expert_id].expect("expert not on expected GPU");
            let gate_up_offset = local_slot * moe.gate_up_expert_stride;
            let down_offset = local_slot * moe.down_expert_stride;
            if moe.has_gate_proj {
                let gate_byte_offset = gate_up_offset;
                let up_byte_offset = gate_up_offset + eis * moe.gate_up_row_stride;
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_gate.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(gate_byte_offset) },
                    input_ptr,
                    eis as u32,
                    hs as u32,
                    fmt,
                    &worker.compute_stream,
                )?;
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(up_byte_offset) },
                    input_ptr,
                    eis as u32,
                    hs as u32,
                    fmt,
                    &worker.compute_stream,
                )?;
                kernels.silu_mul.forward(
                    &mut worker.scratch_act,
                    &worker.scratch_gate,
                    &worker.scratch_up,
                    eis as u32,
                    &worker.compute_stream,
                )?;
            } else {
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(gate_up_offset) },
                    input_ptr,
                    eis as u32,
                    hs as u32,
                    fmt,
                    &worker.compute_stream,
                )?;
                kernels.silu_mul.relu_squared(
                    &mut worker.scratch_act,
                    &worker.scratch_up,
                    eis as u32,
                    &worker.compute_stream,
                )?;
            }
            dispatch_proj(
                &kernels.linear_proj,
                worker.scratch_gate.as_ptr() as *mut f32,
                unsafe { down_base.add(down_offset) },
                worker.scratch_act.as_ptr(),
                hs as u32,
                eis as u32,
                fmt,
                &worker.compute_stream,
            )?;
            kernels.residual_add.weighted_accumulate(
                &mut worker.expert_out,
                &worker.scratch_gate,
                weight,
                hs as u32,
                &worker.compute_stream,
            )?;
        }
        worker.compute_done.record(&worker.compute_stream)?;
    }

    let mut active_mask = 0u32;
    for gpu in 1..num_devices {
        if !per_gpu[gpu].is_empty() {
            active_mask |= 1 << gpu;
        }
    }
    Device::set_current(DeviceId(0))?;
    Ok(active_mask)
}

/// GART dispatch: caller copies activation to normed_stage (GART/write-through) via async D2D,
/// records fc1_done event. Workers H2D from normed_stage host pointer — always PCIe-coherent.
/// Avoids RDNA3 L2 coherency issue where P2P reads bypass GPU 0's L2 and see stale VRAM.
/// Populates ffn_down_stage; caller copies to ffn_down if needed.
pub fn dispatch_moe_layer_p2p(
    ctx: &mut MultiGpuContext,
    worker_kernels: &[WorkerKernels],
    moe: &DistributedMoeWeights,
    src_host: *const f32,  // activation in GART host-mapped memory (normed_stage), PCIe-coherent
    ffn_down_stage: &mut MappedHostBuffer<f32>,
    moe_expert_ids: &MappedHostBuffer<i32>,
    moe_expert_weights: &MappedHostBuffer<f32>,
    k: usize,
    src_size: usize, // latent_size for Nemotron-H, hs for standard
    eis: usize,
    _stream0: &Stream,
) -> HipResult<()> {
    let expert_ids: &[i32] =
        unsafe { std::slice::from_raw_parts(moe_expert_ids.host_ptr() as *const i32, k) };
    let expert_weights: &[f32] =
        unsafe { std::slice::from_raw_parts(moe_expert_weights.host_ptr() as *const f32, k) };

    let num_devices = ctx.num_devices;
    let mut per_gpu: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_devices];
    for j in 0..k {
        let eid = expert_ids[j] as usize;
        let gpu = moe.expert_device[eid];
        per_gpu[gpu].push((eid, expert_weights[j]));
    }

    unsafe {
        std::ptr::write_bytes(ffn_down_stage.host_ptr(), 0, src_size);
    }

    // Broadcast activation to each worker via H2D from GART (normed_stage).
    // fc1_done fires after async D2D to normed_stage completes, so src_host is valid.
    // H2D from GART is always PCIe-coherent — no L2 stale data issue.
    let act_host: *const std::ffi::c_void = src_host as *const std::ffi::c_void;
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        let worker = &ctx.workers[gpu];
        Device::set_current(worker.device)?;
        unsafe {
            ffi::hipMemsetAsync(
                worker.expert_out.as_ptr() as *mut std::ffi::c_void,
                0,
                src_size * 4,
                worker.compute_stream.raw(),
            );
        }
        // Wait for fc1_done (fired after async D2D to GART), then H2D from GART host pointer.
        MultiGpuContext::stream_wait_event(&worker.transfer_stream, &ctx.fc1_done)?;
        unsafe {
            let rc = ffi::hipMemcpyAsync(
                worker.activation_in.as_ptr() as *mut std::ffi::c_void,
                act_host,
                src_size * 4,
                ffi::hipMemcpyHostToDevice,
                worker.transfer_stream.raw(),
            );
            if rc != 0 {
                return Err(braidinfer_hip::HipError(rc));
            }
        }
        worker.broadcast_done.record(&worker.transfer_stream)?;
    }

    let sync_debug = std::env::var("SYNC_DEBUG").is_ok();
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        let worker = &mut ctx.workers[gpu];
        let kernels = &worker_kernels[gpu];
        let buf = &moe.expert_buffers[gpu];
        MultiGpuContext::stream_wait_event(&worker.compute_stream, &worker.broadcast_done)?;
        Device::set_current(worker.device)?;
        if sync_debug {
            eprintln!("SYNC_DEBUG: p2p dispatch GPU{gpu} {} experts", per_gpu[gpu].len());
        }
        let input_ptr = worker.activation_in.as_ptr();
        let gate_up_base = buf.gate_up.as_ptr();
        let down_base = buf.down.as_ptr();
        let fmt = moe.weight_format;
        let expert_in_dim = moe.gate_up_in_dim as u32;

        for &(expert_id, weight) in &per_gpu[gpu] {
            let local_slot = buf.slot_map[expert_id].expect("expert not on expected GPU");
            let gate_up_offset = local_slot * moe.gate_up_expert_stride;
            let down_offset = local_slot * moe.down_expert_stride;

            if moe.has_gate_proj {
                let up_byte_offset = gate_up_offset + eis * moe.gate_up_row_stride;
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_gate.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(gate_up_offset) },
                    input_ptr,
                    eis as u32,
                    expert_in_dim,
                    fmt,
                    &worker.compute_stream,
                )?;
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(up_byte_offset) },
                    input_ptr,
                    eis as u32,
                    expert_in_dim,
                    fmt,
                    &worker.compute_stream,
                )?;
                kernels.silu_mul.forward(
                    &mut worker.scratch_act,
                    &worker.scratch_gate,
                    &worker.scratch_up,
                    eis as u32,
                    &worker.compute_stream,
                )?;
            } else {
                dispatch_proj(
                    &kernels.linear_proj,
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(gate_up_offset) },
                    input_ptr,
                    eis as u32,
                    expert_in_dim,
                    fmt,
                    &worker.compute_stream,
                )?;
                kernels.silu_mul.relu_squared(
                    &mut worker.scratch_act,
                    &worker.scratch_up,
                    eis as u32,
                    &worker.compute_stream,
                )?;
            }

            dispatch_proj(
                &kernels.linear_proj,
                worker.scratch_gate.as_ptr() as *mut f32,
                unsafe { down_base.add(down_offset) },
                worker.scratch_act.as_ptr(),
                expert_in_dim,
                eis as u32,
                fmt,
                &worker.compute_stream,
            )?;
            kernels.residual_add.weighted_accumulate(
                &mut worker.expert_out,
                &worker.scratch_gate,
                weight,
                expert_in_dim,
                &worker.compute_stream,
            )?;
        }
        worker.compute_done.record(&worker.compute_stream)?;
        if sync_debug {
            worker.compute_done.synchronize()?;
            eprintln!("SYNC_DEBUG: GPU{gpu} compute done OK");
        }
    }

    // Async gather: overlap D2H transfers from all GPUs
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        let worker = &ctx.workers[gpu];
        Device::set_current(worker.device)?;
        MultiGpuContext::stream_wait_event(&worker.transfer_stream, &worker.compute_done)?;
        unsafe {
            let rc = ffi::hipMemcpyAsync(
                worker.gather_host.as_ptr() as *mut std::ffi::c_void,
                worker.expert_out.as_ptr() as *const std::ffi::c_void,
                src_size * 4,
                ffi::hipMemcpyDeviceToHost,
                worker.transfer_stream.raw(),
            );
            if rc != 0 {
                return Err(braidinfer_hip::HipError(rc));
            }
        }
        worker.transfer_done.record(&worker.transfer_stream)?;
    }
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() {
            continue;
        }
        ctx.workers[gpu].transfer_done.synchronize()?;
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(ctx.workers[gpu].gather_host.as_ptr(), src_size) };
        let out: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(ffn_down_stage.host_ptr(), src_size) };
        for i in 0..src_size {
            out[i] += src[i];
        }
    }

    Device::set_current(DeviceId(0))?;
    Ok(())
}
