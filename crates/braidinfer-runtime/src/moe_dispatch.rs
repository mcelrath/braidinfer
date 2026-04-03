//! Event-based multi-GPU MoE expert dispatch.
//! CPU reads expert_ids from GPU 0, broadcasts activation to worker GPUs,
//! launches expert FFN in parallel, gathers results back to GPU 0.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::device::Device;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};

use crate::kernel::{LinearProjKernel, SiluMulKernel, ResidualAddKernel};
use crate::multi_gpu::MultiGpuContext;
use crate::weights::DistributedMoeWeights;

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

/// Dispatch one MoE layer across multiple GPUs.
///
/// Assumes: OP_MOE_GATE has already run on GPU 0, producing expert_ids and weights
/// in the activation buffers. This function handles the expert FFN + accumulation.
pub fn dispatch_moe_layer(
    ctx: &mut MultiGpuContext,
    worker_kernels: &[WorkerKernels],
    moe: &DistributedMoeWeights,
    normed: &DeviceBuffer<f32>,
    ffn_down: &mut DeviceBuffer<f32>,
    moe_expert_ids: &DeviceBuffer<i32>,
    moe_expert_weights: &DeviceBuffer<f32>,
    k: usize,
    hs: usize,
    eis: usize,
    stream0: &Stream,
) -> HipResult<()> {
    // 1. Read expert_ids + weights from GPU 0
    stream0.synchronize()?;
    let mut expert_ids = vec![0i32; k];
    let mut expert_weights = vec![0.0f32; k];
    moe_expert_ids.copy_to_host(&mut expert_ids)?;
    moe_expert_weights.copy_to_host(&mut expert_weights)?;

    // 2. Group experts by GPU
    let num_devices = ctx.num_devices;
    let mut per_gpu: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_devices];
    for j in 0..k {
        let eid = expert_ids[j] as usize;
        let gpu = moe.expert_device[eid];
        per_gpu[gpu].push((eid, expert_weights[j]));
    }

    // 3. Zero output buffers
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() { continue; }
        let worker = &ctx.workers[gpu];
        Device::set_current(worker.device)?;
        unsafe {
            ffi::hipMemsetAsync(
                worker.expert_out.as_ptr() as *mut std::ffi::c_void,
                0, hs * 4, worker.compute_stream.raw(),
            );
        }
    }
    Device::set_current(DeviceId(0))?;
    unsafe {
        ffi::hipMemsetAsync(
            ffn_down.as_mut_ptr() as *mut std::ffi::c_void,
            0, hs * 4, stream0.raw(),
        );
    }

    // 4. Broadcast activation from GPU 0 to worker GPUs
    for gpu in 1..num_devices {
        if per_gpu[gpu].is_empty() { continue; }
        let worker = &ctx.workers[gpu];
        MultiGpuContext::memcpy_peer_async(
            worker.activation_in.as_ptr() as *mut std::ffi::c_void, worker.device,
            normed.as_ptr() as *const std::ffi::c_void, DeviceId(0),
            hs * 4, &worker.transfer_stream,
        )?;
        worker.broadcast_done.record(&worker.transfer_stream)?;
    }

    // 5. Launch expert FFN on each GPU
    for gpu in 0..num_devices {
        if per_gpu[gpu].is_empty() { continue; }

        let worker = &mut ctx.workers[gpu];
        let kernels = &worker_kernels[gpu];
        let buf = &moe.expert_buffers[gpu];

        if gpu > 0 {
            MultiGpuContext::stream_wait_event(&worker.compute_stream, &worker.broadcast_done)?;
        }

        Device::set_current(worker.device)?;

        let input_ptr: *const f32 = if gpu == 0 {
            normed.as_ptr()
        } else {
            worker.activation_in.as_ptr()
        };

        // Weight base pointers: GPU 0 uses original packed buffer, others use per-GPU buffer
        let gate_up_base: *const u8 = if gpu == 0 {
            moe.gpu0_gate_up_base
        } else {
            buf.gate_up.as_ptr()
        };
        let down_base: *const u8 = if gpu == 0 {
            moe.gpu0_down_base
        } else {
            buf.down.as_ptr()
        };

        for &(expert_id, weight) in &per_gpu[gpu] {
            let local_slot = buf.slot_map[expert_id]
                .expect("expert not on expected GPU");

            let gate_up_offset = local_slot * moe.gate_up_expert_stride;
            let down_offset = local_slot * moe.down_expert_stride;

            if moe.has_gate_proj {
                let gate_byte_offset = gate_up_offset;
                let up_byte_offset = gate_up_offset + eis * moe.gate_up_row_stride;

                kernels.linear_proj.forward_packed_ptr(
                    worker.scratch_gate.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(gate_byte_offset) },
                    input_ptr, eis as u32, hs as u32,
                    "linear_proj_pcg32_q4", &worker.compute_stream,
                )?;
                kernels.linear_proj.forward_packed_ptr(
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { gate_up_base.add(up_byte_offset) },
                    input_ptr, eis as u32, hs as u32,
                    "linear_proj_pcg32_q4", &worker.compute_stream,
                )?;
                kernels.silu_mul.forward(
                    &mut worker.scratch_act,
                    &worker.scratch_gate,
                    &worker.scratch_up,
                    eis as u32, &worker.compute_stream,
                )?;
            } else {
                // relu²: up + relu²
                kernels.linear_proj.forward_packed_ptr(
                    worker.scratch_up.as_ptr() as *mut f32,
                    unsafe { buf.gate_up.as_ptr().add(gate_up_offset) },
                    input_ptr, eis as u32, hs as u32,
                    "linear_proj_pcg32_q4", &worker.compute_stream,
                )?;
                kernels.silu_mul.relu_squared(
                    &mut worker.scratch_act,
                    &worker.scratch_up,
                    eis as u32, &worker.compute_stream,
                )?;
            }

            // down_proj → scratch_gate (reuse as scratch for down output)
            kernels.linear_proj.forward_packed_ptr(
                worker.scratch_gate.as_ptr() as *mut f32,
                unsafe { down_base.add(down_offset) },
                worker.scratch_act.as_ptr(),
                hs as u32, eis as u32,
                "linear_proj_pcg32_q4", &worker.compute_stream,
            )?;

            // weighted accumulate: expert_out += weight * down_output
            kernels.residual_add.weighted_accumulate(
                &mut worker.expert_out,
                &worker.scratch_gate, // reused as down_proj output
                weight,
                hs as u32, &worker.compute_stream,
            )?;
        }

        worker.compute_done.record(&worker.compute_stream)?;
    }

    // 6. Gather: copy worker expert_out to GPU 0 and accumulate into ffn_down
    Device::set_current(DeviceId(0))?;

    // GPU 0's local results: ffn_down += expert_out[0]
    if !per_gpu[0].is_empty() {
        MultiGpuContext::stream_wait_event(stream0, &ctx.workers[0].compute_done)?;
        worker_kernels[0].residual_add.weighted_accumulate(
            ffn_down,
            &ctx.workers[0].expert_out,
            1.0,
            hs as u32, stream0,
        )?;
    }

    // Remote results: P2P copy then accumulate (sequentially through gather_stream)
    for gpu in 1..num_devices {
        if per_gpu[gpu].is_empty() { continue; }
        let worker = &ctx.workers[gpu];

        MultiGpuContext::stream_wait_event(&ctx.gather_stream, &worker.compute_done)?;

        // P2P copy worker's expert_out → GPU 0's workers[0].activation_in (reuse as gather buf)
        MultiGpuContext::memcpy_peer_async(
            ctx.workers[0].activation_in.as_ptr() as *mut std::ffi::c_void, DeviceId(0),
            worker.expert_out.as_ptr() as *const std::ffi::c_void, worker.device,
            hs * 4, &ctx.gather_stream,
        )?;

        // Wait for copy, then accumulate on main stream
        ctx.gather_done.record(&ctx.gather_stream)?;
        MultiGpuContext::stream_wait_event(stream0, &ctx.gather_done)?;

        worker_kernels[0].residual_add.weighted_accumulate(
            ffn_down,
            &ctx.workers[0].activation_in,
            1.0,
            hs as u32, stream0,
        )?;
    }

    Ok(())
}
