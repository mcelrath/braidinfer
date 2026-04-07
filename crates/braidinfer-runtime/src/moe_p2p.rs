//! GPU-native P2P MoE dispatch via persistent cooperative worker kernels.
//!
//! # Architecture
//!
//! GPU 0 runs `megakernel_f32` (persistent cooperative kernel). When it encounters
//! `OP_MOE_DISPATCH`, `op_moe_dispatch` in megakernel_moe_barrier.hip:
//!   1. Writes expert_ids/weights/activation_ptr into `MoeWorkItem` (GART/host-mapped)
//!   2. Bumps `seq_counter` to trigger workers on GPUs 1-3
//!   3. Computes GPU 0's local experts (all SMs cooperate)
//!   4. Polls `MoeWorkItem.ack_flags[w]` for all workers
//!   5. Sums `output_slots[gpu * hs]` into `final_output`
//!
//! Workers (GPUs 1-3) run `moe_worker_kernel` (moe_worker.hip):
//!   - Poll `MoeWorkItem.seq_num` for new work
//!   - P2P-copy activation from GPU 0 VRAM
//!   - Compute local experts (config looked up by layer_idx)
//!   - P2P-write result to `output_slots[my_gpu * hs]` on GPU 0 VRAM
//!   - Write `ack_flags[my_gpu] = seq`
//!
//! # Shutdown
//!
//! `shutdown.write_volatile(1)` → worker kernel writes `done_flag=1` before return.
//! Drop polls done_flag (30s timeout) before freeing GPU resources.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::Device;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::mem::ManuallyDrop;

use crate::weights::DistributedMoeWeights;

/// MoeExpertEntry layout (must match moe_work_queue.h).
#[repr(C)]
struct MoeExpertEntry {
    global_expert_id: u32,
    _pad: u32,
    gate_up_ptr: u64,
    down_ptr: u64,
}

/// MoeWorkerConfig layout (must match moe_work_queue.h).
#[repr(C)]
struct MoeWorkerConfig {
    my_gpu_id: u32,
    num_experts_local: u32,
    gate_up_row_stride: u32,
    down_row_stride: u32,
    hidden_size: u32,
    expert_intermediate_size: u32,
    _pad: [u32; 2],
    entries: [MoeExpertEntry; 256],
}

const CONFIG_SIZE: usize = std::mem::size_of::<MoeWorkerConfig>();

fn multiprocessor_count(device: DeviceId) -> HipResult<u32> {
    let mut val = 0i32;
    braidinfer_hip::error::check(unsafe {
        ffi::hipDeviceGetAttribute(&mut val, 63, device.0 as i32)
    })?;
    Ok(val as u32)
}

/// Per-worker GPU state (GPUs 1-3).
pub struct MoeWorkerGpu {
    pub device: DeviceId,
    /// Per-layer config pointer array on this GPU's VRAM: `MoeWorkerConfig*[num_layers]`.
    layer_config_ptrs: DeviceBuffer<u64>,
    /// Per-layer config blobs on this GPU's VRAM (kept alive for kernel lifetime).
    _config_storage: Vec<DeviceBuffer<u8>>,
    pub local_activation: DeviceBuffer<f32>,
    pub scratch_gate: DeviceBuffer<f32>,
    pub scratch_up: DeviceBuffer<f32>,
    pub scratch_act: DeviceBuffer<f32>,
    pub local_output: DeviceBuffer<f32>,
    /// Host-mapped shutdown flag (write 1 to initiate shutdown).
    pub shutdown: MappedHostBuffer<u32>,
    /// Host-mapped done flag (kernel writes 1 before returning after shutdown).
    pub done: MappedHostBuffer<u32>,
    pub stream: Stream,
    pub module: Module,
}

/// GPU-native P2P MoE dispatch context.
pub struct MoeP2pContext {
    /// Shared work queue in GART memory (MoeWorkItem layout, host-mapped).
    pub work_queue: MappedHostBuffer<u8>,
    /// Monotonic dispatch sequence counter (GART, host-mapped u32).
    pub seq_counter: MappedHostBuffer<u32>,
    /// Expert output accumulation buffer on GPU 0 VRAM: `float[num_gpus * hidden_size]`.
    pub output_slots: DeviceBuffer<f32>,
    /// GPU 0 per-layer config pointer array on GPU 0 VRAM: `MoeWorkerConfig*[num_layers]`.
    pub gpu0_layer_config_ptrs: DeviceBuffer<u64>,
    /// GPU 0 per-layer config blobs on GPU 0 VRAM (kept alive).
    _gpu0_config_storage: Vec<DeviceBuffer<u8>>,
    /// GPU 0 scratch buffers for expert computation.
    pub gpu0_scratch_gate: DeviceBuffer<f32>,
    pub gpu0_scratch_up: DeviceBuffer<f32>,
    pub gpu0_scratch_act: DeviceBuffer<f32>,
    /// Persistent worker states for GPUs 1-3 (ManuallyDrop: freed only after done_flags set).
    pub workers: Vec<ManuallyDrop<MoeWorkerGpu>>,
    pub num_gpus: usize,
    pub hidden_size: usize,
}

/// Size of MoeWorkItem in bytes (must fit the struct in moe_work_queue.h).
/// seq_num(4)+layer_idx(4)+num_active(4)+hidden_size(4)+expert_intermediate_size(4)+
/// has_gate_proj(4)+num_workers(4)+_pad0(4) = 32
/// expert_ids[32]*4 = 128, expert_weights[32]*4 = 128
/// activation_ptr(8) + output_slots_ptr(8) = 16
/// ack_flags[8]*4 = 32
/// Total = 336; round to 512 for alignment safety.
pub const MOE_WORK_QUEUE_SIZE: usize = 512;

impl MoeP2pContext {
    /// Initialize GPU-native P2P MoE dispatch.
    ///
    /// `gpu0`: GPU 0 device (runs megakernel / op_moe_dispatch).
    /// `worker_devices`: GPUs 1..N that will run moe_worker_kernel.
    /// `dist_moe_by_layer`: per-layer distributed MoE weights (use `None` for non-MoE layers).
    /// `num_total_layers`: total model layer count (max layer_idx + 1).
    pub fn init(
        gpu0: DeviceId,
        worker_devices: &[DeviceId],
        hidden_size: usize,
        expert_intermediate_size: usize,
        num_total_layers: usize,
        dist_moe_by_layer: &[Option<&DistributedMoeWeights>],
        shared_mem: u32,
    ) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let num_workers = worker_devices.len();
        let num_gpus = num_workers + 1;

        // Allocate shared GART resources (GPU 0 context)
        Device::set_current(gpu0)?;
        let work_queue = MappedHostBuffer::<u8>::alloc(MOE_WORK_QUEUE_SIZE)?;
        let seq_counter = MappedHostBuffer::<u32>::alloc(1)?;
        let output_slots = DeviceBuffer::<f32>::alloc(gpu0, num_gpus * hidden_size)?;

        let (gpu0_layer_config_ptrs, gpu0_config_storage) = build_layer_configs(
            gpu0,
            0,
            num_total_layers,
            dist_moe_by_layer,
            hidden_size,
            expert_intermediate_size,
            |dist, eid| {
                let buf = &dist.expert_buffers[0];
                buf.slot_map[eid].map(|slot| {
                    let gu = unsafe {
                        dist.gpu0_gate_up_base
                            .add(slot * dist.gate_up_expert_stride)
                    } as u64;
                    let dn =
                        unsafe { dist.gpu0_down_base.add(slot * dist.down_expert_stride) } as u64;
                    (gu, dn, buf.local_expert_count as u32)
                })
            },
        )?;

        let gpu0_scratch_gate = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        let gpu0_scratch_up = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        let gpu0_scratch_act = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;

        // Launch moe_worker_kernel on each worker GPU
        let mut workers = Vec::with_capacity(num_workers);
        for (w_idx, &device) in worker_devices.iter().enumerate() {
            let gpu_id = (w_idx + 1) as u32;
            Device::set_current(device)?;

            let (layer_config_ptrs, config_storage) = build_layer_configs(
                device,
                gpu_id,
                num_total_layers,
                dist_moe_by_layer,
                hidden_size,
                expert_intermediate_size,
                |dist, eid| {
                    let buf = &dist.expert_buffers[gpu_id as usize];
                    buf.slot_map[eid].map(|slot| {
                        let gu =
                            unsafe { buf.gate_up.as_ptr().add(slot * dist.gate_up_expert_stride) }
                                as u64;
                        let dn =
                            unsafe { buf.down.as_ptr().add(slot * dist.down_expert_stride) } as u64;
                        (gu, dn, buf.local_expert_count as u32)
                    })
                },
            )?;

            let local_activation = DeviceBuffer::<f32>::alloc(device, hidden_size)?;
            let scratch_gate = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let scratch_up = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let scratch_act = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let local_output = DeviceBuffer::<f32>::alloc(device, hidden_size)?;
            let shutdown = MappedHostBuffer::<u32>::alloc(1)?;
            let done = MappedHostBuffer::<u32>::alloc(1)?;
            let stream = Stream::new(device)?;
            let module = Module::load(device, &kernel_dir.join("moe_worker.hsaco"))?;
            let func = module.get_function("moe_worker_kernel")?;

            // Args: work_queue, shutdown, layer_configs, done_flag,
            //       local_activation, scratch_gate, scratch_up, scratch_act, local_output
            let mut wq_ptr = work_queue.device_ptr() as *mut std::ffi::c_void;
            let mut sd_ptr = shutdown.device_ptr() as *mut std::ffi::c_void;
            let mut lc_ptr = layer_config_ptrs.as_ptr() as *mut std::ffi::c_void;
            let mut df_ptr = done.device_ptr() as *mut std::ffi::c_void;
            let mut la_ptr = local_activation.as_ptr() as *mut std::ffi::c_void;
            let mut sg_ptr = scratch_gate.as_ptr() as *mut std::ffi::c_void;
            let mut su_ptr = scratch_up.as_ptr() as *mut std::ffi::c_void;
            let mut sa_ptr = scratch_act.as_ptr() as *mut std::ffi::c_void;
            let mut lo_ptr = local_output.as_ptr() as *mut std::ffi::c_void;

            let mut args: [*mut std::ffi::c_void; 9] = [
                std::ptr::addr_of_mut!(wq_ptr).cast(),
                std::ptr::addr_of_mut!(sd_ptr).cast(),
                std::ptr::addr_of_mut!(lc_ptr).cast(),
                std::ptr::addr_of_mut!(df_ptr).cast(),
                std::ptr::addr_of_mut!(la_ptr).cast(),
                std::ptr::addr_of_mut!(sg_ptr).cast(),
                std::ptr::addr_of_mut!(su_ptr).cast(),
                std::ptr::addr_of_mut!(sa_ptr).cast(),
                std::ptr::addr_of_mut!(lo_ptr).cast(),
            ];

            let bpsm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
            let num_cus = multiprocessor_count(device)?;
            let num_blocks = (bpsm as u32 * num_cus).max(num_cus);
            func.launch_cooperative(
                (num_blocks, 1, 1),
                (256, 1, 1),
                shared_mem,
                &stream,
                &mut args,
            )?;
            eprintln!(
                "  MoE worker GPU {}: launched ({num_blocks} blocks, {shared_mem}B shared)",
                device.0
            );

            workers.push(ManuallyDrop::new(MoeWorkerGpu {
                device,
                layer_config_ptrs,
                _config_storage: config_storage,
                local_activation,
                scratch_gate,
                scratch_up,
                scratch_act,
                local_output,
                shutdown,
                done,
                stream,
                module,
            }));
        }

        Ok(MoeP2pContext {
            work_queue,
            seq_counter,
            output_slots,
            gpu0_layer_config_ptrs,
            _gpu0_config_storage: gpu0_config_storage,
            gpu0_scratch_gate,
            gpu0_scratch_up,
            gpu0_scratch_act,
            workers,
            num_gpus,
            hidden_size,
        })
    }
}

impl Drop for MoeP2pContext {
    fn drop(&mut self) {
        for worker in &self.workers {
            unsafe {
                std::ptr::write_volatile(worker.shutdown.host_ptr(), 1u32);
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        for worker in &self.workers {
            loop {
                let done = unsafe { std::ptr::read_volatile(worker.done.host_ptr()) };
                if done != 0 {
                    break;
                }
                if std::time::Instant::now() > deadline {
                    eprintln!(
                        "braidinfer: moe_worker shutdown timeout on GPU {}",
                        worker.device.0
                    );
                    break;
                }
                std::hint::spin_loop();
            }
        }
        for worker in &mut self.workers {
            unsafe {
                ManuallyDrop::drop(worker);
            }
        }
    }
}

/// Build per-layer config pointer array and config blobs on a GPU device.
///
/// `get_expert_ptrs(dist, eid)` → `Some((gate_up_ptr, down_ptr, num_local_experts))` or `None`.
fn build_layer_configs(
    device: DeviceId,
    gpu_id: u32,
    num_layers: usize,
    dist_moe_by_layer: &[Option<&DistributedMoeWeights>],
    hidden_size: usize,
    expert_intermediate_size: usize,
    get_expert_ptrs: impl Fn(&DistributedMoeWeights, usize) -> Option<(u64, u64, u32)>,
) -> HipResult<(DeviceBuffer<u64>, Vec<DeviceBuffer<u8>>)> {
    let mut ptr_array = vec![0u64; num_layers];
    let mut config_storage: Vec<DeviceBuffer<u8>> = Vec::new();

    for (layer_idx, maybe_dist) in dist_moe_by_layer.iter().enumerate() {
        if layer_idx >= num_layers {
            break;
        }
        let Some(dist) = maybe_dist else { continue };

        // Row strides for Q4 PcG32: num_groups * 20 bytes/group.
        // gate_up rows: output=eis, input=hs → num_groups = (hs+31)/32
        // down rows: output=hs, input=eis → num_groups = (eis+31)/32
        let gate_up_row_stride = ((hidden_size + 31) / 32 * 20) as u32;
        let down_row_stride = ((expert_intermediate_size + 31) / 32 * 20) as u32;

        // Build config for this layer
        let mut cfg = MoeWorkerConfig {
            my_gpu_id: gpu_id,
            num_experts_local: 0, // filled below
            gate_up_row_stride,
            down_row_stride,
            hidden_size: hidden_size as u32,
            expert_intermediate_size: expert_intermediate_size as u32,
            _pad: [0; 2],
            entries: unsafe { std::mem::zeroed() },
        };
        let mut local_count = 0u32;
        for eid in 0..dist.num_experts.min(256) {
            if let Some((gu_ptr, dn_ptr, cnt)) = get_expert_ptrs(dist, eid) {
                cfg.entries[eid] = MoeExpertEntry {
                    global_expert_id: eid as u32,
                    _pad: 0,
                    gate_up_ptr: gu_ptr,
                    down_ptr: dn_ptr,
                };
                local_count = cnt;
            }
        }
        cfg.num_experts_local = local_count;

        // Upload to device
        let config_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&cfg as *const MoeWorkerConfig as *const u8, CONFIG_SIZE)
        };
        let mut dev_buf = DeviceBuffer::<u8>::alloc(device, CONFIG_SIZE)?;
        dev_buf.copy_from_host(config_bytes)?;
        ptr_array[layer_idx] = dev_buf.as_ptr() as u64;
        config_storage.push(dev_buf);
    }

    // Upload pointer array to device
    let mut ptr_buf = DeviceBuffer::<u64>::alloc(device, num_layers)?;
    ptr_buf.copy_from_host(&ptr_array)?;
    Ok((ptr_buf, config_storage))
}
