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

use crate::quant::WeightFormat;
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
    hidden_size: u32,
    expert_intermediate_size: u32,
    weight_format: u32,  // 0=PCG32Q4, 1=RNF4G128 (matches MOE_WEIGHT_FORMAT_* constants)
    _pad: [u32; 2],
    entries: [MoeExpertEntry; 512],
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
    _layer_config_ptrs: DeviceBuffer<u64>,
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
    /// GART timing buffer: [N_TIMING_SLOTS * 4] u64 cycle timestamps.
    /// Layout per slot i: [t_work_start, t_copy_done, t_experts_done, t_output_done].
    pub timing_buf: MappedHostBuffer<u64>,
    pub stream: Stream,
    pub module: Module,
}

/// GPU-native P2P MoE dispatch context.
pub struct MoeP2pContext {
    /// Shared work queue in GART memory (MoeWorkItem fixed fields + activation_cache[gate_up_in_dim]).
    pub work_queue: MappedHostBuffer<u8>,
    /// Monotonic dispatch sequence counter (GART, host-mapped u32).
    pub seq_counter: MappedHostBuffer<u32>,
    /// Expert output accumulation buffer on GPU 0 VRAM: `float[num_gpus * hidden_size]`.
    pub output_slots: DeviceBuffer<f32>,
    /// GPU 0 per-layer config pointer array on GPU 0 VRAM: `MoeWorkerConfig*[num_layers]`.
    pub gpu0_layer_config_ptrs: DeviceBuffer<u64>,
    /// GPU 0 per-layer config blobs on GPU 0 VRAM (kept alive).
    _gpu0_config_storage: Vec<DeviceBuffer<u8>>,
    /// GPU 0 VRAM activation staging buffer: [MAX_PREFILL_BATCH × gate_up_in_dim].
    /// Prefill path copies activations here via hipMemcpy; decode path uses activation directly.
    /// Workers P2P-read via activation_ptr field (set to VRAM ptr) after __threadfence_system().
    pub activation_staging: DeviceBuffer<f32>,
    /// GPU 0 scratch buffers for expert computation.
    pub gpu0_scratch_gate: DeviceBuffer<f32>,
    pub gpu0_scratch_up: DeviceBuffer<f32>,
    pub gpu0_scratch_act: DeviceBuffer<f32>,
    /// Persistent worker states for GPUs 1-3 (ManuallyDrop: freed only after done_flags set).
    pub workers: Vec<ManuallyDrop<MoeWorkerGpu>>,
    pub num_gpus: usize,
    pub hidden_size: usize,
}

/// Fixed-field size of MoeWorkItem (bytes), excluding the flexible activation_cache[] tail.
/// seq_num(4)+batch_size(4)+layer_idx(4)+num_active(4)+hidden_size(4)+
/// eis(4)+has_gate_proj(4)+num_workers(4)+gate_up_in_dim(4) = 36
/// expert_ids[64*32]*4 = 8192, expert_weights[64*32]*4 = 8192
/// _pad_align(4) to align activation_ptr to 8 bytes (offset 16420 → 16424)
/// activation_ptr(8)+output_slots_ptr(8) = 16
/// ack_flags[8]*4 = 32
/// Total fixed = 36 + 8192 + 8192 + 4 + 16 + 32 = 16472 bytes.
/// Full work_queue size = MOE_WORK_QUEUE_FIXED + batch_size * gate_up_in_dim * 4.
pub const MOE_WORK_QUEUE_FIXED: usize = 16472;

/// Maximum tokens per batched prefill dispatch (must match MOE_MAX_PREFILL_BATCH in moe_work_queue.h).
pub const MAX_PREFILL_BATCH: usize = 64;
/// Maximum active experts per token (must match MOE_MAX_ACTIVE_EXPERTS in moe_work_queue.h).
pub const MAX_ACTIVE_EXPERTS: usize = 32;

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
        gate_up_in_dim: usize,
        expert_intermediate_size: usize,
        num_total_layers: usize,
        dist_moe_by_layer: &[Option<&DistributedMoeWeights>],
        shared_mem: u32,
    ) -> HipResult<Self> {
        let kernel_dir = crate::kernel::kernel_dir();
        let num_workers = worker_devices.len();
        let num_gpus = num_workers + 1;

        // Allocate shared GART resources (GPU 0 context).
        // Use alloc (NOT alloc_portable): MTYPE_UC (write-through) ensures GPU→CPU signaling
        // for ack_flags, seq_num, and done_flag is immediately visible without L2 caching.
        // alloc_portable uses MTYPE_NC which may cache GPU writes — unsafe for polling.
        // Workers access work_queue and seq_counter via hipHostGetDevicePointer per GPU
        // (called below in the worker launch loop), which works even without portable flag
        // because P2P is enabled between all GPUs at init time.
        // Work queue includes flexible activation_cache[] at the end, sized to gate_up_in_dim.
        Device::set_current(gpu0)?;
        // Allocate GART work queue sized for max batch (MAX_PREFILL_BATCH tokens).
        // The flexible activation_cache tail holds batch_size * gate_up_in_dim floats.
        let wq_size = MOE_WORK_QUEUE_FIXED + MAX_PREFILL_BATCH * gate_up_in_dim * std::mem::size_of::<f32>();
        let work_queue = MappedHostBuffer::<u8>::alloc(wq_size)?;
        let seq_counter = MappedHostBuffer::<u32>::alloc(1)?;
        // Output slots sized for MAX_PREFILL_BATCH × num_gpus × hidden_size.
        // Decode uses only the first (0 * num_gpus + gpu) * hs slot (batch_size=1).
        let output_slots = DeviceBuffer::<f32>::alloc(gpu0, MAX_PREFILL_BATCH * num_gpus * hidden_size)?;


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

        // Activation staging: GPU 0 VRAM for prefill batches (workers P2P-read after __threadfence_system).
        let activation_staging = DeviceBuffer::<f32>::alloc(gpu0, MAX_PREFILL_BATCH * gate_up_in_dim)?;
        // scratch_gate is reused for gate output (eis elements) AND down output (gupd elements).
        // Must be max(eis, gupd) = max(expert_intermediate_size, gate_up_in_dim).
        let scratch_gate_size = expert_intermediate_size.max(gate_up_in_dim);
        let gpu0_scratch_gate = DeviceBuffer::<f32>::alloc(gpu0, scratch_gate_size)?;
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

            let local_activation = DeviceBuffer::<f32>::alloc(device, gate_up_in_dim)?;
            // scratch_gate reused for gate output (eis) and down output (gupd): allocate max.
            let scratch_gate = DeviceBuffer::<f32>::alloc(device, scratch_gate_size)?;
            let scratch_up = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let scratch_act = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let local_output = DeviceBuffer::<f32>::alloc(device, hidden_size)?;
            let shutdown = MappedHostBuffer::<u32>::alloc(1)?;
            let done = MappedHostBuffer::<u32>::alloc(1)?;
            // Timing buffer: 64 slots × 4 timestamps each (GART, CPU-readable without memcpy).
            let timing_buf = MappedHostBuffer::<u64>::alloc(64 * 4)?;
            unsafe { std::ptr::write_bytes(timing_buf.host_ptr(), 0, 64 * 4); }
            let stream = Stream::new(device)?;
            let module = Module::load(device, &kernel_dir.join("moe_worker.hsaco"))?;
            let func = module.get_function("moe_worker_kernel")?;

            // Args: work_queue, shutdown, layer_configs, done_flag,
            //       local_activation, scratch_gate, scratch_up, scratch_act, local_output, gpu_id
            // Re-query device pointer for work_queue from CURRENT GPU context.
            // hipHostGetDevicePointer returns the VA for the current GPU's GPUVM page tables.
            // Even with hipHostMallocPortable, VA may differ per GPU on AMD discrete GPUs.
            let mut wq_dp: *mut std::ffi::c_void = std::ptr::null_mut();
            braidinfer_hip::error::check(unsafe {
                ffi::hipHostGetDevicePointer(
                    &mut wq_dp,
                    work_queue.host_ptr() as *mut std::ffi::c_void,
                    0,
                )
            })?;
            let mut wq_ptr = wq_dp;
            let mut sd_ptr = shutdown.device_ptr() as *mut std::ffi::c_void;
            let mut lc_ptr = layer_config_ptrs.as_ptr() as *mut std::ffi::c_void;
            let mut df_ptr = done.device_ptr() as *mut std::ffi::c_void;
            let mut la_ptr = local_activation.as_ptr() as *mut std::ffi::c_void;
            let mut sg_ptr = scratch_gate.as_ptr() as *mut std::ffi::c_void;
            let mut su_ptr = scratch_up.as_ptr() as *mut std::ffi::c_void;
            let mut sa_ptr = scratch_act.as_ptr() as *mut std::ffi::c_void;
            let mut lo_ptr = local_output.as_ptr() as *mut std::ffi::c_void;
            let mut gid = gpu_id;
            let mut tb_ptr = timing_buf.device_ptr() as *mut std::ffi::c_void;
            // watchdog: NULL disables the watchdog (Phase 2 wires a real WatchdogState here).
            let mut wd_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut args: [*mut std::ffi::c_void; 12] = [
                std::ptr::addr_of_mut!(wq_ptr).cast(),
                std::ptr::addr_of_mut!(sd_ptr).cast(),
                std::ptr::addr_of_mut!(lc_ptr).cast(),
                std::ptr::addr_of_mut!(df_ptr).cast(),
                std::ptr::addr_of_mut!(la_ptr).cast(),
                std::ptr::addr_of_mut!(sg_ptr).cast(),
                std::ptr::addr_of_mut!(su_ptr).cast(),
                std::ptr::addr_of_mut!(sa_ptr).cast(),
                std::ptr::addr_of_mut!(lo_ptr).cast(),
                std::ptr::addr_of_mut!(gid).cast(),
                std::ptr::addr_of_mut!(tb_ptr).cast(),
                std::ptr::addr_of_mut!(wd_ptr).cast(),
            ];

            let num_cus = multiprocessor_count(device)?;
            // Use exactly num_cus blocks (1 per CU): safest cooperative launch count.
            // Each block polls independently; more blocks = more parallelism but risks
            // cooperative constraint violations if bpsm * num_cus exceeds hardware capacity.
            let num_blocks = num_cus;
            func.launch_cooperative(
                (num_blocks, 1, 1),
                (256, 1, 1),
                shared_mem,
                &stream,
                &mut args,
            )?;
            // Wait until kernel reaches past cg::this_grid() (done_flag >= 0xAA02).
            let t0 = std::time::Instant::now();
            loop {
                let v = unsafe { std::ptr::read_volatile(done.host_ptr()) };
                if v >= 0xAA02 { break; }
                if crate::persistent_dispatch::shutdown_requested() {
                    panic!("moe_worker GPU {} startup interrupted: SIGINT/SIGTERM", gpu_id);
                }
                if t0.elapsed().as_millis() > 5000 {
                    panic!("GPU {} moe_worker_kernel failed to start (done_flag={v:#x})", gpu_id);
                }
                std::hint::spin_loop();
            }
            eprintln!("  moe_worker_kernel GPU {}: started (done_flag=0xAA02)", gpu_id);

            workers.push(ManuallyDrop::new(MoeWorkerGpu {
                device,
                _layer_config_ptrs: layer_config_ptrs,
                _config_storage: config_storage,
                local_activation,
                scratch_gate,
                scratch_up,
                scratch_act,
                local_output,
                shutdown,
                done,
                timing_buf,
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
            activation_staging,
            gpu0_scratch_gate,
            gpu0_scratch_up,
            gpu0_scratch_act,
            workers,
            num_gpus,
            hidden_size,
        })
    }

    // Offsets into the MoeWorkItem byte buffer (must match moe_work_queue.h layout).
    // seq_num(4)+batch_size(4)+layer_idx(4)+num_active(4)+hidden_size(4)+
    // eis(4)+has_gate_proj(4)+num_workers(4)+gate_up_in_dim(4) = 36
    // expert_ids[64*32]*4 = 8192 at offset 36
    // expert_weights[64*32]*4 = 8192 at offset 8228
    // _pad_align(4) at offset 16420 (alignment padding before activation_ptr)
    // activation_ptr(8) at offset 16424
    // output_slots_ptr(8) at offset 16432
    // ack_flags[8]*4 = 32 at offset 16440
    // activation_cache[] at MOE_WORK_QUEUE_FIXED = 16472
    const OFF_BATCH_SIZE: usize = 4;
    const OFF_LAYER_IDX: usize = 8;
    const OFF_NUM_ACTIVE: usize = 12;
    const OFF_HIDDEN_SIZE: usize = 16;
    const OFF_EIS: usize = 20;
    const OFF_HAS_GATE: usize = 24;
    const OFF_NUM_WORKERS: usize = 28;
    const OFF_GATE_UP_IN_DIM: usize = 32;
    const OFF_EXPERT_IDS: usize = 36;
    const OFF_EXPERT_WEIGHTS: usize = Self::OFF_EXPERT_IDS + MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS * 4;
    // +4 for _pad_align, then activation_ptr (8 bytes), then output_slots_ptr
    const OFF_ACTIVATION_PTR: usize = Self::OFF_EXPERT_WEIGHTS + MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS * 4 + 4;
    const OFF_OUTPUT_SLOTS_PTR: usize = Self::OFF_ACTIVATION_PTR + 8;
    const OFF_ACK_FLAGS: usize = Self::OFF_OUTPUT_SLOTS_PTR + 8;

    /// CPU-initiated batched MoE dispatch for prefill.
    ///
    /// Writes batch_size tokens' activations + routing into the GART work queue,
    /// triggers all worker GPUs, and returns the dispatch sequence number.
    /// Caller should do GPU 0 local expert computation, then call `poll_prefill_batch_ack`.
    ///
    /// `activations`: [batch_size × gate_up_in_dim] flat slice
    /// `expert_ids`: [batch_size × k] flat slice (row-major, k per token)
    /// `expert_weights`: [batch_size × k] flat slice
    pub fn trigger_prefill_batch(
        &mut self,
        activations: &[f32],
        expert_ids: &[i32],
        expert_weights: &[f32],
        batch_size: usize,
        k: usize,
        layer_idx: u32,
        hs: usize,
        eis: usize,
        has_gate_proj: bool,
        gate_up_in_dim: usize,
    ) -> u32 {
        assert!(batch_size <= MAX_PREFILL_BATCH);
        assert!(k <= MAX_ACTIVE_EXPERTS);
        let num_workers = self.workers.len();
        // Copy activations to GPU 0 VRAM staging buffer. Workers P2P-read via activation_ptr
        // after GPU 0's __threadfence_system() flushes L2 before seq_num write.
        // Safe: GPU 0's persistent worker is NOT running during prefill.
        self.activation_staging
            .copy_from_host(&activations[..batch_size * gate_up_in_dim])
            .expect("activation_staging hipMemcpy failed");
        let wq_ptr = self.work_queue.host_ptr() as *mut u8;
        unsafe {
            (wq_ptr.add(Self::OFF_BATCH_SIZE) as *mut u32).write_volatile(batch_size as u32);
            (wq_ptr.add(Self::OFF_LAYER_IDX) as *mut u32).write_volatile(layer_idx);
            (wq_ptr.add(Self::OFF_NUM_ACTIVE) as *mut u32).write_volatile(k as u32);
            (wq_ptr.add(Self::OFF_HIDDEN_SIZE) as *mut u32).write_volatile(hs as u32);
            (wq_ptr.add(Self::OFF_EIS) as *mut u32).write_volatile(eis as u32);
            (wq_ptr.add(Self::OFF_HAS_GATE) as *mut u32).write_volatile(has_gate_proj as u32);
            (wq_ptr.add(Self::OFF_NUM_WORKERS) as *mut u32).write_volatile(num_workers as u32);
            (wq_ptr.add(Self::OFF_GATE_UP_IN_DIM) as *mut u32).write_volatile(gate_up_in_dim as u32);
            // output_slots_ptr
            (wq_ptr.add(Self::OFF_OUTPUT_SLOTS_PTR) as *mut u64)
                .write_volatile(self.output_slots.as_ptr() as u64);
            // expert routing: [t * MAX_ACTIVE_EXPERTS + j]
            let ids_dst = wq_ptr.add(Self::OFF_EXPERT_IDS) as *mut i32;
            let wts_dst = wq_ptr.add(Self::OFF_EXPERT_WEIGHTS) as *mut f32;
            for t in 0..batch_size {
                for j in 0..k {
                    ids_dst.add(t * MAX_ACTIVE_EXPERTS + j).write_volatile(expert_ids[t * k + j]);
                    wts_dst.add(t * MAX_ACTIVE_EXPERTS + j).write_volatile(expert_weights[t * k + j]);
                }
            }
            // Write activation_ptr: GPU 0 VRAM staging (workers P2P-read after __threadfence_system).
            // GPU 0's persistent worker is NOT running during prefill, so hipMemcpy is safe.
            (wq_ptr.add(Self::OFF_ACTIVATION_PTR) as *mut u64)
                .write_volatile(self.activation_staging.as_ptr() as u64);
            // clear ack flags
            let ack_ptr = wq_ptr.add(Self::OFF_ACK_FLAGS) as *mut u32;
            for w in 1..=num_workers {
                ack_ptr.add(w).write_volatile(0u32);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let seq_ptr = self.seq_counter.host_ptr();
            let seq = seq_ptr.read_volatile().wrapping_add(1);
            seq_ptr.write_volatile(seq);
            (wq_ptr as *mut u32).write_volatile(seq); // seq_num triggers workers
            seq
        }
    }

    /// Poll ack flags from all worker GPUs after `trigger_prefill_batch`.
    /// Call after GPU 0 has finished its local expert computation.
    pub fn poll_prefill_batch_ack(&self, seq: u32) {
        let num_workers = self.workers.len();
        let wq_ptr = self.work_queue.host_ptr() as *const u8;
        let ack_ptr = unsafe { wq_ptr.add(Self::OFF_ACK_FLAGS) as *const u32 };
        for w in 1..=num_workers {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let ack = unsafe { ack_ptr.add(w).read_volatile() };
                if ack == seq { break; }
                if crate::persistent_dispatch::shutdown_requested() {
                    panic!("MoE prefill ack interrupted: SIGINT/SIGTERM (gpu={w}, seq={seq})");
                }
                if std::time::Instant::now() > deadline {
                    panic!("MoE worker GPU {} prefill batch ack timeout (seq={seq})", w);
                }
                std::hint::spin_loop();
            }
        }
    }

    /// Print timing analysis from the worker GPU timing buffers.
    /// GPU clock frequency for 7900XTX (RDNA3): ~2500 MHz.
    /// Call after at least one token has been generated.
    pub fn print_timing_report(&self, gpu_clock_mhz: f64) {
        let cycles_per_us = gpu_clock_mhz / 1000.0;
        for (i, worker) in self.workers.iter().enumerate() {
            let buf = worker.timing_buf.host_ptr();
            let mut slots_used = 0u32;
            // Count non-zero slots
            for s in 0..64 {
                let t0 = unsafe { std::ptr::read_volatile(buf.add(s * 4)) };
                if t0 == 0 { break; }
                slots_used += 1;
            }
            if slots_used == 0 {
                eprintln!("  Worker GPU {}: no timing data (timing_buf all zero)", i + 1);
                continue;
            }
            eprintln!("  Worker GPU {} timing ({} layers, clock={} MHz):", i + 1, slots_used, gpu_clock_mhz as u64);
            let mut total_outer_us = 0.0f64;
            let mut total_copy_us = 0.0f64;
            let mut total_expert_us = 0.0f64;
            let mut total_output_us = 0.0f64;
            for s in 0..(slots_used as usize) {
                let t0 = unsafe { std::ptr::read_volatile(buf.add(s * 4    )) }; // work_start
                let t1 = unsafe { std::ptr::read_volatile(buf.add(s * 4 + 1)) }; // copy_done
                let t2 = unsafe { std::ptr::read_volatile(buf.add(s * 4 + 2)) }; // experts_done
                let t3 = unsafe { std::ptr::read_volatile(buf.add(s * 4 + 3)) }; // output_done
                if t0 == 0 || t1 < t0 || t2 < t1 || t3 < t2 { continue; }
                total_copy_us   += (t1 - t0) as f64 / cycles_per_us;
                total_expert_us += (t2 - t1) as f64 / cycles_per_us;
                total_output_us += (t3 - t2) as f64 / cycles_per_us;
                // Outer sync cost: gap between this slot's output_done and next slot's work_start
                if s + 1 < slots_used as usize {
                    let t_next = unsafe { std::ptr::read_volatile(buf.add((s + 1) * 4)) };
                    if t_next > t3 {
                        total_outer_us += (t_next - t3) as f64 / cycles_per_us;
                    }
                }
            }
            let n = (slots_used - 1).max(1) as f64;
            eprintln!("    Outer sync+poll avg: {:.1} us/layer  (TOTAL {:.1} us)",
                total_outer_us / n, total_outer_us);
            eprintln!("    Activation copy avg: {:.1} us/layer  (TOTAL {:.1} us)",
                total_copy_us / slots_used as f64, total_copy_us);
            eprintln!("    Expert compute  avg: {:.1} us/layer  (TOTAL {:.1} us)",
                total_expert_us / slots_used as f64, total_expert_us);
            eprintln!("    Output copy     avg: {:.1} us/layer  (TOTAL {:.1} us)",
                total_output_us / slots_used as f64, total_output_us);
            let total = total_outer_us + total_copy_us + total_expert_us + total_output_us;
            eprintln!("    Total measured:      {:.1} us  ({:.1} tok/s if dominant)",
                total, 1e6 / total);
        }
    }
}

impl Drop for MoeP2pContext {
    fn drop(&mut self) {
        // ALWAYS request shutdown (even on panic). The kernel polls shutdown at the top
        // of its instruction loop. Use a short timeout on panic, longer on clean exit.
        // On timeout we leak HIP resources rather than risk hipFree deadlocking against
        // a still-running cooperative kernel.
        let panicking = std::thread::panicking();
        let timeout = if panicking {
            std::time::Duration::from_secs(2)
        } else {
            std::time::Duration::from_secs(5)
        };
        for worker in &self.workers {
            unsafe {
                std::ptr::write_volatile(worker.shutdown.host_ptr(), 1u32);
            }
        }
        let deadline = std::time::Instant::now() + timeout;
        let mut worker_done = vec![false; self.workers.len()];
        for (idx, worker) in self.workers.iter().enumerate() {
            loop {
                let done = unsafe { std::ptr::read_volatile(worker.done.host_ptr()) };
                if done != 0 {
                    worker_done[idx] = true;
                    break;
                }
                if std::time::Instant::now() > deadline {
                    eprintln!(
                        "braidinfer: moe_worker shutdown timeout on GPU {} (leaking) {}",
                        worker.device.0,
                        if panicking { "[panic]" } else { "" }
                    );
                    break;
                }
                std::hint::spin_loop();
            }
        }
        for (idx, worker) in self.workers.iter_mut().enumerate() {
            if worker_done[idx] {
                unsafe {
                    ManuallyDrop::drop(worker);
                }
            }
        }
        if panicking {
            std::process::exit(1);
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

        // Row stride from DistributedMoeWeights (authoritative: uses actual gate_up_in_dim,
        // which may be moe_latent_size < hidden_size for Nemotron-H).
        let gate_up_row_stride = dist.gate_up_row_stride as u32;

        // Map WeightFormat to MOE_WEIGHT_FORMAT_* constants (must match moe_work_queue.h).
        let weight_format_code = match dist.weight_format {
            WeightFormat::Rnf4G128 => 1u32, // MOE_WEIGHT_FORMAT_RNF4G128
            _ => 0u32,                       // MOE_WEIGHT_FORMAT_PCG32Q4 (default)
        };

        // Build config for this layer
        let mut cfg = MoeWorkerConfig {
            my_gpu_id: gpu_id,
            num_experts_local: 0, // filled below
            gate_up_row_stride,
            hidden_size: hidden_size as u32,
            expert_intermediate_size: expert_intermediate_size as u32,
            weight_format: weight_format_code,
            _pad: [0; 2],
            entries: unsafe { std::mem::zeroed() },
        };
        assert!(
            dist.num_experts <= 512,
            "MoeWorkerConfig::entries[512] too small for model with {} experts (layer {})",
            dist.num_experts,
            layer_idx
        );
        let mut local_count = 0u32;
        for eid in 0..dist.num_experts {
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

        // Upload to device (synchronous — called before any kernel is launched on this device)
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
