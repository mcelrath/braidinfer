//! Shared MoE state (post-unified-worker, epic braidinfer-0hu Phase 2-5).
//!
//! After the unified-worker cutover, `moe_worker_kernel` is gone — every GPU
//! runs `persistent_worker.hsaco`, dispatched via `PersistentDispatch`.
//!
//! `MoeP2pContext` retains the buffer/state ownership it had before:
//!   - `output_slots`: GPU 0 VRAM (UC-mapped) `[MAX_PREFILL_BATCH × num_gpus × hs]`,
//!     written via P2P by workers' `OP_MOE_FFN_REMOTE`, summed by GPU 0's
//!     `op_moe_dispatch` (now CPU-orchestrated).
//!   - `gpu0_layer_config_ptrs` + GPU 0 scratch + `activation_staging`: GPU 0's
//!     local-expert state used by `op_moe_dispatch`.
//!   - `workers[w]`: per-worker MoE state — `local_activation`, `local_output`,
//!     `scratch_*`, plus the per-layer `MoeWorkerConfig*` pointer array on each
//!     worker GPU. CPU populates these into `OP_MOE_FFN_REMOTE` instructions
//!     and dispatches them via the worker's persistent_worker mailbox.
//!
//! There are no longer any kernel modules or streams owned by this context —
//! all GPU work runs through `PersistentDispatch`. The `MoeP2pContext` name
//! and module location are preserved for now to minimize cutover diff; a
//! follow-up will fold its state into `PersistentDispatch::GpuWorker`.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::device::DeviceGuard;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};

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

/// Per-worker GPU MoE state (no kernel modules in unified-worker design).
pub struct MoeWorkerGpu {
    pub device: DeviceId,
    /// Per-layer config pointer array on this GPU's VRAM: `MoeWorkerConfig*[num_layers]`.
    /// Indexed by layer_idx; CPU reads `[layer_idx]` and passes the pointer in
    /// `OP_MOE_FFN_REMOTE.config`.
    pub layer_config_ptrs: DeviceBuffer<u64>,
    /// Per-layer config pointer values mirrored on the host (same as the device
    /// buffer above) — CPU dispatch needs these to populate OP_MOE_FFN_REMOTE
    /// without round-tripping to the GPU.
    pub layer_config_ptrs_host: Vec<u64>,
    /// Per-layer config blobs on this GPU's VRAM (kept alive for kernel lifetime).
    _config_storage: Vec<DeviceBuffer<u8>>,
    pub local_activation: DeviceBuffer<f32>,
    pub scratch_gate: DeviceBuffer<f32>,
    pub scratch_up: DeviceBuffer<f32>,
    pub scratch_act: DeviceBuffer<f32>,
    pub local_output: DeviceBuffer<f32>,
}

/// GPU-native P2P MoE dispatch context.
pub struct MoeP2pContext {
    /// Shared work queue in GART memory (MoeWorkItem fixed fields + activation_cache[gate_up_in_dim]).
    pub work_queue: MappedHostBuffer<u8>,
    /// Monotonic dispatch sequence counter (GART, host-mapped u32).
    pub seq_counter: MappedHostBuffer<u32>,
    /// Expert output accumulation buffer. Workers P2P-write expert outputs
    /// here; GPU 0's POST op reads them.
    ///
    /// **Memory class — 2026-05-14 fix for braidinfer-snl §11.4 wedge:**
    /// Was `DeviceBuffer<f32>::alloc_uncached(gpu0, ...)` (GPU 0 UC VRAM).
    /// That made worker writes cross-GPU peer-VRAM UC stores, which per
    /// `kernels/rdna3/rdna3_peer.h` V7 reproducer evidence wedge MES at 4+
    /// GPUs under multi-GPU PCIe pressure. Switched to portable host-
    /// mapped UC: the hazard envelope says host-mapped UC has never
    /// reproduced the wedge across V0/V5 n=10 4-GPU trials. Workers still
    /// write cross-PCIe (now to host memory instead of peer VRAM); GPU 0's
    /// POST reads back through hipHostGetDevicePointer.
    pub output_slots: MappedHostBuffer<f32>,
    /// Per-GPU device pointer to `output_slots` (one per device, including
    /// GPU 0). hipHostGetDevicePointer returns a device-context-specific
    /// pointer; even with `hipHostMallocPortable` the address is valid
    /// only for the GPU context that was current at the time of the call.
    pub output_slots_dev_ptrs: Vec<*mut f32>,
    /// Host-mapped UC staging buffer for the MoE activation input
    /// (`act.normed` for standard MoE, `act.moe_latent` for Nemotron-H).
    ///
    /// **Input-side companion to `output_slots`** (snl 2026-05-15 follow-up).
    /// Workers previously P2P-READ from GPU 0's cached `act.normed` (cross-
    /// GPU peer-VRAM read of cached memory). At 4+ GPUs concurrent worker
    /// reads pressure GPU 0's L2 and the PCIe root-complex non-posted-read
    /// path (ea bridge #242 mechanism: posted-write congestion in the
    /// upstream port stalls non-posted MMIO reads → MES driver-side
    /// timeout). Moving the read source to host-mapped UC removes the
    /// cross-GPU peer path entirely.
    ///
    /// Sized `MAX_PREFILL_BATCH × hidden_size`. Decode uses only first hs.
    pub moe_act_uc_handoff: MappedHostBuffer<f32>,
    /// Per-GPU device pointer to `moe_act_uc_handoff`. Index 0 = GPU 0,
    /// 1.. = workers in worker-index order. Same pattern as
    /// `output_slots_dev_ptrs`.
    pub moe_act_uc_handoff_dev_ptrs: Vec<*mut f32>,
    /// GPU 0 per-layer config pointer array on GPU 0 VRAM: `MoeWorkerConfig*[num_layers]`.
    pub gpu0_layer_config_ptrs: DeviceBuffer<u64>,
    /// GPU 0 per-layer config blobs on GPU 0 VRAM (kept alive).
    _gpu0_config_storage: Vec<DeviceBuffer<u8>>,
    /// Portable host-mapped activation staging buffer: [MAX_PREFILL_BATCH × gate_up_in_dim].
    /// CPU writes activations directly via `host_ptr()` (no DMA, no kernel launch).
    /// Workers P2P-read via per-worker device pointers in `activation_staging_dev_ptrs`.
    /// Replaces a former `DeviceBuffer<f32>::alloc_uncached(gpu0, ...)` which deadlocked
    /// when GPU 0's persistent_worker held all CUs (eh2).
    pub activation_staging: MappedHostBuffer<f32>,
    /// Per-worker device pointer to `activation_staging`. Indexed by worker_idx
    /// (0 = first worker = GPU 1). Even with `hipHostMallocPortable`, the device
    /// address returned by `hipHostGetDevicePointer` is valid only for the GPU
    /// context that was current at the time of the call. We therefore retrieve
    /// one device pointer per worker GPU at init time.
    pub activation_staging_dev_ptrs: Vec<*mut f32>,
    /// GPU 0 scratch buffers for expert computation.
    pub gpu0_scratch_gate: DeviceBuffer<f32>,
    pub gpu0_scratch_up: DeviceBuffer<f32>,
    pub gpu0_scratch_act: DeviceBuffer<f32>,
    /// GPU 0 cached-local accumulator for op_moe_dispatch (PRE). Workers
    /// already stage into local `lo` and barrierless-copy to UC `out_p2p`
    /// at exit (megakernel_moe.hip op_moe_ffn_remote). Mirror that pattern
    /// on GPU 0: stage zero/accumulate into `gpu0_acc` (cached), then
    /// final barrierless copy to UC `output_slots[0..gupd]`. Eliminates
    /// the §11.4 PCIe-write-before-barrier hazard at coop_zero and
    /// per-expert coop_weighted_acc sites. Size = hidden_size (the slot
    /// stride; gupd ≤ hs).
    pub gpu0_acc: DeviceBuffer<f32>,
    /// Per-worker MoE state for GPUs 1..N-1. No kernel modules — workers run
    /// `persistent_worker.hsaco` via `PersistentDispatch`.
    pub workers: Vec<MoeWorkerGpu>,
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
        // DeviceGuard pins the function body to gpu0 and restores the caller's
        // device when this guard drops at function return. The per-worker
        // inner DeviceGuard below temporarily switches to each worker device
        // and restores gpu0 at end of each iteration.
        let _gpu0_guard = DeviceGuard::switch_to(gpu0)?;
        // Allocate GART work queue sized for max batch (MAX_PREFILL_BATCH tokens).
        // The flexible activation_cache tail holds batch_size * gate_up_in_dim floats.
        let wq_size = MOE_WORK_QUEUE_FIXED + MAX_PREFILL_BATCH * gate_up_in_dim * std::mem::size_of::<f32>();
        let work_queue = MappedHostBuffer::<u8>::alloc(wq_size)?;
        let seq_counter = MappedHostBuffer::<u32>::alloc(1)?;
        // Output slots sized for MAX_PREFILL_BATCH × num_gpus × hidden_size.
        // Decode uses only the first (0 * num_gpus + gpu) * hs slot (batch_size=1).
        //
        // 2026-05-14 (braidinfer-snl §11.4 fix): allocated as portable host-
        // mapped UC instead of GPU 0 UC VRAM. kernels/rdna3/rdna3_peer.h
        // documents that cross-GPU peer-VRAM UC stores wedge MES at 4+ GPUs
        // under multi-GPU PCIe pressure (V7 reproducer, 3/10 wedge rate at
        // 4-GPU), but host-mapped UC alone never reproduced the wedge
        // across V0/V5 trials. Per-worker + GPU 0 device pointers via
        // hipHostGetDevicePointer in the worker-launch loop below.
        // alloc_coherent (hipHostMallocCoherent) forces fine-grained UC on
        // BOTH CPU and GPU sides; alloc_portable may use MTYPE_NC (cached)
        // and would defeat the no-L2-staleness intent.
        let output_slots = MappedHostBuffer::<f32>::alloc_portable_coherent(
            MAX_PREFILL_BATCH * num_gpus * hidden_size,
        )?;
        let mut output_slots_dev_ptrs: Vec<*mut f32> = Vec::with_capacity(num_gpus);
        // GPU 0 first.
        let mut gpu0_output_slots_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            braidinfer_hip::error::check(braidinfer_hip::ffi::hipHostGetDevicePointer(
                &mut gpu0_output_slots_dev,
                output_slots.host_ptr() as *mut std::ffi::c_void,
                0,
            ))?;
        }
        output_slots_dev_ptrs.push(gpu0_output_slots_dev as *mut f32);

        // Input-side companion (snl 2026-05-15): host-mapped UC for the
        // MoE activation handoff. Sized for max prefill batch × hidden_size
        // (decode uses first hs only). Same allocator + per-GPU dev-ptr
        // pattern as output_slots.
        let moe_act_uc_handoff = MappedHostBuffer::<f32>::alloc_portable_coherent(
            MAX_PREFILL_BATCH * hidden_size,
        )?;
        let mut moe_act_uc_handoff_dev_ptrs: Vec<*mut f32> = Vec::with_capacity(num_gpus);
        let mut gpu0_handoff_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            braidinfer_hip::error::check(braidinfer_hip::ffi::hipHostGetDevicePointer(
                &mut gpu0_handoff_dev,
                moe_act_uc_handoff.host_ptr() as *mut std::ffi::c_void,
                0,
            ))?;
        }
        moe_act_uc_handoff_dev_ptrs.push(gpu0_handoff_dev as *mut f32);


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

        // Activation staging: portable host-mapped (system RAM, mapped into all
        // GPU contexts). CPU writes directly via host_ptr(); workers P2P-read
        // via device_ptr() over PCIe. Empirical bench (kernels/diagnostic/
        // p2p_read_bw_bench/) shows ~7x faster aggregate bandwidth than UC VRAM
        // P2P at 4 concurrent worker reads (40 GB/s vs 6 GB/s at 64MB).
        // Coherence validated by 77r.2.2 (host_mapped_coh_bench/): worker reads
        // see latest CPU writes both with and without explicit gl1_inv.
        // Required for braidinfer-eh2: copy_from_host deadlocks under GPU 0's
        // persistent_worker; direct CPU writes to host RAM bypass that hazard.
        let activation_staging = MappedHostBuffer::<f32>::alloc_portable(MAX_PREFILL_BATCH * gate_up_in_dim)?;
        // scratch_gate is reused for gate output (eis elements) AND down output (gupd elements).
        // Must be max(eis, gupd) = max(expert_intermediate_size, gate_up_in_dim).
        let scratch_gate_size = expert_intermediate_size.max(gate_up_in_dim);
        let gpu0_scratch_gate = DeviceBuffer::<f32>::alloc(gpu0, scratch_gate_size)?;
        let gpu0_scratch_up = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        let gpu0_scratch_act = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        // §11.4 HAZARD avoidance for op_moe_dispatch — see field doc.
        let gpu0_acc = DeviceBuffer::<f32>::alloc(gpu0, hidden_size)?;

        // Allocate per-worker MoE state on each worker GPU. No kernel launch —
        // workers run `persistent_worker.hsaco` via `PersistentDispatch`, dispatched
        // with `OP_MOE_FFN_REMOTE` from the CPU.
        let _ = shared_mem;
        let _ = kernel_dir;
        let mut workers = Vec::with_capacity(num_workers);
        let mut activation_staging_dev_ptrs: Vec<*mut f32> = Vec::with_capacity(num_workers);
        for (w_idx, &device) in worker_devices.iter().enumerate() {
            let gpu_id = (w_idx + 1) as u32;
            let _worker_guard = DeviceGuard::switch_to(device)?;
            // Get a worker-specific device pointer to the portable host-mapped
            // activation_staging. Even with `hipHostMallocPortable`, AMD ROCm
            // requires this call from each GPU's context to obtain a valid
            // device address for that GPU; the device pointer from a different
            // context can page-fault on access.
            let mut act_dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                braidinfer_hip::error::check(braidinfer_hip::ffi::hipHostGetDevicePointer(
                    &mut act_dev_ptr,
                    activation_staging.host_ptr() as *mut std::ffi::c_void,
                    0,
                ))?;
            }
            activation_staging_dev_ptrs.push(act_dev_ptr as *mut f32);
            // Per-worker device pointer to output_slots (snl §11.4 fix).
            let mut out_dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                braidinfer_hip::error::check(braidinfer_hip::ffi::hipHostGetDevicePointer(
                    &mut out_dev_ptr,
                    output_slots.host_ptr() as *mut std::ffi::c_void,
                    0,
                ))?;
            }
            output_slots_dev_ptrs.push(out_dev_ptr as *mut f32);
            // Per-worker device pointer to the MoE activation handoff buffer
            // (snl input-side fix, follow-up to output_slots).
            let mut handoff_dev: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                braidinfer_hip::error::check(braidinfer_hip::ffi::hipHostGetDevicePointer(
                    &mut handoff_dev,
                    moe_act_uc_handoff.host_ptr() as *mut std::ffi::c_void,
                    0,
                ))?;
            }
            moe_act_uc_handoff_dev_ptrs.push(handoff_dev as *mut f32);

            // §11.18 (d-buffers) per udi #2619: peer-side UTCL2 TLB warm-up.
            // hipHostGetDevicePointer above only resolves the peer's virtual
            // address for the GART page; per §11.18 L1900-1904 the GPU VA TLB
            // entry is committed in this peer's UTCL2 ONLY on first touch. A
            // 4-byte D2H read via hipMemcpy from each peer-view dev_ptr forces
            // that TLB commit during init, before the first decode-time read
            // would otherwise PERMISSION_FAULT or read stale.
            {
                let mut scratch: u32 = 0;
                let scratch_ptr = &mut scratch as *mut u32 as *mut std::ffi::c_void;
                for &peer_view in &[act_dev_ptr, out_dev_ptr, handoff_dev] {
                    unsafe {
                        braidinfer_hip::error::check(braidinfer_hip::ffi::hipMemcpy(
                            scratch_ptr,
                            peer_view,
                            4,
                            braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                        ))?;
                    }
                }
            }

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
            // Mirror the layer config pointer values on the host so CPU dispatch
            // can fill OP_MOE_FFN_REMOTE.config without GPU reads.
            let mut layer_config_ptrs_host = vec![0u64; num_total_layers];
            layer_config_ptrs.copy_to_host(&mut layer_config_ptrs_host)?;

            let local_activation = DeviceBuffer::<f32>::alloc(device, gate_up_in_dim)?;
            // scratch_gate reused for gate output (eis) and down output (gupd): allocate max.
            let scratch_gate = DeviceBuffer::<f32>::alloc(device, scratch_gate_size)?;
            let scratch_up = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let scratch_act = DeviceBuffer::<f32>::alloc(device, expert_intermediate_size)?;
            let local_output = DeviceBuffer::<f32>::alloc(device, hidden_size)?;

            workers.push(MoeWorkerGpu {
                device,
                layer_config_ptrs,
                layer_config_ptrs_host,
                _config_storage: config_storage,
                local_activation,
                scratch_gate,
                scratch_up,
                scratch_act,
                local_output,
            });
            eprintln!("  MoE worker state allocated on GPU {} (no separate kernel — runs via persistent_worker)", gpu_id);
        }

        Ok(MoeP2pContext {
            output_slots_dev_ptrs,
            moe_act_uc_handoff,
            moe_act_uc_handoff_dev_ptrs,
            work_queue,
            seq_counter,
            output_slots,
            gpu0_layer_config_ptrs,
            _gpu0_config_storage: gpu0_config_storage,
            activation_staging,
            activation_staging_dev_ptrs,
            gpu0_scratch_gate,
            gpu0_scratch_up,
            gpu0_scratch_act,
            gpu0_acc,
            workers,
            num_gpus,
            hidden_size,
        })
    }

    /// Worker-specific device pointer to the activation staging buffer.
    /// Use this when building OP_MOE_FFN_REMOTE for `worker_idx` (0-based —
    /// 0 = GPU 1). See field doc on `activation_staging_dev_ptrs`.
    pub fn activation_staging_dev_ptr_for(&self, worker_idx: usize) -> *mut f32 {
        self.activation_staging_dev_ptrs[worker_idx]
    }

    /// Build an OP_MOE_FFN_REMOTE instruction for one token on `worker_idx`
    /// (0-based — worker_idx 0 is GPU 1, etc.). Dispatches into the worker's
    /// persistent_worker mailbox via the caller's `PersistentDispatch`.
    ///
    /// `activation_p2p_dev_ptr`: GPU 0 VRAM activation pointer (worker P2P-reads via this VA).
    /// `output_slot_p2p_dev_ptr`: GPU 0 VRAM output-slot for this (token, worker_idx) pair.
    /// `expert_ids_p2p`, `expert_weights_p2p`: GPU 0 VRAM (or host-mapped) expert routing.
    /// `layer_idx`: index into worker's `layer_config_ptrs` to select per-layer expert config.
    pub(crate) fn build_ffn_remote_inst(
        &self,
        worker_idx: usize,
        layer_idx: usize,
        activation_p2p_dev_ptr: *const f32,
        output_slot_p2p_dev_ptr: *mut f32,
        expert_ids_p2p: *const i32,
        expert_weights_p2p: *const f32,
        k: usize,
        eis: usize,
        hs: usize,
        gupd: usize,
        has_gate_proj: bool,
        relu_sq: bool,
    ) -> crate::megakernel::Instruction {
        let w = &self.workers[worker_idx];
        let cfg_ptr = w.layer_config_ptrs_host[layer_idx] as *const std::ffi::c_void;
        // grid_x is unused inside op_moe_ffn_remote (which uses the full grid).
        crate::megakernel::instructions::MoeFfnRemoteInst::new(
            1, // grid_x — kernel uses cooperative full grid; value irrelevant
            activation_p2p_dev_ptr,
            output_slot_p2p_dev_ptr,
            expert_ids_p2p,
            expert_weights_p2p,
            cfg_ptr,
            w.local_activation.as_ptr() as *mut f32,
            w.local_output.as_ptr() as *mut f32,
            w.scratch_gate.as_ptr() as *mut f32,
            w.scratch_up.as_ptr() as *mut f32,
            w.scratch_act.as_ptr() as *mut f32,
            k as u32,
            eis as u32,
            hs as u32,
            gupd as u32,
            has_gate_proj,
            relu_sq,
        ).into_inst()
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
