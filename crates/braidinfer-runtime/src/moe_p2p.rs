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
use braidinfer_hip::staging::CrossGpuStaging;

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

/// Per-MoE-layer routing parameters needed by the CPU worker-dispatch path
/// (`dispatch_moe_workers_decode_async`). Populated at megakernel compile time
/// from the same source-of-truth values (model config + DistributedMoeWeights +
/// MoeP2pContext pointers) used to emit OP_MOE_DISPATCH — never read back from
/// compiled instruction words, so the worker-dispatch path is decoupled from
/// the GPU-0 PRE opcode's wire layout (bd 1hik).
///
/// All pointers reference activation buffers / shared P2P buffers whose device
/// addresses are stable for the model's lifetime (allocated once at init).
#[derive(Clone, Copy)]
pub(crate) struct DecodeMoeParams {
    /// `p2p.output_slots_dev_ptrs[0]` — base of [batch × num_gpus × hs] expert
    /// output slot grid. Workers write per-(token, gpu) slots; POST sums them.
    pub output_slots: *mut f32,
    /// `act.moe_expert_ids` device pointer (populated each step by OP_MOE_GATE).
    pub expert_ids: *const i32,
    /// `act.moe_expert_weights` device pointer (populated each step by OP_MOE_GATE).
    pub expert_weights: *const f32,
    /// hidden_size (slot stride: output_slots[gpu_id * hs..gpu_id * hs + hs]).
    pub hs: u32,
    /// gate_up_in_dim (expert input dimension; equals hs for standard MoE,
    /// `moe_latent_size` for Nemotron-H latent MoE).
    pub gupd: u32,
    /// num_active experts per token.
    pub k: u32,
    /// expert_intermediate_size.
    pub eis: u32,
    /// True for gated MoE (gate→silu_mul), false for non-gated (relu²).
    pub has_gate_proj: bool,
    /// Convenience: `!has_gate_proj` — non-gated path uses relu_squared.
    pub relu_sq: bool,
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
    ///
    /// Owned as a [`CrossGpuStaging<f32>`]: the per-GPU device pointers are
    /// resolved at construction and accessed via `.dev_ptr(gpu_idx)`. Indexing
    /// convention: `gpu_idx == gpu_id` (slice order `[gpu0, worker0, worker1,
    /// ...]` passed to `CrossGpuStaging::alloc`). `.host_ptr()` is the CPU
    /// view used for CPU-side zero init and post-step readback.
    pub output_slots: CrossGpuStaging<f32>,
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
    ///
    /// Owned as a [`CrossGpuStaging<f32>`]: indexing convention `gpu_idx ==
    /// gpu_id` (slice order `[gpu0, worker0, worker1, ...]`), same as
    /// [`Self::output_slots`].
    pub moe_act_uc_handoff: CrossGpuStaging<f32>,
    /// GPU 0 per-layer config pointer array on GPU 0 VRAM: `MoeWorkerConfig*[num_layers]`.
    pub gpu0_layer_config_ptrs: DeviceBuffer<u64>,
    /// GPU 0 per-layer config blobs on GPU 0 VRAM (kept alive).
    _gpu0_config_storage: Vec<DeviceBuffer<u8>>,
    /// Portable host-mapped activation staging buffer: [MAX_PREFILL_BATCH × gate_up_in_dim].
    /// CPU writes activations directly via `host_ptr()` (no DMA, no kernel launch).
    /// Workers P2P-read via per-worker device pointers in `activation_staging_dev_ptrs`.
    /// Replaces a former `DeviceBuffer<f32>::alloc_uncached(gpu0, ...)` which deadlocked
    /// when GPU 0's persistent_worker held all CUs (eh2).
    ///
    /// Owned as a [`CrossGpuStaging<f32>`]: same `gpu_idx == gpu_id`
    /// indexing as [`Self::output_slots`]. GPU 0's view is needed because
    /// Step 1's D2D-copy stages prefill_normed into this buffer. Workers
    /// use the [`Self::activation_staging_dev_ptr_for`] accessor (which
    /// remaps worker_idx → gpu_id = worker_idx + 1).
    pub activation_staging: CrossGpuStaging<f32>,
    /// bd 9gmh Phase 1 (udi msg #3440): GPU 0 VRAM UC staging — workers P2P-read this
    /// via the patched MTYPE_UC peer-VRAM mapping (kernel patch 0001 active in
    /// linux-p2p 7.0.9). Host-mapped portable_coherent is asymmetric on gfx1100
    /// multi-GPU: non-allocator-GPU reads of an allocator-GPU's host-mapped UC
    /// see stale data even after CPU/allocator-GPU reads succeed. GPU 0 VRAM with
    /// peer-VRAM-UC mapping fixes this. Sized [MAX_PREFILL_BATCH × gate_up_in_dim].
    pub activation_staging_vram: DeviceBuffer<f32>,
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
    /// bd i7gl: GPU 0 local-activation buffer for self-dispatched OP_MOE_FFN_REMOTE.
    /// Mirrors MoeWorkerGpu.local_activation but on GPU 0. Sized gate_up_in_dim
    /// (= gupd). op_moe_ffn_remote's leading coop_copy stages the activation
    /// from activation_p2p (which on GPU 0 self-points to prefill_normed_dev)
    /// into this buffer before the per-expert GEMV loop.
    pub gpu0_local_activation: DeviceBuffer<f32>,
    /// bd i7gl: host-side copy of gpu0_layer_config_ptrs device values, so
    /// CPU can read per-layer MoeWorkerConfig* without device→host copy
    /// during per-token dispatch. Same pattern as MoeWorkerGpu.layer_config_ptrs_host.
    pub gpu0_layer_config_ptrs_host: Vec<u64>,
    /// bd 9gmh Phase 1: host-mapped per-token expert routing populated by
    /// OP_MOE_GATE in moe_ffn_forward_prefill_batched Step 1. CPU reads
    /// directly to drive Step 2.5 worker dispatches and Step 3 GPU 0 local
    /// expert compute. Sized [MAX_PREFILL_BATCH × MAX_ACTIVE_EXPERTS].
    pub per_token_expert_ids: MappedHostBuffer<i32>,
    pub per_token_expert_weights: MappedHostBuffer<f32>,
    /// bd 9gmh Phase 1 drain sentinel (udi msg #3421): GPU 0 VRAM u32 used as
    /// op_d2d_copy's signal_ptr at end of Step 1. MUST be VRAM, not host-mapped UC —
    /// the kernel's __atomic_store_n(SYSTEM) compiles to buffer_atomic_swap_b32 with
    /// glc=1 expecting a memory-controller return-ack; host UC has no ack generator
    /// → wave wedges in vmcnt-pending → process D-state.
    pub step1_drain_sentinel: DeviceBuffer<u32>,
    /// bd 9gmh Phase 1H: shared-expert dot-product scratch. Used between
    /// OP_LINEAR_PROJ(out_dim=1, in_dim=hs) (computes scratch[0] =
    /// dot(shared_expert_gate, normed)) and OP_SIGMOID_WEIGHTED_ADD
    /// (reads scratch[0]). Single f32 per program; safe to share across
    /// per-token programs since each token waits for its program to
    /// complete before the next dispatches.
    pub shared_expert_gate_scratch: DeviceBuffer<f32>,
    /// bd 9gmh Phase 1G: pre-zeroed f32 buffer used as source for OP_D2D_COPY
    /// when we need to zero ffn_down (replaces hipMemsetAsync on self.stream
    /// which can't run while persistent_worker holds CUs). Size = hidden_size.
    pub gpu0_zero_buffer: DeviceBuffer<f32>,
    /// Per-worker MoE state for GPUs 1..N-1. No kernel modules — workers run
    /// `persistent_worker.hsaco` via `PersistentDispatch`.
    pub workers: Vec<MoeWorkerGpu>,
    pub num_gpus: usize,
    pub hidden_size: usize,
    /// Per-layer MoE routing parameters consumed by the CPU worker-dispatch
    /// path (`dispatch_moe_workers_decode_async`). `None` for non-MoE layers;
    /// `Some(...)` populated by `compile_multi_gpu_p2p` from the same source
    /// of truth used to emit OP_MOE_DISPATCH. Lookup is by `layer_idx`.
    pub(crate) decode_params: Vec<Option<DecodeMoeParams>>,
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
        // §11.4 cross-GPU staging buffers. Indexing convention `gpu_idx ==
        // gpu_id`: slice order is `[gpu0, worker0, worker1, ...]`.
        let mut gpu_id_devices: Vec<DeviceId> = Vec::with_capacity(num_gpus);
        gpu_id_devices.push(gpu0);
        gpu_id_devices.extend_from_slice(worker_devices);
        let mut output_slots = CrossGpuStaging::<f32>::alloc(
            MAX_PREFILL_BATCH * num_gpus * hidden_size,
            &gpu_id_devices,
        )?;

        // Input-side companion (snl 2026-05-15): host-mapped UC for the
        // MoE activation handoff. Sized for max prefill batch × hidden_size
        // (decode uses first hs only). Same allocator + per-GPU dev-ptr
        // pattern as output_slots.
        let mut moe_act_uc_handoff = CrossGpuStaging::<f32>::alloc(
            MAX_PREFILL_BATCH * hidden_size,
            &gpu_id_devices,
        )?;


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
        // bd 9gmh Phase 1F: GPU 0's persistent_worker writes rmsnorm output here;
        // peer GPUs P2P-read in Step 2.5. Must be portable_coherent (MTYPE_UC) —
        // gfx11 has no L2 writeback, so a cached host-mapped allocation would
        // leave worker writes stuck in GPU 0's L2 (per udi 2026-05-22). Replaces
        // a prior SDMA-D2H-from-VRAM design that was non-deterministically reading
        // stale VRAM dirtied at L2.
        // Indexing convention `gpu_idx == gpu_id`: GPU 0 D2D-copies into
        // `.dev_ptr(0)`, worker `w` P2P-reads via `.dev_ptr(w + 1)`. The
        // worker-side accessor `activation_staging_dev_ptr_for(worker_idx)`
        // returns `.dev_ptr(worker_idx + 1)`.
        let mut activation_staging = CrossGpuStaging::<f32>::alloc(
            MAX_PREFILL_BATCH * gate_up_in_dim,
            &gpu_id_devices,
        )?;
        // bd 9gmh Phase 1: GPU 0 VRAM staging — destination for Step 1's final D2D-copy.
        // Workers P2P-read; kernel patch 0001 forces MTYPE_UC for the peer-VRAM mapping
        // so worker reads bypass GPU 0's L2. Local alloc is cached (default) — fast
        // for GPU 0's own writes in Step 1 + reads in Step 3.
        let mut activation_staging_vram = DeviceBuffer::<f32>::alloc(gpu0, MAX_PREFILL_BATCH * gate_up_in_dim)?;
        // scratch_gate is reused for gate output (eis elements) AND down output (gupd elements).
        // Must be max(eis, gupd) = max(expert_intermediate_size, gate_up_in_dim).
        let scratch_gate_size = expert_intermediate_size.max(gate_up_in_dim);
        let mut gpu0_scratch_gate = DeviceBuffer::<f32>::alloc(gpu0, scratch_gate_size)?;
        let mut gpu0_scratch_up = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        let mut gpu0_scratch_act = DeviceBuffer::<f32>::alloc(gpu0, expert_intermediate_size)?;
        // §11.4 HAZARD avoidance for op_moe_dispatch — see field doc.
        let mut gpu0_acc = DeviceBuffer::<f32>::alloc(gpu0, hidden_size)?;
        // bd i7gl: GPU 0 local_activation for self-dispatched OP_MOE_FFN_REMOTE.
        let mut gpu0_local_activation = DeviceBuffer::<f32>::alloc(gpu0, gate_up_in_dim)?;
        // bd i7gl: host-side mirror of gpu0_layer_config_ptrs values for
        // per-token dispatch (avoids device→host copy in hot path).
        let mut gpu0_layer_config_ptrs_host = vec![0u64; num_total_layers];
        gpu0_layer_config_ptrs.copy_to_host(&mut gpu0_layer_config_ptrs_host)?;

        // bd 9gmh Phase 1: per-token routing host-mapped buffers (CPU reads
        // directly to drive worker dispatch). MAX_ACTIVE_EXPERTS = 32 (per
        // moe_p2p constant). 64 * 32 * 4 bytes = 8 KiB each — negligible.
        let per_token_expert_ids = MappedHostBuffer::<i32>::alloc(MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS)?;
        let per_token_expert_weights = MappedHostBuffer::<f32>::alloc(MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS)?;
        // bd 9gmh Phase 1: VRAM sentinel for Step 1's D2D-copy producer-readback drain
        // (per udi msg #3421 — MUST be VRAM, host-mapped UC wedges the SYSTEM-scope atomic).
        let step1_drain_sentinel = DeviceBuffer::<u32>::alloc(gpu0, 1)?;

        // bd 9gmh Phase 1H: 1-element scratch for shared-expert dot product
        // (OP_LINEAR_PROJ(1×hs) writes, OP_SIGMOID_WEIGHTED_ADD reads).
        // bd 9gmh Phase 1: per-token slot, padded to 128-byte L0 cache line stride.
        // Single-element scratch broke: OP_LINEAR_PROJ(grid_x=1) only block 0 writes
        // scratch[0]; OP_SIGMOID_WEIGHTED_ADD (grid_x=8) blocks 1-7 read scratch[0]
        // — their L0 has stale (or uninit garbage = NaN) cached entries. Per-token
        // separate cache lines = each token's read sees its own producer's writes.
        let mut shared_expert_gate_scratch = DeviceBuffer::<f32>::alloc(gpu0, MAX_PREFILL_BATCH * 32)?;

        // bd 9gmh Phase 1G: pre-zeroed VRAM buffer for OP_D2D_COPY-based
        // zeroing (replaces hipMemsetAsync which can't run while persistent
        // worker holds CUs). Initialize once at startup via copy_from_host
        // (safe — runs before any worker spawn).
        let mut gpu0_zero_buffer = DeviceBuffer::<f32>::alloc(gpu0, hidden_size)?;
        let zeros = vec![0.0f32; hidden_size];
        gpu0_zero_buffer.copy_from_host(&zeros)?;

        // bd 9gmh: zero-init all VRAM scratch buffers at allocation. RDNA3 L0
        // caches indexed by VA; cold-miss reads of uninitialized VRAM pages can
        // return arbitrary bit patterns (often NaN-encoded leftovers from prior
        // workloads). For any buffer that a kernel might READ before its first
        // WRITE within a token, the read sees garbage. Zero init makes cold reads
        // return 0.0 deterministically — finite, safe.
        let zeros_big = vec![0.0f32; MAX_PREFILL_BATCH * gate_up_in_dim];
        activation_staging_vram.copy_from_host(&zeros_big[..MAX_PREFILL_BATCH * gate_up_in_dim])?;
        gpu0_scratch_gate.copy_from_host(&zeros_big[..scratch_gate_size])?;
        gpu0_scratch_up.copy_from_host(&zeros_big[..expert_intermediate_size])?;
        gpu0_scratch_act.copy_from_host(&zeros_big[..expert_intermediate_size])?;
        gpu0_acc.copy_from_host(&zeros_big[..hidden_size])?;
        shared_expert_gate_scratch.copy_from_host(&zeros_big[..MAX_PREFILL_BATCH * 32])?;
        // Zero host-mapped UC buffers via CPU writes.
        output_slots.zero();
        moe_act_uc_handoff.zero();
        activation_staging.zero();
        unsafe {
            std::ptr::write_bytes(per_token_expert_ids.host_ptr(), 0, MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS);
            std::ptr::write_bytes(per_token_expert_weights.host_ptr(), 0, MAX_PREFILL_BATCH * MAX_ACTIVE_EXPERTS);
        }
        // step1_drain_sentinel: signal-only, no reads of stale → skip.

        // Allocate per-worker MoE state on each worker GPU. No kernel launch —
        // workers run `persistent_worker.hsaco` via `PersistentDispatch`, dispatched
        // with `OP_MOE_FFN_REMOTE` from the CPU.
        let _ = shared_mem;
        let _ = kernel_dir;
        let mut workers = Vec::with_capacity(num_workers);
        for (w_idx, &device) in worker_devices.iter().enumerate() {
            let gpu_id = (w_idx + 1) as u32;
            let _worker_guard = DeviceGuard::switch_to(device)?;
            // Per-worker peer-views resolved by CrossGpuStaging::alloc above.
            // gpu_id indexing: output_slots/moe_act_uc_handoff use slice
            // `[gpu0, worker0, ...]` (so `gpu_id` directly). activation_staging
            // uses `worker_devices` only (so `w_idx`).
            let act_dev_ptr = activation_staging.dev_ptr(gpu_id as usize) as *mut std::ffi::c_void;
            let out_dev_ptr = output_slots.dev_ptr(gpu_id as usize) as *mut std::ffi::c_void;
            let handoff_dev = moe_act_uc_handoff.dev_ptr(gpu_id as usize) as *mut std::ffi::c_void;

            // §11.18 (d-buffers) per udi #2619: peer-side UTCL2 TLB warm-up.
            // hipHostGetDevicePointer only resolves the peer's virtual
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
            moe_act_uc_handoff,
            work_queue,
            seq_counter,
            output_slots,
            gpu0_layer_config_ptrs,
            _gpu0_config_storage: gpu0_config_storage,
            activation_staging,
            activation_staging_vram,
            gpu0_scratch_gate,
            gpu0_scratch_up,
            gpu0_scratch_act,
            gpu0_acc,
            gpu0_local_activation,
            gpu0_layer_config_ptrs_host,
            per_token_expert_ids,
            per_token_expert_weights,
            step1_drain_sentinel,
            shared_expert_gate_scratch,
            gpu0_zero_buffer,
            workers,
            num_gpus,
            hidden_size,
            // bd 1hik: populated by compile_multi_gpu_p2p; None until then.
            decode_params: vec![None; num_total_layers],
        })
    }

    /// Worker-specific device pointer to the activation staging buffer.
    /// Use this when building OP_MOE_FFN_REMOTE for `worker_idx` (0-based —
    /// 0 = GPU 1). See field doc on `activation_staging_dev_ptrs`.
    pub fn activation_staging_dev_ptr_for(&self, worker_idx: usize) -> *mut f32 {
        // worker_idx 0 = GPU 1 (gpu_id 1) under gpu_idx == gpu_id convention.
        self.activation_staging.dev_ptr(worker_idx + 1)
    }

    /// Worker's local_output VRAM pointer (worker-local addressable).
    /// Used by the BRAIDINFER_MOE_NO_P2P_WRITE diagnostic probe.
    pub fn local_output_ptr_for(&self, worker_idx: usize) -> *mut f32 {
        self.workers[worker_idx].local_output.as_ptr() as *mut f32
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
        // bd el1f Phase A: pair with Step 1's D2dCopyInst::with_signal.
        // wait_ptr=null disables (e.g. decode path); wait_seq should be
        // (layer_idx + 1) for prefill to match the producer's monotonic seq.
        wait_ptr: *const u32,
        wait_seq: u64,
    ) -> crate::megakernel::Instruction {
        let w = &self.workers[worker_idx];
        // bd 0hu3-b: pass the worker-local config_array (worker VRAM, worker
        // VA) and let the kernel index into it. Each worker's array contains
        // VAs valid in its own context.
        let config_array =
            w.layer_config_ptrs.as_ptr() as *const *const std::ffi::c_void;
        // grid_x is unused inside op_moe_ffn_remote (which uses the full grid).
        crate::megakernel::instructions::MoeFfnRemoteInst::new(
            1, // grid_x — kernel uses cooperative full grid; value irrelevant
            activation_p2p_dev_ptr,
            output_slot_p2p_dev_ptr,
            expert_ids_p2p,
            expert_weights_p2p,
            config_array,
            layer_idx as u64,
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
        )
        .with_wait(wait_ptr, wait_seq)
        .into_inst()
    }

    /// bd i7gl: build an OP_MOE_FFN_REMOTE instruction for self-dispatch on
    /// GPU 0. Mirrors `build_ffn_remote_inst` but reads GPU-0-side scratch
    /// buffers (gpu0_*) instead of a `workers[w]` entry. Used by prefill
    /// Step 3 to unify GPU 0's expert compute with the peer-worker code path.
    /// activation_p2p / output_slot_p2p self-point at GPU 0's own buffers
    /// (prefill_normed_dev / output_slots[0]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_ffn_remote_inst_gpu0(
        &self,
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
        wait_ptr: *const u32,
        wait_seq: u64,
    ) -> crate::megakernel::Instruction {
        // bd 0hu3-b: pass GPU 0's config_array (GPU 0 VRAM, GPU 0 VA), kernel
        // dereferences at config_array[layer_idx]. The entries are VAs valid
        // in GPU 0's own context — and this instruction is consumed exclusively
        // by GPU 0 (compile_inner_p2p megakernel + persistent_worker).
        let config_array =
            self.gpu0_layer_config_ptrs.as_ptr() as *const *const std::ffi::c_void;
        crate::megakernel::instructions::MoeFfnRemoteInst::new(
            1, // grid_x — kernel uses cooperative full grid; irrelevant
            activation_p2p_dev_ptr,
            output_slot_p2p_dev_ptr,
            expert_ids_p2p,
            expert_weights_p2p,
            config_array,
            layer_idx as u64,
            self.gpu0_local_activation.as_ptr() as *mut f32,
            self.gpu0_acc.as_ptr() as *mut f32,
            self.gpu0_scratch_gate.as_ptr() as *mut f32,
            self.gpu0_scratch_up.as_ptr() as *mut f32,
            self.gpu0_scratch_act.as_ptr() as *mut f32,
            k as u32,
            eis as u32,
            hs as u32,
            gupd as u32,
            has_gate_proj,
            relu_sq,
        )
        .with_wait(wait_ptr, wait_seq)
        .into_inst()
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
