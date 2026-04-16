// Persistent cooperative worker kernel for GPU-initiated MoE expert dispatch.
// Launched once per worker GPU (GPUs 1..N-1), polls host-mapped work queue.
// Uses shared coop_* GEMV/activation primitives from moe_expert_ops.h.
//
// Design note: all blocks (1 per CU × 48 CUs = 12288 threads) spin at grid.sync()
// between work items. This is required for cooperative kernel correctness: all blocks
// must reach grid.sync() together, so they must all be running. This consumes the full
// GPU while idle — an intentional tradeoff for low dispatch latency.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "moe_work_queue.h"
#include "moe_expert_ops.h"

// GPU cycle counter. __builtin_readcyclecounter() works on gfx1100 without feature flags.
// On RDNA3 this reads the shader clock (cycles; divide by GPU MHz to get microseconds).
__device__ __forceinline__ uint64_t gpu_clock() {
    return (uint64_t)__builtin_readcyclecounter();
}

// Timing slots: 4 u64 timestamps × N_TIMING_SLOTS iterations.
// Slot layout per iteration i (indices 4*i .. 4*i+3):
//   [0]: t_work_start  — after outer grid.sync() (start of layer work)
//   [1]: t_copy_done   — after coop_copy activation
//   [2]: t_experts_done — after all experts (last coop_weighted_acc)
//   [3]: t_output_done — after coop_copy output to GPU 0
// CPU computes:
//   inner_sync_overhead = t_work_start[i+1] - t_output_done[i] (outer sync + poll)
//   activation_copy     = t_copy_done[i] - t_work_start[i]
//   expert_compute      = t_experts_done[i] - t_copy_done[i]
//   output_copy         = t_output_done[i] - t_experts_done[i]
#define N_TIMING_SLOTS 64
#define TIMING_FIELDS 4

extern "C" __global__ __launch_bounds__(256) void moe_worker_kernel(
    volatile MoeWorkItem* work_queue,
    volatile MoeShutdownFlag* shutdown_flag,
    MoeWorkerConfig** layer_configs,  // per-layer configs indexed by work_queue->layer_idx
    volatile uint32_t* done_flag,     // CPU polls this after shutdown to know kernel exited
    float* local_activation,  // [hidden_size] local VRAM
    float* scratch_gate,      // [expert_intermediate_size] local VRAM
    float* scratch_up,        // [expert_intermediate_size] local VRAM
    float* scratch_act,       // [expert_intermediate_size] local VRAM
    float* local_output,      // [hidden_size] local VRAM — accumulated then P2P written
    uint32_t gpu_id,          // this worker's GPU index (explicit, avoids scanning layer_configs)
    volatile uint64_t* timing_buf  // [N_TIMING_SLOTS * TIMING_FIELDS] GART — NULL to disable
) {
    namespace cg = cooperative_groups;

    // Signal CPU that kernel started before cg::this_grid() (to detect early crash).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *done_flag = 0xAA01u;
        __threadfence_system();
    }

    cg::grid_group grid = cg::this_grid();

    // Signal CPU that kernel is past cg::this_grid() (CPU waits for this before proceeding).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *done_flag = 0xAA02u;
        __threadfence_system();
    }

    uint32_t my_gpu = gpu_id;
    uint32_t last_seq = 0;
    uint32_t timing_slot = 0;

    while (true) {
        // Single-thread poll: only block 0 thread 0 checks host-mapped memory.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (shutdown_flag->shutdown) {
                    work_queue->seq_num = 0xFFFFFFFFu;
                    break;
                }
                uint32_t seq = work_queue->seq_num;
                if (seq > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        grid.sync();

        // Acquire fence: ensures all wq field writes from GPU 0 (done before seq_num RELEASE)
        // are visible to this GPU before we read activation_ptr, output_slots_ptr, etc.
        __threadfence_system();

        uint32_t seq = work_queue->seq_num;
        if (seq == 0xFFFFFFFFu) {
            // Signal CPU that kernel has exited
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                *done_flag = 1u;
            }
            return;
        }
        if (seq <= last_seq) continue;

        // t_work_start: timestamp after outer grid.sync — all blocks have passed the barrier.
        // Recorded by block 0 thread 0; valid across all blocks after grid.sync().
        if (timing_buf && threadIdx.x == 0 && blockIdx.x == 0 && timing_slot < N_TIMING_SLOTS) {
            timing_buf[timing_slot * TIMING_FIELDS + 0] = gpu_clock();
        }

        uint32_t layer_idx = work_queue->layer_idx;
        MoeWorkerConfig* config = layer_configs[layer_idx];

        uint32_t hs = work_queue->hidden_size;
        uint32_t eis = work_queue->expert_intermediate_size;
        uint32_t k = work_queue->num_active;
        uint32_t has_gate = work_queue->has_gate_proj;
        // gate_up_in_dim: expert input dimension (= hidden_size for standard MoE,
        // = moe_latent_size for Nemotron-H). Default to hs for backward compat.
        uint32_t gupd = (work_queue->gate_up_in_dim > 0) ? work_queue->gate_up_in_dim : hs;
        // activation_cache: GPU 0 wrote activation here (GART, bypassing L2) before writing seq_num.
        // Workers read from GART → always see fresh data (no L2-stale issue from GPU 0's VRAM).
        const float* activation_src = (const float*)work_queue->activation_cache;
        float* output_slots = (float*)work_queue->output_slots_ptr;
        // Output slot stride is always hidden_size so GPU 0's sum loop uses consistent offsets.
        float* my_output_slot = output_slots + my_gpu * hs;
        // 1. Copy activation from work queue GART cache to local VRAM (gupd elements)
        coop_copy(local_activation, activation_src, gupd, grid);

        // t_copy_done: after activation copy (one inner grid.sync done).
        if (timing_buf && threadIdx.x == 0 && blockIdx.x == 0 && timing_slot < N_TIMING_SLOTS) {
            timing_buf[timing_slot * TIMING_FIELDS + 1] = gpu_clock();
        }

        grid.sync();

        // 2. Zero local output
        coop_zero(local_output, hs, grid);

        // 3. Process each expert assigned to this GPU
        for (uint32_t j = 0; j < k; j++) {
            int32_t eid = work_queue->expert_ids[j];
            float ew = work_queue->expert_weights[j];
            if (eid < 0 || eid >= 512) continue;

            MoeExpertEntry& entry = config->entries[eid];
            if (entry.gate_up_ptr == 0) continue;

            const unsigned char* gate_up_data = (const unsigned char*)entry.gate_up_ptr;
            const unsigned char* down_data = (const unsigned char*)entry.down_ptr;

            if (has_gate) {
                coop_gemv_pcg32(scratch_gate, gate_up_data,
                                local_activation, eis, gupd, grid);
                const unsigned char* up_data = gate_up_data +
                    (long long)eis * config->gate_up_row_stride;
                coop_gemv_pcg32(scratch_up, up_data,
                                local_activation, eis, gupd, grid);
                coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid);
            } else {
                coop_gemv_pcg32(scratch_up, gate_up_data,
                                local_activation, eis, gupd, grid);
                coop_relu_squared(scratch_act, scratch_up, eis, grid);
            }

            // down: eis → gupd (= gate_up_in_dim, not necessarily hidden_size)
            coop_gemv_pcg32(scratch_gate, down_data,
                            scratch_act, gupd, eis, grid);
            coop_weighted_acc(local_output, scratch_gate, ew, gupd, grid);
        }

        // t_experts_done: after all expert GEMVs.
        if (timing_buf && threadIdx.x == 0 && blockIdx.x == 0 && timing_slot < N_TIMING_SLOTS) {
            timing_buf[timing_slot * TIMING_FIELDS + 2] = gpu_clock();
        }

        // 4. P2P write local_output to GPU 0's output_slots (gupd elements)
        // __threadfence_system() ensures the write is visible to GPU 0 after
        // it polls ack_flags (which polls MTYPE_UC host-mapped memory).
        __threadfence_system();
        coop_copy(my_output_slot, local_output, gupd, grid);

        // t_output_done: after output P2P copy.
        if (timing_buf && threadIdx.x == 0 && blockIdx.x == 0 && timing_slot < N_TIMING_SLOTS) {
            timing_buf[timing_slot * TIMING_FIELDS + 3] = gpu_clock();
            timing_slot++;
        }

        // 5. Ack via host-mapped flag
        // __threadfence_system() after coop_copy ensures P2P writes to output_slots
        // are committed before the ack is visible to GPU 0.
        __threadfence_system();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            work_queue->ack_flags[my_gpu] = seq;
        }

        last_seq = seq;
        grid.sync();
    }
}
