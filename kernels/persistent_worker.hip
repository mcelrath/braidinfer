// Persistent cooperative worker kernel: polls host-mapped work queue for operations.
// Replaces the megakernel — same op implementations, but CPU-scheduled via memory writes.
// One instance per GPU, launched once at init, runs for model lifetime.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#include "moe_work_queue.h"      // MoeWorkerConfig + MoeExpertEntry types for OP_MOE_FFN_REMOTE
#include "watchdog.h"
#include "rdna3_sync.h"          // braidinfer::rdna3::atomic_block_barrier
#include "op_profile.h"          // OP_PROFILE_BEGIN/END (no-op when -DBRAIDINFER_OP_PROFILE unset)

// Per-TU GridBarrierState for atomic_block_barrier. Allocated in cache-coherent
// VRAM by the loader (zero-initialized __device__ static). Replaces grid.sync()
// system-coverage barrier with ~115-155x faster CTA-arrival barrier.
// One state suffices: barrier protocol resets counter on each pass via
// generation flip, no ABA risk for non-overlapping barriers.
__device__ static braidinfer::rdna3::GridBarrierState g_gbs;

// Include all megakernel op implementations (same as megakernel.hip)
#include "megakernel_ops.hip"
#include "moe_expert_ops.h"      // coop_* helpers shared with moe_worker (used by OP_MOE_FFN_REMOTE)
#include "megakernel_moe.hip"
#include "megakernel_moe_dispatch.hip"

// Work queue for CPU → GPU communication (host-mapped memory).
// CPU writes an instruction + increments seq_num. Worker polls seq_num.
#define MAX_BATCH_INSTRUCTIONS 256

struct WorkerQueue {
    volatile uint32_t seq_num;      // monotonic counter, triggers worker
    volatile uint32_t shutdown;     // 1 = exit
    uint32_t num_instructions;      // instructions in this batch (1..64)
    uint32_t _pad;
    uint64_t inst[MAX_BATCH_INSTRUCTIONS * INST_SIZE_WORDS]; // instruction batch
    volatile uint32_t ack;          // worker writes seq_num when done
    volatile uint32_t done;         // kernel writes 1 when exiting (for Drop polling)
    volatile uint32_t progress_pc;  // worker writes pc before each instruction (for timeout diagnosis)
    uint32_t _pad2;
    // op_profile: GPU-resident DeviceBuffer<u64> base pointer, size 2 *
    // BRAIDINFER_OP_PROFILE_NUM_SLOTS. Null when -DBRAIDINFER_OP_PROFILE
    // is unset. See kernels/op_profile.h. Plan: PLAN-op-profile.md.
    uint64_t* op_profile;
};

extern "C" __global__ __launch_bounds__(256, 2) void persistent_worker(
    volatile WorkerQueue* queue,
    WatchdogState* watchdog  // host-mapped watchdog state; NULL to disable
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("persistent_worker: started, queue=%p\n", (void*)queue);
    }

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;
    uint32_t watchdog_beat_ctr = 0;

    while (true) {
        // Mark idle BEFORE entering the inner poll loop. Distinguishes idle
        // wait (no batch yet) from hang for the host watchdog.
        if (watchdog) watchdog_signal_exited(watchdog);

        // Single-thread poll to avoid PCIe storm
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (queue->shutdown) {
                    // Sentinel: set seq to max so all threads see shutdown
                    queue->seq_num = 0xFFFFFFFFu;
                    break;
                }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        // Clear `exited` — kernel is now actively processing a batch.
        if (watchdog) watchdog_init(watchdog, &watchdog_beat_ctr);

        // Watchdog: cooperative exit if force_exit set.
        if (watchdog && watchdog_poll_and_check(watchdog, &g_gbs, 0u)) {
            watchdog_signal_exit(&queue->done);
            return;
        }
        if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);

        uint32_t seq = queue->seq_num;
        if (seq == 0xFFFFFFFFu) {
            if (threadIdx.x == 0 && blockIdx.x == 0) queue->done = 1;
            return;
        }

        uint32_t num_inst = queue->num_instructions;
        if (num_inst == 0) num_inst = 1; // backward compat: 0 means 1

        // bz0 fix: invalidate per-WGP L1 at batch boundary. Activation buffer writes by
        // prior batch may leave stale L1 lines on different WGPs.
        if (threadIdx.x == 0) {
            asm volatile("buffer_gl1_inv" ::: "memory");
        }
        __syncthreads();

        // Process batch of instructions with grid.sync() between each
        for (uint32_t pc = 0; pc < num_inst; pc++) {
        // Report progress so CPU can identify stuck instruction on timeout.
        if (threadIdx.x == 0 && blockIdx.x == 0) queue->progress_pc = pc;
        // Copy instruction to shared memory (avoids PCIe reads in virtual block loop)
        __shared__ u64 sinst[INST_SIZE_WORDS];
        const u64* src = (const u64*)(queue->inst + pc * INST_SIZE_WORDS);
        if (threadIdx.x < INST_SIZE_WORDS) {
            sinst[threadIdx.x] = src[threadIdx.x];
        }
        __syncthreads();
        const u64* inst = sinst;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);

        // Per-instruction-dispatch profiling. "One call" = one opcode
        // dispatch including its post-op grid barrier. Macros are no-ops
        // unless -DBRAIDINFER_OP_PROFILE is set. See PLAN-op-profile.md.
        OP_PROFILE_BEGIN(opcode, queue->op_profile);

        if (opcode == OP_BARRIER) {
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        } else if (opcode == OP_MOE_FFN) {
            op_moe_ffn(inst, grid, &g_gbs);
        } else if (opcode == OP_MOE_FFN_REMOTE) {
            op_moe_ffn_remote(inst, grid, &g_gbs);
        } else if (opcode == OP_MOE_DISPATCH) {
            op_moe_dispatch(inst, grid, &g_gbs, watchdog);
        } else if (opcode == OP_MOE_DISPATCH_POST) {
            op_moe_dispatch_post(inst, grid, &g_gbs, watchdog);
        } else if (opcode == OP_LINEAR_PROJ_PCG32) {
            op_linear_proj_pcg32_full(inst, grid_x);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        } else if (opcode == OP_LINEAR_PROJ_RNF4) {
            op_linear_proj_rnf4_full(inst, grid_x);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        } else {
            // Virtual block loop
            for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
                switch (opcode) {
                    case OP_RMSNORM:      op_rmsnorm(inst, vb); break;
                    case OP_LINEAR_PROJ:  op_linear_proj(inst, vb); break;
                    case OP_CONV1D:       op_conv1d(inst, vb); break;
                    case OP_GDN_GATE:     op_gdn_gate(inst, vb); break;
                    case OP_GDN_RECUR:    op_gdn_recurrent(inst, vb); break;
                    case OP_RMSNORM_GATE: op_rmsnorm_gated(inst, vb); break;
                    case OP_RESIDUAL_ADD: op_residual_add(inst, vb); break;
                    case OP_QK_NORM:      op_qk_norm(inst, vb); break;
                    case OP_MROPE:        op_mrope(inst, vb); break;
                    case OP_GQA_ATTN:     op_gqa_attn(inst, vb); break;
                    case OP_OUTPUT_GATE:  op_output_gate(inst, vb); break;
                    case OP_FFN_GATE_UP:  op_ffn_gate_up(inst, vb); break;
                    case OP_FFN_DOWN_RES: op_ffn_down_res(inst, vb); break;
                    case OP_EMBEDDING:    op_embedding(inst, vb); break;
                    case OP_LM_HEAD:      op_lm_head(inst, vb); break;
                    case OP_D2D_COPY:     op_d2d_copy(inst, vb); break;
                    case OP_ATTN_PAGED:   op_attn_paged(inst, vb); break;
                    case OP_ATTN_PREFILL: op_attn_prefill(inst, vb); break;
                    case OP_DEINTERLEAVE: op_deinterleave(inst, vb); break;
                    case OP_KV_QUANTIZE:  op_kv_quantize(inst, vb); break;
                    case OP_ATTN_PAGED_Q: op_attn_paged_quant(inst, vb); break;
                    case OP_MOE_GATE:    op_moe_gate(inst, vb); break;
                    case OP_LINEAR_PROJ_RNF4:  op_linear_proj_rnf4(inst, vb); break;
                    case OP_LINEAR_PROJ_PCG32: op_linear_proj_pcg32(inst, vb); break;
                    case OP_RMSNORM_WX:  op_rmsnorm_wx(inst, vb); break;
                    case OP_SILU_MUL:    op_silu_mul(inst, vb); break;
                    case OP_FFN_GATE_UP_RNF4:  op_ffn_gate_up_rnf4(inst, vb); break;
                    case OP_FFN_DOWN_RES_RNF4: op_ffn_down_res_rnf4(inst, vb); break;
                    case OP_SIGMOID_WEIGHTED_ADD: op_sigmoid_weighted_add(inst, vb); break;
                    case OP_SCALE_ADD:   op_scale_add(inst, vb); break;
                    case OP_RELU_SQ:     op_relu_sq(inst, vb); break;
                    case OP_MAMBA2_CONV1D:    op_mamba2_conv1d(inst, vb); break;
                    case OP_SSM_UPDATE:       op_ssm_update(inst, vb); break;
                    case OP_MAMBA2_NORM_GATED: op_mamba2_norm_gated(inst, vb); break;
                    case OP_CONV1D_3X:         op_conv1d_3x(inst, vb); break;
                    case OP_FFN_GATE_UP_WX:    op_ffn_gate_up_wx(inst, vb); break;
                    case OP_FFN_GATE_UP_RNF4_WX: op_ffn_gate_up_rnf4_wx(inst, vb); break;
                    case OP_LINEAR_PROJ_2X:    op_linear_proj_2x(inst, vb); break;
                    default: break;
                }
            }
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        }

        OP_PROFILE_END(opcode, queue->op_profile);

        } // end for (pc = 0; pc < num_inst; pc++)

        // Ack: flush ALL threads' writes to system-visible memory, then write ack.
        // grid.sync() ensures all threads completed (device-scope). But P2P readers on
        // GPUs 1-3 bypass GPU 0's L2 and read VRAM directly. grid.sync() alone does NOT
        // flush L2 → VRAM for P2P visibility. __threadfence() from EACH thread
        // ensures that thread's cache lines are written back to system-visible memory.
        // Must be called by ALL threads (not just thread 0,0) to cover all written cache lines.
        __threadfence();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            queue->ack = seq;
        }
        last_seq = seq;
    }
}
