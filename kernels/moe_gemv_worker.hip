// Lean MoE GEMV worker: handles OP_EXPERT_FFN with high block count.
// Uses virtual block loops and grid.sync() between phases — same pattern as
// the fat persistent_worker.hip OP_EXPERT_FFN handling.
// Cooperative launch: grid.sync at top of outer loop ensures all blocks
// converge before block 0 polls, so inner grid.sync() in phases is safe.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#include "moe_expert_ops.h"
#include "watchdog.h"
#include "rdna3_reduce.h"
#include "rdna3_sync.h"

// Per-TU GridBarrierState: zero-init __device__ static, cache-coherent VRAM.
__device__ static braidinfer::rdna3::GridBarrierState g_gbs;

// Phase 2' cross-block shutdown flag (AGENT-scope cached VRAM).
// Set by block 0 thread 0 in poll loop; read by all blocks post-barrier.
// Same pattern as g_shutdown_seen in megakernel.hip (commit 8cf8084).
__device__ static uint32_t g_mgw_shutdown_seen;

#define MOE_MAX_BATCH 16

struct MoeWorkerQueue {
    volatile uint32_t seq_num;
    volatile uint32_t shutdown;
    uint32_t num_instructions;
    uint32_t _pad;
    uint64_t inst[MOE_MAX_BATCH * INST_SIZE_WORDS];
    volatile uint32_t ack;
    volatile uint32_t debug_stage;  // written by GPU for CPU-side diagnostics
    volatile uint32_t done;         // kernel writes 1 when exiting (for Drop polling)
    uint32_t _pad2[1];
};

// Per-row Q4 PcG32 GEMV helper.
// Computes output[row] = dot(weight_row, input). One call per row.
__device__ static void lean_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int row, int out_dim, int in_dim
) {
    if (row >= out_dim) return;
    const int tid = threadIdx.x, stride = blockDim.x;
    const int gs = 32, gb = 20;
    const int ng = (in_dim + gs - 1) / gs;
    const unsigned char* rd = weight + (long long)row * ng * gb;
    extern __shared__ float shm[];
    float acc = 0.0f;
    for (int g = tid; g < ng; g += stride) {
        const unsigned char* gp = rd + g * gb;
        float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
        float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
        int base = g * gs;
        int cnt = min(gs, in_dim - base);
        for (int i = 0; i < cnt; i += 2) {
            unsigned char byte = gp[i / 2];
            acc += ((float)(byte & 0xF) * sc + mn) * input[base + i]
                 + ((float)((byte >> 4) & 0xF) * sc + mn) * input[base + i + 1];
        }
    }
    // Wave32 sum (DPP butterfly — same pairings as the prior shfl_down loop).
    acc = braidinfer::rdna3::wave32_reduce_sum(acc);
    if ((tid & 31) == 0) shm[tid >> 5] = acc;
    __syncthreads();
    // 8-warp combine: load shm[0..7] into the first 8 lanes of wave 0 and run
    // an 8-wide subwave butterfly. Lane-0 result is bit-exact with the prior
    // `if (tid<8) { ... shfl 4,2,1 ... }` butterfly (same pairings on the
    // active lanes). DPP needs all source lanes to have valid input, so we
    // load 0.0f for lanes >= 8.
    {
        float v8 = (tid < 8) ? shm[tid] : 0.0f;
        v8 = braidinfer::rdna3::subwave_reduce_sum<8>(v8);
        if (tid == 0) output[row] = v8;
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(256) void moe_gemv_worker(
    volatile MoeWorkerQueue* queue,
    float* scratch_gate,
    float* scratch_up,
    float* scratch_act,
    WatchdogState* watchdog  // host-mapped watchdog state; NULL to disable
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;
    uint32_t watchdog_beat_ctr = 0;

    while (true) {
        // grid.sync() at TOP — all blocks converge here before block 0 polls.
        // Same pattern as persistent_worker.hip. This ensures inner grid.sync()
        // calls in the OP_EXPERT_FFN phases are safe.
        //
        // Phase 2' fix: NO UC writes inside the poll loop or before this
        // barrier. queue->seq_num=sentinel deferred to after the barrier
        // (same 4fg.5 pattern as megakernel.hip). Shutdown communicated via
        // g_mgw_shutdown_seen (__device__ global, AGENT-scope VRAM, no PCIe).
        // The barrier's RELEASE+ACQUIRE fence pair makes it visible to all
        // blocks after the barrier returns.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (queue->shutdown) { g_mgw_shutdown_seen = 1u; break; }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);

        // Phase 2' deferred-ack: write ack for the PREVIOUS batch AFTER
        // the barrier so no vscnt stall precedes this barrier. AGENT scope:
        // UC memory reaches host DRAM via PCIe regardless of scope; SYSTEM
        // scope would emit vscnt before the watchdog/dispatch barrier that
        // follows. last_seq == 0 on first iteration; writing ack=0 is no-op.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            braidinfer::rdna3::host_uc_store_agent(&queue->ack, last_seq);
        }

        // Shutdown: sentinel + done written POST-barrier (4fg.5 pattern).
        // No barrier follows these writes so no vscnt stall.
        // All blocks read g_mgw_shutdown_seen (set pre-barrier, visible post-
        // barrier via the fence pair), so all blocks return together.
        if (g_mgw_shutdown_seen) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                queue->seq_num = 0xFFFFFFFFu;
                queue->done = 1;
            }
            return;
        }

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
        if (num_inst == 0) num_inst = 1;

        __shared__ u64 sinst[INST_SIZE_WORDS];
        for (uint32_t pc = 0; pc < num_inst; pc++) {
            if (threadIdx.x < INST_SIZE_WORDS)
                sinst[threadIdx.x] = ((const u64*)(queue->inst))[pc * INST_SIZE_WORDS + threadIdx.x];
            __syncthreads();

            // OP_EXPERT_FFN: gate+up+silu+down with virtual block loops and grid.sync().
            // inst[1]=sg(unused), [2]=su(unused), [3]=sa(unused), [4]=output (host-mapped),
            // [5]=gate_up, [6]=down, [7]=activation, [8]=(eis<<32)|hs,
            // [9]=gu_row_stride, [10]=has_gate, [11]=expert_weight (f32 bits)
            float* output = (float*)(uintptr_t)sinst[4];
            const unsigned char* gate_up = (const unsigned char*)(uintptr_t)sinst[5];
            const unsigned char* down    = (const unsigned char*)(uintptr_t)sinst[6];
            const float* activation = (const float*)(uintptr_t)sinst[7];
            int eis = (int)(sinst[8] >> 32), hs = (int)(sinst[8] & 0xFFFFFFFF);
            int gu_rs = (int)sinst[9], has_gate = (int)sinst[10];
            u32 ew_bits = (u32)sinst[11];
            float ew; __builtin_memcpy(&ew, &ew_bits, sizeof(float));

            if (has_gate) {
                // Gate proj → scratch_gate[0..eis]
                for (int row = blockIdx.x; row < eis; row += gridDim.x)
                    lean_gemv_pcg32(scratch_gate, gate_up, activation, row, eis, hs);
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
                // Up proj → scratch_up[0..eis]
                const unsigned char* up_data = gate_up + (long long)eis * gu_rs;
                for (int row = blockIdx.x; row < eis; row += gridDim.x)
                    lean_gemv_pcg32(scratch_up, up_data, activation, row, eis, hs);
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
                // SiLU: scratch_act[i] = silu(scratch_gate[i]) * scratch_up[i]
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis; i += gridDim.x * blockDim.x) {
                    float g = scratch_gate[i];
                    scratch_act[i] = (g / (1.0f + expf(-g))) * scratch_up[i];
                }
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            } else {
                // Up proj only → scratch_up[0..eis]
                for (int row = blockIdx.x; row < eis; row += gridDim.x)
                    lean_gemv_pcg32(scratch_up, gate_up, activation, row, eis, hs);
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
                // ReLU²: scratch_act[i] = relu(scratch_up[i])²
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis; i += gridDim.x * blockDim.x) {
                    float x = scratch_up[i]; float r = x > 0.0f ? x : 0.0f; scratch_act[i] = r * r;
                }
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            }

            // Down proj: scratch_gate[0..hs] = down × scratch_act
            for (int row = blockIdx.x; row < hs; row += gridDim.x)
                lean_gemv_pcg32(scratch_gate, down, scratch_act, row, hs, eis);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            // Weighted accumulate: output[i] += ew * scratch_gate[i]
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hs; i += gridDim.x * blockDim.x)
                output[i] += ew * scratch_gate[i];
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        }

        // Phase 2': queue->ack is written at the TOP of the next outer-loop
        // iteration AFTER the barrier (not here). See deferred-ack comment above.
        last_seq = seq;
    }
}
