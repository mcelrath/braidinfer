// Lean MoE GEMV worker: handles OP_EXPERT_FFN with high block count.
// Uses per-block scratch buffers to avoid cross-block grid.sync() in gate→silu pipeline.
// Pattern: grid.sync() at TOP of outer loop — same as fat persistent_worker.hip.
// This ensures all blocks sync on every iteration before block 0 polls.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#include "moe_expert_ops.h"

#define MOE_MAX_BATCH 16

struct MoeWorkerQueue {
    volatile uint32_t seq_num;
    volatile uint32_t shutdown;
    uint32_t num_instructions;
    uint32_t _pad;
    uint64_t inst[MOE_MAX_BATCH * INST_SIZE_WORDS];
    volatile uint32_t ack;
    uint32_t _pad2[3];
};

// Per-row Q4 PcG32 GEMV helper (same as in moe_worker.hip).
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
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if ((tid & 31) == 0) shm[tid >> 5] = acc;
    __syncthreads();
    if (tid < 8) {
        acc = shm[tid];
        for (int o = 4; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if (tid == 0) output[row] = acc;
    }
    __syncthreads();
}

// OP_EXPERT_FFN per-block: gate+up+silu+down with per-block scratch (no grid.sync).
// inst[1]=sg, [2]=su, [3]=sa (per-block slices), [4]=output, [5]=gate_up, [6]=down,
// [7]=activation, [8]=(eis<<32)|hs, [9]=gu_row_stride, [10]=has_gate, [11]=ew bits.
__device__ static void lean_expert_ffn_block(const u64* inst) {
    float* sg = (float*)(uintptr_t)inst[1];
    float* su = (float*)(uintptr_t)inst[2];
    float* sa = (float*)(uintptr_t)inst[3];
    float* output = (float*)(uintptr_t)inst[4];
    const unsigned char* gate_up = (const unsigned char*)(uintptr_t)inst[5];
    const unsigned char* down    = (const unsigned char*)(uintptr_t)inst[6];
    const float* activation = (const float*)(uintptr_t)inst[7];
    int eis = (int)(inst[8] >> 32), hs = (int)(inst[8] & 0xFFFFFFFF);
    int gu_rs = (int)inst[9], has_gate = (int)inst[10];
    u32 ew_bits = (u32)inst[11];
    float ew; __builtin_memcpy(&ew, &ew_bits, sizeof(float));

    if (has_gate) {
        for (int r = 0; r < eis; r++) lean_gemv_pcg32(sg, gate_up, activation, r, eis, hs);
        for (int r = 0; r < eis; r++) lean_gemv_pcg32(su, gate_up + (long long)eis * gu_rs, activation, r, eis, hs);
        for (int i = threadIdx.x; i < eis; i += blockDim.x) {
            float g = sg[i]; sa[i] = (g / (1.0f + expf(-g))) * su[i];
        }
        __syncthreads();
    } else {
        for (int r = 0; r < eis; r++) lean_gemv_pcg32(su, gate_up, activation, r, eis, hs);
        for (int i = threadIdx.x; i < eis; i += blockDim.x) {
            float x = su[i]; float r = x > 0.0f ? x : 0.0f; sa[i] = r * r;
        }
        __syncthreads();
    }
    // Down proj: each block computes its own rows via virtual block loop
    for (int r = blockIdx.x; r < hs; r += gridDim.x) {
        lean_gemv_pcg32(sg, down, sa, r, hs, eis);
        // sg[r] now holds dot(down_row_r, sa). BUT sg is shared with gate/up by offset 0.
        // sg has eis slots. r can be > eis for hs > eis. Need separate output.
        // SIMPLEST: use output directly via inline GEMV with ew scaling
    }
    // Actually, inline the down+accumulate to avoid needing a separate buffer:
    // Each block writes to disjoint output rows (no race).
    extern __shared__ float shm[];
    for (int row = blockIdx.x; row < hs; row += gridDim.x) {
        const int tid = threadIdx.x, stride = blockDim.x;
        const int gs = 32, gb = 20, ng = (eis + gs - 1) / gs;
        const unsigned char* rd = down + (long long)row * ng * gb;
        float acc = 0.0f;
        for (int g = tid; g < ng; g += stride) {
            const unsigned char* gp = rd + g * gb;
            float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
            float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
            int base = g * gs, cnt = min(gs, eis - base);
            for (int i = 0; i < cnt; i += 2) {
                unsigned char byte = gp[i / 2];
                acc += ((float)(byte & 0xF) * sc + mn) * sa[base + i]
                     + ((float)((byte >> 4) & 0xF) * sc + mn) * sa[base + i + 1];
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if ((tid & 31) == 0) shm[tid >> 5] = acc;
        __syncthreads();
        if (tid < 8) {
            acc = shm[tid];
            for (int o = 4; o > 0; o >>= 1) acc += __shfl_down(acc, o);
            if (tid == 0) output[row] += ew * acc; // each block writes DIFFERENT rows
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256) void moe_gemv_worker(
    volatile MoeWorkerQueue* queue,
    float* scratch_gate_all,
    float* scratch_up_all,
    float* scratch_act_all
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;

    while (true) {
        // Block 0 thread 0 polls. All other blocks idle here (they reach grid.sync below).
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (queue->shutdown) { queue->seq_num = 0xFFFFFFFFu; break; }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        // ALL blocks converge: block 0 arrives after poll, others from previous grid.sync
        grid.sync();

        uint32_t seq = queue->seq_num;
        if (seq == 0xFFFFFFFFu) return;

        uint32_t num_inst = queue->num_instructions;
        if (num_inst == 0) num_inst = 1;

        __shared__ u64 sinst[INST_SIZE_WORDS];
        for (uint32_t pc = 0; pc < num_inst; pc++) {
            if (threadIdx.x < INST_SIZE_WORDS)
                sinst[threadIdx.x] = ((const u64*)(queue->inst))[pc * INST_SIZE_WORDS + threadIdx.x];
            __syncthreads();

            // Per-block scratch slices: each block gets its own eis-sized gate/up/act buffers
            int eis = (int)(sinst[8] >> 32);
            float* sg = scratch_gate_all + (long long)blockIdx.x * eis;
            float* su = scratch_up_all   + (long long)blockIdx.x * eis;
            float* sa = scratch_act_all  + (long long)blockIdx.x * eis;

            u64 s1 = sinst[1], s2 = sinst[2], s3 = sinst[3];
            sinst[1] = (u64)sg; sinst[2] = (u64)su; sinst[3] = (u64)sa;
            lean_expert_ffn_block(sinst);
            sinst[1] = s1; sinst[2] = s2; sinst[3] = s3;
        }

        // All blocks done — sync before ack
        grid.sync();

        last_seq = seq;
        if (threadIdx.x == 0 && blockIdx.x == 0)
            queue->ack = seq;
    }
}
