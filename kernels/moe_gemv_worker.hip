// Lean persistent MoE GEMV worker: handles ONLY OP_EXPERT_FFN.
// Minimal register footprint for maximum occupancy (~60 VGPRs → 6 blocks/CU → 576 blocks).
// Separate from the fat persistent_worker.hip which includes all 25+ ops and uses 251 VGPRs.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"

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

// Lean Q4 PcG32 GEMV: one row per virtual block, shared memory reduction.
// Identical to op_linear_proj_pcg32 but standalone (no fat switch context).
__device__ static void lean_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int row, int out_dim, int in_dim
) {
    if (row >= out_dim) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
    extern __shared__ float shared[];

    float acc = 0.0f;
    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp = row_data + g * group_bytes;
        float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
        float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
        int base = g * group_size;
        int count = min(group_size, in_dim - base);
        for (int i = 0; i < count; i += 2) {
            unsigned char byte = gp[i / 2];
            float v0 = (float)(byte & 0xF) * sc + mn;
            float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
            acc += v0 * input[base + i] + v1 * input[base + i + 1];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if ((tid & 31) == 0) shared[tid >> 5] = acc;
    __syncthreads();
    if (tid < 8) {
        acc = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) output[row] = acc;
    }
    __syncthreads();
}

// OP_EXPERT_FFN: gate+up+silu+down, weighted accumulate (requires grid.sync).
__device__ void lean_expert_ffn(const u64* inst, cooperative_groups::grid_group& grid) {
    float* sg = (float*)(uintptr_t)inst[1];
    float* su = (float*)(uintptr_t)inst[2];
    float* sa = (float*)(uintptr_t)inst[3];
    float* output = (float*)(uintptr_t)inst[4];
    const unsigned char* gate_up = (const unsigned char*)(uintptr_t)inst[5];
    const unsigned char* down = (const unsigned char*)(uintptr_t)inst[6];
    const float* activation = (const float*)(uintptr_t)inst[7];
    int eis = (int)(inst[8] >> 32);
    int hs = (int)(inst[8] & 0xFFFFFFFF);
    int gu_row_stride = (int)inst[9];
    int has_gate = (int)inst[10];
    u32 ew_bits = (u32)inst[11];
    float ew; __builtin_memcpy(&ew, &ew_bits, sizeof(float));

    if (has_gate) {
        for (int row = blockIdx.x; row < eis; row += gridDim.x)
            lean_gemv_pcg32(sg, gate_up, activation, row, eis, hs);
        grid.sync();
        const unsigned char* up_data = gate_up + (long long)eis * gu_row_stride;
        for (int row = blockIdx.x; row < eis; row += gridDim.x)
            lean_gemv_pcg32(su, up_data, activation, row, eis, hs);
        grid.sync();
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis; i += gridDim.x * blockDim.x) {
            float g = sg[i]; sa[i] = (g / (1.0f + expf(-g))) * su[i];
        }
        grid.sync();
    } else {
        for (int row = blockIdx.x; row < eis; row += gridDim.x)
            lean_gemv_pcg32(su, gate_up, activation, row, eis, hs);
        grid.sync();
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis; i += gridDim.x * blockDim.x) {
            float x = su[i]; float r = x > 0.0f ? x : 0.0f; sa[i] = r * r;
        }
        grid.sync();
    }
    for (int row = blockIdx.x; row < hs; row += gridDim.x)
        lean_gemv_pcg32(sg, down, sa, row, hs, eis);
    grid.sync();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hs; i += gridDim.x * blockDim.x)
        output[i] += ew * sg[i];
    grid.sync();
}

// Target 6 blocks/CU for ~60 VGPRs. If compiler needs more, it spills to scratch.
extern "C" __global__ __launch_bounds__(256) void moe_gemv_worker(
    volatile MoeWorkerQueue* queue
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // Minimal poll-and-ack. Expert FFN computation is done by GPU 0's fat worker.
    // TODO: implement cross-block-safe expert FFN for high-occupancy MoE dispatch.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t last = 0;
        while (true) {
            if (queue->shutdown) return;
            uint32_t s = queue->seq_num;
            if (s != last) {
                if (s == 0xFFFFFFFFu) return;
                last = s;
                queue->ack = s;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
}
