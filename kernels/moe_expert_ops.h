// Cooperative expert FFN operations shared between megakernel (GPU 0) and
// persistent worker kernels (GPUs 1..N-1).
// All functions are cooperative — all blocks participate, grid.sync() at end.
#pragma once
#include <hip/hip_cooperative_groups.h>
#include "bf16_utils.h"

// Q4 PcG32 GEMV: one row per block in virtual block loop.
// Thread utilization: tpg (threads-per-group) threads cooperate on each quantization group.
// tpg = blockDim.x / num_groups (rounded down to power of 2, capped at 16).
//   in_dim=4096 (num_groups=128): tpg=2, 100% utilization (vs 50% naive).
//   in_dim=2048 (num_groups=64):  tpg=4, 100% utilization (vs 25% naive).
//   in_dim=512  (num_groups=16):  tpg=16, 100% utilization (vs 6% naive).
// Cap at 16: ensures elems_per_thread >= 2 (avoids sub-byte nibble indexing).
__device__ inline void coop_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    // Compute tpg: power-of-2, capped at 16.
    int raw_tpg = blockDim.x / max(num_groups, 1);
    raw_tpg = min(raw_tpg, 16);
    int tpg = 1;
    while (tpg * 2 <= raw_tpg) tpg <<= 1;
    const int groups_per_block = blockDim.x / tpg;
    const int lane = threadIdx.x % tpg;
    const int group_in_block = threadIdx.x / tpg;
    const int elems_per_thread = group_size / tpg; // always even (tpg <= 16, group_size=32)
    const int bytes_per_thread = elems_per_thread / 2;
    const int byte_off = lane * bytes_per_thread;
    extern __shared__ float shared[];

    for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
        float acc = 0.0f;
        for (int g = group_in_block; g < num_groups; g += groups_per_block) {
            const unsigned char* gp = row_data + g * group_bytes;
            float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
            float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
            int elem_base = g * group_size + lane * elems_per_thread;
            int count = min(elems_per_thread, in_dim - elem_base);
            for (int b = 0; b < bytes_per_thread && b * 2 < count; b++) {
                unsigned char byte = gp[byte_off + b];
                float v0 = (float)(byte & 0xF) * sc + mn;
                float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                acc += v0 * input[elem_base + b * 2];
                if (b * 2 + 1 < count) acc += v1 * input[elem_base + b * 2 + 1];
            }
        }
        // Reduce within tpg threads (warp shuffle)
        for (int offset = tpg / 2; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        // lane==0 of each tpg-group stores its partial sum
        if (lane == 0) shared[group_in_block] = acc;
        __syncthreads();
        // Thread 0 sums all groups_per_block partial sums
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int g = 0; g < groups_per_block; g++) total += shared[g];
            output[row] = total;
        }
        __syncthreads();
    }
    grid.sync();
}

__device__ inline void coop_silu_mul(
    float* output, const float* gate, const float* up, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        float g = gate[i];
        output[i] = (g / (1.0f + expf(-g))) * up[i];
    }
    grid.sync();
}

__device__ inline void coop_relu_squared(
    float* output, const float* input, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        float x = input[i];
        float r = x > 0.0f ? x : 0.0f;
        output[i] = r * r;
    }
    grid.sync();
}

__device__ inline void coop_weighted_acc(
    float* output, const float* input, float weight, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        output[i] += weight * input[i];
    }
    grid.sync();
}

__device__ inline void coop_zero(float* buf, int size,
                                  cooperative_groups::grid_group& grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        buf[i] = 0.0f;
    }
    grid.sync();
}

__device__ inline void coop_copy(float* dst, const float* src, int count,
                                  cooperative_groups::grid_group& grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
    grid.sync();
}
