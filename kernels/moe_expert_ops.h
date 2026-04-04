// Cooperative expert FFN operations shared between megakernel (GPU 0) and
// persistent worker kernels (GPUs 1..N-1).
// All functions are cooperative — all blocks participate, grid.sync() at end.
#pragma once
#include <hip/hip_cooperative_groups.h>
#include "bf16_utils.h"

// Q4 PcG32 GEMV: one row per block in virtual block loop.
__device__ inline void coop_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    extern __shared__ float shared[];

    for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
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
