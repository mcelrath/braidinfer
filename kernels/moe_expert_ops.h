// Cooperative expert FFN operations shared between megakernel (GPU 0) and
// persistent worker kernels (GPUs 1..N-1).
// All functions are cooperative — all blocks participate, grid.sync() at end.
#pragma once
#include <hip/hip_cooperative_groups.h>
#include "bf16_utils.h"
#include "quant_consts.h"
#include "rdna3_reduce.h"

// Runtime dispatch for sub-wave sum reduction. tpg ∈ {1,2,4,8,16}.
// W==1: input is already the per-thread "sum" (no reduction needed).
// W∈{2,4,8,16}: dispatch to braidinfer::rdna3::subwave_reduce_sum<W>.
// Same butterfly pairings as the prior `for (offset=tpg/2; offset>0; offset>>=1)
// acc += __shfl_down(acc, offset)` loop, so lane-0 result is bit-exact.
__device__ __forceinline__ float subwave_reduce_dynamic(float v, int tpg) {
    switch (tpg) {
        case 16: return braidinfer::rdna3::subwave_reduce_sum<16>(v);
        case 8:  return braidinfer::rdna3::subwave_reduce_sum<8>(v);
        case 4:  return braidinfer::rdna3::subwave_reduce_sum<4>(v);
        case 2:  return braidinfer::rdna3::subwave_reduce_sum<2>(v);
        default: return v;  // tpg == 1: no reduction
    }
}

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
        acc = subwave_reduce_dynamic(acc, tpg);
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

// Fused gate+up GEMV: compute gate_proj and up_proj simultaneously using all blocks.
// Odd-indexed blocks handle gate rows; even-indexed blocks handle up rows.
// This eliminates one grid.sync() vs sequential coop_gemv_pcg32 calls.
// gate_weight and up_weight must be the same shape (eis × gupd).
__device__ inline void coop_gemv_pcg32_fused_gate_up(
    float* gate_out, const unsigned char* gate_weight,
    float* up_out,   const unsigned char* up_weight,
    const float* input, int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    int raw_tpg = blockDim.x / max(num_groups, 1);
    raw_tpg = min(raw_tpg, 16);
    int tpg = 1;
    while (tpg * 2 <= raw_tpg) tpg <<= 1;
    const int groups_per_block = blockDim.x / tpg;
    const int lane = threadIdx.x % tpg;
    const int group_in_block = threadIdx.x / tpg;
    const int elems_per_thread = group_size / tpg;
    const int bytes_per_thread = elems_per_thread / 2;
    const int byte_off = lane * bytes_per_thread;
    extern __shared__ float shared[];

    // Split blocks: even → gate, odd → up. Each block processes every 2nd row (stride 2 in block space).
    const bool is_gate = (blockIdx.x & 1) == 0;
    const int half_grid = max(gridDim.x / 2, 1);
    const int half_block = blockIdx.x / 2;
    float* my_out = is_gate ? gate_out : up_out;
    const unsigned char* my_weight = is_gate ? gate_weight : up_weight;

    for (int row = half_block; row < out_dim; row += half_grid) {
        const unsigned char* row_data = my_weight + (long long)row * num_groups * group_bytes;
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
        acc = subwave_reduce_dynamic(acc, tpg);
        if (lane == 0) shared[group_in_block] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int g = 0; g < groups_per_block; g++) total += shared[g];
            my_out[row] = total;
        }
        __syncthreads();
    }
    grid.sync();
}

// Q8 RNF4G128 GEMV: one row per block in virtual block loop.
// RNF4 layout: group_size=128, group_bytes=132.
//   Bytes [0..63]:   main NF4 nibbles (128 values, low nibble = even elem, high = odd)
//   Bytes [64..65]:  bf16 absmax1 (main scale)
//   Bytes [66..129]: residual NF4 nibbles (128 values)
//   Bytes [130..131]: bf16 absmax2 (residual scale)
// Dequant: val = NF4_TABLE[nibble1] * absmax1 + NF4_TABLE[nibble2] * absmax2
// Thread utilization: 2 threads per group (group_size=128, each thread covers 64 elems = 32 bytes + 32 residual bytes).
__device__ inline void coop_gemv_rnf4(
    float* output, const unsigned char* weight, const float* input,
    int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    // tpg: power-of-2, capped at 8 (each thread handles 16 pairs = 32 nibbles from main + 32 residual)
    int raw_tpg = blockDim.x / max(num_groups, 1);
    raw_tpg = min(raw_tpg, 8);
    int tpg = 1;
    while (tpg * 2 <= raw_tpg) tpg <<= 1;
    const int groups_per_block = blockDim.x / tpg;
    const int lane = threadIdx.x % tpg;
    const int group_in_block = threadIdx.x / tpg;
    // Each thread covers group_size/tpg elements; packed as group_size/(tpg*2) byte pairs.
    const int elems_per_thread = group_size / tpg;
    const int bytes_per_thread = elems_per_thread / 2; // nibble-pairs per thread
    const int byte_off = lane * bytes_per_thread;      // byte offset into main nibble block
    extern __shared__ float shared[];

    for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
        float acc = 0.0f;
        for (int g = group_in_block; g < num_groups; g += groups_per_block) {
            const unsigned char* gp = row_data + g * group_bytes;
            float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
            float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
            int elem_base = g * group_size + lane * elems_per_thread;
            int count = min(elems_per_thread, in_dim - elem_base);
            for (int b = 0; b < bytes_per_thread && b * 2 < count; b++) {
                unsigned char m = gp[byte_off + b];            // main nibbles
                unsigned char r = gp[66 + byte_off + b];      // residual nibbles (offset 66 = 64 main + 2 absmax1)
                float v0 = NF4_TABLE[m & 0xF] * absmax1 + NF4_TABLE[r & 0xF] * absmax2;
                float v1 = NF4_TABLE[(m >> 4) & 0xF] * absmax1 + NF4_TABLE[(r >> 4) & 0xF] * absmax2;
                acc += v0 * input[elem_base + b * 2];
                if (b * 2 + 1 < count) acc += v1 * input[elem_base + b * 2 + 1];
            }
        }
        acc = subwave_reduce_dynamic(acc, tpg);
        if (lane == 0) shared[group_in_block] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int g = 0; g < groups_per_block; g++) total += shared[g];
            output[row] = total;
        }
        __syncthreads();
    }
    grid.sync();
}

// Fused gate+up RNF4 GEMV: same split as coop_gemv_pcg32_fused_gate_up.
// Even blocks → gate, odd blocks → up.
__device__ inline void coop_gemv_rnf4_fused_gate_up(
    float* gate_out, const unsigned char* gate_weight,
    float* up_out,   const unsigned char* up_weight,
    const float* input, int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    int raw_tpg = blockDim.x / max(num_groups, 1);
    raw_tpg = min(raw_tpg, 8);
    int tpg = 1;
    while (tpg * 2 <= raw_tpg) tpg <<= 1;
    const int groups_per_block = blockDim.x / tpg;
    const int lane = threadIdx.x % tpg;
    const int group_in_block = threadIdx.x / tpg;
    const int elems_per_thread = group_size / tpg;
    const int bytes_per_thread = elems_per_thread / 2;
    const int byte_off = lane * bytes_per_thread;
    extern __shared__ float shared[];

    const bool is_gate = (blockIdx.x & 1) == 0;
    const int half_grid = max(gridDim.x / 2, 1);
    const int half_block = blockIdx.x / 2;
    float* my_out = is_gate ? gate_out : up_out;
    const unsigned char* my_weight = is_gate ? gate_weight : up_weight;

    for (int row = half_block; row < out_dim; row += half_grid) {
        const unsigned char* row_data = my_weight + (long long)row * num_groups * group_bytes;
        float acc = 0.0f;
        for (int g = group_in_block; g < num_groups; g += groups_per_block) {
            const unsigned char* gp = row_data + g * group_bytes;
            float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
            float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
            int elem_base = g * group_size + lane * elems_per_thread;
            int count = min(elems_per_thread, in_dim - elem_base);
            for (int b = 0; b < bytes_per_thread && b * 2 < count; b++) {
                unsigned char m = gp[byte_off + b];
                unsigned char r = gp[66 + byte_off + b];
                float v0 = NF4_TABLE[m & 0xF] * absmax1 + NF4_TABLE[r & 0xF] * absmax2;
                float v1 = NF4_TABLE[(m >> 4) & 0xF] * absmax1 + NF4_TABLE[(r >> 4) & 0xF] * absmax2;
                acc += v0 * input[elem_base + b * 2];
                if (b * 2 + 1 < count) acc += v1 * input[elem_base + b * 2 + 1];
            }
        }
        acc = subwave_reduce_dynamic(acc, tpg);
        if (lane == 0) shared[group_in_block] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int g = 0; g < groups_per_block; g++) total += shared[g];
            my_out[row] = total;
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
