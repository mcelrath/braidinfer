// MoE prefill helpers: gather, scatter-accumulate, and batched PCG32Q4 matmul.
// Used by single-GPU batched MoE prefill to amortize expert weight loading across tokens.
#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Gather count rows from [n x in_dim] source into [count x in_dim] output.
// indices[j] is the source row index for output row j.
extern "C" __global__ void moe_gather_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ indices,
    const int count,
    const int in_dim
) {
    int j = blockIdx.x;
    if (j >= count) return;
    int src_row = indices[j];
    const float* src = input + (long long)src_row * in_dim;
    float* dst = output + (long long)j * in_dim;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Scatter-add weighted: for j in 0..count, output[indices[j]*out_dim..] += weights[j]*input[j*out_dim..]
// output is [n x out_dim], input is [count x out_dim]. Sequential experts guarantee no conflict.
extern "C" __global__ void moe_scatter_add_weighted_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const int count,
    const int out_dim
) {
    int j = blockIdx.x;
    if (j >= count) return;
    int dst_row = indices[j];
    float w = weights[j];
    const float* src = input + (long long)j * out_dim;
    float* dst = output + (long long)dst_row * out_dim;
    for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
        dst[i] += w * src[i];
    }
}

// Zero [n x dim] batch output buffer. Used to initialize ffn_out before scatter.
extern "C" __global__ void moe_zero_batch_f32(
    float* __restrict__ output,
    const int total_elems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) output[idx] = 0.0f;
}

// Batched PCG32Q4 matmul: output[batch x out_dim] = weight[out_dim x in_dim] @ input[batch x in_dim]
// One block per output row; loops over batch inside.
// Weight format: PCG32Q4 (group_size=32, 20 bytes/group: 16 bytes nibble-packed + 2 bf16 min + 2 bf16 scale)
extern "C" __global__ void linear_proj_pcg32_batched(
    float* __restrict__ output,     // [batch x out_dim]
    const unsigned char* __restrict__ weight, // PCG32Q4 packed
    const float* __restrict__ input, // [batch x in_dim]
    const int out_dim,
    const int in_dim,
    const int batch
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;

    extern __shared__ float shared[];

    for (int row = (int)blockIdx.x; row < out_dim; row += (int)gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
        for (int b = 0; b < batch; b++) {
            const float* inp = input + (long long)b * in_dim;
            float acc = 0.0f;
            for (int g = tid; g < num_groups; g += stride) {
                const unsigned char* gp = row_data + (long long)g * group_bytes;
                float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                int base = g * group_size;
                int cnt = (base + group_size <= in_dim) ? group_size : (in_dim - base);
                for (int i = 0; i < cnt; i += 2) {
                    unsigned char byte = gp[i / 2];
                    float v0 = (float)(byte & 0xF) * sc + mn;
                    float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                    acc += v0 * inp[base + i];
                    if (base + i + 1 < in_dim) acc += v1 * inp[base + i + 1];
                }
            }
            shared[tid] = acc;
            __syncthreads();
            for (int s = stride / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            if (tid == 0) output[(long long)b * out_dim + row] = shared[0];
            __syncthreads();
        }
    }
}

// Batched silu: output[i] = silu(gate[i]) * up[i] for flat arrays of total_size = batch * eis
extern "C" __global__ void silu_mul_batched_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = silu_g * up[idx];
}

// Batched relu²: output[i] = relu(input[i])²
extern "C" __global__ void relu_sq_batched_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    float x = input[idx];
    float r = (x > 0.0f) ? x : 0.0f;
    output[idx] = r * r;
}
