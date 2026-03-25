#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Per-head RMSNorm applied to Q and K after projection, before RoPE.
// One block per head. Reduction via shared memory.
//
// q:        [num_q_heads, head_dim]  f32 (in-place)
// k:        [num_kv_heads, head_dim] f32 (in-place)
// q_weight: [head_dim]              BF16
// k_weight: [head_dim]              BF16
// Grid: num_q_heads + num_kv_heads blocks

extern "C" __global__ void qk_norm_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const unsigned short* __restrict__ q_weight,
    const unsigned short* __restrict__ k_weight,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float eps
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float* head_ptr;
    const unsigned short* weight_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + head_idx * head_dim;
        weight_ptr = q_weight;
    } else {
        head_ptr = k + (head_idx - num_q_heads) * head_dim;
        weight_ptr = k_weight;
    }

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float v = head_ptr[i];
        sum_sq += v * v;
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += stride) {
        head_ptr[i] = head_ptr[i] * rms * (1.0f + bf16_to_f32(weight_ptr[i]));
    }
}
