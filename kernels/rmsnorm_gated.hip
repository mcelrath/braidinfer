#include <hip/hip_runtime.h>

// Gated RMSNorm (norm_before_gate=True): output = rms_norm(x) * weight * silu(z)
// Used by GDN (Qwen3.5) output norm.
// x, z: [num_heads, value_dim]
// weight: [value_dim]
// output: [num_heads, value_dim]
// One block per head.

extern "C" __global__ void rmsnorm_gated_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,  // f32 weight (not BF16) — output_norm/linear_attn.norm are the only f32 weights in the model
    const int num_heads,
    const int value_dim,
    const float eps
) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xh = x + h * value_dim;
    const float* zh = z + h * value_dim;
    float* oh = output + h * value_dim;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < value_dim; i += stride) {
        float v = xh[i];
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

    float rms = rsqrtf(shared[0] / (float)value_dim + eps);

    for (int i = tid; i < value_dim; i += stride) {
        float normed = xh[i] * rms * weight[i];
        float zi = zh[i];
        float silu_z = zi / (1.0f + expf(-zi));
        oh[i] = normed * silu_z;
    }
}

// Gated RMSNorm (norm_before_gate=False): output = rms_norm(x * silu(z)) * weight
// Used by Mamba2 (Nemotron-H) output norm.
// Norm is computed over the gated product, not the raw x.
extern "C" __global__ void rmsnorm_gated_post_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    const int num_heads,
    const int value_dim,
    const float eps
) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xh = x + h * value_dim;
    const float* zh = z + h * value_dim;
    float* oh = output + h * value_dim;

    extern __shared__ float shared[];

    // Pass 1: compute sum_sq of (x * silu(z))
    float sum_sq = 0.0f;
    for (int i = tid; i < value_dim; i += stride) {
        float zi = zh[i];
        float silu_z = zi / (1.0f + expf(-zi));
        float v = xh[i] * silu_z;
        sum_sq += v * v;
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)value_dim + eps);

    // Pass 2: recompute x * silu(z), apply norm and weight
    for (int i = tid; i < value_dim; i += stride) {
        float zi = zh[i];
        float silu_z = zi / (1.0f + expf(-zi));
        float v = xh[i] * silu_z;
        oh[i] = v * rms * weight[i];
    }
}
