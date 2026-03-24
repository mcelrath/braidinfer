#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
// One block per row. Reduction within block via shared memory.

extern "C" __global__ void rmsnorm_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + row * hidden_size;
    float* y = output + row * hidden_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Block reduction
    extern __shared__ float shared[];
    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // Normalize and scale
    for (int i = tid; i < hidden_size; i += stride) {
        y[i] = x[i] * rms * (1.0f + weight[i]);
    }
}
