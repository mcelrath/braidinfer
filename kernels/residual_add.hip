#include <hip/hip_runtime.h>

// Residual add: y = x + residual (element-wise)

extern "C" __global__ void residual_add_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = x[idx] + residual[idx];
}
