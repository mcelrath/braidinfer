#include <hip/hip_runtime.h>

// SiLU gating: y = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Element-wise, one thread per element.

extern "C" __global__ void silu_mul_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = silu_g * up[idx];
}
