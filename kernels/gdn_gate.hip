#include <hip/hip_runtime.h>

// GDN gate computation: gate = exp(-exp(A_log) * softplus(a + dt_bias))
// Produces decay factors in (0, 1).
//
// A_log:   [num_heads]
// a:       [num_heads]
// dt_bias: [num_heads]
// gate:    [num_heads]  output

extern "C" __global__ void gdn_gate_f32(
    float* __restrict__ gate,
    const float* __restrict__ A_log,
    const float* __restrict__ a,
    const float* __restrict__ dt_bias,
    const int num_heads
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;

    float x = a[h] + dt_bias[h];
    // softplus(x) = log(1 + exp(x)), numerically stable form
    float sp = x > 20.0f ? x : log1pf(expf(x));
    float decay = -expf(A_log[h]) * sp;
    gate[h] = expf(decay);
}
