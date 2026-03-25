#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// GDN gate computation: gate = exp(-exp(A_log) * softplus(a + dt_bias))
// Produces decay factors in (0, 1).
//
// A_log:   [num_heads] f32 (stored in log space)
// a:       [num_heads] f32 (activation)
// dt_bias: [num_heads] BF16
// gate:    [num_heads] f32 output

extern "C" __global__ void gdn_gate_f32(
    float* __restrict__ gate,
    const float* __restrict__ A_log,
    const float* __restrict__ a,
    const unsigned short* __restrict__ dt_bias,
    const int num_heads
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;

    float x = a[h] + bf16_to_f32(dt_bias[h]);
    // softplus(x) = log(1 + exp(x)), numerically stable form
    float sp = x > 20.0f ? x : log1pf(expf(x));
    float decay = -expf(A_log[h]) * sp;
    gate[h] = expf(decay);
}
