#include <hip/hip_runtime.h>

// Causal 1D convolution decode update (single token).
// Shifts conv state left, inserts new input, computes depthwise conv + SiLU.
//
// state:   [conv_dim, kernel_size-1]  (updated in-place)
// input:   [conv_dim]
// weight:  [conv_dim, kernel_size]
// output:  [conv_dim]
// One thread per channel.

extern "C" __global__ void causal_conv1d_update_f32(
    float* __restrict__ state,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int conv_dim,
    const int kernel_size
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int hist = kernel_size - 1;
    float* s = state + ch * hist;
    const float* w = weight + ch * kernel_size;

    // Shift state left: s[0..hist-2] = s[1..hist-1]
    for (int i = 0; i < hist - 1; i++) {
        s[i] = s[i + 1];
    }
    // Insert new input at end
    s[hist - 1] = input[ch];

    // Compute depthwise conv: sum over kernel_size elements
    float acc = 0.0f;
    for (int k = 0; k < hist; k++) {
        acc += s[k] * w[k];
    }
    acc += input[ch] * w[hist];

    // SiLU activation
    output[ch] = acc / (1.0f + expf(-acc));
}
