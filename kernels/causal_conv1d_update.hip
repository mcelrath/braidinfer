#include <hip/hip_runtime.h>

// Causal 1D convolution decode update (single token).
// Computes depthwise conv + SiLU, then updates state for next step.
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

    // Compute depthwise conv: state holds previous (kernel_size-1) inputs
    // conv = sum(s[k] * w[k] for k in 0..hist) + input * w[hist]
    float acc = 0.0f;
    for (int k = 0; k < hist; k++) {
        acc += s[k] * w[k];
    }
    acc += input[ch] * w[hist];

    // SiLU activation
    output[ch] = acc / (1.0f + expf(-acc));

    // Update state: shift left and insert new input for next step
    for (int i = 0; i < hist - 1; i++) {
        s[i] = s[i + 1];
    }
    s[hist - 1] = input[ch];
}
