#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Causal 1D convolution decode update (single token).
// Computes depthwise conv + SiLU, then updates state for next step.
//
// state:   [conv_dim, kernel_size-1]  f32 (updated in-place)
// input:   [conv_dim]                 f32
// weight:  [conv_dim, kernel_size]    BF16
// output:  [conv_dim]                 f32
// One thread per channel.

extern "C" __global__ void causal_conv1d_update_f32(
    float* __restrict__ state,
    const float* __restrict__ input,
    const unsigned short* __restrict__ weight,
    float* __restrict__ output,
    const int conv_dim,
    const int kernel_size
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int hist = kernel_size - 1;
    float* s = state + ch * hist;
    const unsigned short* w = weight + ch * kernel_size;

    // Compute depthwise conv: state holds previous (kernel_size-1) inputs.
    // nn.Conv1d uses correlation (not convolution): w[0] is most recent, w[kernel-1] is oldest.
    // State: s[0]=oldest, s[hist-1]=most recent. Current input = newest.
    // conv = input * w[0] + s[hist-1] * w[1] + ... + s[0] * w[hist]
    float acc = input[ch] * bf16_to_f32(w[0]);
    for (int k = 0; k < hist; k++) {
        acc += s[hist - 1 - k] * bf16_to_f32(w[k + 1]);
    }

    // SiLU activation
    output[ch] = acc / (1.0f + expf(-acc));

    // Update state: shift left and insert new input for next step
    for (int i = 0; i < hist - 1; i++) {
        s[i] = s[i + 1];
    }
    s[hist - 1] = input[ch];
}

// Variant with bias (for Mamba2 which has conv1d.bias).
// Same as above but adds per-channel bias before activation.
extern "C" __global__ void causal_conv1d_update_bias_f32(
    float* __restrict__ state,
    const float* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int conv_dim,
    const int kernel_size
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int hist = kernel_size - 1;
    float* s = state + ch * hist;
    const unsigned short* w = weight + ch * kernel_size;

    // nn.Conv1d correlation order: w[0]=current, w[1]=most recent history, w[hist]=oldest
    float acc = input[ch] * bf16_to_f32(w[0]);
    for (int k = 0; k < hist; k++) {
        acc += s[hist - 1 - k] * bf16_to_f32(w[k + 1]);
    }
    acc += bias[ch];

    // SiLU activation
    output[ch] = acc / (1.0f + expf(-acc));

    for (int i = 0; i < hist - 1; i++) {
        s[i] = s[i + 1];
    }
    s[hist - 1] = input[ch];
}
