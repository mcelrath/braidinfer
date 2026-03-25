#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Fused FFN decode kernel — two sub-kernels, 2 launches instead of 6.
//
// Sub-kernel 1: RMSNorm(input) -> gate_proj + up_proj -> silu(gate)*up -> scratch
// Sub-kernel 2: down_proj(scratch) + residual(input) -> output
//
// All weights are BF16, activations are f32.
// hidden_size = 1024, intermediate_size = 3584 for Qwen3.5-0.8B

// ffn_fused_gate_up_f32:
//   grid:  (intermediate_size, 1, 1)  — one block per output row of gate/up
//   block: (block_size, 1, 1)         — threads collaborate on dot product over hidden_size
//   smem:  block_size * sizeof(float) for reduction
extern "C" __global__ void ffn_fused_gate_up_f32(
    float* __restrict__ scratch,                  // [intermediate_size] — silu(gate)*up output
    const float* __restrict__ input,              // [hidden_size]
    const unsigned short* __restrict__ rms_weight,// [hidden_size] BF16
    const unsigned short* __restrict__ w_gate,    // [intermediate_size, hidden_size] BF16
    const unsigned short* __restrict__ w_up,      // [intermediate_size, hidden_size] BF16
    int hidden_size,
    int intermediate_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= intermediate_size) return;

    extern __shared__ float shared[];

    // Step 1: compute RMS norm scale (each block recomputes — avoids writing norm'd vector)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = input[i];
        sum_sq += v * v;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms_scale = rsqrtf(shared[0] / (float)hidden_size + eps);
    __syncthreads();

    // Step 2: gate dot product and up dot product in the same loop
    const unsigned short* wg = w_gate + row * hidden_size;
    const unsigned short* wu = w_up   + row * hidden_size;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float normed = input[i] * rms_scale * (1.0f + bf16_to_f32(rms_weight[i]));
        acc_gate += bf16_to_f32(wg[i]) * normed;
        acc_up   += bf16_to_f32(wu[i]) * normed;
    }

    // Reduction for gate
    shared[tid] = acc_gate;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float gate_val = shared[0];
    __syncthreads();

    // Reduction for up
    shared[tid] = acc_up;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float up_val = shared[0];

    // Step 3: silu(gate) * up
    if (tid == 0) {
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        scratch[row] = silu_g * up_val;
    }
}

// ffn_fused_down_residual_f32:
//   grid:  (hidden_size, 1, 1)         — one block per output element
//   block: (block_size, 1, 1)          — threads collaborate on dot product over intermediate_size
//   smem:  block_size * sizeof(float)
extern "C" __global__ void ffn_fused_down_residual_f32(
    float* __restrict__ output,                // [hidden_size]
    const float* __restrict__ input,           // [hidden_size] — original input (residual)
    const unsigned short* __restrict__ w_down, // [hidden_size, intermediate_size] BF16
    const float* __restrict__ scratch,         // [intermediate_size]
    int hidden_size,
    int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= hidden_size) return;

    const unsigned short* w = w_down + row * intermediate_size;

    float acc = 0.0f;
    for (int i = tid; i < intermediate_size; i += stride) {
        acc += bf16_to_f32(w[i]) * scratch[i];
    }

    extern __shared__ float shared[];
    shared[tid] = acc;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared[0] + input[row];
    }
}
