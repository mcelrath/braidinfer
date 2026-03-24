#include <hip/hip_runtime.h>

// Attention output gating: output = attn_output * sigmoid(gate)
// attn_output, gate: [size]
// output: [size]

extern "C" __global__ void output_gate_f32(
    float* __restrict__ output,
    const float* __restrict__ attn_output,
    const float* __restrict__ gate,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float g = gate[idx];
    output[idx] = attn_output[idx] * (1.0f / (1.0f + expf(-g)));
}
