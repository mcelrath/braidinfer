#include <hip/hip_runtime.h>

// Deinterleave: split [q0,gate0, q1,gate1, ...] per head into separate q and gate tensors.
// One element per thread. Grid: ceil(num_heads * head_dim / 256) blocks x 256 threads.
//
// src:     [num_heads, 2*head_dim]  (interleaved q and gate)
// dst_q:   [num_heads, head_dim]
// dst_gate:[num_heads, head_dim]

extern "C" __global__ void deinterleave_f32(
    float* __restrict__ dst_q,
    float* __restrict__ dst_gate,
    const float* __restrict__ src,
    const int num_heads,
    const int head_dim
) {
    int total = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int head = idx / head_dim;
    int elem = idx % head_dim;
    int src_base = head * 2 * head_dim;

    dst_q[idx]    = src[src_base + elem];
    dst_gate[idx] = src[src_base + head_dim + elem];
}
