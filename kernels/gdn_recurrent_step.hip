#include <hip/hip_runtime.h>

// Gated DeltaNet recurrent decode step (single token, single layer).
// One block per head.
//
// For each head h:
//   gate = sigmoid(g[h])
//   beta = sigmoid(b[h])
//   S[h] = gate * S[h] + beta * outer(v[h], k[h])   // state: [key_dim x value_dim]
//   o[h] = S[h]^T @ q[h]   i.e. o[j] = sum_i S[i,j] * q[i]
//
// Layout:
//   q, k: [num_heads, key_dim]
//   v:    [num_heads, value_dim]
//   g, b: [num_heads]
//   state:[num_heads, key_dim, value_dim]  (in-place update)
//   output:[num_heads, value_dim]

extern "C" __global__ void gdn_recurrent_step_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ b,
    float* __restrict__ state,
    float* __restrict__ output,
    const int key_dim,
    const int value_dim
) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float gate = 1.0f / (1.0f + expf(-g[h]));
    const float beta = 1.0f / (1.0f + expf(-b[h]));

    const float* qh = q + h * key_dim;
    const float* kh = k + h * key_dim;
    const float* vh = v + h * value_dim;
    float* sh = state + h * key_dim * value_dim;
    float* oh = output + h * value_dim;

    // Update state: S[i,j] = gate * S[i,j] + beta * v[j] * k[i]
    // Total elements: key_dim * value_dim
    const int total = key_dim * value_dim;
    for (int idx = tid; idx < total; idx += block_size) {
        const int i = idx / value_dim;
        const int j = idx % value_dim;
        sh[idx] = gate * sh[idx] + beta * vh[j] * kh[i];
    }

    __syncthreads();

    // Output: o[j] = sum_i S[i,j] * q[i]
    for (int j = tid; j < value_dim; j += block_size) {
        float sum = 0.0f;
        for (int i = 0; i < key_dim; i++) {
            sum += sh[i * value_dim + j] * qh[i];
        }
        oh[j] = sum;
    }
}
