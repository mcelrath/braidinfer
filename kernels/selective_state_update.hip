#include <hip/hip_runtime.h>

// Mamba2 selective state update (SSM recurrence step)
//
// Per-head computation:
//   dt_val = softplus(dt[h] + dt_bias[h])
//   dA = exp(dt_val * (-exp(A_log[h])))
//   For each (d, s):
//     ssm_state[h,d,s] = dA * ssm_state[h,d,s] + dt_val * x[h,d] * B[g,s]
//   For each d:
//     y[h,d] = sum_s(ssm_state[h,d,s] * C[g,s]) + D[h] * x[h,d]
//
// where g = h / heads_per_group (n_groups groups, each has num_heads/n_groups heads)
//
// One block per head. 256 threads per block.
// state shape: [num_heads, head_dim, state_size]

__device__ __forceinline__ float softplus(float x) {
    // softplus(x) = log(1 + exp(x))
    // For numerical stability: if x > 20, softplus ≈ x
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

extern "C" __global__ void selective_state_update_f32(
    float* __restrict__ ssm_state,   // [num_heads, head_dim, state_size] — in-place update
    const float* __restrict__ x,     // [num_heads * head_dim] — input (reshaped from intermediate)
    const float* __restrict__ dt,    // [num_heads] — timestep (before softplus)
    const float* __restrict__ dt_bias, // [num_heads]
    const float* __restrict__ A_log, // [num_heads] — log of decay
    const float* __restrict__ B,     // [n_groups * state_size] — input-to-state
    const float* __restrict__ C,     // [n_groups * state_size] — state-to-output
    const float* __restrict__ D,     // [num_heads] — skip connection
    float* __restrict__ output,      // [num_heads * head_dim] — output
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups
) {
    const int h = blockIdx.x; // one block per head
    if (h >= num_heads) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int heads_per_group = num_heads / n_groups;
    const int g = h / heads_per_group;

    // Compute per-head scalars
    float dt_val = softplus(dt[h] + dt_bias[h]);
    float A = -expf(A_log[h]);
    float dA = expf(dt_val * A);

    // Pointers for this head
    float* state_h = ssm_state + (long long)h * head_dim * state_size;
    const float* x_h = x + h * head_dim;
    const float* B_g = B + g * state_size;
    const float* C_g = C + g * state_size;
    float D_h = D[h];

    // Phase 1: Update state — ssm_state[h,d,s] = dA * state + dt * x[d] * B[s]
    // Total elements: head_dim * state_size (e.g., 64*128 = 8192)
    int total = head_dim * state_size;
    for (int idx = tid; idx < total; idx += stride) {
        int d = idx / state_size;
        int s = idx % state_size;
        float old_state = state_h[d * state_size + s];
        state_h[d * state_size + s] = dA * old_state + dt_val * x_h[d] * B_g[s];
    }

    __syncthreads(); // Ensure state is fully updated before reading

    // Phase 2: Compute output — y[d] = sum_s(state[d,s] * C[s]) + D * x[d]
    for (int d = tid; d < head_dim; d += stride) {
        float sum = 0.0f;
        for (int s = 0; s < state_size; s++) {
            sum += state_h[d * state_size + s] * C_g[s];
        }
        output[h * head_dim + d] = sum + D_h * x_h[d];
    }
}
