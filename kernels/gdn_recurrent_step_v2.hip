#include <hip/hip_runtime.h>

// GDN recurrent decode step v2: correct gate computation and QK L2-norm.
// One block per head.
//
// Gate is pre-computed (already in (0,1) via gdn_gate kernel).
// Q and K are L2-normalized per-head before outer product.
//
// q, k: [num_heads, key_dim]
// v:    [num_heads, value_dim]
// gate: [num_heads]  -- pre-computed decay in (0,1)
// b:    [num_heads]  -- beta (sigmoid applied here)
// state:[num_heads, key_dim, value_dim]  (updated in-place)
// output:[num_heads, value_dim]

extern "C" __global__ void gdn_recurrent_step_v2_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ b,
    float* __restrict__ state,
    float* __restrict__ output,
    const int key_dim,
    const int value_dim
) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float decay = gate[h];
    const float beta = 1.0f / (1.0f + expf(-b[h]));

    const float* qh = q + h * key_dim;
    const float* kh = k + h * key_dim;
    const float* vh = v + h * value_dim;
    float* sh = state + h * key_dim * value_dim;
    float* oh = output + h * value_dim;

    // Compute L2 norms of q and k using shared memory reduction
    extern __shared__ float shared[];
    float* sq = shared;
    float* sk = shared + block_size;

    float sq_acc = 0.0f;
    float sk_acc = 0.0f;
    for (int i = tid; i < key_dim; i += block_size) {
        float qi = qh[i];
        float ki = kh[i];
        sq_acc += qi * qi;
        sk_acc += ki * ki;
    }
    sq[tid] = sq_acc;
    sk[tid] = sk_acc;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sq[tid] += sq[tid + s];
            sk[tid] += sk[tid + s];
        }
        __syncthreads();
    }

    float q_inv_norm = rsqrtf(sq[0] + 1e-8f);
    float k_inv_norm = rsqrtf(sk[0] + 1e-8f);

    // Update state: S[i,j] = decay * S[i,j] + beta * v[j] * k_norm[i]
    const int total = key_dim * value_dim;
    for (int idx = tid; idx < total; idx += block_size) {
        const int i = idx / value_dim;
        const int j = idx % value_dim;
        float k_norm_i = kh[i] * k_inv_norm;
        sh[idx] = decay * sh[idx] + beta * vh[j] * k_norm_i;
    }

    __syncthreads();

    // Output: o[j] = sum_i S[i,j] * q_norm[i]
    for (int j = tid; j < value_dim; j += block_size) {
        float sum = 0.0f;
        for (int i = 0; i < key_dim; i++) {
            float q_norm_i = qh[i] * q_inv_norm;
            sum += sh[i * value_dim + j] * q_norm_i;
        }
        oh[j] = sum;
    }
}
