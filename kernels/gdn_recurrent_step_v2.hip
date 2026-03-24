#include <hip/hip_runtime.h>

// GDN recurrent decode step v2: Gated Delta Rule with QK L2-norm.
// One block per head.
//
// Implements the gated delta rule recurrence:
//   q_norm = l2norm(q), k_norm = l2norm(k)
//   q_scaled = q_norm / sqrt(key_dim)
//   decay = gate (pre-computed, in (0,1))
//   state *= decay
//   kv_mem[j] = sum_i state[i,j] * k_norm[i]
//   delta[j] = (v[j] - kv_mem[j]) * beta
//   state[i,j] += k_norm[i] * delta[j]
//   output[j] = sum_i state[i,j] * q_scaled[i]
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

    float q_inv_norm = rsqrtf(sq[0] + 1e-6f);
    float k_inv_norm = rsqrtf(sk[0] + 1e-6f);
    float q_scale = q_inv_norm * rsqrtf((float)key_dim);

    // Step 1: state *= decay
    const int total = key_dim * value_dim;
    for (int idx = tid; idx < total; idx += block_size) {
        sh[idx] *= decay;
    }
    __syncthreads();

    // Step 2: kv_mem[j] = sum_i state[i,j] * k_norm[i], then delta[j] = (v[j] - kv_mem[j]) * beta
    // Step 3: state[i,j] += k_norm[i] * delta[j]
    // We process one j (value_dim column) per thread iteration
    for (int j = tid; j < value_dim; j += block_size) {
        float kv_mem = 0.0f;
        for (int i = 0; i < key_dim; i++) {
            kv_mem += sh[i * value_dim + j] * (kh[i] * k_inv_norm);
        }
        float delta = (vh[j] - kv_mem) * beta;
        for (int i = 0; i < key_dim; i++) {
            sh[i * value_dim + j] += (kh[i] * k_inv_norm) * delta;
        }
    }
    __syncthreads();

    // Step 4: output[j] = sum_i state[i,j] * q_scaled[i]
    for (int j = tid; j < value_dim; j += block_size) {
        float sum = 0.0f;
        for (int i = 0; i < key_dim; i++) {
            sum += sh[i * value_dim + j] * (qh[i] * q_scale);
        }
        oh[j] = sum;
    }
}
