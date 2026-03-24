#include <hip/hip_runtime.h>

// Fused GDN decode layer: 3 sub-kernels replacing 9 separate launches.
//
// Scratch buffer layout (floats), with nk = num_heads*key_dim, nv = num_heads*value_dim:
//   q:             [nk]   offset 0
//   k:             [nk]   offset nk
//   v:             [nv]   offset 2*nk
//   g:             [nh]   offset 2*nk + nv
//   b:             [nh]   offset 2*nk + nv + nh
//   recurrent_out: [nv]   offset 2*nk + nv + 2*nh
//
// For Qwen3.5-0.8B: hidden=1024, num_heads=16, key_dim=128, value_dim=128, nh=16
//   nk=2048, nv=2048, offsets: q=0, k=2048, v=4096, g=6144, b=6160, rec_out=6176
//   total scratch: 8224 floats

// ---------------------------------------------------------------------------
// Sub-kernel 1: RMSNorm + Q/K/V/gate/beta projections
//
// Grid: (2*nk + nv + 2*nh) blocks, 256 threads each.
// Block i produces scratch[i] via dot(w_row_i, rms_norm(input)).
// RMSNorm is recomputed by every block (avoids cross-block sync).
//
// Block-to-output mapping:
//   [0,          nk)           -> q
//   [nk,         2*nk)         -> k
//   [2*nk,       2*nk+nv)      -> v
//   [2*nk+nv,    2*nk+nv+nh)   -> g
//   [2*nk+nv+nh, 2*nk+nv+2*nh) -> b
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_fused_proj_f32(
    float* __restrict__ scratch,
    const float* __restrict__ input,
    const float* __restrict__ rms_weight,
    const float* __restrict__ w_q,
    const float* __restrict__ w_k,
    const float* __restrict__ w_v,
    const float* __restrict__ w_g,
    const float* __restrict__ w_b,
    const int hidden_size,
    const int num_heads,
    const int key_dim,
    const int value_dim,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int nk = num_heads * key_dim;
    const int nv = num_heads * value_dim;
    const int nh = num_heads;

    // Shared memory: first stride floats for RMSNorm reduction, reused for dot product.
    extern __shared__ float shared[];

    // --- RMSNorm: compute 1/rms of input ---
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = input[i];
        sum_sq += val * val;
    }
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // --- Determine weight row and output pointer ---
    const float* w;
    float* out;

    if (row < nk) {
        w   = w_q + row * hidden_size;
        out = scratch + row;
    } else if (row < 2 * nk) {
        const int r = row - nk;
        w   = w_k + r * hidden_size;
        out = scratch + nk + r;
    } else if (row < 2 * nk + nv) {
        const int r = row - 2 * nk;
        w   = w_v + r * hidden_size;
        out = scratch + 2 * nk + r;
    } else if (row < 2 * nk + nv + nh) {
        const int r = row - 2 * nk - nv;
        w   = w_g + r * hidden_size;
        out = scratch + 2 * nk + nv + r;
    } else {
        const int r = row - 2 * nk - nv - nh;
        w   = w_b + r * hidden_size;
        out = scratch + 2 * nk + nv + nh + r;
    }

    // --- Dot product: w[row] . (input * rms * rms_weight) ---
    __syncthreads();
    float acc = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc += w[i] * (input[i] * rms * rms_weight[i]);
    }
    shared[tid] = acc;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        *out = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Sub-kernel 2: GDN recurrent step per head
//
// Reads q/k/v/g/b from scratch, updates persistent state in-place,
// writes recurrent_out into scratch at offset 2*nk+nv+2*nh.
// Grid: num_heads blocks.
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_fused_recurrent_f32(
    float* __restrict__ scratch,
    float* __restrict__ state,
    const int num_heads,
    const int key_dim,
    const int value_dim
) {
    const int h          = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    const int nk = num_heads * key_dim;
    const int nv = num_heads * value_dim;
    const int nh = num_heads;

    const float* q_base   = scratch;
    const float* k_base   = scratch + nk;
    const float* v_base   = scratch + 2 * nk;
    const float* g_base   = scratch + 2 * nk + nv;
    const float* b_base   = scratch + 2 * nk + nv + nh;
    float*       rec_base = scratch + 2 * nk + nv + 2 * nh;

    const float gate = 1.0f / (1.0f + expf(-g_base[h]));
    const float beta = 1.0f / (1.0f + expf(-b_base[h]));

    const float* qh = q_base + h * key_dim;
    const float* kh = k_base + h * key_dim;
    const float* vh = v_base + h * value_dim;
    float*       sh = state  + h * key_dim * value_dim;
    float*       oh = rec_base + h * value_dim;

    // Update state: S[i,j] = gate * S[i,j] + beta * v[j] * k[i]
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

// ---------------------------------------------------------------------------
// Sub-kernel 3: Output projection + residual add
//
// Reads recurrent_out from scratch (offset 2*nk+nv+2*nh), applies W_o [hidden x rec_dim],
// adds residual (input), writes to output.
// Grid: hidden_size blocks.
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_fused_output_f32(
    float* __restrict__ output,
    const float* __restrict__ scratch,
    const float* __restrict__ input,
    const float* __restrict__ w_o,
    const int hidden_size,
    const int num_heads,
    const int key_dim,
    const int value_dim
) {
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= hidden_size) return;

    const int nk     = num_heads * key_dim;
    const int nv     = num_heads * value_dim;
    const int nh     = num_heads;
    const int rec_dim = nv;  // recurrent output dim = num_heads * value_dim

    const float* rec_out = scratch + 2 * nk + nv + 2 * nh;
    const float* w       = w_o + row * rec_dim;

    float acc = 0.0f;
    for (int i = tid; i < rec_dim; i += stride) {
        acc += w[i] * rec_out[i];
    }

    __shared__ float shared[256];
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
