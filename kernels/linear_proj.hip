#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Matrix-vector multiply: y = W @ x
// W is [out_dim x in_dim] row-major BF16, x is [in_dim] f32, y is [out_dim] f32.
// One block per output row, dot product with shared mem reduction.
// For decode batch=1 only.

extern "C" __global__ void linear_proj_f32(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= out_dim) return;

    const unsigned short* w = weight + (long long)row * in_dim;

    float acc = 0.0f;

    // 128-bit vectorized loads: 8 bf16 values per load (hits peak BW ~960 GB/s)
    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ulong2* w_vec = (const ulong2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ulong2 wv = w_vec[vi];
        unsigned short ws[8];
        __builtin_memcpy(ws, &wv, 16);

        int base = vi * vec_elements;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += bf16_to_f32(ws[j]) * input[base + j];
        }
    }

    // Handle remainder (in_dim not multiple of 8)
    int remainder_start = vec_count * vec_elements;
    for (int i = remainder_start + tid; i < in_dim; i += stride) {
        acc += bf16_to_f32(w[i]) * input[i];
    }

    __shared__ float shared[256];
    shared[tid] = acc;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared[0];
    }
}
