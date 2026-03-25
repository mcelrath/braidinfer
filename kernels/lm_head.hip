#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// LM head: y = W @ x
// W is [vocab_size x hidden_size] BF16, x is [hidden_size] f32, y is [vocab_size] f32.
// One block per output row (vocab token), dot product with shared mem reduction.

extern "C" __global__ void lm_head_f32(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int vocab_size,
    const int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= vocab_size) return;

    const unsigned short* w = weight + (long long)row * hidden_size;

    float acc = 0.0f;

    const int vec_elements = 8;
    const int vec_count = hidden_size / vec_elements;
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
    int rem = vec_count * vec_elements;
    for (int i = rem + tid; i < hidden_size; i += stride) {
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
