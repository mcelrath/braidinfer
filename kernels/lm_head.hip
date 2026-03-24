#include <hip/hip_runtime.h>

// LM head: y = W @ x
// W is [vocab_size x hidden_size], x is [hidden_size], y is [vocab_size]
// One block per output row (vocab token), dot product with shared mem reduction.

extern "C" __global__ void lm_head_f32(
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ input,
    const int vocab_size,
    const int hidden_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= vocab_size) return;

    const float* w = weight + (long long)row * hidden_size;

    float acc = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc += w[i] * input[i];
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
