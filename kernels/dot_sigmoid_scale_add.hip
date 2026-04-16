#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Fused: scale = sigmoid(dot(bf16_weight, f32_input)), then output[i] += scale * src[i]
// Eliminates CPU round-trip for shared expert gate weighting.
// Launch with 1 block, block_size threads, shmem = (block_size/warpSize)*sizeof(float)

extern "C" __global__ void dot_sigmoid_scale_add_bf16_f32(
    float* __restrict__ output,
    const float* __restrict__ src,
    const float* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const int size
) {
    extern __shared__ float shmem[];

    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    // Parallel partial dot product
    float partial = 0.0f;
    for (int i = tid; i < size; i += bs)
        partial += input[i] * bf16_to_f32(weight[i]);

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down(partial, offset);

    // Store warp sums into shared memory
    if (tid % warpSize == 0)
        shmem[tid / warpSize] = partial;
    __syncthreads();

    // Thread 0 sums warps, computes sigmoid, broadcasts via shmem[0]
    if (tid == 0) {
        float total = 0.0f;
        const int num_warps = (bs + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++)
            total += shmem[w];
        shmem[0] = 1.0f / (1.0f + expf(-total));
    }
    __syncthreads();

    const float scale = shmem[0];
    for (int i = tid; i < size; i += bs)
        output[i] += scale * src[i];
}
