#include <hip/hip_runtime.h>

// GPU-resident argmax: finds index of maximum element in a float array.
// Two-phase reduction: Phase 1 finds per-block max, Phase 2 reduces blocks.
// For single-kernel simplicity, uses one block with enough threads.
//
// For vocab_size up to 262144: one block of 256 threads, each scans ~1024 elements.
// Output: single i32 (token ID) — eliminates 500KB+ logit transfer over PCIe.

extern "C" __global__ void argmax_f32(
    const float* __restrict__ input,
    int* __restrict__ output,  // single element: argmax index
    const int size
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Each thread finds local max over its strided range
    float local_max = -1e30f;
    int local_idx = 0;
    for (int i = tid; i < size; i += stride) {
        float val = input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Shared memory reduction
    extern __shared__ char smem[];
    float* smax = (float*)smem;
    int* sidx = (int*)(smem + blockDim.x * sizeof(float));

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sidx[0];
    }
}
