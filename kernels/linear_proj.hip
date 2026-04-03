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
    const ull2* w_vec = (const ull2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ull2 wv = __builtin_nontemporal_load(&w_vec[vi]);
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


#include "quant_consts.h"


// Matrix-vector multiply with rnf4_g128 quantized weights.
// Weight layout per row: groups of 128 elements, each group is 132 bytes:
//   [64B idx1_packed | 2B absmax1_bf16 | 64B idx2_packed | 2B absmax2_bf16]
// idx packed: 2 values per byte, low nibble first.
// Dequant: value = NF4_TABLE[idx1] * absmax1 + NF4_TABLE[idx2] * absmax2
//
// Optimization: 4 rows per block, input cached in shared memory,
// pair processing (2 elements per byte), wave32 shuffle reduction.

#define ROWS_PER_BLOCK 4

extern "C" __global__ void linear_proj_rnf4_g128(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int base_row = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int sub_tid = tid % 64;    // thread within sub-group (64 threads per row)
    const int sub_row = tid / 64;    // which of the 4 rows this thread handles

    const int GROUP_SIZE = 128;
    const int GROUP_BYTES = 132;
    const int num_groups = (in_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    const long long row_stride = (long long)num_groups * GROUP_BYTES;

    // Cache input in shared memory (shared across all 4 rows)
    extern __shared__ float sh[];
    float* sh_input = sh;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        sh_input[i] = input[i];
    }
    __syncthreads();

    int row = base_row + sub_row;
    if (row >= out_dim) return;

    const unsigned char* row_data = weight + row * row_stride;
    float acc = 0.0f;

    // 64 threads per row, each processes element pairs strided
    const int n_pairs = in_dim / 2;
    for (int pair = sub_tid; pair < n_pairs; pair += 64) {
        int g = (pair * 2) / GROUP_SIZE;
        int i = (pair * 2) % GROUP_SIZE;
        int byte_idx = i / 2;

        const unsigned char* gp = row_data + g * GROUP_BYTES;
        unsigned char b1 = gp[byte_idx];
        unsigned char b2 = gp[66 + byte_idx];

        float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
        float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));

        float v0 = NF4_TABLE[b1 & 0xF] * absmax1 + NF4_TABLE[b2 & 0xF] * absmax2;
        float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * absmax1 + NF4_TABLE[(b2 >> 4) & 0xF] * absmax2;

        int base = pair * 2;
        acc += v0 * sh_input[base] + v1 * sh_input[base + 1];
    }

    // Wave32 reduction within the 64-thread sub-group
    // First reduce within wave (32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    // Cross-wave within sub-group: 64 threads = 2 waves
    float* wave_sums = sh_input + in_dim + sub_row * 2;
    if (sub_tid == 0) wave_sums[0] = acc;
    if (sub_tid == 32) wave_sums[1] = acc;
    __syncthreads();

    if (sub_tid == 0) {
        output[row] = wave_sums[0] + wave_sums[1];
    }
}

#undef ROWS_PER_BLOCK


// Matrix-vector multiply with PcG32Q4 (asymmetric 4-bit, group_size=32) quantized weights.
// Weight layout per row: groups of 32 elements, each group is 20 bytes:
//   [16B idx_packed | 2B min_f16 | 2B scale_f16]
// idx packed: 2 values per byte, low nibble first, values 0-15.
// Dequant: value = idx * scale + min

extern "C" __global__ void linear_proj_pcg32_q4(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= out_dim) return;

    const int group_size = 32;
    const int group_bytes = 20;  // 16 + 2 + 2
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;

    float acc = 0.0f;

    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp = row_data + g * group_bytes;
        const unsigned char* idx_packed = gp;
        float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
        float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));

        int base = g * group_size;
        int count = min(group_size, in_dim - base);

        for (int i = 0; i < count; i += 2) {
            unsigned char byte = idx_packed[i / 2];
            int idx_lo = byte & 0xF;
            int idx_hi = (byte >> 4) & 0xF;

            float v0 = (float)idx_lo * sc + mn;
            acc += v0 * input[base + i];

            if (i + 1 < count) {
                float v1 = (float)idx_hi * sc + mn;
                acc += v1 * input[base + i + 1];
            }
        }
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
