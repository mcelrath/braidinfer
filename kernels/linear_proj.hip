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


// NF4 codebook: 16 quantile-matched levels for N(0,1), from QLoRA
__constant__ float NF4_TABLE[16] = {
    -1.0f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.0910500f,  0.0f,
     0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
     0.4407098f,  0.5626170f,  0.7229568f,  1.0f,
};


// Matrix-vector multiply with rnf4_g128 quantized weights.
// Weight layout per row: groups of 128 elements, each group is 132 bytes:
//   [64B idx1_packed | 2B absmax1_f16 | 64B idx2_packed | 2B absmax2_f16]
// idx packed: 2 values per byte, low nibble first.
// Dequant: value = NF4_TABLE[idx1] * absmax1 + NF4_TABLE[idx2] * absmax2
//
// row_stride = num_groups_per_row * 132 bytes
// num_groups_per_row = ceil(in_dim / 128)

extern "C" __global__ void linear_proj_rnf4_g128(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,  // packed rnf4 data
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    if (row >= out_dim) return;

    const int group_size = 128;
    const int group_bytes = 132;  // 64 + 2 + 64 + 2
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;

    float acc = 0.0f;

    // Each thread processes multiple groups
    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp = row_data + g * group_bytes;
        const unsigned char* idx1_packed = gp;
        float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
        const unsigned char* idx2_packed = gp + 66;
        float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));

        int base = g * group_size;
        int count = min(group_size, in_dim - base);

        for (int i = 0; i < count; i += 2) {
            unsigned char byte1 = idx1_packed[i / 2];
            unsigned char byte2 = idx2_packed[i / 2];

            int i1_lo = byte1 & 0xF;
            int i1_hi = (byte1 >> 4) & 0xF;
            int i2_lo = byte2 & 0xF;
            int i2_hi = (byte2 >> 4) & 0xF;

            float v0 = NF4_TABLE[i1_lo] * absmax1 + NF4_TABLE[i2_lo] * absmax2;
            acc += v0 * input[base + i];

            if (i + 1 < count) {
                float v1 = NF4_TABLE[i1_hi] * absmax1 + NF4_TABLE[i2_hi] * absmax2;
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
