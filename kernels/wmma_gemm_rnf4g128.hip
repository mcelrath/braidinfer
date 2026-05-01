#include <hip/hip_runtime.h>
#include "bf16_utils.h"
#include "quant_consts.h"

// f32 -> bf16 via bit-truncation of mantissa (round-toward-zero, no NaN/Inf special handling).
__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int bits;
    __builtin_memcpy(&bits, &v, 4);
    return (unsigned short)(bits >> 16);
}

// WMMA-accelerated batched GEMM with inline RNF4G128 dequant.
//
// Weight format (per output row): num_groups groups of 132 bytes.
// Each 132-byte group covers 128 elements:
//   bytes  0.. 63: idx1_packed (64 bytes, 2 nibbles each = 128 nibbles for indices 0-127)
//   bytes 64.. 65: absmax1 (bf16)
//   bytes 66..129: idx2_packed (64 bytes, 2 nibbles each = 128 nibbles for indices 0-127)
//   bytes 130..131: absmax2 (bf16)
//
// Dequant for element pair at byte_idx:
//   b1 = idx1_packed[byte_idx], b2 = idx2_packed[byte_idx]
//   v0 = NF4_TABLE[b1 & 0xF] * absmax1 + NF4_TABLE[b2 & 0xF] * absmax2
//   v1 = NF4_TABLE[(b1 >> 4) & 0xF] * absmax1 + NF4_TABLE[(b2 >> 4) & 0xF] * absmax2
//
// A (activations) is [M, K] in f32.
// B (weights) is [N, num_groups * 132] in RNF4G128 format (N = out_dim).
// C (output) is [M, N] in f32.
//
// Grid:  [ceil(M/16), ceil(N/16)]
// Block: [32] — one wave32 per block.
// K must be a multiple of 128 (group size).

typedef unsigned short ushort16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__launch_bounds__(32, 2)
extern "C" __global__ void wmma_gemm_rnf4g128(
    float*                __restrict__ C,
    const float*          __restrict__ A,
    const unsigned char*  __restrict__ B,
    int M, int N, int K
) {
    const int row_tile = blockIdx.x * 16;
    const int col_tile = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_tile >= M || col_tile >= N) return;

    const int my_row = row_tile + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int my_col = col_tile + (tid & 15);
    const int safe_col = (my_col < N) ? my_col : (N - 1);

    const int GROUP_SIZE = 128;
    const int GROUP_BYTES = 132;
    const int num_groups = K / GROUP_SIZE;

    const unsigned char* b_row = B + (long long)safe_col * num_groups * GROUP_BYTES;

    float8 acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < num_groups; g++) {
        const unsigned char* gp = b_row + g * GROUP_BYTES;

        float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
        float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));

        // Process the 128 elements in this group as 8 K-tiles of 16 elements.
        // Elements within the group are pairs: element 2*bi+0 and 2*bi+1 share byte_idx=bi.
        // Group layout: 64 byte pairs, so byte_idx in [0, 64).
        // K-tile of 16 elements = 8 bytes from idx1 and 8 bytes from idx2.
        for (int kt = 0; kt < 8; kt++) {
            int k_start = g * GROUP_SIZE + kt * 16;
            int byte_base = kt * 8;

            // Load A[safe_row, k_start..k_start+16] -> a_reg (bf16)
            ushort16 a_reg;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float av = (k_start + i < K) ? A[(long long)safe_row * K + k_start + i] : 0.0f;
                a_reg[i] = f32_to_bf16_bits(av);
            }

            // Dequant B: 16 elements from this K-tile -> b_reg (bf16)
            // Element pair at byte_base+byte_offset: byte_offset in [0,8)
            // element 2*byte_offset+0: low nibble of idx1 and idx2
            // element 2*byte_offset+1: high nibble of idx1 and idx2
            ushort16 b_reg;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char b1 = gp[byte_base + i];
                unsigned char b2 = gp[66 + byte_base + i];
                float v0 = NF4_TABLE[b1 & 0xF] * absmax1 + NF4_TABLE[b2 & 0xF] * absmax2;
                float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * absmax1 + NF4_TABLE[(b2 >> 4) & 0xF] * absmax2;
                b_reg[2 * i + 0] = f32_to_bf16_bits(v0);
                b_reg[2 * i + 1] = f32_to_bf16_bits(v1);
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_reg, b_reg, acc);
        }
    }

    // Write output: acc[j] = C[ row_tile + 2*j + (tid>>4) ][ col_tile + (tid & 15) ]
    const int out_col = col_tile + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_tile + 2 * j + (tid >> 4);
            if (out_row < M) {
                C[(long long)out_row * N + out_col] = acc[j];
            }
        }
    }
}
