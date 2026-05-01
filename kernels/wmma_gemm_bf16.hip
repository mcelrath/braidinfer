#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// WMMA-accelerated batched GEMM: C = A @ B.T
// A is [M, K] in bf16 (activations, row-major)
// B is [N, K] in bf16 (weights transposed, row-major, i.e. B[n, k] = original_weight[n, k])
// C is [M, N] in f32 (output)
//
// Grid:  [ceil(M/16), ceil(N/16)]
// Block: [32] — one wave32 per block
//
// gfx1100 RDNA3: uses __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32.
// Compiled with -mno-wavefrontsize64.
//
// RDNA3 WMMA output layout (wave32, 16x16x16):
//   acc[j] = C[ 2*j + (tid>>4) ][ tid & 15 ]
//   j in [0, 8), so each thread owns 8 output elements in a 16-col stripe.
//
// bf16 input register: ushort16 (16 x uint16 = 16 bf16 values).
// Each thread in the 32-thread wave holds 16 K-dimension bf16 values for its row/col.
// For a 16xK tile: threads 0-15 own rows 0-15 (A), threads 16-31 alias to rows 0-15.
// Similarly for B (columns).

typedef unsigned short ushort16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__launch_bounds__(32, 2)
extern "C" __global__ void wmma_gemm_bf16(
    float*          __restrict__ C,
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    int M, int N, int K
) {
    const int row_tile = blockIdx.x * 16;
    const int col_tile = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_tile >= M || col_tile >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row = row_tile + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    // Each thread owns one column in the 16-col tile for B.
    const int my_col = col_tile + (tid & 15);
    const int safe_col = (my_col < N) ? my_col : (N - 1);

    float8 acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // K must be a multiple of 16. Process in 16-element K-tiles.
    for (int k = 0; k < K; k += 16) {
        // Load 16 bf16 values for A[safe_row, k..k+16]
        ushort16 a_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int ki = k + i;
            a_reg[i] = (ki < K) ? A[(long long)safe_row * K + ki] : (unsigned short)0;
        }

        // Load 16 bf16 values for B[safe_col, k..k+16] (B is transposed weight: [N, K])
        ushort16 b_reg;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int ki = k + i;
            b_reg[i] = (ki < K) ? B[(long long)safe_col * K + ki] : (unsigned short)0;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_reg, b_reg, acc);
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
