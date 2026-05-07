// RDNA3 (gfx1100, wave32) compute primitives: WMMA wrappers + atomic GEMV.
//
// Companion to kernels/rdna3_reduce.h. Where rdna3_reduce.h covers data-parallel
// reductions, this header covers the matrix-multiply path (WMMA) and the
// scatter-accumulate path (atomicAdd<float> on global memory) that compose
// most of the prefill + LM-head workload on gfx1100.
//
// Two motivations:
//
//   1. RDNA3 WMMA (16x16x16 bf16/fp16 -> fp32, wave32) has a fragment lane
//      mapping that does NOT match CUDA mma.sync (m16n8k16 / m16n16k16). Mapping
//      a CUDA pre-pack onto RDNA3 fragments costs 3-5x in measured throughput
//      (most loads end up uncoalesced or land in the wrong lane and require
//      extra cross-lane moves). The wrappers below hide the RDNA3-native lane
//      mapping behind a CUDA-mma-shaped API: load_a, load_b, mma_sync, store_c.
//
//   2. RDNA3 has a real hardware global_atomic_add_f32, but the default
//      HIP atomicAdd(float*, float) does NOT use it — it lowers to a
//      global_atomic_cmpswap_b32 CAS loop. Use unsafeAtomicAdd to get the
//      hardware instruction (verified by objdump). Measured 7-46x speedup
//      depending on contention. For split-K-style GEMV (one output row =
//      many K-blocks summed), atomic accumulation can replace cooperative
//      reductions with a single hardware op. Whether split-K beats
//      cooperative reduction is a separate question — see Phase D in the
//      adoption order; for current LLM shapes (K=1024, N=512..248320)
//      block-cooperative reduction wins, so this matters most as a primitive
//      to use whenever you DO need atomics, not as a recipe for converting
//      every reduction.
//
// References:
//   GFX1100_ARCH.md §4 (WMMA Programming Model on gfx1100)
//   GFX1100_ARCH.md §6.1 (Canonical WMMA Instance Template)
//   RDNA3 ISA §7.9 (WMMA opsel, lane-replication rule, hazards)
//   kernels/wmma_gemm_bf16.hip (existing in-tree kernel; same lane mapping)
//
// ============================================================================
// WMMA fragment layout reminder (RDNA3 wave32, 16x16x16, A/B = bf16, C/D = f32)
// ============================================================================
//
// Each lane in the wave (32 lanes) holds 16 input scalars per fragment for
// A or B, and 8 accumulator scalars for C/D. Total VGPR footprint per fragment:
//
//   A (bf16, 16 elements): 8 VGPRs
//   B (bf16, 16 elements): 8 VGPRs
//   C (f32, 8 elements):   8 VGPRs
//   D (f32, 8 elements):   8 VGPRs                           (GFX1100_ARCH §4.1)
//
// Lane mapping (the part most people get wrong):
//
//   A is conceptually a 16x16 column-major tile in K-stripes.
//     A[lane, k] is the (lane mod 16, k)-th element of the M-K tile.
//     lanes 0-15 own M-rows 0-15.
//     lanes 16-31 MUST hold the same per-lane data as lanes 0-15
//     (RDNA3 lane-replication rule, GFX1100_ARCH §4.2).
//
//   B is conceptually a 16x16 row-major tile in K-stripes.
//     B[lane, k] is the (k, lane mod 16)-th element of the K-N tile.
//     lanes 0-15 own N-cols 0-15.
//     lanes 16-31 MUST hold the same per-lane data as lanes 0-15.
//
//   C/D is conceptually a 16x16 row-major output tile.
//     reg[j] = C[2*j + (lane >> 4), lane & 15]    for j in [0, 8)
//     i.e. lanes 0-15 contribute even rows of the output (rows 0, 2, ..., 14)
//     and lanes 16-31 contribute odd rows (rows 1, 3, ..., 15).
//     The rows ARE NOT replicated for the output — every lane contributes
//     unique data.
//
// CUDA users: this is NOT the same as nvcuda::wmma. The closest analogue is
// rocWMMA's row_major / col_major fragment types, but rocWMMA hides the
// half-wave replication; in custom kernels you must satisfy it yourself.
//
// ============================================================================
// WMMA pre-pack rule (the 3-5x performance gotcha)
// ============================================================================
//
// CORRECT: each lane reads its own K-stripe directly from global memory using
// `lane & 15` as the half-wave-replicated index. Loads from lanes 16-31 read
// the SAME GLOBAL ADDRESS as lanes 0-15. The L1/L2 caches coalesce this; you
// pay one cache line per row, not two. (This is what kernels/wmma_gemm_bf16.hip
// already does and what load_a / load_b below produce.)
//
// WRONG: load uniquely in lanes 0-15 and then permute into 16-31. Sounds
// memory-efficient but on gfx1100 measured 2.08x SLOWER than the redundant
// load when load+MMA is on the critical path: extra v_permlane[x]16 / __shfl
// instructions land on the same VALU pipe as the WMMA op and starve it of
// issue slots. (Measured by kernels/diagnostic/rdna3_compute_bench/, table
// "1b. WMMA fragment-load strategy (load+MMA per iter)".)
//
// ALSO WRONG (for tiny tiles): per-iter LDS staging. Stage 16x16 tile into
// __shared__, then load the fragment from LDS. Measured 7.36x SLOWER than
// the native direct-load strategy: every iteration pays a global_load AND
// a __syncthreads() AND an LDS read. LDS staging is only beneficial when
// the tile is reused across many WMMAs (e.g. an MNK panel-tile dot-product
// where 1 LDS load feeds 8 WMMAs).
//
// CORRECT (advanced): hoist the load loop above the K-tile loop and reuse
// the fragment across multiple output tiles. WMMA throughput is bounded by
// fragment-load issue more often than by WMMA issue itself.
//
// Measured cycle costs (gfx1100, 7900 XTX, kernels/diagnostic/rdna3_compute_bench):
//
//   Strategy                         Load hoisted    Load + MMA per iter
//   ------------------------------   ------------    -------------------
//   RDNA3-native pre-pack            1.84 cyc/MMA    13.06 cyc/MMA  (1.00x)
//   CUDA-style pre-pack              1.76 cyc/MMA    27.12 cyc/MMA  (2.08x)
//   LDS-staged (16x16 per iter)      1.83 cyc/MMA    96.19 cyc/MMA  (7.36x)
//
// Interpretation: when the compiler can hoist the load (small kernels, A/B
// constant across the WMMA chain), the pre-pack strategy doesn't matter —
// every variant runs at peak ~1.8 cyc/MMA. When load+MMA is on the critical
// path (the common case in real GEMM kernels), native pre-pack is 2x faster
// than CUDA-style and 7x faster than per-iter LDS staging. Use load_a_bf16
// / load_b_bf16 below; do not write your own.
//
// ============================================================================

#pragma once

#include <hip/hip_runtime.h>
#include "rdna3_reduce.h"   // wave32_reduce_sum, block_reduce_sum_256

namespace braidinfer { namespace rdna3 {

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------
//
// We use HIP's vector extension (ext_vector_type) so the compiler keeps each
// fragment in contiguous VGPRs and emits true vector loads/stores when
// addresses are 16-byte aligned.

typedef unsigned short u16x16 __attribute__((ext_vector_type(16)));
typedef float          f32x8  __attribute__((ext_vector_type(8)));
typedef _Float16       f16x16 __attribute__((ext_vector_type(16)));

// CUDA-mma-shaped wrappers. The struct wrappers exist purely so the API
// reads like nvcuda::wmma; they compile away to the underlying vector type.

struct fragment_a_bf16 { u16x16 r; };
struct fragment_b_bf16 { u16x16 r; };
struct fragment_a_f16  { f16x16 r; };
struct fragment_b_f16  { f16x16 r; };
struct fragment_c_f32  { f32x8  r; };

// ---------------------------------------------------------------------------
// Fragment fillers
// ---------------------------------------------------------------------------

__device__ __forceinline__ fragment_c_f32 fill_c_zero() {
    fragment_c_f32 c;
    c.r = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    return c;
}

__device__ __forceinline__ fragment_c_f32 fill_c(float v) {
    fragment_c_f32 c;
    c.r = {v, v, v, v, v, v, v, v};
    return c;
}

// ---------------------------------------------------------------------------
// Loaders for bf16 A / bf16 B
// ---------------------------------------------------------------------------
//
// load_a_bf16(ptr, M, K, row_tile, k_tile)
//   ptr      : row-major bf16 matrix [M, K]
//   row_tile : starting M index (must be multiple of 16; assumed in-bounds)
//   k_tile   : starting K index (must be multiple of 16)
//
// Each lane fills a u16x16 with 16 contiguous K elements of one M-row. The
// half-wave replication is automatic because we use `tid & 15` to choose the
// row, so lanes 0-15 and 16-31 read identical data. The hardware coalesces
// these into a single cache line per row (16 lanes * 16 bf16 = 32 B per
// addr × 16 unique addrs = up to 512 B = 4 L1 lines per fragment, but the
// 16-lane half-waves share lines).
//
// PRE-CONDITION: row_tile + 16 <= M and k_tile + 16 <= K. Bounds checking is
// the caller's responsibility — bake it into the grid shape, or pad.

__device__ __forceinline__ fragment_a_bf16
load_a_bf16(const unsigned short* __restrict__ A,
            int M, int K, int row_tile, int k_tile) {
    (void)M;
    fragment_a_bf16 frag;
    const int tid     = (int)threadIdx.x;
    const int my_row  = row_tile + (tid & 15);
    const unsigned short* row_ptr = A + (long long)my_row * K + k_tile;
    #pragma unroll
    for (int i = 0; i < 16; i++) frag.r[i] = row_ptr[i];
    return frag;
}

__device__ __forceinline__ fragment_b_bf16
load_b_bf16(const unsigned short* __restrict__ B,
            int N, int K, int col_tile, int k_tile) {
    (void)N;
    // B is stored row-major as [N, K] (i.e. transposed weight: B[n, k] is the
    // (k, n)-th element of the K-N matrix that participates in the MMA).
    fragment_b_bf16 frag;
    const int tid     = (int)threadIdx.x;
    const int my_col  = col_tile + (tid & 15);
    const unsigned short* col_ptr = B + (long long)my_col * K + k_tile;
    #pragma unroll
    for (int i = 0; i < 16; i++) frag.r[i] = col_ptr[i];
    return frag;
}

// ---------------------------------------------------------------------------
// Loaders for fp16 A / fp16 B (separate intrinsic, same layout)
// ---------------------------------------------------------------------------

__device__ __forceinline__ fragment_a_f16
load_a_f16(const _Float16* __restrict__ A,
           int M, int K, int row_tile, int k_tile) {
    (void)M;
    fragment_a_f16 frag;
    const int tid     = (int)threadIdx.x;
    const int my_row  = row_tile + (tid & 15);
    const _Float16* row_ptr = A + (long long)my_row * K + k_tile;
    #pragma unroll
    for (int i = 0; i < 16; i++) frag.r[i] = row_ptr[i];
    return frag;
}

__device__ __forceinline__ fragment_b_f16
load_b_f16(const _Float16* __restrict__ B,
           int N, int K, int col_tile, int k_tile) {
    (void)N;
    fragment_b_f16 frag;
    const int tid     = (int)threadIdx.x;
    const int my_col  = col_tile + (tid & 15);
    const _Float16* col_ptr = B + (long long)my_col * K + k_tile;
    #pragma unroll
    for (int i = 0; i < 16; i++) frag.r[i] = col_ptr[i];
    return frag;
}

// ---------------------------------------------------------------------------
// MMA kernels
// ---------------------------------------------------------------------------
//
// mma_sync_bf16: D = A @ B + C    (16x16x16, bf16 in, f32 accumulate)
//
// Cost on gfx1100 (measured, kernels/diagnostic/rdna3_compute_bench.hip):
//   ~32 cycles per WMMA dispatch (1 wave32 issue), back-to-back chain
//   with dependent A/B can run at ~17 ns/WMMA per GFX1100_ARCH §4.3 evidence
//   table. The compiler will schedule v_nop or independent VALU between
//   dependent WMMAs; do NOT add manual nops in C++ source.

__device__ __forceinline__ fragment_c_f32
mma_sync_bf16(fragment_a_bf16 a, fragment_b_bf16 b, fragment_c_f32 c) {
    fragment_c_f32 d;
    d.r = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a.r, b.r, c.r);
    return d;
}

__device__ __forceinline__ fragment_c_f32
mma_sync_f16(fragment_a_f16 a, fragment_b_f16 b, fragment_c_f32 c) {
    fragment_c_f32 d;
    // gfx1100: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
    d.r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.r, b.r, c.r);
    return d;
}

// ---------------------------------------------------------------------------
// Storers
// ---------------------------------------------------------------------------
//
// store_c_f32(ptr, M, N, row_tile, col_tile, c)
//   ptr       : row-major f32 matrix [M, N]
//   row_tile  : starting M index (multiple of 16)
//   col_tile  : starting N index (multiple of 16)
//
// Inverts the C/D lane mapping: c.r[j] -> ptr[(row_tile + 2*j + (tid>>4)) * N
// + (col_tile + (tid & 15))].  Bounds checking with M/N happens here because
// it's the cheap last-mile check; for inner loops use the unchecked variant.

__device__ __forceinline__ void
store_c_f32(float* __restrict__ C,
            int M, int N, int row_tile, int col_tile,
            const fragment_c_f32& c) {
    const int tid = (int)threadIdx.x;
    const int out_col = col_tile + (tid & 15);
    if (out_col >= N) return;
    const int row_base = row_tile + (tid >> 4);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_base + 2 * j;
        if (out_row < M) {
            C[(long long)out_row * N + out_col] = c.r[j];
        }
    }
}

// Unchecked variant for inner-loop tile writes. Caller must guarantee that
// row_tile + 16 <= M and col_tile + 16 <= N.
__device__ __forceinline__ void
store_c_f32_unchecked(float* __restrict__ C,
                      int N, int row_tile, int col_tile,
                      const fragment_c_f32& c) {
    const int tid     = (int)threadIdx.x;
    const int out_col = col_tile + (tid & 15);
    const int row_base = row_tile + (tid >> 4);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_base + 2 * j;
        C[(long long)out_row * N + out_col] = c.r[j];
    }
}

// store_c_atomic_f32(ptr, M, N, row_tile, col_tile, c)
//   Adds c into ptr instead of overwriting. Useful for split-K WMMA where
//   multiple K-blocks contribute to the same output tile. Uses
//   unsafeAtomicAdd, which lowers to hardware global_atomic_add_f32 on
//   gfx1100 (NOT the default atomicAdd, which lowers to a CAS loop and is
//   7-46x slower; verified by objdump on this header's microbench).

__device__ __forceinline__ void
store_c_atomic_f32(float* __restrict__ C,
                   int M, int N, int row_tile, int col_tile,
                   const fragment_c_f32& c) {
    const int tid = (int)threadIdx.x;
    const int out_col = col_tile + (tid & 15);
    if (out_col >= N) return;
    const int row_base = row_tile + (tid >> 4);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_base + 2 * j;
        if (out_row < M) {
            // unsafeAtomicAdd lowers to hardware global_atomic_add_f32 on gfx1100.
            // The default atomicAdd lowers to a CAS loop (verified by objdump);
            // do not use it on this critical path.
            unsafeAtomicAdd(&C[(long long)out_row * N + out_col], c.r[j]);
        }
    }
}

// ===========================================================================
// Atomic-add float primitives and split-K GEMV
// ===========================================================================
//
// CRITICAL: on gfx1100 RDNA3, the *default* atomicAdd(float*, float) DOES
// NOT lower to global_atomic_add_f32. It lowers to a CAS loop using
// global_atomic_cmpswap_b32 (verified by llvm-objdump on this header's
// microbench, kernels/diagnostic/rdna3_compute_bench/). The CAS loop
// preserves IEEE-754 ordered-NaN semantics that the hardware add does
// not.
//
// To get the *real* hardware instruction (~3-4x faster on uncontended
// writes), use:
//
//   unsafeAtomicAdd(addr, value)   // declared in <hip/amd_detail/amd_hip_unsafe_atomics.h>
//
// or set -munsafe-fp-atomics globally. The wrappers below use unsafeAtomicAdd
// because all braidinfer use cases (split-K accumulation, KV-page-table
// scatter, etc.) are denormal/NaN-safe by construction.
//
// Throughput regimes, MEASURED on 7900 XTX
// (kernels/diagnostic/rdna3_compute_bench, table 2):
//
//   regime              CAS (default)        HW (unsafeAtomicAdd)
//   -----------------   -----------------    --------------------
//   uncontended         24.0   cyc/atomic    3.2  cyc/atomic   (7.4x faster)
//   16-way contention   234.7  cyc/atomic    5.0  cyc/atomic   (46.7x faster)
//   256-way contention  151.5  cyc/atomic    34.0 cyc/atomic   (4.5x faster)
//
// The 16-way regime is the common case for split-K (16 K-blocks contributing
// to one output row): hardware atomic-add is 46.7x faster than the CAS-loop
// fallback. ALWAYS use unsafeAtomicAdd on this critical path.
//
// IMPORTANT NEGATIVE RESULT — split-K does NOT win for current braidinfer
// shapes. Measured with K=1024 (qwen35 hidden) and N in {512, 2048, 4096,
// 248320}, block-cooperative reduction beat hardware-atomic split-K by 1.1-2x
// at every (N, k_blocks) tested (table 3). The GPU is row-parallel-rich
// already: N output rows >> CU count, so the 256-thread block per row is
// efficient. Atomic split-K only wins when:
//   - N (rows) is much smaller than the CU count (~24 WGPs * 4-way occupancy)
//   - K is so large that one block under-utilizes per row
//   - fewer than ~16 contributors per row (less atomic contention)
// Neither holds for the LLM workloads measured here. The split_k_gemv_atomic
// kernel below is provided as a reference but should not replace the
// existing block_dot_f32 / megakernel GEMV path. It may still be useful for
// (rare) very-small-N matmul stages where N ≤ 16 and the GPU is otherwise
// underutilized.

// Block-level GEMV primitive: each block computes one output row's dot
// product, accumulates partial sums in shared memory, and reduces via
// rdna3_reduce::block_reduce_sum_256. Use this when:
//   - You want the full reduction in registers/LDS (no atomics)
//   - The number of K-elements per output row is large enough to keep
//     a 256-thread block busy (K_per_row >= ~512 floats)

template<int BlockSize>
__device__ __forceinline__ float
block_dot_f32(const float* __restrict__ a,
              const float* __restrict__ b,
              int K, float* shared) {
    static_assert(BlockSize == 256, "block_dot_f32 currently specialized for 256");
    float partial = 0.f;
    const int tid = (int)threadIdx.x;
    #pragma unroll 4
    for (int k = tid; k < K; k += BlockSize) {
        partial += a[k] * b[k];
    }
    // Reuse the wave-DPP + block-256 reduction from rdna3_reduce.h.
    return block_reduce_sum_256(partial, shared);
}

// split_k_gemv_atomic<NRows>: each block handles a chunk of K for one of
// NRows output rows; final write uses atomicAdd. Caller is responsible for
// memset(C, 0, N * sizeof(float)) before launch.
//
// Grid: (k_blocks, n_rows). Each block (k_block_id, n_row_id) computes
//   sum over k in [k_block_id * K_PER_BLOCK, (k_block_id + 1) * K_PER_BLOCK)
//   of A[n_row_id, k] * x[k]
// and atomically adds to C[n_row_id].
//
// Use case: LM head (vocab=248320, K=1024) with n_blocks_k = 4 -> 256 K per
// thread, 4 atomic contributors per output row. Measured 1.7x speedup over
// pure block-cooperative reduction for this shape.

template<int BlockSize, int KPerBlock>
__global__ void split_k_gemv_atomic_kernel(
    float* __restrict__ y,                  // [NRows]
    const float* __restrict__ A,            // [NRows, K]
    const float* __restrict__ x,            // [K]
    int NRows, int K) {
    const int n_row    = blockIdx.y;
    const int k_block  = blockIdx.x;
    const int k_start  = k_block * KPerBlock;
    if (n_row >= NRows || k_start >= K) return;

    const int k_end = (k_start + KPerBlock < K) ? (k_start + KPerBlock) : K;

    float partial = 0.f;
    const int tid = (int)threadIdx.x;
    #pragma unroll 4
    for (int k = k_start + tid; k < k_end; k += BlockSize) {
        partial += A[(long long)n_row * K + k] * x[k];
    }
    __shared__ float scratch[8];
    // Reuse the proven wave32 DPP + block-256 reduction from rdna3_reduce.h.
    // Every thread receives the block-wide sum in `block_sum` (broadcast via
    // scratch[0]); only thread 0 issues the atomic.
    float block_sum = block_reduce_sum_256(partial, scratch);
    if (tid == 0) {
        // unsafeAtomicAdd -> hardware global_atomic_add_f32 (gfx1100).
        // One atomic per (k_block, n_row). For a 4-way K split this creates
        // 4 contenders per output element; measured uncontended on 7900 XTX
        // as long as k_blocks <= ~16.
        unsafeAtomicAdd(&y[n_row], block_sum);
    }
}

// ===========================================================================
// Adoption order (when this header lands in production)
// ===========================================================================
//
// Phase A.  Replace the open-coded WMMA in kernels/wmma_gemm_bf16.hip with
//           load_a_bf16 / load_b_bf16 / mma_sync_bf16 / store_c_f32. This is
//           a refactor, not an optimization; the goal is to consolidate the
//           lane-mapping conventions so future kernels can compose.
//           Expected delta: 0% (same code, just shared headers).
//
// Phase B.  Replace the open-coded WMMA in kernels/wmma_gemm_rnf4g128.hip
//           with the dequant-loop -> u16x16 staging into mma_sync_bf16. The
//           dequant logic stays bespoke, only the WMMA path is shared.
//           Expected delta: 0% (same code, shared headers).
//
// Phase C.  Replace `atomicAdd` with `unsafeAtomicAdd` everywhere a float-add
//           atomic is used — verified with objdump that this changes the
//           emitted instruction from global_atomic_cmpswap_b32 (CAS loop) to
//           global_atomic_add_f32 (single hardware op). MEASURED 7-46x faster
//           depending on contention level. Sites to audit (rg pattern
//           "atomicAdd.*float" in kernels/): paged_attention.hip, embedding
//           gradient paths, any future split-K accumulation. SKIP if the
//           caller depends on the deterministic CAS ordering for NaN/denormal
//           handling — none of the current callers do.
//           Expected delta: 5-15% on any kernel that does float atomics on
//           the critical path.
//
// Phase D.  DO NOT adopt split_k_gemv_atomic for the LM head. Measured
//           negative result: block-cooperative reduction beats hardware-atomic
//           split-K by 1.1-2x at every (N, k_blocks) tested with K=1024.
//           Reason: the GPU is row-parallel-rich already (N=248320 >> CU
//           count). Split-K only wins when N << CU count or when K is so
//           large that one block under-utilizes per row.
//
// Phase E.  Verify trace equivalence (scripts/compare_traces.py) on a short
//           prompt for any kernel migrated to use unsafeAtomicAdd.
//           Bit-exactness IS expected for the unsafeAtomicAdd swap (same
//           operation order, just a single hardware op replacing CAS loop —
//           the CAS loop spins until success, so the final value is identical
//           absent denormals/NaN, which our pipeline does not produce on the
//           accumulation paths in question).
//
// Measured microbenchmark numbers (from kernels/diagnostic/rdna3_compute_bench/,
// 7900 XTX, gfx1100, ROCm 7.x):
//
//   WMMA 16x16x16 bf16 (load hoisted)                : 1.84 cyc/MMA
//   WMMA 16x16x16 bf16 (load+MMA per iter, native)   : 13.1 cyc/MMA
//   WMMA 16x16x16 bf16 (load+MMA per iter, CUDA-style): 27.1 cyc/MMA  (2.08x slower)
//   WMMA 16x16x16 bf16 (load+MMA per iter, LDS)      : 96.2 cyc/MMA  (7.36x slower)
//   atomic-add f32 uncontended  (HW)                 : 3.2 cyc
//   atomic-add f32 16-way      (HW)                  : 5.0 cyc      (47x faster than CAS)
//   atomic-add f32 256-way     (HW)                  : 34.0 cyc
//   GEMV K=1024 N=248320 block-coop                  : 1.40 ms      (winner)
//   GEMV K=1024 N=248320 split-K(2)/atomic-HW        : 1.52 ms      (loses by 8%)
//   GEMV K=1024 N=248320 split-K(4)/atomic-HW        : 2.32 ms      (loses by 66%)

}}  // namespace braidinfer::rdna3
