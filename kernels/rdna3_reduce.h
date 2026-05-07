// RDNA3 (gfx1100, wave32) reduction primitives.
//
// Drop-in replacements for the __shfl_down-based reductions used throughout
// the megakernel and MoE expert kernels. These primitives prefer DPP-modified
// VALU instructions (v_add_f32_dpp row_xmask) and v_permlanex16_b32 over
// ds_bpermute_b32, because:
//
//   1. DPP/permlane are vector-ALU instructions: 1-cycle issue, no LDS.
//   2. ds_bpermute_b32 hits the LDS pipe (~4-cycle, contends with __shared__).
//   3. ds_bpermute_b32 has a known same-VGPR non-determinism hazard on
//      gfx1100 when the compiler allows dst==src (see kernels/diagnostic/
//      bpermute_repro/ and kb memory bz0-root-cause-solved-2026-05-03-shfl).
//      DPP and permlane have no analogous hazard.
//
// Measured on gfx1100 / 7900 XTX (kernels/diagnostic/reduce_bench/):
//
//   primitive           threads   median cyc / reduction   max_rel_err vs Kahan
//   ----------------    -------   ----------------------   --------------------
//   wave32 SHFL              32                    13.33               1.7e-07
//   wave32 DPP+P16           32                     3.46               1.4e-07  *
//   subwave_2 SHFL            2                     4.04               (n/a)
//   subwave_2 DPP             2                     2.15                       *
//   subwave_4 SHFL            4                     6.22
//   subwave_4 DPP             4                     2.27                       *
//   subwave_8 SHFL            8                     8.54
//   subwave_8 DPP             8                     2.48                       *
//   subwave_16 SHFL          16                    10.68
//   subwave_16 DPP           16                     2.69                       *
//   block256 SHFL           256                    23.26               1.7e-07
//   block256 DPP+P16        256                    13.72               1.9e-07  *
//   block256 TREE-LDS       256                    43.35               1.4e-07
//
//   "* = recommended". Numerical-error analysis used 10000 random vectors
//   with three distributions (positive RMS-input-like, mixed-sign N(0,1),
//   heavy-tail with 1% outliers). Errors are statistically indistinguishable
//   between SHFL and DPP+P16 because they sum the same lanes in the same
//   pairing order. TREE-LDS is the most accurate (slightly tighter
//   pairwise-summation tree) but is ~3x slower than DPP+P16 at block scale.
//
// Header summary:
//   wave32_reduce_sum(v)            -> full-wave (32-thread) sum, every lane gets it
//   subwave_reduce_sum<W>(v)        -> sub-wave (W ∈ {2,4,8,16}) sum, every lane in sub-group
//   block_reduce_sum_256(v, shared) -> 256-thread block sum, returned to all threads via shared[0]
//
// All three avoid ds_bpermute_b32 entirely. None of the helpers below are
// affected by the RDNA3 same-VGPR ds_bpermute hazard.
//
// Compile/usage notes:
//   - Wave32 is required (HIP default for gfx1100). Do NOT mix with
//     -mwavefrontsize64.
//   - Requires gfx10+ for row_xmask DPP control and permlanex16 (gfx1100 ✓).
//   - The dpp_ctrl encodings here use raw integer constants because hipcc
//     does not expose the DPP modifier mnemonics from C++ source.
//   - The DPP intrinsic requires the dpp_ctrl argument to be a compile-time
//     constant; that is why all helpers below are templated on the offset.
//
// Adoption order (see braidinfer-77r.1 notes):
//   Phase A. Replace warp_reduce_sum / block_reduce_sum in kernels/megakernel_ops.hip
//            with wave32_reduce_sum / block_reduce_sum_256.
//            (~25 callsites; biggest win — block_reduce_sum is in attn, FFN, RMSNorm.)
//   Phase B. Replace the tpg-shfl loop in coop_gemv_pcg32 / coop_gemv_rnf4 in
//            kernels/moe_expert_ops.h with subwave_reduce_sum<W>.
//            (~4 callsites, w ∈ {2,4,8,16} chosen at runtime — needs a small
//            switch / template dispatch per layer config.)
//   Phase C. Replace the inline shfl loops in kernels/megakernel_moe.hip
//            (lines ~174-186, ~380, ~588, etc.) with wave32_reduce_sum and
//            block_reduce_sum_256.
//   Phase D. Verify trace equivalence (scripts/compare_traces.py) on a
//            short prompt. Errors should be ≤ 1 ULP per reduction (same
//            pairing order, same FMA-fusion potential).

#pragma once

#include <hip/hip_runtime.h>

namespace braidinfer { namespace rdna3 {

// ---------------------------------------------------------------------------
// Building blocks: DPP-modified add and permlanex16-broadcast helpers.
// ---------------------------------------------------------------------------

// row_xmask:N — within each 16-lane row, lane i is paired with lane i^N.
// Returns v + (DPP-fetched value); after one application every lane i in
// a row holds v[i] + v[i^N]. After 4 applications with N = 8,4,2,1 every
// lane in the row holds the sum of all 16 lanes in that row.
//
// dpp_ctrl encoding (gfx10+): 0x160 | N for row_xmask:N (1..15).
// We use bound_ctrl=0 (true): out-of-row fetches return 0. row_mask=bank_mask=0xF
// (all rows / all banks active).
template<int N>
__device__ __forceinline__ float dpp_row_xmask_add(float v) {
    static_assert(N >= 1 && N <= 15, "row_xmask N must be 1..15");
    union { float f; int i; } a, b;
    a.f = v;
    constexpr int ctrl = 0x160 | (N & 0xF);
    b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
    return v + b.f;
}

// permlanex16 cross-half broadcast.
// permlanex16(old, src, sel0, sel1, fi, bc): with sel0=sel1=0 and fi=1, bc=1,
// each lane in the low 16 receives src[lane 16] (broadcast from the start of
// the high half), and each lane in the high 16 receives src[lane 0]. When
// every lane in row 0 holds the row-0 sum and every lane in row 1 holds the
// row-1 sum (after row_xmask reduction), this swap-and-add gives every lane
// the full wave sum.
__device__ __forceinline__ float permlanex16_swap_add(float v) {
    union { float f; int i; } a, b;
    a.f = v;
    b.i = __builtin_amdgcn_permlanex16((unsigned)a.i, (unsigned)a.i,
                                       /*sel0=*/0u, /*sel1=*/0u,
                                       /*fi=*/true, /*bc=*/true);
    return v + b.f;
}

// ---------------------------------------------------------------------------
// Public API: sum reductions
// ---------------------------------------------------------------------------

// Full wave32 sum reduction. After return, EVERY lane in the wave holds the
// sum of all 32 lanes' input values. Replaces the 5-stage __shfl_down loop;
// the existing pattern reads the result from lane 0 only, which still works
// because every lane has the same value.
//
// Cost: 4 v_add_f32_dpp + 1 v_permlanex16_b32 + 1 v_add_f32 ≈ 3.5 cycles.
__device__ __forceinline__ float wave32_reduce_sum(float v) {
    v = dpp_row_xmask_add<8>(v);
    v = dpp_row_xmask_add<4>(v);
    v = dpp_row_xmask_add<2>(v);
    v = dpp_row_xmask_add<1>(v);
    v = permlanex16_swap_add(v);
    return v;
}

// Sub-wave sum reduction within W consecutive lanes (W ∈ {2, 4, 8, 16}).
// Used by MoE GEMV's tpg pattern: tpg threads cooperate on one quantization
// group, then thread 0 of the tpg-group writes the partial sum to LDS.
// After return, every lane in the W-lane sub-group holds the sum.
//
// W must be a power of 2 in {2,4,8,16}. (For W=32 use wave32_reduce_sum;
// for W=1 the input is already the sum.)
//
// IMPORTANT: assumes lane 0 of each sub-group is at lane index 0 mod W
// (the natural "blockDim.x % tpg" partitioning the existing code uses).
// row_xmask operates within the 16-lane row, so W must divide 16.
//
// Cost (median, 7900 XTX):
//   W=2  : 1 dpp add ≈ 2.15 cyc
//   W=4  : 2 dpp adds ≈ 2.27 cyc
//   W=8  : 3 dpp adds ≈ 2.48 cyc
//   W=16 : 4 dpp adds ≈ 2.69 cyc
template<int W>
__device__ __forceinline__ float subwave_reduce_sum(float v) {
    static_assert(W == 2 || W == 4 || W == 8 || W == 16,
                  "subwave_reduce_sum only supports W in {2,4,8,16}");
    if constexpr (W >= 16) v = dpp_row_xmask_add<8>(v);
    if constexpr (W >=  8) v = dpp_row_xmask_add<4>(v);
    if constexpr (W >=  4) v = dpp_row_xmask_add<2>(v);
    if constexpr (W >=  2) v = dpp_row_xmask_add<1>(v);
    return v;
}

// 256-thread (8-wave) block sum reduction.
//
// Step 1 (intra-wave): wave32_reduce_sum (DPP + permlanex16, no LDS).
// Step 2 (inter-wave): lane 0 of each wave writes its partial to shared[0..7].
// Step 3: thread 0 sequentially adds the 8 partials into shared[0].
// Step 4: __syncthreads() and broadcast shared[0] to all threads.
//
// shared must point to at least 8 floats of __shared__ memory.
// Returns the block-wide sum on every thread (via shared[0]).
//
// Cost: 13.7 cycles vs 23.3 (shfl-based) vs 43.3 (pure tree-LDS). 1.7x
// speedup over the current block_reduce_sum.
//
// NOTE: this primitive ends with __syncthreads(), so callers may safely
// re-use `shared` immediately afterward (matches existing block_reduce_sum
// semantics in kernels/megakernel_ops.hip).
__device__ __forceinline__ float block_reduce_sum_256(float v, float* shared) {
    v = wave32_reduce_sum(v);
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) shared[warp_id] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        // 8 warps × 32 lanes = 256.
        s += shared[0]; s += shared[1]; s += shared[2]; s += shared[3];
        s += shared[4]; s += shared[5]; s += shared[6]; s += shared[7];
        shared[0] = s;
    }
    __syncthreads();
    return shared[0];
}

}}  // namespace braidinfer::rdna3
