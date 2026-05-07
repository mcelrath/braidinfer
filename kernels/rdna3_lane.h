// RDNA3 (gfx1100, wave32) non-reduction lane primitives.
//
// Companion to kernels/rdna3_reduce.h, which covers wave-sum reductions via
// DPP butterfly + permlanex16. THIS header covers the rest of the
// CUDA-shaped lane vocabulary that maps cleanly onto single RDNA3 VALU/LDS
// instructions and avoids ds_bpermute_b32:
//
//   wave_ballot(pred)              -> __ballot_sync replacement, wave32 mask in u32
//   wave_any / wave_all / wave_count_predicate
//   lane_id() / wave_id_in_block()
//   lane_broadcast<src>(v)         -> v_readlane_b32 (1 src lane → all)
//   lane_broadcast(v, src)         -> v_readfirstlane / v_readlane (dynamic src)
//   lane_write<dst>(reg, v)        -> v_writelane_b32 (insert into one lane)
//   wave_swizzle<Ctrl>(v)          -> ds_swizzle_b32 (LDS, no-bpermute, fixed ctrl)
//   wave_quad_swap<N>(v)           -> ds_swizzle quad-perm (XOR within 4-lane group)
//   wave_xor_butterfly<N>(v)       -> ds_swizzle bitmask XOR pattern (any N in 1..15)
//   subwave_tile<W>                -> sub-wave broadcast / shfl helpers via permlane*
//   subwave_tile<W>::shfl_xor(v,N) -> XOR shfl within W-lane sub-group (no LDS)
//
// All helpers run on the VALU or LDS pipe; none lower to ds_bpermute_b32, so
// they are immune to the gfx1100 same-VGPR ds_bpermute hazard documented in
// kernels/diagnostic/bpermute_repro/ (cf. kb memory bz0-root-cause-solved-2026-05-03-shfl).
//
// Why these matter (vs CUDA equivalents on RDNA3):
//   - HIP's __ballot returns u64 wave64 mask even on wave32 hardware; the
//     CUDA shape is u32 wave-mask. wave_ballot() forces the wave32 form.
//   - HIP's __shfl_sync does not exist; __shfl{xor,up,down,_sync} all lower
//     to ds_bpermute_b32. Lane primitives here use v_readlane / v_writelane
//     / ds_swizzle / permlane16 instead — all 1-cycle VALU or single-instruction
//     LDS-pipe ops, NOT the multi-cycle bpermute path.
//   - cooperative_groups::tile_partition<N> (CUDA) lowers on AMD to scalar
//     ds_bpermute via runtime lane indices. subwave_tile<W> uses fixed-pattern
//     primitives (DPP, permlane16, ds_swizzle) so the compiler can emit
//     single VALU ops with no cross-pipe latency.
//
// Header summary with measured cycles (kernels/diagnostic/lane_bench/
// rdna3_lane_bench.hip, 7900 XTX gfx1100 ROCm 7.1.x, blockDim=32, ITER=4096,
// 64 blocks; median s_memrealtime ticks per op):
//
//   primitive                          cycles/op   notes
//   --------------------------------   ---------   --------------------------
//   wave_ballot(pred)                  2.88        v_cmp_*_e32 → vcc_lo (SGPR)
//   ballot HIP-w64 (reference)         2.88        same; HIP wraps to u64
//   lane_broadcast<src>(v) const src   2.08        v_readlane_b32, imm src
//   lane_broadcast(v, src_lane)        2.76        v_readlane_b32 with SGPR src
//   lane_read_first(v)                 2.08        v_readfirstlane_b32
//   __shfl(v, src) (reference)         3.90        ds_bpermute_b32 (LDS pipe)
//   wave_xor_butterfly<N>(v)           3.68        ds_swizzle_b32 BITMASK_PERM
//   wave_quad_swap<N>(v)               3.69        ds_swizzle_b32 QUAD_PERM
//   __shfl_xor (reference)             3.14        ds_bpermute_b32
//   subwave_tile<W>::broadcast<R>      3.68        ds_swizzle_b32 BROADCAST,W,R
//   subwave_tile<W>::shfl_xor<N> DPP   2.07        v_mov_b32_dpp row_xmask:N
//   __shfl_xor (subwave reference)     3.13        ds_bpermute_b32
//   wave32_reduce_max (DPP+permlane16) 5.00        4×DPP-mov + max + permlane
//   block256 max LDS-tree (reference) 46.18        9-stage shared-mem tree
//   block256 max DPP+LDS              15.31        wave_reduce_max + 8-warp LDS
//
// Headline ratios:
//   3.0x  block256 max:                    DPP+LDS       vs LDS-tree
//   1.9x  lane_broadcast<const>:           v_readlane    vs __shfl
//   1.5x  subwave_tile<8>::shfl_xor<4>:    DPP           vs __shfl_xor
//   9.2x  wave32_reduce_max:               DPP+permlane  vs LDS-tree-max-256
//
// Where the lane primitives unlock perf wins in braidinfer:
//   - Online-softmax max-broadcast in op_attn_paged / op_gqa_attn (kernels/
//     megakernel_ops.hip:1612, :1811, :1990, :2077, :2178, :2203): each
//     timestep needs a wave-wide max of `score`, then broadcast m_new back
//     to all lanes for correction = expf(m - m_new). Currently the related
//     all-block max-reduction (e.g. residual-quant scale finder at lines
//     2178/2203) uses an LDS-tree max @ 46 cycles per reduce. Replacing
//     with `block256_reduce_max_dpp_lds` (wave32_reduce_max + 8-warp LDS
//     tip) measures 15.3 cycles — 3.0x speedup, 30 cyc saved per softmax
//     timestep × seq_len. wave32_reduce_max alone (intra-wave, when only
//     a wave needs it) is 9.2x faster than the LDS path at 5.0 cycles.
//   - MoE expert-skip ballot (kernels/megakernel_moe.hip:174-186, :380, :588):
//     before launching coop_gemv across an expert, we want
//     `if (wave_any(eid_assigned == self_eid))` to early-exit waves that have
//     no work. Currently a __syncthreads() + LDS flag; ballot is single-cycle.
//   - subwave_tile<W>::broadcast for tpg-style intra-group sharing in
//     coop_gemv_pcg32 / coop_gemv_rnf4 (moe_expert_ops.h:56,121,183,244):
//     the per-quant-group scale only needs to be loaded by one lane and
//     broadcast to its W-lane sub-group. Replace `__shfl(scale, 0, W)` with
//     `subwave_tile<W>::broadcast(scale, 0)`.
//
// Compile/usage:
//   - Wave32 only (default for gfx1100 HIP). Mixing with -mwavefrontsize64
//     will silently miscompile wave_ballot's u32 width.
//   - Requires gfx10+ for v_permlane*/ds_swizzle bitmask-perm (gfx1100 OK).
//   - All `static_assert`s use the wave32 lane indexing convention
//     (lane = threadIdx.x & 31).
//
// Design notes / non-obvious RDNA3 lane semantics vs CUDA:
//   - `__builtin_amdgcn_readlane(v, src)` requires `src` to be uniform across
//     the wave (lowers to v_readlane reading SGPR). It is NOT a per-lane
//     gather — every lane gets the SAME src lane's value. CUDA's __shfl_sync
//     supports per-lane src; that lowers to ds_bpermute and is in rdna3_reduce.h
//     territory if we ever need it. We do NOT need it for braidinfer's
//     existing patterns (broadcast-from-fixed-lane is what every callsite uses).
//   - `__builtin_amdgcn_ballot_w32(pred)` returns a u32 mask of which lanes
//     evaluated `pred` true. It is the natural wave32 form. HIP's plain
//     `__ballot(pred)` returns u64; on wave32 hardware the upper 32 bits are
//     always 0 so `(uint32_t)__ballot(pred)` is equivalent but emits a
//     redundant 64-bit move. We prefer the explicit w32 builtin.
//   - ds_swizzle_b32 with bitmask-perm uses control word
//     ctrl = (and_mask << 10) | (or_mask << 5) | xor_mask, with top bit 0.
//     XOR butterflies (lane ^= N) are the most useful pattern: ctrl = N
//     (and=0x1F, or=0). The control MUST be a compile-time constant; that's
//     why these helpers are templated.
//   - ds_swizzle quad-perm (top-bit set) reorders within each group of 4 lanes
//     using a 2-bit-per-lane lookup. Useful for fixed shuffles that DPP
//     can't express (e.g. arbitrary mapping within 4-lane groups). Encoded as
//     ctrl = 0x8000 | (l3<<6 | l2<<4 | l1<<2 | l0).
//   - There is NO equivalent of __activemask on RDNA3: predication is exposed
//     via EXEC, but EXEC is implicit and not user-readable as a value the way
//     CUDA's activemask is. wave_ballot(true) gives the active-lane mask
//     (all bits set for any lane that runs the instruction), which is the
//     practical equivalent.
//
// (See rdna3_lane_bench.hip for cycle measurements and asm verification.)

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace braidinfer { namespace rdna3 {

// ---------------------------------------------------------------------------
// Lane / wave identity
// ---------------------------------------------------------------------------

// Lane id within wave32. Equivalent to threadIdx.x & 31, but expressed as
// the AMDGCN intrinsic so the compiler can keep it in an SGPR-friendly form.
__device__ __forceinline__ uint32_t lane_id() {
    return (uint32_t)__builtin_amdgcn_mbcnt_hi(
                         ~0u,
                         __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Wave id within block (0..7 for blockDim.x=256). Computed from threadIdx.x
// because RDNA3 has no dedicated wave-id intrinsic exposed to HIP at C++ level.
__device__ __forceinline__ uint32_t wave_id_in_block() {
    return (uint32_t)(threadIdx.x >> 5);
}

// ---------------------------------------------------------------------------
// Ballot / any / all (wave32 active-lane mask)
// ---------------------------------------------------------------------------

// CUDA __ballot_sync replacement for wave32. Returns a 32-bit mask whose
// bit i is set iff lane i was active AND `pred` evaluated true on lane i.
//
// HIP's plain __ballot(pred) returns u64 even on wave32 (upper 32 bits zero);
// this helper returns u32 directly via the explicit wave32 builtin so callers
// can use it as the natural CUDA shape (popc, ctz, branch on != 0, etc.).
__device__ __forceinline__ uint32_t wave_ballot(bool pred) {
    return (uint32_t)__builtin_amdgcn_ballot_w32(pred);
}

// Active-lane mask (every lane that executes this instruction). Drop-in for
// CUDA __activemask().
__device__ __forceinline__ uint32_t wave_active_mask() {
    return (uint32_t)__builtin_amdgcn_ballot_w32(true);
}

// CUDA __any_sync / __all_sync over the wave (active lanes only).
__device__ __forceinline__ bool wave_any(bool pred) {
    return __builtin_amdgcn_ballot_w32(pred) != 0;
}

__device__ __forceinline__ bool wave_all(bool pred) {
    return __builtin_amdgcn_ballot_w32(!pred) == 0;
}

// Count active lanes for which pred is true. Useful for histogram-style
// in-wave counting (e.g. how many experts are assigned to this wave's slot).
__device__ __forceinline__ int wave_count_predicate(bool pred) {
    return __builtin_popcount((uint32_t)__builtin_amdgcn_ballot_w32(pred));
}

// ---------------------------------------------------------------------------
// Direct lane read / write (v_readlane / v_writelane)
// ---------------------------------------------------------------------------
//
// These are the single-instruction RDNA3 primitives that CUDA approximates
// with __shfl_sync(v, src). On RDNA3 the readlane intrinsic generates
// v_readlane_b32 (VALU, ~1 cycle, no LDS) — strictly better than
// ds_bpermute_b32 for the broadcast-from-fixed-lane pattern.
//
// `__builtin_amdgcn_readlane` requires the source-lane index to be uniform
// across the wave (it lowers to v_readlane_b32 reading from an SGPR). For
// compile-time-constant src lane this is trivially satisfied.

// Broadcast value from a single fixed source lane to every lane in the wave.
// Compile-time src lane → single v_readlane_b32 instruction, ~1 cycle.
template<int SrcLane>
__device__ __forceinline__ float lane_broadcast(float v) {
    static_assert(SrcLane >= 0 && SrcLane < 32, "SrcLane must be in 0..31 for wave32");
    union { float f; int i; } u;
    u.f = v;
    int x = __builtin_amdgcn_readlane(u.i, SrcLane);
    union { int i; float f; } o; o.i = x;
    return o.f;
}

// Broadcast int variant.
template<int SrcLane>
__device__ __forceinline__ int lane_broadcast_i32(int v) {
    static_assert(SrcLane >= 0 && SrcLane < 32, "SrcLane must be in 0..31 for wave32");
    return __builtin_amdgcn_readlane(v, SrcLane);
}

// Dynamic-source-lane broadcast. `src_lane` MUST be uniform across the wave
// (typically it lives in an SGPR — e.g. a value already reduced or a scalar
// loop counter). If different lanes pass different src_lane, behavior is
// undefined per the AMDGCN ISA (v_readlane reads SGPR).
//
// Use lane_broadcast<C>(v) instead when src is a compile-time constant.
__device__ __forceinline__ float lane_broadcast(float v, int src_lane) {
    union { float f; int i; } u;
    u.f = v;
    int x = __builtin_amdgcn_readlane(u.i, src_lane);
    union { int i; float f; } o; o.i = x;
    return o.f;
}

// Read first active lane's value. Useful when you've already reduced a value
// and want to broadcast it without caring which lane currently holds it.
__device__ __forceinline__ float lane_read_first(float v) {
    union { float f; int i; } u;
    u.f = v;
    int x = __builtin_amdgcn_readfirstlane(u.i);
    union { int i; float f; } o; o.i = x;
    return o.f;
}

__device__ __forceinline__ int lane_read_first_i32(int v) {
    return __builtin_amdgcn_readfirstlane(v);
}

// Write `v` (the value supplied by lane 0, or any wave-uniform value) into
// lane DstLane of `reg`, leaving all other lanes' value of `reg` unchanged.
//
// Maps to a single v_writelane_b32 instruction. The DstLane index AND the
// source value `v` must be wave-uniform — v_writelane_b32 takes both from
// SGPRs at the ISA level. (Per AMDGCN semantics, the value to insert must
// be uniform across the wave; if you pass a per-lane value the compiler
// will emit a v_readfirstlane to coerce it, which is then what gets written.)
//
// There is no clang builtin for writelane; we emit inline asm directly.
//
// Common uses: build a small per-wave vector by accumulating one lane at a
// time, or insert a scalar correction into a specific lane after computation.
template<int DstLane>
__device__ __forceinline__ float lane_write(float reg, float v) {
    static_assert(DstLane >= 0 && DstLane < 32, "DstLane must be in 0..31 for wave32");
    union { float f; int i; } ru, vu;
    ru.f = reg; vu.f = v;
    int rv = ru.i;
    // v_writelane_b32 vDst, sSrc, sLane (destructively writes one lane of vDst).
    // We pre-load rv with the existing value of `reg` then overwrite lane DstLane.
    asm volatile(
        "v_writelane_b32 %0, %1, %2"
        : "+v"(rv)
        : "s"(vu.i), "n"(DstLane)
    );
    union { int i; float f; } o; o.i = rv;
    return o.f;
}

template<int DstLane>
__device__ __forceinline__ int lane_write_i32(int reg, int v) {
    static_assert(DstLane >= 0 && DstLane < 32, "DstLane must be in 0..31 for wave32");
    int rv = reg;
    asm volatile(
        "v_writelane_b32 %0, %1, %2"
        : "+v"(rv)
        : "s"(v), "n"(DstLane)
    );
    return rv;
}

// ---------------------------------------------------------------------------
// ds_swizzle_b32 — fixed-pattern lane permutation (LDS pipe, no LDS allocation)
// ---------------------------------------------------------------------------
//
// ds_swizzle_b32 is a single instruction that permutes lanes within groups of
// 32 lanes using a compile-time-constant pattern. Despite the "ds_" prefix
// it does NOT touch LDS memory — it uses the LDS data path purely as a
// crossbar. Latency measured at ~3.7 cycles per op on gfx1100 (slightly more
// than DPP's ~2 cycles, but ds_swizzle reaches patterns DPP cannot).
//
// Two control modes:
//
//   QDMode (quad-dword reorder): top bit of ctrl set, low 8 bits hold a
//     2-bit-per-lane lookup. Within each 4-lane group, lane i takes the value
//     from lane(group_base + lookup[i]). ctrl = 0x8000 | (l3<<6 | l2<<4 | l1<<2 | l0).
//
//   BitMaskPerm (XOR/AND/OR per lane): top bit clear, ctrl encodes
//     (xor_mask<<10) | (or_mask<<5) | and_mask. Each lane i fetches from
//     lane ((i AND and_mask) OR or_mask) XOR xor_mask. (Note: the bit-field
//     layout is XOR-high / OR-mid / AND-low — verified empirically against
//     LLVM 17/ROCm 7.1.x via swizzle_probe.hip; some published ISA refs put
//     AND in the high bits, but the LLVM intrinsic uses XOR-high.) The most
//     useful sub-mode is pure XOR (and=0x1F, or=0) for butterfly patterns.
//
// All ctrl values must be compile-time constants — that's why the helpers
// are templated.

// Raw escape hatch for callers that know their own ctrl encoding.
template<int Ctrl>
__device__ __forceinline__ float wave_swizzle(float v) {
    union { float f; int i; } u;
    u.f = v;
    int x = __builtin_amdgcn_ds_swizzle(u.i, Ctrl);
    union { int i; float f; } o; o.i = x;
    return o.f;
}

template<int Ctrl>
__device__ __forceinline__ int wave_swizzle_i32(int v) {
    return __builtin_amdgcn_ds_swizzle(v, Ctrl);
}

// XOR butterfly within the wave: lane i exchanges value with lane i ^ N.
// Implemented via BitMaskPerm with xor_mask=N, and_mask=0x1F, or_mask=0.
//
// For N=1,2,4,8,16: classic butterfly stage. For N=16 this is the cross-half
// swap that DPP cannot reach (DPP row_xmask is limited to N <= 15 within a
// 16-lane row); on gfx1100 the alternative is permlanex16, which has the
// same effect but a longer latency than ds_swizzle for this specific pattern.
//
// N must be in 1..31 (any single-bit or composite XOR pattern across 32 lanes).
template<int N>
__device__ __forceinline__ float wave_xor_butterfly(float v) {
    static_assert(N >= 1 && N <= 31, "wave_xor_butterfly N must be in 1..31");
    constexpr int ctrl = ((N & 0x1F) << 10) | (0 << 5) | 0x1F;
    return wave_swizzle<ctrl>(v);
}

template<int N>
__device__ __forceinline__ int wave_xor_butterfly_i32(int v) {
    static_assert(N >= 1 && N <= 31, "wave_xor_butterfly N must be in 1..31");
    constexpr int ctrl = ((N & 0x1F) << 10) | (0 << 5) | 0x1F;
    return wave_swizzle_i32<ctrl>(v);
}

// Quad swap: within each 4-lane group, swap lanes (0<->N, 1<->N^1, etc).
// N must be 1, 2, or 3 (XOR pattern within a 4-lane group).
//   N=1: pair-swap        (0<->1, 2<->3)
//   N=2: half-swap        (0<->2, 1<->3)
//   N=3: full-reverse     (0<->3, 1<->2)
//
// Implemented via QDMode: lookup[i] = i ^ N gives the XOR pattern.
template<int N>
__device__ __forceinline__ float wave_quad_swap(float v) {
    static_assert(N >= 1 && N <= 3, "wave_quad_swap N must be 1, 2, or 3");
    // Lookup: each lane i takes from lane (i ^ N) within its 4-lane group.
    constexpr int l0 = (0 ^ N) & 0x3;
    constexpr int l1 = (1 ^ N) & 0x3;
    constexpr int l2 = (2 ^ N) & 0x3;
    constexpr int l3 = (3 ^ N) & 0x3;
    constexpr int ctrl = 0x8000 | (l3 << 6) | (l2 << 4) | (l1 << 2) | l0;
    return wave_swizzle<ctrl>(v);
}

// Quad reverse (lane i within each 4-group ↔ lane 3-i). Convenience alias for
// the most common quad permutation.
__device__ __forceinline__ float wave_quad_reverse(float v) {
    return wave_quad_swap<3>(v);
}

// ---------------------------------------------------------------------------
// Wave-wide max via DPP butterfly (companion to rdna3_reduce.h sum)
// ---------------------------------------------------------------------------
//
// Every callsite of online-softmax max in braidinfer (op_attn_paged,
// op_gqa_attn) computes `m_new = max(m, score)` per lane then needs the
// wave-wide max broadcast back. The current code uses the LDS-tree max in
// shared memory (~25-40 cycles); butterfly-DPP max-fold + permlanex16 is
// the same shape as wave32_reduce_sum but with `fmaxf` in place of `+`,
// giving ~3-4 cycles for the entire op.
//
// (Lives in rdna3_lane.h rather than rdna3_reduce.h because it's a max,
//  not a sum, and rdna3_reduce.h scope was bounded to sums; consider folding
//  these two headers together later.)

template<int N>
__device__ __forceinline__ float dpp_row_xmask_max(float v) {
    static_assert(N >= 1 && N <= 15, "row_xmask N must be 1..15");
    union { float f; int i; } a, b;
    a.f = v;
    constexpr int ctrl = 0x160 | (N & 0xF);
    b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
    return fmaxf(v, b.f);
}

__device__ __forceinline__ float permlanex16_swap_max(float v) {
    union { float f; int i; } a, b;
    a.f = v;
    b.i = __builtin_amdgcn_permlanex16((unsigned)a.i, (unsigned)a.i,
                                       0u, 0u, /*fi=*/true, /*bc=*/true);
    return fmaxf(v, b.f);
}

// Full wave32 max: every lane ends up with the wave-wide max of its input.
// Cost: 4 v_max_f32_dpp + 1 v_permlanex16_b32 + 1 v_max_f32 ≈ 3.5 cycles.
__device__ __forceinline__ float wave32_reduce_max(float v) {
    v = dpp_row_xmask_max<8>(v);
    v = dpp_row_xmask_max<4>(v);
    v = dpp_row_xmask_max<2>(v);
    v = dpp_row_xmask_max<1>(v);
    v = permlanex16_swap_max(v);
    return v;
}

// Sub-wave max within W consecutive lanes (W ∈ {2,4,8,16}). Mirrors
// rdna3::subwave_reduce_sum.
template<int W>
__device__ __forceinline__ float subwave_reduce_max(float v) {
    static_assert(W == 2 || W == 4 || W == 8 || W == 16,
                  "subwave_reduce_max only supports W in {2,4,8,16}");
    if constexpr (W >= 16) v = dpp_row_xmask_max<8>(v);
    if constexpr (W >=  8) v = dpp_row_xmask_max<4>(v);
    if constexpr (W >=  4) v = dpp_row_xmask_max<2>(v);
    if constexpr (W >=  2) v = dpp_row_xmask_max<1>(v);
    return v;
}

// ---------------------------------------------------------------------------
// subwave_tile<W>: cooperative_groups::tile_partition<W> shaped abstraction
// ---------------------------------------------------------------------------
//
// CUDA exposes thread_block_tile<W> for sub-warp tiles with shfl_xor /
// shfl_up / shfl_down / broadcast members. On AMDGCN the templated
// implementation lowers each of those to ds_bpermute_b32 with a runtime lane
// index — which is the slow path. subwave_tile<W> below maps the same
// surface area to fixed-pattern DPP / permlane / readlane primitives so each
// op is a single VALU instruction.
//
// Constraints:
//   - W ∈ {2, 4, 8, 16} (must divide 16 because DPP row_xmask operates within
//     a 16-lane row). For W=32 use the wave32_reduce_* primitives directly.
//   - Lane 0 of each W-tile must be at lane index 0 mod W (the natural
//     blockDim.x % W partitioning braidinfer's MoE coop_gemv already uses).
//   - All shfl_xor lane-offsets N must satisfy N < W and be compile-time
//     constants.
template<int W>
struct subwave_tile {
    static_assert(W == 2 || W == 4 || W == 8 || W == 16,
                  "subwave_tile only supports W in {2,4,8,16}");
    static constexpr int width = W;

    // Lane index within the W-tile (0..W-1). Equal to lane_id() & (W-1).
    __device__ __forceinline__ static uint32_t thread_rank() {
        return lane_id() & (W - 1);
    }

    // XOR-shfl within the W-tile: each lane exchanges value with lane (rank ^ N).
    // N must be < W and a compile-time constant.
    template<int N>
    __device__ __forceinline__ static float shfl_xor(float v) {
        static_assert(N >= 1 && N < W, "shfl_xor N must be in 1..W-1");
        union { float f; int i; } a, b;
        a.f = v;
        constexpr int ctrl = 0x160 | (N & 0xF);
        b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
        return b.f;
    }

    // Sub-tile sum reduction — every lane in the tile ends up with the sum.
    __device__ __forceinline__ static float reduce_sum(float v) {
        if constexpr (W >= 16) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x168, 0xF, 0xF, 0);
            v += b.f;
        }
        if constexpr (W >= 8) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x164, 0xF, 0xF, 0);
            v += b.f;
        }
        if constexpr (W >= 4) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x162, 0xF, 0xF, 0);
            v += b.f;
        }
        if constexpr (W >= 2) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x161, 0xF, 0xF, 0);
            v += b.f;
        }
        return v;
    }

    // Sub-tile max reduction.
    __device__ __forceinline__ static float reduce_max(float v) {
        if constexpr (W >= 16) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x168, 0xF, 0xF, 0);
            v = fmaxf(v, b.f);
        }
        if constexpr (W >= 8) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x164, 0xF, 0xF, 0);
            v = fmaxf(v, b.f);
        }
        if constexpr (W >= 4) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x162, 0xF, 0xF, 0);
            v = fmaxf(v, b.f);
        }
        if constexpr (W >= 2) {
            union { float f; int i; } a, b;
            a.f = v; b.i = __builtin_amdgcn_update_dpp(0, a.i, 0x161, 0xF, 0xF, 0);
            v = fmaxf(v, b.f);
        }
        return v;
    }

    // Broadcast lane(src_rank)'s value to all lanes in the W-tile.
    // src_rank must be in 0..W-1 and a compile-time constant.
    //
    // Implementation note: v_readlane reads from the absolute wave lane index,
    // so we need a way to give each W-tile its own broadcast source. For the
    // common case src_rank == 0, every W-tile's lane 0 sits at lane index
    // (tile_id * W). DPP row_broadcast (ctrl 0x142..0x14F) cannot directly
    // express "broadcast lane 0 of each W-group" for arbitrary W; ds_swizzle
    // BitMaskPerm with and_mask = ~(W-1), or_mask = src_rank, xor_mask = 0
    // does exactly that (every lane fetches from the lane at offset src_rank
    // within its own W-tile).
    template<int SrcRank>
    __device__ __forceinline__ static float broadcast(float v) {
        static_assert(SrcRank >= 0 && SrcRank < W, "SrcRank must be in 0..W-1");
        // BitMaskPerm: lane_i_fetches_from = ((i AND and_mask) OR or_mask) XOR xor_mask
        //   and_mask = 0x1F & ~(W-1)   (zero out the in-tile rank bits)
        //   or_mask  = SrcRank          (force in-tile rank to SrcRank)
        // Bit layout (LLVM intrinsic, verified empirically): XOR<<10 | OR<<5 | AND.
        constexpr int and_mask = 0x1F & ~(W - 1);
        constexpr int or_mask  = SrcRank;
        constexpr int ctrl = (0 << 10) | (or_mask << 5) | (and_mask & 0x1F);
        return wave_swizzle<ctrl>(v);
    }

    // Predicate-true count within the W-tile (popcount of in-tile ballot bits).
    __device__ __forceinline__ static int count_predicate(bool pred) {
        uint32_t mask = (uint32_t)__builtin_amdgcn_ballot_w32(pred);
        // Extract the W bits belonging to this lane's tile.
        uint32_t tile_base = lane_id() & ~(W - 1);
        uint32_t tile_mask = ((1u << W) - 1u) << tile_base;
        return __builtin_popcount(mask & tile_mask);
    }

    // Any/all within the W-tile.
    __device__ __forceinline__ static bool any(bool pred) {
        return count_predicate(pred) > 0;
    }
    __device__ __forceinline__ static bool all(bool pred) {
        return count_predicate(pred) == W;
    }
};

}}  // namespace braidinfer::rdna3
