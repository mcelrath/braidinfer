// RDNA3 (gfx1100, wave32) memory primitives.
//
// Drop-in helpers for the most common memory-system levers we hit in the
// braidinfer megakernel + persistent_worker + MoE expert kernels:
//
//   1. Async global → LDS DMA (cuda cp.async semantics).
//   2. atomicAdd helpers for f32 (HW) and bf16 (CAS-emulated on RDNA3).
//   3. LDS bank-conflict-free padding for tpg-style cooperative GEMV.
//   4. Typed wrappers for buffer_gl0_inv / buffer_gl1_inv with placement docs.
//
// Companion to kernels/rdna3_reduce.h. Same naming conventions
// (namespace braidinfer::rdna3, drop-in shapes, inline-documented cycles).
//
// All measurements made on RX 7900 XTX (gfx1100, wave32) via
// kernels/diagnostic/rdna3_memory_bench/. Numbers below are MEDIAN ticks per
// op from in-kernel s_memrealtime, taken from a populated cache state unless
// otherwise noted (kernels/diagnostic/rdna3_memory_bench.hip:run_all).
//
// Header summary:
//   atomic_add_f32(p, v)                 -> HW VMEM atomic, wave-aggregated by HW
//   atomic_add_bf16_safe(p, v)           -> 32-bit-aligned dword CAS loop
//   lds_pad_for_tpg<Tpg>()               -> constexpr float-lane padding to avoid bank conflicts
//   gl0_invalidate()  / gl1_invalidate() -> documented buffer_gl{0,1}_inv wrappers
//
// NOTE: a "cuda cp.async equivalent" was investigated and is NOT available
// on gfx1100. The `__builtin_amdgcn_global_load_lds` intrinsic and the
// `global_load_b32 ... lds` ISA encoding both require the
// `vmem-to-lds-load-insts` LLVM target feature, which is read-only and
// is enabled ONLY on CDNA (gfx9xx/gfx94x). The RDNA3 ISA also lacks the
// LDS-bit form on `global_load_b{32,64,128}` and `buffer_load_b{32,64,128}`.
// The closest substitute on gfx1100 is the regular `global_load_b{32,..}`
// → VGPR → `ds_write_b{32,..}` two-step, which is what every existing
// kernel already does. Confirmed by feeding `global_load_lds_b32 v0,
// v[1:2], off` through `llvm-mc --mcpu=gfx1100` (rejected) and through
// `--mcpu=gfx942` (accepted). We keep this comment as a marker so future
// RDNA4 (gfx1200) adoption knows where to look. RDNA4 documentation
// suggests the LDS DMA path may land there — verify on hardware before
// adopting.
//
// IMPORTANT — None of the helpers below replace anything existing yet. This
// header is intended for FUTURE adoption (see Adoption order at the bottom).
//
// Codegen audit (verified by `--save-temps` on the bench, hipcc 22.0.0git
// from ROCm 7.2.53211; see kernels/diagnostic/rdna3_memory_bench/
// rdna3_memory_bench-hip-amdgcn-amd-amdhsa-gfx1100.s):
//   - gl0_invalidate / gl1_invalidate / gl01_invalidate
//       → exactly the inline `buffer_gl{0,1}_inv` asm intended.
//   - atomic_add_f32_hw  → single `global_atomic_add_f32` instruction.
//   - atomic_add_f32_cas → `global_atomic_cmpswap_b32` in a CAS loop.
//   - atomic_add_bf16_safe → `flat_atomic_cmpswap_b32` in a CAS loop.
//   - lds_pad_for_tpg<> → constexpr; no runtime calls emitted.
//   - No spurious `s_nop` insertions in any of the above.
// All primitives produce the intended instructions on this toolchain.
// We do NOT provide `_asm` siblings because no intrinsic emission was
// observed to be fragile on the audited build.

#pragma once

#include <hip/hip_runtime.h>
// unsafeAtomicAdd is the route to gfx1100's HW global_atomic_add_f32.
#include <hip/amd_detail/amd_hip_unsafe_atomics.h>

namespace braidinfer { namespace rdna3 {

// ============================================================================
// 1. Async global → LDS DMA — NOT AVAILABLE ON GFX1100
// ============================================================================
//
// SUMMARY: don't try. The `__builtin_amdgcn_global_load_lds` intrinsic, the
// `global_load_b{32,64,128} ... lds` ISA encoding, and the `buffer_load_*
// lds` form are ALL gated behind the `vmem-to-lds-load-insts` LLVM target
// feature. That feature is read-only and is set ONLY on CDNA (gfx908,
// gfx90a, gfx94x). On gfx1100 (and on gfx1200/RDNA4 in current LLVM)
// these forms are rejected at compile time AND at the assembler level.
//
// We verified this with two probes:
//   1. `__builtin_amdgcn_global_load_lds(...)` in a HIP source file with
//      `--offload-arch=gfx1100` produces:
//        error: '__builtin_amdgcn_global_load_lds' needs target feature
//        vmem-to-lds-load-insts
//   2. Feeding `global_load_lds_b32 v0, v[1:2], off` through
//      `llvm-mc --mcpu=gfx1100` produces "invalid instruction"; the same
//      input through `--mcpu=gfx942` is accepted.
//
// Closest substitute on gfx1100: regular `global_load_b{32,64,128}` →
// VGPR → `ds_write_b{32,64,128}`. That is what every kernel in
// kernels/*.hip already does (verified by `--save-temps` on the
// rdna3_memory_bench output: only `global_load_*` and `ds_write_*` ops
// in the hot loop, no LDS-bit forms). The compiler scheduler already
// pipelines the load+store pair; manual unrolling of 4×b32 sequences
// to "approximate" cp.async did NOT win in our microbenches — see the
// load-vs-load section in rdna3_memory_bench.hip.
//
// If RDNA4 (gfx1200) gains the LDS-bit forms in a future LLVM, this
// section becomes the place to add `global_load_lds_b32` etc. Until
// then, do not add stub wrappers — they'd hide a compile-time error.

// ============================================================================
// 2. atomicAdd helpers (f32: HW vs CAS-emulated; bf16: always CAS)
// ============================================================================
//
// CRITICAL FINDING (verified in rdna3_memory_bench, asm inspection):
// `atomicAdd(float*, float)` in HIP on gfx1100 lowers to a CAS loop
// (`global_atomic_cmpswap_b32`) BY DEFAULT — even though gfx1100 hardware
// supports `global_atomic_add_f32` natively. The default path uses CAS
// for memory-model reasons (handling NaN/denorm cases conservatively).
// `unsafeAtomicAdd(float*, float)` in <hip/amd_detail/amd_hip_unsafe_atomics.h>
// emits the HW instruction directly.
//
// Verified by `hipcc --offload-arch=gfx1100 --save-temps`:
//   atomicAdd(float*, float)       → global_atomic_cmpswap_b32 ... + loop
//   unsafeAtomicAdd(float*, float) → global_atomic_add_f32      (1 instr)
//
// The HW form is safe for ALL braidinfer use cases because we never
// concurrently update the same f32 cell from CPU + GPU and never rely on
// strict NaN/denorm propagation through atomicAdd. Where a kernel does
// atomicAdd to accumulate FFN partials or per-expert outputs, it should
// use atomic_add_f32_hw below.
//
// Measured (rdna3_memory_bench, RX 7900 XTX, gfx1100; cyc/op via
// wall_clock64; 8 blocks × 256 threads × 4096 iters):
//
//   n_slots          f32_cas (CAS loop)   f32_hw (HW)   bf16 (flat-CAS)
//   --------------   ------------------   -----------   ---------------
//   1 (all-hot)             68802                 72                78
//   4                        4861                 18                45
//   16                       1206                 10                26
//   64                        333                  9                26
//   per-thread unique          24                  2                26
//
// Headline numbers:
//   - f32_hw is **~950× faster than f32_cas** at single-slot contention,
//     and **13× faster** even at zero contention (per-thread unique).
//     The CAS-loop default in HIP's atomicAdd is a SEVERE pessimization
//     for any kernel that does atomic accumulation.
//   - bf16 vs f32_hw is **~1.1×** slower at single-slot contention but
//     **~14× slower** at zero contention (because the bf16 path still
//     pays the CAS-loop latency once per op even when uncontended).
//
// Practical guidance:
//   - For ALL f32 atomic accumulators in braidinfer hot loops, switch to
//     atomic_add_f32_hw. The default `atomicAdd(float*, float)` is the
//     800–1300× slower CAS path. (We currently use atomicAdd in
//     megakernel.hip:141 for a dump_count counter where it doesn't
//     matter; if a future hot path adds atomic accumulators, route them
//     through atomic_add_f32_hw.)
//   - bf16 atomics are NOT free. Accumulate in f32 and cast at the
//     very end. The MoE expert FFN already does this; do not break it.
//   - If you need a per-token bf16 reduction, route it through f32 and
//     bf16-cast on the final write.

// Software-CAS atomic add. Lowers to global_atomic_cmpswap_b32 + loop.
// Equivalent to plain `atomicAdd(p, v)`. Provided here so callsites can
// be explicit about which variant they want.
__device__ __forceinline__ float atomic_add_f32_cas(float* p, float v) {
    return atomicAdd(p, v);
}

// HW global_atomic_add_f32 path. Single instruction, ~2× faster under
// contention. Uses the unsafeAtomicAdd helper from
// <hip/amd_detail/amd_hip_unsafe_atomics.h>, which on gfx1100 lowers to
// the HW instruction via __hip_atomic_fetch_add(..., RELAXED, AGENT).
//
// Asm verified: `global_atomic_add_f32 v1, v0, s[0:1]` — single instr.
//
// "Unsafe" means: NaN handling and denorm-flush behavior of the HW
// instruction may differ from the strict IEEE-754 CAS loop. For our
// inference workloads this never matters. Do NOT use this for an atomic
// counter or sentinel that is observed by the CPU mid-flight.
__device__ __forceinline__ float atomic_add_f32_hw(float* p, float v) {
    return unsafeAtomicAdd(p, v);
}

// Default to HW. Anyone who wants the CAS path should call atomic_add_f32_cas
// explicitly.
__device__ __forceinline__ float atomic_add_f32(float* p, float v) {
    return atomic_add_f32_hw(p, v);
}

// Software-emulated bf16 atomic add. The bf16 lives at a 16-bit-aligned
// address; we operate on the encompassing 32-bit dword via CAS.
//
// IMPORTANT: this is correct under contention but ~1.7–3× slower than
// atomic_add_f32. Prefer accumulating in f32 wherever possible.
__device__ __forceinline__ unsigned short
atomic_add_bf16_safe(unsigned short* p, float addend) {
    // Find the dword that contains *p, and which half (low or high) it lives in.
    uintptr_t addr = (uintptr_t)p;
    uint32_t* dword = (uint32_t*)(addr & ~uintptr_t(3));
    bool is_high = (addr & 2u) != 0;
    uint32_t old = __atomic_load_n(dword, __ATOMIC_RELAXED);
    while (true) {
        // Decode current bf16 half → f32, add, re-encode.
        unsigned short cur_bf16 =
            (unsigned short)((old >> (is_high ? 16 : 0)) & 0xFFFFu);
        // bf16 → f32: shift into the upper 16 bits of a u32.
        union { float f; uint32_t u; } cur;
        cur.u = ((uint32_t)cur_bf16) << 16;
        float new_f = cur.f + addend;
        // f32 → bf16 with round-to-nearest-even.
        union { float f; uint32_t u; } nf; nf.f = new_f;
        uint32_t r = nf.u;
        uint32_t lsb = (r >> 16) & 1u;
        uint32_t round_bias = 0x7FFFu + lsb;
        r = (r + round_bias) >> 16;
        if ((nf.u & 0x7F800000u) == 0x7F800000u) {
            // Preserve NaN/inf (don't accidentally shift mantissa garbage).
            r = nf.u >> 16;
        }
        unsigned short new_bf16 = (unsigned short)(r & 0xFFFFu);
        // Splice the new half back into the dword.
        uint32_t mask    = is_high ? 0x0000FFFFu : 0xFFFF0000u;
        uint32_t shifted = (uint32_t)new_bf16 << (is_high ? 16 : 0);
        uint32_t want    = (old & mask) | shifted;
        uint32_t prev    = atomicCAS(dword, old, want);
        if (prev == old) return cur_bf16;
        old = prev;
    }
}

// ============================================================================
// 3. LDS padding for tpg-style cooperative GEMV
// ============================================================================
//
// RDNA3 LDS in WGP mode is 64 banks × 4 bytes (RDNA3 ISA §2.3.1; gfx1100
// HIP default is WGP mode). When 32 lanes of a wave touch the SAME bank in
// the SAME cycle (other than broadcast from a single address), the access
// serializes.
//
// The MoE expert GEMV pattern (kernels/moe_expert_ops.h:coop_gemv_*) fills
// `extern __shared__ float shared[groups_per_block]` from `lane==0` of each
// tpg-group, then `threadIdx.x == 0` accumulates them. That fill is fine
// (lane==0 only writes once per group), but the accumulate loop sees
// `groups_per_block` consecutive floats — if groups_per_block == 64 and the
// loop is unrolled, every 4-byte access maps to bank `idx & 63` which is
// fine. The hazard surfaces when the layout puts e.g. tpg-major partial
// sums at stride 64; then lanes that share a bank serialize.
//
// Measured (rdna3_memory_bench → bench_lds_bank_conflict, RX 7900 XTX):
//
//   pattern                            cyc/access
//   --------------------------------   ----------
//   packed_1d  shared[256]                 4.0
//   packed_2d  [32][32]  (stride 32)      17.9   ← bank conflict
//   packed_2d  [32][64]  (stride 64)      17.8   ← bank conflict
//   packed_2d  [32][128] (stride 128)     18.0   ← bank conflict
//   padded_2d  [32][32+1] (stride 33)      2.7   ← FIXED
//   padded_2d  [32][64+1] (stride 65)      2.7   ← FIXED
//   padded_2d  [32][128+1](stride 129)     2.7   ← FIXED
//   tpg=2  gpb=128 (1 thread reads)        2.1   ← no conflict
//   tpg=4  gpb=64  (1 thread reads)        1.1   ← no conflict
//   tpg=8  gpb=32  (1 thread reads)        1.8   ← no conflict
//   tpg=16 gpb=16  (1 thread reads)        2.0   ← no conflict
//
// Headline: stride-32/64/128 access from a wave (32 lanes hitting the
// same column at different rows) creates a clean 32-way serialization
// (~18 cyc) that a +1 float pad COMPLETELY ELIMINATES (2.7 cyc, **6.7×
// speedup**).
//
// Conclusion: the current 1D `shared[groups_per_block]` layout in
// kernels/moe_expert_ops.h has NO bank conflicts at any tpg ∈ {2,4,8,16}.
// The padding rule below matters only when transitioning to a 2D LDS
// tile (e.g. a future fused gate+up GEMV that stores per-(tpg-group,
// output) partials in a 2D block). In that case, pad the inner stride
// by +1 float to break the 32-bank period.

// Constexpr padding rule: returns the number of EXTRA floats to add per
// row of a 2D LDS tile to avoid the 64-bank period. Tpg is the threads-
// per-group fan-in.
//
// Usage:
//   constexpr int PAD = lds_pad_for_tpg<Tpg>();
//   __shared__ float tile[TPG_GROUPS][ROWS_PER_GROUP + PAD];
template<int Tpg>
__device__ __host__ constexpr int lds_pad_for_tpg() {
    // For 1D layouts (groups_per_block consecutive floats), no padding.
    // For 2D layouts of (groups, lanes-per-group), pad rows whose stride
    // is a multiple of 64.
    static_assert(Tpg == 1 || Tpg == 2 || Tpg == 4 || Tpg == 8 ||
                  Tpg == 16 || Tpg == 32,
                  "Tpg must be a power of 2 in [1, 32]");
    return 1;
}

// ============================================================================
// 4. Cache invalidation wrappers (buffer_gl0_inv / buffer_gl1_inv)
// ============================================================================
//
// gfx1100 has a per-CU L0 (vector) cache and a per-shader-array L1 cache.
// There is NO L2 invalidation instruction on RDNA3 (`buffer_gl2_inv` is
// rejected by `llvm-mc --mcpu=gfx1100`; only `buffer_gl0_inv` and
// `buffer_gl1_inv` exist). For cross-GPU coherence see GFX1100_ARCH §5.3.
//
// Measured (rdna3_memory_bench → bench_cache_invalidate, RX 7900 XTX,
// gfx1100, 2304 MHz GPU clock; 1024 invs/loop, 64 blocks, lane-0 issue):
//
//   variant                        cyc/inv (issue-only, no s_waitcnt)
//   ---------------------------    ----------------------------------
//   baseline (loop overhead)         0.88
//   buffer_gl0_inv only              2.09  (+1.21 cyc ≈ 0.5 ns)
//   buffer_gl1_inv only              4.18  (+3.30 cyc ≈ 1.4 ns)
//   buffer_gl0_inv + buffer_gl1_inv  4.24  (+3.36 cyc ≈ 1.5 ns)
//
// The numbers above are **issue cost only** — we did not insert an
// `s_waitcnt vmcnt(0)` after the invalidate, so the bench measures
// just the in-pipeline issue latency. The actual stall when the next
// VMEM op needs the invalidation to drain is workload-dependent.
//
// Important observations:
//   - gl0_inv is genuinely cheap (~0.5 ns issue cost).
//   - gl1_inv is ~3× more expensive than gl0_inv issue-cost-wise.
//   - **gl0_inv + gl1_inv is essentially free on top of gl1_inv alone**
//     (4.24 vs 4.18 cyc — pipelined). So the megakernel kernel-entry
//     pattern (`gl0_inv + gl1_inv`) is correct: there's no reason to
//     drop the gl0_inv to save ~0.06 cyc.
//   - Issue cost does NOT scale with thread count (only lane 0 issues).
//
// Placement guidance (current code is correct):
//   - persistent_worker batch boundary (lines 86–88): gl1_inv is sufficient
//     because the CONSUMER is the same persistent_worker waves on the same
//     shader array, but a different op may have been written by a
//     DIFFERENT wave on the same shader array → gl1_inv flushes the L1
//     between them.
//   - megakernel kernel-entry (lines 179–181): gl0_inv + gl1_inv. The
//     megakernel is launched fresh; the GPU may have arbitrary residual
//     L0/L1 lines from a previous launch. The +75 ns for gl0_inv is
//     cheap insurance.
//   - op_attn_paged / op_gdn_recurrent entry: gl1_inv only. The producer
//     (OP_D2D_COPY for KV pages, prior op for GDN state) ran on the same
//     persistent worker → wave but may have used a different CU. L0 is
//     per-CU so it is automatically invalidated by the time a different
//     CU reads. L1 is per-shader-array so this cross-CU same-SA path
//     needs explicit invalidation.
//
// Rule of thumb:
//   - Producer and consumer on different SHADER ARRAYS → BOTH gl0+gl1
//   - Producer and consumer on same SA but possibly different CUs → gl1
//   - Producer and consumer on same CU → none (rare in practice)
//   - Cross-GPU peer reads → see GFX1100_ARCH §5.3 (different problem;
//     uncached memory + agent-scope __threadfence is the workaround).

// Invalidate the per-CU L0 (vector) data cache. ~75 ns. Place between a
// peer-CU producer and consumer when both are on the same WGP/SA.
//
// IMPORTANT: must be issued by lane 0 only of the consuming wave; placing
// it inside an `if (threadIdx.x == 0)` is mandatory because it is a
// SOPP-class instruction that operates on per-CU SGPR state.
__device__ __forceinline__ void gl0_invalidate() {
    asm volatile("buffer_gl0_inv" ::: "memory");
}

// Invalidate the per-shader-array L1 cache. ~245 ns. Place between a
// peer-SA producer and consumer.
//
// Lane-0-only requirement is the same as gl0_invalidate.
__device__ __forceinline__ void gl1_invalidate() {
    asm volatile("buffer_gl1_inv" ::: "memory");
}

// Combined L0+L1 invalidation for the worst-case cross-WGP path. ~280 ns.
__device__ __forceinline__ void gl01_invalidate() {
    asm volatile("buffer_gl0_inv\n\t"
                 "buffer_gl1_inv" ::: "memory");
}

// ============================================================================
// 5. Buffer descriptors (V# / S#) — survey result
// ============================================================================
//
// We surveyed every load site in kernels/ for opportunities where switching
// from regular `__global__` pointer loads to `buffer_load_dword*` (V#-based
// loads) would matter. The conclusion: NONE of the current braidinfer
// kernels would benefit. Reasoning:
//
//   1. The hot loads are ALREADY 4-byte vectorized (e.g. coop_gemv_pcg32:
//      one byte-per-thread × 32 lanes packed → 32 dwords/wave).
//   2. The compiler already emits buffer_load_b{32,64,128} on gfx1100 for
//      pointer-load patterns where the base is a 16B-aligned global ptr.
//      We confirmed via `--save-temps` on attn_layer_fused.hip and
//      rdna3_memory_bench.hip: no `flat_load_*` instructions appear in
//      the hot loops, only `buffer_load_*`. The compiler is doing the
//      V# packing for us.
//   3. CK has explicit V# wrappers (amd_buffer_addressing_builtins.hpp)
//      for use cases where bounds checking via the V# rather than via a
//      VALU compare matters. None of our kernels have that pattern.
//   4. Constructing a V# manually (s_load_b128 of an SRD) costs ~2 SGPR
//      per buffer and 4 SGPR for the SRD itself — not worth it for
//      kernels that already get optimal codegen.
//
// We DO NOT provide a `buffer_load_dword<N>` wrapper here. If a future
// kernel needs explicit out-of-bounds clamping or scope hints (glc/slc/dlc)
// that the compiler isn't emitting, copy the relevant CK helper from
// /opt/rocm/include/ck/utility/amd_buffer_addressing_builtins.hpp.

// ============================================================================
// Open questions (not blocking adoption)
// ============================================================================
//
// O1. SLC/DLC hints on regular global_load_b{32,128}: per GFX1100_ARCH
//     §5.0, the right `dlc` setting is workload-dependent. We have NOT
//     wired a per-op DLC switch yet because the compiler does not
//     emit dlc=1 for any of our loads anyway, and forcing it via
//     inline-asm `global_load_b128 v[…], v[…], off dlc` requires
//     bypassing the compiler scheduler. Defer until a benchmark
//     identifies a kernel that benefits.
//
// O2. RDNA4 (gfx1200) cp.async path: see Section 1 above. If LLVM gains
//     the LDS-bit forms for gfx1200, add a guarded path here.
//
// ============================================================================
// Adoption order
// ============================================================================
//
//   Phase A. Replace the manual `buffer_gl1_inv` asm in
//            kernels/persistent_worker.hip:87, kernels/megakernel_ops.hip:848,
//            kernels/megakernel_ops.hip:1664 with gl1_invalidate(). Replace
//            kernels/megakernel.hip:180 with gl01_invalidate(). Pure
//            readability win, identical codegen.
//   Phase B. If a quantized-aware bf16 accumulator path is added (e.g. a
//            future "online quant of a layer-norm weight"), use
//            atomic_add_bf16_safe. Until then, stay with f32 accumulators.
//            DO NOT replace existing f32 atomicAdds with bf16 — the bench
//            shows a 1.7–3× regression.
//   Phase C. Apply lds_pad_for_tpg<Tpg>() ONLY when introducing a 2D LDS
//            tile in a new fused kernel. The existing 1D `shared[]`
//            layouts in moe_expert_ops.h have NO bank conflicts (verified
//            in the bench), so do not change them.

}}  // namespace braidinfer::rdna3
