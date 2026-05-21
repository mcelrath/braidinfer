// SPDX-License-Identifier: MIT
// rdna3_barrier.h — RDNA3 (gfx1100, wave32) synchronization primitives.
// Migrated from kernels/rdna3_sync.h (2026-05-13).
//
// CANONICAL HAZARD (composable_kernel/GFX1100_ARCH.md §11.4):
//   atomic_block_barrier (default) is the production primitive. The V2/ASM/V4
//   variants and BRAIDINFER_USE_GRID_SYNC produce wrong or wedge-prone output
//   on multi-GPU MoE workloads (exterior_algebra-zuk Phase 2/2'/2''
//   2026-05-12; persistent_skeleton_repro V7 3/10 wedge 2026-05-13). They are
//   preserved as durable A/B scaffolding for future investigators but require
//   RDNA3_I_KNOW_WHAT_IM_DOING to enable, preventing accidental reintroduction.
//
// To enable any variant: define BOTH the variant macro AND
// RDNA3_I_KNOW_WHAT_IM_DOING.
//
// RDNA3 (gfx1100, wave32) synchronization primitives.
//
// Provides scope-explicit fence wrappers and faster grid-wide barrier
// alternatives to cooperative_groups::grid_group::sync(). The default
// cg grid.sync() invokes a system-coverage release/acquire pattern that is
// stronger than what most megakernel boundaries actually require.
//
// All fences here lower to `__builtin_amdgcn_fence(order, scope)` with
// scope strings as they appear in the LLVM AMDGPU backend:
//
//   ""           -> system / cross-agent (gfx1100: HANGS for live spin-waits;
//                   safe at kernel-end via UC memory pattern, see GFX1100_ARCH.md §5.3)
//   "agent"      -> device-wide L2-coherent fence (this is what __threadfence() emits)
//   "workgroup"  -> CU-local fence (this is what __threadfence_block() emits)
//   "wavefront"  -> single-wave fence (lightest; rare practical use)
//
// Microbenchmark numbers in comments are median in-kernel cycles measured
// on RX 7900 XTX / gfx1100 via kernels/diagnostic/rdna3_sync_bench. Run that
// binary to refresh numbers when toolchain or driver changes.
//
// Header summary:
//   fence_block()                 — workgroup-scope acq_rel fence
//   fence_device()                — agent-scope acq_rel fence (== __threadfence)
//   fence_system_uc()             — system-scope acq_rel fence (caller MUST verify
//                                   either UC memory or pre-kernel-end placement;
//                                   bare system fence on live spin-wait WILL HANG)
//   fence_release_block/device()  — release-only variants
//   fence_acquire_block/device()  — acquire-only variants
//
//   barrier_within_wave()         — alias for the wave-implicit lock-step (no insn)
//   barrier_workgroup()           — fast __syncthreads with cleaner intent name
//
//   atomic_block_barrier(ctr)     — hand-rolled grid barrier via atomicAdd on a
//                                   uint counter. ~1.5x faster than grid.sync()
//                                   on gfx1100 because it skips the full
//                                   release/acquire pair grid.sync() emits.
//   fast_grid_sync(grid, ctr)     — drop-in safer variant: cooperative_groups
//                                   grid.sync() if no counter provided, else the
//                                   atomic_block_barrier path. Same semantics.
//
// IMPORTANT — when atomic_block_barrier is NOT a drop-in for grid.sync():
//   * It is a CTA-arrival barrier, not a stream-of-writes fence. After it
//     returns, all blocks are past the barrier point but their PRIOR memory
//     writes are only guaranteed visible at agent scope (via the embedded
//     __threadfence). For consumers in the SAME persistent kernel reading
//     L2-cached data, this is sufficient; for cross-GPU peer reads use the
//     UC-memory pattern from GFX1100_ARCH.md §5.3 instead.
//   * It does NOT serve as a system-coverage fence. cg::grid_group::sync()
//     also does not — that is a misconception we want to dispel.
//   * It requires the counter to be a single 4-byte uint, INITIALIZED to 0,
//     and reset between barriers (the helper handles reset by detecting
//     wraparound to gridDim.x).
//
// Compile/usage notes:
//   - Wave32 only (HIP default for gfx1100). Don't mix with -mwavefrontsize64.
//   - Counter for atomic_block_barrier must be in cache-coherent VRAM.
//     Allocate via plain hipMalloc; do NOT use hipDeviceMallocUncached —
//     UC writes are very slow and would defeat the speedup.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace braidinfer { namespace rdna3 {

// ---------------------------------------------------------------------------
// Scope-explicit memory fences
// ---------------------------------------------------------------------------
// __builtin_amdgcn_fence(order, scope):
//   order: __ATOMIC_ACQUIRE=2, __ATOMIC_RELEASE=3, __ATOMIC_ACQ_REL=4,
//          __ATOMIC_SEQ_CST=5
//   scope: "" (system), "agent" (device), "workgroup" (CTA), "wavefront"

__device__ __forceinline__ void fence_block() {
    // Same as __threadfence_block() — included here for symmetry / explicit
    // intent at call sites (so reviewers don't have to recall HIP defaults).
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
}

__device__ __forceinline__ void fence_device() {
    // Same as __threadfence(). Drains this thread's stores to L2 such that
    // any other thread on this GPU performing an acquire load sees them.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
}

__device__ __forceinline__ void fence_system_uc() {
    // System-scope fence. ON gfx1100 THIS HANGS IF USED AS THE SOLE COHERENCE
    // MECHANISM FOR A LIVE SPIN-WAIT. The ISA has no L2 sys-invalidate
    // instruction (`buffer_invl2`/`buffer_gl2_inv` rejected by llvm-mc) so the
    // fence cannot actually push L2 across a peer link. Per GFX1100_ARCH.md
    // §5.3 the working pattern is `fence_device()` + UC-mapped target memory,
    // which is what callers should use. This wrapper exists so that any code
    // that thinks it needs system fence is forced to acknowledge the gfx1100
    // hazard at the call site (rename to fence_system_uc to flag the
    // requirement to use UC memory).
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
}

__device__ __forceinline__ void fence_release_block() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
}
__device__ __forceinline__ void fence_release_device() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}
__device__ __forceinline__ void fence_acquire_block() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
__device__ __forceinline__ void fence_acquire_device() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// ---------------------------------------------------------------------------
// Within-block barriers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void barrier_workgroup() {
    // Identical to __syncthreads(). Provided for symmetry.
    __syncthreads();
}

__device__ __forceinline__ void barrier_within_wave() {
    // Wave32 lanes execute in lockstep; no barrier instruction needed.
    // Wrapped here because some patterns (e.g., DPP after a divergent if)
    // require a wavefront fence to settle EXEC mask interactions.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    __builtin_amdgcn_wave_barrier();
}

// ---------------------------------------------------------------------------
// Grid-wide ("device-wide") barrier
// ---------------------------------------------------------------------------
//
// atomic_block_barrier:
//   Hand-rolled barrier across all blocks of a cooperative launch. Replaces
//   cooperative_groups::grid_group::sync() in cases where the only requirement
//   is "all blocks have arrived; their prior stores are visible at agent
//   scope". This avoids the system-scope coverage cg::grid_group::sync()
//   includes.
//
// Protocol (last-arriver pattern, no spin-wait by other blocks):
//
//   __syncthreads();              // gather all threads in this block
//   if (threadIdx.x == 0)         // one increment per block
//       __threadfence();
//       prev = atomicAdd(ctr, 1)
//       if (prev + 1 == gridDim.x):
//           atomicStore(ctr, 0)   // reset for next barrier (no extra alloc)
//           atomicStore(release, generation+1)  // last block signals
//       else:
//           spin on (atomic_load(release) == generation+1)
//   __syncthreads();              // broadcast to all threads in block
//
// The pattern below uses TWO 4-byte slots: `arrived_ctr` (counts arrivals,
// reset to 0 after release) and `gen` (monotonically increases each barrier
// pass). This avoids ABA on the counter alone.
//
// Caller responsibility:
//   - Allocate `state` via hipMalloc, sizeof(GridBarrierState) = 8B, init to 0.
//   - One state per "barrier site" if you want barriers to overlap pipelined
//     phases; otherwise reuse one site.
//   - This API REQUIRES a cooperative launch (gridDim.x ≤ max cooperative
//     blocks for the kernel). Same launch constraint as cg::grid_group::sync.

struct GridBarrierState {
    unsigned int arrived;
    unsigned int generation;
};

#ifdef BRAIDINFER_USE_GRID_SYNC
#  ifndef RDNA3_I_KNOW_WHAT_IM_DOING
#    error "BRAIDINFER_USE_GRID_SYNC: cg::grid_group::sync produces incorrect output on multi-GPU MoE decode (zuk Phase 2 2026-05-12). Define RDNA3_I_KNOW_WHAT_IM_DOING to enable for A/B testing."
#  endif
#endif

__device__ __forceinline__ void atomic_block_barrier(GridBarrierState* state) {
#ifdef BRAIDINFER_USE_GRID_SYNC
    // braidinfer-pky.2 Phase 0b diagnostic (2026-05-12): swap to
    // cooperative_groups::grid_group::sync to test whether the wedge is
    // specific to atomic_block_barrier's atomic-add+spin protocol or a
    // broader RDNA3 multi-GPU synchronization issue. Per kb
    // rdna3-grid-sync-vs-atomic-block-barrier-gfx1100, cg::grid.sync is
    // ~115-155x slower per call (~11k cyc at 192 blocks vs ~96 cyc) — but
    // if it sidesteps the wedge, the +2% end-to-end cost is acceptable on
    // multi-GPU MoE decode. Build with BRAIDINFER_USE_GRID_SYNC=1.
    //
    // The function still takes `state` to keep call sites unchanged; the
    // pointer is unused. cooperative_groups::this_grid().sync() must run
    // on a cooperative launch, which persistent_worker is.
    (void)state;
    cooperative_groups::this_grid().sync();
#else
    __syncthreads();
    __shared__ unsigned int s_target_gen;

    if (threadIdx.x == 0) {
        // Drain this block's stores to L2 BEFORE counter increment, so the
        // last arriver's `generation` flip implies all blocks' data is
        // visible at agent scope (i.e., readable by any other block).
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        unsigned int target_gen =
            __hip_atomic_load(&state->generation, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) + 1u;
        s_target_gen = target_gen;
        unsigned int prev =
            __hip_atomic_fetch_add(&state->arrived, 1u, __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == (unsigned int)gridDim.x) {
            // Last arriver: reset counter and bump generation atomically
            // enough — no other block will be incrementing because they are
            // all waiting on `generation`.
            __hip_atomic_store(&state->arrived, 0u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&state->generation, target_gen,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            // Non-last block: spin on generation.
            for (;;) {
                unsigned int g =
                    __hip_atomic_load(&state->generation, __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_AGENT);
                if (g == target_gen) break;
                // s_sleep(1) not (0): per ea KB root-cause-zuk-phase-2-2ab,
                // s_sleep(0) starves MES of cycles to schedule REMOVE_QUEUE
                // messages under 4-GPU concurrent cooperative kernels →
                // MES deadlock → MODE1 reset cascade (observed 2026-05-16
                // at t≈55000s, all 4 MCIO cards wedged simultaneously).
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
#endif
}

// Phase-2 variant: identical to atomic_block_barrier but replaces the
// __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent") release fence with
// inline GCN that emits only s_waitcnt lgkmcnt(0)+vmcnt(0) + s_barrier,
// deliberately OMITTING s_waitcnt_vscnt null,0x0.
//
// Motivation (exterior_algebra-zuk Phase 2, 2026-05-12):
//   Phase 1 disassembly confirmed that the compiler inserts
//   `s_waitcnt_vscnt null, 0x0` immediately before every `s_barrier` in
//   the inlined atomic_block_barrier body.  That instruction drains the
//   SYSTEM-scope vector-store completion counter (tracks stores issued with
//   SYSTEM scope across PCIe).  Under 4-GPU PCIe pressure the counter does
//   not drain promptly, blocking the wave indefinitely — the wedge.
//
// Trade-off: in-flight SYSTEM-scope stores are not guaranteed visible to
// peer GPUs at barrier exit.  Callers that read cross-GPU data must ensure
// the producer side has completed its UC/peer write before the consumer
// issues its load (the 4fg.5 deferred-read pattern).  For the current MoE
// decode megakernel, peer reads follow a separate signalling flag (not this
// barrier), so the relaxed guarantee is safe.
//
// Enable via -DBRAIDINFER_BARRIER_V2 (see macro below, or build.rs).
__device__ __forceinline__ void atomic_block_barrier_v2(GridBarrierState* state) {
    __syncthreads();
    __shared__ unsigned int s_target_gen;

    if (threadIdx.x == 0) {
        // Drain this block's stores visible at agent scope WITHOUT draining
        // the SYSTEM-scope vscnt counter.  s_waitcnt lgkmcnt(0) ensures LDS
        // and scalar writes are complete; vmcnt(0) ensures vector memory
        // (global_load/store) writes are visible within the agent (device).
        // The s_barrier that follows serialises the wave with other waves in
        // the CU — keeping the subsequent atomicAdd from racing with still-
        // in-flight intra-device stores — without stalling on inter-GPU PCIe.
        asm volatile(
            "s_waitcnt lgkmcnt(0) vmcnt(0)\n\t"
            "s_barrier\n\t"
            "buffer_gl0_inv\n\t"
            ::: "memory");

        unsigned int target_gen =
            __hip_atomic_load(&state->generation, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) + 1u;
        s_target_gen = target_gen;
        unsigned int prev =
            __hip_atomic_fetch_add(&state->arrived, 1u, __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == (unsigned int)gridDim.x) {
            // Last arriver: reset counter, then release-store generation.
            __hip_atomic_store(&state->arrived, 0u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&state->generation, target_gen,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            // Non-last block: spin on generation.
            for (;;) {
                unsigned int g =
                    __hip_atomic_load(&state->generation, __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_AGENT);
                if (g == target_gen) break;
                // s_sleep(1) not (0): per ea KB root-cause-zuk-phase-2-2ab,
                // s_sleep(0) starves MES of cycles to schedule REMOVE_QUEUE
                // messages under 4-GPU concurrent cooperative kernels →
                // MES deadlock → MODE1 reset cascade (observed 2026-05-16
                // at t≈55000s, all 4 MCIO cards wedged simultaneously).
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}

// Phase-2 compile-time A/B switch.  Pass -DBRAIDINFER_BARRIER_V2 to hipcc
// (or set BRAIDINFER_BARRIER_V2=1 when building with Cargo, which gates the
// define in build.rs) to route all atomic_block_barrier() call sites to
// atomic_block_barrier_v2() without touching any caller.  The macro MUST be
// placed after both function definitions so the function bodies compile with
// their canonical names (placing it before causes a redefinition error because
// the function-body declaration of atomic_block_barrier would itself expand to
// atomic_block_barrier_v2, colliding with the explicit definition above).
// Call sites in .hip files that include this header will see the redirect.
#ifdef BRAIDINFER_BARRIER_V2
#  ifndef RDNA3_I_KNOW_WHAT_IM_DOING
#    error "BRAIDINFER_BARRIER_V2: omits s_waitcnt_vscnt; not safe for cross-GPU consumers. Define RDNA3_I_KNOW_WHAT_IM_DOING to enable."
#  endif
#define atomic_block_barrier atomic_block_barrier_v2
#endif

// ASM-tightened grid barrier. Same protocol as atomic_block_barrier but
// avoids the C-level memory model (uses explicit s_waitcnt + global_atomic).
// Empirical (gfx1100): atomic_block_barrier already compiles to optimal
// global_atomic_add + s_waitcnt + s_sleep loop. The ASM version is here as a
// reference / fallback in case future toolchain regressions inflate the C
// path. Keep both so we can A/B at any time.
__device__ __forceinline__ void asm_block_barrier(GridBarrierState* state) {
    __syncthreads();
    __shared__ unsigned int s_target_gen;
    if (threadIdx.x == 0) {
        // Drain prior stores to L2 (agent scope = device-wide visibility).
        // Ensures non-last blocks see writer's data after acquiring generation.
        asm volatile("s_waitcnt vmcnt(0)\n"
                     "buffer_gl1_inv\n"
                     "s_waitcnt_vscnt null, 0x0\n"
                     ::: "memory");

        unsigned int target_gen;
        unsigned int prev;
        // Atomic read-modify of arrived; load generation.
        target_gen = state->generation + 1u;
        s_target_gen = target_gen;
        prev = atomicAdd(&state->arrived, 1u);

        if (prev + 1u == (unsigned int)gridDim.x) {
            __hip_atomic_store(&state->arrived, 0u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            // Release-store generation: one wait+inv before the store ensures
            // the reset of `arrived` is visible before non-last blocks
            // observe `generation == target_gen`.
            asm volatile("s_waitcnt vmcnt(0)\n"
                         "buffer_gl1_inv\n"
                         "s_waitcnt_vscnt null, 0x0\n"
                         ::: "memory");
            __hip_atomic_store(&state->generation, target_gen,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            // Spin-acquire on generation. s_sleep 0 nudges scheduler to give
            // other waves a turn; s_setprio 0 demotes this poller wave so it
            // doesn't starve the last arriver from running its release.
            asm volatile("s_setprio 0" ::: "memory");
            unsigned int g;
            do {
                asm volatile("s_sleep 0\n"
                             "buffer_gl1_inv\n"
                             "s_waitcnt vmcnt(0)\n" ::: "memory");
                g = __hip_atomic_load(&state->generation, __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_AGENT);
            } while (g != target_gen);
            asm volatile("s_setprio 1" ::: "memory");
        }
    }
    __syncthreads();
}

// Phase-3' compile-time A/B switch.  Pass -DBRAIDINFER_BARRIER_ASM to hipcc
// (or set BRAIDINFER_BARRIER_ASM=1 when building with Cargo, which gates the
// define in build.rs) to route all atomic_block_barrier() call sites to
// asm_block_barrier() without touching any caller.  The macro MUST be placed
// after asm_block_barrier's function body (above) so the function compiles
// with its canonical name before the redirect takes effect.
// Root cause this targets (exterior_algebra-zuk Phase 2', 2026-05-12):
//   atomic_block_barrier's spin loop does:
//     global_load -> s_waitcnt vmcnt(0) -> buffer_gl1_inv
//   i.e. INVALIDATE AFTER LOAD — each iteration reads stale GL1 (per-shader-
//   array L1) and never sees the updated state->generation.
//   asm_block_barrier's spin loop does:
//     buffer_gl1_inv -> s_waitcnt vmcnt(0) -> load
//   i.e. INVALIDATE BEFORE LOAD — forces a fresh L2 read every iteration.
#ifdef BRAIDINFER_BARRIER_ASM
#  ifndef RDNA3_I_KNOW_WHAT_IM_DOING
#    error "BRAIDINFER_BARRIER_ASM: experimental asm spin loop; superseded by V4 (omits s_sleep preemption). Define RDNA3_I_KNOW_WHAT_IM_DOING to enable."
#  endif
#define atomic_block_barrier asm_block_barrier
#endif

// Phase-4 variant: same as asm_block_barrier but the spin loop omits
// s_sleep 0. Reason: s_sleep on gfx1100 causes hardware preemption — when
// ALL blocks of a cooperative grid hit the same s_sleep in their spin path,
// the entire grid preempts simultaneously and no block advances
// state->generation. Replacing s_sleep with s_nop (or nothing) keeps the
// wave resident in the CU. s_setprio 0 (already present) prevents the
// spinning wave from monopolizing CU resources.
//
// Root cause (exterior_algebra-zuk Phase 2'', 2026-05-12): Phase 2'' agent
// disassembled persistent_worker.hsaco and found the wave parks at PC 0x1C88
// = `s_sleep 0` in atomic_block_barrier spin loop. s_sleep on gfx1100 causes
// HARDWARE PREEMPTION — saves wave registers to memory, marks CU idle. ALL
// blocks of cooperative grid preempt simultaneously at the same s_sleep; no
// last-arriver advances state->generation. Classic cooperative-grid +
// preemption deadlock. Both v1 and asm_block_barrier contain this point.
// Enable via -DBRAIDINFER_BARRIER_V4 (see macro below, or build.rs).
__device__ __forceinline__ void atomic_block_barrier_v4(GridBarrierState* state) {
    __syncthreads();
    __shared__ unsigned int s_target_gen;
    if (threadIdx.x == 0) {
        asm volatile("s_waitcnt vmcnt(0)\n"
                     "buffer_gl1_inv\n"
                     "s_waitcnt_vscnt null, 0x0\n"
                     ::: "memory");
        unsigned int target_gen = state->generation + 1u;
        s_target_gen = target_gen;
        unsigned int prev = atomicAdd(&state->arrived, 1u);
        if (prev + 1u == (unsigned int)gridDim.x) {
            __hip_atomic_store(&state->arrived, 0u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            asm volatile("s_waitcnt vmcnt(0)\n"
                         "buffer_gl1_inv\n"
                         "s_waitcnt_vscnt null, 0x0\n"
                         ::: "memory");
            __hip_atomic_store(&state->generation, target_gen,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            asm volatile("s_setprio 0" ::: "memory");
            unsigned int g;
            do {
                asm volatile("s_nop 0x7f\n"        // NOP delay, no preemption
                             "buffer_gl1_inv\n"
                             "s_waitcnt vmcnt(0)\n" ::: "memory");
                g = __hip_atomic_load(&state->generation, __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_AGENT);
            } while (g != target_gen);
            asm volatile("s_setprio 1" ::: "memory");
        }
    }
    __syncthreads();
}

#ifdef BRAIDINFER_BARRIER_V4
#  ifndef RDNA3_I_KNOW_WHAT_IM_DOING
#    error "BRAIDINFER_BARRIER_V4: experimental variant (omits s_sleep, INVALIDATE-BEFORE-LOAD). Default atomic_block_barrier is the production primitive. Define RDNA3_I_KNOW_WHAT_IM_DOING to enable."
#  endif
#define atomic_block_barrier atomic_block_barrier_v4
#endif

// ---------------------------------------------------------------------------
// Diagnostic helpers (not for production)
// ---------------------------------------------------------------------------

// Reset a GridBarrierState from device side BEFORE first use within a kernel.
// Idempotent if already zero. Use only from <<<1,1>>>-style preamble or
// from threadIdx.x == 0 && blockIdx.x == 0 with __syncthreads() afterward.
__device__ __forceinline__ void grid_barrier_reset(GridBarrierState* state) {
    state->arrived    = 0u;
    state->generation = 0u;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

}}  // namespace braidinfer::rdna3
