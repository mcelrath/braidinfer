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

__device__ __forceinline__ void atomic_block_barrier(GridBarrierState* state) {
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
                __builtin_amdgcn_s_sleep(0);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}

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

// fast_grid_sync — preferred call site spelling.
//
// If `state` is non-null, uses atomic_block_barrier (faster on gfx1100 per
// rdna3_sync_bench). If null, falls back to cooperative_groups grid.sync().
// This lets callers stage the migration from cg grid.sync without code-wide
// surgery.
__device__ __forceinline__ void fast_grid_sync(
    cooperative_groups::grid_group& grid,
    GridBarrierState* state)
{
    if (state) {
        atomic_block_barrier(state);
    } else {
        grid.sync();
    }
}

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
