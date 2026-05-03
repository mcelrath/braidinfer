#pragma once
// Shared watchdog primitive for all persistent cooperative kernels.
//
// Allocated in a single pinned-host page (hipHostMallocMapped | hipHostMallocCoherent).
// Host writes force_exit=1 to request emergency exit; kernel polls every N iterations.
//
// CORRECTNESS: force_exit is broadcast via a __device__ global flag shared across all
// blocks before grid.sync(). A per-block __shared__ flag would NOT propagate across
// blocks — each block has its own shared memory. Using __device__ global ensures all
// blocks see the same value after the broadcast block writes it.
//
// gfx1100 HAZARD: __hip_atomic_load(SYSTEM scope) hangs on gfx1100 (see docs/P2P.md).
// Use volatile reads for polling host-mapped memory. __hip_atomic_store(SYSTEM) works.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

struct WatchdogState {
    volatile uint32_t force_exit;        // host writes 1 to request emergency exit
    volatile uint64_t progress_counter;  // kernel increments at progress points
    volatile uint32_t last_op_id;        // telemetry: which op was running at last beat
    volatile uint32_t _pad;              // alignment padding
    volatile uint64_t last_beat_us;      // telemetry: gpu_clock() at last beat (not wall clock)
};

// Device-global broadcast flag. Thread 0/block 0 writes the exit decision here;
// all blocks read it after grid.sync(). Must be __device__ (not __shared__) so it
// is visible across all blocks in the cooperative grid.
__device__ bool __watchdog_should_exit;

// Poll host-mapped force_exit and broadcast decision to all blocks.
//
// Call sites: outer poll loop, between major work phases.
// Protocol:
//   1. Thread 0, block 0 reads force_exit via volatile load (avoids gfx1100 atomic_load hang).
//   2. Writes decision to __watchdog_should_exit (__device__ global, visible to all blocks).
//   3. grid.sync() — all blocks wait here, so the decision is fully visible on return.
//   4. All blocks return the shared decision.
//
// If this returns true, the caller MUST return from the kernel immediately (all blocks
// return together, so no block is left waiting at the next grid.sync()).
__device__ __forceinline__ bool watchdog_poll_and_check(
    WatchdogState* ws,
    cooperative_groups::grid_group& grid,
    uint32_t op_id)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Volatile read: avoids __hip_atomic_load SYSTEM scope hang on gfx1100.
        uint32_t fe = *(volatile uint32_t*)&ws->force_exit;
        __watchdog_should_exit = (fe != 0);
        ws->last_op_id = op_id;
    }
    grid.sync();
    return __watchdog_should_exit;
}

// Progress beat: increment counter every K iterations to signal liveness to host watchdog.
// K=100 at s_sleep(1) ≈ 1µs/iteration → one SYSTEM-scope store per 100µs.
// Host thread declares no-progress if counter unchanged for WATCHDOG_NO_PROGRESS_MS (default 2s).
__device__ __forceinline__ void watchdog_beat(WatchdogState* ws, uint32_t* local_counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t c = ++(*local_counter);
        if ((c % 100) == 0) {
            __hip_atomic_store(&ws->progress_counter, (uint64_t)c,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// Emit done_flag=1 and return.  Used at cooperative-exit point: all blocks must
// be calling this together (after watchdog_poll_and_check returned true and the caller
// completed any cleanup grid.sync() calls needed before return).
__device__ __forceinline__ void watchdog_signal_exit(volatile uint32_t* done_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (done_flag) *done_flag = 1u;
        __threadfence_system();
    }
}
