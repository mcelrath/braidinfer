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
    volatile uint32_t exited;            // kernel writes 1 on clean exit (host pauses no-progress timer)
    volatile uint64_t progress_counter;  // kernel increments at progress points
    volatile uint32_t last_op_id;        // telemetry: which op was running at last beat
    volatile uint32_t _pad;              // alignment padding
    volatile uint64_t last_beat_us;      // telemetry: gpu_clock() at last beat (not wall clock)
};

// Device-global broadcast flag. Thread 0/block 0 writes the exit decision here;
// all blocks read it after grid.sync(). Must be __device__ (not __shared__) so it
// is visible across all blocks in the cooperative grid.
//
// NOTE: Each .hip file is compiled to an independent .hsaco (device code object)
// and loaded separately at runtime — they are not linked together. Each kernel
// that includes this header gets its own private copy of the flag, which is
// correct: each cooperative kernel runs independently and needs its own broadcast.
// The conventional "extern + separate .hip" pattern applies to statically-linked
// TUs; it does NOT apply here and would break device symbol resolution at runtime.
__device__ bool __watchdog_should_exit;

// COOPERATIVE EXIT GRANULARITY: watchdog_poll_and_check is called ONLY between
// top-level instructions (opcode dispatch boundaries), NOT inside compute-heavy ops
// (op_moe_ffn, op_linear_proj_*, op_attn_paged, etc.). A wedge inside a compute
// op escalates directly to process abort (via the host watchdog thread's grace
// period expiry), bypassing cooperative exit.
//
// This is acceptable for current ops since each compute op is bounded:
//   - op_linear_proj_*:  O(d²) GEMV, completes in milliseconds
//   - op_attn_paged:     O(n²) per token, bounded by seq_len
//   - op_moe_ffn:        bounded by num_active_experts × expert_size
//
// If any future op can run longer than the watchdog's no-progress timeout
// (default 2s), add intra-op watchdog_beat() calls at safe checkpoints.
// A full watchdog_poll_and_check() inside a compute op requires all blocks to
// reach the call simultaneously (grid.sync() precondition), which may not be
// achievable mid-op without significant restructuring.
//
// Poll host-mapped force_exit and broadcast decision to all blocks.
//
// Call sites: outer poll loop, between major work phases (top-level instructions only).
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

// Initialize local counter from the device-visible progress_counter.
// CRITICAL for kernels that re-launch on the same WatchdogState (e.g. megakernel
// per execute()): without this, every launch starts local_counter at 0 and
// stores 100, 200, 300, ... — identical sequence each launch — so the host
// watchdog sees the counter "stuck" at the same value across launches and
// spuriously aborts. By seeding from the device-visible value, each launch
// continues the monotonic sequence.
//
// Long-lived persistent kernels (persistent_worker, moe_worker) call this once
// at start and the value will be 0 — same as before. Only matters for kernels
// that launch repeatedly against the same state.
__device__ __forceinline__ void watchdog_init(WatchdogState* ws, uint32_t* local_counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && ws) {
        // gfx1100 hazard: __hip_atomic_load(SYSTEM scope) hangs. Use volatile
        // read instead — the WatchdogState page is host-mapped MTYPE=UC, so
        // the read goes directly to system memory bypassing GPU L2.
        *local_counter = *(volatile uint64_t*)&ws->progress_counter;
        // Clear the exited flag — kernel is now running again. SYSTEM-scope
        // store works on gfx1100 (per watchdog.h comment); use it for explicit
        // release ordering.
        __hip_atomic_store(&ws->exited, 0u,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// Mark the kernel as cleanly exited so the host watchdog pauses its
// no-progress timer until the next launch (which calls watchdog_init).
// Without this, the host keeps polling a frozen progress_counter and
// spuriously fires force_exit during the gap between launches (most
// commonly: per-segment prefill where the host runs MoE dispatch +
// next-segment compile between mk.execute() calls).
__device__ __forceinline__ void watchdog_signal_exited(WatchdogState* ws) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && ws) {
        __hip_atomic_store(&ws->exited, 1u,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
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
        __threadfence();
    }
}
