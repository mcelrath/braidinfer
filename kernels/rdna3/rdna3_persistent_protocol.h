// SPDX-License-Identifier: MIT
// rdna3_persistent_protocol.h — Canonical mailbox-poll protocol for the
// braidinfer persistent_worker family on gfx1100.
//
// Why this header exists:
//   The persistent worker's outer-loop body (poll → barrier → ack →
//   dispatch → loop) has subtle ordering invariants. Phase 2' deferred-
//   ack (commit 8cf8084) wrote ack=last_seq at the TOP of each outer iter
//   AFTER the iter's inner-poll completed. This produced a protocol
//   deadlock: host blocked on ack=N, worker's iter N+1 inner-poll blocked
//   on seq=N+1 which host wouldn't send before ack=N. Wedge fingerprint:
//   stuck_pc=PC_IN_POLL, ack=0, first dispatch never returns. The bug took
//   10+ hours to root-cause in 2026-05-14.
//
// Canonical contract (any persistent worker MUST satisfy):
//   1. ack=seq is written in the SAME iter as the dispatch that processed
//      seq, AFTER all dispatch_opcode calls + threadfence, BEFORE looping
//      back to the next inner-poll.
//   2. AGENT scope on the ack store. SYSTEM-scope emits s_waitcnt_vscnt
//      that wedges across the next iter's atomic_block_barrier under
//      multi-GPU PCIe pressure (§11.4).
//   3. Cross-block shutdown propagation uses a __device__ static flag in
//      the user's TU. Set by block 0 thread 0 inside the inner poll; the
//      atomic_block_barrier's RELEASE+ACQUIRE fence pair makes the store
//      visible to all blocks after the barrier.
//   4. Single thread (block 0 thread 0) writes to host-mapped queue fields.
//      Multi-thread/multi-block writes to host UC pages are unsafe.
//
// API surface:
//   persistent_iter_poll_barrier — block 0 thread 0 polls seq_num; on
//     observing seq>last_seq or shutdown, all blocks barrier together.
//     Returns kPersistentShutdown if exit requested.
//   persistent_iter_ack — block 0 thread 0 writes ack=seq with AGENT scope.
//     Call after dispatch_opcode loop completes + threadfence.
//
// Standing regression test:
//   kernels/diagnostic/persistent_skeleton_repro/prod_kernel_test —
//   loads megakernel.hsaco's `persistent_worker` symbol, dispatches 3
//   sequential batches, asserts ack matches seq each time. Runs in ~1
//   second standalone (no model load) and catches any protocol regression
//   instantly. Wire into cargo test / CI gate.

#ifndef BRAIDINFER_RDNA3_PERSISTENT_PROTOCOL_H
#define BRAIDINFER_RDNA3_PERSISTENT_PROTOCOL_H

#include <stdint.h>
#include "../worker_queue.h"
#include "rdna3_barrier.h"
#include "rdna3_peer.h"  // host_uc_store_agent

namespace braidinfer { namespace rdna3 {

enum PersistentIterResult : uint32_t {
    kPersistentContinue = 0,
    kPersistentShutdown = 1,
};

// Sentinel progress_pc values. >= 0x10000000 so they never collide with
// per-instruction pc values (0..255).
#define BRAIDINFER_PC_OUTER         0x10000001u
#define BRAIDINFER_PC_IN_POLL       0x10000002u
#define BRAIDINFER_PC_POST_POLL     0x10000003u
#define BRAIDINFER_PC_POST_BARRIER  0x10000004u
#define BRAIDINFER_PC_PRE_DISPATCH  0x10000005u

// One iteration of the canonical persistent worker protocol up through
// the post-barrier sync. The caller is responsible for invoking
// dispatch_opcode for inst[0..num_inst_out) AFTER this returns
// kPersistentContinue, then calling persistent_iter_ack.
//
// MUST be called from every block in the cooperative grid (all blocks
// participate in the trailing atomic_block_barrier). User-provided
// `shutdown_seen_flag` is a pointer to a __device__ static uint32_t in
// the calling TU; it is set by block 0 thread 0 when host shutdown is
// observed, and all blocks read it post-barrier.
// Queue type must expose, at compatible offsets:
//   volatile uint32_t seq_num         // host triggers
//   volatile uint32_t shutdown        // host requests exit
//   uint32_t num_instructions         // batch size
//   volatile uint32_t ack             // worker publishes
//   volatile uint32_t done            // worker writes 1 on exit
// progress_pc_field is a host-readable diagnostic field, may be nullptr.
// For WorkerQueue use &queue->progress_pc; for MoeWorkerQueue use
// &queue->debug_stage; for queues without one, pass nullptr.
template <typename Queue>
__device__ __forceinline__ PersistentIterResult
persistent_iter_poll_barrier(
    volatile Queue* queue,
    volatile uint32_t* progress_pc_field,
    GridBarrierState* gbs,
    uint32_t last_seq,
    uint32_t* shutdown_seen_flag,
    uint32_t* seq_out,
    uint32_t* num_inst_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (progress_pc_field) {
            host_uc_store_agent(progress_pc_field, BRAIDINFER_PC_OUTER);
            host_uc_store_agent(progress_pc_field, BRAIDINFER_PC_IN_POLL);
        }
        while (true) {
            // L0/L1 invalidation is documented silently no-op on gfx11+ in
            // cooperative-kernel polling context (kb gl-inv-noop-gfx11) but
            // retained for code intent.
            asm volatile("buffer_gl0_inv\n\t" "buffer_gl1_inv\n\t" ::: "memory");
            if (queue->shutdown) {
                *shutdown_seen_flag = 1u;
                break;
            }
            uint32_t s = queue->seq_num;
            if (s > last_seq) break;
            __builtin_amdgcn_s_sleep(1);
        }
        if (progress_pc_field)
            host_uc_store_agent(progress_pc_field, BRAIDINFER_PC_POST_POLL);
    }
    atomic_block_barrier(gbs);
    if (threadIdx.x == 0 && blockIdx.x == 0 && progress_pc_field)
        host_uc_store_agent(progress_pc_field, BRAIDINFER_PC_POST_BARRIER);

    if (*shutdown_seen_flag) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            queue->seq_num = 0xFFFFFFFFu;
            queue->done = 1;
        }
        return kPersistentShutdown;
    }

    // Read seq and num_inst once, per-thread, after the barrier's
    // ACQUIRE fence has propagated the host's writes.
    uint32_t seq = queue->seq_num;
    uint32_t num_inst = queue->num_instructions;
    if (num_inst == 0) num_inst = 1;  // backward compat
    *seq_out = seq;
    *num_inst_out = num_inst;
    if (threadIdx.x == 0 && blockIdx.x == 0 && progress_pc_field)
        host_uc_store_agent(progress_pc_field, BRAIDINFER_PC_PRE_DISPATCH);
    return kPersistentContinue;
}

// Canonical ack write. Call AFTER the dispatch_opcode loop completes and
// AFTER a __threadfence(). AGENT scope is mandatory — SYSTEM-scope wedges
// across the next iter's barrier under multi-GPU PCIe pressure.
template <typename Queue>
__device__ __forceinline__ void
persistent_iter_ack(volatile Queue* queue, uint32_t seq) {
#ifdef BRAIDINFER_ACK_DRAIN_VSCNT
    // braidinfer-snl Option (D): drain SYSTEM-scope vector store counter
    // before ack. Ensures any cross-GPU PCIe writes from this iter
    // (e.g., MoE activation production on GPU 0, or expert output
    // writes from workers) are visible to OTHER GPUs before the
    // host sees ack=seq. Without this, multi-GPU MoE shows ~33%
    // non-determinism (benchmark_results/regression/2026-05-14_post_wedge_fix/
    // pky2_moe_4gpu_30runs.log).
    //
    // §11.4 risk: vscnt drain on a thread that diverges across blocks
    // (rare) could stall multi-block convergence at the next barrier.
    // Mitigation: only block 0 thread 0 drains here, and the NEXT
    // barrier is at the top of the next outer iter (post-poll, many
    // ops away). Single-thread drain — no cross-block dependency.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        host_uc_store_agent(&queue->ack, seq);
    }
}

// Tag macro for the CI grep gate. Any persistent_worker entry point must
// contain this token in its body (verifies the canonical protocol is in
// use rather than a hand-rolled outer loop).
#define BRAIDINFER_PERSISTENT_PROTOCOL_CANONICAL  /* present-marker */

}}  // namespace braidinfer::rdna3

#endif  // BRAIDINFER_RDNA3_PERSISTENT_PROTOCOL_H
