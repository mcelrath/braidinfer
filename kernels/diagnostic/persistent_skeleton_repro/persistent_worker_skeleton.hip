// Minimal reproduction skeleton for the multi-GPU persistent-cooperative-kernel
// wedge. Strips kernels/megakernel.hip persistent_worker (lines 115-283) down
// to the surface pattern that is suspected of triggering the wedge:
//
//   outer while(true)
//     UC-store @ progress_pc = OUTER       (matches production marker discipline)
//     [§11.4 PCIe-write-before-barrier site, retained for fidelity]
//     UC-store @ progress_pc = IN_POLL
//     while: poll q->seq_num until s > last_seq
//     UC-store @ progress_pc = POST_POLL
//     atomic_block_barrier   <-- gated by variant (V2 skips)
//     UC-store @ progress_pc = POST_BARRIER
//     write ack
//     (V3: return after one iteration; otherwise loop)
//
// Variants (selected at host launch time, passed as int kernel arg):
//   V0 (=0): 4-GPU persistent + post-poll barrier         — target wedge
//   V1 (=1): 2-GPU persistent + post-poll barrier         — control (§5.5 envelope)
//   V2 (=2): 4-GPU persistent, NO post-poll barrier       — isolates barrier
//   V3 (=3): 4-GPU non-persistent (exit after one iter)   — isolates persistent
//   V4a(=4): peer-GPU dispatch over UC device queue       — separate fixture (NOT YET)
//
// Diagnostic fields written to host_uc_store_agent so the host can read on
// timeout: block_alive_count (atomicAdd at entry per block), progress_pc
// sentinel values. Match kernels/worker_queue.h + commit 64ce6f1's diagnostics.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>

#include "../../rdna3_sync.h"
#include "../../rdna3_coherence.h"

namespace cg = cooperative_groups;
using braidinfer::rdna3::GridBarrierState;
using braidinfer::rdna3::atomic_block_barrier;
using braidinfer::rdna3::host_uc_store_agent;

// Per-TU GridBarrierState, mirrors production kernels/megakernel.hip:16.
__device__ static GridBarrierState g_gbs;

// Layout matches the on-the-wire host-side struct in host_runner.cpp.
// All 32-bit fields, all volatile from kernel POV (host-mapped UC). The
// queue->seq_num is the load-bearing read in the wedge.
struct WedgeReproQueue {
    volatile uint32_t seq_num;             // CPU writes, worker polls
    volatile uint32_t ack;                 // worker writes, CPU polls
    volatile uint32_t progress_pc;         // worker writes sentinels
    volatile uint32_t block_alive_count;   // worker atomicAdds at entry
    volatile uint32_t shutdown;            // CPU writes 1 for clean exit
    volatile uint32_t completed_dispatches;// worker increments per ack
    uint32_t _pad[2];                      // align to 32 bytes
};

// Variant tag. Keep in sync with host_runner.cpp + README.md.
#define VARIANT_V0_PERSIST_BARRIER 0
#define VARIANT_V1_PERSIST_BARRIER_2GPU 1
#define VARIANT_V2_PERSIST_NOBARRIER 2
#define VARIANT_V3_SINGLELAUNCH 3
#define VARIANT_V4A_PEER_UC_DEVICE 4  // not implemented in this binary

// Sentinel progress_pc values. Match production kernels/megakernel.hip values
// so a future investigator reading both logs gets identical semantics.
#define PC_OUTER       0x10000001u
#define PC_IN_POLL     0x10000002u
#define PC_POST_POLL   0x10000003u
#define PC_POST_BARRIER 0x10000004u
#define PC_PRE_ACK     0x10000005u

extern "C" __global__ __launch_bounds__(256, 2)
void persistent_worker_skeleton(
    volatile WedgeReproQueue* q,
    int variant
) {
    // Every block's thread 0 atomicAdds at kernel entry — confirms cooperative
    // grid is fully scheduled (read on host on timeout).
    if (threadIdx.x == 0) {
        atomicAdd((unsigned int*)&q->block_alive_count, 1u);
    }

    cg::grid_group grid = cg::this_grid();
    uint32_t last_seq = 0;

    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            host_uc_store_agent(&q->progress_pc, PC_OUTER);

        // Single-thread poll. Block 0 thread 0 only. Other threads/blocks
        // proceed to the barrier (if enabled) and wait for block 0.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            host_uc_store_agent(&q->progress_pc, PC_IN_POLL);
            while (true) {
                if (q->shutdown) break;
                uint32_t s = q->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
            host_uc_store_agent(&q->progress_pc, PC_POST_POLL);
        }

        // V0/V1/V3: post-poll grid barrier. V2 skips to isolate whether the
        // barrier interaction is load-bearing.
        if (variant != VARIANT_V2_PERSIST_NOBARRIER) {
            atomic_block_barrier(&g_gbs);
            if (threadIdx.x == 0 && blockIdx.x == 0)
                host_uc_store_agent(&q->progress_pc, PC_POST_BARRIER);
        }

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (q->shutdown) {
                // Clean exit path. Match production OP_HALT signaling shape:
                // write ack to a known sentinel so the host can distinguish
                // clean exit from wedge.
                host_uc_store_agent(&q->ack, 0xFFFFFFFFu);
                return;
            }
            host_uc_store_agent(&q->progress_pc, PC_PRE_ACK);
            // Sample seq fresh (in case shutdown wasn't set but ack still
            // needs to reflect the latest dispatch).
            uint32_t seq = q->seq_num;
            host_uc_store_agent(&q->ack, seq);
            host_uc_store_agent(&q->completed_dispatches, q->completed_dispatches + 1u);
            last_seq = seq;
        }

        // V3: non-persistent. Exit after one ack instead of looping.
        if (variant == VARIANT_V3_SINGLELAUNCH) {
            return;
        }
    }
}
