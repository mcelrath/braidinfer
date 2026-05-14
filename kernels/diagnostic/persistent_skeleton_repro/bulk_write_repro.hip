// bulk_write_repro: minimal isolation of bulk-write-before-trigger as the
// production-wedge mechanism. Mirrors crates/braidinfer-runtime/src/
// persistent_dispatch.rs dispatch_batch() write pattern within a single
// persistent kernel lifetime. Within-lifetime, isolated trigger writes (the
// prior persistent_skeleton_repro V0 baseline) PASS 100/100; production
// dispatch (32KB inst[] before seq_num trigger) wedges on first dispatch.
//
// Layout matches kernels/worker_queue.h field-for-field:
//   seq_num         u32 @ 0   — host trigger; worker polls
//   shutdown        u32 @ 4
//   num_instructions u32 @ 8
//   block_alive_count u32 @ 12
//   inst[256*16]    u64 @ 16  — 32768B bulk-write region
//   ack             u32 @ after inst
//   progress_pc     u32
//
// Host BULK_BYTES env var selects how many bytes of inst[] are written per
// dispatch (0..32768). Cache boundaries on gfx1100 RX 7900 XTX (rocminfo):
//   L1 = 32 KB, L2 = 6 MB, L3 (Infinity) = 96 MB. Sweep across to localize
//   which cache level is implicated.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

#define INST_SIZE_WORDS 16
#define MAX_BATCH_INSTRUCTIONS 256

struct ReproQueue {
    volatile uint32_t seq_num;          // 0
    volatile uint32_t shutdown;         // 4
    uint32_t num_instructions;          // 8
    volatile uint32_t block_alive_count; // 12
    uint64_t inst[MAX_BATCH_INSTRUCTIONS * INST_SIZE_WORDS]; // 16 .. 32784
    volatile uint32_t ack;
    volatile uint32_t progress_pc;
    volatile uint32_t completed_dispatches;
    uint32_t _pad;
};

#define PC_OUTER        0x10000001u
#define PC_IN_POLL      0x10000002u
#define PC_POST_POLL    0x10000003u
#define PC_PRE_ACK      0x10000005u

extern "C" __global__ __launch_bounds__(256, 2)
void bulk_repro_worker(volatile ReproQueue* q) {
    if (threadIdx.x == 0) {
        atomicAdd((unsigned int*)&q->block_alive_count, 1u);
    }
    cg::grid_group grid = cg::this_grid();
    __shared__ uint32_t s_last_seq;
    uint32_t last_seq = 0;
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            q->progress_pc = PC_OUTER;
            q->progress_pc = PC_IN_POLL;
            while (true) {
                asm volatile("buffer_gl0_inv\n\t" "buffer_gl1_inv\n\t" ::: "memory");
                if (q->shutdown) { last_seq = 0xFFFFFFFFu; break; }
                uint32_t s = q->seq_num;
                if (s > last_seq) { last_seq = s; break; }
                __builtin_amdgcn_s_sleep(1);
            }
            q->progress_pc = PC_POST_POLL;
        }
        // Broadcast last_seq through q->seq_num (host-mapped, all blocks read).
        // 0xFFFFFFFF in q->shutdown signals exit; all blocks check it post-barrier.
        grid.sync();
        if (q->shutdown) {
            if (threadIdx.x == 0 && blockIdx.x == 0) q->ack = 0xFFFFFFFFu;
            return;
        }
        // Re-sync last_seq from q->seq_num for non-block0 threads.
        if (threadIdx.x != 0 || blockIdx.x != 0) last_seq = q->seq_num;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            q->progress_pc = PC_PRE_ACK;
            q->ack = last_seq;
            q->completed_dispatches = q->completed_dispatches + 1u;
        }
        grid.sync();
    }
}
