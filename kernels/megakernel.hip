#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#include "watchdog.h"
#include "rdna3_sync.h"          // braidinfer::rdna3::atomic_block_barrier
#include "op_profile.h"          // OP_PROFILE_BEGIN/END (no-op when -DBRAIDINFER_OP_PROFILE unset)
#include "dump.h"                // dump_instruction_output (used by megakernel_dispatch)

// Megakernel: persistent cooperative kernel that executes a program of instructions.
// Launched once with max cooperative blocks (384 on gfx1100 with 256 threads).
// Each instruction specifies an operation and how many "virtual blocks" it needs.
// Physical blocks loop over virtual blocks: for (vb = blockIdx.x; vb < grid_x; vb += gridDim.x)

// Per-TU GridBarrierState: zero-init __device__ static, cache-coherent VRAM.
// One state suffices — the barrier protocol resets the counter via generation flip.
__device__ static braidinfer::rdna3::GridBarrierState g_gbs;

// 4fg.5: cross-block shutdown flag in AGENT-scope cached VRAM (no PCIe drain
// hazard). Set by block 0 thread 0 when queue->shutdown is observed; read by
// all blocks after the post-poll barrier. Communicates shutdown without
// requiring a SYSTEM-scope UC write before the barrier (kb
// rdna3-atomic-block-barrier-multi-gpu-fundamental-issue corrected mechanism).
__device__ static uint32_t g_shutdown_seen;

#include "megakernel_ops.hip"
#include "megakernel_moe.hip"
#include "megakernel_moe_dispatch.hip"
#include "worker_queue.h"               // struct WorkerQueue (persistent_worker entry)
#include "megakernel_dispatch.hip"  // unified dispatch_opcode helper (zqw)

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 18  // 18 u64s per instruction = 144 bytes

extern "C" __global__ __launch_bounds__(256) void megakernel_f32(
    const u64* __restrict__ program,
    int num_instructions,
    WatchdogState* watchdog,  // host-mapped watchdog state; NULL to disable
    uint64_t* op_profile      // op_profile counter buffer (null = disabled)
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // bz0 probe v3: kernel-entry L1 invalidation WITHOUT s_waitcnt (which has hang
    // pathology on gfx1100 — confirmed by PCIe-coherency investigation: vmcnt(0) blocks
    // forever if any vmem op response is delayed). RDNA3 cooperative-launch AQL packets
    // use AGENT scope which flushes/invalidates L2 but NOT per-WGP L1.
    if (threadIdx.x == 0) {
        asm volatile("buffer_gl0_inv\n\t" "buffer_gl1_inv\n\t" ::: "memory");
    }
    __syncthreads();

    // Dump state: encoded in first instruction if OP_NOP with dump_base != 0
    // First instruction words: [0]=opcode(NOP), [1]=dump_base ptr, [2]=dump_capacity, [3]=dump_count ptr
    char* dump_base = nullptr;
    int dump_capacity = 0;
    int* dump_count = nullptr;
    if (num_instructions > 0 && inst_opcode(program) == OP_NOP && ((const NopInst*)program)->dump_buf != nullptr) {
        const NopInst* nop = (const NopInst*)program;
        dump_base = (char*)nop->dump_buf;
        dump_capacity = (int)nop->max_slots;
        dump_count = (int*)nop->dump_counter;
    }

    uint32_t watchdog_beat_ctr = 0;
    // Seed counter from prior value so successive kernel launches against the
    // same WatchdogState produce a monotonically advancing sequence (otherwise
    // every launch stores 100, 200, ... — identical across launches — and the
    // host watchdog spuriously declares no-progress between launches).
    if (watchdog) watchdog_init(watchdog, &watchdog_beat_ctr);

    for (int pc = 0; pc < num_instructions; pc++) {
        // Watchdog: check force_exit between instructions. All blocks reach this point
        // via the grid.sync() at the end of each instruction, so cooperative exit is safe.
        if (watchdog && watchdog_poll_and_check(watchdog, &g_gbs, (uint32_t)pc)) {
            return;  // all blocks return together; no done_flag (megakernel has no persistent done)
        }
        if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);

        const u64* inst = program + pc * INST_SIZE;
        u32 opcode = inst_opcode(inst);

        if (opcode == OP_HALT) break;

        // Unified dispatch (zqw): all opcode routing lives in
        // dispatch_opcode in megakernel_dispatch.hip; both this
        // megakernel entry and the persistent_worker entry call it.
        dispatch_opcode(
            inst, grid, &g_gbs,
            dump_base, dump_capacity, dump_count,
            op_profile, pc, watchdog
        );
    }

    // Signal clean exit so host watchdog pauses its no-progress timer until
    // the next launch (otherwise it spuriously fires on the frozen
    // progress_counter during inter-segment dispatch gaps).
    if (watchdog) watchdog_signal_exited(watchdog);
}

// ---- persistent_worker entry point (zqw merge) ----
// Persistent cooperative worker kernel: polls host-mapped WorkerQueue for
// operations. CPU-scheduled via memory writes (no HIP API calls in hot path).
// One instance per GPU, launched once at init, runs for model lifetime.
//
// Same op-dispatch chain as megakernel_f32 above (via dispatch_opcode helper).
// Differences from megakernel_f32 are confined to the driver loop:
//   - mailbox-poll lifecycle (wait for seq_num, ack on completion)
//   - per-batch L1 invalidation
//   - per-pc instruction copy to LDS
//   - shutdown-flag handling
//   - __launch_bounds__(256, 2) instead of (256) for cooperative occupancy
extern "C" __global__ __launch_bounds__(256, 2) void persistent_worker(
    volatile WorkerQueue* queue,
    WatchdogState* watchdog  // host-mapped watchdog state; NULL to disable
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("persistent_worker: started, queue=%p\n", (void*)queue);
    }

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;
    uint32_t watchdog_beat_ctr = 0;

    while (true) {
        // Mark idle BEFORE entering the inner poll loop. Distinguishes idle
        // wait (no batch yet) from hang for the host watchdog.
        if (watchdog) watchdog_signal_exited(watchdog);

        // Single-thread poll to avoid PCIe storm.
        // 4fg.5: NO PCIe writes here — the upcoming atomic_block_barrier
        // emits s_waitcnt_vscnt that stalls indefinitely under multi-GPU
        // PCIe pressure if a SYSTEM-scope UC write is in flight. Defer
        // queue->seq_num=sentinel and queue->done=1 to AFTER the barrier.
        // Communicate shutdown via g_shutdown_seen (AGENT-scope VRAM, no
        // PCIe). The barrier's RELEASE+ACQUIRE fence pair makes the
        // pre-barrier store visible to all blocks post-barrier.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (queue->shutdown) {
                    g_shutdown_seen = 1u;
                    break;
                }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        // Clear `exited` — kernel is now actively processing a batch.
        if (watchdog) watchdog_init(watchdog, &watchdog_beat_ctr);

        // 4fg.5: shutdown detection via g_shutdown_seen (AGENT-scope, set
        // before barrier; visible to all blocks via barrier's fence pair).
        // The host-mapped writes (queue->seq_num=sentinel, queue->done=1)
        // happen AFTER all barriers — no more PCIe-write-before-barrier
        // hazard. No fence after the writes: host poll will see them within
        // microseconds via natural PCIe propagation.
        if (g_shutdown_seen) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                queue->seq_num = 0xFFFFFFFFu;
                queue->done = 1;
            }
            return;
        }
        uint32_t seq = queue->seq_num;

        // Watchdog: cooperative exit if force_exit set (non-shutdown path).
        if (watchdog && watchdog_poll_and_check(watchdog, &g_gbs, 0u)) {
            watchdog_signal_exit(&queue->done);
            return;
        }
        if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);

        uint32_t num_inst = queue->num_instructions;
        if (num_inst == 0) num_inst = 1; // backward compat: 0 means 1

        // bz0 fix: invalidate per-WGP L1 at batch boundary. Activation buffer writes by
        // prior batch may leave stale L1 lines on different WGPs.
        if (threadIdx.x == 0) {
            asm volatile("buffer_gl1_inv" ::: "memory");
        }
        __syncthreads();

        // Snapshot trace-dump pointers for this batch (zqw): C-style cast
        // strips volatile from the queue's field types. Set by Rust
        // PersistentDispatch when Model::trace is active; null = trace disabled.
        char* dump_base = (char*)queue->dump_base;
        int dump_capacity = (int)queue->dump_capacity;
        int* dump_count = (int*)queue->dump_count;
        uint64_t* op_profile = (uint64_t*)queue->op_profile;

        // Process batch of instructions; dispatch helper handles opcode routing,
        // post-op barrier, and trace dump.
        for (uint32_t pc = 0; pc < num_inst; pc++) {
            // Copy instruction to shared memory (avoids PCIe reads in virtual block loop)
            __shared__ u64 sinst[INST_SIZE_WORDS];
            const u64* src = (const u64*)(queue->inst + pc * INST_SIZE_WORDS);
            if (threadIdx.x < INST_SIZE_WORDS) {
                sinst[threadIdx.x] = src[threadIdx.x];
            }
            __syncthreads();
            const u64* inst = sinst;

            // OP_HALT: in-band shutdown. 4fg.5 fix — align all blocks via a
            // grid barrier BEFORE the host-mapped writes, so the writes have
            // no following barrier to stall on. Per kb 4fg-4 the previous
            // code had a 20% rate of "done=1 set but kernel never
            // terminates": a block lagging in the prior op's trailing
            // barrier would never reach OP_HALT, and MES wouldn't fire AQL
            // completion. The barrier here forces all blocks to converge at
            // OP_HALT before any of them write done=1 + return. After the
            // barrier, the host-mapped UC writes are unfenced — they
            // propagate to host via PCIe within microseconds and the host's
            // poll picks them up.
            //
            // IMPORTANT: progress_pc=pc must NOT be written before the
            // OP_HALT check — that store is PCIe-bound and would stall the
            // OP_HALT barrier's leading s_waitcnt_vscnt. Defer to past the
            // OP_HALT check below.
            u32 opcode = inst_opcode(inst);
            if (opcode == OP_HALT) {
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    queue->done = 1;
                }
                return;
            }
            // Report progress so CPU can identify stuck instruction on
            // timeout. Safe to write here: dispatch_opcode does significant
            // compute before its trailing barrier, so the PCIe write drains
            // before any s_waitcnt_vscnt fires.
            if (threadIdx.x == 0 && blockIdx.x == 0) queue->progress_pc = pc;
            dispatch_opcode(
                inst, grid, &g_gbs,
                dump_base, dump_capacity, dump_count,
                op_profile, (int)pc, watchdog
            );
        }

        // Ack: flush ALL threads' writes to system-visible memory, then write ack.
        // grid.sync() ensures all threads completed (device-scope). But P2P readers on
        // GPUs 1-3 bypass GPU 0's L2 and read VRAM directly. grid.sync() alone does NOT
        // flush L2 → VRAM for P2P visibility. __threadfence() from EACH thread
        // ensures that thread's cache lines are written back to system-visible memory.
        // Must be called by ALL threads (not just thread 0,0) to cover all written cache lines.
        __threadfence();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            queue->ack = seq;
        }
        last_seq = seq;
    }
}
