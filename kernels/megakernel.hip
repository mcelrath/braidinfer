#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#define RDNA3_PERSISTENT_POLLING_VALIDATED 1
#include "rdna3/rdna3_persistent.h"
#include "rdna3/rdna3_barrier.h"          // braidinfer::rdna3::atomic_block_barrier
#include "rdna3/rdna3_persistent_protocol.h"  // canonical poll/ack helper
#include "rdna3/rdna3_mailbox.h"              // mailbox_load_descriptor (volatile-safe payload read)
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
#include "op_kv_write_paged_batch.hip"  // bd srg6.4 Phase 3b: OP_KV_WRITE_PAGED_BATCH
#include "worker_queue.h"               // struct WorkerQueue (persistent_worker entry)
#include "megakernel_dispatch.hip"  // unified dispatch_opcode helper (zqw)

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 18  // 18 u64s per instruction = 144 bytes

// ---- persistent_worker entry point (zqw merge) ----
// Persistent cooperative worker kernel: polls host-mapped WorkerQueue for
// operations. CPU-scheduled via memory writes (no HIP API calls in hot path).
// One instance per GPU, launched once at init, runs for model lifetime.
//
// Calls dispatch_opcode (kernels/megakernel_dispatch.hip) for per-opcode work.
// Driver loop responsibilities:
//   - mailbox-poll lifecycle (wait for seq_num, ack on completion)
//   - per-batch L1 invalidation
//   - per-pc instruction copy to LDS
//   - shutdown-flag handling
//   - __launch_bounds__(256, 2) for cooperative occupancy
// (Prior to bd 9gmh Phase 4 / commit 5953598, a sibling `megakernel_f32`
// one-shot entry existed; it was deleted once persistent dispatch covered
// both prefill and decode.)
extern "C" __global__ __launch_bounds__(256, 2) void persistent_worker(
    volatile WorkerQueue* queue,
    WatchdogState* watchdog  // host-mapped watchdog state; NULL to disable
) {
    // Phase 0b wedge diagnostic 2026-05-13 (#5 disambiguation): every block's
    // thread 0 increments block_alive_count at kernel entry, BEFORE any
    // barrier or poll. DispatchError::Timeout reads this on the host side:
    //   count <  num_blocks  => cooperative grid never fully scheduled
    //   count == num_blocks  => all blocks running; wedge is downstream
    if (threadIdx.x == 0) {
        atomicAdd((unsigned int*)&queue->block_alive_count, 1u);
    }
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;
    uint32_t watchdog_beat_ctr = 0;

    // Phase 0b wedge diagnostic 2026-05-13: sentinel values written to
    // progress_pc at distinguishing code points (>= 0x10000000 so they
    // never collide with valid pc values 0..255). Lets the host's
    // DispatchError::Timeout message pinpoint the wedge site:
    //   0                  -> block 0 thread 0 never wrote (extreme #5 / kernel didn't run)
    //   0x10000001 OUTER   -> entered outer while(true), wedged before poll
    //   0x10000002 IN_POLL -> wedged inside inner spin, never saw seq>last_seq
    //   0x10000003 POST_POLL -> saw seq, exited poll, wedged before barrier
    //   0x10000004 POST_BAR -> barrier completed, wedged before instruction loop
    //   0x10000005 PRE_LOOP -> about to read first instruction
    //   0..255 (pc value)  -> wedged downstream during dispatch_opcode
    //
    // Writes use host_uc_store_agent (AGENT scope) — same pattern as ack /
    // progress_pc writes elsewhere; avoids the SYSTEM-scope vscnt emit that
    // is §11.4-hazardous before atomic_block_barrier.

    while (true) {
        // Mark idle BEFORE the canonical poll/barrier.
        if (watchdog) watchdog_signal_exited(watchdog);

        // Canonical poll-barrier: handles inner-poll, shutdown propagation,
        // grid barrier, and reads seq+num_inst once all blocks are
        // synchronized. See kernels/rdna3/rdna3_persistent_protocol.h.
        uint32_t seq, num_inst;
        if (braidinfer::rdna3::persistent_iter_poll_barrier(
                queue, &queue->progress_pc, &g_gbs, last_seq,
                &g_shutdown_seen, &seq, &num_inst)
                == braidinfer::rdna3::kPersistentShutdown) {
            return;
        }

        // Clear `exited` — kernel is now actively processing a batch.
        if (watchdog) watchdog_init(watchdog, &watchdog_beat_ctr);

        // Watchdog: cooperative exit if force_exit set (non-shutdown path).
        if (watchdog && watchdog_poll_and_check(watchdog, &g_gbs, 0u)) {
            watchdog_signal_exit(&queue->done);
            return;
        }
        if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);

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

        if (threadIdx.x == 0 && blockIdx.x == 0)
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000005u);

        // Process batch of instructions; dispatch helper handles opcode routing,
        // post-op barrier, and trace dump.
        for (uint32_t pc = 0; pc < num_inst; pc++) {
            // Copy instruction to shared memory (avoids PCIe reads in virtual block loop)
            // bd 4e2m / udi #3221: use mailbox_load_descriptor (rdna3_mailbox.h)
            // to preserve volatile through pointer arithmetic. The prior
            // `(const u64*)` cast stripped volatile, causing ROCm clang to emit
            // plain L2-cached global_load_b64 — empirically necessary-but-not-
            // sufficient correctness fix (closes the cast-strip bug at the
            // shader level; cold-start race lives below L1/L2, see COHERENCE.md).
            __shared__ u64 sinst[INST_SIZE_WORDS];
            if (threadIdx.x < INST_SIZE_WORDS) {
                sinst[threadIdx.x] = braidinfer::rdna3::mailbox_load_descriptor(
                    &queue->inst[pc * INST_SIZE_WORDS + threadIdx.x]
                );
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
            // OP_HALT check — that store (even AGENT-scope) precedes the
            // OP_HALT barrier if placed here. Defer to past the OP_HALT
            // check below.
            u32 opcode = inst_opcode(inst);
            if (opcode == OP_HALT) {
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    queue->done = 1;
                }
                return;
            }
            // Report progress AFTER dispatch_opcode returns (Phase 2' fix).
            // Writing progress_pc BEFORE dispatch_opcode would put a UC
            // volatile store before the trailing atomic_block_barrier inside
            // dispatch_opcode, emitting s_waitcnt_vscnt before the barrier's
            // leading vscnt — the wedge site under 4-GPU PCIe pressure.
            // Deferring past dispatch_opcode means the trailing barrier of
            // the PREVIOUS op has already completed; the next barrier is the
            // trailing barrier of THIS op, which follows significant compute.
            // The CPU reads progress_pc only on timeout, so one-instruction
            // staleness is acceptable.
            dispatch_opcode(
                inst, grid, &g_gbs,
                dump_base, dump_capacity, dump_count,
                op_profile, (int)pc, watchdog
            );
            // Phase 2': AGENT scope (was volatile). Diagnostic write only;
            // CPU reads on timeout, not real-time. Same reasoning as 4fg.5
            // watchdog_beat: UC memory needs no SYSTEM fence, and SYSTEM scope
            // emits s_waitcnt_vscnt before the next dispatch's trailing barrier.
            if (threadIdx.x == 0 && blockIdx.x == 0)
                braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, (uint32_t)pc);
            // bd 6k9m: beat the watchdog PER INSTRUCTION, not once per batch.
            // Previously the only beat was before this loop (one beat per batch),
            // so a large but HEALTHY batch — e.g. qwen35_27b's 64-layer prefill:
            // many fast ops whose total exceeds the 2s no-progress window — tripped
            // the watchdog and was FALSELY aborted (op_id=0/counter=1 signature).
            // Beating each instruction resets the no-progress window per op: the
            // watchdog still catches a genuinely STUCK op (no beat for 2s within a
            // single instruction) but tolerates a long sequence of fast ops.
            // watchdog_beat is block0/thr0-only and flushes to host only every
            // 100th beat (throttled), so the per-instruction cost is a local
            // increment, not a PCIe write.
            if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);
        }

        // Flush all threads' writes to system-visible memory.
        __threadfence();
        // Canonical ack write. AGENT scope inside the helper.
        braidinfer::rdna3::persistent_iter_ack(queue, seq);
        last_seq = seq;
    }
}

