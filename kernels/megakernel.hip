#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#define RDNA3_PERSISTENT_POLLING_VALIDATED 1
#include "rdna3/rdna3_persistent.h"
#include "rdna3/rdna3_barrier.h"          // braidinfer::rdna3::atomic_block_barrier
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
    // Phase 0b wedge diagnostic 2026-05-13 (#5 disambiguation): every block's
    // thread 0 increments block_alive_count at kernel entry, BEFORE any
    // barrier or poll. DispatchError::Timeout reads this on the host side:
    //   count <  num_blocks  => cooperative grid never fully scheduled
    //   count == num_blocks  => all blocks running; wedge is downstream
    if (threadIdx.x == 0) {
        atomicAdd((unsigned int*)&queue->block_alive_count, 1u);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("persistent_worker: started, queue=%p\n", (void*)queue);
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
        if (threadIdx.x == 0 && blockIdx.x == 0)
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000001u);
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
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000002u);
            while (true) {
                // Per-iteration L0/L1 invalidation. Documented silently no-op
                // on gfx11+ in cooperative-kernel context (kb 2026-05-14
                // gl-inv-noop-gfx11) but retained for code intent.
                asm volatile("buffer_gl0_inv\n\t" "buffer_gl1_inv\n\t" ::: "memory");
                if (queue->shutdown) {
                    g_shutdown_seen = 1u;
                    break;
                }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000003u);
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        if (threadIdx.x == 0 && blockIdx.x == 0)
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000004u);
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

        // Phase 2' deferred-ack pattern REMOVED 2026-05-14 — it caused a
        // protocol deadlock (ack=N written at top of iter N+1 AFTER iter
        // N+1's inner-poll, but host blocked on ack=N before sending
        // seq=N+1). Ack write now lives post-dispatch at the END of this
        // iter (see below). See GFX1100_ARCH.md §11.15.
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

        if (threadIdx.x == 0 && blockIdx.x == 0)
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000005u);

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
        }

        // Flush all threads' writes to system-visible memory.
        __threadfence();
        // Immediate ack=seq. The Phase 2' deferred-ack pattern (ack written
        // at top of next iter's body at line 225) deadlocks: ack=N is only
        // written in iter N+1, but iter N+1's inner poll waits for seq=N+1
        // which the host won't send until ack=N visible. Verified empirically
        // 2026-05-14 via standalone prod_kernel_test. AGENT scope avoids
        // §11.4 SYSTEM-scope hazard with the next iter's barrier.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            braidinfer::rdna3::host_uc_store_agent(&queue->ack, seq);
        }
        last_seq = seq;
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void simple_persistent_worker(
    volatile WorkerQueue* queue,
    WatchdogState* watchdog
) {
    if (threadIdx.x == 0) {
        atomicAdd((unsigned int*)&queue->block_alive_count, 1u);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("simple_persistent_worker: started, queue=%p\n", (void*)queue);
    }
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    // Force dispatch_opcode body to be present in this function via an
    // unreachable branch. Tests whether the size/register-pressure of
    // having that body inlined changes correctness of the inner poll loop.
    if (queue->shutdown == 0xDEADBEEFu) {
        const u64 fake_inst[INST_SIZE_WORDS] = {0};
        dispatch_opcode(fake_inst, grid, &g_gbs,
                        nullptr, 0, nullptr, nullptr, 0, nullptr);
    }
    uint32_t last_seq = 0;
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000001u);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Same write pattern as production: host_uc_store_agent.
            braidinfer::rdna3::host_uc_store_agent(&queue->progress_pc, 0x10000002u);
            while (true) {
                if (queue->shutdown) {
                    queue->ack = 0xFFFFFFFFu;
                    queue->done = 1u;
                    return;
                }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
            queue->progress_pc = 0x10000003u;
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            uint32_t seq = queue->seq_num;
            queue->ack = seq;
            last_seq = seq;
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
    }
}
