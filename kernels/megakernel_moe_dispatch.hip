// bd 0hu3-b: op_moe_dispatch retired. GPU 0's MoE expert PRE-compute is now
// emitted as OP_MOE_FFN_REMOTE self-dispatch (see compile_inner_p2p in
// crates/braidinfer-runtime/src/megakernel/megakernel_compile.rs). Every GPU
// now runs the same op_moe_ffn_remote kernel for MoE expert compute in
// decode. This file retains only OP_MOE_DISPATCH_POST (the cross-GPU sum
// step), which still uses the MoeDispatchInst layout.
//
// Historical: op_moe_dispatch (PRE) was the "zero gpu0_acc + run GPU 0
// experts + barrierless copy to output_slots[0]" pattern. op_moe_ffn_remote
// implements the same pattern with self-pointing pointers when invoked on
// GPU 0; the wasted same-GPU `coop_copy(la, act_p2p)` is accepted.
#include "moe_work_queue.h"
#include "moe_expert_ops.h"
#define RDNA3_PERSISTENT_POLLING_VALIDATED 1
#include "rdna3/rdna3_persistent.h"
#include "rdna3/rdna3_barrier.h"
#include "rdna3/rdna3_mailbox.h"  // yef5.2 P1d: sentinel_spin_load_u32 for the option-c POST

// OP_MOE_DISPATCH_POST (opcode 45): sum output_slots across all GPUs into final_output.
//
// Fired on GPU 0 AFTER the CPU has waited for all worker OP_MOE_FFN_REMOTE acks.
// At this point output_slots[g * hs + 0..gupd] for all g in [0..total_gpus) are
// populated (workers via P2P writes, GPU 0 via OP_MOE_DISPATCH PRE) and visible
// to GPU 0 (output_slots is MTYPE=UC, so reads bypass L2).
//
// Reuses MoeDispatchInst layout. Reads only:
//   inst[2]  = output_slots_ptr     (float*, GPU 0 VRAM/UC, [num_gpus * hs])
//   inst[3]  = final_output_ptr     (float*, GPU 0 VRAM, [gupd])
//   inst[7]  = (..   << 32) | hs    (slot stride; gupd <= hs)
//   inst[15] = total_gpus
//   inst[16] = gate_up_in_dim
__device__ __attribute__((noinline)) void op_moe_dispatch_post(const u64* inst, cooperative_groups::grid_group& grid,
                                                                  braidinfer::rdna3::GridBarrierState* gbs,
                                                                  WatchdogState* watchdog = nullptr) {
    (void)watchdog;
    const MoeDispatchInst* i = (const MoeDispatchInst*)inst;
    float* output_slots = (float*)(uintptr_t)i->output_slots;
    float* final_output = (float*)(uintptr_t)i->final_output;
    int hs = (int)(i->num_workers_hs & 0xFFFFFFFF);
    int total_gpus = (int)i->num_gpus;
    int gupd = (int)i->gate_up_in_dim;
    if (gupd == 0) gupd = hs;

    // yef5.2 P1d (option c): if result_descs is set, the workers wrote their outputs
    // into their OWN UC-VRAM (no cross-GPU store) and AGENT-RELEASE-signalled per-worker
    // result sentinels. GPU0 acquire-spins each sentinel, invalidates L0/L1, then
    // PEER-READS each worker's UC-VRAM (fresh via §5.3) and sums — no output_slots
    // (host-UC) read, fixing the B1 §11.19(x)-stale POST. result_descs==0 => old path below.
    if (i->result_descs != 0) {
        const MoeResultDesc* descs = (const MoeResultDesc*)(uintptr_t)i->result_descs;
        int num_workers = total_gpus - 1;
        unsigned int rseq = (unsigned int)i->seq_counter; // per-step (position+1), bumped by decode loop
        // yef5.2.8 double-buffer: spin on + read the (rseq%2) slot (the worker signals/writes the
        // same slot). stride = gupd. Always-on (correct for both the gpu0_pre reclaim and async).
        const unsigned int dbuf_slot = rseq & 1u;
        const unsigned long dbuf_off = (unsigned long)dbuf_slot * (unsigned long)gupd;
        // Acquire-spin each worker's result sentinel (block 0 / thread 0; bare spin +
        // s_sleep, NO in-loop stores — §11.4 vscnt-drain-safe, mirrors op_moe_ffn_remote:381).
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int w = 0; w < num_workers; w++) {
                while (braidinfer::rdna3::sentinel_spin_load_u32(
                           ((const unsigned int*)descs[w].result_sentinel) + dbuf_slot) != rseq) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
        braidinfer::rdna3::atomic_block_barrier(gbs);
        // §11.18(b) consumer-side invalidation: discard stale L0/L1 lines for the peer
        // UC-VRAM (gl2_inv n/a on gfx11; the glc+dlc sentinel load already confirmed the
        // producer write reached VRAM). Mirrors op_moe_ffn_remote:409-415.
        if ((threadIdx.x & 63) == 0) asm volatile("buffer_gl0_inv" ::: "memory");
        if (threadIdx.x == 0) asm volatile("buffer_gl1_inv" ::: "memory");
        __syncthreads();
        // Sum the workers' UC-VRAM outputs (cross-GPU peer-read, fresh).
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gupd;
             idx += gridDim.x * blockDim.x) {
            float sum = 0.0f;
            for (int w = 0; w < num_workers; w++) {
                sum += descs[w].local_output_uc[dbuf_off + idx]; // yef5.2.8: read the (rseq%2) slot
            }
            final_output[idx] = sum;
        }
        braidinfer::rdna3::atomic_block_barrier(gbs);
        return;
    }

    // Sum gupd elements per slot, slot stride = hs.
    // P3 (braidinfer-4n5.7): GPU 0 no longer contributes an expert output to
    // output_slots[0] (it runs no experts). Start sum at g=1 to skip the
    // stale slot 0 and accumulate only worker outputs (slots 1..total_gpus).
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gupd;
         idx += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int g = 1; g < total_gpus; g++) {
            sum += output_slots[g * hs + idx];
        }
        final_output[idx] = sum;
    }
    braidinfer::rdna3::atomic_block_barrier(gbs);
}
