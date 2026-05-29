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
