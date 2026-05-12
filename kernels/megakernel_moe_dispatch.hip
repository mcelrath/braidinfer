// OP_MOE_DISPATCH (post-Phase-7 split): GPU 0's local-expert PRE stage of MoE.
//
// Split design (epic braidinfer-0hu Phase 7) restores parallelism between GPU 0
// local-expert compute and worker MoE compute. The CPU sequence is:
//   1. CPU fires OP_MOE_FFN_REMOTE on each worker (async, no wait)
//   2. CPU fires GPU 0 batch containing OP_MOE_DISPATCH (this op) — zero + GPU 0 experts.
//      This runs CONCURRENTLY with workers (workers on GPUs 1+, GPU 0 here).
//   3. CPU waits for all worker acks AND for GPU 0's batch ack to drain.
//   4. CPU fires next GPU 0 batch starting with OP_MOE_DISPATCH_POST — sums output_slots.
//
// Restored parallelism: ~25% throughput recovery on multi-GPU MoE decode.
// Cost: 1 extra dispatch round-trip per MoE layer (~10 µs).
//
// This op (OP_MOE_DISPATCH, opcode 34) now does:
//   1. Zeros GPU 0's slot in output_slots[0..gupd]
//   2. Runs GPU 0's local experts (entries[eid].gate_up_ptr != 0)
//
// The sum step is a separate OP_MOE_DISPATCH_POST (opcode 45). See op_moe_dispatch_post.
//
// work_queue and seq_counter fields are unused but preserved for ABI stability.
//
// Instruction layout (17 u64 words):
//   inst[1]  = work_queue_ptr       (UNUSED — preserved for layout)
//   inst[2]  = output_slots_ptr     (float*, GPU 0 VRAM/UC, [total_gpus * hs])
//   inst[3]  = final_output_ptr     (float*, GPU 0 VRAM, [hs] accumulated result)
//   inst[4]  = expert_ids_ptr       (int32_t*, host-mapped — OP_MOE_GATE output)
//   inst[5]  = expert_weights_ptr   (float*, host-mapped — OP_MOE_GATE output)
//   inst[6]  = seq_counter_ptr      (UNUSED — preserved for layout)
//   inst[7]  = (num_workers << 32) | hidden_size   [hidden_size for output slot stride]
//   inst[8]  = (layer_idx << 32) | num_active (k)
//   inst[9]  = (expert_intermediate_size << 32) | has_gate_proj
//   inst[10] = activation_ptr       (float*, GPU 0 VRAM — [gate_up_in_dim] expert activation)
//   inst[11] = gpu0_config_ptr      (MoeWorkerConfig**, GPU 0 VRAM array indexed by layer_idx)
//   inst[12] = scratch_gate_ptr     (float*, GPU 0 VRAM [eis])
//   inst[13] = scratch_up_ptr       (float*, GPU 0 VRAM [eis])
//   inst[14] = scratch_act_ptr      (float*, GPU 0 VRAM [eis])
//   inst[15] = total_gpus           (num_workers + 1, for output_slots indexing)
//   inst[16] = gate_up_in_dim       (expert input dim; = hidden_size for standard MoE,
//                                    = moe_latent_size for Nemotron-H)
#include "moe_work_queue.h"
#include "moe_expert_ops.h"
#include "watchdog.h"
#include "rdna3_sync.h"

__device__ __attribute__((noinline)) void op_moe_dispatch(const u64* inst, cooperative_groups::grid_group& grid,
                                                            braidinfer::rdna3::GridBarrierState* gbs,
                                                            WatchdogState* watchdog = nullptr) {
    (void)watchdog;
    const MoeDispatchInst* i = (const MoeDispatchInst*)inst;
    float* output_slots = (float*)(uintptr_t)i->output_slots;
    float* final_output = (float*)(uintptr_t)i->final_output;
    const int32_t* src_expert_ids = (const int32_t*)(uintptr_t)i->expert_ids;
    const float* src_expert_weights = (const float*)(uintptr_t)i->expert_weights;
    int hs = (int)(i->num_workers_hs & 0xFFFFFFFF);  // hidden_size (output slot stride)
    int layer_idx = (int)(i->layer_k >> 32);
    int k = (int)(i->layer_k & 0xFFFFFFFF);
    int eis = (int)(i->eis_gate >> 32);
    int has_gate = (int)(i->eis_gate & 0xFFFFFFFF);
    const float* activation = (const float*)(uintptr_t)i->activation;
    // layer_config_ptrs is MoeWorkerConfig**: array of per-layer config ptrs indexed by layer_idx
    MoeWorkerConfig* gpu0_config = ((MoeWorkerConfig**)(uintptr_t)i->layer_config_ptrs)[layer_idx];
    float* scratch_gate = (float*)(uintptr_t)i->scratch_gate;
    float* scratch_up = (float*)(uintptr_t)i->scratch_up;
    float* scratch_act = (float*)(uintptr_t)i->scratch_act;
    int total_gpus = (int)i->num_gpus;
    // gate_up_in_dim: expert input dimension (= hs for standard MoE, moe_latent_size for Nemotron-H)
    int gupd = (int)i->gate_up_in_dim;
    if (gupd == 0) gupd = hs;  // backward compat: default to hidden_size

    // GPU 0's output slot is at index 0 in output_slots (stride = hs for consistent slot layout).
    // §11.4 HAZARD avoidance (braidinfer-pky.2 Phase 0b followup): the
    // per-expert accumulation stages into `gpu0_acc` (cached local VRAM)
    // and does ONE barrierless copy to UC `gpu0_output` at the end. The
    // intermediate coop_zero / coop_weighted_acc barriers now operate on
    // cached memory — no PCIe-write-before-barrier hazard. Mirrors the
    // worker-side pattern in op_moe_ffn_remote (megakernel_moe.hip:401).
    float* gpu0_output = output_slots; // [0..gupd) — final UC destination
    float* gpu0_acc = (float*)(uintptr_t)i->gpu0_acc; // [0..gupd) — cached local

    // 1. Zero the cached accumulator (not the UC slot).
    // Do NOT zero final_output — the sum loop overwrites it completely.
    // Do NOT zero worker output slots — those were P2P-written by workers and the
    // CPU has waited for their ack before firing this op.
    coop_zero(gpu0_acc, gupd, grid, gbs);

    // (Worker-trigger section removed — CPU now drives OP_MOE_FFN_REMOTE on workers
    //  before this op fires, and CPU waits for worker ack before this batch dispatch.)

    // 3. GPU 0 computes its local experts (all blocks cooperate).
    // Workers on GPUs 1-3 have already finished their experts and written
    // to output_slots[g * hs] for g >= 1 — guaranteed because CPU waited for
    // worker ack before firing this op. UC mapping on output_slots ensures
    // no stale L2 entries on GPU 0 reads.
    for (int j = 0; j < k; j++) {
        int32_t eid = src_expert_ids[j];
        float ew = src_expert_weights[j];
        if (eid < 0 || eid >= 512) continue;

        MoeExpertEntry& entry = gpu0_config->entries[eid];
        if (entry.gate_up_ptr == 0) continue; // not on GPU 0

        const unsigned char* gate_up_data = (const unsigned char*)entry.gate_up_ptr;
        const unsigned char* down_data = (const unsigned char*)entry.down_ptr;

        if (gpu0_config->weight_format == MOE_WEIGHT_FORMAT_RNF4G128) {
            if (has_gate) {
                const unsigned char* up_data = gate_up_data +
                    (long long)eis * gpu0_config->gate_up_row_stride;
                coop_gemv_rnf4_fused_gate_up(scratch_gate, gate_up_data,
                                               scratch_up, up_data,
                                               activation, eis, gupd, grid, gbs);
                coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid, gbs);
            } else {
                coop_gemv_rnf4(scratch_up, gate_up_data, activation, eis, gupd, grid, gbs);
                coop_relu_squared(scratch_act, scratch_up, eis, grid, gbs);
            }
            // down: eis → gupd
            coop_gemv_rnf4(scratch_gate, down_data, scratch_act, gupd, eis, grid, gbs);
        } else {
            // PCG32Q4 (default)
            if (has_gate) {
                const unsigned char* up_data = gate_up_data +
                    (long long)eis * gpu0_config->gate_up_row_stride;
                coop_gemv_pcg32_fused_gate_up(scratch_gate, gate_up_data,
                                               scratch_up, up_data,
                                               activation, eis, gupd, grid, gbs);
                coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid, gbs);
            } else {
                coop_gemv_pcg32(scratch_up, gate_up_data, activation, eis, gupd, grid, gbs);
                coop_relu_squared(scratch_act, scratch_up, eis, grid, gbs);
            }
            // down: eis → gupd
            coop_gemv_pcg32(scratch_gate, down_data, scratch_act, gupd, eis, grid, gbs);
        }

        coop_weighted_acc(gpu0_acc, scratch_gate, ew, gupd, grid, gbs);
    }

    // PRE complete: GPU 0 has accumulated its local experts into `gpu0_acc`.
    // Final step: barrierless copy gpu0_acc → output_slots[0..gupd] (cross-
    // type write: cached → UC). No trailing atomic_block_barrier — kernel
    // exit / persistent_worker's next batch-boundary buffer_gl1_inv handles
    // visibility. The __threadfence pair flushes (a) cached stores before
    // copy starts and (b) UC stores before kernel exit. Mirrors
    // op_moe_ffn_remote's exit pattern (megakernel_moe.hip:425-432).
    // Sum is performed by OP_MOE_DISPATCH_POST after the CPU has wait_ack'd
    // workers + GPU 0 PRE — POST reads `output_slots`, so the kernel-exit
    // fence between this op and POST guarantees visibility.
    (void)final_output;
    (void)total_gpus;

    __threadfence();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gupd;
         idx += gridDim.x * blockDim.x) {
        gpu0_output[idx] = gpu0_acc[idx];
    }
    __threadfence();
}

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
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < gupd;
         idx += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int g = 0; g < total_gpus; g++) {
            sum += output_slots[g * hs + idx];
        }
        final_output[idx] = sum;
    }
    braidinfer::rdna3::atomic_block_barrier(gbs);
}
