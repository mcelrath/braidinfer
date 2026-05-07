// OP_MOE_DISPATCH (post-unified-worker): GPU 0's local-expert + sum stage of MoE.
//
// In the unified-worker design (epic braidinfer-0hu Phase 2-5), the CPU emits
// OP_MOE_FFN_REMOTE on each worker GPU's persistent_worker BEFORE this op runs,
// and waits for all workers to ack before firing the GPU 0 batch containing
// this op. By the time op_moe_dispatch executes, all worker output_slots are
// already populated and visible (workers' P2P writes + UC mapping).
//
// This op now:
//   1. Zeros GPU 0's slot in output_slots
//   2. Runs GPU 0's local experts (entries[eid].gate_up_ptr != 0)
//   3. Sums output_slots across all GPUs into final_output
//
// Worker-trigger and ack-poll sections have been removed — CPU orchestrates
// the worker dispatch/wait around this op (see decode.rs / moe_p2p.rs).
// work_queue and seq_counter fields are now unused but preserved in the
// instruction layout for ABI stability (no kernel-side reads).
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

__device__ __attribute__((noinline)) void op_moe_dispatch(const u64* inst, cooperative_groups::grid_group& grid, WatchdogState* watchdog = nullptr) {
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

    // GPU 0's output slot is at index 0 in output_slots (stride = hs for consistent slot layout)
    float* gpu0_output = output_slots; // [0..gupd)

    // 1. Zero GPU 0's output slot.
    // Do NOT zero final_output — the sum loop overwrites it completely.
    // Do NOT zero worker output slots — those were P2P-written by workers and the
    // CPU has waited for their ack before firing this op.
    coop_zero(gpu0_output, gupd, grid);

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
                                               activation, eis, gupd, grid);
                coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid);
            } else {
                coop_gemv_rnf4(scratch_up, gate_up_data, activation, eis, gupd, grid);
                coop_relu_squared(scratch_act, scratch_up, eis, grid);
            }
            // down: eis → gupd
            coop_gemv_rnf4(scratch_gate, down_data, scratch_act, gupd, eis, grid);
        } else {
            // PCG32Q4 (default)
            if (has_gate) {
                const unsigned char* up_data = gate_up_data +
                    (long long)eis * gpu0_config->gate_up_row_stride;
                coop_gemv_pcg32_fused_gate_up(scratch_gate, gate_up_data,
                                               scratch_up, up_data,
                                               activation, eis, gupd, grid);
                coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid);
            } else {
                coop_gemv_pcg32(scratch_up, gate_up_data, activation, eis, gupd, grid);
                coop_relu_squared(scratch_act, scratch_up, eis, grid);
            }
            // down: eis → gupd
            coop_gemv_pcg32(scratch_gate, down_data, scratch_act, gupd, eis, grid);
        }

        coop_weighted_acc(gpu0_output, scratch_gate, ew, gupd, grid);
    }

    // (Worker ack-poll removed — CPU has already waited for all workers before
    //  firing this batch.)

    // Acquire fence on GPU 0's writes to gpu0_output (other GPU output_slots are
    // already visible via UC mapping). Workers' P2P writes were ack'd to CPU
    // before this op; UC ensures no stale L2 on GPU 0's read of those slots.
    __threadfence();

    // 5. Sum all output_slots into final_output (gupd elements per slot, slot stride = hs).
    // GPU 0 slot (index 0) is in local VRAM — no stale L2 issue.
    // Worker slots (indices 1..num_workers) were P2P-written — GPU 0 reads fresh VRAM (L2 miss).
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < gupd;
         i += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int g = 0; g < total_gpus; g++) {
            sum += output_slots[g * hs + i];
        }
        final_output[i] = sum;
    }
    grid.sync();
}
