// OP_MOE_DISPATCH: GPU-initiated MoE dispatch with GPU 0 expert execution.
//
// GPU 0 computes its local experts WHILE workers on GPUs 1-3 compute theirs.
// After GPU 0 finishes its experts, it polls worker acks, then sums all results.
//
// Instruction layout (17 u64 words):
//   inst[1]  = work_queue_ptr       (MoeWorkItem*, host-mapped)
//   inst[2]  = output_slots_ptr     (float*, GPU 0 VRAM, [total_gpus * hs])
//   inst[3]  = final_output_ptr     (float*, GPU 0 VRAM, [hs] accumulated result)
//   inst[4]  = expert_ids_ptr       (int32_t*, host-mapped — OP_MOE_GATE output)
//   inst[5]  = expert_weights_ptr   (float*, host-mapped — OP_MOE_GATE output)
//   inst[6]  = seq_counter_ptr      (volatile uint32_t*, host-mapped)
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

__device__ __attribute__((noinline)) void op_moe_dispatch(const u64* inst, cooperative_groups::grid_group& grid) {
    const MoeDispatchInst* i = (const MoeDispatchInst*)inst;
    volatile MoeWorkItem* wq = (volatile MoeWorkItem*)(uintptr_t)i->work_queue;
    float* output_slots = (float*)(uintptr_t)i->output_slots;
    float* final_output = (float*)(uintptr_t)i->final_output;
    const int32_t* src_expert_ids = (const int32_t*)(uintptr_t)i->expert_ids;
    const float* src_expert_weights = (const float*)(uintptr_t)i->expert_weights;
    volatile uint32_t* seq_counter = (volatile uint32_t*)(uintptr_t)i->seq_counter;
    int num_workers = (int)(i->num_workers_hs >> 32);
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
    // Do NOT zero worker output slots — GPU 0 would cache those in L2.
    coop_zero(gpu0_output, gupd, grid);

    // 2. Block 0 thread 0: populate work queue and trigger workers
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int j = 0; j < k && j < MOE_MAX_ACTIVE_EXPERTS; j++) {
            wq->expert_ids[j] = src_expert_ids[j];
            wq->expert_weights[j] = src_expert_weights[j];
        }
        wq->batch_size = 1;  // decode: single token per dispatch
        wq->layer_idx = layer_idx;
        wq->num_active = k;
        wq->hidden_size = hs;
        wq->expert_intermediate_size = eis;
        wq->has_gate_proj = has_gate;
        wq->gate_up_in_dim = gupd;
        wq->activation_ptr = (uint64_t)activation;  // GPU 0 VRAM; workers P2P-read after __threadfence_system()
        wq->output_slots_ptr = (uint64_t)output_slots;
        wq->num_workers = num_workers;

        // ack_flags are indexed by GPU ID; workers are GPUs 1..num_workers
        for (int w = 1; w <= num_workers; w++)
            wq->ack_flags[w] = 0;

        // Flush GPU 0 L2 to HBM so workers can P2P-read activation_ptr (VRAM) directly.
        // wq field writes (GART) also bypass L2 → visible before seq_num write.
        __threadfence_system();

        uint32_t seq = *seq_counter + 1;
        *seq_counter = seq;
        wq->seq_num = seq;  // triggers workers
    }
    grid.sync();

    // Read seq for later ack polling
    uint32_t seq = *seq_counter;

    // 3. GPU 0 computes its local experts (all blocks cooperate).
    // Workers on GPUs 1-3 are computing in parallel.
    for (int j = 0; j < k; j++) {
        int32_t eid = src_expert_ids[j];
        float ew = src_expert_weights[j];
        if (eid < 0 || eid >= 512) continue;

        MoeExpertEntry& entry = gpu0_config->entries[eid];
        if (entry.gate_up_ptr == 0) continue; // not on GPU 0

        const unsigned char* gate_up_data = (const unsigned char*)entry.gate_up_ptr;
        const unsigned char* down_data = (const unsigned char*)entry.down_ptr;

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

        // down: eis → gupd (= gate_up_in_dim, not necessarily hidden_size)
        coop_gemv_pcg32(scratch_gate, down_data, scratch_act, gupd, eis, grid);
        coop_weighted_acc(gpu0_output, scratch_gate, ew, gupd, grid);
    }

    // 4. Block 0 thread 0: poll worker acks (GPU 0's experts are done by now)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int w = 0; w < num_workers; w++) {
            while (wq->ack_flags[w + 1] != seq) {
                __builtin_amdgcn_s_sleep(0);
            }
        }
    }
    grid.sync();

    // Acquire fence: ensure all worker P2P writes to output_slots are visible before sum.
    // Workers use __threadfence_system() before writing ack_flags; this matching fence on
    // GPU 0 side ensures the HIP memory model guarantees ordering (not just empirical RDNA3 behavior).
    __threadfence_system();

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
