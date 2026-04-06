
// OP_BARRIER: Park megakernel, signal CPU to dispatch multi-GPU MoE, wait for resume.
//
// Instruction layout:
//   inst[1] = barrier_flag_ptr  (mapped host uint32_t: GPU writes layer_idx+1 to signal CPU)
//   inst[2] = resume_flag_ptr   (mapped host uint32_t: CPU writes 1; GPU resets to 0)
//   inst[3] = layer_idx         (which MoE layer; passed to CPU to look up dispatch params)
//
// Only block 0 executes (grid_x=1). Other blocks skip the vb loop and hit grid.sync().
// Block 0 writes barrier_flag, spins on resume_flag, then exits and hits grid.sync().
// Expert FFN results are written by CPU-dispatched kernels before resume is signaled;
// megakernel reads them fresh (no stale L2 — no prior read of those addresses).
__device__ void op_barrier(const u64* inst, u32 vb) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    volatile uint32_t* barrier_flag = (volatile uint32_t*)(uintptr_t)inst[1];
    volatile uint32_t* resume_flag  = (volatile uint32_t*)(uintptr_t)inst[2];
    uint32_t layer_idx = (uint32_t)inst[3];

    *barrier_flag = layer_idx + 1;
    while (*resume_flag == 0) {
        __builtin_amdgcn_s_sleep(7);
    }
    *resume_flag = 0;
}

// OP_MOE_DISPATCH: GPU-initiated MoE dispatch with GPU 0 expert execution.
//
// GPU 0 computes its local experts WHILE workers on GPUs 1-3 compute theirs.
// After GPU 0 finishes its experts, it polls worker acks, then sums all results.
//
// Instruction layout (17 u64 words):
//   inst[1]  = work_queue_ptr       (MoeWorkItem*, host-mapped)
//   inst[2]  = output_slots_ptr     (float*, GPU 0 VRAM, [total_gpus * hs])
//   inst[3]  = final_output_ptr     (float*, GPU 0 VRAM, [hs] — accumulated result)
//   inst[4]  = expert_ids_ptr       (int32_t*, host-mapped — OP_MOE_GATE output)
//   inst[5]  = expert_weights_ptr   (float*, host-mapped — OP_MOE_GATE output)
//   inst[6]  = seq_counter_ptr      (volatile uint32_t*, host-mapped)
//   inst[7]  = (num_workers << 32) | hidden_size   [num_workers = GPUs 1..N-1]
//   inst[8]  = (layer_idx << 32) | num_active (k)
//   inst[9]  = (expert_intermediate_size << 32) | has_gate_proj
//   inst[10] = activation_ptr       (float*, GPU 0 VRAM — normed activation)
//   inst[11] = gpu0_config_ptr      (MoeWorkerConfig*, GPU 0 device memory)
//   inst[12] = scratch_gate_ptr     (float*, GPU 0 VRAM [eis])
//   inst[13] = scratch_up_ptr       (float*, GPU 0 VRAM [eis])
//   inst[14] = scratch_act_ptr      (float*, GPU 0 VRAM [eis])
//   inst[15] = total_gpus           (num_workers + 1, for output_slots indexing)
#include "moe_work_queue.h"
#include "moe_expert_ops.h"

__device__ void op_moe_dispatch(const u64* inst, cooperative_groups::grid_group& grid) {
    volatile MoeWorkItem* wq = (volatile MoeWorkItem*)(uintptr_t)inst[1];
    float* output_slots = (float*)(uintptr_t)inst[2];
    float* final_output = (float*)(uintptr_t)inst[3];
    const int32_t* src_expert_ids = (const int32_t*)(uintptr_t)inst[4];
    const float* src_expert_weights = (const float*)(uintptr_t)inst[5];
    volatile uint32_t* seq_counter = (volatile uint32_t*)(uintptr_t)inst[6];
    int num_workers = (int)(inst[7] >> 32);
    int hs = (int)(inst[7] & 0xFFFFFFFF);
    int layer_idx = (int)(inst[8] >> 32);
    int k = (int)(inst[8] & 0xFFFFFFFF);
    int eis = (int)(inst[9] >> 32);
    int has_gate = (int)(inst[9] & 0xFFFFFFFF);
    const float* activation = (const float*)(uintptr_t)inst[10];
    // inst[11] is MoeWorkerConfig**: array of per-layer config ptrs indexed by layer_idx
    MoeWorkerConfig* gpu0_config = ((MoeWorkerConfig**)(uintptr_t)inst[11])[layer_idx];
    float* scratch_gate = (float*)(uintptr_t)inst[12];
    float* scratch_up = (float*)(uintptr_t)inst[13];
    float* scratch_act = (float*)(uintptr_t)inst[14];
    int total_gpus = (int)inst[15];

    // GPU 0's output slot is at index 0 in output_slots
    float* gpu0_output = output_slots; // [0..hs)

    // 1. Zero final_output and GPU 0's output slot (all blocks cooperate).
    // Do NOT zero worker output slots — GPU 0 would cache those in L2.
    coop_zero(final_output, hs, grid);
    coop_zero(gpu0_output, hs, grid);

    // 2. Block 0 thread 0: populate work queue and trigger workers
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int j = 0; j < k && j < MOE_MAX_ACTIVE_EXPERTS; j++) {
            wq->expert_ids[j] = src_expert_ids[j];
            wq->expert_weights[j] = src_expert_weights[j];
        }
        wq->layer_idx = layer_idx;
        wq->num_active = k;
        wq->hidden_size = hs;
        wq->expert_intermediate_size = eis;
        wq->has_gate_proj = has_gate;
        wq->activation_ptr = (uint64_t)activation;
        wq->output_slots_ptr = (uint64_t)output_slots;
        wq->num_workers = num_workers;

        for (int w = 0; w < num_workers; w++)
            wq->ack_flags[w] = 0;

        // Ensure GPU 0 VRAM zeros are committed before workers read output_slots
        __threadfence();

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
        if (eid < 0 || eid >= 256) continue;

        MoeExpertEntry& entry = gpu0_config->entries[eid];
        if (entry.gate_up_ptr == 0) continue; // not on GPU 0

        const unsigned char* gate_up_data = (const unsigned char*)entry.gate_up_ptr;
        const unsigned char* down_data = (const unsigned char*)entry.down_ptr;

        if (has_gate) {
            coop_gemv_pcg32(scratch_gate, gate_up_data, activation, eis, hs, grid);
            const unsigned char* up_data = gate_up_data +
                (long long)eis * gpu0_config->gate_up_row_stride;
            coop_gemv_pcg32(scratch_up, up_data, activation, eis, hs, grid);
            coop_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid);
        } else {
            coop_gemv_pcg32(scratch_up, gate_up_data, activation, eis, hs, grid);
            coop_relu_squared(scratch_act, scratch_up, eis, grid);
        }

        coop_gemv_pcg32(scratch_gate, down_data, scratch_act, hs, eis, grid);
        coop_weighted_acc(gpu0_output, scratch_gate, ew, hs, grid);
    }

    // 4. Block 0 thread 0: poll worker acks (GPU 0's experts are done by now)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int w = 0; w < num_workers; w++) {
            while (wq->ack_flags[w] != seq) {
                __builtin_amdgcn_s_sleep(0);
            }
        }
    }
    grid.sync();

    // Acquire fence: ensure all worker P2P writes to output_slots are visible before sum.
    // Workers use __threadfence_system() before writing ack_flags; this matching fence on
    // GPU 0 side ensures the HIP memory model guarantees ordering (not just empirical RDNA3 behavior).
    __threadfence_system();

    // 5. Sum all output_slots into final_output (all blocks cooperate).
    // GPU 0 slot (index 0) is in local VRAM — no stale L2 issue.
    // Worker slots (indices 1..num_workers) were P2P-written — GPU 0 reads fresh VRAM (L2 miss).
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hs;
         i += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int g = 0; g < total_gpus; g++) {
            sum += output_slots[g * hs + i];
        }
        final_output[i] = sum;
    }
    grid.sync();
}
