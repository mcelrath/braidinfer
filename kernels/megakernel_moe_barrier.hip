
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

// OP_MOE_DISPATCH: GPU-initiated MoE expert dispatch via persistent worker kernels.
//
// Instruction layout:
//   inst[1] = work_queue_ptr     (MoeWorkItem*, host-mapped)
//   inst[2] = output_slots_ptr   (float*, GPU 0 VRAM, [num_workers * hs])
//   inst[3] = final_output_ptr   (float*, GPU 0 VRAM, [hs] — accumulated result)
//   inst[4] = num_workers
//   inst[5] = hidden_size
//   inst[6] = seq_num            (current sequence number to write)
//   inst[7] = layer_idx
//
// All blocks participate. Block 0 thread 0 triggers workers and polls acks.
// After ack, all blocks cooperatively sum output_slots → final_output.
#include "moe_work_queue.h"

__device__ void op_moe_dispatch(const u64* inst, cooperative_groups::grid_group& grid) {
    volatile MoeWorkItem* wq = (volatile MoeWorkItem*)(uintptr_t)inst[1];
    float* output_slots = (float*)(uintptr_t)inst[2];
    float* final_output = (float*)(uintptr_t)inst[3];
    int num_workers = (int)inst[4];
    int hs = (int)inst[5];
    uint32_t seq = (uint32_t)inst[6];
    uint32_t layer_idx = (uint32_t)inst[7];

    // 1. Zero output_slots and final_output (all blocks cooperate)
    int total = num_workers * hs + hs;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        if (i < num_workers * hs)
            output_slots[i] = 0.0f;
        else
            final_output[i - num_workers * hs] = 0.0f;
    }
    grid.sync();

    // 2. Block 0 thread 0: trigger workers by writing seq_num
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        wq->layer_idx = layer_idx;
        __threadfence_system();
        wq->seq_num = seq;

        // 3. Poll ack_flags until all workers done
        for (int w = 0; w < num_workers; w++) {
            while (wq->ack_flags[w] != seq) {
                __builtin_amdgcn_s_sleep(0);
            }
        }
    }
    grid.sync();

    // 4. Sum output_slots into final_output (all blocks cooperate)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hs;
         i += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int w = 0; w < num_workers; w++) {
            sum += output_slots[w * hs + i];
        }
        final_output[i] = sum;
    }
    grid.sync();
}
