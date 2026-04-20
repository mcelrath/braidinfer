// Persistent cooperative worker kernel: polls host-mapped work queue for operations.
// Replaces the megakernel — same op implementations, but CPU-scheduled via memory writes.
// One instance per GPU, launched once at init, runs for model lifetime.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"

// Include all megakernel op implementations (same as megakernel.hip)
#include "megakernel_ops.hip"
#include "megakernel_moe.hip"
#include "megakernel_moe_dispatch.hip"

// Work queue for CPU → GPU communication (host-mapped memory).
// CPU writes an instruction + increments seq_num. Worker polls seq_num.
#define MAX_BATCH_INSTRUCTIONS 256

struct WorkerQueue {
    volatile uint32_t seq_num;      // monotonic counter, triggers worker
    volatile uint32_t shutdown;     // 1 = exit
    uint32_t num_instructions;      // instructions in this batch (1..64)
    uint32_t _pad;
    uint64_t inst[MAX_BATCH_INSTRUCTIONS * INST_SIZE_WORDS]; // instruction batch
    volatile uint32_t ack;          // worker writes seq_num when done
    volatile uint32_t done;         // kernel writes 1 when exiting (for Drop polling)
    volatile uint32_t progress_pc;  // worker writes pc before each instruction (for timeout diagnosis)
    uint32_t _pad2;
};

extern "C" __global__ __launch_bounds__(256, 2) void persistent_worker(
    volatile WorkerQueue* queue
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("persistent_worker: started, queue=%p\n", (void*)queue);
    }

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    uint32_t last_seq = 0;

    while (true) {
        // Single-thread poll to avoid PCIe storm
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (queue->shutdown) {
                    // Sentinel: set seq to max so all threads see shutdown
                    queue->seq_num = 0xFFFFFFFFu;
                    break;
                }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        grid.sync();

        uint32_t seq = queue->seq_num;
        if (seq == 0xFFFFFFFFu) {
            if (threadIdx.x == 0 && blockIdx.x == 0) queue->done = 1;
            return;
        }

        uint32_t num_inst = queue->num_instructions;
        if (num_inst == 0) num_inst = 1; // backward compat: 0 means 1

        // Process batch of instructions with grid.sync() between each
        for (uint32_t pc = 0; pc < num_inst; pc++) {
        // Report progress so CPU can identify stuck instruction on timeout.
        if (threadIdx.x == 0 && blockIdx.x == 0) queue->progress_pc = pc;
        // Copy instruction to shared memory (avoids PCIe reads in virtual block loop)
        __shared__ u64 sinst[INST_SIZE_WORDS];
        const u64* src = (const u64*)(queue->inst + pc * INST_SIZE_WORDS);
        if (threadIdx.x < INST_SIZE_WORDS) {
            sinst[threadIdx.x] = src[threadIdx.x];
        }
        __syncthreads();
        const u64* inst = sinst;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);

        if (opcode == OP_BARRIER) {
            grid.sync();
        } else if (opcode == OP_MOE_FFN) {
            op_moe_ffn(inst, grid);
        } else if (opcode == OP_MOE_DISPATCH) {
            op_moe_dispatch(inst, grid);
        } else if (opcode == OP_LINEAR_PROJ_PCG32) {
            op_linear_proj_pcg32_full(inst, grid_x);
            grid.sync();
        } else if (opcode == OP_LINEAR_PROJ_RNF4) {
            op_linear_proj_rnf4_full(inst, grid_x);
            grid.sync();
        } else {
            // Virtual block loop
            for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
                switch (opcode) {
                    case OP_RMSNORM:      op_rmsnorm(inst, vb); break;
                    case OP_LINEAR_PROJ:  op_linear_proj(inst, vb); break;
                    case OP_CONV1D:       op_conv1d(inst, vb); break;
                    case OP_GDN_GATE:     op_gdn_gate(inst, vb); break;
                    case OP_GDN_RECUR:    op_gdn_recurrent(inst, vb); break;
                    case OP_RMSNORM_GATE: op_rmsnorm_gated(inst, vb); break;
                    case OP_RESIDUAL_ADD: op_residual_add(inst, vb); break;
                    case OP_QK_NORM:      op_qk_norm(inst, vb); break;
                    case OP_MROPE:        op_mrope(inst, vb); break;
                    case OP_GQA_ATTN:     op_gqa_attn(inst, vb); break;
                    case OP_OUTPUT_GATE:  op_output_gate(inst, vb); break;
                    case OP_FFN_GATE_UP:  op_ffn_gate_up(inst, vb); break;
                    case OP_FFN_DOWN_RES: op_ffn_down_res(inst, vb); break;
                    case OP_EMBEDDING:    op_embedding(inst, vb); break;
                    case OP_LM_HEAD:      op_lm_head(inst, vb); break;
                    case OP_D2D_COPY:     op_d2d_copy(inst, vb); break;
                    case OP_ATTN_PAGED:   op_attn_paged(inst, vb); break;
                    case OP_ATTN_PREFILL: op_attn_prefill(inst, vb); break;
                    case OP_DEINTERLEAVE: op_deinterleave(inst, vb); break;
                    case OP_KV_QUANTIZE:  op_kv_quantize(inst, vb); break;
                    case OP_ATTN_PAGED_Q: op_attn_paged_quant(inst, vb); break;
                    case OP_MOE_GATE:    op_moe_gate(inst, vb); break;
                    case OP_LINEAR_PROJ_RNF4:  op_linear_proj_rnf4(inst, vb); break;
                    case OP_LINEAR_PROJ_PCG32: op_linear_proj_pcg32(inst, vb); break;
                    case OP_RMSNORM_WX:  op_rmsnorm_wx(inst, vb); break;
                    case OP_SILU_MUL:    op_silu_mul(inst, vb); break;
                    case OP_FFN_GATE_UP_RNF4:  op_ffn_gate_up_rnf4(inst, vb); break;
                    case OP_FFN_DOWN_RES_RNF4: op_ffn_down_res_rnf4(inst, vb); break;
                    case OP_SIGMOID_WEIGHTED_ADD: op_sigmoid_weighted_add(inst, vb); break;
                    case OP_SCALE_ADD:   op_scale_add(inst, vb); break;
                    case OP_RELU_SQ:     op_relu_sq(inst, vb); break;
                    case OP_MAMBA2_CONV1D:    op_mamba2_conv1d(inst, vb); break;
                    case OP_SSM_UPDATE:       op_ssm_update(inst, vb); break;
                    case OP_MAMBA2_NORM_GATED: op_mamba2_norm_gated(inst, vb); break;
                    case OP_CONV1D_3X:         op_conv1d_3x(inst, vb); break;
                    default: break;
                }
            }
            grid.sync();
        }

        } // end for (pc = 0; pc < num_inst; pc++)

        // Ack: flush ALL threads' writes to system-visible memory, then write ack.
        // grid.sync() ensures all threads completed (device-scope). But P2P readers on
        // GPUs 1-3 bypass GPU 0's L2 and read VRAM directly. grid.sync() alone does NOT
        // flush L2 → VRAM for P2P visibility. __threadfence_system() from EACH thread
        // ensures that thread's cache lines are written back to system-visible memory.
        // Must be called by ALL threads (not just thread 0,0) to cover all written cache lines.
        __threadfence_system();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            queue->ack = seq;
        }
        last_seq = seq;
    }
}
