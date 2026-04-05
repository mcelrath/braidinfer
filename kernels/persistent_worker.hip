// Persistent cooperative worker kernel: polls host-mapped work queue for operations.
// Replaces the megakernel — same op implementations, but CPU-scheduled via memory writes.
// One instance per GPU, launched once at init, runs for model lifetime.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"

// Include all megakernel op implementations (same as megakernel.hip)
#include "megakernel_ops.hip"
#include "megakernel_moe.hip"

// Per-row Q4 PcG32 GEMV for OP_EXPERT_FFN compound instruction.
// output[row] = dot(weight_row, input). One block per row in virtual block loop.
__device__ void op_linear_proj_pcg32_row(
    float* output, const unsigned char* weight, const float* input,
    int row, int out_dim, int in_dim
) {
    if (row >= out_dim) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
    extern __shared__ float shared[];

    float acc = 0.0f;
    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp = row_data + g * group_bytes;
        float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
        float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
        int base = g * group_size;
        int count = min(group_size, in_dim - base);
        for (int i = 0; i < count; i += 2) {
            unsigned char byte = gp[i / 2];
            float v0 = (float)(byte & 0xF) * sc + mn;
            float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
            acc += v0 * input[base + i] + v1 * input[base + i + 1];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if ((tid & 31) == 0) shared[tid >> 5] = acc;
    __syncthreads();
    if (tid < 8) {
        acc = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) output[row] = acc;
    }
    __syncthreads();
}

// Work queue for CPU → GPU communication (host-mapped memory).
// CPU writes an instruction + increments seq_num. Worker polls seq_num.
struct WorkerQueue {
    volatile uint32_t seq_num;      // monotonic counter, 0 = no work
    volatile uint32_t shutdown;     // 1 = exit
    uint32_t _pad[2];
    uint64_t inst[INST_SIZE_WORDS]; // single instruction to execute
    volatile uint32_t ack;          // worker writes seq_num when done
    uint32_t _pad2[3];
};

extern "C" __global__ __launch_bounds__(256) void persistent_worker(
    volatile WorkerQueue* queue
) {
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
        if (seq == 0xFFFFFFFFu) return; // shutdown

        // Copy instruction to shared memory (avoids PCIe reads in virtual block loop)
        __shared__ u64 sinst[INST_SIZE_WORDS];
        if (threadIdx.x < INST_SIZE_WORDS) {
            sinst[threadIdx.x] = queue->inst[threadIdx.x];
        }
        __syncthreads();
        const u64* inst = sinst;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);

        if (threadIdx.x == 0 && blockIdx.x == 0 && grid_x > 10000) {
            printf("WORKER: seq=%u opcode=%u grid_x=%u\n", seq, opcode, grid_x);
        }

        // OP_EXPERT_FFN: compound gate+up+silu+down for one expert (internal grid.sync)
        if (opcode == OP_EXPERT_FFN) {
            float* sg = (float*)(uintptr_t)inst[1];
            float* su = (float*)(uintptr_t)inst[2];
            float* sa = (float*)(uintptr_t)inst[3];
            float* eo = (float*)(uintptr_t)inst[4];
            const unsigned char* gate_up = (const unsigned char*)(uintptr_t)inst[5];
            const unsigned char* down    = (const unsigned char*)(uintptr_t)inst[6];
            const float* activation = (const float*)(uintptr_t)inst[7];
            int eis = (int)(inst[8] >> 32);
            int hs  = (int)(inst[8] & 0xFFFFFFFF);
            int gu_row_stride = (int)inst[9];
            int has_gate = (int)inst[10];

            if (has_gate) {
                // Gate proj
                for (int row = blockIdx.x; row < eis; row += gridDim.x) {
                    op_linear_proj_pcg32_row(sg, gate_up, activation, row, eis, hs);
                }
                grid.sync();
                // Up proj
                const unsigned char* up_data = gate_up + (long long)eis * gu_row_stride;
                for (int row = blockIdx.x; row < eis; row += gridDim.x) {
                    op_linear_proj_pcg32_row(su, up_data, activation, row, eis, hs);
                }
                grid.sync();
                // SiLU mul
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis;
                     i += gridDim.x * blockDim.x) {
                    float g = sg[i];
                    sa[i] = (g / (1.0f + expf(-g))) * su[i];
                }
                grid.sync();
            } else {
                // Up only → relu²
                for (int row = blockIdx.x; row < eis; row += gridDim.x) {
                    op_linear_proj_pcg32_row(su, gate_up, activation, row, eis, hs);
                }
                grid.sync();
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < eis;
                     i += gridDim.x * blockDim.x) {
                    float x = su[i]; float r = x > 0.0f ? x : 0.0f; sa[i] = r * r;
                }
                grid.sync();
            }
            // Down proj
            for (int row = blockIdx.x; row < hs; row += gridDim.x) {
                op_linear_proj_pcg32_row(eo, down, sa, row, hs, eis);
            }
            grid.sync();
            goto ack;
        }

        // Dispatch: same switch as megakernel, minus OP_HALT/OP_BARRIER/OP_MOE_DISPATCH
        if (opcode == OP_MOE_FFN) {
            op_moe_ffn(inst, grid);
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
                    default: break;
                }
            }
            if (threadIdx.x == 0 && blockIdx.x == 0 && grid_x > 10000) {
                printf("WORKER: block0 finished loop, entering grid.sync (grid_x=%u)\n", grid_x);
            }
            grid.sync();
            if (threadIdx.x == 0 && blockIdx.x == 0 && grid_x > 10000) {
                printf("WORKER: grid.sync done\n");
            }
        }

        // Ack: write seq_num to host-mapped flag
        ack:
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            queue->ack = seq;
        }

        last_seq = seq;
    }
}
