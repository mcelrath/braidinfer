#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"

// Megakernel: persistent cooperative kernel that executes a program of instructions.
// Launched once with max cooperative blocks (384 on gfx1100 with 256 threads).
// Each instruction specifies an operation and how many "virtual blocks" it needs.
// Physical blocks loop over virtual blocks: for (vb = blockIdx.x; vb < grid_x; vb += gridDim.x)

#include "megakernel_ops.hip"
#include "megakernel_moe.hip"
#include "megakernel_moe_dispatch.hip"

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 18  // 18 u64s per instruction = 144 bytes

// Dump mode: per-instruction activation capture
// dump_base: GPU buffer for storing instruction outputs (NULL = disabled)
// dump_capacity: max number of slots
// Each slot: [opcode(u32) inst_idx(u32) size(u32) pad(u32)] [data(float...)]
#define DUMP_HEADER_INTS 4
#define DUMP_MAX_FLOATS  8192
#define DUMP_SLOT_BYTES  (DUMP_HEADER_INTS * 4 + DUMP_MAX_FLOATS * 4)

__device__ void dump_instruction_output(
    char* dump_base, int dump_capacity, int* dump_count,
    const u64* inst, int opcode, int pc
) {
    if (dump_base == nullptr) return;
    if (blockIdx.x != 0) return;  // Only block 0 dumps

    // Get output pointer and size from instruction (all structs have output/result at word[1])
    float* output = (float*)(inst[1]);
    int size = 0;

    // Determine output size from opcode + typed struct fields
    switch (opcode) {
        case OP_RMSNORM: case OP_RMSNORM_WX:
            size = (int)((const RmsNormInst*)inst)->dim;
            break;
        case OP_LINEAR_PROJ: case OP_LINEAR_PROJ_RNF4: case OP_LINEAR_PROJ_PCG32:
            size = (int)((const LinearProjInst*)inst)->out_dim;
            break;
        case OP_LINEAR_PROJ_2X:
            // grid_x covers both A and B regions; size used only for dump/trace bounds.
            size = (int)((const LinearProj2xInst*)inst)->out_dim * 2;
            break;
        case OP_RMSNORM_GATE: {
            const RmsNormGateInst* g = (const RmsNormGateInst*)inst;
            size = (int)g->num_heads * (int)g->vd;
            break;
        }
        case OP_RESIDUAL_ADD:
            size = (int)((const ResidualAddInst*)inst)->n;
            break;
        case OP_SILU_MUL:
            size = (int)((const SiluMulInst*)inst)->size;
            break;
        case OP_EMBEDDING:
            size = (int)((const EmbeddingInst*)inst)->hs;
            break;
        case OP_FFN_GATE_UP:
        case OP_FFN_GATE_UP_WX:
        case OP_FFN_GATE_UP_RNF4:
        case OP_FFN_GATE_UP_RNF4_WX:
            size = (int)((const FfnGateUpInst*)inst)->intermediate;
            break;
        case OP_FFN_DOWN_RES:
        case OP_FFN_DOWN_RES_RNF4:
            size = (int)((const FfnDownResInst*)inst)->hs;
            break;
        case OP_GQA_ATTN: {
            const GqaAttnInst* g = (const GqaAttnInst*)inst;
            size = (int)g->nqh * (int)g->hd;
            break;
        }
        case OP_OUTPUT_GATE:
            size = (int)((const OutputGateInst*)inst)->size;
            break;
        case OP_MOE_GATE: {
            const MoeGateInst* g = (const MoeGateInst*)inst;
            output = g->expert_weights;
            size = (int)g->k;
            break;
        }
        case OP_MOE_FFN: {
            const MoeFfnInst* g = (const MoeFfnInst*)inst;
            output = g->ffn_down;
            size = (int)g->hs_eis & 0xFFFF;
            break;
        }
        default:
            return;  // Skip non-compute ops
    }

    if (size <= 0) return;
    if (size > DUMP_MAX_FLOATS) size = DUMP_MAX_FLOATS;

    // Allocate slot (thread 0 only)
    int slot;
    if (threadIdx.x == 0) {
        slot = atomicAdd(dump_count, 1);
    }
    __shared__ int s_slot;
    if (threadIdx.x == 0) s_slot = slot;
    __syncthreads();
    slot = s_slot;

    if (slot >= dump_capacity) return;

    char* slot_ptr = dump_base + (long long)slot * DUMP_SLOT_BYTES;

    // Write header
    if (threadIdx.x == 0) {
        ((int*)slot_ptr)[0] = opcode;
        ((int*)slot_ptr)[1] = pc;
        ((int*)slot_ptr)[2] = size;
        ((int*)slot_ptr)[3] = 0;
    }

    // Copy output data cooperatively
    float* dst = (float*)(slot_ptr + DUMP_HEADER_INTS * 4);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = output[i];
    }
}

extern "C" __global__ __launch_bounds__(256) void megakernel_f32(
    const u64* __restrict__ program,
    int num_instructions
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

    for (int pc = 0; pc < num_instructions; pc++) {
        const u64* inst = program + pc * INST_SIZE;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);

        if (opcode == OP_HALT) break;
        if (opcode == OP_BARRIER) { grid.sync(); continue; }

        // OP_MOE_FFN manages its own blocks and grid.sync() internally
        if (opcode == OP_MOE_FFN) {
            op_moe_ffn(inst, grid);
            // op_moe_ffn does its own grid.sync(); skip outer sync
            if (dump_base) {
                dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
                grid.sync();
            }
            continue;
        }

        // OP_LINEAR_PROJ_PCG32/RNF4: tiled-LDS optimized GEMV. Manages its own row loop
        // (loads input tile once per block, processes all assigned rows per tile).
        if (opcode == OP_LINEAR_PROJ_PCG32) {
            op_linear_proj_pcg32_full(inst, grid_x);
            grid.sync();
            if (dump_base) { dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc); grid.sync(); }
            continue;
        }
        if (opcode == OP_LINEAR_PROJ_RNF4) {
            op_linear_proj_rnf4_full(inst, grid_x);
            grid.sync();
            if (dump_base) { dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc); grid.sync(); }
            continue;
        }

        // OP_MOE_DISPATCH: GPU-initiated MoE — triggers workers, polls acks, sums results.
        // Manages its own grid.sync() internally.
        if (opcode == OP_MOE_DISPATCH) {
            op_moe_dispatch(inst, grid);
            continue;
        }

        // Virtual block loop: each physical block handles multiple virtual blocks
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
                case OP_RMSNORM_WX: op_rmsnorm_wx(inst, vb); break;
                case OP_SILU_MUL:   op_silu_mul(inst, vb); break;
                case OP_FFN_GATE_UP_RNF4:  op_ffn_gate_up_rnf4(inst, vb); break;
                case OP_FFN_DOWN_RES_RNF4: op_ffn_down_res_rnf4(inst, vb); break;
                case OP_SIGMOID_WEIGHTED_ADD: op_sigmoid_weighted_add(inst, vb); break;
                case OP_SCALE_ADD:   op_scale_add(inst, vb); break;
                case OP_RELU_SQ:     op_relu_sq(inst, vb); break;
                case OP_MAMBA2_CONV1D:    op_mamba2_conv1d(inst, vb); break;
                case OP_SSM_UPDATE:       op_ssm_update(inst, vb); break;
                case OP_MAMBA2_NORM_GATED: op_mamba2_norm_gated(inst, vb); break;
                case OP_CONV1D_3X:         op_conv1d_3x(inst, vb); break;
                case OP_FFN_GATE_UP_WX:    op_ffn_gate_up_wx(inst, vb); break;
                case OP_FFN_GATE_UP_RNF4_WX: op_ffn_gate_up_rnf4_wx(inst, vb); break;
                case OP_LINEAR_PROJ_2X:    op_linear_proj_2x(inst, vb); break;
                default:
                    if (threadIdx.x == 0 && blockIdx.x == 0)
                        printf("MEGAKERNEL: unknown opcode %u at pc %d\n", opcode, pc);
                    break;
            }
        }

        // Dump output when enabled (no grid.sync here; caller inserts BarrierInst after dump groups)
        grid.sync();

        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            grid.sync();
        }
    }
}
