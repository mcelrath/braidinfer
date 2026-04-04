#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "bf16_utils.h"

// Megakernel: persistent cooperative kernel that executes a program of instructions.
// Launched once with max cooperative blocks (384 on gfx1100 with 256 threads).
// Each instruction specifies an operation and how many "virtual blocks" it needs.
// Physical blocks loop over virtual blocks: for (vb = blockIdx.x; vb < grid_x; vb += gridDim.x)

// Instruction layout (136 bytes, 17 uint64s):
//   [0]:  opcode (low 31) | FLAG_NO_SYNC (bit 31) | grid_x (high 32)
//   [1..16]: up to 16 argument slots (pointers or packed scalars)
//   [5]:  arg4
//   [6]:  arg5
//   [7]:  arg6
//   [8]:  arg7
//   [9]:  arg8
//   [10]: arg9
//   [11]: arg10
//   [12]: arg11
//   [13]: arg12
//   [14]: arg13
//   [15]: arg14

#include "opcodes.h"
#include "quant_consts.h"

#define FLAG_NO_SYNC   0x80000000u  // bit 31: skip grid.sync() after this instruction

typedef unsigned long long u64;
typedef unsigned int u32;

// Helper to extract fields from instruction
__device__ __forceinline__ u32 inst_opcode(const u64* inst) {
    return (u32)(inst[0] & 0x7FFFFFFF);  // mask out FLAG_NO_SYNC
}
__device__ __forceinline__ bool inst_no_sync(const u64* inst) {
    return (inst[0] & FLAG_NO_SYNC) != 0;
}
__device__ __forceinline__ u32 inst_grid_x(const u64* inst) {
    return (u32)(inst[0] >> 32);
}
__device__ __forceinline__ float* inst_fptr(const u64* inst, int idx) {
    return (float*)(inst[idx]);
}
__device__ __forceinline__ const unsigned short* inst_u16ptr(const u64* inst, int idx) {
    return (const unsigned short*)(inst[idx]);
}
__device__ __forceinline__ const void* inst_ptr(const u64* inst, int idx) {
    return (const void*)(inst[idx]);
}
__device__ __forceinline__ int inst_int(const u64* inst, int idx) {
    return (int)(inst[idx]);
}
__device__ __forceinline__ float inst_float(const u64* inst, int idx) {
    u32 bits = (u32)inst[idx];
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}


#include "megakernel_ops.hip"
#include "megakernel_moe.hip"
#include "megakernel_moe_barrier.hip"

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 17  // 17 u64s per instruction = 136 bytes (expanded for QK-norm k_norm_weight ptr at slot 16)

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

    // Get output pointer and size from instruction (inst[1] is output for all compute ops)
    float* output = inst_fptr(inst, 1);
    int size = 0;

    // Determine output size from opcode + instruction args
    switch (opcode) {
        case OP_RMSNORM: case OP_RMSNORM_WX:
            size = inst_int(inst, 4); // hidden_size
            break;
        case OP_LINEAR_PROJ: case OP_LINEAR_PROJ_RNF4: case OP_LINEAR_PROJ_PCG32:
            size = inst_int(inst, 4); // out_dim (total for batch=1)
            break;
        case OP_RMSNORM_GATE:
            size = inst_int(inst, 4) * inst_int(inst, 5); // num_heads * value_dim
            break;
        case OP_RESIDUAL_ADD:
            size = inst_int(inst, 4); // size
            break;
        case OP_SILU_MUL:
            size = inst_int(inst, 4); // size
            break;
        case OP_EMBEDDING:
            size = inst_int(inst, 4); // hidden_size
            break;
        case OP_FFN_GATE_UP:
        case OP_FFN_GATE_UP_RNF4:
            size = inst_int(inst, 7); // intermediate_size
            break;
        case OP_FFN_DOWN_RES:
        case OP_FFN_DOWN_RES_RNF4:
            size = inst_int(inst, 5); // hidden_size
            break;
        case OP_GQA_ATTN:
            size = inst_int(inst, 4) * inst_int(inst, 6); // nqh * hd
            break;
        case OP_OUTPUT_GATE:
            size = inst_int(inst, 4); // size
            break;
        case OP_MOE_GATE:
            output = inst_fptr(inst, 3); // expert_weights
            size = inst_int(inst, 5);    // k
            break;
        case OP_MOE_FFN:
            output = inst_fptr(inst, 4); // ffn_down (accumulated output)
            size = inst_int(inst, 10) & 0xFFFF; // hs
            break;
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

    // Dump state: encoded in first instruction if OP_NOP with dump_base != 0
    // First instruction words: [0]=opcode(NOP), [1]=dump_base ptr, [2]=dump_capacity, [3]=dump_count ptr
    char* dump_base = nullptr;
    int dump_capacity = 0;
    int* dump_count = nullptr;
    if (num_instructions > 0 && inst_opcode(program) == OP_NOP && inst_ptr(program, 1) != nullptr) {
        dump_base = (char*)inst_ptr(program, 1);
        dump_capacity = inst_int(program, 2);
        dump_count = (int*)inst_ptr(program, 3);
    }

    for (int pc = 0; pc < num_instructions; pc++) {
        const u64* inst = program + pc * INST_SIZE;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);
        bool no_sync = inst_no_sync(inst);

        if (opcode == OP_HALT) break;

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

        // OP_BARRIER: block 0 thread 0 signals CPU and spins; all other blocks wait at grid.sync().
        // Must be outside the virtual block loop so blocks 1..N-1 can reach grid.sync() immediately
        // while block 0 spins. After CPU resumes block 0, block 0 also calls grid.sync() to reunite.
        if (opcode == OP_BARRIER) {
            op_barrier(inst, blockIdx.x);
            grid.sync();
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
                default: break;
            }
        }

        if (!no_sync) grid.sync();

        // Dump output AFTER grid.sync() — all blocks finished writing
        if (dump_base && !no_sync) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            grid.sync();  // Ensure dump completes before next instruction
        }
    }
}
