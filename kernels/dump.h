// Per-instruction trace dump helper used by the persistent_worker mailbox
// path via kernels/megakernel_dispatch.hip. bd 9gmh Phase 4: the prior
// megakernel_f32 one-shot path was deleted; only persistent_worker remains.
//
// Block 0 atomically allocates a slot in dump_buffer, writes a header
// {opcode, pc, size, pad}, then copies the instruction's output region
// cooperatively across threadIdx.x. Other blocks no-op.
//
// Output region is determined per-opcode from the typed *Inst structs
// in megakernel_common.h (which must be included before this header).
//
// History: this header was extracted from kernels/megakernel.hip in the
// braidinfer-zqw merge so dump support could be shared across entry points.
#ifndef BRAIDINFER_DUMP_H
#define BRAIDINFER_DUMP_H

#include <hip/hip_runtime.h>
#include "megakernel_common.h"

// Slot layout: [opcode(u32) inst_idx(u32) size(u32) pad(u32)] [data(float...)]
#define DUMP_HEADER_INTS 4
#define DUMP_MAX_FLOATS  8192
#define DUMP_SLOT_BYTES  (DUMP_HEADER_INTS * 4 + DUMP_MAX_FLOATS * 4)

__device__ __forceinline__ void dump_instruction_output(
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
        case OP_SCALE_ADD:
            // GDN/Mamba2 blocks end with OP_SCALE_ADD (in-place residual via
            // scaled add). Output is the post-residual hidden state — exactly
            // what PostMixer{layer} probes want.
            size = (int)((const ScaleAddInst*)inst)->size;
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
        case OP_ATTN_PAGED: {
            const AttnPagedInst* g = (const AttnPagedInst*)inst;
            size = (int)g->nqh * (int)g->hd;
            break;
        }
        // OP_ATTN_PAGED_Q: skipped from dump (scratch field at word[1] is uint64 not pointer;
        // default `output = inst[1]` would deref a non-pointer value as a float*).
        // OP_ATTN_PREFILL: skipped from dump (only fires during prefill, not bench_coherence
        // decode comparison; size n*nqh*hd can be large; not the audit target anyway).
        case OP_GDN_GATE: {
            const GdnGateInst* g = (const GdnGateInst*)inst;
            size = (int)g->num_heads;
            break;
        }
        case OP_QK_NORM: {
            // dump q only (k is at word 2; we lose K coverage but enough for diagnosis)
            const QkNormInst* g = (const QkNormInst*)inst;
            int batch = (int)g->batch; if (batch <= 0) batch = 1;
            size = batch * (int)g->nqh * (int)g->hd;
            break;
        }
        case OP_MROPE: {
            // dump q only
            const MropeInst* g = (const MropeInst*)inst;
            int batch = (int)g->batch; if (batch <= 0) batch = 1;
            size = batch * (int)g->nqh * (int)g->hd;
            break;
        }
        case OP_DEINTERLEAVE: {
            const DeinterleaveInst* g = (const DeinterleaveInst*)inst;
            int batch = (int)g->batch; if (batch <= 0) batch = 1;
            size = batch * (int)g->num_heads * (int)g->head_dim;
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
    if (output == nullptr) return;  // null-guard: some ops have output @ word[1] = 0 (uninit / unused)
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

#endif  // BRAIDINFER_DUMP_H
