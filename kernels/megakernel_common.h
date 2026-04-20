// Common definitions shared between megakernel.hip and persistent_worker.hip.
// Instruction layout, helper functions, type aliases.
#pragma once
#include "opcodes.h"
#include "bf16_utils.h"
#include "quant_consts.h"
#include <stdint.h>
#include <stddef.h>

#define INST_SIZE_WORDS 18          // 18 u64s per instruction = 144 bytes

typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u32 inst_opcode(const u64* inst) {
    return (u32)(inst[0]);
}
__device__ __forceinline__ u32 inst_grid_x(const u64* inst) {
    return (u32)(inst[0] >> 32);
}
// ─── Per-opcode typed instruction structs (C99, matching Rust #[repr(C)]) ───

// OP_NOP (opcode 0)
typedef struct {
    uint64_t opcode_gridx;
    const uint8_t* dump_buf;
    int64_t max_slots;
    const int32_t* dump_counter;
    uint64_t _pad[14];
} NopInst;
static_assert(sizeof(NopInst) == INST_SIZE_WORDS * 8, "NopInst size mismatch");

// OP_RMSNORM / OP_RMSNORM_WX (opcodes 1, 27)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* input;
    const uint16_t* weight;
    int64_t dim;
    uint64_t eps_bits;
    uint64_t _pad[12];
} RmsNormInst;
static_assert(sizeof(RmsNormInst) == INST_SIZE_WORDS * 8, "RmsNormInst size mismatch");
static_assert(offsetof(RmsNormInst, output) == 8, "RmsNormInst.output offset");

// OP_LINEAR_PROJ / OP_LINEAR_PROJ_RNF4 / OP_LINEAR_PROJ_PCG32 (opcodes 2, 25, 26)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const uint8_t* weight;
    const float* input;
    int64_t out_dim;
    int64_t in_dim;
    int64_t batch;
    uint64_t _pad[11];
} LinearProjInst;
static_assert(sizeof(LinearProjInst) == INST_SIZE_WORDS * 8, "LinearProjInst size mismatch");
static_assert(offsetof(LinearProjInst, output) == 8, "LinearProjInst.output offset");

// OP_CONV1D (opcode 3)
typedef struct {
    uint64_t opcode_gridx;
    float* state;
    const float* input;
    const uint16_t* weight;
    float* output;
    int64_t dim;
    int64_t kernel_size;
    uint64_t _pad[11];
} Conv1dInst;
static_assert(sizeof(Conv1dInst) == INST_SIZE_WORDS * 8, "Conv1dInst size mismatch");
static_assert(offsetof(Conv1dInst, state) == 8, "Conv1dInst.state offset");

// OP_GDN_GATE (opcode 4)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* a_proj;
    const float* a_log;
    const uint16_t* dt_bias;
    int64_t num_heads;
    uint64_t _pad[12];
} GdnGateInst;
static_assert(sizeof(GdnGateInst) == INST_SIZE_WORDS * 8, "GdnGateInst size mismatch");
static_assert(offsetof(GdnGateInst, output) == 8, "GdnGateInst.output offset");

// OP_GDN_RECUR (opcode 5)
typedef struct {
    uint64_t opcode_gridx;
    const float* q;
    const float* k;
    const float* v;
    const float* gate;
    const float* b_proj;
    float* state;
    float* output;
    int64_t kd;
    int64_t vd;
    int64_t gqa_group;
    uint64_t num_heads;
    uint64_t _pad[6];
} GdnRecurInst;
static_assert(sizeof(GdnRecurInst) == INST_SIZE_WORDS * 8, "GdnRecurInst size mismatch");
static_assert(offsetof(GdnRecurInst, q) == 8, "GdnRecurInst.q offset");

// OP_RMSNORM_GATE (opcode 6)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* x;
    const float* z;
    const float* weight;
    int64_t num_heads;
    int64_t vd;
    uint64_t eps_bits;
    uint64_t _pad[10];
} RmsNormGateInst;
static_assert(sizeof(RmsNormGateInst) == INST_SIZE_WORDS * 8, "RmsNormGateInst size mismatch");
static_assert(offsetof(RmsNormGateInst, output) == 8, "RmsNormGateInst.output offset");

// OP_RESIDUAL_ADD (opcode 7)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* src;
    const float* residual;
    int64_t n;
    uint64_t _pad[13];
} ResidualAddInst;
static_assert(sizeof(ResidualAddInst) == INST_SIZE_WORDS * 8, "ResidualAddInst size mismatch");
static_assert(offsetof(ResidualAddInst, output) == 8, "ResidualAddInst.output offset");

// OP_QK_NORM (opcode 8)
typedef struct {
    uint64_t opcode_gridx;
    float* q;
    float* k;
    const uint16_t* q_norm;
    const uint16_t* k_norm;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    uint64_t eps_bits;
    int64_t batch;
    uint64_t _pad[8];
} QkNormInst;
static_assert(sizeof(QkNormInst) == INST_SIZE_WORDS * 8, "QkNormInst size mismatch");
static_assert(offsetof(QkNormInst, q) == 8, "QkNormInst.q offset");

// OP_MROPE (opcode 9)
typedef struct {
    uint64_t opcode_gridx;
    float* q;
    float* k;
    const float* inv_freq;
    const int32_t* pos_ids;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    int64_t rd;
    int64_t s0;
    int64_t s1;
    int64_t s2;
    int64_t batch;
    uint64_t _pad[5];
} MropeInst;
static_assert(sizeof(MropeInst) == INST_SIZE_WORDS * 8, "MropeInst size mismatch");
static_assert(offsetof(MropeInst, q) == 8, "MropeInst.q offset");

// OP_GQA_ATTN (opcode 10)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* q;
    const float* k_cache;
    const float* v_cache;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    int64_t seq_len;
    int64_t max_seq_len;
    int64_t q_head_start;
    uint64_t _pad[7];
} GqaAttnInst;
static_assert(sizeof(GqaAttnInst) == INST_SIZE_WORDS * 8, "GqaAttnInst size mismatch");
static_assert(offsetof(GqaAttnInst, output) == 8, "GqaAttnInst.output offset");

// OP_OUTPUT_GATE (opcode 11)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* attn_out;
    const float* gate;
    int64_t size;
    uint64_t _pad[13];
} OutputGateInst;
static_assert(sizeof(OutputGateInst) == INST_SIZE_WORDS * 8, "OutputGateInst size mismatch");
static_assert(offsetof(OutputGateInst, output) == 8, "OutputGateInst.output offset");

// OP_FFN_GATE_UP / OP_FFN_GATE_UP_RNF4 (opcodes 12, 30)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* hidden;
    const uint16_t* norm_weight;
    const uint8_t* w_gate;
    const uint8_t* w_up;
    int64_t hs;
    int64_t intermediate;
    uint64_t eps_bits;
    int64_t batch;
    uint64_t _pad[8];
} FfnGateUpInst;
static_assert(sizeof(FfnGateUpInst) == INST_SIZE_WORDS * 8, "FfnGateUpInst size mismatch");
static_assert(offsetof(FfnGateUpInst, output) == 8, "FfnGateUpInst.output offset");

// OP_FFN_DOWN_RES / OP_FFN_DOWN_RES_RNF4 (opcodes 13, 31)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* residual;
    const uint8_t* w_down;
    const float* ffn_act;
    int64_t hs;
    int64_t intermediate;
    int64_t batch;
    uint64_t _pad[10];
} FfnDownResInst;
static_assert(sizeof(FfnDownResInst) == INST_SIZE_WORDS * 8, "FfnDownResInst size mismatch");
static_assert(offsetof(FfnDownResInst, output) == 8, "FfnDownResInst.output offset");

// OP_EMBEDDING (opcode 14)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const uint16_t* embed_weight;
    int64_t token_id;
    int64_t hs;
    uint64_t _pad[13];
} EmbeddingInst;
static_assert(sizeof(EmbeddingInst) == INST_SIZE_WORDS * 8, "EmbeddingInst size mismatch");
static_assert(offsetof(EmbeddingInst, output) == 8, "EmbeddingInst.output offset");

// OP_D2D_COPY (opcode 17)
typedef struct {
    uint64_t opcode_gridx;
    float* dst;
    const float* src;
    int64_t n_elems;
    uint64_t _pad[14];
} D2dCopyInst;
static_assert(sizeof(D2dCopyInst) == INST_SIZE_WORDS * 8, "D2dCopyInst size mismatch");
static_assert(offsetof(D2dCopyInst, dst) == 8, "D2dCopyInst.dst offset");

// OP_ATTN_PAGED (opcode 18)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* q;
    uint64_t page_table;
    uint64_t pos_table;
    const float* inv_freq;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    int64_t seq_len;
    int64_t chunk_tokens;
    int64_t rd;
    uint64_t layer_k_offset;
    uint64_t layer_v_offset;
    uint64_t partial_state;
    uint64_t _pad1;
    const uint16_t* k_norm;
    uint64_t _pad2;
} AttnPagedInst;
static_assert(sizeof(AttnPagedInst) == INST_SIZE_WORDS * 8, "AttnPagedInst size mismatch");
static_assert(offsetof(AttnPagedInst, output) == 8, "AttnPagedInst.output offset");

// OP_ATTN_PREFILL (opcode 19)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* q;
    const float* k_cache;
    const float* v_cache;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    int64_t start_pos;
    int64_t n;
    int64_t max_seq_len;
    uint64_t _pad[7];
} AttnPrefillInst;
static_assert(sizeof(AttnPrefillInst) == INST_SIZE_WORDS * 8, "AttnPrefillInst size mismatch");
static_assert(offsetof(AttnPrefillInst, output) == 8, "AttnPrefillInst.output offset");

// OP_DEINTERLEAVE (opcode 20)
typedef struct {
    uint64_t opcode_gridx;
    float* dst_q;
    float* dst_gate;
    const float* src;
    int64_t num_heads;
    int64_t head_dim;
    int64_t batch;
    uint64_t _pad[11];
} DeinterleaveInst;
static_assert(sizeof(DeinterleaveInst) == INST_SIZE_WORDS * 8, "DeinterleaveInst size mismatch");
static_assert(offsetof(DeinterleaveInst, dst_q) == 8, "DeinterleaveInst.dst_q offset");

// OP_ATTN_PAGED_Q (opcode 22)
typedef struct {
    uint64_t opcode_gridx;
    uint64_t scratch;
    const float* q;
    uint64_t quant_page_table;
    uint64_t pos_table;
    const float* inv_freq;
    int64_t nqh;
    int64_t nkh;
    int64_t hd;
    int64_t quant_seq_len;
    int64_t chunk_tokens;
    int64_t rd;
    int64_t q1d;
    int64_t q1s;
    int64_t rd_off;
    int64_t rs;
    const uint16_t* k_norm;
    uint64_t _pad;
} AttnPagedQInst;
static_assert(sizeof(AttnPagedQInst) == INST_SIZE_WORDS * 8, "AttnPagedQInst size mismatch");

// OP_KV_QUANTIZE (opcode — see opcodes.h)
// words[1]=src(f32), [2]=q1_data(u8), [3]=q1_scale(f32*), [4]=r_data(u8),
//        [5]=r_scale(f32*), [6]=num_kv_heads, [7]=head_dim, [8]=chunk_tokens
typedef struct {
    uint64_t opcode_gridx;
    const float* src;
    uint8_t* q1_data;
    float* q1_scale;
    uint8_t* r_data;
    float* r_scale;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t chunk_tokens;
    int32_t _pad0;
    uint64_t _pad[10];
} KvQuantizeInst;
static_assert(sizeof(KvQuantizeInst) == INST_SIZE_WORDS * 8, "KvQuantizeInst size mismatch");
static_assert(offsetof(KvQuantizeInst, src) == 8, "KvQuantizeInst.src offset");
static_assert(offsetof(KvQuantizeInst, num_kv_heads) == 48, "KvQuantizeInst.num_kv_heads offset");

// OP_MOE_GATE (opcode 23)
typedef struct {
    uint64_t opcode_gridx;
    const float* scores;
    int32_t* expert_ids;
    float* expert_weights;
    int64_t ne;
    int64_t k;
    int64_t gate_mode;
    uint64_t rsf_bits;
    const uint8_t* bias;
    uint64_t _pad[9];
} MoeGateInst;
static_assert(sizeof(MoeGateInst) == INST_SIZE_WORDS * 8, "MoeGateInst size mismatch");
static_assert(offsetof(MoeGateInst, scores) == 8, "MoeGateInst.scores offset");

// OP_MOE_FFN (opcode 24)
typedef struct {
    uint64_t opcode_gridx;
    const int32_t* expert_ids;
    const float* expert_weights;
    const float* normed;
    float* ffn_down;
    const uint8_t* gate_up_data;
    uint64_t gate_up_expert_stride;
    const uint8_t* down_data;
    uint64_t down_expert_stride;
    int64_t k;
    int64_t hs_eis;
    int64_t flags;
    const float* expert_gate;
    const float* expert_up;
    const float* expert_act;
    const float* expert_out;
    uint64_t gate_up_row_stride;
    uint64_t _pad;
} MoeFfnInst;
static_assert(sizeof(MoeFfnInst) == INST_SIZE_WORDS * 8, "MoeFfnInst size mismatch");
static_assert(offsetof(MoeFfnInst, expert_ids) == 8, "MoeFfnInst.expert_ids offset");

// OP_SIGMOID_WEIGHTED_ADD (opcode 32)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* scalar;
    const float* input;
    int64_t n;
    uint64_t _pad[13];
} SigmoidWeightedAddInst;
static_assert(sizeof(SigmoidWeightedAddInst) == INST_SIZE_WORDS * 8, "SigmoidWeightedAddInst size mismatch");
static_assert(offsetof(SigmoidWeightedAddInst, output) == 8, "SigmoidWeightedAddInst.output offset");

// OP_SCALE_ADD (opcode 36)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* src;
    uint64_t scale_bits;
    int64_t size;
    uint64_t _pad[13];
} ScaleAddInst;
static_assert(sizeof(ScaleAddInst) == INST_SIZE_WORDS * 8, "ScaleAddInst size mismatch");
static_assert(offsetof(ScaleAddInst, output) == 8, "ScaleAddInst.output offset");

// OP_RELU_SQ (opcode 37)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* input;
    int64_t size;
    uint64_t _pad[14];
} ReluSqInst;
static_assert(sizeof(ReluSqInst) == INST_SIZE_WORDS * 8, "ReluSqInst size mismatch");
static_assert(offsetof(ReluSqInst, output) == 8, "ReluSqInst.output offset");

// OP_MAMBA2_CONV1D (opcode 38)
typedef struct {
    uint64_t opcode_gridx;
    float* state;
    const float* input;
    const uint16_t* weight;
    const float* bias;
    float* output;
    int64_t conv_dim;
    int64_t kernel_size;
    uint64_t _pad[10];
} Mamba2Conv1dInst;
static_assert(sizeof(Mamba2Conv1dInst) == INST_SIZE_WORDS * 8, "Mamba2Conv1dInst size mismatch");
static_assert(offsetof(Mamba2Conv1dInst, state) == 8, "Mamba2Conv1dInst.state offset");

// OP_MAMBA2_NORM_GATED (opcode 39)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* x;
    const float* z;
    const float* weight;
    int64_t num_heads;
    int64_t value_dim;
    uint64_t eps_bits;
    uint64_t _pad[10];
} Mamba2NormGatedInst;
static_assert(sizeof(Mamba2NormGatedInst) == INST_SIZE_WORDS * 8, "Mamba2NormGatedInst size mismatch");
static_assert(offsetof(Mamba2NormGatedInst, output) == 8, "Mamba2NormGatedInst.output offset");

// OP_SSM_UPDATE (opcode 29)
typedef struct {
    uint64_t opcode_gridx;
    float* state;
    const float* x;
    const float* dt;
    const float* dt_bias;
    const float* a_log;
    const float* b;
    const float* c;
    const float* d_weight;
    float* output;
    int64_t nh;
    int64_t hd;
    int64_t sd;
    int64_t ng;
    uint64_t _pad[4];
} SsmUpdateInst;
static_assert(sizeof(SsmUpdateInst) == INST_SIZE_WORDS * 8, "SsmUpdateInst size mismatch");
static_assert(offsetof(SsmUpdateInst, state) == 8, "SsmUpdateInst.state offset");

// OP_SILU_MUL (opcode 28)
typedef struct {
    uint64_t opcode_gridx;
    float* output;
    const float* gate;
    const float* up;
    int64_t size;
    uint64_t _pad[13];
} SiluMulInst;
static_assert(sizeof(SiluMulInst) == INST_SIZE_WORDS * 8, "SiluMulInst size mismatch");
static_assert(offsetof(SiluMulInst, output) == 8, "SiluMulInst.output offset");

// OP_MOE_DISPATCH (opcode 34)
typedef struct {
    uint64_t opcode_gridx;
    uint64_t work_queue;
    uint64_t output_slots;
    uint64_t final_output;
    uint64_t expert_ids;
    uint64_t expert_weights;
    uint64_t seq_counter;
    uint64_t num_workers_hs;    // (num_workers << 32) | hidden_size
    uint64_t layer_k;           // (layer_idx << 32) | k
    uint64_t eis_gate;          // (eis << 32) | has_gate
    uint64_t activation;
    uint64_t layer_config_ptrs;
    uint64_t scratch_gate;
    uint64_t scratch_up;
    uint64_t scratch_act;
    uint64_t num_gpus;
    uint64_t gate_up_in_dim;
    uint64_t _pad;
} MoeDispatchInst;
static_assert(sizeof(MoeDispatchInst) == INST_SIZE_WORDS * 8, "MoeDispatchInst size mismatch");

// OP_CONV1D_3X (opcode 40) — fused Q+K+V causal conv1d in one instruction
// grid_x = 2*blocks_qk + blocks_v; vb < blocks_qk → Q, < 2*blocks_qk → K, else → V
typedef struct {
    uint64_t opcode_gridx;
    float* q_state;
    const float* q_input;
    const uint16_t* q_weight;
    float* q_output;
    float* k_state;
    const float* k_input;
    const uint16_t* k_weight;
    float* k_output;
    float* v_state;
    const float* v_input;
    const uint16_t* v_weight;
    float* v_output;
    int64_t qk_dim;
    int64_t v_dim;
    int64_t kernel_size;
    uint64_t blocks_qk_v;  // low32=blocks_qk, high32=blocks_v
    uint64_t _pad;
} Conv1d3xInst;
static_assert(sizeof(Conv1d3xInst) == INST_SIZE_WORDS * 8, "Conv1d3xInst size mismatch");
static_assert(offsetof(Conv1d3xInst, q_state) == 8, "Conv1d3xInst.q_state offset");

// OP_BARRIER (opcode 33)
typedef struct {
    uint64_t opcode_gridx;
    const uint32_t* barrier_flag;
    const uint32_t* resume_flag;
    int64_t layer_idx;
    uint64_t _pad[14];
} BarrierInst;
static_assert(sizeof(BarrierInst) == INST_SIZE_WORDS * 8, "BarrierInst size mismatch");
static_assert(offsetof(BarrierInst, barrier_flag) == 8, "BarrierInst.barrier_flag offset");
