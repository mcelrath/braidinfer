// Megakernel opcodes — single source of truth.
// Parsed by build.rs to generate Rust constants.
// Included by megakernel.hip for GPU-side dispatch.

#define OP_NOP           0
#define OP_RMSNORM       1
#define OP_LINEAR_PROJ   2
#define OP_CONV1D        3
#define OP_GDN_GATE      4
#define OP_GDN_RECUR     5
#define OP_RMSNORM_GATE  6
#define OP_RESIDUAL_ADD  7
#define OP_QK_NORM       8
#define OP_MROPE         9
#define OP_GQA_ATTN     10
#define OP_OUTPUT_GATE  11
#define OP_FFN_GATE_UP  12
#define OP_FFN_DOWN_RES 13
#define OP_EMBEDDING    14
#define OP_LM_HEAD      15
#define OP_HALT         16
#define OP_D2D_COPY     17
#define OP_ATTN_PAGED   18
#define OP_ATTN_PREFILL 19
#define OP_DEINTERLEAVE 20
#define OP_KV_QUANTIZE  21
#define OP_ATTN_PAGED_Q 22
#define OP_MOE_GATE     23
#define OP_MOE_FFN      24  // Reserved — MoE uses kernel-by-kernel dispatch (see braidinfer-cea.7)
#define OP_LINEAR_PROJ_RNF4  25
#define OP_LINEAR_PROJ_PCG32 26
#define OP_RMSNORM_WX   27  // w*x variant (Llama, OLMoE) vs (1+w)*x (Qwen3.5)
#define OP_SILU_MUL     28  // output[i] = silu(gate[i]) * up[i]
