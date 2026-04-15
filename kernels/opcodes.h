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
#define OP_SSM_UPDATE   29  // Mamba2 selective state update
#define OP_FFN_GATE_UP_RNF4  30  // Fused RMSNorm + gate_proj + up_proj + SiLU*up (rnf4 weights)
#define OP_FFN_DOWN_RES_RNF4 31  // Fused down_proj + residual add (rnf4 weights)
#define OP_SIGMOID_WEIGHTED_ADD 32  // output[i] += sigmoid(scalar[0]) * input[i]
#define OP_BARRIER      33  // Park megakernel; CPU dispatches multi-GPU MoE; resume via mapped host flag
#define OP_MOE_DISPATCH 34  // GPU-initiated MoE: write work queue, poll ack flags, sum output slots
#define OP_EXPERT_FFN   35  // Compound: gate+up+silu+down (Q4 PcG32) with internal grid.sync()
#define OP_SCALE_ADD    36  // output[i] += scale * src[i]; args: [1]=output, [2]=src, [3]=scale(f32 bits), [4]=size
#define OP_RELU_SQ      37  // output[i] = relu(input[i])^2; args: [1]=output, [2]=input, [3]=size
#define OP_MAMBA2_CONV1D    38  // causal conv1d with f32 bias + SiLU; args: [1]=state(w), [2]=input, [3]=weight(u16), [4]=bias(f32), [5]=output(w), [6]=conv_dim, [7]=kernel_size
#define OP_MAMBA2_NORM_GATED 39 // rms_norm(x*silu(z))*weight; args: [1]=output(w), [2]=x, [3]=z, [4]=weight, [5]=num_heads, [6]=value_dim, [7]=eps
