#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "bf16_utils.h"

// Megakernel: persistent cooperative kernel that executes a program of instructions.
// Launched once with max cooperative blocks (384 on gfx1100 with 256 threads).
// Each instruction specifies an operation and how many "virtual blocks" it needs.
// Physical blocks loop over virtual blocks: for (vb = blockIdx.x; vb < grid_x; vb += gridDim.x)

// Instruction layout (128 bytes, 16 uint64s):
//   [0]:  opcode (low 32) | grid_x (high 32)  — grid_x = number of virtual blocks
//   [1]:  arg0 (pointer or scalar)
//   [2]:  arg1
//   [3]:  arg2
//   [4]:  arg3
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

#define OP_NOP          0
#define OP_RMSNORM      1
#define OP_LINEAR_PROJ  2
#define OP_CONV1D       3
#define OP_GDN_GATE     4
#define OP_GDN_RECUR    5
#define OP_RMSNORM_GATE 6
#define OP_RESIDUAL_ADD 7
#define OP_QK_NORM      8
#define OP_MROPE        9
#define OP_GQA_ATTN    10
#define OP_OUTPUT_GATE 11
#define OP_FFN_GATE_UP 12
#define OP_FFN_DOWN_RES 13
#define OP_EMBEDDING   14
#define OP_LM_HEAD     15
#define OP_HALT        16
#define OP_D2D_COPY    17
#define OP_ATTN_PAGED  18

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
__device__ __forceinline__ int inst_int(const u64* inst, int idx) {
    return (int)(inst[idx]);
}
__device__ __forceinline__ float inst_float(const u64* inst, int idx) {
    union { u64 u; float f; } conv;
    conv.u = inst[idx];
    return conv.f;
}

// ============================================================
// Operation implementations
// Each takes the instruction args and a virtual block index.
// threadIdx.x is the thread within the block (0..255).
// ============================================================

// OP_RMSNORM: y = x * rsqrt(mean(x²)+eps) * (1+weight)
// args: [1]=output, [2]=input, [3]=weight, [4]=hidden_size, [5]=eps_bits
// grid_x = batch (default 1). vblock = batch index. Each block normalizes one row.
__device__ void op_rmsnorm(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const float* input_base = inst_fptr(inst, 2);
    const unsigned short* weight = inst_u16ptr(inst, 3);
    int hidden_size = inst_int(inst, 4);
    float eps = inst_float(inst, 5);

    const float* input = input_base + (long long)vblock * hidden_size;
    float* output = output_base + (long long)vblock * hidden_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = input[i];
        sum_sq += v * v;
    }
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        output[i] = input[i] * rms * (1.0f + bf16_to_f32(weight[i]));
    }
}

// OP_LINEAR_PROJ: y = W @ x  (one virtual block per output row)
// args: [1]=output, [2]=weight, [3]=input, [4]=out_dim, [5]=in_dim, [6]=batch_size (0 or 1 = single token)
// When batch > 1: input is [batch × in_dim], output is [batch × out_dim].
// grid_x = out_dim. Each block handles one weight row for ALL batch elements.
// Weight loaded once (from L2 cache), dot products accumulated per batch element in registers.
#define MAX_PREFILL_BATCH 64
__device__ void op_linear_proj(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const unsigned short* weight = inst_u16ptr(inst, 2);
    const float* input_base = inst_fptr(inst, 3);
    int out_dim = inst_int(inst, 4);
    int in_dim = inst_int(inst, 5);
    int batch = inst_int(inst, 6);
    if (batch <= 1) batch = 1;

    int row = vblock;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned short* w = weight + (long long)row * in_dim;

    // Single-token fast path (no register array overhead)
    if (batch == 1) {
        float acc = 0.0f;
        const int vec_elements = 8;
        const int vec_count = in_dim / vec_elements;
        const ull2* w_vec = (const ull2*)w;

        for (int vi = tid; vi < vec_count; vi += stride) {
            ull2 wv = __builtin_nontemporal_load(&w_vec[vi]);
            unsigned short ws[8];
            __builtin_memcpy(ws, &wv, 16);
            int base = vi * vec_elements;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc += bf16_to_f32(ws[j]) * input_base[base + j];
        }
        int rem = vec_count * vec_elements;
        for (int i = rem + tid; i < in_dim; i += stride) acc += bf16_to_f32(w[i]) * input_base[i];

        extern __shared__ float shared[];
        shared[tid] = acc;
        __syncthreads();
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        if (tid == 0) output_base[row] = shared[0];
        return;
    }

    // Batched path: accumulate per-batch dot products, weight loaded once from L2 cache
    float acc[MAX_PREFILL_BATCH];
    for (int b = 0; b < batch; b++) acc[b] = 0.0f;

    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ull2* w_vec = (const ull2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ull2 wv = w_vec[vi];
        unsigned short ws[8];
        __builtin_memcpy(ws, &wv, 16);
        float wf[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) wf[j] = bf16_to_f32(ws[j]);

        int base = vi * vec_elements;
        for (int b = 0; b < batch; b++) {
            const float* inp = input_base + (long long)b * in_dim + base;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[b] += wf[j] * inp[j];
        }
    }
    int rem = vec_count * vec_elements;
    for (int i = rem + tid; i < in_dim; i += stride) {
        float wf = bf16_to_f32(w[i]);
        for (int b = 0; b < batch; b++) acc[b] += wf * input_base[(long long)b * in_dim + i];
    }

    // Reduce each batch element (shared memory reduction)
    extern __shared__ float shared[];
    for (int b = 0; b < batch; b++) {
        shared[tid] = acc[b];
        __syncthreads();
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        if (tid == 0) output_base[(long long)b * out_dim + row] = shared[0];
        __syncthreads();
    }
}

// OP_CONV1D: causal conv1d update (one virtual block per channel)
// args: [1]=state, [2]=input, [3]=weight, [4]=output, [5]=conv_dim, [6]=kernel_size
__device__ void op_conv1d(const u64* inst, u32 vblock) {
    float* state = inst_fptr(inst, 1);
    const float* input = inst_fptr(inst, 2);
    const unsigned short* weight = inst_u16ptr(inst, 3);
    float* output = inst_fptr(inst, 4);
    int conv_dim = inst_int(inst, 5);
    int kernel_size = inst_int(inst, 6);

    int ch = vblock * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    int hist = kernel_size - 1;
    float* s = state + ch * hist;
    const unsigned short* w = weight + ch * kernel_size;

    float acc = 0.0f;
    for (int k = 0; k < hist; k++) {
        acc += s[k] * bf16_to_f32(w[k]);
    }
    acc += input[ch] * bf16_to_f32(w[hist]);
    output[ch] = acc / (1.0f + expf(-acc));

    for (int i = 0; i < hist - 1; i++) {
        s[i] = s[i + 1];
    }
    s[hist - 1] = input[ch];
}

// OP_GDN_GATE: gate = exp(-exp(A_log) * softplus(a + dt_bias))
// args: [1]=gate_out, [2]=a_input, [3]=A_log, [4]=dt_bias, [5]=num_heads
__device__ void op_gdn_gate(const u64* inst, u32 vblock) {
    float* gate_out = inst_fptr(inst, 1);
    const float* a_input = inst_fptr(inst, 2);
    const float* A_log = inst_fptr(inst, 3);
    const unsigned short* dt_bias = inst_u16ptr(inst, 4);
    int num_heads = inst_int(inst, 5);

    int h = vblock * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;

    float a = a_input[h];
    float A = expf(A_log[h]);
    float sp = log1pf(expf(a + bf16_to_f32(dt_bias[h])));
    gate_out[h] = expf(-A * sp);
}

// OP_GDN_RECUR: gated delta rule recurrence (one block per head)
// args: [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b, [6]=state|output_packed
//   state and output are packed: state_ptr = arg6, output_ptr = arg6 + state_size
//   Actually, let's use two args: [6]=state, [7-low]=output (but we only have 7 slots)
//   Re-pack: arg6 = state_ptr, we need output ptr too.
//   Solution: use inst[7] for additional args
// args: [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b_and_key_dim, [6]=state, [7-low]=output
//   b_and_key_dim: low 32 = key_dim, high 32 = value_dim... no, b is a pointer.
//   We need 8 args but only have 7 slots (inst[1]..inst[7]).
//   Let's pack key_dim and value_dim into one u64: low32=key_dim, high32=value_dim
//   and use: [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b, [6]=state, [7]=output
//   dims passed as grid_x encoding... no.
//   Actually we have inst[0..7] = 8 slots. inst[0] is opcode|grid_x.
//   So inst[1]..inst[7] = 7 arg slots. We need: q,k,v,gate,b,state,output + key_dim,value_dim = 9 values.
//   Pack dims: key_dim|value_dim into a single u64.
//   Use: [1]=q, [2]=k, [3]=v, [4]=gate_and_b_packed, [5]=state, [6]=output, [7]=dims
//   gate_and_b_packed: low 64 bits... no, these are pointers.
//   Better: increase instruction size to 12 u64s (96 bytes). Still cache-line friendly.
//   OR: for complex ops, use an indirection — args point to a separate args buffer.
//   Simplest: make instruction 16 u64s (128 bytes = 2 cache lines). Plenty of room.

// Let me redesign: 128-byte instructions (16 x u64)
// [0]: opcode (low32) | grid_x (high32)
// [1..15]: up to 15 argument slots

// For GDN recurrent:
// [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b, [6]=state, [7]=output, [8]=key_dim, [9]=value_dim
__device__ void op_gdn_recurrent(const u64* inst, u32 vblock) {
    const float* q = inst_fptr(inst, 1);
    const float* k = inst_fptr(inst, 2);
    const float* v = inst_fptr(inst, 3);
    const float* gate = inst_fptr(inst, 4);
    const float* b = inst_fptr(inst, 5);
    float* state = inst_fptr(inst, 6);
    float* output = inst_fptr(inst, 7);
    int key_dim = inst_int(inst, 8);
    int value_dim = inst_int(inst, 9);

    const int h = vblock;  // one block per head
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float decay = gate[h];
    const float beta = 1.0f / (1.0f + expf(-b[h]));

    const float* qh = q + h * key_dim;
    const float* kh = k + h * key_dim;
    const float* vh = v + h * value_dim;
    float* sh = state + h * key_dim * value_dim;
    float* oh = output + h * value_dim;

    extern __shared__ float shared[];
    float* sq = shared;
    float* sk = shared + block_size;

    float sq_acc = 0.0f, sk_acc = 0.0f;
    for (int i = tid; i < key_dim; i += block_size) {
        sq_acc += qh[i] * qh[i];
        sk_acc += kh[i] * kh[i];
    }
    sq[tid] = sq_acc;
    sk[tid] = sk_acc;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) { sq[tid] += sq[tid + s]; sk[tid] += sk[tid + s]; }
        __syncthreads();
    }
    float q_inv_norm = rsqrtf(sq[0] + 1e-6f);
    float k_inv_norm = rsqrtf(sk[0] + 1e-6f);
    float q_scale = q_inv_norm * rsqrtf((float)key_dim);

    int total = key_dim * value_dim;
    for (int idx = tid; idx < total; idx += block_size) sh[idx] *= decay;
    __syncthreads();

    for (int j = tid; j < value_dim; j += block_size) {
        float kv_mem = 0.0f;
        for (int i = 0; i < key_dim; i++) kv_mem += sh[j * key_dim + i] * (kh[i] * k_inv_norm);
        float delta = (vh[j] - kv_mem) * beta;
        for (int i = 0; i < key_dim; i++) sh[j * key_dim + i] += (kh[i] * k_inv_norm) * delta;
    }
    __syncthreads();

    for (int j = tid; j < value_dim; j += block_size) {
        float sum = 0.0f;
        for (int i = 0; i < key_dim; i++) sum += sh[j * key_dim + i] * (qh[i] * q_scale);
        oh[j] = sum;
    }
}

// OP_RMSNORM_GATE: RMSNorm(x) * SiLU(z) with direct weight multiply (no 1+w)
// args: [1]=output, [2]=x, [3]=z, [4]=weight, [5]=num_heads, [6]=value_dim, [7]=eps
__device__ void op_rmsnorm_gated(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* x = inst_fptr(inst, 2);
    const float* z = inst_fptr(inst, 3);
    const float* weight = inst_fptr(inst, 4);
    int value_dim = inst_int(inst, 6);
    float eps = inst_float(inst, 7);

    int h = vblock;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xh = x + h * value_dim;
    const float* zh = z + h * value_dim;
    float* oh = output + h * value_dim;

    extern __shared__ float shared[];
    float sum_sq = 0.0f;
    for (int i = tid; i < value_dim; i += stride) { float v = xh[i]; sum_sq += v * v; }
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / (float)value_dim + eps);

    for (int i = tid; i < value_dim; i += stride) {
        float normed = xh[i] * rms * weight[i];
        float zi = zh[i];
        oh[i] = normed * zi / (1.0f + expf(-zi));
    }
}

// OP_RESIDUAL_ADD: output = a + b
// args: [1]=output, [2]=a, [3]=b, [4]=size
__device__ void op_residual_add(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* a = inst_fptr(inst, 2);
    const float* b = inst_fptr(inst, 3);
    int size = inst_int(inst, 4);

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = a[idx] + b[idx];
}

// OP_FFN_GATE_UP: fused RMSNorm + gate_proj + up_proj + SiLU*up
// args: [1]=scratch, [2]=input, [3]=rms_weight, [4]=w_gate, [5]=w_up,
//       [6]=hidden_size, [7]=intermediate_size, [8]=eps, [9]=batch_size (0/1=single)
// When batch > 1: input is [batch × hidden_size], scratch is [batch × intermediate_size].
// vblock = batch_idx * intermediate_size + row.
__device__ void op_ffn_gate_up(const u64* inst, u32 vblock) {
    float* scratch_base = inst_fptr(inst, 1);
    const float* input_base = inst_fptr(inst, 2);
    const unsigned short* rms_weight = inst_u16ptr(inst, 3);
    const unsigned short* w_gate = inst_u16ptr(inst, 4);
    const unsigned short* w_up = inst_u16ptr(inst, 5);
    int hidden_size = inst_int(inst, 6);
    int intermediate_size = inst_int(inst, 7);
    float eps = inst_float(inst, 8);
    int batch = inst_int(inst, 9);
    if (batch <= 1) batch = 1;

    int row = vblock % intermediate_size;
    int batch_idx = vblock / intermediate_size;
    if (batch_idx >= batch) return;

    const float* input = input_base + (long long)batch_idx * hidden_size;
    float* scratch = scratch_base + (long long)batch_idx * intermediate_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) { float v = input[i]; sum_sq += v * v; }
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms_scale = rsqrtf(shared[0] / (float)hidden_size + eps);
    __syncthreads();

    const unsigned short* wg = w_gate + (long long)row * hidden_size;
    const unsigned short* wu = w_up + (long long)row * hidden_size;
    float acc_gate = 0.0f, acc_up = 0.0f;

    const int vec_elements = 8;
    const int vec_count = hidden_size / vec_elements;
    const ull2* wg_vec = (const ull2*)wg;
    const ull2* wu_vec = (const ull2*)wu;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ull2 gv = __builtin_nontemporal_load(&wg_vec[vi]);
        ull2 uv = __builtin_nontemporal_load(&wu_vec[vi]);
        unsigned short gs[8], us[8];
        __builtin_memcpy(gs, &gv, 16);
        __builtin_memcpy(us, &uv, 16);
        int base = vi * vec_elements;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float normed = input[base + j] * rms_scale * (1.0f + bf16_to_f32(rms_weight[base + j]));
            acc_gate += bf16_to_f32(gs[j]) * normed;
            acc_up += bf16_to_f32(us[j]) * normed;
        }
    }
    int rem = vec_count * vec_elements;
    for (int i = rem + tid; i < hidden_size; i += stride) {
        float normed = input[i] * rms_scale * (1.0f + bf16_to_f32(rms_weight[i]));
        acc_gate += bf16_to_f32(wg[i]) * normed;
        acc_up += bf16_to_f32(wu[i]) * normed;
    }

    shared[tid] = acc_gate;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float gate_val = shared[0];
    __syncthreads();

    shared[tid] = acc_up;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float up_val = shared[0];

    if (tid == 0) {
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        scratch[row] = silu_g * up_val;
    }
}

// OP_FFN_DOWN_RES: down_proj(scratch) + residual
// args: [1]=output, [2]=residual, [3]=w_down, [4]=scratch, [5]=hidden_size, [6]=intermediate_size, [7]=batch_size
// When batch > 1: output/residual are [batch × hidden_size], scratch is [batch × intermediate_size].
// vblock = batch_idx * hidden_size + row.
__device__ void op_ffn_down_res(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const float* residual_base = inst_fptr(inst, 2);
    const unsigned short* w_down = inst_u16ptr(inst, 3);
    const float* scratch_base = inst_fptr(inst, 4);
    int hidden_size = inst_int(inst, 5);
    int intermediate_size = inst_int(inst, 6);
    int batch = inst_int(inst, 7);
    if (batch <= 1) batch = 1;

    int row = vblock % hidden_size;
    int batch_idx = vblock / hidden_size;
    if (batch_idx >= batch) return;

    float* output = output_base + (long long)batch_idx * hidden_size;
    const float* residual = residual_base + (long long)batch_idx * hidden_size;
    const float* scratch = scratch_base + (long long)batch_idx * intermediate_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned short* w = w_down + (long long)row * intermediate_size;

    float acc = 0.0f;

    const int vec_elements = 8;
    const int vec_count = intermediate_size / vec_elements;
    const ull2* w_vec = (const ull2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ull2 wv = __builtin_nontemporal_load(&w_vec[vi]);
        unsigned short ws[8];
        __builtin_memcpy(ws, &wv, 16);
        int base = vi * vec_elements;
        #pragma unroll
        for (int j = 0; j < 8; j++) acc += bf16_to_f32(ws[j]) * scratch[base + j];
    }
    int rem = vec_count * vec_elements;
    for (int i = rem + tid; i < intermediate_size; i += stride) acc += bf16_to_f32(w[i]) * scratch[i];

    extern __shared__ float shared[];
    shared[tid] = acc;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[row] = shared[0] + residual[row];
}

// OP_EMBEDDING: output[i] = table[token_id * hidden_size + i]
// args: [1]=output, [2]=table, [3]=token_id, [4]=hidden_size
__device__ void op_embedding(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const unsigned short* table = inst_u16ptr(inst, 2);
    int token_id = inst_int(inst, 3);
    int hidden_size = inst_int(inst, 4);

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        output[idx] = bf16_to_f32(table[(long long)token_id * hidden_size + idx]);
    }
}

// OP_LM_HEAD: logits[row] = dot(weight[row], input) — same as linear_proj but may be large
// Reuses op_linear_proj logic
__device__ void op_lm_head(const u64* inst, u32 vblock) {
    op_linear_proj(inst, vblock);
}

// OP_QK_NORM: per-head RMSNorm on Q and K with (1+weight)
// args: [1]=q, [2]=k, [3]=q_weight, [4]=k_weight, [5]=num_q_heads, [6]=num_kv_heads, [7]=head_dim, [8]=eps
__device__ void op_qk_norm(const u64* inst, u32 vblock) {
    float* q = inst_fptr(inst, 1);
    float* k = inst_fptr(inst, 2);
    const unsigned short* q_weight = inst_u16ptr(inst, 3);
    const unsigned short* k_weight = inst_u16ptr(inst, 4);
    int num_q_heads = inst_int(inst, 5);
    int num_kv_heads = inst_int(inst, 6);
    int head_dim = inst_int(inst, 7);
    float eps = inst_float(inst, 8);

    int head_idx = vblock;
    float* head_ptr;
    const unsigned short* weight_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + head_idx * head_dim;
        weight_ptr = q_weight;
    } else {
        head_ptr = k + (head_idx - num_q_heads) * head_dim;
        weight_ptr = k_weight;
    }

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) { float v = head_ptr[i]; sum_sq += v * v; }
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += stride) {
        head_ptr[i] = head_ptr[i] * rms * (1.0f + bf16_to_f32(weight_ptr[i]));
    }
}

// OP_D2D_COPY: memcpy device-to-device
// args: [1]=dst, [2]=src, [3]=count (in floats)
__device__ void op_d2d_copy(const u64* inst, u32 vblock) {
    float* dst = inst_fptr(inst, 1);
    const float* src = inst_fptr(inst, 2);
    int count = inst_int(inst, 3);

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] = src[idx];
}

// OP_MROPE: multimodal rotary position embedding (one block per head)
// args: [1]=q, [2]=k, [3]=inv_freq, [4]=position_ids, [5]=num_q_heads,
//       [6]=num_kv_heads, [7]=head_dim, [8]=rope_dim,
//       [9]=section0_pairs, [10]=section1_pairs, [11]=section2_pairs
__device__ void op_mrope(const u64* inst, u32 vblock) {
    float* q = inst_fptr(inst, 1);
    float* k = inst_fptr(inst, 2);
    const float* inv_freq = inst_fptr(inst, 3);
    const int* position_ids = (const int*)(inst[4]);
    int num_q_heads = inst_int(inst, 5);
    int head_dim = inst_int(inst, 7);
    int section0_pairs = inst_int(inst, 9);
    int section1_pairs = inst_int(inst, 10);
    int rope_dim = inst_int(inst, 8);

    int head_idx = vblock;
    float* head_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + head_idx * head_dim;
    } else {
        head_ptr = k + (head_idx - num_q_heads) * head_dim;
    }

    int total_pairs = rope_dim / 2;
    for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
        int section = (pair < section0_pairs) ? 0 : (pair < section0_pairs + section1_pairs) ? 1 : 2;
        int pos = position_ids[section];
        float theta = (float)pos * inv_freq[pair];
        float cos_t = cosf(theta), sin_t = sinf(theta);
        int i0 = pair * 2, i1 = pair * 2 + 1;
        float x0 = head_ptr[i0], x1 = head_ptr[i1];
        head_ptr[i0] = x0 * cos_t - x1 * sin_t;
        head_ptr[i1] = x0 * sin_t + x1 * cos_t;
    }
}

// OP_GQA_ATTN: GQA decode attention (one block per Q head)
// args: [1]=output, [2]=q, [3]=k_cache, [4]=v_cache,
//       [5]=num_q_heads, [6]=num_kv_heads, [7]=head_dim, [8]=seq_len, [9]=max_seq_len
__device__ void op_gqa_attn(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* q = inst_fptr(inst, 2);
    const float* k_cache = inst_fptr(inst, 3);
    const float* v_cache = inst_fptr(inst, 4);
    int num_q_heads = inst_int(inst, 5);
    int num_kv_heads = inst_int(inst, 6);
    int head_dim = inst_int(inst, 7);
    int seq_len = inst_int(inst, 8);
    int max_seq_len = inst_int(inst, 9);

    int q_head = vblock;
    int gqa_group = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_group;
    float scale = rsqrtf((float)head_dim);
    const float* q_ptr = q + q_head * head_dim;

    extern __shared__ float shmem[];
    float* partial = shmem;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Single-pass online softmax with register V accumulation
    // acc[] holds one element per strided position; head_dim/block_size must be <= 4
    if (head_dim > 4 * block_size) __builtin_trap();
    float m = -1e30f, d = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += block_size) dot += q_ptr[i] * k_ptr[i];
        partial[tid] = dot;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        float score = partial[0] * scale;
        float m_new = fmaxf(m, score);
        float correction = expf(m - m_new);
        float w = expf(score - m_new);
        d = d * correction + w;
        const float* v_ptr = v_cache + (t * num_kv_heads + kv_head) * head_dim;
        int slot = 0;
        for (int i = tid; i < head_dim; i += block_size) {
            acc[slot] = acc[slot] * correction + w * v_ptr[i];
            slot++;
        }
        m = m_new;
        __syncthreads();
    }

    // Write output once (normalize by d)
    float* out_ptr = output + q_head * head_dim;
    int slot = 0;
    for (int i = tid; i < head_dim; i += block_size) {
        out_ptr[i] = acc[slot] / d;
        slot++;
    }
}

// OP_OUTPUT_GATE: output = attn_output * sigmoid(gate)
// args: [1]=output, [2]=attn_output, [3]=gate, [4]=size
__device__ void op_output_gate(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* attn_output = inst_fptr(inst, 2);
    const float* gate = inst_fptr(inst, 3);
    int size = inst_int(inst, 4);

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        output[idx] = attn_output[idx] / (1.0f + expf(-g));
    }
}

// OP_ATTN_PAGED: Paged GQA decode attention with lazy RoPE recurrence applied to K.
// One virtual block per Q head. Walks paged KV chunks, applies RoPE to pre-RoPE K
// via trig recurrence (4 FMA per pair per token step — no cos/sin lookup table).
// args: [1]=output, [2]=q (post-RoPE), [3]=page_table (u64* chunk base ptrs),
//       [4]=position_table (int* per-token positions), [5]=inv_freq (float*, rope_dim/2 freqs),
//       [6]=num_q_heads, [7]=num_kv_heads, [8]=head_dim, [9]=seq_len,
//       [10]=chunk_tokens(64), [11]=rope_dim, [12]=layer_k_offset_bytes, [13]=layer_v_offset_bytes
__device__ void op_attn_paged(const u64* inst, u32 vblock) {
    float* output          = inst_fptr(inst, 1);
    const float* q         = inst_fptr(inst, 2);
    const u64* page_table  = (const u64*)(inst[3]);
    const int* position_table = (const int*)(inst[4]);
    const float* inv_freq  = inst_fptr(inst, 5);
    int num_q_heads        = inst_int(inst, 6);
    int num_kv_heads       = inst_int(inst, 7);
    int head_dim           = inst_int(inst, 8);
    int seq_len            = inst_int(inst, 9);
    int chunk_tokens       = inst_int(inst, 10);
    int rope_dim           = inst_int(inst, 11);
    long long layer_k_offset = (long long)(u64)(inst[12]);
    long long layer_v_offset = (long long)(u64)(inst[13]);

    int q_head = vblock;
    if (q_head >= num_q_heads) return;

    int gqa_group = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_group;
    float scale = rsqrtf((float)head_dim);
    const float* q_ptr = q + q_head * head_dim;

    int tid = threadIdx.x;
    int block_size = blockDim.x;  // 256

    extern __shared__ float shmem[];
    float* sh_k   = shmem;          // head_dim floats
    float* partial = shmem;         // reused after dot product

    int rope_pairs = rope_dim / 2;

    // Precompute per-step trig increments (model-constant inv_freq).
    // Thread tid < rope_pairs computes cos_freq[tid] = cos(inv_freq[tid])
    // These are the same for every chunk so compute once.
    float cos_freq = 0.0f, sin_freq = 0.0f;
    if (tid < rope_pairs) {
        float freq = inv_freq[tid];
        cos_freq = cosf(freq);
        sin_freq = sinf(freq);
    }

    // Online softmax state
    float m = -1e30f, d = 0.0f;
    // V accumulation: head_dim=256, block_size=256, so exactly 1 element per thread
    float v_acc = 0.0f;

    int num_chunks = (seq_len + chunk_tokens - 1) / chunk_tokens;

    // Trig recurrence state per thread (for tid < rope_pairs)
    float cos_state = 0.0f, sin_state = 0.0f;

    for (int c = 0; c < num_chunks; c++) {
        const char* chunk_base = (const char*)(page_table[c]);
        const float* k_base = (const float*)(chunk_base + layer_k_offset);
        const float* v_base = (const float*)(chunk_base + layer_v_offset);
        int tokens_in_chunk = chunk_tokens;
        if (c == num_chunks - 1) {
            tokens_in_chunk = seq_len - c * chunk_tokens;
        }

        // Seed trig recurrence at chunk start from position_table
        if (tid < rope_pairs) {
            int pos0 = position_table[c * chunk_tokens];
            float theta0 = (float)pos0 * inv_freq[tid];
            cos_state = cosf(theta0);
            sin_state = sinf(theta0);
        }

        for (int t = 0; t < tokens_in_chunk; t++) {
            int token_idx = t * num_kv_heads + kv_head;

            // Load K into shared memory (head_dim == block_size == 256, one element per thread)
            if (tid < head_dim) {
                sh_k[tid] = k_base[token_idx * head_dim + tid];
            }
            __syncthreads();

            // Apply lazy RoPE rotation to sh_k (in-place, rope_dim elements only)
            // Threads 0..rope_pairs-1 handle pairs (tid*2, tid*2+1)
            if (tid < rope_pairs) {
                int i0 = tid * 2;
                int i1 = tid * 2 + 1;
                float x0 = sh_k[i0];
                float x1 = sh_k[i1];
                sh_k[i0] = x0 * cos_state - x1 * sin_state;
                sh_k[i1] = x0 * sin_state + x1 * cos_state;
            }
            __syncthreads();

            // Q·K dot product (one element per thread since head_dim == block_size)
            float dot = 0.0f;
            if (tid < head_dim) {
                dot = q_ptr[tid] * sh_k[tid];
            }
            // Reduce dot via shared memory (reuse sh_k alias partial)
            partial[tid] = dot;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) partial[tid] += partial[tid + s];
                __syncthreads();
            }
            float score = partial[0] * scale;

            // Online softmax update
            float m_new = fmaxf(m, score);
            float correction = expf(m - m_new);
            float w = expf(score - m_new);
            d = d * correction + w;

            // V accumulation (no RoPE on V)
            if (tid < head_dim) {
                float vval = v_base[token_idx * head_dim + tid];
                v_acc = v_acc * correction + w * vval;
            }
            m = m_new;

            // Advance trig recurrence to next token:
            // cos(θ+Δ) = cosθ·cosΔ - sinθ·sinΔ
            // sin(θ+Δ) = sinθ·cosΔ + cosθ·sinΔ
            if (tid < rope_pairs) {
                float cos_new = cos_state * cos_freq - sin_state * sin_freq;
                float sin_new = sin_state * cos_freq + cos_state * sin_freq;
                cos_state = cos_new;
                sin_state = sin_new;
            }
        }
    }

    // Write normalized output
    float* out_ptr = output + q_head * head_dim;
    if (tid < head_dim) {
        out_ptr[tid] = v_acc / d;
    }
}

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 16  // 16 u64s per instruction = 128 bytes

extern "C" __global__ void megakernel_f32(
    const u64* __restrict__ program,
    int num_instructions
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    for (int pc = 0; pc < num_instructions; pc++) {
        const u64* inst = program + pc * INST_SIZE;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);
        bool no_sync = inst_no_sync(inst);

        if (opcode == OP_HALT) break;

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
                default: break;
            }
        }

        if (!no_sync) grid.sync();
    }
}
