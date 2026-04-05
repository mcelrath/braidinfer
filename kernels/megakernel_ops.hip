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
#define B_TILE 8
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

    // Batched path: tile batch dimension to keep register pressure low (B_TILE accumulators)
    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ull2* w_vec = (const ull2*)w;
    extern __shared__ float shared[];

    for (int b0 = 0; b0 < batch; b0 += B_TILE) {
        int tile_end = (b0 + B_TILE < batch) ? b0 + B_TILE : batch;
        int tile_size = tile_end - b0;

        float acc[B_TILE];
        #pragma unroll
        for (int t = 0; t < B_TILE; t++) acc[t] = 0.0f;

        for (int vi = tid; vi < vec_count; vi += stride) {
            ull2 wv = __builtin_nontemporal_load(&w_vec[vi]);
            unsigned short ws[8];
            __builtin_memcpy(ws, &wv, 16);
            float wf[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) wf[j] = bf16_to_f32(ws[j]);

            int base = vi * vec_elements;
            for (int t = 0; t < tile_size; t++) {
                const float* inp = input_base + (long long)(b0 + t) * in_dim + base;
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[t] += wf[j] * inp[j];
            }
        }
        int rem = vec_count * vec_elements;
        for (int i = rem + tid; i < in_dim; i += stride) {
            float wf = bf16_to_f32(w[i]);
            for (int t = 0; t < tile_size; t++) acc[t] += wf * input_base[(long long)(b0 + t) * in_dim + i];
        }

        for (int t = 0; t < tile_size; t++) {
            shared[tid] = acc[t];
            __syncthreads();
            for (int s = stride / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = shared[0];
            __syncthreads();
        }
    }
}

// OP_SILU_MUL: output[i] = silu(gate[i]) * up[i]
// args: [1]=output, [2]=gate, [3]=up, [4]=size
__device__ void op_silu_mul(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* gate = inst_fptr(inst, 2);
    const float* up = inst_fptr(inst, 3);
    int size = inst_int(inst, 4);

    // vblock indexes chunks of blockDim.x elements
    int base = vblock * blockDim.x + threadIdx.x;
    if (base < size) {
        float g = gate[base];
        float silu_g = g / (1.0f + expf(-g));
        output[base] = silu_g * up[base];
    }
}

// OP_LINEAR_PROJ_RNF4: y = W_rnf4 @ x (one virtual block per output row, decode only)
// args: [1]=output, [2]=weight (packed rnf4), [3]=input, [4]=out_dim, [5]=in_dim
// Weight layout per row: groups of 128 elements, each 132 bytes.
//   group = [idx1_packed(64B), absmax1(2B), idx2_packed(64B), absmax2(2B)] = 132 bytes
//   idx1/idx2 are packed nibbles: 2 elements per byte, 64 bytes = 128 elements.
//
// Optimized: all 256 threads cooperate on one row, processing element pairs in stride.
// Each thread handles in_dim/512 pairs (2 elements each). No shared memory for input
// (read from global — L2 cached after first row in this block).
__device__ void op_linear_proj_rnf4(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const unsigned char* weight = (const unsigned char*)inst_ptr(inst, 2);
    const float* input_base = inst_fptr(inst, 3);
    int out_dim = inst_int(inst, 4);
    int in_dim = inst_int(inst, 5);
    int batch = inst_int(inst, 6);
    if (batch <= 1) batch = 1;

    int row = vblock;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
    const int num_pairs = in_dim / 2;
    extern __shared__ float shared[];

    if (batch == 1) {
        // Single-token path: flat loop over pairs with bit-shift group indexing.
        // group_size=128 (power of 2): use shifts for group/local_pair computation.
        float acc = 0.0f;
        for (int pair = tid; pair < num_pairs; pair += block_size) {
            int elem = pair << 1;     // pair * 2
            int g = elem >> 7;        // elem / 128
            int local_pair = (elem & 127) >> 1; // (elem % 128) / 2
            const unsigned char* gp = row_data + g * group_bytes;
            float am1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
            float am2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
            unsigned char b1 = gp[local_pair];
            unsigned char b2 = gp[66 + local_pair];
            float v0 = NF4_TABLE[b1 & 0xF] * am1 + NF4_TABLE[b2 & 0xF] * am2;
            float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * am1 + NF4_TABLE[(b2 >> 4) & 0xF] * am2;
            acc += v0 * input_base[elem] + v1 * input_base[elem + 1];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if ((tid & 31) == 0) shared[tid >> 5] = acc;
        __syncthreads();
        if (tid < 8) {
            acc = shared[tid];
            for (int offset = 4; offset > 0; offset >>= 1)
                acc += __shfl_down(acc, offset);
            if (tid == 0) output_base[row] = acc;
        }
        return;
    }

    // Batched path: tile over batch, dequant weight pairs once per tile
    for (int b0 = 0; b0 < batch; b0 += B_TILE) {
        int tile_end = (b0 + B_TILE < batch) ? b0 + B_TILE : batch;
        int tile_size = tile_end - b0;

        float acc[B_TILE];
        #pragma unroll
        for (int t = 0; t < B_TILE; t++) acc[t] = 0.0f;

        for (int pair = tid; pair < num_pairs; pair += block_size) {
            int elem = pair * 2;
            int g = elem / group_size;
            int local_pair = (elem % group_size) / 2;
            const unsigned char* gp = row_data + g * group_bytes;
            float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
            float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
            unsigned char b1 = gp[local_pair];
            unsigned char b2 = gp[66 + local_pair];
            float v0 = NF4_TABLE[b1 & 0xF] * absmax1 + NF4_TABLE[b2 & 0xF] * absmax2;
            float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * absmax1 + NF4_TABLE[(b2 >> 4) & 0xF] * absmax2;

            for (int t = 0; t < tile_size; t++) {
                const float* inp = input_base + (long long)(b0 + t) * in_dim;
                acc[t] += v0 * inp[elem] + v1 * inp[elem + 1];
            }
        }

        // Reduce via shared memory (same pattern as bf16 batched path)
        for (int t = 0; t < tile_size; t++) {
            shared[tid] = acc[t];
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = shared[0];
            __syncthreads();
        }
    }
}

// OP_LINEAR_PROJ_PCG32: y = W_pcg32 @ x (one virtual block per output row, decode only)
// args: [1]=output, [2]=weight (packed pcg32), [3]=input, [4]=out_dim, [5]=in_dim
// Weight layout per row: groups of 32 elements, each 20 bytes.
__device__ void op_linear_proj_pcg32(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const unsigned char* weight = (const unsigned char*)inst_ptr(inst, 2);
    const float* input_base = inst_fptr(inst, 3);
    int out_dim = inst_int(inst, 4);
    int in_dim = inst_int(inst, 5);
    int batch = inst_int(inst, 6);
    if (batch <= 1) batch = 1;

    int row = vblock;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
    extern __shared__ float shared[];

    if (batch == 1) {
        // Single-token fast path
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
                acc += v0 * input_base[base + i] + v1 * input_base[base + i + 1];
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
            if (tid == 0) output_base[row] = acc;
        }
        return;
    }

    // Batched path: loop over batch, reuse single-token logic per element
    for (int b = 0; b < batch; b++) {
        const float* input = input_base + (long long)b * in_dim;
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
        // Reduce via shared memory
        shared[tid] = acc;
        __syncthreads();
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        if (tid == 0) output_base[(long long)b * out_dim + row] = shared[0];
        __syncthreads();
    }
}

// OP_RMSNORM_WX: y = x * w / sqrt(mean(x^2) + eps) — standard w*x variant (Llama, OLMoE)
// Same args as OP_RMSNORM: [1]=output, [2]=input, [3]=weight, [4]=hidden_size, [5]=eps_bits
__device__ void op_rmsnorm_wx(const u64* inst, u32 vblock) {
    float* output_base = inst_fptr(inst, 1);
    const float* input_base = inst_fptr(inst, 2);
    const unsigned short* weight = inst_u16ptr(inst, 3);
    int hidden_size = inst_int(inst, 4);
    float eps = inst_float(inst, 5);

    // vblock = batch index (same as op_rmsnorm)
    float* output = output_base + (long long)vblock * hidden_size;
    const float* input = input_base + (long long)vblock * hidden_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = input[i];
        sum_sq += val * val;
    }
    extern __shared__ float shared[];
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        output[i] = input[i] * rms * bf16_to_f32(weight[i]);
    }
}

// OP_CONV1D: causal conv1d update with SiLU activation (one virtual block per channel)
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
    output[ch] = acc / (1.0f + expf(-acc)); // SiLU: x * sigmoid(x)

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
// args: [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b, [6]=state, [7]=output, [8]=key_dim, [9]=value_dim
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
    int gqa_group = inst_int(inst, 10);
    if (gqa_group <= 0) gqa_group = 1;

    const int h = vblock;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float decay = gate[h];
    const float beta = 1.0f / (1.0f + expf(-b[h]));

    const int kh_idx = h / gqa_group;
    const float* qh = q + kh_idx * key_dim;
    const float* kh = k + kh_idx * key_dim;
    const float* vh = v + h * value_dim;
    float* sh = state + h * key_dim * value_dim;
    float* oh = output + h * value_dim;

    // Shared memory: [0..block_size) for reductions, [block_size..block_size+key_dim) for k_norm cache
    extern __shared__ float shared[];
    float* sq = shared;
    float* sk = shared + block_size;

    // Q/K norm reduction
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

    // Pre-normalize k into shared memory (avoids redundant k[i]*k_inv_norm in inner loops)
    float* k_sh = shared + block_size;
    for (int i = tid; i < key_dim; i += block_size) k_sh[i] = kh[i] * k_inv_norm;
    __syncthreads();

    // Decay state: all 256 threads participate
    int total = key_dim * value_dim;
    for (int idx = tid; idx < total; idx += block_size) sh[idx] *= decay;
    __syncthreads();

    // Delta rule: compute kv_mem[j] = S[j,:] . k_norm, then delta[j] = beta*(v[j] - kv_mem[j])
    // Use shared memory to store kv_mem results (reuse sq region)
    float* kv_mem_sh = shared; // shared[0..value_dim)

    // Step 1: kv_mem[j] = dot(S[j,:], k_norm) — threads 0..value_dim-1
    for (int j = tid; j < value_dim; j += block_size) {
        float kv_mem = 0.0f;
        for (int i = 0; i < key_dim; i++) kv_mem += sh[j * key_dim + i] * k_sh[i];
        kv_mem_sh[j] = (vh[j] - kv_mem) * beta; // store delta directly
    }
    __syncthreads();

    // Step 2: S[j,i] += k_norm[i] * delta[j] — i-major loop, ALL threads participate
    // Each thread handles a stripe of the (j,i) space
    for (int idx = tid; idx < total; idx += block_size) {
        int j = idx / key_dim;
        int i = idx % key_dim;
        sh[idx] += k_sh[i] * kv_mem_sh[j];
    }
    __syncthreads();

    // Q readout: output[j] = S[j,:] . q_scaled
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

// OP_SIGMOID_WEIGHTED_ADD: output[i] += sigmoid(scalar[0]) * input[i]
// args: [1]=output (float*, [size]), [2]=scalar (float*, [1] — dot product to sigmoid),
//       [3]=input (float*, [size]), [4]=size
// Used for shared expert gating: output = ffn_down, scalar = gate@normed, input = shared_expert_out
__device__ void op_sigmoid_weighted_add(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* scalar = inst_fptr(inst, 2);
    const float* input = inst_fptr(inst, 3);
    int size = inst_int(inst, 4);

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) {
        float w = 1.0f / (1.0f + expf(-scalar[0]));
        output[idx] += w * input[idx];
    }
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

// OP_FFN_GATE_UP_RNF4: fused RMSNorm + gate_proj + up_proj + SiLU*up (rnf4 weights, decode n=1)
// args: [1]=scratch, [2]=input, [3]=rms_weight, [4]=w_gate (rnf4), [5]=w_up (rnf4),
//       [6]=hidden_size, [7]=intermediate_size, [8]=eps
// vblock = output row index in [0, intermediate_size)
__device__ void op_ffn_gate_up_rnf4(const u64* inst, u32 vblock) {
    float* scratch = inst_fptr(inst, 1);
    const float* input = inst_fptr(inst, 2);
    const unsigned short* rms_weight = inst_u16ptr(inst, 3);
    const unsigned char* w_gate = (const unsigned char*)inst_ptr(inst, 4);
    const unsigned char* w_up = (const unsigned char*)inst_ptr(inst, 5);
    int hidden_size = inst_int(inst, 6);
    int intermediate_size = inst_int(inst, 7);
    float eps = inst_float(inst, 8);

    int row = vblock;
    if (row >= intermediate_size) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (hidden_size + group_size - 1) / group_size;

    extern __shared__ float shared[];

    // RMSNorm: compute sum of squares
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

    const unsigned char* wg_row = w_gate + (long long)row * num_groups * group_bytes;
    const unsigned char* wu_row = w_up  + (long long)row * num_groups * group_bytes;
    float acc_gate = 0.0f, acc_up = 0.0f;

    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp_g = wg_row + g * group_bytes;
        const unsigned char* gp_u = wu_row + g * group_bytes;

        float absmax_g1 = bf16_to_f32(*(const unsigned short*)(gp_g + 64));
        float absmax_g2 = bf16_to_f32(*(const unsigned short*)(gp_g + 130));
        float absmax_u1 = bf16_to_f32(*(const unsigned short*)(gp_u + 64));
        float absmax_u2 = bf16_to_f32(*(const unsigned short*)(gp_u + 130));

        int base = g * group_size;
        int count = min(group_size, hidden_size - base);
        for (int i = 0; i < count; i += 2) {
            unsigned char bg1 = gp_g[i / 2];
            unsigned char bg2 = (gp_g + 66)[i / 2];
            unsigned char bu1 = gp_u[i / 2];
            unsigned char bu2 = (gp_u + 66)[i / 2];

            float normed0 = input[base + i] * rms_scale * (1.0f + bf16_to_f32(rms_weight[base + i]));
            float wg0 = NF4_TABLE[bg1 & 0xF] * absmax_g1 + NF4_TABLE[bg2 & 0xF] * absmax_g2;
            float wu0 = NF4_TABLE[bu1 & 0xF] * absmax_u1 + NF4_TABLE[bu2 & 0xF] * absmax_u2;
            acc_gate += wg0 * normed0;
            acc_up   += wu0 * normed0;

            if (i + 1 < count) {
                float normed1 = input[base + i + 1] * rms_scale * (1.0f + bf16_to_f32(rms_weight[base + i + 1]));
                float wg1 = NF4_TABLE[(bg1 >> 4) & 0xF] * absmax_g1 + NF4_TABLE[(bg2 >> 4) & 0xF] * absmax_g2;
                float wu1 = NF4_TABLE[(bu1 >> 4) & 0xF] * absmax_u1 + NF4_TABLE[(bu2 >> 4) & 0xF] * absmax_u2;
                acc_gate += wg1 * normed1;
                acc_up   += wu1 * normed1;
            }
        }
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

// OP_FFN_DOWN_RES_RNF4: down_proj(scratch) + residual (rnf4 weights, decode n=1)
// args: [1]=output, [2]=residual, [3]=w_down (rnf4), [4]=scratch, [5]=hidden_size, [6]=intermediate_size
// vblock = output row index in [0, hidden_size)
__device__ void op_ffn_down_res_rnf4(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* residual = inst_fptr(inst, 2);
    const unsigned char* w_down = (const unsigned char*)inst_ptr(inst, 3);
    const float* scratch = inst_fptr(inst, 4);
    int hidden_size = inst_int(inst, 5);
    int intermediate_size = inst_int(inst, 6);

    int row = vblock;
    if (row >= hidden_size) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (intermediate_size + group_size - 1) / group_size;
    const unsigned char* row_data = w_down + (long long)row * num_groups * group_bytes;

    float acc = 0.0f;
    for (int g = tid; g < num_groups; g += stride) {
        const unsigned char* gp = row_data + g * group_bytes;
        float absmax1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
        float absmax2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
        int base = g * group_size;
        int count = min(group_size, intermediate_size - base);
        for (int i = 0; i < count; i += 2) {
            unsigned char b1 = gp[i / 2];
            unsigned char b2 = (gp + 66)[i / 2];
            float v0 = NF4_TABLE[b1 & 0xF] * absmax1 + NF4_TABLE[b2 & 0xF] * absmax2;
            acc += v0 * scratch[base + i];
            if (i + 1 < count) {
                float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * absmax1 + NF4_TABLE[(b2 >> 4) & 0xF] * absmax2;
                acc += v1 * scratch[base + i + 1];
            }
        }
    }

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
    int batch = inst_int(inst, 9);
    if (batch <= 0) batch = 1;

    int heads_per_token = num_q_heads + num_kv_heads;
    int token_idx = vblock / heads_per_token;
    int head_idx = vblock % heads_per_token;

    float* head_ptr;
    const unsigned short* weight_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + (token_idx * num_q_heads + head_idx) * head_dim;
        weight_ptr = q_weight;
    } else {
        head_ptr = k + (token_idx * num_kv_heads + (head_idx - num_q_heads)) * head_dim;
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
    int num_kv_heads = inst_int(inst, 6);
    int head_dim = inst_int(inst, 7);
    int rope_dim = inst_int(inst, 8);
    int section0_pairs = inst_int(inst, 9);
    int section1_pairs = inst_int(inst, 10);
    int batch = inst_int(inst, 12);
    if (batch <= 0) batch = 1;

    int heads_per_token = num_q_heads + num_kv_heads;
    int token_idx = vblock / heads_per_token;
    int head_idx = vblock % heads_per_token;

    float* head_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + (token_idx * num_q_heads + head_idx) * head_dim;
    } else {
        head_ptr = k + (token_idx * num_kv_heads + (head_idx - num_q_heads)) * head_dim;
    }

    // Per-token position_ids: 3 ints per token (temporal, height, width)
    const int* tok_pos = position_ids + token_idx * 3;

    int total_pairs = rope_dim / 2;
    for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
        int section = (pair < section0_pairs) ? 0 : (pair < section0_pairs + section1_pairs) ? 1 : 2;
        int pos = tok_pos[section];
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
    if (head_dim > 4 * block_size) return; // head_dim too large for fixed acc[4] array
    float m = -1e30f, d = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + ((long)kv_head * max_seq_len + t) * head_dim;
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
        const float* v_ptr = v_cache + ((long)kv_head * max_seq_len + t) * head_dim;
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
// One virtual block per Q head. Walks paged KV chunks, applies QK-norm + RoPE to pre-norm K
// via trig recurrence (4 FMA per pair per token step — no cos/sin lookup table).
// Cache stores pre-QK-norm K for quantization quality (post-norm K is ±0.06, catastrophic for Q4).
// args: [1]=output, [2]=q (post-QK-norm, post-RoPE), [3]=page_table (u64* chunk base ptrs),
//       [4]=position_table (int* per-token positions), [5]=inv_freq (float*, rope_dim/2 freqs),
//       [6]=num_q_heads, [7]=num_kv_heads, [8]=head_dim, [9]=seq_len,
//       [10]=chunk_tokens(64), [11]=rope_dim, [12]=layer_k_offset_bytes, [13]=layer_v_offset_bytes,
//       [14]=partial_state (float* or 0; if non-null, init (m,d,v_acc) from quantized pass scratch)
//       [16]=k_norm_weight (bf16* or null; if non-null, apply QK-norm after K load/dequant)
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
    const float* partial_state = (const float*)(inst[14]); // 0 = none

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

    float cos_freq = 0.0f, sin_freq = 0.0f;
    if (tid < rope_pairs) {
        float freq = inv_freq[tid];
        cos_freq = cosf(freq);
        sin_freq = sinf(freq);
    }

    // Online softmax state — init from quantized pass if available
    float m = -1e30f, d = 0.0f;
    float v_acc = 0.0f;
    if (partial_state != nullptr) {
        int stride_per_head = 2 + head_dim;
        const float* ps = partial_state + q_head * stride_per_head;
        m = ps[0];
        d = ps[1];
        if (tid < head_dim) {
            v_acc = ps[2 + tid];
        }
    }

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
            int token_idx = kv_head * chunk_tokens + t;

            // Load K into shared memory (head_dim == block_size == 256, one element per thread)
            if (tid < head_dim) {
                sh_k[tid] = k_base[token_idx * head_dim + tid];
            }
            __syncthreads();

            // Apply QK-norm to cached K (cache stores pre-norm K for quantization quality).
            // k[i] = k[i] * rsqrt(sum_sq/head_dim + eps) * (1 + k_norm_weight[i])
            // k_norm_weight is passed as inst[16] (bf16 pointer, or null for non-QK-norm models)
            // NOTE: partial aliases sh_k (both = shmem). Save K to register before reduction.
            {
                const unsigned short* kn_w = (const unsigned short*)(inst[16]);
                if (kn_w != nullptr) {
                    float my_k = (tid < head_dim) ? sh_k[tid] : 0.0f;
                    partial[tid] = my_k * my_k;  // overwrites sh_k via alias
                    __syncthreads();
                    for (int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) partial[tid] += partial[tid + s];
                        __syncthreads();
                    }
                    float k_rms = rsqrtf(partial[0] / (float)head_dim + 1e-6f);
                    if (tid < head_dim) {
                        float w = 1.0f + bf16_to_f32(kn_w[tid]);
                        sh_k[tid] = my_k * k_rms * w;  // use saved register value
                    }
                    __syncthreads();
                }
            }

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
                float vval = v_base[(kv_head * chunk_tokens + t) * head_dim + tid];
                v_acc = v_acc * correction + w * vval;
            }
            m = m_new;

            // Advance trig recurrence to next token (assumes contiguous positions within chunk).
            // TODO(braidinfer-lcl): For prefix caching with non-contiguous positions,
            // recompute cos/sin from position_table[c*chunk_tokens+t] instead.
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

// OP_ATTN_PAGED_Q: Paged GQA attention with residual_pc int4 quantized KV cache.
// Writes PARTIAL softmax state to scratch buffer (not final output).
// OP_ATTN_PAGED reads the scratch and merges with f32 active chunk.
// Cache stores pre-QK-norm K — QK-norm applied after dequant for quantization quality.
// args: [1]=scratch (float*, layout [nqh × (2+head_dim)]: m, d, v_acc per head),
//       [2]=q (post-QK-norm, post-RoPE), [3]=page_table (u64*, quantized chunks only),
//       [4]=position_table (int*), [5]=inv_freq (float*),
//       [6]=num_q_heads, [7]=num_kv_heads, [8]=head_dim, [9]=seq_len (of quantized chunks),
//       [10]=chunk_tokens(64), [11]=rope_dim,
//       [12]=layer_k_q1data_off, [13]=layer_k_q1scale_off,
//       [14]=layer_k_rdata_off, [15]=layer_k_rscale_off
//       [16]=k_norm_weight (bf16* or null; if non-null, apply QK-norm after dequant)
// V offsets = K offsets + per_kv stride, computed from nkh/ct/hd.
__device__ void op_attn_paged_quant(const u64* inst, u32 vblock) {
    float* scratch         = inst_fptr(inst, 1);
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
    // Byte offsets within each chunk for this layer's K quantized data
    long long k_q1data_off  = (long long)(u64)(inst[12]);
    long long k_q1scale_off = (long long)(u64)(inst[13]);
    long long k_rdata_off   = (long long)(u64)(inst[14]);
    long long k_rscale_off  = (long long)(u64)(inst[15]);

    // V offsets: per_kv bytes after K offsets
    int data_bytes = num_kv_heads * chunk_tokens * head_dim / 2;
    int scale_bytes = num_kv_heads * head_dim * 4;
    int per_kv = 2 * (data_bytes + scale_bytes);
    long long v_q1data_off  = k_q1data_off + per_kv;
    long long v_q1scale_off = k_q1scale_off + per_kv;
    long long v_rdata_off   = k_rdata_off + per_kv;
    long long v_rscale_off  = k_rscale_off + per_kv;

    int q_head = vblock;
    if (q_head >= num_q_heads) return;

    int gqa_group = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_group;
    float attn_scale = rsqrtf((float)head_dim);
    const float* q_ptr = q + q_head * head_dim;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float shmem[];
    float* sh_k = shmem;       // [head_dim]
    float* partial = shmem;    // reused for reduction

    int rope_pairs = rope_dim / 2;

    float cos_freq = 0.0f, sin_freq = 0.0f;
    if (tid < rope_pairs) {
        float freq = inv_freq[tid];
        cos_freq = cosf(freq);
        sin_freq = sinf(freq);
    }

    float m = -1e30f, d_softmax = 0.0f;
    float v_acc = 0.0f;

    int num_chunks = (seq_len + chunk_tokens - 1) / chunk_tokens;
    float cos_state = 0.0f, sin_state = 0.0f;

    for (int c = 0; c < num_chunks; c++) {
        const char* chunk_base = (const char*)(page_table[c]);
        int tokens_in_chunk = chunk_tokens;
        if (c == num_chunks - 1) {
            tokens_in_chunk = seq_len - c * chunk_tokens;
        }

        // Load K scales for this head into registers (once per chunk)
        // K q1_scale layout: [nkh, hd] f32 at chunk_base + k_q1scale_off
        const float* k_q1s = (const float*)(chunk_base + k_q1scale_off);
        const float* k_rs  = (const float*)(chunk_base + k_rscale_off);
        float k_s1 = 0.0f, k_s2 = 0.0f;
        if (tid < head_dim) {
            k_s1 = k_q1s[kv_head * head_dim + tid];
            k_s2 = k_rs[kv_head * head_dim + tid];
        }

        // Load V scales
        const float* v_q1s = (const float*)(chunk_base + v_q1scale_off);
        const float* v_rs  = (const float*)(chunk_base + v_rscale_off);
        float v_s1 = 0.0f, v_s2 = 0.0f;
        if (tid < head_dim) {
            v_s1 = v_q1s[kv_head * head_dim + tid];
            v_s2 = v_rs[kv_head * head_dim + tid];
        }

        // K/V packed data pointers
        const unsigned char* k_q1d = (const unsigned char*)(chunk_base + k_q1data_off);
        const unsigned char* k_rd  = (const unsigned char*)(chunk_base + k_rdata_off);
        const unsigned char* v_q1d = (const unsigned char*)(chunk_base + v_q1data_off);
        const unsigned char* v_rd  = (const unsigned char*)(chunk_base + v_rdata_off);

        // Seed trig recurrence
        if (tid < rope_pairs) {
            int pos0 = position_table[c * chunk_tokens];
            float theta0 = (float)pos0 * inv_freq[tid];
            cos_state = cosf(theta0);
            sin_state = sinf(theta0);
        }

        for (int t = 0; t < tokens_in_chunk; t++) {
            // Dequantize K for this token into shared memory
            if (tid < head_dim) {
                int pair = t / 2;
                int idx = (kv_head * (chunk_tokens / 2) + pair) * head_dim + tid;
                unsigned char q1_byte = k_q1d[idx];
                unsigned char r_byte = k_rd[idx];
                int q1_nibble, r_nibble;
                if (t % 2 == 0) {
                    q1_nibble = (int)(q1_byte & 0xF);
                    r_nibble = (int)(r_byte & 0xF);
                } else {
                    q1_nibble = (int)((q1_byte >> 4) & 0xF);
                    r_nibble = (int)((r_byte >> 4) & 0xF);
                }
                // Sign extend from 4-bit
                if (q1_nibble > 7) q1_nibble -= 16;
                if (r_nibble > 7) r_nibble -= 16;
                sh_k[tid] = (float)q1_nibble * k_s1 + (float)r_nibble * k_s2;
            }
            __syncthreads();

            // Apply QK-norm to dequantized K (cache stores pre-norm K for quantization quality).
            // k_norm_weight passed as inst[16] (bf16 pointer, or 0 for non-QK-norm models)
            // NOTE: partial aliases sh_k (both = shmem). Save K to register before reduction.
            {
                const unsigned short* kn_w = (const unsigned short*)(inst[16]);
                if (kn_w != nullptr) {
                    float my_k = (tid < head_dim) ? sh_k[tid] : 0.0f;
                    partial[tid] = my_k * my_k;
                    __syncthreads();
                    for (int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) partial[tid] += partial[tid + s];
                        __syncthreads();
                    }
                    float k_rms = rsqrtf(partial[0] / (float)head_dim + 1e-6f);
                    if (tid < head_dim) {
                        float w = 1.0f + bf16_to_f32(kn_w[tid]);
                        sh_k[tid] = my_k * k_rms * w;
                    }
                    __syncthreads();
                }
            }

            // Apply lazy RoPE
            if (tid < rope_pairs) {
                int i0 = tid * 2;
                int i1 = tid * 2 + 1;
                float x0 = sh_k[i0];
                float x1 = sh_k[i1];
                sh_k[i0] = x0 * cos_state - x1 * sin_state;
                sh_k[i1] = x0 * sin_state + x1 * cos_state;
            }
            __syncthreads();

            // Q·K dot product
            float dot = 0.0f;
            if (tid < head_dim) dot = q_ptr[tid] * sh_k[tid];
            partial[tid] = dot;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) partial[tid] += partial[tid + s];
                __syncthreads();
            }
            float score = partial[0] * attn_scale;

            // Online softmax
            float m_new = fmaxf(m, score);
            float correction = expf(m - m_new);
            float w = expf(score - m_new);
            d_softmax = d_softmax * correction + w;

            // Dequantize V and accumulate (no RoPE on V)
            if (tid < head_dim) {
                int pair = t / 2;
                int idx = (kv_head * (chunk_tokens / 2) + pair) * head_dim + tid;
                unsigned char vq1_byte = v_q1d[idx];
                unsigned char vr_byte = v_rd[idx];
                int vq1, vr;
                if (t % 2 == 0) {
                    vq1 = (int)(vq1_byte & 0xF);
                    vr = (int)(vr_byte & 0xF);
                } else {
                    vq1 = (int)((vq1_byte >> 4) & 0xF);
                    vr = (int)((vr_byte >> 4) & 0xF);
                }
                if (vq1 > 7) vq1 -= 16;
                if (vr > 7) vr -= 16;
                float vval = (float)vq1 * v_s1 + (float)vr * v_s2;
                v_acc = v_acc * correction + w * vval;
            }
            m = m_new;

            // Advance trig recurrence
            if (tid < rope_pairs) {
                float cos_new = cos_state * cos_freq - sin_state * sin_freq;
                float sin_new = sin_state * cos_freq + cos_state * sin_freq;
                cos_state = cos_new;
                sin_state = sin_new;
            }
        }
    }

    // Write partial state: [m, d, v_acc[0..head_dim]] per head
    int stride_per_head = 2 + head_dim;
    float* head_scratch = scratch + q_head * stride_per_head;
    if (tid == 0) {
        head_scratch[0] = m;
        head_scratch[1] = d_softmax;
    }
    if (tid < head_dim) {
        head_scratch[2 + tid] = v_acc;
    }
}

// OP_ATTN_PREFILL: Batched causal GQA attention for prefill.
// N queries attend to flat KV cache with causal masking (query t sees KV 0..start_pos+t).
// One virtual block per (token, q_head) pair: vblock = token_idx * num_q_heads + q_head.
// args: [1]=output([N*nqh*hd]), [2]=q([N*nqh*hd]), [3]=k_cache, [4]=v_cache,
//       [5]=num_q_heads, [6]=num_kv_heads, [7]=head_dim, [8]=start_pos,
//       [9]=num_tokens(N), [10]=max_seq_len
__device__ void op_attn_prefill(const u64* inst, u32 vblock) {
    float* output = inst_fptr(inst, 1);
    const float* q = inst_fptr(inst, 2);
    const float* k_cache = inst_fptr(inst, 3);
    const float* v_cache = inst_fptr(inst, 4);
    int num_q_heads = inst_int(inst, 5);
    int num_kv_heads = inst_int(inst, 6);
    int head_dim = inst_int(inst, 7);
    int start_pos = inst_int(inst, 8);
    int num_tokens = inst_int(inst, 9);
    int max_seq_len = inst_int(inst, 10);

    int token_idx = vblock / num_q_heads;
    int q_head = vblock % num_q_heads;
    if (token_idx >= num_tokens) return;

    int gqa_group = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_group;
    float scale = rsqrtf((float)head_dim);
    const float* q_ptr = q + (token_idx * num_q_heads + q_head) * head_dim;

    // Causal: this query at position start_pos+token_idx sees KV 0..start_pos+token_idx
    int seq_len = start_pos + token_idx + 1;

    extern __shared__ float shmem[];
    float* partial = shmem;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Online softmax with register V accumulation (same as op_gqa_attn)
    if (head_dim > 4 * block_size) return; // head_dim too large for fixed acc[4] array
    float m = -1e30f, d = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + ((long)kv_head * max_seq_len + t) * head_dim;
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
        const float* v_ptr = v_cache + ((long)kv_head * max_seq_len + t) * head_dim;
        int slot = 0;
        for (int i = tid; i < head_dim; i += block_size) {
            acc[slot] = acc[slot] * correction + w * v_ptr[i];
            slot++;
        }
        m = m_new;
        __syncthreads();
    }

    float* out_ptr = output + (token_idx * num_q_heads + q_head) * head_dim;
    int slot = 0;
    for (int i = tid; i < head_dim; i += block_size) {
        out_ptr[i] = acc[slot] / d;
        slot++;
    }
}

// OP_DEINTERLEAVE: Split [batch * num_heads * 2*hd] interleaved → two [batch * num_heads * hd] outputs.
// For each head h, copies src[h*2*hd .. h*2*hd+hd] → dst_even[h*hd], src[h*2*hd+hd .. h*2*hd+2*hd] → dst_odd[h*hd].
// args: [1]=dst_even, [2]=dst_odd, [3]=src, [4]=num_heads, [5]=head_dim, [6]=batch
// grid_x = total elements / blockDim.x
__device__ void op_deinterleave(const u64* inst, u32 vblock) {
    float* dst_even = inst_fptr(inst, 1);
    float* dst_odd = inst_fptr(inst, 2);
    const float* src = inst_fptr(inst, 3);
    int num_heads = inst_int(inst, 4);
    int head_dim = inst_int(inst, 5);
    int batch = inst_int(inst, 6);
    if (batch <= 0) batch = 1;

    int total = batch * num_heads * head_dim;
    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // idx maps to (batch_idx, head, element)
    int elem = idx % head_dim;
    int rest = idx / head_dim;
    int head = rest % num_heads;
    int batch_idx = rest / num_heads;

    int src_offset = (batch_idx * num_heads + head) * 2 * head_dim;
    int dst_offset = (batch_idx * num_heads + head) * head_dim + elem;

    dst_even[dst_offset] = src[src_offset + elem];
    dst_odd[dst_offset] = src[src_offset + head_dim + elem];
}

// OP_KV_QUANTIZE: Quantize a full f32 chunk (64 tokens) to residual_pc int4.
// One virtual block per channel (nkh * head_dim). Each block quantizes 64 values.
// args: [1]=f32_src (staging buffer region for this K or V, layout [nkh, chunk_tokens, hd])
//       [2]=q1_data_dst (int4 packed output, [nkh * chunk_tokens * hd / 2] bytes)
//       [3]=q1_scale_dst (f32 per-channel scales, [nkh * hd])
//       [4]=r_data_dst (residual int4 packed)
//       [5]=r_scale_dst (residual f32 scales)
//       [6]=num_kv_heads, [7]=head_dim, [8]=chunk_tokens (64)
// grid_x = nkh * head_dim (one block per channel)
__device__ void op_kv_quantize(const u64* inst, u32 vblock) {
    const float* src = inst_fptr(inst, 1);
    unsigned char* q1_data = (unsigned char*)(inst[2]);
    float* q1_scale = inst_fptr(inst, 3);
    unsigned char* r_data = (unsigned char*)(inst[4]);
    float* r_scale = inst_fptr(inst, 5);
    int nkh = inst_int(inst, 6);
    int hd = inst_int(inst, 7);
    int ct = inst_int(inst, 8);

    int channel = vblock;
    int total_channels = nkh * hd;
    if (channel >= total_channels) return;

    int head = channel / hd;
    int dim = channel % hd;
    int tid = threadIdx.x;

    // Each channel has ct (64) values at stride hd: src[head * ct * hd + t * hd + dim]
    // With 256 threads and 64 values, first 64 threads each handle one value
    extern __shared__ float shmem[];
    float* vals = shmem;             // [ct] f32 values for this channel
    float* residuals = shmem + ct;   // [ct] residuals

    // Load values into shared memory
    if (tid < ct) {
        vals[tid] = src[(long long)head * ct * hd + tid * hd + dim];
    }
    __syncthreads();

    // Pass 1: find max absolute value for scale
    float local_max = 0.0f;
    if (tid < ct) {
        local_max = fabsf(vals[tid]);
    }
    // Reduce max across threads (only first ct=64 threads have data, but reduce all 256)
    shmem[256 + tid] = local_max;  // use extra shmem region for reduction
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) shmem[256 + tid] = fmaxf(shmem[256 + tid], shmem[256 + tid + s]);
        __syncthreads();
    }
    float max_val = shmem[256];
    float s1 = (max_val > 0.0f) ? max_val / 7.0f : 1.0f;  // symmetric: [-8,7] mapped to [-8s,7s]
    float inv_s1 = 1.0f / s1;

    // Quantize pass 1 and compute residuals
    if (tid < ct) {
        float v = vals[tid];
        int q = __float2int_rn(v * inv_s1);
        q = max(-8, min(7, q));
        vals[tid] = (float)q;  // store quantized int for packing
        residuals[tid] = v - (float)q * s1;
    }
    __syncthreads();

    // Pass 2: find max absolute residual
    float local_rmax = 0.0f;
    if (tid < ct) {
        local_rmax = fabsf(residuals[tid]);
    }
    shmem[256 + tid] = local_rmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) shmem[256 + tid] = fmaxf(shmem[256 + tid], shmem[256 + tid + s]);
        __syncthreads();
    }
    float rmax_val = shmem[256];
    float s2 = (rmax_val > 0.0f) ? rmax_val / 7.0f : 1.0f;
    float inv_s2 = 1.0f / s2;

    // Quantize residuals
    if (tid < ct) {
        float r = residuals[tid];
        int q = __float2int_rn(r * inv_s2);
        residuals[tid] = (float)max(-8, min(7, q));
    }
    __syncthreads();

    // Pack int4 values: 2 values per byte, low-nibble-first
    // Layout: q1_data[head * ct/2 * hd + pair * hd + dim] for pair in 0..ct/2
    // Each thread packs one pair (2 consecutive tokens)
    if (tid < ct / 2) {
        int even = (int)vals[tid * 2];
        int odd = (int)vals[tid * 2 + 1];
        unsigned char packed = ((unsigned char)(odd & 0xF) << 4) | ((unsigned char)(even & 0xF));
        long long data_idx = (long long)head * (ct / 2) * hd + tid * hd + dim;
        q1_data[data_idx] = packed;

        int r_even = (int)residuals[tid * 2];
        int r_odd = (int)residuals[tid * 2 + 1];
        unsigned char r_packed = ((unsigned char)(r_odd & 0xF) << 4) | ((unsigned char)(r_even & 0xF));
        r_data[data_idx] = r_packed;
    }

    // Write scales (one thread per channel)
    if (tid == 0) {
        q1_scale[channel] = s1;
        r_scale[channel] = s2;
    }
}
