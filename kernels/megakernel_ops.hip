// ============================================================
// Operation implementations
// Each takes the instruction args and a virtual block index.
// threadIdx.x is the thread within the block (0..255).
// ============================================================

// Wave32 warp-reduce sum (RDNA3/gfx1100). Works within a 32-thread wavefront.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

// Block-reduce sum across 256 threads (8 warps × 32 lanes).
// Uses shared[0..7] for warp partial sums. Ends with __syncthreads() so callers
// can safely do sequential block_reduce_sum calls or reuse shared immediately.
// Returns total sum to thread 0; other threads return their warp's partial sum.
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    val = warp_reduce_sum(val);
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        val = 0.0f;
        // blockDim.x == 256 → 8 warps; unroll manually
        val += shared[0]; val += shared[1]; val += shared[2]; val += shared[3];
        val += shared[4]; val += shared[5]; val += shared[6]; val += shared[7];
        shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// bz0 experiment 1: pure sequential reduction, no __shfl_down. All threads
// write to shared, thread 0 sums sequentially. Diagnostic — eliminates
// shfl as a possible non-determinism source. Slower than block_reduce_sum.
__device__ __forceinline__ float sequential_reduce_sum_256(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 256; i++) sum += shared[i];
        shared[0] = sum;
    }
    __syncthreads();
    return shared[0];
}

// Shared-memory tree reduction: 256 → 1 in 8 parallel stages, no __shfl_down.
// Confirmed deterministic on gfx1100 wave32 (replaces shfl-based reduction at
// op_attn_paged K-rms which was source of bz0). Faster than sequential.
__device__ __forceinline__ float tree_reduce_sum_256(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// OP_RMSNORM: y = x * rsqrt(mean(x²)+eps) * (1+weight)
// args: [1]=output, [2]=input, [3]=weight, [4]=hidden_size, [5]=eps_bits
// grid_x = batch (default 1). vblock = batch index. Each block normalizes one row.
__device__ __attribute__((noinline)) void op_rmsnorm(const u64* inst, u32 vblock) {
    const RmsNormInst* i = (const RmsNormInst*)inst;
    float* output_base = i->output;
    const float* input_base = i->input;
    const uint16_t* weight = i->weight;
    int hidden_size = (int)i->dim;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);

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
    float rms = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)hidden_size + eps);

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
__device__ __attribute__((noinline)) void op_linear_proj(const u64* inst, u32 vblock) {
    const LinearProjInst* i = (const LinearProjInst*)inst;
    float* output_base = i->output;
    const uint16_t* weight = (const uint16_t*)i->weight;
    const float* input_base = i->input;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
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
        float total = block_reduce_sum(acc, shared);
        if (tid == 0) output_base[row] = total;
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
            float total = block_reduce_sum(acc[t], shared);
            if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = total;
        }
    }
}

// OP_LINEAR_PROJ_2X: bf16 fused two linear projections sharing the same input.
// vb < out_dim → write to output_a using weight_a; vb < 2*out_dim → write to output_b using
// weight_b. Same input/in_dim/out_dim/batch for both. Used by GDN w_a + w_b (always bf16).
// Saves one grid.sync vs two sequential OP_LINEAR_PROJ instructions.
__device__ __attribute__((noinline)) void op_linear_proj_2x(const u64* inst, u32 vblock) {
    const LinearProj2xInst* i = (const LinearProj2xInst*)inst;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
    if (batch <= 1) batch = 1;

    // Block routing: which sub-projection is this block working on?
    float* output_base;
    const uint16_t* weight;
    int row;
    if ((int)vblock < out_dim) {
        output_base = i->output_a;
        weight = i->weight_a;
        row = (int)vblock;
    } else if ((int)vblock < 2 * out_dim) {
        output_base = i->output_b;
        weight = i->weight_b;
        row = (int)vblock - out_dim;
    } else {
        return;
    }

    const float* input_base = i->input;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned short* w = weight + (long long)row * in_dim;
    extern __shared__ float shared[];

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
        for (int k = rem + tid; k < in_dim; k += stride) acc += bf16_to_f32(w[k]) * input_base[k];
        float total = block_reduce_sum(acc, shared);
        if (tid == 0) output_base[row] = total;
        return;
    }

    // Batched: tile B_TILE batch rows at a time
    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ull2* w_vec = (const ull2*)w;
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
        for (int k = rem + tid; k < in_dim; k += stride) {
            float wf = bf16_to_f32(w[k]);
            for (int t = 0; t < tile_size; t++) acc[t] += wf * input_base[(long long)(b0 + t) * in_dim + k];
        }
        for (int t = 0; t < tile_size; t++) {
            float total = block_reduce_sum(acc[t], shared);
            if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = total;
        }
    }
}

// OP_SILU_MUL: output[i] = silu(gate[i]) * up[i]
// args: [1]=output, [2]=gate, [3]=up, [4]=size
__device__ __attribute__((noinline)) void op_silu_mul(const u64* inst, u32 vblock) {
    const SiluMulInst* i = (const SiluMulInst*)inst;
    float* output = i->output;
    const float* gate = i->gate;
    const float* up = i->up;
    int size = (int)i->size;

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
// Tiled LDS optimization constants for linear_proj quantized ops.
// Each tile is 7680 floats (30KB). Shared memory layout per block:
//   [0..7]          warp reduction sums (8 floats = 32 bytes)
//   [8..8+7679]     input tile (7680 floats = 30720 bytes)
//   [8+7680..8+7680+255]  partial outputs per row (256 floats = 1024 bytes)
// Total: (8+7680+256)*4 = 31776 bytes. 2 blocks/CU: 63552 < 65536 ✓
#define LPROJ_TILE_FLOATS    7680   // 240*32 PCG32 groups or 60*128 RNF4 groups
#define LPROJ_MAX_ROWS        256   // max rows per block; limits out_dim ≤ 256*gridDim.x
#define SHARED_LPROJ_TOTAL  31776   // (8 + LPROJ_TILE_FLOATS + LPROJ_MAX_ROWS) * 4 bytes per block

// OP_LINEAR_PROJ_RNF4 (full, tiled-LDS): loads input tile once per block,
// processes all assigned rows per tile. One call replaces the virtual block loop.
// batch=1 (decode) only; batch>1 falls back to per-row approach.
__device__ __attribute__((noinline)) void op_linear_proj_rnf4_full(const u64* inst, u32 grid_x) {
    const LinearProjInst* i = (const LinearProjInst*)inst;
    float* output_base = i->output;
    const unsigned char* weight = i->weight;
    const float* input_base = i->input;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
    if (batch <= 1) batch = 1;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    extern __shared__ float shared[];

    if (batch == 1) {
        float* lds_tile    = shared + 8;
        float* row_partial = shared + 8 + LPROJ_TILE_FLOATS;

        // Count and initialize row partials
        int num_rows = 0;
        for (u32 r = blockIdx.x; r < (u32)out_dim; r += gridDim.x) num_rows++;
        for (int ri = tid; ri < num_rows; ri += stride) row_partial[ri] = 0.0f;
        __syncthreads();

        const int groups_per_tile = LPROJ_TILE_FLOATS / group_size;  // 60
        const int n_tiles = (num_groups + groups_per_tile - 1) / groups_per_tile;
        const int num_pairs = in_dim / 2;

        for (int t = 0; t < n_tiles; t++) {
            int g0 = t * groups_per_tile;
            int g1 = min(g0 + groups_per_tile, num_groups);
            int tile_start = g0 * group_size;
            int tile_len = min((g1 - g0) * group_size, in_dim - tile_start);

            // Load input tile cooperatively
            for (int i = tid; i < tile_len; i += stride) lds_tile[i] = input_base[tile_start + i];
            __syncthreads();

            int ri = 0;
            for (u32 row = blockIdx.x; row < (u32)out_dim; row += gridDim.x, ri++) {
                const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
                float acc = 0.0f;
                // process pairs covering this tile's groups
                int p0 = (tile_start) / 2;
                int p1 = min((g1 * group_size) / 2, num_pairs);
                for (int pair = p0 + tid; pair < p1; pair += stride) {
                    int elem = pair * 2;
                    int g = elem >> 7;        // elem / 128
                    int local_pair = (elem & 127) >> 1;
                    const unsigned char* gp = row_data + g * group_bytes;
                    float am1 = bf16_to_f32(*(const unsigned short*)(gp + 64));
                    float am2 = bf16_to_f32(*(const unsigned short*)(gp + 130));
                    unsigned char b1 = gp[local_pair];
                    unsigned char b2 = gp[66 + local_pair];
                    float v0 = NF4_TABLE[b1 & 0xF] * am1 + NF4_TABLE[b2 & 0xF] * am2;
                    float v1 = NF4_TABLE[(b1 >> 4) & 0xF] * am1 + NF4_TABLE[(b2 >> 4) & 0xF] * am2;
                    int local_elem = elem - tile_start;
                    acc += v0 * lds_tile[local_elem] + v1 * lds_tile[local_elem + 1];
                }
                for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
                if ((tid & 31) == 0) shared[tid >> 5] = acc;
                __syncthreads();
                // Reduce 8 warp partials with thread 0 only. The previous code did
                // __shfl_down inside an `if (tid < 8)` block — on RDNA3 wave32, shfl
                // from an inactive lane is undefined and observably non-deterministic
                // (AMD ISA: lane reads source register; if source lane is masked off
                // its register state is implementation-defined). bz0 Source 3 root cause.
                if (tid == 0) {
                    float total = shared[0] + shared[1] + shared[2] + shared[3]
                                + shared[4] + shared[5] + shared[6] + shared[7];
                    row_partial[ri] += total;
                }
                __syncthreads();
            }
            __syncthreads();  // Before overwriting lds_tile with next tile
        }

        for (u32 row = blockIdx.x, ri = 0; row < (u32)out_dim; row += gridDim.x, ri++) {
            if (tid == 0) output_base[row] = row_partial[ri];
        }
        return;
    }

    // batch > 1 (prefill): existing per-row approach
    const int num_pairs = in_dim / 2;
    for (u32 row = blockIdx.x; row < (u32)out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
        for (int b0 = 0; b0 < batch; b0 += B_TILE) {
            int tile_end = (b0 + B_TILE < batch) ? b0 + B_TILE : batch;
            int tile_size = tile_end - b0;
            float acc[B_TILE];
            #pragma unroll
            for (int t = 0; t < B_TILE; t++) acc[t] = 0.0f;
            for (int pair = tid; pair < num_pairs; pair += stride) {
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
            for (int t = 0; t < tile_size; t++) {
                float total = block_reduce_sum(acc[t], shared);
                if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = total;
            }
        }
    }
}

// Optimized: all 256 threads cooperate on one row, processing element pairs in stride.
// Each thread handles in_dim/512 pairs (2 elements each). No shared memory for input
// (read from global — L2 cached after first row in this block).
__device__ __attribute__((noinline)) void op_linear_proj_rnf4(const u64* inst, u32 vblock) {
    const LinearProjInst* i = (const LinearProjInst*)inst;
    float* output_base = i->output;
    const unsigned char* weight = i->weight;
    const float* input_base = i->input;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
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
        // bz0 fix: see comment in op_linear_proj_rnf4_full above. shfl from inactive
        // lanes is undefined on RDNA3 wave32; use sequential thread-0 sum.
        if (tid == 0) {
            float total = shared[0] + shared[1] + shared[2] + shared[3]
                        + shared[4] + shared[5] + shared[6] + shared[7];
            output_base[row] = total;
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
            float total = block_reduce_sum(acc[t], shared);
            if (tid == 0) output_base[(long long)(b0 + t) * out_dim + row] = total;
        }
    }
}

// OP_LINEAR_PROJ_PCG32 (full, tiled-LDS): same semantics as OP_LINEAR_PROJ_PCG32 but
// loads input in tiles into LDS once per block, then processes all assigned rows per tile.
// One call handles all virtual blocks; must NOT be called from the virtual block loop.
__device__ __attribute__((noinline)) void op_linear_proj_pcg32_full(const u64* inst, u32 grid_x) {
    const LinearProjInst* i = (const LinearProjInst*)inst;
    float* output_base = i->output;
    const unsigned char* weight = i->weight;
    const float* input_base = i->input;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
    if (batch <= 1) batch = 1;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    extern __shared__ float shared[];

    if (batch == 1) {
        float* lds_tile    = shared + 8;
        float* row_partial = shared + 8 + LPROJ_TILE_FLOATS;

        int num_rows = 0;
        for (u32 r = blockIdx.x; r < (u32)out_dim; r += gridDim.x) num_rows++;
        for (int ri = tid; ri < num_rows; ri += stride) row_partial[ri] = 0.0f;
        __syncthreads();

        const int groups_per_tile = LPROJ_TILE_FLOATS / group_size;  // 240
        const int n_tiles = (num_groups + groups_per_tile - 1) / groups_per_tile;

        for (int t = 0; t < n_tiles; t++) {
            int g0 = t * groups_per_tile;
            int g1 = min(g0 + groups_per_tile, num_groups);
            int tile_start = g0 * group_size;
            int tile_len = min((g1 - g0) * group_size, in_dim - tile_start);

            // Load input tile cooperatively
            for (int i = tid; i < tile_len; i += stride) lds_tile[i] = input_base[tile_start + i];
            __syncthreads();

            int ri = 0;
            for (u32 row = blockIdx.x; row < (u32)out_dim; row += gridDim.x, ri++) {
                const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
                float acc = 0.0f;
                for (int g = g0 + tid; g < g1; g += stride) {
                    const unsigned char* gp = row_data + g * group_bytes;
                    float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                    float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                    int local_base = (g - g0) * group_size;
                    int count = min(group_size, in_dim - g * group_size);
                    for (int i = 0; i < count; i += 2) {
                        unsigned char byte = gp[i / 2];
                        float v0 = (float)(byte & 0xF) * sc + mn;
                        float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                        acc += v0 * lds_tile[local_base + i] + v1 * lds_tile[local_base + i + 1];
                    }
                }
                for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
                if ((tid & 31) == 0) shared[tid >> 5] = acc;
                __syncthreads();
                // Reduce 8 warp partials with thread 0 only. The previous code did
                // __shfl_down inside an `if (tid < 8)` block — on RDNA3 wave32, shfl
                // from an inactive lane is undefined and observably non-deterministic
                // (AMD ISA: lane reads source register; if source lane is masked off
                // its register state is implementation-defined). bz0 Source 3 root cause.
                if (tid == 0) {
                    float total = shared[0] + shared[1] + shared[2] + shared[3]
                                + shared[4] + shared[5] + shared[6] + shared[7];
                    row_partial[ri] += total;
                }
                __syncthreads();
            }
            __syncthreads();  // Before overwriting lds_tile with next tile
        }

        for (u32 row = blockIdx.x, ri = 0; row < (u32)out_dim; row += gridDim.x, ri++) {
            if (tid == 0) {
                output_base[row] = row_partial[ri];
            }
        }
        return;
    }

    // batch > 1 (prefill): existing per-row, per-batch approach
    for (u32 row = blockIdx.x; row < (u32)out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
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
            float total = block_reduce_sum(acc, shared);
            if (tid == 0) output_base[(long long)b * out_dim + row] = total;
        }
    }
}

// OP_LINEAR_PROJ_PCG32: y = W_pcg32 @ x (one virtual block per output row, decode only)
// args: [1]=output, [2]=weight (packed pcg32), [3]=input, [4]=out_dim, [5]=in_dim
// Weight layout per row: groups of 32 elements, each 20 bytes.
__device__ __attribute__((noinline)) void op_linear_proj_pcg32(const u64* inst, u32 vblock) {
    const LinearProjInst* i = (const LinearProjInst*)inst;
    float* output_base = i->output;
    const unsigned char* weight = i->weight;
    const float* input_base = i->input;
    int out_dim = (int)i->out_dim;
    int in_dim = (int)i->in_dim;
    int batch = (int)i->batch;
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
        // bz0 fix: see comment in op_linear_proj_rnf4_full above. shfl from inactive
        // lanes is undefined on RDNA3 wave32; use sequential thread-0 sum.
        if (tid == 0) {
            float total = shared[0] + shared[1] + shared[2] + shared[3]
                        + shared[4] + shared[5] + shared[6] + shared[7];
            output_base[row] = total;
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
        float total = block_reduce_sum(acc, shared);
        if (tid == 0) output_base[(long long)b * out_dim + row] = total;
    }
}

// OP_RMSNORM_WX: y = x * w / sqrt(mean(x^2) + eps) — standard w*x variant (Llama, OLMoE)
// Same args as OP_RMSNORM: [1]=output, [2]=input, [3]=weight, [4]=hidden_size, [5]=eps_bits
__device__ __attribute__((noinline)) void op_rmsnorm_wx(const u64* inst, u32 vblock) {
    const RmsNormInst* i = (const RmsNormInst*)inst;
    float* output_base = i->output;
    const float* input_base = i->input;
    const uint16_t* weight = i->weight;
    int hidden_size = (int)i->dim;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);

    // DEBUG: print input[0] to track hidden state evolution
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
    float rms = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        output[i] = input[i] * rms * bf16_to_f32(weight[i]);
    }
}

// OP_CONV1D: causal conv1d update with SiLU activation (one virtual block per channel)
// args: [1]=state, [2]=input, [3]=weight, [4]=output, [5]=conv_dim, [6]=kernel_size
__device__ __attribute__((noinline)) void op_conv1d(const u64* inst, u32 vblock) {
    const Conv1dInst* i = (const Conv1dInst*)inst;
    float* state = i->state;
    const float* input = i->input;
    const uint16_t* weight = i->weight;
    float* output = i->output;
    int conv_dim = (int)i->dim;
    int kernel_size = (int)i->kernel_size;

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

// OP_CONV1D_3X: fused Q+K+V causal conv1d; vb routing: <blocks_qk→Q, <2*blocks_qk→K, else→V
__device__ __attribute__((noinline)) void op_conv1d_3x(const u64* inst, u32 vblock) {
    const Conv1d3xInst* i = (const Conv1d3xInst*)inst;
    u32 blocks_qk = (u32)(i->blocks_qk_v & 0xFFFFFFFFu);
    u32 blocks_v  = (u32)(i->blocks_qk_v >> 32);

    float* state;
    const float* input;
    const uint16_t* weight;
    float* output;
    int conv_dim;

    if (vblock < blocks_qk) {
        state = i->q_state; input = i->q_input; weight = i->q_weight; output = i->q_output;
        conv_dim = (int)i->qk_dim;
    } else if (vblock < 2u * blocks_qk) {
        vblock -= blocks_qk;
        state = i->k_state; input = i->k_input; weight = i->k_weight; output = i->k_output;
        conv_dim = (int)i->qk_dim;
    } else {
        vblock -= 2u * blocks_qk;
        state = i->v_state; input = i->v_input; weight = i->v_weight; output = i->v_output;
        conv_dim = (int)i->v_dim;
    }

    int kernel_size = (int)i->kernel_size;
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
    output[ch] = acc / (1.0f + expf(-acc)); // SiLU

    for (int k = 0; k < hist - 1; k++) {
        s[k] = s[k + 1];
    }
    s[hist - 1] = input[ch];
}

// OP_GDN_GATE: gate = exp(-exp(A_log) * softplus(a + dt_bias))
// args: [1]=gate_out, [2]=a_input, [3]=A_log, [4]=dt_bias, [5]=num_heads
__device__ __attribute__((noinline)) void op_gdn_gate(const u64* inst, u32 vblock) {
    const GdnGateInst* i = (const GdnGateInst*)inst;
    float* gate_out = i->output;
    const float* a_input = i->a_proj;
    const float* A_log = i->a_log;
    const uint16_t* dt_bias = i->dt_bias;
    int num_heads = (int)i->num_heads;

    int h = vblock * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;

    float a = a_input[h];
    float A = expf(A_log[h]);
    float sp = log1pf(expf(a + bf16_to_f32(dt_bias[h])));
    gate_out[h] = expf(-A * sp);
}

// OP_GDN_RECUR: gated delta rule recurrence (one block per head)
// args: [1]=q, [2]=k, [3]=v, [4]=gate, [5]=b, [6]=state, [7]=output, [8]=key_dim, [9]=value_dim
__device__ __attribute__((noinline)) void op_gdn_recurrent(const u64* inst, u32 vblock) {
    // bz0 fix: invalidate per-WGP L1 on op entry. Multi-block-per-head GDN reads/writes
    // a HBM state buffer; consumer WGPs may have stale L1 lines.
    if (threadIdx.x == 0) {
        asm volatile("buffer_gl1_inv" ::: "memory");
    }
    __syncthreads();

    const GdnRecurInst* i = (const GdnRecurInst*)inst;
    const float* q = i->q;
    const float* k = i->k;
    const float* v = i->v;
    const float* gate = i->gate;
    const float* b = i->b_proj;
    float* state = i->state;
    float* output = i->output;
    int key_dim = (int)i->kd;
    int value_dim = (int)i->vd;
    int gqa_group = (int)i->gqa_group;
    int num_heads = (int)i->num_heads;
    if (gqa_group <= 0) gqa_group = 1;
    if (num_heads <= 0) num_heads = 1;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Distribute multiple blocks per head: blocks_per_head blocks handle one head's j-partition
    const int blocks_per_head = max((int)gridDim.x / num_heads, 1);
    const int h = (int)vblock % num_heads;
    const int block_in_head = (int)vblock / num_heads;
    const int j_start = (block_in_head * value_dim) / blocks_per_head;
    const int j_end = ((block_in_head + 1) * value_dim) / blocks_per_head;

    const float decay = gate[h];
    const float beta = 1.0f / (1.0f + expf(-b[h]));

    const int kh_idx = h / gqa_group;
    const float* qh = q + kh_idx * key_dim;
    const float* kh = k + kh_idx * key_dim;
    const float* vh = v + h * value_dim;
    float* sh = state + h * key_dim * value_dim;
    float* oh = output + h * value_dim;

    // Shared memory layout:
    //   [0..block_size)              sq: reduction buffer
    //   [block_size..2*block_size)   sk: reduction buffer
    //   [2*block_size..2*block_size+key_dim)   k_sh: normalized k vector
    //   [2*block_size+key_dim..)     kv_mem_sh: delta values for j in [j_start, j_end)
    extern __shared__ float shared[];
    float* sq = shared;
    float* sk = shared + block_size;
    float* k_sh = shared + 2 * block_size;
    float* kv_mem_sh = shared + 2 * block_size + key_dim;

    // Q/K norm reduction over full key_dim
    float sq_acc = 0.0f, sk_acc = 0.0f;
    for (int ii = tid; ii < key_dim; ii += block_size) {
        sq_acc += qh[ii] * qh[ii];
        sk_acc += kh[ii] * kh[ii];
    }
    float q_inv_norm = rsqrtf(block_reduce_sum(sq_acc, sq) + 1e-6f);
    float k_inv_norm = rsqrtf(block_reduce_sum(sk_acc, sk) + 1e-6f);
    float q_scale = q_inv_norm * rsqrtf((float)key_dim);

    for (int ii = tid; ii < key_dim; ii += block_size) k_sh[ii] = kh[ii] * k_inv_norm;
    __syncthreads();

    // Decay state for this block's j partition
    const int idx_start = j_start * key_dim;
    const int idx_end = j_end * key_dim;
    for (int idx = idx_start + tid; idx < idx_end; idx += block_size) sh[idx] *= decay;
    __syncthreads();

    // Step 1: kv_mem[j] = dot(S[j,:], k_norm) for j in [j_start, j_end)
    for (int j = j_start + tid; j < j_end; j += block_size) {
        float kv_mem = 0.0f;
        for (int ii = 0; ii < key_dim; ii++) kv_mem += sh[j * key_dim + ii] * k_sh[ii];
        kv_mem_sh[j - j_start] = (vh[j] - kv_mem) * beta;
    }
    __syncthreads();

    // Step 2: S[j,i] += k_norm[i] * delta[j] for j in [j_start, j_end)
    for (int idx = idx_start + tid; idx < idx_end; idx += block_size) {
        int j = idx / key_dim - j_start;
        int ii = idx % key_dim;
        sh[idx] += k_sh[ii] * kv_mem_sh[j];
    }
    __syncthreads();

    // Q readout: output[j] = S[j,:] . q_scaled for j in [j_start, j_end)
    for (int j = j_start + tid; j < j_end; j += block_size) {
        float sum = 0.0f;
        for (int ii = 0; ii < key_dim; ii++) sum += sh[j * key_dim + ii] * (qh[ii] * q_scale);
        oh[j] = sum;
    }
}

// OP_RMSNORM_GATE: RMSNorm(x) * SiLU(z) with direct weight multiply (no 1+w)
// args: [1]=output, [2]=x, [3]=z, [4]=weight, [5]=num_heads, [6]=value_dim, [7]=eps
__device__ __attribute__((noinline)) void op_rmsnorm_gated(const u64* inst, u32 vblock) {
    const RmsNormGateInst* i = (const RmsNormGateInst*)inst;
    float* output = i->output;
    const float* x = i->x;
    const float* z = i->z;
    const float* weight = i->weight;
    int value_dim = (int)i->vd;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);

    int h = vblock;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xh = x + h * value_dim;
    const float* zh = z + h * value_dim;
    float* oh = output + h * value_dim;

    extern __shared__ float shared[];
    float sum_sq = 0.0f;
    for (int i = tid; i < value_dim; i += stride) { float v = xh[i]; sum_sq += v * v; }
    float rms = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)value_dim + eps);

    for (int i = tid; i < value_dim; i += stride) {
        float normed = xh[i] * rms * weight[i];
        float zi = zh[i];
        oh[i] = normed * zi / (1.0f + expf(-zi));
    }
}

// OP_RESIDUAL_ADD: output = a + b
// args: [1]=output, [2]=a, [3]=b, [4]=size
__device__ __attribute__((noinline)) void op_residual_add(const u64* inst, u32 vblock) {
    const ResidualAddInst* i = (const ResidualAddInst*)inst;
    float* output = i->output;
    const float* a = i->src;
    const float* b = i->residual;
    int size = (int)i->n;

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = a[idx] + b[idx];
}

// OP_SIGMOID_WEIGHTED_ADD: output[i] += sigmoid(scalar[0]) * input[i]
// args: [1]=output (float*, [size]), [2]=scalar (float*, [1] — dot product to sigmoid),
//       [3]=input (float*, [size]), [4]=size
// Used for shared expert gating: output = ffn_down, scalar = gate@normed, input = shared_expert_out
__device__ __attribute__((noinline)) void op_sigmoid_weighted_add(const u64* inst, u32 vblock) {
    const SigmoidWeightedAddInst* i = (const SigmoidWeightedAddInst*)inst;
    float* output = i->output;
    const float* scalar = i->scalar;
    const float* input = i->input;
    int size = (int)i->n;

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) {
        float w = 1.0f / (1.0f + expf(-scalar[0]));
        output[idx] += w * input[idx];
    }
}

// OP_SCALE_ADD: output[i] += scale * src[i]
// args: [1]=output (float*), [2]=src (float*), [3]=scale (f32 bits packed as u32), [4]=size
__device__ __attribute__((noinline)) void op_scale_add(const u64* inst, u32 vblock) {
    const ScaleAddInst* i = (const ScaleAddInst*)inst;
    float* output = i->output;
    const float* src = i->src;
    u32 scale_bits = (u32)i->scale_bits;
    float scale; __builtin_memcpy(&scale, &scale_bits, sizeof(float));
    int size = (int)i->size;
    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] += scale * src[idx];
}

// OP_MAMBA2_CONV1D: causal conv1d with f32 bias + SiLU activation (Mamba2)
// args: [1]=state(w), [2]=input, [3]=weight(u16/bf16), [4]=bias(f32), [5]=output(w), [6]=conv_dim, [7]=kernel_size
// One virtual block per 256 channels. grid_x = ceil(conv_dim / 256).
__device__ __attribute__((noinline)) void op_mamba2_conv1d(const u64* inst, u32 vblock) {
    const Mamba2Conv1dInst* i = (const Mamba2Conv1dInst*)inst;
    float* state = i->state;
    const float* input = i->input;
    const uint16_t* weight = i->weight;
    const float* bias = i->bias;
    float* output = i->output;
    int conv_dim = (int)i->conv_dim;
    int kernel_size = (int)i->kernel_size;

    int ch = vblock * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    int hist = kernel_size - 1;
    float* s = state + ch * hist;
    const unsigned short* w = weight + ch * kernel_size;

    float acc = 0.0f;
    for (int k = 0; k < hist; k++) acc += s[k] * bf16_to_f32(w[k]);
    acc += input[ch] * bf16_to_f32(w[hist]);
    acc += bias[ch];
    output[ch] = acc / (1.0f + expf(-acc));

    for (int i = 0; i < hist - 1; i++) s[i] = s[i + 1];
    s[hist - 1] = input[ch];
}

// OP_SSM_UPDATE: Mamba2 selective state update (one block per head)
// args: [1]=state(w), [2]=x, [3]=dt, [4]=dt_bias, [5]=A_log, [6]=B, [7]=C, [8]=D, [9]=output(w),
//       [10]=num_heads, [11]=head_dim, [12]=state_size, [13]=n_groups
// grid_x = num_heads. One block per head, 256 threads.
__device__ __attribute__((noinline)) void op_ssm_update(const u64* inst, u32 vblock) {
    const SsmUpdateInst* i = (const SsmUpdateInst*)inst;
    float* ssm_state = i->state;
    const float* x = i->x;
    const float* dt = i->dt;
    const float* dt_bias = i->dt_bias;
    const float* A_log = i->a_log;
    const float* B = i->b;
    const float* C = i->c;
    const float* D = i->d_weight;
    float* output = i->output;
    int num_heads = (int)i->nh;
    int head_dim = (int)i->hd;
    int state_size = (int)i->sd;
    int n_groups = (int)i->ng;

    const int h = vblock;
    if (h >= num_heads) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int heads_per_group = num_heads / n_groups;
    const int g = h / heads_per_group;

    float dt_val = dt[h] + dt_bias[h];
    dt_val = dt_val > 20.0f ? dt_val : logf(1.0f + expf(dt_val)); // softplus
    float dA = expf(dt_val * (-expf(A_log[h])));

    float* state_h = ssm_state + (long long)h * head_dim * state_size;
    const float* x_h = x + h * head_dim;
    const float* B_g = B + g * state_size;
    const float* C_g = C + g * state_size;
    float D_h = D[h];

    int total = head_dim * state_size;
    for (int idx = tid; idx < total; idx += stride) {
        int d = idx / state_size;
        int s = idx % state_size;
        state_h[d * state_size + s] = dA * state_h[d * state_size + s] + dt_val * x_h[d] * B_g[s];
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += stride) {
        float sum = 0.0f;
        for (int s = 0; s < state_size; s++) sum += state_h[d * state_size + s] * C_g[s];
        output[h * head_dim + d] = sum + D_h * x_h[d];
    }
}

// OP_MAMBA2_NORM_GATED: rms_norm(x * silu(z)) * weight (norm_before_gate=False, Mamba2 style)
// args: [1]=output(w), [2]=x(ssm_out), [3]=z(gate), [4]=weight, [5]=num_heads, [6]=value_dim, [7]=eps
// grid_x = num_heads. One block per head.
__device__ __attribute__((noinline)) void op_mamba2_norm_gated(const u64* inst, u32 vblock) {
    const Mamba2NormGatedInst* i = (const Mamba2NormGatedInst*)inst;
    float* output = i->output;
    const float* x = i->x;
    const float* z = i->z;
    const float* weight = i->weight;
    int value_dim = (int)i->value_dim;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);

    int h = vblock;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xh = x + h * value_dim;
    const float* zh = z + h * value_dim;
    float* oh = output + h * value_dim;
    const float* wh = weight + h * value_dim;

    extern __shared__ float shared[];

    float sum_sq = 0.0f;
    for (int i = tid; i < value_dim; i += stride) {
        float zi = zh[i];
        float silu_z = zi / (1.0f + expf(-zi));
        float v = xh[i] * silu_z;
        sum_sq += v * v;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)value_dim + eps);

    for (int i = tid; i < value_dim; i += stride) {
        float zi = zh[i];
        float silu_z = zi / (1.0f + expf(-zi));
        oh[i] = xh[i] * silu_z * rms * wh[i];
    }
}

// OP_RELU_SQ: output[i] = relu(input[i])^2
// args: [1]=output (float*), [2]=input (float*), [3]=size
__device__ __attribute__((noinline)) void op_relu_sq(const u64* inst, u32 vblock) {
    const ReluSqInst* i = (const ReluSqInst*)inst;
    float* output = i->output;
    const float* input = i->input;
    int size = (int)i->size;
    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < size) { float x = input[idx]; float r = x > 0.0f ? x : 0.0f; output[idx] = r * r; }
}

// OP_FFN_GATE_UP / OP_FFN_GATE_UP_WX: fused RMSNorm + gate_proj + up_proj + SiLU*up
// args: [1]=scratch, [2]=input, [3]=rms_weight, [4]=w_gate, [5]=w_up,
//       [6]=hidden_size, [7]=intermediate_size, [8]=eps, [9]=batch_size (0/1=single)
// ONE_PLUS_W=true: (1+w)*x normalization (Qwen3.5). false: w*x (Llama/Mistral).
template<bool ONE_PLUS_W>
__device__ __attribute__((always_inline)) void op_ffn_gate_up_impl(const u64* inst, u32 vblock) {
    const FfnGateUpInst* i = (const FfnGateUpInst*)inst;
    float* scratch_base = i->output;
    const float* input_base = i->hidden;
    const uint16_t* rms_weight = i->norm_weight;
    const uint16_t* w_gate = (const uint16_t*)i->w_gate;
    const uint16_t* w_up = (const uint16_t*)i->w_up;
    int hidden_size = (int)i->hs;
    int intermediate_size = (int)i->intermediate;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);
    int batch = (int)i->batch;
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
    float rms_scale = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)hidden_size + eps);

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
            float w_factor = ONE_PLUS_W ? (1.0f + bf16_to_f32(rms_weight[base + j]))
                                        : bf16_to_f32(rms_weight[base + j]);
            float normed = input[base + j] * rms_scale * w_factor;
            acc_gate += bf16_to_f32(gs[j]) * normed;
            acc_up += bf16_to_f32(us[j]) * normed;
        }
    }
    int rem = vec_count * vec_elements;
    for (int i = rem + tid; i < hidden_size; i += stride) {
        float w_factor = ONE_PLUS_W ? (1.0f + bf16_to_f32(rms_weight[i]))
                                    : bf16_to_f32(rms_weight[i]);
        float normed = input[i] * rms_scale * w_factor;
        acc_gate += bf16_to_f32(wg[i]) * normed;
        acc_up += bf16_to_f32(wu[i]) * normed;
    }

    float gate_val = block_reduce_sum(acc_gate, shared);
    float up_val = block_reduce_sum(acc_up, shared);

    if (tid == 0) {
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        scratch[row] = silu_g * up_val;
    }
}

__device__ __attribute__((noinline)) void op_ffn_gate_up(const u64* inst, u32 vblock) {
    op_ffn_gate_up_impl<true>(inst, vblock);
}

__device__ __attribute__((noinline)) void op_ffn_gate_up_wx(const u64* inst, u32 vblock) {
    op_ffn_gate_up_impl<false>(inst, vblock);
}

// OP_FFN_DOWN_RES: down_proj(scratch) + residual
// args: [1]=output, [2]=residual, [3]=w_down, [4]=scratch, [5]=hidden_size, [6]=intermediate_size, [7]=batch_size
// When batch > 1: output/residual are [batch × hidden_size], scratch is [batch × intermediate_size].
// vblock = batch_idx * hidden_size + row.
__device__ __attribute__((noinline)) void op_ffn_down_res(const u64* inst, u32 vblock) {
    const FfnDownResInst* i = (const FfnDownResInst*)inst;
    float* output_base = i->output;
    const float* residual_base = i->residual;
    const uint16_t* w_down = (const uint16_t*)i->w_down;
    const float* scratch_base = i->ffn_act;
    int hidden_size = (int)i->hs;
    int intermediate_size = (int)i->intermediate;
    int batch = (int)i->batch;
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
    float total = block_reduce_sum(acc, shared);
    if (tid == 0) output[row] = total + residual[row];
}

// OP_FFN_GATE_UP_RNF4: fused RMSNorm + gate_proj + up_proj + SiLU*up (rnf4 weights)
// args: [1]=scratch, [2]=input, [3]=rms_weight, [4]=w_gate (rnf4), [5]=w_up (rnf4),
//       [6]=hidden_size, [7]=intermediate_size, [8]=eps, [9]=batch_size (0/1=single)
// When batch > 1: input is [batch × hidden_size], scratch is [batch × intermediate_size].
// vblock = batch_idx * intermediate_size + row.
template<bool ONE_PLUS_W>
__device__ __attribute__((always_inline)) void op_ffn_gate_up_rnf4_impl(const u64* inst, u32 vblock) {
    const FfnGateUpInst* i = (const FfnGateUpInst*)inst;
    float* scratch_base = i->output;
    const float* input_base = i->hidden;
    const uint16_t* rms_weight = i->norm_weight;
    const unsigned char* w_gate = i->w_gate;
    const unsigned char* w_up = i->w_up;
    int hidden_size = (int)i->hs;
    int intermediate_size = (int)i->intermediate;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);
    int batch = (int)i->batch;
    if (batch <= 1) batch = 1;

    int row = vblock % intermediate_size;
    int batch_idx = vblock / intermediate_size;
    if (batch_idx >= batch) return;

    const float* input = input_base + (long long)batch_idx * hidden_size;
    float* scratch = scratch_base + (long long)batch_idx * intermediate_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 128;
    const int group_bytes = 132;
    const int num_groups = (hidden_size + group_size - 1) / group_size;

    extern __shared__ float shared[];

    // RMSNorm: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) { float v = input[i]; sum_sq += v * v; }
    float rms_scale = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)hidden_size + eps);

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

            float w_factor0 = ONE_PLUS_W ? (1.0f + bf16_to_f32(rms_weight[base + i]))
                                         : bf16_to_f32(rms_weight[base + i]);
            float normed0 = input[base + i] * rms_scale * w_factor0;
            float wg0 = NF4_TABLE[bg1 & 0xF] * absmax_g1 + NF4_TABLE[bg2 & 0xF] * absmax_g2;
            float wu0 = NF4_TABLE[bu1 & 0xF] * absmax_u1 + NF4_TABLE[bu2 & 0xF] * absmax_u2;
            acc_gate += wg0 * normed0;
            acc_up   += wu0 * normed0;

            if (i + 1 < count) {
                float w_factor1 = ONE_PLUS_W ? (1.0f + bf16_to_f32(rms_weight[base + i + 1]))
                                             : bf16_to_f32(rms_weight[base + i + 1]);
                float normed1 = input[base + i + 1] * rms_scale * w_factor1;
                float wg1 = NF4_TABLE[(bg1 >> 4) & 0xF] * absmax_g1 + NF4_TABLE[(bg2 >> 4) & 0xF] * absmax_g2;
                float wu1 = NF4_TABLE[(bu1 >> 4) & 0xF] * absmax_u1 + NF4_TABLE[(bu2 >> 4) & 0xF] * absmax_u2;
                acc_gate += wg1 * normed1;
                acc_up   += wu1 * normed1;
            }
        }
    }

    float gate_val = block_reduce_sum(acc_gate, shared);
    float up_val = block_reduce_sum(acc_up, shared);

    if (tid == 0) {
        float silu_g = gate_val / (1.0f + expf(-gate_val));
        scratch[row] = silu_g * up_val;
    }
}

__device__ __attribute__((noinline)) void op_ffn_gate_up_rnf4(const u64* inst, u32 vblock) {
    op_ffn_gate_up_rnf4_impl<true>(inst, vblock);
}

__device__ __attribute__((noinline)) void op_ffn_gate_up_rnf4_wx(const u64* inst, u32 vblock) {
    op_ffn_gate_up_rnf4_impl<false>(inst, vblock);
}

// OP_FFN_DOWN_RES_RNF4: down_proj(scratch) + residual (rnf4 weights)
// args: [1]=output, [2]=residual, [3]=w_down (rnf4), [4]=scratch, [5]=hidden_size, [6]=intermediate_size, [7]=batch_size
// When batch > 1: output/residual are [batch × hidden_size], scratch is [batch × intermediate_size].
// vblock = batch_idx * hidden_size + row.
__device__ __attribute__((noinline)) void op_ffn_down_res_rnf4(const u64* inst, u32 vblock) {
    const FfnDownResInst* i = (const FfnDownResInst*)inst;
    float* output_base = i->output;
    const float* residual_base = i->residual;
    const unsigned char* w_down = i->w_down;
    const float* scratch_base = i->ffn_act;
    int hidden_size = (int)i->hs;
    int intermediate_size = (int)i->intermediate;
    int batch = (int)i->batch;
    if (batch <= 1) batch = 1;

    int row = vblock % hidden_size;
    int batch_idx = vblock / hidden_size;
    if (batch_idx >= batch) return;

    float* output = output_base + (long long)batch_idx * hidden_size;
    const float* residual = residual_base + (long long)batch_idx * hidden_size;
    const float* scratch = scratch_base + (long long)batch_idx * intermediate_size;

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
    float total = block_reduce_sum(acc, shared);
    if (tid == 0) output[row] = total + residual[row];
}

// OP_EMBEDDING: output[i] = table[token_id * hidden_size + i]
// args: [1]=output, [2]=table, [3]=token_id, [4]=hidden_size
__device__ __attribute__((noinline)) void op_embedding(const u64* inst, u32 vblock) {
    const EmbeddingInst* i = (const EmbeddingInst*)inst;
    float* output = i->output;
    const uint16_t* table = i->embed_weight;
    int token_id = (int)i->token_id;
    int hidden_size = (int)i->hs;

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        output[idx] = bf16_to_f32(table[(long long)token_id * hidden_size + idx]);
    }
}

// OP_LM_HEAD: logits[row] = dot(weight[row], input) — same as linear_proj but may be large
// Reuses op_linear_proj logic
__device__ __attribute__((noinline)) void op_lm_head(const u64* inst, u32 vblock) {
    op_linear_proj(inst, vblock);
}

// OP_QK_NORM: per-head RMSNorm on Q and K with (1+weight)
// args: [1]=q, [2]=k, [3]=q_weight, [4]=k_weight, [5]=num_q_heads, [6]=num_kv_heads, [7]=head_dim, [8]=eps
__device__ __attribute__((noinline)) void op_qk_norm(const u64* inst, u32 vblock) {
    const QkNormInst* i = (const QkNormInst*)inst;
    float* q = i->q;
    float* k = i->k;
    const uint16_t* q_weight = i->q_norm;
    const uint16_t* k_weight = i->k_norm;
    int num_q_heads = (int)i->nqh;
    int num_kv_heads = (int)i->nkh;
    int head_dim = (int)i->hd;
    float eps; uint32_t eps_bits32 = (uint32_t)i->eps_bits; __builtin_memcpy(&eps, &eps_bits32, 4);
    int batch = (int)i->batch;
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
    float rms = rsqrtf(block_reduce_sum(sum_sq, shared) / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += stride) {
        head_ptr[i] = head_ptr[i] * rms * (1.0f + bf16_to_f32(weight_ptr[i]));
    }
}

// OP_D2D_COPY: memcpy device-to-device
// args: [1]=dst, [2]=src, [3]=count (in floats)
__device__ __attribute__((noinline)) void op_d2d_copy(const u64* inst, u32 vblock) {
    const D2dCopyInst* i = (const D2dCopyInst*)inst;
    float* dst = i->dst;
    const float* src = i->src;
    int count = (int)i->n_elems;

    int idx = vblock * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] = src[idx];
}

// OP_MROPE: multimodal rotary position embedding (one block per head)
// args: [1]=q, [2]=k, [3]=inv_freq, [4]=position_ids, [5]=num_q_heads,
//       [6]=num_kv_heads, [7]=head_dim, [8]=rope_dim,
//       [9]=section0_pairs, [10]=section1_pairs, [11]=section2_pairs
__device__ __attribute__((noinline)) void op_mrope(const u64* inst, u32 vblock) {
    const MropeInst* i = (const MropeInst*)inst;
    float* q = i->q;
    float* k = i->k;
    const float* inv_freq = i->inv_freq;
    const int* position_ids = i->pos_ids;
    int num_q_heads = (int)i->nqh;
    int num_kv_heads = (int)i->nkh;
    int head_dim = (int)i->hd;
    int rope_dim = (int)i->rd;
    int section0_pairs = (int)i->s0;
    int section1_pairs = (int)i->s1;
    int batch = (int)i->batch;
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
__device__ __attribute__((noinline)) void op_gqa_attn(const u64* inst, u32 vblock) {
    const GqaAttnInst* i = (const GqaAttnInst*)inst;
    float* output = i->output;
    const float* q = i->q;
    const float* k_cache = i->k_cache;
    const float* v_cache = i->v_cache;
    int num_q_heads = (int)i->nqh;   // global nqh (for gqa_group)
    int num_kv_heads = (int)i->nkh;  // global nkh
    int head_dim = (int)i->hd;
    int seq_len = (int)i->seq_len;
    int max_seq_len = (int)i->max_seq_len;
    int q_head_start = (int)i->q_head_start; // gpu_i * local_nqh

    int q_head = vblock;                   // local Q head index (q buffer offset)
    int gqa_group = num_q_heads / num_kv_heads;
    int kv_head = (q_head + q_head_start) / gqa_group;
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
        float score = block_reduce_sum(dot, partial) * scale;
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
__device__ __attribute__((noinline)) void op_output_gate(const u64* inst, u32 vblock) {
    const OutputGateInst* i = (const OutputGateInst*)inst;
    float* output = i->output;
    const float* attn_output = i->attn_out;
    const float* gate = i->gate;
    int size = (int)i->size;

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
__device__ __attribute__((noinline)) void op_attn_paged(const u64* inst, u32 vblock) {
    // bz0 fix: invalidate per-WGP L1 on op entry. KV chunk pool is written by prior op
    // (OP_D2D_COPY) on potentially different WGPs from the consumer.
    if (threadIdx.x == 0) {
        asm volatile("buffer_gl1_inv" ::: "memory");
    }
    __syncthreads();

    const AttnPagedInst* i = (const AttnPagedInst*)inst;
    float* output          = i->output;
    const float* q         = i->q;
    const u64* page_table  = (const u64*)(i->page_table);
    const int* position_table = (const int*)(i->pos_table);
    const float* inv_freq  = i->inv_freq;
    int num_q_heads        = (int)i->nqh;
    int num_kv_heads       = (int)i->nkh;
    int head_dim           = (int)i->hd;
    int seq_len            = (int)i->seq_len;
    int chunk_tokens       = (int)i->chunk_tokens;
    int rope_dim           = (int)i->rd;
    long long layer_k_offset = (long long)i->layer_k_offset;
    long long layer_v_offset = (long long)i->layer_v_offset;
    const float* partial_state = (const float*)(i->partial_state); // 0 = none
    // Source 1: read eps from instruction (was hardcoded 1e-6f, broken for non-Qwen3.5)
    float eps;
    {
        uint32_t eps_bits32 = (uint32_t)i->eps_bits;
        __builtin_memcpy(&eps, &eps_bits32, 4);
    }
    // Source 2: read mRoPE section sizes (low32 = section0_pairs, high32 = section1_pairs)
    int mrope_section0_pairs = (int)(i->mrope_sections & 0xFFFFFFFFu);
    int mrope_section1_pairs = (int)(i->mrope_sections >> 32);

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

    for (int c = 0; c < num_chunks; c++) {
        const char* chunk_base = (const char*)(page_table[c]);
        const float* k_base = (const float*)(chunk_base + layer_k_offset);
        const float* v_base = (const float*)(chunk_base + layer_v_offset);
        int tokens_in_chunk = chunk_tokens;
        if (c == num_chunks - 1) {
            tokens_in_chunk = seq_len - c * chunk_tokens;
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
            // k_norm_weight is passed as k_norm field of AttnPagedInst
            // NOTE: partial aliases sh_k (both = shmem). Save K to register before reduction.
            // Source 1 fix: use eps from instruction (was hardcoded 1e-6f).
            {
                const uint16_t* kn_w = i->k_norm;
                if (kn_w != nullptr) {
                    float my_k = (tid < head_dim) ? sh_k[tid] : 0.0f;
                    // bz0 fix: tree reduction (no __shfl_down). A/B test confirmed
                    // K-rms with block_reduce_sum (shfl) is the source of bz0
                    // single-GPU non-determinism on gfx1100 wave32.
                    float k_rms = rsqrtf(tree_reduce_sum_256(my_k * my_k, partial) / (float)head_dim + eps);
                    if (tid < head_dim) {
                        float w = 1.0f + bf16_to_f32(kn_w[tid]);
                        sh_k[tid] = my_k * k_rms * w;  // use saved register value
                    }
                }
            }

            // Apply lazy RoPE rotation to sh_k (in-place, rope_dim elements only).
            // Source 2 fix: mRoPE sections — each pair belongs to a temporal/height/width
            // section, and uses the corresponding position from a 3-int per-token table.
            // For text-only models with mrope_sections=[0,0,0], all pairs fall into
            // section 2; we read position_table[*3 + 2]. Writers populate all 3 ints
            // with the same scalar position, so this is equivalent to standard RoPE.
            // Threads 0..rope_pairs-1 handle pairs (tid*2, tid*2+1).
            if (tid < rope_pairs) {
                int section;
                if (mrope_section0_pairs == 0 && mrope_section1_pairs == 0) {
                    // Standard (non-mRoPE) models: use section 0 (cleaner default).
                    section = 0;
                } else {
                    section = (tid < mrope_section0_pairs)
                        ? 0
                        : (tid < mrope_section0_pairs + mrope_section1_pairs ? 1 : 2);
                }
                // position_table layout: 3 ints per token (temporal, height, width).
                int pos = position_table[(c * chunk_tokens + t) * 3 + section];
                float theta = (float)pos * inv_freq[tid];
                float cos_state = cosf(theta);
                float sin_state = sinf(theta);
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
            // Q·K dot reduction. A/B test confirmed shfl is fine here (only
            // K-rms's reduction shows bz0); use the fast block_reduce_sum.
            float score = block_reduce_sum(dot, partial) * scale;

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
__device__ __attribute__((noinline)) void op_attn_paged_quant(const u64* inst, u32 vblock) {
    const AttnPagedQInst* i = (const AttnPagedQInst*)inst;
    float* scratch         = (float*)(i->scratch);
    const float* q         = i->q;
    const u64* page_table  = (const u64*)(i->quant_page_table);
    const int* position_table = (const int*)(i->pos_table);
    const float* inv_freq  = i->inv_freq;
    int num_q_heads        = (int)i->nqh;
    int num_kv_heads       = (int)i->nkh;
    int head_dim           = (int)i->hd;
    int seq_len            = (int)i->quant_seq_len;
    int chunk_tokens       = (int)i->chunk_tokens;
    int rope_dim           = (int)i->rd;
    // Byte offsets within each chunk for this layer's K quantized data
    long long k_q1data_off  = (long long)i->q1d;
    long long k_q1scale_off = (long long)i->q1s;
    long long k_rdata_off   = (long long)i->rd_off;
    long long k_rscale_off  = (long long)i->rs;

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

    float m = -1e30f, d_softmax = 0.0f;
    float v_acc = 0.0f;

    int num_chunks = (seq_len + chunk_tokens - 1) / chunk_tokens;

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
            // k_norm_weight passed as k_norm field of AttnPagedQInst
            // NOTE: partial aliases sh_k (both = shmem). Save K to register before reduction.
            {
                const uint16_t* kn_w = i->k_norm;
                if (kn_w != nullptr) {
                    float my_k = (tid < head_dim) ? sh_k[tid] : 0.0f;
                    float k_rms = rsqrtf(block_reduce_sum(my_k * my_k, partial) / (float)head_dim + 1e-6f);
                    if (tid < head_dim) {
                        float w = 1.0f + bf16_to_f32(kn_w[tid]);
                        sh_k[tid] = my_k * k_rms * w;
                    }
                }
            }

            // Apply lazy RoPE — position_table now stores 3 ints per token (mRoPE).
            // For text models all 3 sections share the same value; section-0 is canonical.
            if (tid < rope_pairs) {
                int pos = position_table[(c * chunk_tokens + t) * 3];
                float theta = (float)pos * inv_freq[tid];
                float cos_state = cosf(theta);
                float sin_state = sinf(theta);
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
            float score = block_reduce_sum(dot, partial) * attn_scale;

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
__device__ __attribute__((noinline)) void op_attn_prefill(const u64* inst, u32 vblock) {
    const AttnPrefillInst* i = (const AttnPrefillInst*)inst;
    float* output = i->output;
    const float* q = i->q;
    const float* k_cache = i->k_cache;
    const float* v_cache = i->v_cache;
    int num_q_heads = (int)i->nqh;
    int num_kv_heads = (int)i->nkh;
    int head_dim = (int)i->hd;
    int start_pos = (int)i->start_pos;
    int num_tokens = (int)i->n;
    int max_seq_len = (int)i->max_seq_len;

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
        float score = block_reduce_sum(dot, partial) * scale;
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
__device__ __attribute__((noinline)) void op_deinterleave(const u64* inst, u32 vblock) {
    const DeinterleaveInst* i = (const DeinterleaveInst*)inst;
    float* dst_even = i->dst_q;
    float* dst_odd = i->dst_gate;
    const float* src = i->src;
    int num_heads = (int)i->num_heads;
    int head_dim = (int)i->head_dim;
    int batch = (int)i->batch;
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
__device__ __attribute__((noinline)) void op_kv_quantize(const u64* inst, u32 vblock) {
    const KvQuantizeInst* i = (const KvQuantizeInst*)inst;
    const float* src = i->src;
    unsigned char* q1_data = i->q1_data;
    float* q1_scale = i->q1_scale;
    unsigned char* r_data = i->r_data;
    float* r_scale = i->r_scale;
    int nkh = i->num_kv_heads;
    int hd = i->head_dim;
    int ct = i->chunk_tokens;

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
