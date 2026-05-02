
// OP_MOE_GATE: Router top-k expert selection from pre-computed scores.
// Assumes scores already computed via OP_LINEAR_PROJ (hidden @ gate^T → [num_experts]).
// This kernel does: top-k selection + softmax/sigmoid/norm_topk + write expert_ids and weights.
// args: [1]=scores (float*, [num_experts] — output of linear_proj)
//       [2]=expert_ids (int*, [k] — output)
//       [3]=expert_weights (float*, [k] — output)
//       [4]=num_experts, [5]=k (top-k count)
//       [6]=gate_mode (0=softmax, 1=norm_topk, 2=sigmoid)
//       [7]=routed_scaling_factor (float)
//       [8]=correction_bias (float* or NULL — added to scores before selection)
// grid_x = 1 (single block processes one token's routing)
__device__ __attribute__((noinline)) void op_moe_gate(const u64* inst, u32 vblock) {
    if (vblock > 0) return;
    const MoeGateInst* i = (const MoeGateInst*)inst;
    const float* scores = i->scores;
    int* expert_ids = i->expert_ids;
    float* expert_weights = i->expert_weights;
    int num_experts = (int)i->ne;
    int k = (int)i->k;
    int gate_mode = (int)i->gate_mode;
    float scaling; uint32_t rsf_bits32 = (uint32_t)i->rsf_bits; __builtin_memcpy(&scaling, &rsf_bits32, 4);
    const float* correction_bias = (const float*)i->bias;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ float shmem[];
    float* sh_sel = shmem;           // [512] — selection scores (biased)
    float* sh_raw = shmem + 512;     // [512] — raw scores (for weight computation)

    // Load scores into shared memory
    for (int i = tid; i < num_experts; i += stride) {
        float s = scores[i];
        if (gate_mode == 2) {
            float sig = 1.0f / (1.0f + expf(-s));
            sh_raw[i] = sig;
            sh_sel[i] = correction_bias ? sig + correction_bias[i] : sig;
        } else {
            sh_raw[i] = s;
            sh_sel[i] = correction_bias ? s + correction_bias[i] : s;
        }
    }
    __syncthreads();

    if (tid != 0) return;

    // Top-k selection from sh_sel
    int ids[32];
    for (int j = 0; j < k; j++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int i = 0; i < num_experts; i++) {
            if (sh_sel[i] > best) {
                best = sh_sel[i];
                best_idx = i;
            }
        }
        ids[j] = best_idx;
        expert_ids[j] = best_idx;
        sh_sel[best_idx] = -1e30f;
    }

    // Compute weights from raw scores of selected experts
    if (gate_mode == 2) {
        // Sigmoid: renormalize + scale
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += sh_raw[ids[j]];
        for (int j = 0; j < k; j++) {
            expert_weights[j] = (sh_raw[ids[j]] / sum) * scaling;
        }
    } else {
        // Softmax over selected experts' raw scores
        float selected[32];
        float max_s = -1e30f;
        for (int j = 0; j < k; j++) {
            selected[j] = sh_raw[ids[j]];
            if (selected[j] > max_s) max_s = selected[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            selected[j] = expf(selected[j] - max_s);
            sum += selected[j];
        }
        float scale = (gate_mode == 1) ? scaling : 1.0f;
        for (int j = 0; j < k; j++) {
            expert_weights[j] = (selected[j] / sum) * scale;
        }
    }
}

// OP_MOE_FFN: Fused MoE FFN for all k experts in a single megakernel op.
// Loops over k selected experts using expert_ids from OP_MOE_GATE output.
// Each expert: gate_proj + up_proj + silu_mul (or relu²) + down_proj + weighted accumulate.
// Internal grid.sync() between matmul phases (all blocks cooperate on each matmul).
//
// args: [1]=expert_ids (int*, [k]), [2]=expert_weights (float*, [k]),
//       [3]=input (float*, [hs] — normed), [4]=output (float*, [hs] — accumulation, zeroed by this op),
//       [5]=gate_up_base (u8*), [6]=gate_up_expert_stride (bytes),
//       [7]=down_base (u8*), [8]=down_expert_stride (bytes),
//       [9]=k (num active), [10]=(hs | eis<<16), [11]=flags (bit0: has_gate_proj, bit1: relu_sq),
//       [12]=scratch_gate (float*, [eis]), [13]=scratch_up (float*, [eis]),
//       [14]=scratch_act (float*, [eis]), [15]=scratch_out (float*, [hs]),
//       [16]=gate_up_row_stride (bytes per row — for computing gate vs up offset within one expert)
//
// Weight format: PcG32Q4 (group_size=32, 20 bytes/group). Other formats use fallback path.
// grid_x should be max(eis, hs) to give enough virtual blocks for all matmuls.
template<typename GridGroup>
__device__ __attribute__((noinline)) void op_moe_ffn(const u64* inst, GridGroup& grid) {
    const MoeFfnInst* i = (const MoeFfnInst*)inst;
    const int* expert_ids = i->expert_ids;
    const float* expert_weights = i->expert_weights;
    const float* input = i->normed;
    float* output = i->ffn_down;
    const unsigned char* gate_up_base = i->gate_up_data;
    long long gate_up_expert_stride = (long long)i->gate_up_expert_stride;
    const unsigned char* down_base = i->down_data;
    long long down_expert_stride = (long long)i->down_expert_stride;
    int k = (int)i->k;
    int packed_dims = (int)i->hs_eis;
    int hs = packed_dims & 0xFFFF;
    int eis = (packed_dims >> 16) & 0xFFFF;
    int flags = (int)i->flags;
    bool has_gate_proj = flags & 1;
    bool relu_sq = flags & 2;
    float* scratch_gate = (float*)i->expert_gate;
    float* scratch_up = (float*)i->expert_up;
    float* scratch_act = (float*)i->expert_act;
    float* scratch_out = (float*)i->expert_out;
    long long gate_up_row_stride = (long long)i->gate_up_row_stride;

    int tid = threadIdx.x;

    // Zero output buffer (all blocks cooperate)
    for (int i = blockIdx.x * blockDim.x + tid; i < hs; i += gridDim.x * blockDim.x) {
        output[i] = 0.0f;
    }
    grid.sync();

    // PcG32Q4 matmul helper (inline): row_data points to group data, dot with input
    // Returns dot product for one output row, reduced to thread 0.
    // Caller provides row_data, in_dim; result is in thread 0's return value.

    for (int ei = 0; ei < k; ei++) {
        int eid = expert_ids[ei];
        float w = expert_weights[ei];

        const unsigned char* expert_gate_up = gate_up_base + eid * gate_up_expert_stride;
        const unsigned char* expert_down = down_base + eid * down_expert_stride;

        // --- Gate proj (or up-only for relu²): expert_gate_up @ input → scratch_gate/scratch_up ---
        if (has_gate_proj) {
            // Gate proj: first eis rows of this expert's gate_up weight
            // Each virtual block = one output row
            for (int row = (int)blockIdx.x; row < eis; row += (int)gridDim.x) {
                const int group_size = 32;
                const int group_bytes = 20;
                const int num_groups = (hs + group_size - 1) / group_size;
                const unsigned char* row_data = expert_gate_up + (long long)row * gate_up_row_stride;
                float acc = 0.0f;
                for (int g = tid; g < num_groups; g += (int)blockDim.x) {
                    const unsigned char* gp = row_data + g * group_bytes;
                    float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                    float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                    int base = g * group_size;
                    int count = min(group_size, hs - base);
                    for (int i = 0; i < count; i += 2) {
                        unsigned char byte = gp[i / 2];
                        float v0 = (float)(byte & 0xF) * sc + mn;
                        float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                        acc += v0 * input[base + i] + v1 * input[base + i + 1];
                    }
                }
                // Wave32 reduction
                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_down(acc, offset);
                extern __shared__ float shared[];
                if ((tid & 31) == 0) shared[tid >> 5] = acc;
                __syncthreads();
                // bz0: sequential thread-0 sum of 8 warp partials. shfl from inactive
                // lanes is undefined on RDNA3 wave32.
                if (tid == 0) {
                    float total = shared[0] + shared[1] + shared[2] + shared[3]
                                + shared[4] + shared[5] + shared[6] + shared[7];
                    scratch_gate[row] = total;
                }
            }
            grid.sync();

            // Up proj: next eis rows
            for (int row = (int)blockIdx.x; row < eis; row += (int)gridDim.x) {
                const int group_size = 32;
                const int group_bytes = 20;
                const int num_groups = (hs + group_size - 1) / group_size;
                const unsigned char* row_data = expert_gate_up + (long long)(eis + row) * gate_up_row_stride;
                float acc = 0.0f;
                for (int g = tid; g < num_groups; g += (int)blockDim.x) {
                    const unsigned char* gp = row_data + g * group_bytes;
                    float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                    float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                    int base = g * group_size;
                    int count = min(group_size, hs - base);
                    for (int i = 0; i < count; i += 2) {
                        unsigned char byte = gp[i / 2];
                        float v0 = (float)(byte & 0xF) * sc + mn;
                        float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                        acc += v0 * input[base + i] + v1 * input[base + i + 1];
                    }
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_down(acc, offset);
                extern __shared__ float shared[];
                if ((tid & 31) == 0) shared[tid >> 5] = acc;
                __syncthreads();
                if (tid == 0) {
                    float total = shared[0] + shared[1] + shared[2] + shared[3]
                                + shared[4] + shared[5] + shared[6] + shared[7];
                    scratch_up[row] = total;
                }
            }
            grid.sync();

            // SiLU(gate) * up → scratch_act
            for (int i = (int)blockIdx.x * (int)blockDim.x + tid; i < eis; i += (int)gridDim.x * (int)blockDim.x) {
                float g = scratch_gate[i];
                float silu_g = g / (1.0f + expf(-g));
                scratch_act[i] = silu_g * scratch_up[i];
            }
        } else {
            // relu² path: up_proj only (eis rows), then relu²
            for (int row = (int)blockIdx.x; row < eis; row += (int)gridDim.x) {
                const int group_size = 32;
                const int group_bytes = 20;
                const int num_groups = (hs + group_size - 1) / group_size;
                const unsigned char* row_data = expert_gate_up + (long long)row * gate_up_row_stride;
                float acc = 0.0f;
                for (int g = tid; g < num_groups; g += (int)blockDim.x) {
                    const unsigned char* gp = row_data + g * group_bytes;
                    float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                    float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                    int base = g * group_size;
                    int count = min(group_size, hs - base);
                    for (int i = 0; i < count; i += 2) {
                        unsigned char byte = gp[i / 2];
                        float v0 = (float)(byte & 0xF) * sc + mn;
                        float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                        acc += v0 * input[base + i] + v1 * input[base + i + 1];
                    }
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_down(acc, offset);
                extern __shared__ float shared[];
                if ((tid & 31) == 0) shared[tid >> 5] = acc;
                __syncthreads();
                if (tid == 0) {
                    float total = shared[0] + shared[1] + shared[2] + shared[3]
                                + shared[4] + shared[5] + shared[6] + shared[7];
                    scratch_up[row] = total;
                }
            }
            grid.sync();

            // relu²: output[i] = relu(x)^2
            for (int i = (int)blockIdx.x * (int)blockDim.x + tid; i < eis; i += (int)gridDim.x * (int)blockDim.x) {
                float x = scratch_up[i];
                float r = x > 0.0f ? x : 0.0f;
                scratch_act[i] = r * r;
            }
        }
        grid.sync();

        // Down proj: expert_down @ scratch_act → scratch_out [hs]
        for (int row = (int)blockIdx.x; row < hs; row += (int)gridDim.x) {
            const int group_size = 32;
            const int group_bytes = 20;
            const int num_groups = (eis + group_size - 1) / group_size;
            const unsigned char* row_data = expert_down + (long long)row * (long long)(num_groups * group_bytes);
            float acc = 0.0f;
            for (int g = tid; g < num_groups; g += (int)blockDim.x) {
                const unsigned char* gp = row_data + g * group_bytes;
                float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
                float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
                int base = g * group_size;
                int count = min(group_size, eis - base);
                for (int i = 0; i < count; i += 2) {
                    unsigned char byte = gp[i / 2];
                    float v0 = (float)(byte & 0xF) * sc + mn;
                    float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                    acc += v0 * scratch_act[base + i] + v1 * scratch_act[base + i + 1];
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                acc += __shfl_down(acc, offset);
            extern __shared__ float shared[];
            if ((tid & 31) == 0) shared[tid >> 5] = acc;
            __syncthreads();
            if (tid == 0) {
                float total = shared[0] + shared[1] + shared[2] + shared[3]
                            + shared[4] + shared[5] + shared[6] + shared[7];
                scratch_out[row] = total;
            }
        }
        grid.sync();

        // Weighted accumulate: output += w * scratch_out
        for (int i = (int)blockIdx.x * (int)blockDim.x + tid; i < hs; i += (int)gridDim.x * (int)blockDim.x) {
            output[i] += w * scratch_out[i];
        }
        grid.sync();
    }
}

