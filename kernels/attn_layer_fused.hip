#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused attention decode layer for Qwen3.5-0.8B (and compatible GQA models).
// Split into 4 sub-kernels to respect grid-sync limitations:
//   1. attn_fused_proj_f32      — RMSNorm + QKV projections → scratch
//   2. attn_fused_rope_cache_f32 — mRoPE on Q,K in scratch + write K,V to KV cache
//   3. attn_fused_attention_f32 — GQA decode attention → attn_out in scratch
//   4. attn_fused_output_f32    — output projection + residual → final output
//
// Scratch layout (floats):
//   [0       .. 2048)  q:       [num_q_heads=8,  head_dim=256]
//   [2048    .. 2560)  k:       [num_kv_heads=2, head_dim=256]
//   [2560    .. 3072)  v:       [num_kv_heads=2, head_dim=256]
//   [3072    .. 5120)  attn_out:[num_q_heads=8,  head_dim=256]
// Total: 5120 floats = 20 KB

// ---------------------------------------------------------------------------
// Sub-kernel 1: RMSNorm + QKV projections
// Grid:  (max(q_out_dim, k_out_dim, v_out_dim)) = q_out_dim = num_q_heads*head_dim
//        blocks process rows of W_q, W_k, W_v independently.
// Block: 256 threads (dot-product reduction along hidden_size=1024).
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_fused_proj_f32(
    float* __restrict__ scratch,          // [5120]: q at 0, k at 2048, v at 2560
    const float* __restrict__ input,      // [hidden_size]
    const float* __restrict__ rms_weight, // [hidden_size]
    const float* __restrict__ w_q,        // [num_q_heads*head_dim, hidden_size]
    const float* __restrict__ w_k,        // [num_kv_heads*head_dim, hidden_size]
    const float* __restrict__ w_v,        // [num_kv_heads*head_dim, hidden_size]
    const int hidden_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float eps
) {
    // Phase A: RMSNorm. We need the scale factor before projections.
    // Use block 0 to compute rms and broadcast via a scratch float.
    // But we have many blocks. Better: each block independently computes rms.
    // hidden_size=1024, block=256 threads: 4 iters each → fast.

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Compute sum-of-squares (shared across all blocks, same input)
    __shared__ float shmem[256];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = input[i];
        sum_sq += val * val;
    }
    shmem[tid] = sum_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shmem[0] / (float)hidden_size + eps);

    // Phase B: Projection. blockIdx.x identifies which output element to compute.
    // Layout: rows [0 .. q_out_dim) → W_q, [q_out_dim .. q_out_dim+kv_out_dim) → W_k,
    //         [q_out_dim+kv_out_dim .. q_out_dim+2*kv_out_dim) → W_v
    const int q_out_dim  = num_q_heads  * head_dim;  // 2048
    const int kv_out_dim = num_kv_heads * head_dim;  // 512
    const int row = blockIdx.x;

    const float* w_row;
    float* out_ptr;

    if (row < q_out_dim) {
        w_row   = w_q + (long)row * hidden_size;
        out_ptr = scratch + row;                       // q at offset 0
    } else if (row < q_out_dim + kv_out_dim) {
        int kr = row - q_out_dim;
        w_row   = w_k + (long)kr * hidden_size;
        out_ptr = scratch + q_out_dim + kr;            // k at offset 2048
    } else {
        int vr = row - q_out_dim - kv_out_dim;
        w_row   = w_v + (long)vr * hidden_size;
        out_ptr = scratch + q_out_dim + kv_out_dim + vr; // v at offset 2560
    }

    // dot(w_row, rms_normalized_input)
    float acc = 0.0f;
    for (int i = tid; i < hidden_size; i += block_size) {
        float normed = input[i] * rms * (1.0f + rms_weight[i]);
        acc += w_row[i] * normed;
    }
    shmem[tid] = acc;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        *out_ptr = shmem[0];
    }
}

// ---------------------------------------------------------------------------
// Sub-kernel 2: mRoPE on Q and K in scratch + write K,V to KV cache
// Grid:  (num_q_heads + num_kv_heads) blocks — one per head (matching mrope.hip)
// Block: 32 threads (32 pairs max for rope_dim=64)
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_fused_rope_cache_f32(
    float* __restrict__ scratch,          // Q at 0, K at q_out_dim
    float* __restrict__ k_cache,          // [max_seq_len, num_kv_heads, head_dim]
    float* __restrict__ v_cache,          // [max_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ inv_freq,   // [rope_dim/2]
    const int* __restrict__ position_ids, // [3]: temporal, height, width
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int rope_dim,
    const int section0_pairs,
    const int section1_pairs,
    const int section2_pairs,
    const int seq_pos,      // index in cache to write (current token position)
    const int max_seq_len
) {
    const int head_idx  = blockIdx.x;
    const int tid       = threadIdx.x;
    const int q_out_dim = num_q_heads * head_dim;
    const int kv_out_dim = num_kv_heads * head_dim;

    // --- mRoPE (same logic as mrope.hip) ---
    float* head_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = scratch + head_idx * head_dim;
    } else {
        head_ptr = scratch + q_out_dim + (head_idx - num_q_heads) * head_dim;
    }

    const int total_pairs = rope_dim / 2;
    for (int pair = tid; pair < total_pairs; pair += blockDim.x) {
        int section;
        if (pair < section0_pairs) {
            section = 0;
        } else if (pair < section0_pairs + section1_pairs) {
            section = 1;
        } else {
            section = 2;
        }
        int pos = position_ids[section];
        float theta     = (float)pos * inv_freq[pair];
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        int i0 = pair * 2;
        int i1 = pair * 2 + 1;
        float x0 = head_ptr[i0];
        float x1 = head_ptr[i1];
        head_ptr[i0] = x0 * cos_theta - x1 * sin_theta;
        head_ptr[i1] = x0 * sin_theta + x1 * cos_theta;
    }
    __syncthreads();

    // --- Write K and V to cache for KV heads ---
    // Only KV head blocks do the cache write
    if (head_idx >= num_q_heads) {
        int kv_head = head_idx - num_q_heads;
        const float* k_ptr = scratch + q_out_dim + kv_head * head_dim;
        const float* v_ptr = scratch + q_out_dim + kv_out_dim + kv_head * head_dim;

        float* k_cache_slot = k_cache + ((long)seq_pos * num_kv_heads + kv_head) * head_dim;
        float* v_cache_slot = v_cache + ((long)seq_pos * num_kv_heads + kv_head) * head_dim;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            k_cache_slot[i] = k_ptr[i];
            v_cache_slot[i] = v_ptr[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-kernel 3: GQA decode attention
// Reads Q from scratch[0..q_out_dim), K/V from cache[0..seq_len).
// Writes attn_out to scratch[q_out_dim + kv_out_dim*2 ..] (offset 3072).
// Grid:  (num_q_heads) blocks — one per Q head
// Block: 256 threads (matches gqa_attention.hip pattern for head_dim=256)
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_fused_attention_f32(
    float* __restrict__ scratch,          // q at 0; writes attn_out at offset 3072
    const float* __restrict__ k_cache,   // [max_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_cache,   // [max_seq_len, num_kv_heads, head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,       // total positions in cache (after write)
    const int max_seq_len
) {
    const int q_out_dim   = num_q_heads  * head_dim;  // 2048
    const int kv_out_dim  = num_kv_heads * head_dim;  // 512
    const int attn_offset = q_out_dim + kv_out_dim * 2; // 3072

    const int q_head     = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;
    const int gqa_group  = num_q_heads / num_kv_heads;
    const int kv_head    = q_head / gqa_group;
    const float scale    = 1.0f / sqrtf((float)head_dim);

    const float* q_ptr = scratch + q_head * head_dim;
    float*       o_ptr = scratch + attn_offset + q_head * head_dim;

    extern __shared__ float shmem[];

    // Pass 1: online softmax to compute global (m, d)
    float m = -1e30f;
    float d = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + ((long)t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += block_size) {
            dot += q_ptr[i] * k_ptr[i];
        }
        shmem[tid] = dot;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) shmem[tid] += shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float score = shmem[0] * scale;
            float m_new = (score > m) ? score : m;
            d = d * expf(m - m_new) + expf(score - m_new);
            m = m_new;
            shmem[0] = m;
            shmem[1] = d;
        }
        __syncthreads();
        m = shmem[0];
        d = shmem[1];
        __syncthreads();
    }

    // Zero output
    for (int i = tid; i < head_dim; i += block_size) {
        o_ptr[i] = 0.0f;
    }
    __syncthreads();

    // Pass 2: accumulate weighted values
    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + ((long)t * num_kv_heads + kv_head) * head_dim;
        const float* v_ptr = v_cache + ((long)t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += block_size) {
            dot += q_ptr[i] * k_ptr[i];
        }
        shmem[tid] = dot;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) shmem[tid] += shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            shmem[0] = shmem[0] * scale;
        }
        __syncthreads();
        float score  = shmem[0];
        float weight = expf(score - m) / d;
        for (int i = tid; i < head_dim; i += block_size) {
            o_ptr[i] += weight * v_ptr[i];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Sub-kernel 4: Output projection + residual
// Reads attn_out from scratch[3072 .. 5120) → [num_q_heads*head_dim=2048]
// W_o is [hidden_size, num_q_heads*head_dim] = [1024, 2048]
// output[i] = dot(W_o[i], attn_out) + input[i]
// Grid:  (hidden_size) blocks — one per output row
// Block: 256 threads
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_fused_output_f32(
    float* __restrict__ output,        // [hidden_size]
    const float* __restrict__ scratch, // attn_out at offset 3072
    const float* __restrict__ input,   // [hidden_size] — residual
    const float* __restrict__ w_o,     // [hidden_size, num_q_heads*head_dim]
    const int hidden_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int q_out_dim   = num_q_heads  * head_dim;  // 2048
    const int kv_out_dim  = num_kv_heads * head_dim;  // 512
    const int attn_offset = q_out_dim + kv_out_dim * 2; // 3072

    const float* attn_out = scratch + attn_offset;

    const int row        = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    if (row >= hidden_size) return;

    const float* w_row = w_o + (long)row * q_out_dim;

    __shared__ float shmem[256];
    float acc = 0.0f;
    for (int i = tid; i < q_out_dim; i += block_size) {
        acc += w_row[i] * attn_out[i];
    }
    shmem[tid] = acc;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        output[row] = shmem[0] + input[row];
    }
}
