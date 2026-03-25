#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// GQA Decode Attention (single new token against KV cache)
// num_q_heads / num_kv_heads = GQA group size (e.g. 8 for Qwen3.5-0.8B)
//
// For each q head h:
//   kv_head = h / gqa_group
//   score[t] = dot(q[h], k_cache[t, kv_head]) / sqrt(head_dim)
//   output[h] = sum_t(softmax(score)[t] * v_cache[t, kv_head])
//
// Uses online softmax (Milakov & Gimelshein) to avoid materializing score vector.
// One block per Q head. Threads collaborate on dot products and reduction.

extern "C" __global__ void gqa_attention_decode_f32(
    float* __restrict__ output,        // [num_q_heads, head_dim]
    const float* __restrict__ q,       // [num_q_heads, head_dim]
    const float* __restrict__ k_cache, // [max_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_cache, // [max_seq_len, num_kv_heads, head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,
    const int max_seq_len
) {
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int gqa_group = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_group;

    const float scale = 1.0f / sqrtf((float)head_dim);
    const float* q_ptr = q + q_head * head_dim;

    // Shared memory layout:
    // [0..block_size): partial dot products
    // [block_size..2*block_size): reduction for online softmax numerator
    extern __shared__ float shmem[];
    float* partial = shmem;

    // Single-pass online softmax with register V accumulation
    // acc[] holds one element per strided position; head_dim/block_size must be <= 4
    if (head_dim > 4 * block_size) __builtin_trap();
    float m = -1e30f;
    float d = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head) * head_dim;
        const float* v_ptr = v_cache + (t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += block_size) {
            dot += q_ptr[i] * k_ptr[i];
        }
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
