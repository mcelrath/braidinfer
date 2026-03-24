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

    // Online softmax state (per-thread, then reduced)
    float m = -1e30f;  // running max
    float d = 0.0f;    // running denominator
    // Running accumulator for output: stored as head_dim values — too large for
    // registers per thread if head_dim=128 and block=128. Instead, we do two
    // passes: first compute online softmax to get m,d; second compute output.
    // But that requires re-reading KV cache twice — acceptable for correctness.
    //
    // Pass 1: compute m and d via online softmax over all seq positions

    // Each thread computes a partial dot product for a subset of seq positions
    // and maintains its own (m, d) state.
    // We use a strided loop: thread tid processes positions tid, tid+block_size, ...
    // Then we reduce (m, d) across threads.

    // For simplicity: each thread owns a contiguous chunk of the head_dim dot product.
    // For a given seq position t, all threads collaborate on the dot product.
    // Then one thread (tid=0) updates (m, d).
    // This serializes seq positions but parallelizes the dot product.

    // Pass 1: online softmax
    m = -1e30f;
    d = 0.0f;

    // We'll store per-position scores to reuse in pass 2 — but seq_len can be
    // large. Instead, do two full passes over KV cache.

    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += block_size) {
            dot += q_ptr[i] * k_ptr[i];
        }
        // Reduce dot across threads
        partial[tid] = dot;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        // tid=0 updates m, d and broadcasts via shared mem
        if (tid == 0) {
            float score = partial[0] * scale;
            float m_new = (score > m) ? score : m;
            d = d * expf(m - m_new) + expf(score - m_new);
            m = m_new;
            partial[0] = m;  // broadcast m
            partial[1] = d;  // broadcast d
        }
        __syncthreads();
        m = partial[0];
        d = partial[1];
        __syncthreads();
    }

    // Pass 2: compute output with known (m, d)
    float* out_ptr = output + q_head * head_dim;
    // Initialize output accumulators per-thread
    // Each thread accumulates its subset of head_dim dims
    // Use partial2 for the running softmax denominator recheck
    // Zero out output
    for (int i = tid; i < head_dim; i += block_size) {
        out_ptr[i] = 0.0f;
    }
    __syncthreads();

    // Recompute running max for pass 2 (same m we ended with)
    // We need to recompute scores. m is stable (final value).
    for (int t = 0; t < seq_len; t++) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head) * head_dim;
        const float* v_ptr = v_cache + (t * num_kv_heads + kv_head) * head_dim;

        // Compute dot(q, k[t])
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
        // Broadcast score
        if (tid == 0) {
            partial[0] = partial[0] * scale;
        }
        __syncthreads();
        float score = partial[0];
        float weight = expf(score - m) / d;

        // Accumulate weight * v[t] into output
        for (int i = tid; i < head_dim; i += block_size) {
            out_ptr[i] += weight * v_ptr[i];
        }
        __syncthreads();
    }
}
