#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = ((unsigned int)h) << 16;
    return v.f;
}

extern "C" __global__ void paged_attention_decode_f32(
    float* __restrict__ output,
    const float* __restrict__ q,
    const unsigned long long* __restrict__ page_table,
    const int* __restrict__ position_table,
    const float* __restrict__ inv_freq,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,
    const int chunk_tokens,
    const int rope_dim,
    const long long layer_k_offset,
    const long long layer_v_offset,
    const unsigned short* __restrict__ k_norm_weight
) {
    const int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int gqa_group = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_group;
    const float scale = rsqrtf((float)head_dim);
    const float* q_ptr = q + q_head * head_dim;

    extern __shared__ float shmem[];
    float* sh_k = shmem;
    float* partial = shmem;

    const int rope_pairs = rope_dim / 2;
    float m = -1e30f;
    float d = 0.0f;
    float v_acc = 0.0f;
    const int num_chunks = (seq_len + chunk_tokens - 1) / chunk_tokens;

    for (int c = 0; c < num_chunks; c++) {
        const char* chunk_base = (const char*)(page_table[c]);
        const float* k_base = (const float*)(chunk_base + layer_k_offset);
        const float* v_base = (const float*)(chunk_base + layer_v_offset);
        int tokens_in_chunk = chunk_tokens;
        if (c == num_chunks - 1) {
            tokens_in_chunk = seq_len - c * chunk_tokens;
        }

        for (int t = 0; t < tokens_in_chunk; t++) {
            const int token_idx = kv_head * chunk_tokens + t;
            if (tid < head_dim) {
                sh_k[tid] = k_base[token_idx * head_dim + tid];
            }
            __syncthreads();

            if (k_norm_weight != nullptr) {
                float my_k = (tid < head_dim) ? sh_k[tid] : 0.0f;
                partial[tid] = my_k * my_k;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) partial[tid] += partial[tid + s];
                    __syncthreads();
                }
                float k_rms = rsqrtf(partial[0] / (float)head_dim + 1e-6f);
                if (tid < head_dim) {
                    float w = 1.0f + bf16_to_f32(k_norm_weight[tid]);
                    sh_k[tid] = my_k * k_rms * w;
                }
                __syncthreads();
            }

            if (tid < rope_pairs) {
                const int pos = position_table[c * chunk_tokens + t];
                const float theta = (float)pos * inv_freq[tid];
                const float cos_state = cosf(theta);
                const float sin_state = sinf(theta);
                const int i0 = tid * 2;
                const int i1 = tid * 2 + 1;
                const float x0 = sh_k[i0];
                const float x1 = sh_k[i1];
                sh_k[i0] = x0 * cos_state - x1 * sin_state;
                sh_k[i1] = x0 * sin_state + x1 * cos_state;
            }
            __syncthreads();

            float dot = 0.0f;
            if (tid < head_dim) {
                dot = q_ptr[tid] * sh_k[tid];
            }
            partial[tid] = dot;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) partial[tid] += partial[tid + s];
                __syncthreads();
            }
            const float score = partial[0] * scale;
            const float m_new = fmaxf(m, score);
            const float correction = expf(m - m_new);
            const float w = expf(score - m_new);
            d = d * correction + w;

            if (tid < head_dim) {
                const float vval = v_base[(kv_head * chunk_tokens + t) * head_dim + tid];
                v_acc = v_acc * correction + w * vval;
            }
            m = m_new;
        }
    }

    float* out_ptr = output + q_head * head_dim;
    if (tid < head_dim) {
        out_ptr[tid] = v_acc / d;
    }
}
