#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// mRoPE: Multimodal Rotary Position Embedding (in-place)
// Applies rotary embedding to first rope_dim dims of each head.
// For Qwen3.5-0.8B: rope_dim=64, mrope_section=[11,11,10], head_dim=256
//
// For each head, split rope_dim dims into 3 sections; each section uses
// a different position_id (temporal, height, width). For text-only,
// all 3 position_ids are the same (the token position).
//
// One block per head. Threads handle pairs of dims.

extern "C" __global__ void mrope_f32(
    float* __restrict__ q,         // [num_q_heads, head_dim] in-place
    float* __restrict__ k,         // [num_kv_heads, head_dim] in-place
    const float* __restrict__ inv_freq,  // [rope_dim/2]
    const int* __restrict__ position_ids, // [3]: temporal, height, width
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int rope_dim,           // = 32 for Qwen3.5-0.8B
    const int section0_pairs,     // = 11
    const int section1_pairs,     // = 11
    const int section2_pairs      // = 10
) {
    // blockIdx.x = head index (q heads first, then kv heads)
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    float* head_ptr;
    if (head_idx < num_q_heads) {
        head_ptr = q + head_idx * head_dim;
    } else {
        head_ptr = k + (head_idx - num_q_heads) * head_dim;
    }

    // Each pair (2i, 2i+1) maps to a section based on pair index
    // section 0: pairs [0, section0_pairs)
    // section 1: pairs [section0_pairs, section0_pairs + section1_pairs)
    // section 2: pairs [section0_pairs + section1_pairs, rope_dim/2)
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
        float theta = (float)pos * inv_freq[pair];
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        int i0 = pair * 2;
        int i1 = pair * 2 + 1;
        float x0 = head_ptr[i0];
        float x1 = head_ptr[i1];
        head_ptr[i0] = x0 * cos_theta - x1 * sin_theta;
        head_ptr[i1] = x0 * sin_theta + x1 * cos_theta;
    }
}
