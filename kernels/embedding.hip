#include <hip/hip_runtime.h>

// Embedding lookup: y = embed_table[token_id]
// embed_table is [vocab_size x hidden_size], token_id is scalar.
// Single block copies one row.

extern "C" __global__ void embedding_f32(
    float* __restrict__ output,
    const float* __restrict__ embed_table,
    const int token_id,
    const int hidden_size
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* row = embed_table + (long long)token_id * hidden_size;

    for (int i = tid; i < hidden_size; i += stride) {
        output[i] = row[i];
    }
}
