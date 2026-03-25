#include <hip/hip_runtime.h>
#include "bf16_utils.h"

// Embedding lookup: y = embed_table[token_id]
// embed_table is [vocab_size x hidden_size] BF16, output is f32.
// Single block copies one row with bf16→f32 conversion.

extern "C" __global__ void embedding_f32(
    float* __restrict__ output,
    const unsigned short* __restrict__ embed_table,
    const int token_id,
    const int hidden_size
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const unsigned short* row = embed_table + (long long)token_id * hidden_size;

    for (int i = tid; i < hidden_size; i += stride) {
        output[i] = bf16_to_f32(row[i]);
    }
}
