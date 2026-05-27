// op_kv_write_paged_batch — bd srg6.4 Phase 3b.
//
// Paged K-or-V batched writer. Replaces N × OP_D2D_COPY per (token, head)
// pair emitted by the per-token paged write path. One instruction walks
// the page_table internally to locate the destination chunk + in-chunk
// offset for each (token, head) virtual block.
//
// Virtual-block dispatch (grid_x = n_tokens * local_nkh):
//   vblock = blockIdx.x (after the virtual-block wraparound in
//            dispatch_opcode's `for (vb = blockIdx.x; vb < grid_x; ...)`).
//   token       = vblock / local_nkh
//   head_local  = vblock % local_nkh
//   head_global = local_kv_head_start + head_local
//
// Destination layout (matches the per-token paged decode pattern in
// compile_attention.rs:100-126, which writes head h to
// chunk_base + layer_kv_offset
//                + h * (chunk_tokens * head_dim * sizeof(float))
//                + in_chunk * head_dim * sizeof(float)
// for one (token, head) pair):
//
//   abs_pos    = start_pos + token
//   chunk_idx  = abs_pos / chunk_tokens
//   in_chunk   = abs_pos % chunk_tokens
//   chunk_base = (char*)(uintptr_t)page_table[chunk_idx]
//   dst        = (float*)(chunk_base + layer_kv_offset
//                  + head_global * chunk_tokens * head_dim * sizeof(float)
//                  + in_chunk * head_dim * sizeof(float))
//
// Each virtual block writes to a distinct destination, so no intra-op
// cross-block sync is required. The post-op grid barrier in
// dispatch_opcode serializes against the next instruction.

#pragma once
#include "megakernel_common.h"

__device__ inline void op_kv_write_paged_batch(const u64* inst, u32 vb) {
    const KvWritePagedBatchInst* i = (const KvWritePagedBatchInst*)inst;
    const float* src = i->src;
    const uint64_t* page_table = i->page_table;
    uint64_t layer_kv_offset = i->layer_kv_offset;

    int start_pos = (int)(i->start_pos & 0xFFFFFFFFu);
    int n_tokens  = (int)(i->start_pos >> 32);

    int local_nkh      = (int)(i->head_slice & 0xFFFFu);
    int local_kv_start = (int)((i->head_slice >> 16) & 0xFFFFu);
    int head_dim       = (int)((i->head_slice >> 32) & 0xFFFFu);
    int chunk_tokens   = (int)((i->head_slice >> 48) & 0xFFFFu);

    int total = n_tokens * local_nkh;
    if ((int)vb >= total) return;

    int token       = (int)vb / local_nkh;
    int head_local  = (int)vb % local_nkh;
    int head_global = local_kv_start + head_local;

    int abs_pos   = start_pos + token;
    int chunk_idx = abs_pos / chunk_tokens;
    int in_chunk  = abs_pos % chunk_tokens;

    char* chunk_base = (char*)(uintptr_t)page_table[chunk_idx];
    float* dst = (float*)(chunk_base + layer_kv_offset
                          + (size_t)head_global * (size_t)chunk_tokens * (size_t)head_dim * sizeof(float)
                          + (size_t)in_chunk * (size_t)head_dim * sizeof(float));

    const float* src_row = src
        + (size_t)((size_t)token * (size_t)local_nkh + (size_t)head_local) * (size_t)head_dim;

    // Cooperative-copy head_dim floats from src_row -> dst.
    for (int t = (int)threadIdx.x; t < head_dim; t += (int)blockDim.x) {
        dst[t] = src_row[t];
    }
}
