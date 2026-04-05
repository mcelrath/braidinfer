// Lean MoE GEMV worker: single-block expert FFN.
// Regular (non-cooperative) launch with 1 block.
// All 256 threads cooperate on each GEMV row via shared memory reduction.
#include <hip/hip_runtime.h>
#include "megakernel_common.h"

#define MOE_MAX_BATCH 16

struct MoeWorkerQueue {
    volatile uint32_t seq_num;
    volatile uint32_t shutdown;
    uint32_t num_instructions;
    uint32_t _pad;
    uint64_t inst[MOE_MAX_BATCH * INST_SIZE_WORDS];
    volatile uint32_t ack;
    uint32_t _pad2[3];
};

// Single-block GEMV: 256 threads process each row sequentially.
__device__ static void block_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int out_dim, int in_dim
) {
    const int tid = threadIdx.x, stride = blockDim.x;
    const int gs = 32, gb = 20, ng = (in_dim + gs - 1) / gs;
    extern __shared__ float shm[];
    for (int row = 0; row < out_dim; row++) {
        const unsigned char* rd = weight + (long long)row * ng * gb;
        float acc = 0.0f;
        for (int g = tid; g < ng; g += stride) {
            const unsigned char* gp = rd + g * gb;
            float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
            float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
            int base = g * gs, cnt = min(gs, in_dim - base);
            for (int i = 0; i < cnt; i += 2) {
                unsigned char byte = gp[i / 2];
                acc += ((float)(byte & 0xF) * sc + mn) * input[base + i]
                     + ((float)((byte >> 4) & 0xF) * sc + mn) * input[base + i + 1];
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if ((tid & 31) == 0) shm[tid >> 5] = acc;
        __syncthreads();
        if (tid < 8) {
            acc = shm[tid];
            for (int o = 4; o > 0; o >>= 1) acc += __shfl_down(acc, o);
            if (tid == 0) output[row] = acc;
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256) void moe_gemv_worker(
    volatile MoeWorkerQueue* queue,
    float* scratch_gate,
    float* scratch_up,
    float* scratch_act
) {
    if (threadIdx.x == 0) printf("LEAN: kernel started queue=%p initial_seq=%u\n", (void*)queue, queue->seq_num);
    uint32_t last_seq = 0;

    while (true) {
        if (threadIdx.x == 0) {
            int poll_count = 0;
            while (true) {
                if (queue->shutdown) { queue->seq_num = 0xFFFFFFFFu; break; }
                uint32_t s = queue->seq_num;
                if (s > last_seq) break;
                if (++poll_count == 100000000) {
                    printf("LEAN: still polling last_seq=%u cur_seq=%u queue=%p\n", last_seq, s, (void*)queue);
                    poll_count = 0;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();

        uint32_t seq = queue->seq_num;
        if (seq == 0xFFFFFFFFu) return;

        if (threadIdx.x == 0) {
            printf("LEAN: seq=%u num_inst=%u\n", seq, queue->num_instructions);
            // Print first instruction pointers for diagnosis
            const u64* first = (const u64*)(queue->inst);
            printf("LEAN: inst[4]=%p [5]=%p [7]=%p eis=%d hs=%d\n",
                (void*)first[4], (void*)first[5], (void*)first[7],
                (int)(first[8]>>32), (int)(first[8]&0xFFFFFFFF));
        }
        uint32_t num_inst = queue->num_instructions;
        if (num_inst == 0) num_inst = 1;

        __shared__ u64 sinst[INST_SIZE_WORDS];
        for (uint32_t pc = 0; pc < num_inst; pc++) {
            if (threadIdx.x < INST_SIZE_WORDS)
                sinst[threadIdx.x] = ((const u64*)(queue->inst))[pc * INST_SIZE_WORDS + threadIdx.x];
            __syncthreads();

            // OP_EXPERT_FFN
            float* output = (float*)(uintptr_t)sinst[4];
            const unsigned char* gate_up = (const unsigned char*)(uintptr_t)sinst[5];
            const unsigned char* down    = (const unsigned char*)(uintptr_t)sinst[6];
            const float* activation = (const float*)(uintptr_t)sinst[7];
            int eis = (int)(sinst[8] >> 32), hs = (int)(sinst[8] & 0xFFFFFFFF);
            int gu_rs = (int)sinst[9], has_gate = (int)sinst[10];
            u32 ew_bits = (u32)sinst[11];
            float ew; __builtin_memcpy(&ew, &ew_bits, sizeof(float));

            if (has_gate) {
                block_gemv_pcg32(scratch_gate, gate_up, activation, eis, hs);
                block_gemv_pcg32(scratch_up, gate_up + (long long)eis * gu_rs, activation, eis, hs);
                for (int i = threadIdx.x; i < eis; i += blockDim.x) {
                    float g = scratch_gate[i];
                    scratch_act[i] = (g / (1.0f + expf(-g))) * scratch_up[i];
                }
                __syncthreads();
            } else {
                block_gemv_pcg32(scratch_up, gate_up, activation, eis, hs);
                for (int i = threadIdx.x; i < eis; i += blockDim.x) {
                    float x = scratch_up[i]; float r = x > 0.0f ? x : 0.0f; scratch_act[i] = r * r;
                }
                __syncthreads();
            }
            block_gemv_pcg32(scratch_gate, down, scratch_act, hs, eis);
            for (int i = threadIdx.x; i < hs; i += blockDim.x)
                output[i] += ew * scratch_gate[i];
            __syncthreads();
        }

        last_seq = seq;
        if (threadIdx.x == 0) {
            printf("LEAN: acking seq=%u\n", seq);
            queue->ack = seq;
        }
    }
}
