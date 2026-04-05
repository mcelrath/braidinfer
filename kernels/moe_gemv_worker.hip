// Minimal lean worker: 4-arg poll+ack, regular launch.
// DIAGNOSTIC: same as Test A that worked, but launched with func.launch not cooperative.
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

extern "C" __global__ __launch_bounds__(256) void moe_gemv_worker(
    volatile MoeWorkerQueue* queue,
    float* scratch_gate,
    float* scratch_up,
    float* scratch_act
) {
    // Only thread 0 of block 0 runs.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)scratch_gate; (void)scratch_up; (void)scratch_act;

    uint32_t last = 0;
    while (true) {
        if (queue->shutdown) return;
        uint32_t s = queue->seq_num;
        if (s != last) {
            if (s == 0xFFFFFFFFu) return;
            last = s;
            queue->ack = s;
        }
        __builtin_amdgcn_s_sleep(1);
    }
}
