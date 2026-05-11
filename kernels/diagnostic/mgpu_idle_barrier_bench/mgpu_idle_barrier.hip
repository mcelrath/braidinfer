// Focused microbench for the multi-GPU shutdown-path atomic_block_barrier wedge.
//
// PRODUCTION OBSERVATION:
//   q8 4-GPU MoE wedges at SHUTDOWN time ~1/5 to 4/8 runs. Worker is at the
//   atomic_block_barrier called after a prolonged idle period (mailbox spin-poll
//   for shutdown signal). Synthetic mgpu_barrier_repro (.worktrees/diag-mgpu-repro/)
//   tests barrier-under-active-workload and PASSES 10M+ events — it does NOT
//   exercise the long-idle-then-barrier pattern.
//
// HYPOTHESIS:
//   atomic_block_barrier on cached __device__ static counter intermittently wedges
//   when preceded by a prolonged idle period (10ms+ of spin-poll on host-mapped
//   memory) under simultaneous multi-GPU activity. Cause: HW-level scheduler /
//   cache state drift during idle, with all 4 GPUs simultaneously wake-and-barrier.
//
// THIS BENCH:
//   Each GPU runs a single cooperative kernel that performs N iterations:
//     1) Block 0 thread 0 spin-polls a host-mapped trigger byte until set.
//     2) ALL blocks call atomic_block_barrier.
//     3) Block 0 thread 0 writes iteration counter to a host-mapped completion field.
//   Host triggers each iteration sequentially with a configurable idle duration
//   between triggers. Records timeouts as wedges.
//
// BUILD:
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 -std=c++17 \
//       -ffp-contract=fast -mwavefrontsize64 mgpu_idle_barrier.hip \
//       -I../.. -o mgpu_idle_barrier
//
// RUN:
//   python3 scripts/launch-gpu.py -g 4 --timeout 600 -- \
//       ./mgpu_idle_barrier --gpus 4 --iters 100 --idle-ms 10

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

#include "../../rdna3_sync.h"

namespace cg = cooperative_groups;
using braidinfer::rdna3::GridBarrierState;
using braidinfer::rdna3::atomic_block_barrier;

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

struct GpuMailbox {
    volatile uint32_t trigger;       // host writes 1 to start an iteration
    volatile uint32_t completed;     // kernel writes iteration count on barrier exit
    volatile uint32_t shutdown;      // host writes 1 to make kernel exit
    volatile uint32_t progress_stage; // kernel writes 100/101/102 stage marker
    uint32_t          _pad[12];      // pad to 64 bytes
};

extern "C" __global__ __launch_bounds__(256, 2) void idle_barrier_kernel(
    volatile GpuMailbox* __restrict__ mbox,
    GridBarrierState*    __restrict__ gbs,
    int                              total_iters,
    int                              barriers_per_iter,
    float*               __restrict__ p2p_target,  // GPU 0 UC, can be null
    int                              gpu_id)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0 && bid == 0) {
        printf("[gpu] idle_barrier_kernel started: blocks=%d, total_iters=%d, barriers_per_iter=%d\n",
               gridDim.x, total_iters, barriers_per_iter);
    }

    for (int iter = 0; iter < total_iters; iter++) {
        // Phase 1: Block 0 thread 0 spin-polls trigger. Other threads/blocks
        // wait at atomic_block_barrier below (production pattern: only one
        // thread polls mailbox to avoid PCIe storm).
        if (tid == 0 && bid == 0) {
            while (true) {
                if (mbox->shutdown) {
                    mbox->progress_stage = 99;
                    return;
                }
                if (mbox->trigger) {
                    mbox->trigger = 0;  // ack
                    mbox->progress_stage = 100;  // about to barrier
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }

        // The critical barrier — preceded by prolonged idle (poll above).
        // This is the SHUTDOWN-PATH equivalent.
        atomic_block_barrier(gbs);

        // Phase 2: simulate the per-batch activity. P2P writes to GPU 0
        // UC memory + back-to-back atomic_block_barriers — production's
        // coop_copy → atomic_block_barrier pattern in op_moe_ffn_remote.
        for (int b = 0; b < barriers_per_iter; b++) {
            if (p2p_target != nullptr && gpu_id != 0) {
                // Each thread writes ONE float to the GPU 0 P2P target.
                int p2p_idx = ((b * 256) + tid) % 4096;
                float v = (float)(iter * 31 + b * 7 + bid * 3 + tid);
                p2p_target[gpu_id * 4096 + p2p_idx] = v;
                __threadfence();
            }
            atomic_block_barrier(gbs);
        }

        // Phase 3: block 0 thread 0 writes completion. If we reach here,
        // all barriers worked.
        if (tid == 0 && bid == 0) {
            mbox->progress_stage = 101;  // passed barriers
            __threadfence();
            mbox->completed = (uint32_t)(iter + 1);
        }
    }

    if (tid == 0 && bid == 0) {
        mbox->progress_stage = 200;  // clean exit
    }
}

int main(int argc, char** argv) {
    int num_gpus = 4;
    int iters    = 100;
    int idle_ms  = 10;
    int blocks   = 96;
    int timeout_ms = 5000;
    int barriers_per_iter = 100;  // simulate accumulated g_gbs state

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus") && i+1 < argc)        num_gpus    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc)  iters       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--idle-ms") && i+1 < argc) idle_ms    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i+1 < argc) blocks      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timeout-ms") && i+1 < argc) timeout_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--barriers-per-iter") && i+1 < argc) barriers_per_iter = atoi(argv[++i]);
    }

    printf("Config: %d GPUs, %d iters, idle_ms=%d, blocks=%d, timeout_ms=%d, barriers_per_iter=%d\n",
           num_gpus, iters, idle_ms, blocks, timeout_ms, barriers_per_iter);

    // Per-GPU mailboxes (host-mapped UC).
    std::vector<GpuMailbox*> host_mboxes(num_gpus);
    std::vector<GpuMailbox*> dev_mboxes(num_gpus);
    std::vector<GridBarrierState*> gbs_ptrs(num_gpus);
    std::vector<hipStream_t> streams(num_gpus);

    for (int g = 0; g < num_gpus; g++) {
        CHECK(hipSetDevice(g));
        CHECK(hipHostMalloc(&host_mboxes[g], sizeof(GpuMailbox),
                            hipHostMallocMapped));
        memset((void*)host_mboxes[g], 0, sizeof(GpuMailbox));
        CHECK(hipHostGetDevicePointer((void**)&dev_mboxes[g],
                                       host_mboxes[g], 0));
        CHECK(hipMalloc(&gbs_ptrs[g], sizeof(GridBarrierState)));
        CHECK(hipMemset(gbs_ptrs[g], 0, sizeof(GridBarrierState)));
        CHECK(hipStreamCreate(&streams[g]));
    }

    // Allocate P2P UC buffer on GPU 0, get peer ptrs for workers (matches
    // production output_slots usage in op_moe_dispatch_post).
    CHECK(hipSetDevice(0));
    float* p2p_buf_gpu0 = nullptr;
    constexpr unsigned int kHipDeviceMallocUncached = 0x3;
    CHECK(hipExtMallocWithFlags((void**)&p2p_buf_gpu0,
                                 num_gpus * 4096 * sizeof(float),
                                 kHipDeviceMallocUncached));
    CHECK(hipMemset(p2p_buf_gpu0, 0, num_gpus * 4096 * sizeof(float)));
    // Enable P2P access from each worker GPU to GPU 0.
    for (int g = 1; g < num_gpus; g++) {
        CHECK(hipSetDevice(g));
        int can_access = 0;
        CHECK(hipDeviceCanAccessPeer(&can_access, g, 0));
        if (can_access) {
            hipError_t e = hipDeviceEnablePeerAccess(0, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "hipDeviceEnablePeerAccess(0) from gpu %d: %s\n",
                        g, hipGetErrorString(e));
                std::exit(1);
            }
        }
    }

    // Launch the kernel on all GPUs (cooperative).
    for (int g = 0; g < num_gpus; g++) {
        CHECK(hipSetDevice(g));
        void* args[6] = {
            (void*)&dev_mboxes[g],
            (void*)&gbs_ptrs[g],
            (void*)&iters,
            (void*)&barriers_per_iter,
            (void*)&p2p_buf_gpu0,
            (void*)&g,
        };
        CHECK(hipLaunchCooperativeKernel(
            (void*)idle_barrier_kernel,
            dim3(blocks, 1, 1),
            dim3(256, 1, 1),
            args,
            0,  // shared mem
            streams[g]));
    }

    // Drive iterations.
    int wedges = 0;
    int max_wedge_stage = -1;
    int max_wedge_gpu = -1;
    int max_wedge_iter = -1;
    auto t_start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iters; iter++) {
        // Idle delay BEFORE triggering — matches production where worker
        // is in idle poll for some time before the shutdown-path barrier.
        if (idle_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
        }

        // Trigger all GPUs simultaneously.
        for (int g = 0; g < num_gpus; g++) {
            host_mboxes[g]->trigger = 1;
        }

        // Wait for all GPUs to ack the iteration via mbox->completed.
        auto t_iter_start = std::chrono::steady_clock::now();
        std::vector<bool> done(num_gpus, false);
        int done_count = 0;
        while (done_count < num_gpus) {
            for (int g = 0; g < num_gpus; g++) {
                if (!done[g] && host_mboxes[g]->completed >= (uint32_t)(iter + 1)) {
                    done[g] = true;
                    done_count++;
                }
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_iter_start).count();
            if (elapsed > timeout_ms) {
                for (int g = 0; g < num_gpus; g++) {
                    if (!done[g]) {
                        wedges++;
                        int stage = (int)host_mboxes[g]->progress_stage;
                        printf("  WEDGE iter=%d gpu=%d stage=%d (completed=%u, expected>=%d)\n",
                               iter, g, stage, host_mboxes[g]->completed, iter + 1);
                        if (stage > max_wedge_stage) {
                            max_wedge_stage = stage;
                            max_wedge_gpu = g;
                            max_wedge_iter = iter;
                        }
                    }
                }
                break;  // skip remaining iterations on wedge
            }
        }
        if (done_count < num_gpus) break;
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count() / 1000.0;

    printf("\nSummary: %d iters, %d wedges, total=%.2fs\n", iters, wedges, total_s);
    if (wedges > 0) {
        printf("  Last wedge: gpu=%d iter=%d stage=%d\n",
               max_wedge_gpu, max_wedge_iter, max_wedge_stage);
        printf("  (stage 99=saw shutdown, 100=about to barrier, 101=passed barrier, 200=clean)\n");
    }

    // Signal shutdown so kernels exit cleanly.
    for (int g = 0; g < num_gpus; g++) {
        host_mboxes[g]->shutdown = 1;
    }
    // Wait briefly for kernels to drain.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return wedges > 0 ? 1 : 0;
}
