// Host runner for persistent_worker_skeleton wedge reproducer. Single-binary
// multi-variant. Designed for the joint udi+braidinfer characterization of
// the 4-GPU q8 MoE wedge (bd: braidinfer-pky.3 / exterior_algebra-3pd).
//
// Interface (matches udi runner contract):
//   argv:  --variant V0|V1|V2|V3|V4 --n-gpus N --n-trials 1
//   stdout: one JSON object per line. Two event shapes:
//     {"event":"config", ...}                       -- once at startup
//     {"variant":..., "trial":..., "wedged":..., ...} -- per trial
//   exit:  0 on completion, 1 on launch failure
//
// This binary implements V0/V1/V2/V3. V4a (peer-GPU UC device queue) is a
// separate fixture and not yet built.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <atomic>

#define CHECK(call)                                                          \
    do {                                                                     \
        hipError_t e = (call);                                               \
        if (e != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            return 1;                                                        \
        }                                                                    \
    } while (0)

struct WedgeReproQueue {
    volatile uint32_t seq_num;
    volatile uint32_t ack;
    volatile uint32_t progress_pc;
    volatile uint32_t block_alive_count;
    volatile uint32_t shutdown;
    volatile uint32_t completed_dispatches;
    volatile uint32_t watchdog_alive;  // V5-only signal
    uint32_t _pad;
};
static_assert(sizeof(WedgeReproQueue) == 32, "WedgeReproQueue layout drift");

extern "C" __global__ void persistent_worker_skeleton(
    volatile WedgeReproQueue* q, int variant);

static const char* variant_name(int v) {
    switch (v) {
        case 0: return "V0";
        case 1: return "V1";
        case 2: return "V2";
        case 3: return "V3";
        case 4: return "V4";
        case 5: return "V5";
        default: return "Vunknown";
    }
}

static int parse_variant(const char* s) {
    if (strcmp(s, "V0") == 0) return 0;
    if (strcmp(s, "V1") == 0) return 1;
    if (strcmp(s, "V2") == 0) return 2;
    if (strcmp(s, "V3") == 0) return 3;
    if (strcmp(s, "V4") == 0) return 4;
    if (strcmp(s, "V5") == 0) return 5;
    return -1;
}

// Detect runtime git commit. Invokes `git rev-parse --short HEAD` from cwd.
// Falls back to "unknown" on failure.
static std::string detect_git_commit() {
    FILE* p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!p) return "unknown";
    char buf[64] = {0};
    if (fgets(buf, sizeof(buf), p) != nullptr) {
        size_t n = strlen(buf);
        if (n > 0 && buf[n - 1] == '\n') buf[n - 1] = '\0';
    }
    pclose(p);
    if (buf[0] == '\0') return std::string("unknown");
    return std::string(buf);
}

// Detect ROCm version from /opt/rocm/.info/version. Falls back to "unknown".
static std::string detect_rocm_version() {
    FILE* f = fopen("/opt/rocm/.info/version", "r");
    if (!f) return "unknown";
    char buf[64] = {0};
    if (fgets(buf, sizeof(buf), f) != nullptr) {
        size_t n = strlen(buf);
        if (n > 0 && buf[n - 1] == '\n') buf[n - 1] = '\0';
    }
    fclose(f);
    if (buf[0] == '\0') return std::string("unknown");
    return std::string(buf);
}

static std::string detect_hw(int device_id) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device_id) != hipSuccess) return "unknown";
    // gcnArchName is the canonical "gfx1100"-style string on ROCm 7.x.
    return std::string(p.gcnArchName);
}

static int max_active_blocks_per_sm(const void* func, int block_size, size_t shared) {
    int n = 0;
    hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &n, func, block_size, shared);
    (void)e;
    return n;
}

int main(int argc, char** argv) {
    int variant = -1;
    int n_gpus = -1;
    int n_trials = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variant = parse_variant(argv[++i]);
        } else if (strcmp(argv[i], "--n-gpus") == 0 && i + 1 < argc) {
            n_gpus = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-trials") == 0 && i + 1 < argc) {
            n_trials = atoi(argv[++i]);
        }
    }
    if (variant < 0 || variant > 5 || n_gpus < 1) {
        fprintf(stderr,
                "usage: %s --variant V0|V1|V2|V3|V4|V5 --n-gpus N [--n-trials 1]\n",
                argv[0]);
        return 1;
    }
    if (variant == 4) {
        // V4a not implemented in this binary. Emit a config + skip event.
        printf("{\"event\":\"config\",\"hw_detect\":\"%s\",\"rocm_detect\":\"%s\","
               "\"n_gpus\":%d,\"variant\":\"%s\",\"commit\":\"%s\","
               "\"skipped\":\"V4a not implemented in v1 skeleton (peer-GPU dispatcher TODO)\"}\n",
               detect_hw(0).c_str(), detect_rocm_version().c_str(),
               n_gpus, variant_name(variant), detect_git_commit().c_str());
        fflush(stdout);
        return 0;
    }

    // Force n_gpus=2 for V1 (control-axis enforced by variant definition).
    if (variant == 1 && n_gpus != 2) {
        fprintf(stderr,
                "V1 is a 2-GPU control; --n-gpus %d ignored, using 2\n", n_gpus);
        n_gpus = 2;
    }

    int device_count = 0;
    CHECK(hipGetDeviceCount(&device_count));
    if (n_gpus > device_count) {
        fprintf(stderr, "requested --n-gpus %d > available %d\n",
                n_gpus, device_count);
        return 1;
    }

    // Startup config event.
    printf("{\"event\":\"config\",\"hw_detect\":\"%s\",\"rocm_detect\":\"%s\","
           "\"n_gpus\":%d,\"variant\":\"%s\",\"commit\":\"%s\"}\n",
           detect_hw(0).c_str(), detect_rocm_version().c_str(),
           n_gpus, variant_name(variant), detect_git_commit().c_str());
    fflush(stdout);

    // Per-GPU resources.
    struct PerGpu {
        WedgeReproQueue* host_ptr;
        WedgeReproQueue* dev_ptr;
        hipStream_t stream;
        uint32_t num_blocks;
    };
    std::vector<PerGpu> gpus(n_gpus);

    for (int g = 0; g < n_gpus; g++) {
        CHECK(hipSetDevice(g));
        void* host_ptr = nullptr;
        // hipHostMallocMapped: same allocation pattern as production
        // MappedHostBuffer::alloc (matches braidinfer commit 963cd76 mtype
        // probe: mem_type=1 alloc_flags=0x2). This is the V0 envelope.
        CHECK(hipHostMalloc(&host_ptr, sizeof(WedgeReproQueue),
                            hipHostMallocMapped));
        memset(host_ptr, 0, sizeof(WedgeReproQueue));
        void* dev_ptr = nullptr;
        CHECK(hipHostGetDevicePointer(&dev_ptr, host_ptr, 0));
        gpus[g].host_ptr = (WedgeReproQueue*)host_ptr;
        gpus[g].dev_ptr = (WedgeReproQueue*)dev_ptr;
        CHECK(hipStreamCreate(&gpus[g].stream));

        // Match production block sizing: blocks_per_sm * num_cus.
        // hipDeviceAttributeMultiprocessorCount = 63 on gfx1100.
        int num_cus = 0;
        CHECK(hipDeviceGetAttribute(&num_cus,
                                    hipDeviceAttributeMultiprocessorCount, g));
        int bpsm = max_active_blocks_per_sm(
            (const void*)persistent_worker_skeleton, 256, 0);
        if (bpsm < 1) bpsm = 1;
        if (bpsm > 2) bpsm = 2;
        gpus[g].num_blocks = (uint32_t)(bpsm * num_cus);
    }

    int wedge_count = 0;
    int complete_count = 0;
    for (int trial = 1; trial <= n_trials; trial++) {
        // Reset queues.
        for (int g = 0; g < n_gpus; g++) {
            memset(gpus[g].host_ptr, 0, sizeof(WedgeReproQueue));
        }

        // Launch cooperative kernel on each GPU.
        for (int g = 0; g < n_gpus; g++) {
            CHECK(hipSetDevice(g));
            void* dev_ptr = gpus[g].dev_ptr;
            int v = variant;
            void* args[2] = {&dev_ptr, &v};
            dim3 grid(gpus[g].num_blocks, 1, 1);
            dim3 block(256, 1, 1);
            CHECK(hipLaunchCooperativeKernel(
                (const void*)persistent_worker_skeleton,
                grid, block, args, 0, gpus[g].stream));
        }

        // Wait until all blocks of each kernel have called atomicAdd on
        // block_alive_count. Bounded wait: 5 seconds. Lets us distinguish
        // "kernel hadn't started yet when we wrote seq" from "kernel started,
        // worker block 0 is in poll, but didn't see seq."
        auto start_wait = std::chrono::steady_clock::now();
        bool all_alive = false;
        while (!all_alive) {
            all_alive = true;
            for (int g = 0; g < n_gpus; g++) {
                uint32_t bac = gpus[g].host_ptr->block_alive_count;
                if (bac < gpus[g].num_blocks) {
                    all_alive = false;
                    break;
                }
            }
            if (all_alive) break;
            auto el = std::chrono::steady_clock::now() - start_wait;
            if (std::chrono::duration_cast<std::chrono::seconds>(el).count() > 5) {
                fprintf(stderr,
                        "trial %d: timed out waiting for all blocks to start\n",
                        trial);
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // Fire seq=1 to each worker simultaneously.
        auto dispatch_t0 = std::chrono::steady_clock::now();
        for (int g = 0; g < n_gpus; g++) {
            // Plain volatile store via host pointer. Same pattern as
            // crates/braidinfer-runtime/src/persistent_dispatch.rs
            // dispatch_batch_fire (write_volatile).
            __atomic_store_n(&gpus[g].host_ptr->seq_num, 1u, __ATOMIC_RELEASE);
        }

        // Spin-poll each worker's ack with 30s timeout. Match production
        // try_wait_ack semantics (persistent_dispatch.rs:493).
        std::vector<bool> done(n_gpus, false);
        std::vector<uint32_t> ack_seen(n_gpus, 0);
        bool wedged = false;
        int wedge_gpu = -1;
        auto trial_start = std::chrono::steady_clock::now();
        while (true) {
            int remaining = 0;
            for (int g = 0; g < n_gpus; g++) {
                if (done[g]) continue;
                uint32_t a = gpus[g].host_ptr->ack;
                if (a == 1u) {
                    done[g] = true;
                    ack_seen[g] = a;
                } else {
                    remaining++;
                }
            }
            if (remaining == 0) break;
            auto el = std::chrono::steady_clock::now() - trial_start;
            if (std::chrono::duration_cast<std::chrono::seconds>(el).count() > 30) {
                wedged = true;
                for (int g = 0; g < n_gpus; g++) {
                    if (!done[g]) { wedge_gpu = g; break; }
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        // Sample wedge signature on the first stuck GPU (or GPU 0 on success).
        int sig_gpu = (wedge_gpu >= 0) ? wedge_gpu : 0;
        uint32_t seq = gpus[sig_gpu].host_ptr->seq_num;
        uint32_t ack = gpus[sig_gpu].host_ptr->ack;
        uint32_t progress_pc = gpus[sig_gpu].host_ptr->progress_pc;
        uint32_t block_alive_count = gpus[sig_gpu].host_ptr->block_alive_count;
        uint32_t completed = gpus[sig_gpu].host_ptr->completed_dispatches;
        auto trial_end = std::chrono::steady_clock::now();
        uint64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            trial_end - dispatch_t0).count();

        printf("{\"variant\":\"%s\",\"trial\":%d,\"wedged\":%s,"
               "\"wedge_signature\":{\"seq\":%u,\"ack\":%u,"
               "\"progress_pc\":\"0x%08x\",\"block_alive_count\":%u,"
               "\"gpu_id\":%d,\"elapsed_ms\":%llu},"
               "\"completed_dispatches\":%u}\n",
               variant_name(variant), trial,
               wedged ? "true" : "false",
               seq, ack, progress_pc, block_alive_count,
               sig_gpu, (unsigned long long)elapsed_ms, completed);
        fflush(stdout);
        if (wedged) wedge_count++; else complete_count++;

        // Signal shutdown to all workers so they exit cleanly before next
        // trial / process exit. Workers see shutdown=1, write ack=0xFFFFFFFF,
        // return. Then stream sync drains.
        for (int g = 0; g < n_gpus; g++) {
            __atomic_store_n(&gpus[g].host_ptr->shutdown, 1u, __ATOMIC_RELEASE);
        }
        // Best-effort sync. If the kernel is wedged, sync hangs — accept
        // that for now; the outer wrapper script kills the process on
        // launch timeout.
        if (!wedged) {
            for (int g = 0; g < n_gpus; g++) {
                CHECK(hipSetDevice(g));
                CHECK(hipStreamSynchronize(gpus[g].stream));
            }
        }
    }

    return 0;
}
