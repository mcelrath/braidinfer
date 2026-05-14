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
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <linux/types.h>

// pky.2 Design D ioctl (per udi #117) — defined locally to avoid kernel-header
// dependency at build time. Must match include/uapi/linux/kfd_ioctl.h in the
// patched kernel.
struct kfd_ioctl_reset_cooperative_state_args {
    __u32 gpu_id;
    __u32 pad;
};
#define BRAIDINFER_AMDKFD_IOC_RESET_COOP \
    _IOWR('K', 0x28, struct kfd_ioctl_reset_cooperative_state_args)

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
    volatile uint32_t* peer_uc_target; // V7-only: GPU 0 UC slot for this worker
};
static_assert(sizeof(WedgeReproQueue) == 40, "WedgeReproQueue layout drift");

extern "C" __global__ void probe_noncoop_kernel(volatile uint32_t* scratch);

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
        case 7: return "V7";
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
    if (strcmp(s, "V7") == 0) return 7;
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
    if (variant < 0 || variant == 6 || variant > 7 || n_gpus < 1) {
        fprintf(stderr,
                "usage: %s --variant V0|V1|V2|V3|V4|V5|V7 --n-gpus N [--n-trials 1]\n",
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

    // V7 setup: allocate UC device buffer on GPU 0 (§5.5 Rule 1a — Canonical
    // device-resident UC for cross-GPU peer writes). Enable P2P from each
    // worker GPU to GPU 0. Stamp peer_uc_target into each worker's queue.
    // For other variants, peer_uc_target stays nullptr (kernel checks).
    uint32_t* uc_target_base = nullptr;
    if (variant == 7) {
        CHECK(hipSetDevice(0));
        // hipDeviceMallocUncached = 0x3.
        const unsigned int HIP_DEVICE_MALLOC_UNCACHED = 0x3u;
        void* p = nullptr;
        CHECK(hipExtMallocWithFlags(&p, sizeof(uint32_t) * n_gpus,
                                    HIP_DEVICE_MALLOC_UNCACHED));
        uc_target_base = (uint32_t*)p;
        // Initialize to known sentinel for diagnostic (workers overwrite).
        CHECK(hipMemset(uc_target_base, 0, sizeof(uint32_t) * n_gpus));

        // Enable P2P from every worker GPU (1..n_gpus-1) to GPU 0.
        // GPU 0 also writes through its own UC mapping (no P2P needed).
        for (int g = 1; g < n_gpus; g++) {
            CHECK(hipSetDevice(g));
            int can = 0;
            CHECK(hipDeviceCanAccessPeer(&can, g, 0));
            if (!can) {
                fprintf(stderr, "V7: GPU %d cannot access GPU 0 peer; skipping enable\n", g);
                continue;
            }
            hipError_t e = hipDeviceEnablePeerAccess(0, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr,
                        "V7: hipDeviceEnablePeerAccess GPU %d->0 failed: %s\n",
                        g, hipGetErrorString(e));
                return 1;
            }
        }
        // Stamp per-worker slot pointer into each queue.
        for (int g = 0; g < n_gpus; g++) {
            gpus[g].host_ptr->peer_uc_target = uc_target_base + g;
        }
    }

    int wedge_count = 0;
    int complete_count = 0;
    // pky.2 2026-05-14: BRAIDINFER_PERSISTENT_KERNEL=1 keeps ONE persistent
    // worker alive across all trials — each trial increments seq, worker
    // dispatches without exit/relaunch. Tests whether wedge fires within
    // a single kernel lifecycle or only on cooperative-grid RELAUNCH.
    bool persistent_mode = std::getenv("BRAIDINFER_PERSISTENT_KERNEL") != nullptr;

    // In persistent mode, queue alloc + kernel launch happens ONCE outside loop.
    if (persistent_mode) {
        for (int g = 0; g < n_gpus; g++) {
            memset(gpus[g].host_ptr, 0, sizeof(WedgeReproQueue));
            if (variant == 7 && uc_target_base != nullptr) {
                gpus[g].host_ptr->peer_uc_target = uc_target_base + g;
            }
        }
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
        // Wait for all blocks to come alive.
        auto pstart = std::chrono::steady_clock::now();
        bool all_alive = false;
        while (!all_alive) {
            all_alive = true;
            for (int g = 0; g < n_gpus; g++) {
                if (gpus[g].host_ptr->block_alive_count < gpus[g].num_blocks) {
                    all_alive = false; break;
                }
            }
            if (all_alive) break;
            if (std::chrono::steady_clock::now() - pstart > std::chrono::seconds(5)) {
                fprintf(stderr, "persistent mode: blocks didn't come alive\n");
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        fprintf(stderr, "persistent mode: ONE cooperative kernel launched, all blocks alive\n");
    }

    for (int trial = 1; trial <= n_trials; trial++) {
      if (!persistent_mode) {
        // Reset queues. memset clobbers peer_uc_target; re-stamp for V7.
        for (int g = 0; g < n_gpus; g++) {
            memset(gpus[g].host_ptr, 0, sizeof(WedgeReproQueue));
            if (variant == 7 && uc_target_base != nullptr) {
                gpus[g].host_ptr->peer_uc_target = uc_target_base + g;
            }
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
      }

      if (!persistent_mode) {
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
      }
        // In persistent_mode, expected_ack tracks the seq we're firing
        // (1, 2, 3, ...). In non-persistent, it's always 1.
        uint32_t fire_seq = persistent_mode ? (uint32_t)trial : 1u;

        // pky.2 probe 2026-05-13: BRAIDINFER_SKIP_DISPATCH_TRIALS=K means the
        // first K trials skip the seq=1 write entirely — kernel launches,
        // sees shutdown=1 right away, exits cleanly via skeleton's shutdown
        // path. Tests whether the wedge priming requires a completed dispatch
        // or just the cooperative-kernel launch+exit cycle.
        int skip_dispatch_trials = 0;
        if (const char* env = std::getenv("BRAIDINFER_SKIP_DISPATCH_TRIALS")) {
            skip_dispatch_trials = atoi(env);
        }
        bool skip_this_trial = (trial <= skip_dispatch_trials);

        auto dispatch_t0 = std::chrono::steady_clock::now();
        if (skip_this_trial) {
            // No seq=1; just shutdown=1 so kernel exits cleanly via the
            // shutdown path (no dispatch completion in this trial).
            for (int g = 0; g < n_gpus; g++) {
                __atomic_store_n(&gpus[g].host_ptr->shutdown, 1u,
                                 __ATOMIC_RELEASE);
            }
        } else {
            // Fire seq=fire_seq to each worker simultaneously.
            for (int g = 0; g < n_gpus; g++) {
                // Plain volatile store via host pointer. Same pattern as
                // crates/braidinfer-runtime/src/persistent_dispatch.rs
                // dispatch_batch_fire (write_volatile).
                __atomic_store_n(&gpus[g].host_ptr->seq_num, fire_seq,
                                 __ATOMIC_RELEASE);
            }
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
                if (a == fire_seq) {
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

        // Canonical wedge metric (udi bridge msg #13): seq_completed == false
        // means the worker didn't fully complete the dispatch even if it
        // observed seq and wrote ack. V7 wedge symptom: ack=1 visible to
        // host BUT completed_dispatches=0 BUT progress_pc=0x10000005 — kernel
        // got past the post-poll barrier and partway through the post-ack
        // sequence before vscnt-drain hang resolved (or didn't). The
        // existing wedged field stays as "host timed out waiting for ack"
        // for backward compat; seq_completed is the metric the aggregator
        // should bin on.
        bool seq_completed = (completed > 0);
        printf("{\"variant\":\"%s\",\"trial\":%d,\"wedged\":%s,"
               "\"seq_completed\":%s,"
               "\"wedge_signature\":{\"seq\":%u,\"ack\":%u,"
               "\"progress_pc\":\"0x%08x\",\"block_alive_count\":%u,"
               "\"gpu_id\":%d,\"elapsed_ms\":%llu},"
               "\"completed_dispatches\":%u}\n",
               variant_name(variant), trial,
               wedged ? "true" : "false",
               seq_completed ? "true" : "false",
               seq, ack, progress_pc, block_alive_count,
               sig_gpu, (unsigned long long)elapsed_ms, completed);
        fflush(stdout);
        if (wedged || !seq_completed) wedge_count++; else complete_count++;

        // In persistent_mode, the worker keeps running across trials. Only
        // send shutdown after the LAST trial. Otherwise we'd kill the
        // persistent worker after trial 1 and lose the test.
        bool last_trial = (trial == n_trials);
        if (!persistent_mode || last_trial) {
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

        // pky.2 probe 2026-05-13: BRAIDINFER_INTERLEAVE_NONCOOP=1 launches a
        // small non-cooperative kernel between trials. Tests whether any
        // non-cooperative-kernel work clears the priming state set by the
        // prior cooperative-kernel launch+exit cycle.
        // pky.2 probe 2026-05-13: BRAIDINFER_ROTATE_STREAM=1 destroys and
        // recreates the HIP stream between trials. If the priming state is
        // per-HQD (allocated by hipStreamCreate via kfd CREATE_QUEUE ioctl),
        // a fresh HQD should clear it and trial N+1 should succeed. If
        // state is per-PASID, rotation won't help. Discriminator probe per
        // user direction.
        // pky.2 Design D ioctl call (per udi msg #117): after each trial,
        // call AMDKFD_IOC_RESET_COOPERATIVE_STATE to ask MES to NOTIFY_TO_
        // UNMAP_PROCESSES then NOTIFY_WORK_ON_UNMAPPED_QUEUE. Should clear
        // the per-PASID MES SRAM state at -0x3dc and let the next trial's
        // cooperative kernel run cleanly.
        if (std::getenv("BRAIDINFER_RESET_COOP_STATE")) {
            // Open /dev/kfd directly. KFD allows multiple fds per process;
            // they share the kfd_process struct keyed on mm_struct so the
            // ioctl operates on the same process state HIP set up.
            int kfd_fd = open("/dev/kfd", O_RDWR);
            if (kfd_fd < 0) {
                fprintf(stderr, "trial %d: open /dev/kfd FAILED errno=%d\n",
                        trial, errno);
            } else {
                // Iterate KFD topology nodes; for each GPU node, issue ioctl.
                // The ioctl returns -EINVAL for nodes we have no PDD on, so
                // we'll just hit-or-miss. For single-GPU test only one will
                // succeed.
                int success_count = 0;
                DIR* d = opendir("/sys/class/kfd/kfd/topology/nodes");
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d)) != nullptr) {
                        if (e->d_name[0] == '.') continue;
                        char path[256];
                        snprintf(path, sizeof(path),
                                 "/sys/class/kfd/kfd/topology/nodes/%s/gpu_id",
                                 e->d_name);
                        FILE* fp = fopen(path, "r");
                        if (!fp) continue;
                        unsigned int gpu_id = 0;
                        if (fscanf(fp, "%u", &gpu_id) == 1 && gpu_id != 0) {
                            kfd_ioctl_reset_cooperative_state_args args = {};
                            args.gpu_id = gpu_id;
                            args.pad = 0;
                            int r = ioctl(kfd_fd,
                                          BRAIDINFER_AMDKFD_IOC_RESET_COOP,
                                          &args);
                            fprintf(stderr,
                                    "trial %d: ioctl RESET_COOP gpu_id=%u rc=%d errno=%d\n",
                                    trial, gpu_id, r, errno);
                            if (r == 0) success_count++;
                        }
                        fclose(fp);
                    }
                    closedir(d);
                }
                fprintf(stderr, "trial %d: RESET_COOP fired on %d gpu(s)\n",
                        trial, success_count);
                close(kfd_fd);
            }
        }
        if (std::getenv("BRAIDINFER_ROTATE_STREAM")) {
            for (int g = 0; g < n_gpus; g++) {
                CHECK(hipSetDevice(g));
                // Stream destroy waits for outstanding work. Worker should
                // have exited via shutdown above so this is fast.
                CHECK(hipStreamDestroy(gpus[g].stream));
            }
            // pky.2 2026-05-14: BRAIDINFER_ZERO_QUEUE_DELAY_MS lets us validate
            // udi Design A hypothesis from userspace. Sleep BETWEEN destroy and
            // recreate, giving MES scheduler tick time to notice "PASID has
            // zero process-created queues" and (per udi's f000ad50 trace) run
            // the clear at -0x3dc. If wedge clears: Design A path empirically
            // validated. If wedges: queue-destroy-without-explicit-clear-packet
            // path doesn't work; Design C (explicit MES packet) is required.
            int delay_ms = 0;
            if (const char* env = std::getenv("BRAIDINFER_ZERO_QUEUE_DELAY_MS")) {
                delay_ms = atoi(env);
            }
            if (delay_ms > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay_ms));
                fprintf(stderr, "trial %d: zero-queue delay %d ms\n",
                        trial, delay_ms);
            }
            for (int g = 0; g < n_gpus; g++) {
                CHECK(hipSetDevice(g));
                CHECK(hipStreamCreate(&gpus[g].stream));
            }
            fprintf(stderr, "trial %d: rotated stream (delay=%dms)\n",
                    trial, delay_ms);
        }
        if (std::getenv("BRAIDINFER_INTERLEAVE_NONCOOP")) {
            for (int g = 0; g < n_gpus; g++) {
                CHECK(hipSetDevice(g));
                uint32_t* scratch = nullptr;
                CHECK(hipMalloc(&scratch, sizeof(uint32_t)));
                CHECK(hipMemsetAsync(scratch, 0, sizeof(uint32_t),
                                     gpus[g].stream));
                hipLaunchKernelGGL(probe_noncoop_kernel,
                                   dim3(1, 1, 1), dim3(64, 1, 1),
                                   0, gpus[g].stream, scratch);
                CHECK(hipStreamSynchronize(gpus[g].stream));
                CHECK(hipFree(scratch));
            }
            fprintf(stderr, "trial %d: interleaved non-coop probe kernel done\n",
                    trial);
        }
    }

    return 0;
}
