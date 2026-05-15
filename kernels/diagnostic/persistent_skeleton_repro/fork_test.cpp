// pky.2 fork-cost probe 2026-05-13: tests whether fork() without exec() can
// run a cooperative kernel in the child. If yes, measures the per-fork cost
// vs the ~337ms fork+exec cost. If no (child wedges or fails), the workaround
// requires full exec().
//
// Parent does HIP init once. Then forks N times. Each child:
//  - Inherits parent's address space via CoW (libraries already loaded)
//  - Has a fresh mm_struct → fresh PASID when it touches /dev/kfd
//  - Tries a cooperative kernel launch
//  - Reports wall time + result via exit code
// Parent waits, tallies success/wedge rates and times.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

namespace cg = cooperative_groups;

struct Queue {
    volatile uint32_t seq_num;
    volatile uint32_t shutdown;
    volatile uint32_t ack;
    volatile uint32_t block_alive_count;
    volatile uint32_t progress_pc;
    uint32_t _pad[11];
};

extern "C" __global__ __launch_bounds__(256, 2)
void mini_worker(volatile Queue* q) {
    if (threadIdx.x == 0) atomicAdd((unsigned int*)&q->block_alive_count, 1u);
    cg::grid_group grid = cg::this_grid();
    uint32_t last_seq = 0;
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (true) {
                if (q->shutdown) { q->ack = 0xFFFFFFFFu; return; }
                uint32_t s = q->seq_num;
                if (s > last_seq) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        grid.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            q->ack = q->seq_num;
            last_seq = q->seq_num;
        }
        grid.sync();
    }
}

#define CHK(x) { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "child[%d] HIP error %d at %s\n", getpid(), (int)e, #x); \
    _exit(2); } }

// One cooperative-kernel cycle in the child. Returns 0 = success, 1 = wedge.
int child_run() {
    // pky.2 2026-05-13: child re-initializes HIP for its fresh PASID. Without
    // this the child inherits parent's HIP runtime state (queue handles, etc.)
    // which reference parent's amdkfd kfd_process. Re-init forces ROCm to bind
    // to child's mm_struct.
    const char* skip_init = std::getenv("BRAIDINFER_SKIP_CHILD_HIPINIT");
    if (!skip_init) {
        // BRAIDINFER_FORK_INIT_MODE selects which init sequence:
        //   reset+init: hipDeviceReset then hipInit (crashes — frees parent mem)
        //   init_only: just hipInit + hipSetDevice (safer attempt)
        //   nothing (default with skip_init unset): just hipSetDevice
        const char* mode = std::getenv("BRAIDINFER_FORK_INIT_MODE");
        if (mode && std::string(mode) == "reset+init") {
            hipError_t r = hipDeviceReset();
            fprintf(stderr, "child[%d] hipDeviceReset = %d\n", getpid(), (int)r);
            r = hipInit(0);
            fprintf(stderr, "child[%d] hipInit = %d\n", getpid(), (int)r);
            r = hipSetDevice(0);
            fprintf(stderr, "child[%d] hipSetDevice = %d\n", getpid(), (int)r);
        } else {
            // Just hipInit + hipSetDevice — no reset, no attempt to free
            // parent's inherited allocations.
            hipError_t r = hipInit(0);
            fprintf(stderr, "child[%d] hipInit = %d\n", getpid(), (int)r);
            r = hipSetDevice(0);
            fprintf(stderr, "child[%d] hipSetDevice = %d\n", getpid(), (int)r);
        }
    }
    void* host_ptr = nullptr;
    CHK(hipHostMalloc(&host_ptr, sizeof(Queue), hipHostMallocMapped));
    memset(host_ptr, 0, sizeof(Queue));
    void* dev_ptr = nullptr;
    CHK(hipHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    Queue* q_host = (Queue*)host_ptr;
    Queue* q_dev  = (Queue*)dev_ptr;
    hipStream_t stream;
    CHK(hipStreamCreate(&stream));
    int num_cus = 0;
    CHK(hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, 0));
    int bpsm = 2;
    int num_blocks = bpsm * num_cus;
    void* args[1] = {&q_dev};
    if (hipLaunchCooperativeKernel((const void*)mini_worker,
                                   dim3(num_blocks,1,1), dim3(256,1,1),
                                   args, 0, stream) != hipSuccess) {
        return 1;
    }
    // wait for blocks to come alive (max 1s)
    auto t0 = std::chrono::steady_clock::now();
    while (q_host->block_alive_count < (uint32_t)num_blocks) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1))
            return 1;
    }
    __atomic_store_n(&q_host->seq_num, 1u, __ATOMIC_RELEASE);
    // wait for ack (10s timeout)
    auto disp_start = std::chrono::steady_clock::now();
    int result = 0;
    while (q_host->ack != 1) {
        if (std::chrono::steady_clock::now() - disp_start > std::chrono::seconds(10)) {
            result = 1; break;
        }
    }
    __atomic_store_n(&q_host->shutdown, 1u, __ATOMIC_RELEASE);
    if (result == 0) {
        // best-effort sync; if wedged just bail
        hipStreamSynchronize(stream);
    }
    hipHostFree(host_ptr);
    return result;
}

int main(int argc, char** argv) {
    int n_forks = (argc >= 2) ? atoi(argv[1]) : 5;

    // Parent: HIP init + warm up so library + GPU context are loaded.
    CHK(hipSetDevice(0));
    void* warm; CHK(hipMalloc(&warm, 16)); CHK(hipFree(warm));
    printf("parent: HIP warm. forking %d children sequentially.\n", n_forks);
    fflush(stdout);

    for (int i = 0; i < n_forks; i++) {
        auto fork_t0 = std::chrono::steady_clock::now();
        pid_t pid = fork();
        if (pid == 0) {
            // child: run cooperative kernel
            int rc = child_run();
            _exit(rc);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        auto fork_t1 = std::chrono::steady_clock::now();
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            fork_t1 - fork_t0).count();
        printf("fork %d: pid=%d exit=%d wall=%ldms result=%s\n",
               i, pid, exit_code, ms,
               exit_code == 0 ? "SUCCESS" : (exit_code == 1 ? "WEDGED" : "ERROR"));
        fflush(stdout);
    }
    return 0;
}
