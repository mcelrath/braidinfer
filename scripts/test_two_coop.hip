// Test: can two cooperative kernels run simultaneously on separate GPUs,
// each using grid.sync()? This simulates the persistent_worker + lean_worker setup.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <unistd.h>

#define CHECK(x) do { hipError_t e = (x); if (e) { fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

// Kernel 1 (fat): polls host-mapped flag, dispatches via grid.sync
__global__ __launch_bounds__(256) void fat_kernel(volatile int* queue, volatile int* ack) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int last = 0;
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (*queue == last && *queue != -1) __builtin_amdgcn_s_sleep(1);
        }
        grid.sync();
        int seq = *queue;
        if (seq == -1) return;
        // Simple work
        grid.sync();
        last = seq;
        if (threadIdx.x == 0 && blockIdx.x == 0) *ack = seq;
    }
}

// Kernel 2 (lean): polls different flag, uses grid.sync
__global__ __launch_bounds__(256) void lean_kernel(volatile int* queue, volatile int* ack) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int last = 0;
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (*queue == last && *queue != -1) __builtin_amdgcn_s_sleep(1);
        }
        grid.sync();
        int seq = *queue;
        if (seq == -1) return;
        grid.sync();
        last = seq;
        if (threadIdx.x == 0 && blockIdx.x == 0) *ack = seq;
    }
}

int main() {
    int ndev;
    CHECK(hipGetDeviceCount(&ndev));
    if (ndev < 2) { printf("Need 2+ GPUs\n"); return 1; }

    // Alloc host-mapped queues
    int *q0, *q1, *a0, *a1;
    CHECK(hipHostMalloc(&q0, sizeof(int), hipHostMallocMapped));
    CHECK(hipHostMalloc(&q1, sizeof(int), hipHostMallocMapped));
    CHECK(hipHostMalloc(&a0, sizeof(int), hipHostMallocMapped));
    CHECK(hipHostMalloc(&a1, sizeof(int), hipHostMallocMapped));
    *q0 = *q1 = *a0 = *a1 = 0;

    // Launch fat kernel on GPU 0
    CHECK(hipSetDevice(0));
    hipStream_t s0;
    CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
    auto f0 = fat_kernel;
    void* args0[] = {&q0, &a0};
    CHECK(hipModuleLaunchCooperativeKernel(/* can't use this directly */
        // Use hipLaunchCooperativeKernel instead
        nullptr, 0,0,0, 0,0,0, 0, s0, nullptr));
    // Actually use hipLaunchCooperativeKernel
    hipLaunchCooperativeKernel((void*)fat_kernel, dim3(96), dim3(256), args0, 1024, s0);

    // Launch lean kernel on GPU 1
    CHECK(hipSetDevice(1));
    hipStream_t s1;
    CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    void* args1[] = {&q1, &a1};
    hipLaunchCooperativeKernel((void*)lean_kernel, dim3(96), dim3(256), args1, 1024, s1);

    usleep(100000); // let kernels start

    // Test: dispatch 3 rounds to each
    for (int i = 1; i <= 3; i++) {
        *q0 = i;
        *q1 = i;
        int t = 0;
        while (*a0 != i || *a1 != i) {
            usleep(100);
            if (++t > 50000) { printf("TIMEOUT round %d (a0=%d a1=%d)\n", i, *a0, *a1); goto done; }
        }
        printf("Round %d: both GPUs acked\n", i);
    }

done:
    *q0 = *q1 = -1;
    usleep(100000);
    printf("Done\n");
    return 0;
}
