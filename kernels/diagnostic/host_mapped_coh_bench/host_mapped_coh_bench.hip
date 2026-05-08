// 77r.2.2: Test whether MappedHostBuffer::alloc_portable
// (hipHostMallocMapped|hipHostMallocPortable) suffers cross-GPU L2 staleness
// on gfx1100 when read repeatedly from a worker GPU.
//
// Methodology: GPU 0 allocates portable host-mapped memory. CPU writes pattern
// A → GPU 1 reads via portable device_ptr → CPU writes pattern B (no
// invalidation, no fence) → GPU 1 reads again. Validate: does GPU 1 see A or
// B on the second read? With MTYPE_NC, GPU 1's L2 may cache A and return it
// for the second read despite the new CPU write.
//
// Two variants per buffer size:
//   plain      — kernel does not invalidate L1 on entry
//   gl1_inv    — kernel calls buffer_gl1_inv at entry
//
// Buffer sizes: 1KB (single line), 64KB (multi-line), 64MB (>L2). The size
// matters because L2 fill state and replacement determine whether the second
// read hits a stale line. Run multiple iterations to see timing trend.
//
// Build: ./build_bench.sh
// Run:   python3 scripts/launch-gpu.py -g 2 --timeout 600 -- \
//          kernels/diagnostic/host_mapped_coh_bench/host_mapped_coh_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
#include <chrono>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// Worker reads `n_words` 32-bit words from `src` (a portable host-mapped device
// pointer) and writes the additive checksum to `out`. Uses additive sum so the
// result is NOT trivially zero by XOR identity (XOR of base^i for i in 0..256
// collapses to 0 because 256 base copies XOR to 0 and XOR of 0..255 = 0).
extern "C" __global__ __launch_bounds__(256) void worker_read_sum(
    const uint32_t* __restrict__ src,
    uint64_t* __restrict__ out,
    int n_words,
    int do_inv
) {
    if (do_inv) {
        // buffer_gl1_inv invalidates per-WGP L1 cache.
        asm volatile("buffer_gl1_inv" ::: "memory");
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    uint64_t acc = 0;
    for (int i = tid; i < n_words; i += stride) {
        acc += (uint64_t)src[i];
    }
    // Block-reduce via shared memory + atomic to single output.
    __shared__ uint64_t s_acc[256];
    s_acc[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_acc[threadIdx.x] += s_acc[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)out, (unsigned long long)s_acc[0]);
}

uint64_t expected_sum(const uint32_t* p, int n) {
    uint64_t a = 0;
    for (int i = 0; i < n; i++) a += (uint64_t)p[i];
    return a;
}

void fill_pattern(uint32_t* p, int n, uint32_t base) {
    for (int i = 0; i < n; i++) p[i] = base + (uint32_t)i;
}

struct Trial {
    size_t bytes;
    int do_inv;            // 0 = plain, 1 = buffer_gl1_inv
    bool first_read_ok;    // pattern A correctly read
    bool second_read_ok;   // pattern B correctly read (THIS is the staleness check)
    uint64_t observed_b;   // checksum returned for pattern B
    uint64_t expected_b;
};

int main() {
    int n_devices = 0;
    CHECK(hipGetDeviceCount(&n_devices));
    if (n_devices < 2) {
        fprintf(stderr, "need at least 2 GPUs; found %d\n", n_devices);
        return 1;
    }

    // Enable peer access from GPU 1 to anything host-mapped from GPU 0
    // (portable mapping is implicitly visible from all devices).
    CHECK(hipSetDevice(0));
    int can_access = 0;
    hipDeviceCanAccessPeer(&can_access, 1, 0);
    if (can_access) {
        CHECK(hipSetDevice(1));
        // hipDeviceEnablePeerAccess returns hipErrorPeerAccessAlreadyEnabled
        // if already on; ignore that.
        hipError_t e = hipDeviceEnablePeerAccess(0, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "peer enable failed: %s\n", hipGetErrorString(e));
            return 1;
        }
    }

    const size_t sizes[] = { 1024, 64 * 1024, 64 * 1024 * 1024 };
    const char* size_names[] = { "1KB", "64KB", "64MB" };

    printf("Host-mapped portable memory cross-GPU coherence test\n");
    printf("Reader: GPU 1, Writer: CPU.\n\n");
    printf("%-8s %-8s %-12s %-12s %-20s %-20s\n",
           "size", "variant", "first_read", "second_read", "observed_B", "expected_B");

    for (int s = 0; s < 3; s++) {
        size_t bytes = sizes[s];
        int n_words = (int)(bytes / 4);

        for (int do_inv = 0; do_inv < 2; do_inv++) {
            // Allocate portable host-mapped memory from GPU 0's context.
            CHECK(hipSetDevice(0));
            uint32_t* host_ptr = nullptr;
            CHECK(hipHostMalloc((void**)&host_ptr, bytes,
                                hipHostMallocMapped | hipHostMallocPortable));

            // Switch to GPU 1 to compute device-side pointer in its context.
            CHECK(hipSetDevice(1));
            uint32_t* dev_ptr_on_gpu1 = nullptr;
            CHECK(hipHostGetDevicePointer((void**)&dev_ptr_on_gpu1, host_ptr, 0));

            // Output buffer on GPU 1 (u64 for additive sum).
            uint64_t* out_dev = nullptr;
            CHECK(hipMalloc(&out_dev, sizeof(uint64_t)));

            // ---- Pattern A ----
            fill_pattern(host_ptr, n_words, 0xAAAAAAAAu);
            uint64_t exp_a = expected_sum(host_ptr, n_words);
            CHECK(hipMemset(out_dev, 0, sizeof(uint64_t)));

            int blocks = std::min(96, (n_words + 255) / 256);
            worker_read_sum<<<blocks, 256>>>(dev_ptr_on_gpu1, out_dev, n_words, do_inv);
            CHECK(hipDeviceSynchronize());
            uint64_t got_a = 0;
            CHECK(hipMemcpy(&got_a, out_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));
            bool first_ok = (got_a == exp_a);

            // ---- Pattern B (no fence, no invalidation between writes) ----
            // Critical: write from CPU directly to host_ptr, no flush, no
            // signaling. Worker GPU's L2 may still hold pattern-A lines.
            fill_pattern(host_ptr, n_words, 0xBBBBBBBBu);
            uint64_t exp_b = expected_sum(host_ptr, n_words);
            CHECK(hipMemset(out_dev, 0, sizeof(uint64_t)));

            // Launch worker again. If `do_inv` is set, the kernel begins by
            // invalidating per-WGP L1.
            worker_read_sum<<<blocks, 256>>>(dev_ptr_on_gpu1, out_dev, n_words, do_inv);
            CHECK(hipDeviceSynchronize());
            uint64_t got_b = 0;
            CHECK(hipMemcpy(&got_b, out_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));
            bool second_ok = (got_b == exp_b);

            printf("%-8s %-8s %-12s %-12s %-20llu %-20llu\n",
                   size_names[s], do_inv ? "gl1_inv" : "plain",
                   first_ok ? "OK" : "STALE",
                   second_ok ? "OK" : "STALE",
                   (unsigned long long)got_b, (unsigned long long)exp_b);

            CHECK(hipFree(out_dev));
            CHECK(hipHostFree(host_ptr));
        }
    }

    return 0;
}
