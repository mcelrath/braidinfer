// 77r.2.10: Cross-GPU coherence of regular MTYPE_C VRAM with explicit flushes.
//
// Question: today moe_latent is regular cached DeviceBuffer on GPU 0 (MTYPE_C).
// Can workers P2P-read it correctly if GPU 0 issues a fence/invalidate before
// signaling? Or must it be UC?
//
// Test pattern:
//   Sequence per (size, fence_kind):
//     1. GPU 0 writes pattern A via kernel.
//     2. GPU 0 issues fence (none / __threadfence / __threadfence_system /
//        buffer_gl1_inv as applicable).
//     3. GPU 0 sets a host-mapped signal flag.
//     4. Host waits for signal.
//     5. Worker GPU 1 reads via P2P, computes additive sum.
//     6. Verify worker saw pattern A.
//     7. GPU 0 writes pattern B (no fence/signal).
//     8. Worker reads again. Verify pattern B (or detect staleness).
//
// Reference: kb gfx1100-cross-gpu-coherence-pattern-per-composable-kernel.
// On gfx1100, __threadfence_system compiles to NO-OP (no buffer_gl2_inv exists
// in the ISA). Expectation: MTYPE_C VRAM may be stale on workers without UC.
//
// Build: ./build_bench.sh
// Run:   python3 scripts/launch-gpu.py -g 2 --timeout 300 -- \
//          kernels/diagnostic/mtype_c_p2p_coh_bench/mtype_c_p2p_coh_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// Writer kernel: GPU 0 writes pattern[i] = base + i to dst[i]. Optional fence
// at end. Then sets signal flag (host-mapped UC).
template<int FENCE_KIND>
__global__ __launch_bounds__(256) void writer_kernel(
    uint32_t* __restrict__ dst,
    int n_words,
    uint32_t base,
    volatile uint32_t* __restrict__ signal
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n_words; i += stride) {
        dst[i] = base + (uint32_t)i;
    }
    // Fence per template parameter:
    // 0 = none, 1 = __threadfence, 2 = __threadfence_system, 3 = gl1_inv writeback
    if (FENCE_KIND == 1) __threadfence();
    if (FENCE_KIND == 2) __threadfence_system();
    if (FENCE_KIND == 3) {
        // RDNA3 has no gl2 inv; gl1_inv invalidates per-WGP L1.
        // Best-effort writeback: __threadfence forces L1 stores to L2.
        __threadfence();
        asm volatile("buffer_gl1_inv" ::: "memory");
    }
    if (tid == 0) {
        *signal = 1u;
    }
}

// Worker kernel: read source via P2P, compute additive sum.
extern "C" __global__ __launch_bounds__(256) void worker_sum(
    const uint32_t* __restrict__ src,
    uint64_t* __restrict__ out,
    int n_words,
    int do_inv
) {
    if (do_inv) asm volatile("buffer_gl1_inv" ::: "memory");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    uint64_t acc = 0;
    for (int i = tid; i < n_words; i += stride) {
        acc += (uint64_t)src[i];
    }
    __shared__ uint64_t s[256];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)out, (unsigned long long)s[0]);
}

uint64_t expected_sum(uint32_t base, int n) {
    // sum_{i=0..n-1} (base + i) = n*base + n*(n-1)/2
    return (uint64_t)n * base + (uint64_t)n * (n - 1) / 2;
}

int main() {
    int n_devices = 0;
    CHECK(hipGetDeviceCount(&n_devices));
    if (n_devices < 2) {
        fprintf(stderr, "need at least 2 GPUs; found %d\n", n_devices);
        return 1;
    }

    // GPU 0 (writer) - GPU 1 (reader, via P2P).
    int wgpu = 0, rgpu = 1;
    CHECK(hipSetDevice(wgpu));
    hipError_t e = hipDeviceEnablePeerAccess(rgpu, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "peer enable failed: %s\n", hipGetErrorString(e));
        return 1;
    }
    CHECK(hipSetDevice(rgpu));
    e = hipDeviceEnablePeerAccess(wgpu, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "peer enable failed: %s\n", hipGetErrorString(e));
        return 1;
    }

    const size_t sizes[] = { 1024, 64 * 1024, 64 * 1024 * 1024 };
    const char* size_names[] = { "1KB", "64KB", "64MB" };

    printf("MTYPE_C VRAM cross-GPU coherence: writer=GPU 0, reader=GPU 1\n\n");
    printf("%-8s %-18s %-12s %-12s %-22s %-22s\n",
           "size", "fence_kind", "first_read", "second_read",
           "got_b", "expected_b");

    const char* fence_labels[] = {
        "none",
        "__threadfence",
        "__threadfence_system",
        "fence+gl1_inv",
    };

    for (int s = 0; s < 3; s++) {
        size_t bytes = sizes[s];
        int n_words = (int)(bytes / 4);

        for (int fk = 0; fk < 4; fk++) {
            // Allocate MTYPE_C source on GPU 0.
            CHECK(hipSetDevice(wgpu));
            uint32_t* src_dev = nullptr;
            CHECK(hipMalloc((void**)&src_dev, bytes));
            // Host-mapped signal flag (UC by default for hipHostMallocMapped).
            uint32_t* signal_host = nullptr;
            CHECK(hipHostMalloc((void**)&signal_host, sizeof(uint32_t),
                                hipHostMallocMapped));
            *signal_host = 0;
            uint32_t* signal_dev_w = nullptr;
            CHECK(hipHostGetDevicePointer((void**)&signal_dev_w, signal_host, 0));

            // Output sum on GPU 1.
            CHECK(hipSetDevice(rgpu));
            uint64_t* out_dev = nullptr;
            CHECK(hipMalloc((void**)&out_dev, sizeof(uint64_t)));

            // ---- Pattern A ----
            CHECK(hipSetDevice(wgpu));
            *signal_host = 0;
            int blocks = std::min(96, (n_words + 255) / 256);
            uint32_t base_a = 0xAAAA0000u;
            switch (fk) {
                case 0: writer_kernel<0><<<blocks, 256>>>(src_dev, n_words, base_a, signal_dev_w); break;
                case 1: writer_kernel<1><<<blocks, 256>>>(src_dev, n_words, base_a, signal_dev_w); break;
                case 2: writer_kernel<2><<<blocks, 256>>>(src_dev, n_words, base_a, signal_dev_w); break;
                case 3: writer_kernel<3><<<blocks, 256>>>(src_dev, n_words, base_a, signal_dev_w); break;
            }
            CHECK(hipDeviceSynchronize());
            // Spin until host sees signal.
            while (__atomic_load_n(signal_host, __ATOMIC_ACQUIRE) == 0) {}

            // Worker reads.
            CHECK(hipSetDevice(rgpu));
            CHECK(hipMemset(out_dev, 0, sizeof(uint64_t)));
            // Get device pointer for src on GPU 1's context (P2P).
            uint32_t* src_on_gpu1 = src_dev; // P2P device pointer is universal across enabled peers
            int rblocks = std::min(96, (n_words + 255) / 256);
            // do_inv=1 to give the reader best chance — invalidate L1.
            worker_sum<<<rblocks, 256>>>(src_on_gpu1, out_dev, n_words, /*do_inv=*/1);
            CHECK(hipDeviceSynchronize());
            uint64_t got_a = 0;
            CHECK(hipMemcpy(&got_a, out_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));
            uint64_t exp_a = expected_sum(base_a, n_words);
            bool first_ok = (got_a == exp_a);

            // ---- Pattern B (no signal — caller must already-have-pointer) ----
            CHECK(hipSetDevice(wgpu));
            uint32_t base_b = 0xBBBB0000u;
            *signal_host = 0;
            switch (fk) {
                case 0: writer_kernel<0><<<blocks, 256>>>(src_dev, n_words, base_b, signal_dev_w); break;
                case 1: writer_kernel<1><<<blocks, 256>>>(src_dev, n_words, base_b, signal_dev_w); break;
                case 2: writer_kernel<2><<<blocks, 256>>>(src_dev, n_words, base_b, signal_dev_w); break;
                case 3: writer_kernel<3><<<blocks, 256>>>(src_dev, n_words, base_b, signal_dev_w); break;
            }
            CHECK(hipDeviceSynchronize());
            while (__atomic_load_n(signal_host, __ATOMIC_ACQUIRE) == 0) {}

            CHECK(hipSetDevice(rgpu));
            CHECK(hipMemset(out_dev, 0, sizeof(uint64_t)));
            worker_sum<<<rblocks, 256>>>(src_on_gpu1, out_dev, n_words, /*do_inv=*/1);
            CHECK(hipDeviceSynchronize());
            uint64_t got_b = 0;
            CHECK(hipMemcpy(&got_b, out_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));
            uint64_t exp_b = expected_sum(base_b, n_words);
            bool second_ok = (got_b == exp_b);

            printf("%-8s %-18s %-12s %-12s %-22llu %-22llu\n",
                   size_names[s], fence_labels[fk],
                   first_ok ? "OK" : "STALE",
                   second_ok ? "OK" : "STALE",
                   (unsigned long long)got_b, (unsigned long long)exp_b);

            CHECK(hipFree(out_dev));
            CHECK(hipSetDevice(wgpu));
            CHECK(hipFree(src_dev));
            CHECK(hipHostFree(signal_host));
        }
    }

    return 0;
}
