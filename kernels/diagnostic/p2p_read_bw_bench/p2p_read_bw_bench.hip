// 77r.2.9 + 77r.2.12: aggregate read bandwidth from N workers reading the same
// source buffer concurrently. Two sources:
//   GPU0_UC_VRAM:  hipExtMallocWithFlags(... hipDeviceMallocUncached) on GPU 0
//   HOST_PORTABLE: hipHostMalloc(... hipHostMallocMapped|hipHostMallocPortable)
//
// Each worker GPU reads `bytes` bytes via 256-thread blocks doing vec4 loads.
// We launch all N worker kernels with hipEventRecord so they execute
// concurrently on their own streams. Aggregate bandwidth is sum of
// per-worker bandwidth.
//
// Build:  ./build_bench.sh
// Run:    python3 scripts/launch-gpu.py -g 4 --timeout 600 -- \
//           kernels/diagnostic/p2p_read_bw_bench/p2p_read_bw_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// Reader kernel: each thread reads a stride of vec4 (16 bytes). Returns the
// XOR fold to a single output (defeats DCE without slowing the read).
extern "C" __global__ __launch_bounds__(256, 2) void read_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ sink,
    int n_vec4
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < n_vec4; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (tid == 0) sink[0] = acc;
}

struct WorkerCtx {
    int gpu_id;
    hipStream_t stream;
    uint4* sink_dev;       // single uint4 on the worker
    hipEvent_t start, stop;
};

double run_concurrent_read(
    int n_workers,
    const uint4* src_dev_ptr,   // device pointer valid on each worker (portable)
    size_t bytes,
    std::vector<WorkerCtx>& workers
) {
    int n_vec4 = (int)(bytes / 16);
    int blocks = std::min(96, (n_vec4 + 255) / 256);

    // Record start events.
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(workers[w].gpu_id));
        CHECK(hipEventRecord(workers[w].start, workers[w].stream));
    }
    // Launch all worker reads.
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(workers[w].gpu_id));
        read_kernel<<<blocks, 256, 0, workers[w].stream>>>(
            src_dev_ptr, workers[w].sink_dev, n_vec4);
    }
    // Record stop events.
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(workers[w].gpu_id));
        CHECK(hipEventRecord(workers[w].stop, workers[w].stream));
    }
    // Sync.
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(workers[w].gpu_id));
        CHECK(hipStreamSynchronize(workers[w].stream));
    }
    // Per-worker times; aggregate bandwidth = sum(bytes / time_w).
    double agg_gbs = 0.0;
    double max_ms = 0.0;
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(workers[w].gpu_id));
        float ms = 0.0f;
        CHECK(hipEventElapsedTime(&ms, workers[w].start, workers[w].stop));
        double gbs = (double)bytes / 1e9 / ((double)ms / 1000.0);
        agg_gbs += gbs;
        if (ms > max_ms) max_ms = ms;
    }
    // Wall-clock aggregate (should equal max single-worker time):
    double wall_gbs = (double)bytes * n_workers / 1e9 / (max_ms / 1000.0);
    (void)wall_gbs;
    return agg_gbs;
}

int main() {
    int n_devices = 0;
    CHECK(hipGetDeviceCount(&n_devices));
    if (n_devices < 4) {
        fprintf(stderr, "this bench wants 4 GPUs; found %d\n", n_devices);
        return 1;
    }

    // Source GPU = device 0 (the "GPU 0" in production).
    int src_gpu = 0;
    int n_total_workers = 4;

    // Enable peer access from each worker (1..3) to source (0). Workers always
    // read from GPU 0 in the production design.
    for (int w = 1; w < n_total_workers; w++) {
        CHECK(hipSetDevice(w));
        hipError_t e = hipDeviceEnablePeerAccess(src_gpu, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "peer enable failed %d->%d: %s\n", w, src_gpu, hipGetErrorString(e));
            return 1;
        }
    }
    // Also enable GPU 0 → GPU 0 (no-op) and itself reading: just include device 0
    // in the worker pool for the "all 4 read" case (treat GPU 0 reading its own
    // local memory as one of the workers — local read, no PCIe).
    CHECK(hipSetDevice(src_gpu));

    // Allocate context per worker.
    std::vector<WorkerCtx> workers(n_total_workers);
    for (int w = 0; w < n_total_workers; w++) {
        workers[w].gpu_id = w;
        CHECK(hipSetDevice(w));
        CHECK(hipStreamCreate(&workers[w].stream));
        CHECK(hipEventCreate(&workers[w].start));
        CHECK(hipEventCreate(&workers[w].stop));
        CHECK(hipMalloc((void**)&workers[w].sink_dev, sizeof(uint4)));
    }

    const size_t sizes[] = { 4 * 1024, 64 * 1024, 1 * 1024 * 1024,
                              4 * 1024 * 1024, 64 * 1024 * 1024 };
    const char* size_names[] = { "4KB", "64KB", "1MB", "4MB", "64MB" };
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // ---- Allocate the two source buffers ----
    // (1) UC VRAM on GPU 0.
    CHECK(hipSetDevice(src_gpu));
    void* uc_vram = nullptr;
    CHECK(hipExtMallocWithFlags(&uc_vram, sizes[n_sizes - 1], hipDeviceMallocUncached));
    // Initialize to nonzero pattern.
    uint32_t* uc_init = (uint32_t*)std::malloc(sizes[n_sizes - 1]);
    for (size_t i = 0; i < sizes[n_sizes - 1] / 4; i++) uc_init[i] = (uint32_t)(i * 0x9E3779B1u);
    CHECK(hipMemcpy(uc_vram, uc_init, sizes[n_sizes - 1], hipMemcpyHostToDevice));
    std::free(uc_init);

    // (2) Portable host-mapped.
    void* host_portable = nullptr;
    CHECK(hipHostMalloc(&host_portable, sizes[n_sizes - 1],
                        hipHostMallocMapped | hipHostMallocPortable));
    uint32_t* hp_words = (uint32_t*)host_portable;
    for (size_t i = 0; i < sizes[n_sizes - 1] / 4; i++) hp_words[i] = (uint32_t)(i * 0x9E3779B1u);

    // For each source, get device pointer per-worker (portable mapping returns
    // a single device pointer valid from all GPUs; UC-on-GPU0 needs P2P access
    // which we already enabled).
    void* uc_dev_ptr = uc_vram; // same device address, P2P resolves it
    void* hp_dev_ptr = nullptr;
    CHECK(hipHostGetDevicePointer(&hp_dev_ptr, host_portable, 0));

    // ---- Bench loop ----
    auto bench = [&](const char* label, void* src_ptr, int n_workers,
                     std::vector<int>& worker_ids) {
        // Build a temporary worker context vector with only the chosen IDs.
        std::vector<WorkerCtx> chosen;
        for (int id : worker_ids) chosen.push_back(workers[id]);

        for (int s = 0; s < n_sizes; s++) {
            // Warmup
            run_concurrent_read(n_workers, (const uint4*)src_ptr, sizes[s], chosen);
            // Measure (median of 5)
            std::vector<double> samples;
            for (int it = 0; it < 5; it++) {
                samples.push_back(run_concurrent_read(n_workers, (const uint4*)src_ptr,
                                                       sizes[s], chosen));
            }
            std::sort(samples.begin(), samples.end());
            double agg_gbs = samples[2];
            printf("%-12s %5s  workers=%d  aggregate=%7.2f GB/s   (per-worker avg=%6.2f GB/s)\n",
                   label, size_names[s], n_workers, agg_gbs, agg_gbs / n_workers);
        }
    };

    printf("\n=== Source: GPU 0 UC VRAM (P2P reads from workers) ===\n");
    // GPU 0 reading its OWN UC VRAM (local UC read, no PCIe — for reference).
    {
        std::vector<int> ids = {0};
        bench("UC_VRAM(loc)", uc_dev_ptr, 1, ids);
    }
    // 1 worker (GPU 1) P2P-reading GPU 0 UC.
    {
        std::vector<int> ids = {1};
        bench("UC_VRAM(p2p)", uc_dev_ptr, 1, ids);
    }
    // 2 workers (GPU 1, 2).
    {
        std::vector<int> ids = {1, 2};
        bench("UC_VRAM(p2p)", uc_dev_ptr, 2, ids);
    }
    // 3 workers (1, 2, 3).
    {
        std::vector<int> ids = {1, 2, 3};
        bench("UC_VRAM(p2p)", uc_dev_ptr, 3, ids);
    }
    // 4 workers (0, 1, 2, 3) — including GPU 0's local UC read.
    {
        std::vector<int> ids = {0, 1, 2, 3};
        bench("UC_VRAM(p2p)", uc_dev_ptr, 4, ids);
    }

    printf("\n=== Source: portable host-mapped RAM ===\n");
    {
        std::vector<int> ids = {0};
        bench("HOST_PORTABLE", hp_dev_ptr, 1, ids);
    }
    {
        std::vector<int> ids = {1};
        bench("HOST_PORTABLE", hp_dev_ptr, 1, ids);
    }
    {
        std::vector<int> ids = {1, 2};
        bench("HOST_PORTABLE", hp_dev_ptr, 2, ids);
    }
    {
        std::vector<int> ids = {1, 2, 3};
        bench("HOST_PORTABLE", hp_dev_ptr, 3, ids);
    }
    {
        std::vector<int> ids = {0, 1, 2, 3};
        bench("HOST_PORTABLE", hp_dev_ptr, 4, ids);
    }

    return 0;
}
