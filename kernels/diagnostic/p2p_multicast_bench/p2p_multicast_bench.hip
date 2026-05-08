// PCIe multicast probe: does GPU 0 writing the same data to N worker GPUs
// scale linearly (unicast) or stay flat (multicast)?
//
// Standard PCIe transactions are unicast TLPs — each write goes to one
// endpoint. PCIe spec §6.14 defines Multicast Capability but AMD consumer
// GPUs (gfx1100) and the amdgpu driver typically do NOT expose it. This
// bench EMPIRICALLY checks: do 4 concurrent GPU 0 → worker[i] writes
// complete in ~1× single-write time (hardware multicast) or ~4× (unicast)?
//
// Methodology: launch N concurrent kernels, each on its own stream, each
// reading the same data on GPU 0 and writing to a different worker's local
// VRAM via P2P. The kernel runs on the SOURCE (GPU 0) for write, OR on the
// DESTINATION for read — both modeled to factor source vs destination
// bottleneck.
//
// Topology assumed: 4 GPUs all on PCIe 4.0 x16 root complex (no xGMI,
// no PCIe switches). See bd memory pcie-topology-all-4-gpus-03-00-0.
//
// Build: ./build_bench.sh
// Run:   python3 scripts/launch-gpu.py -g 4 --timeout 600 -- \
//          kernels/diagnostic/p2p_multicast_bench/p2p_multicast_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// Write-from-source kernel: runs on GPU 0, reads from local VRAM `src`,
// writes to peer VRAM `dst` via P2P. P2P writes traverse PCIe.
extern "C" __global__ __launch_bounds__(256, 2) void p2p_write_from_src(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int n_vec4
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n_vec4; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    int n_devices = 0;
    CHECK(hipGetDeviceCount(&n_devices));
    if (n_devices < 4) {
        fprintf(stderr, "this bench wants 4 GPUs; found %d\n", n_devices);
        return 1;
    }

    int src_gpu = 0;
    int n_workers = 4; // workers 0..3, but the "0" case is GPU 0 writing to its own VRAM (local store, no PCIe)

    // Enable peer access from GPU 0 (writer) to each worker GPU.
    CHECK(hipSetDevice(src_gpu));
    for (int w = 0; w < n_workers; w++) {
        if (w == src_gpu) continue;
        hipError_t e = hipDeviceEnablePeerAccess(w, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "peer enable %d->%d failed: %s\n",
                    src_gpu, w, hipGetErrorString(e));
            return 1;
        }
    }

    const size_t sizes[] = { 64 * 1024, 1 * 1024 * 1024,
                              16 * 1024 * 1024, 64 * 1024 * 1024 };
    const char* size_names[] = { "64KB", "1MB", "16MB", "64MB" };
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t max_bytes = sizes[n_sizes - 1];

    // Source buffer on GPU 0.
    CHECK(hipSetDevice(src_gpu));
    uint4* src_dev = nullptr;
    CHECK(hipMalloc((void**)&src_dev, max_bytes));
    // Initialize via host upload.
    {
        uint32_t* tmp = (uint32_t*)std::malloc(max_bytes);
        for (size_t i = 0; i < max_bytes / 4; i++) tmp[i] = (uint32_t)(i * 0x9E3779B1u);
        CHECK(hipMemcpy(src_dev, tmp, max_bytes, hipMemcpyHostToDevice));
        std::free(tmp);
    }

    // Destination buffers, one per worker GPU.
    std::vector<uint4*> dst_dev(n_workers, nullptr);
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipSetDevice(w));
        CHECK(hipMalloc((void**)&dst_dev[w], max_bytes));
    }

    // Streams + events on GPU 0 (writer).
    CHECK(hipSetDevice(src_gpu));
    std::vector<hipStream_t> streams(n_workers);
    std::vector<hipEvent_t> ev_start(n_workers), ev_stop(n_workers);
    for (int w = 0; w < n_workers; w++) {
        CHECK(hipStreamCreate(&streams[w]));
        CHECK(hipEventCreate(&ev_start[w]));
        CHECK(hipEventCreate(&ev_stop[w]));
    }

    auto bench_concurrent = [&](int n_concurrent, size_t bytes,
                                const char* label) {
        int n_vec4 = (int)(bytes / 16);
        int blocks = std::min(96, (n_vec4 + 255) / 256);
        // Warmup
        for (int w = 0; w < n_concurrent; w++) {
            p2p_write_from_src<<<blocks, 256, 0, streams[w]>>>(
                src_dev, dst_dev[w], n_vec4);
        }
        for (int w = 0; w < n_concurrent; w++) {
            CHECK(hipStreamSynchronize(streams[w]));
        }

        // Measure (median of 5 trials)
        std::vector<double> wall_samples;
        std::vector<std::vector<double>> per_w_samples(n_concurrent);
        for (int it = 0; it < 5; it++) {
            // Record start on each stream (events submitted to streams).
            for (int w = 0; w < n_concurrent; w++) {
                CHECK(hipEventRecord(ev_start[w], streams[w]));
            }
            // Launch all writes.
            for (int w = 0; w < n_concurrent; w++) {
                p2p_write_from_src<<<blocks, 256, 0, streams[w]>>>(
                    src_dev, dst_dev[w], n_vec4);
            }
            // Record stop on each stream.
            for (int w = 0; w < n_concurrent; w++) {
                CHECK(hipEventRecord(ev_stop[w], streams[w]));
            }
            // Synchronize all.
            for (int w = 0; w < n_concurrent; w++) {
                CHECK(hipStreamSynchronize(streams[w]));
            }
            // Per-stream elapsed.
            double max_ms = 0.0;
            for (int w = 0; w < n_concurrent; w++) {
                float ms = 0.0f;
                CHECK(hipEventElapsedTime(&ms, ev_start[w], ev_stop[w]));
                per_w_samples[w].push_back(ms);
                if (ms > max_ms) max_ms = ms;
            }
            wall_samples.push_back(max_ms);
        }
        std::sort(wall_samples.begin(), wall_samples.end());
        double wall_ms = wall_samples[2];
        // Per-worker median.
        std::vector<double> per_w_med(n_concurrent);
        for (int w = 0; w < n_concurrent; w++) {
            std::sort(per_w_samples[w].begin(), per_w_samples[w].end());
            per_w_med[w] = per_w_samples[w][2];
        }
        double bytes_total = (double)bytes * n_concurrent;
        double agg_gbs = bytes_total / 1e9 / (wall_ms / 1000.0);
        // For unicast: agg_gbs ≈ single_gbs (saturated link).
        // For multicast: agg_gbs ≈ N * single_gbs (data fan-out for free).
        double per_w_gbs = (double)bytes / 1e9 / (per_w_med[0] / 1000.0);
        printf("%-12s  N=%d  size=%-6s  wall=%6.3f ms  agg=%7.2f GB/s  "
               "per-stream=%6.2f GB/s\n",
               label, n_concurrent, "...", wall_ms, agg_gbs, per_w_gbs);
        // Print per-stream times to expose any obvious non-uniformity.
        printf("              per-stream ms: ");
        for (int w = 0; w < n_concurrent; w++) printf("%5.2f ", per_w_med[w]);
        printf("\n");
    };

    printf("\n=== GPU 0 P2P-writes to N concurrent worker GPUs ===\n");
    printf("Unicast hypothesis: N targets => wall time ~ N * single-target time\n");
    printf("Multicast hypothesis: N targets => wall time ~ single-target time\n\n");
    for (int s = 0; s < n_sizes; s++) {
        printf("--- %s ---\n", size_names[s]);
        // 1 target = GPU 0 → GPU 1
        bench_concurrent(1, sizes[s], "1-target");
        // 2 targets concurrent
        bench_concurrent(2, sizes[s], "2-target");
        // 3 targets concurrent
        bench_concurrent(3, sizes[s], "3-target");
        // 4 targets concurrent (including GPU 0 → GPU 0)
        bench_concurrent(4, sizes[s], "4-target");
        printf("\n");
    }

    return 0;
}
