// 77r.2.1: Empirically characterize hipMemcpy / hipMemcpyAsync behavior on
// GPU 0 while a persistent cooperative worker holds all CUs.
//
// The PERSISTENT_WORKER_ACTIVE assertion in crates/braidinfer-hip/src/lib.rs
// guards against a documented "DMA requires CUs" deadlock. This bench tests
// whether that's universally true or whether some paths (async + dedicated
// stream + pinned source) can complete via SDMA without contending for CUs.
//
// Methodology:
//   1. Launch a synthetic cooperative spinner kernel that holds the same
//      block-count as production persistent_worker (96 blocks * 256 threads
//      on RX 7900 XTX) and spins on a host-mapped stop flag.
//   2. From the host (after a small delay to ensure the spinner is running),
//      run each h2d candidate path on a separate POSIX thread with a hard
//      timeout. If the copy returns within the timeout, declare the path
//      "non-deadlocking" and record its completion time. Otherwise, set the
//      stop flag (kills the spinner), join the test thread, and declare
//      "DEADLOCKED".
//   3. Repeat for: hipMemcpy sync, hipMemcpyAsync default stream + sync,
//      hipMemcpyAsync dedicated stream + sync, hipMemcpyAsync dedicated
//      stream + pinned source. Buffer sizes: 4KB, 1MB, 64MB.
//
// Reference for assertion claim: CLAUDE.md "What Causes Hangs in Persistent Mode".
//
// Build: ./build_bench.sh
// Run:   python3 scripts/launch-gpu.py --timeout 600 -- \
//          kernels/diagnostic/dma_under_persistent_bench/dma_under_persistent_bench

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

namespace cg = cooperative_groups;

// Synthetic persistent spinner: 96 cooperative blocks * 256 threads, spins on
// a host-mapped stop flag. Mirrors production persistent_worker block layout.
extern "C" __global__ __launch_bounds__(256, 2) void spinner(volatile uint32_t* stop_flag) {
    cg::grid_group grid = cg::this_grid();
    while (true) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // poll without busy-stalling whole CU
        }
        if (*stop_flag) break;
        // Light CU work to ensure we aren't elided / yielded.
        // Use s_sleep to avoid drowning power but stay resident.
        __builtin_amdgcn_s_sleep(2);
    }
    grid.sync();
}

struct TestResult {
    const char* name;
    size_t bytes;
    bool deadlocked;
    double elapsed_ms;
};

// Run an h2d copy via `fn` on a dedicated thread with a `timeout_ms` cutoff.
// Returns the completion time, or sets `deadlocked=true` if the cutoff hit.
template<typename Fn>
TestResult run_with_timeout(const char* name, size_t bytes, Fn fn, int timeout_ms) {
    std::atomic<bool> done{false};
    auto t0 = std::chrono::steady_clock::now();
    std::thread worker([&]() {
        fn();
        done.store(true, std::memory_order_release);
    });

    auto deadline = t0 + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (done.load(std::memory_order_acquire)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    TestResult r{name, bytes, false, 0.0};
    if (done.load()) {
        worker.join();
        auto t1 = std::chrono::steady_clock::now();
        r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        // Deadlock — caller must set stop flag and let spinner exit.
        r.deadlocked = true;
        r.elapsed_ms = (double)timeout_ms;
        // Don't join — leak the thread. Process will exit anyway.
        worker.detach();
    }
    return r;
}

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, dev));
    printf("device: %s, multiProcessorCount=%d\n", props.name, props.multiProcessorCount);

    // Allocate stop_flag in host-mapped memory.
    uint32_t* stop_flag_host = nullptr;
    CHECK(hipHostMalloc((void**)&stop_flag_host, sizeof(uint32_t),
                        hipHostMallocMapped));
    *stop_flag_host = 0;
    uint32_t* stop_flag_dev = nullptr;
    CHECK(hipHostGetDevicePointer((void**)&stop_flag_dev, stop_flag_host, 0));

    // Allocate device buffers we'll DMA into. Largest first; reuse for smaller.
    constexpr size_t MAX_BYTES = 64 * 1024 * 1024;
    void* d_buf = nullptr;
    CHECK(hipMalloc(&d_buf, MAX_BYTES));

    // Host source buffers: regular malloc + pinned (hipHostMalloc).
    void* h_buf_pageable = std::malloc(MAX_BYTES);
    void* h_buf_pinned = nullptr;
    CHECK(hipHostMalloc(&h_buf_pinned, MAX_BYTES, hipHostMallocDefault));
    std::memset(h_buf_pageable, 0xAA, MAX_BYTES);
    std::memset(h_buf_pinned, 0xBB, MAX_BYTES);

    // Dedicated stream for "non-default-stream" tests.
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    auto run_phase = [&](size_t bytes, std::vector<TestResult>& results) {
        // Phase A: baseline (no spinner). Each path should complete in ms.
        // Phase B: with spinner running. We'll detect deadlock or completion.

        const int timeout_ms = 3000;

        // ---- Phase B: with spinner ----
        // Launch spinner via cooperative kernel API.
        *stop_flag_host = 0;
        void* spinner_args[] = { (void*)&stop_flag_dev };
        // Use 96 blocks * 256 threads (RX 7900 XTX cooperative ceiling for the
        // megakernel). hipModuleLaunchCooperativeKernel is for module-loaded
        // kernels; for HIP-compiled extern "C" we use hipLaunchCooperativeKernel.
        dim3 grid(96, 1, 1), block(256, 1, 1);
        CHECK(hipLaunchCooperativeKernel((void*)spinner, grid, block,
                                         spinner_args, 0, /*stream*/0));
        // Brief delay so the spinner is definitively resident.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto run = [&](const char* name, auto fn) {
            results.push_back(run_with_timeout(name, bytes, fn, timeout_ms));
        };

        // 1. Sync hipMemcpy (pageable source)
        run("sync_hipMemcpy_pageable", [&]() {
            hipMemcpy(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice);
        });
        // 2. hipMemcpyAsync default stream (pageable)
        run("async_default_pageable", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice, 0);
            hipStreamSynchronize(0);
        });
        // 3. hipMemcpyAsync dedicated stream (pageable)
        run("async_dedicated_pageable", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);
        });
        // 4. hipMemcpyAsync dedicated stream (pinned)
        run("async_dedicated_pinned", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pinned, bytes, hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);
        });

        // Stop spinner (whether tests succeeded or hit deadlock detection).
        *stop_flag_host = 1;
        // Wait for spinner to actually exit (it polls every s_sleep(2) cycles).
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // hipDeviceSynchronize will block until spinner exits cleanly.
        // If a test deadlocked, the API queue is already poisoned — try sync,
        // ignore errors so we can report partial results.
        hipDeviceSynchronize();
    };

    // Phase A baseline: no spinner.
    auto run_baseline = [&](size_t bytes, std::vector<TestResult>& results) {
        const int timeout_ms = 3000;
        auto run = [&](const char* name, auto fn) {
            results.push_back(run_with_timeout(name, bytes, fn, timeout_ms));
        };
        run("sync_hipMemcpy_pageable", [&]() {
            hipMemcpy(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice);
        });
        run("async_default_pageable", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice, 0);
            hipStreamSynchronize(0);
        });
        run("async_dedicated_pageable", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pageable, bytes, hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);
        });
        run("async_dedicated_pinned", [&]() {
            hipMemcpyAsync(d_buf, h_buf_pinned, bytes, hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);
        });
    };

    const size_t sizes[] = { 4 * 1024, 1 * 1024 * 1024, 64 * 1024 * 1024 };
    const char* size_names[] = { "4KB", "1MB", "64MB" };

    printf("\n=== BASELINE (no persistent spinner) ===\n");
    printf("%-30s %8s %12s %s\n", "path", "bytes", "elapsed_ms", "status");
    for (int s = 0; s < 3; s++) {
        std::vector<TestResult> results;
        run_baseline(sizes[s], results);
        for (auto& r : results) {
            printf("%-30s %8s %12.3f %s\n",
                   r.name, size_names[s], r.elapsed_ms,
                   r.deadlocked ? "DEADLOCK" : "ok");
        }
    }

    printf("\n=== WITH PERSISTENT SPINNER (96 blocks x 256 threads) ===\n");
    printf("%-30s %8s %12s %s\n", "path", "bytes", "elapsed_ms", "status");
    for (int s = 0; s < 3; s++) {
        std::vector<TestResult> results;
        run_phase(sizes[s], results);
        for (auto& r : results) {
            printf("%-30s %8s %12.3f %s\n",
                   r.name, size_names[s], r.elapsed_ms,
                   r.deadlocked ? "DEADLOCK" : "ok");
        }
    }

    hipFree(d_buf);
    std::free(h_buf_pageable);
    hipHostFree(h_buf_pinned);
    hipHostFree(stop_flag_host);
    hipStreamDestroy(stream);
    return 0;
}
