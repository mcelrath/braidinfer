// Microbenchmark for synchronization primitives on gfx1100.
//
// Measures:
//   1. cg::grid_group::sync()        at gridDim ∈ {48, 96, 192, 384}
//   2. atomic_block_barrier (rdna3_sync.h) at the same gridDims
//   3. __threadfence_block        ("workgroup")  per-call cycles
//   4. __threadfence              ("agent")      per-call cycles
//   5. __threadfence_system       ("")           per-call cycles
//      (system-scope fence; on gfx1100 this is generally unsafe but it
//       still has a measurable issue cost which informs whether call
//       sites that *should* be agent-scope are silently paying system
//       cost)
//
// Cycle measurement: in-kernel wall_clock64() (s_memrealtime) ticks.
// Reported per-operation = (t_end - t_start) / iterations.
//
// Build:
//   ./build_bench.sh
//
// Run (always via launch-gpu.py):
//   python3 scripts/launch-gpu.py --timeout 300 -- \
//     /home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/rdna3_sync_bench/rdna3_sync_bench

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "../../rdna3/rdna3_barrier.h"

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

using namespace braidinfer::rdna3;

namespace cg = cooperative_groups;

// ============================================================================
// Fence cost benchmarks (single-block, single-wave; per-call cost)
// ============================================================================
//
// Each iteration performs: store-to-volatile-global, fence, load-back. This
// prevents the compiler from eliminating the fence (which it WILL do on a
// no-op loop). The store/load adds a fixed ~baseline cycle count; we
// characterize that with a "no fence" variant and subtract.

template<int ITER>
__global__ void bench_baseline_storeload(volatile uint32_t* sink, uint64_t* cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sink[0] = (uint32_t)i;
        uint32_t v = sink[0];
        if (v == 0xDEADBEEF) sink[1] = v; // never executes
    }
    uint64_t t1 = wall_clock64();
    cycles[0] = t1 - t0;
}

template<int ITER>
__global__ void bench_fence_block(volatile uint32_t* sink, uint64_t* cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sink[0] = (uint32_t)i;
        fence_block();
        uint32_t v = sink[0];
        if (v == 0xDEADBEEF) sink[1] = v;
    }
    uint64_t t1 = wall_clock64();
    cycles[0] = t1 - t0;
}

template<int ITER>
__global__ void bench_fence_device(volatile uint32_t* sink, uint64_t* cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sink[0] = (uint32_t)i;
        fence_device();
        uint32_t v = sink[0];
        if (v == 0xDEADBEEF) sink[1] = v;
    }
    uint64_t t1 = wall_clock64();
    cycles[0] = t1 - t0;
}

// System fence: VERY DANGEROUS on gfx1100 in spin-wait contexts. Single-block
// run with no spin-wait reader is OK as a microbenchmark.
template<int ITER>
__global__ void bench_fence_system(volatile uint32_t* sink, uint64_t* cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sink[0] = (uint32_t)i;
        fence_system_uc();
        uint32_t v = sink[0];
        if (v == 0xDEADBEEF) sink[1] = v;
    }
    uint64_t t1 = wall_clock64();
    cycles[0] = t1 - t0;
}

// All-threads variant: every thread executes the fence (more realistic for
// __threadfence() before a peer-visible store, where every thread participates).
template<int ITER>
__global__ __launch_bounds__(256) void bench_fence_device_allthreads(
    volatile uint32_t* sink, uint64_t* cycles)
{
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sink[threadIdx.x] = (uint32_t)i;
        fence_device();
        uint32_t v = sink[threadIdx.x];
        if (v == 0xDEADBEEF) sink[256 + threadIdx.x] = v;
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) cycles[0] = t1 - t0;
}

// ============================================================================
// Grid barrier benchmarks
// ============================================================================

// cg grid.sync() over ITER iterations.
template<int ITER>
__global__ __launch_bounds__(256) void bench_cg_grid_sync(uint64_t* per_block_cycles) {
    cg::grid_group grid = cg::this_grid();
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        grid.sync();
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0) per_block_cycles[blockIdx.x] = t1 - t0;
}

// atomic_block_barrier over ITER iterations.
template<int ITER>
__global__ __launch_bounds__(256) void bench_atomic_block_barrier(
    GridBarrierState* state, uint64_t* per_block_cycles)
{
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        atomic_block_barrier(state);
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0) per_block_cycles[blockIdx.x] = t1 - t0;
}

// ASM-tightened variant — exercises asm_block_barrier path
template<int ITER>
__global__ __launch_bounds__(256) void bench_asm_block_barrier(
    GridBarrierState* state, uint64_t* per_block_cycles)
{
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        asm_block_barrier(state);
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0) per_block_cycles[blockIdx.x] = t1 - t0;
}

// Two-state alternating barrier: avoids the reset path's data dependency
// on the counter (counter for state A is reset by last arriver while state B
// is being entered). A useful comparison.
template<int ITER>
__global__ __launch_bounds__(256) void bench_atomic_block_barrier_double(
    GridBarrierState* a, GridBarrierState* b, uint64_t* per_block_cycles)
{
    uint64_t t0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        atomic_block_barrier((i & 1) ? a : b);
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0) per_block_cycles[blockIdx.x] = t1 - t0;
}

// Coarse barrier: every Nth iteration only. Models "fuse N consecutive
// instructions" — measures whether the cost is per-barrier or per-grid-sweep.
template<int ITER, int EVERY>
__global__ __launch_bounds__(256) void bench_cg_grid_sync_every(uint64_t* per_block_cycles) {
    cg::grid_group grid = cg::this_grid();
    uint64_t t0 = wall_clock64();
    int counter = 0;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        counter++;
        if (counter == EVERY) { grid.sync(); counter = 0; }
    }
    uint64_t t1 = wall_clock64();
    if (threadIdx.x == 0) per_block_cycles[blockIdx.x] = t1 - t0;
}

// ============================================================================
// Poller priority benchmark
// ============================================================================
//
// In persistent_worker.hip the poll loop on queue->seq_num runs at default
// priority. s_setprio 3 (highest) MAY make the polling wave observe writes
// from peer GPUs / host faster by getting more issue cycles. Caveat: it can
// also starve other waves on the same SIMD. Measured here in isolation.
//
// Benchmark: launch a kernel where one block polls a flag while another
// block (on a different CU) writes the flag after a delay. Compare time
// from write-to-poll-observation at priority 0 vs priority 3.

extern "C" __global__ void poller_baseline_writer(volatile uint32_t* flag,
                                                  uint64_t delay_cycles)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    while (wall_clock64() - t0 < delay_cycles) { /* spin */ }
    *flag = 1u;
    __threadfence();
}

extern "C" __global__ void poller_setprio0(volatile uint32_t* flag,
                                           uint32_t* observed_cycles)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    asm volatile("s_setprio 0" ::: "memory");
    uint64_t t0 = wall_clock64();
    while (*flag == 0u) { /* spin */ }
    uint64_t t1 = wall_clock64();
    observed_cycles[0] = (uint32_t)(t1 - t0);
}

extern "C" __global__ void poller_setprio3(volatile uint32_t* flag,
                                           uint32_t* observed_cycles)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    asm volatile("s_setprio 3" ::: "memory");
    uint64_t t0 = wall_clock64();
    while (*flag == 0u) { /* spin */ }
    uint64_t t1 = wall_clock64();
    observed_cycles[0] = (uint32_t)(t1 - t0);
}

// Same but uses s_sleep + invalidate (model of persistent_worker poll path).
extern "C" __global__ void poller_sleep_inv(volatile uint32_t* flag,
                                             uint32_t* observed_cycles)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t t0 = wall_clock64();
    while (true) {
        uint32_t v = *flag;
        if (v != 0u) break;
        asm volatile("s_sleep 1\nbuffer_gl1_inv\ns_waitcnt vmcnt(0)" ::: "memory");
    }
    uint64_t t1 = wall_clock64();
    observed_cycles[0] = (uint32_t)(t1 - t0);
}

// ============================================================================
// Driver
// ============================================================================

struct BarrierResult {
    const char* name;
    int blocks;
    int iters;
    double median_cyc_per_op;
    double mean_cyc_per_op;
};

template<typename Kernel, typename... Args>
double run_cooperative(Kernel kernel, int blocks, int threads, int iters,
                       uint64_t* d_cycles, Args... args)
{
    void* params[] = { (void*)&args..., (void*)&d_cycles };
    dim3 g(blocks), b(threads);

    // Warmup
    CHECK(hipLaunchCooperativeKernel((const void*)kernel, g, b, params, 0, 0));
    CHECK(hipDeviceSynchronize());

    // Timed
    CHECK(hipLaunchCooperativeKernel((const void*)kernel, g, b, params, 0, 0));
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cycles(blocks);
    CHECK(hipMemcpy(cycles.data(), d_cycles, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(cycles.begin(), cycles.end());
    double median = (double)cycles[blocks / 2] / (double)iters;

    double sum = 0;
    for (auto c : cycles) sum += (double)c;
    double mean = sum / blocks / iters;
    return median + 1e-30*mean; // primary return is median; mean stashed in printout below
}

template<typename Kernel, typename... Args>
BarrierResult run_coop_named(const char* name, Kernel kernel,
                             int blocks, int threads, int iters,
                             uint64_t* d_cycles, Args... args)
{
    void* params[] = { (void*)&args..., (void*)&d_cycles };
    dim3 g(blocks), b(threads);

    CHECK(hipLaunchCooperativeKernel((const void*)kernel, g, b, params, 0, 0));
    CHECK(hipDeviceSynchronize());
    CHECK(hipLaunchCooperativeKernel((const void*)kernel, g, b, params, 0, 0));
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cycles(blocks);
    CHECK(hipMemcpy(cycles.data(), d_cycles, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(cycles.begin(), cycles.end());

    double median = (double)cycles[blocks / 2] / (double)iters;
    double sum = 0;
    for (auto c : cycles) sum += (double)c;
    double mean = sum / blocks / iters;
    return { name, blocks, iters, median, mean };
}

// Need a non-templated launcher because run_coop_named's variadic params
// pack pointers in reverse on x86 ABI; the lambdas below are explicit per-kernel.

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s   warpSize=%d   gcnArch=%s\n",
           prop.name, prop.warpSize, prop.gcnArchName);

    // Query cooperative-launch max blocks (for reference).
    int max_blocks = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, (const void*)bench_cg_grid_sync<1>, 256, 0));
    printf("MaxActiveBlocksPerMP for 256-thread bench_cg_grid_sync: %d\n", max_blocks);
    int num_cu = prop.multiProcessorCount;
    printf("multiProcessorCount: %d\n", num_cu);
    printf("Cooperative launch max blocks ~ %d (= occupancy x CUs)\n", max_blocks * num_cu);

    // ------------------------------------------------------------------
    // 1. Per-fence cycle cost (single-block, single-thread)
    // ------------------------------------------------------------------
    {
        constexpr int ITER = 4096;
        uint64_t* d_cycles = nullptr;
        uint32_t* d_sink = nullptr;
        CHECK(hipMalloc(&d_cycles, sizeof(uint64_t)));
        CHECK(hipMalloc(&d_sink, 1024 * sizeof(uint32_t)));

        printf("\n=== Fence cost: cycles per call (single-thread, store+fence+load) ===\n");
        printf("Subtract baseline (store+load, no fence) to isolate fence cost.\n");
        printf("  %-32s  %-14s\n", "primitive", "cyc/call");
        printf("  %-32s  %-14s\n", "------------------------------", "----------");

        auto run = [&](const char* name, void(*k)(volatile uint32_t*, uint64_t*)) {
            k<<<1,1>>>(d_sink, d_cycles); CHECK(hipDeviceSynchronize());
            k<<<1,1>>>(d_sink, d_cycles); CHECK(hipDeviceSynchronize());
            uint64_t c; CHECK(hipMemcpy(&c, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
            printf("  %-32s  %-14.3f\n", name, (double)c / ITER);
        };
        run("baseline (store+load, NO FENCE)", bench_baseline_storeload<ITER>);
        run("fence_block (workgroup)",         bench_fence_block<ITER>);
        run("fence_device (agent)",            bench_fence_device<ITER>);
        run("fence_system_uc (system)",        bench_fence_system<ITER>);

        printf("\n=== Fence cost: all 256 threads issue, per-call ===\n");
        printf("  %-32s  %-14s\n", "primitive", "cyc/call");
        printf("  %-32s  %-14s\n", "------------------------------", "----------");
        bench_fence_device_allthreads<ITER><<<1,256>>>(d_sink, d_cycles); CHECK(hipDeviceSynchronize());
        bench_fence_device_allthreads<ITER><<<1,256>>>(d_sink, d_cycles); CHECK(hipDeviceSynchronize());
        uint64_t cc; CHECK(hipMemcpy(&cc, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
        printf("  %-32s  %-14.3f\n", "fence_device (agent, 256 thr)", (double)cc / ITER);

        CHECK(hipFree(d_cycles));
        CHECK(hipFree(d_sink));
    }

    // ------------------------------------------------------------------
    // 2. Grid-wide barrier cost
    // ------------------------------------------------------------------
    {
        constexpr int ITER = 1024;
        const int block_counts[] = {48, 96, 192, 384};

        printf("\n=== Grid barrier cost: cycles per barrier ===\n");
        printf("  %-30s  %-8s  %-12s  %-12s\n", "primitive", "blocks", "median_cyc", "mean_cyc");
        printf("  %-30s  %-8s  %-12s  %-12s\n", "------------------------------", "------", "----------", "----------");

        for (int n_blocks : block_counts) {
            uint64_t* d_cycles = nullptr;
            CHECK(hipMalloc(&d_cycles, n_blocks * sizeof(uint64_t)));

            // Cooperative launch limit check
            int max_active = 0;
            CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active, (const void*)bench_cg_grid_sync<ITER>, 256, 0));
            int max_total = max_active * num_cu;
            if (n_blocks > max_total) {
                printf("  [SKIP] %d blocks > cooperative limit %d\n", n_blocks, max_total);
                CHECK(hipFree(d_cycles));
                continue;
            }

            // ---- cg grid.sync ----
            {
                void* params[] = { (void*)&d_cycles };
                dim3 g(n_blocks), b(256);
                CHECK(hipLaunchCooperativeKernel((const void*)bench_cg_grid_sync<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());
                CHECK(hipLaunchCooperativeKernel((const void*)bench_cg_grid_sync<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());

                std::vector<uint64_t> cycles(n_blocks);
                CHECK(hipMemcpy(cycles.data(), d_cycles, n_blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
                std::sort(cycles.begin(), cycles.end());
                double median = (double)cycles[n_blocks / 2] / ITER;
                double sum = 0; for (auto c : cycles) sum += (double)c;
                double mean = sum / n_blocks / ITER;
                printf("  %-30s  %-8d  %-12.2f  %-12.2f\n", "cg::grid_group::sync", n_blocks, median, mean);
            }

            // ---- atomic_block_barrier ----
            {
                GridBarrierState* d_state = nullptr;
                CHECK(hipMalloc(&d_state, sizeof(GridBarrierState)));
                GridBarrierState zero = {0u, 0u};
                CHECK(hipMemcpy(d_state, &zero, sizeof(zero), hipMemcpyHostToDevice));

                void* params[] = { (void*)&d_state, (void*)&d_cycles };
                dim3 g(n_blocks), b(256);
                CHECK(hipLaunchCooperativeKernel((const void*)bench_atomic_block_barrier<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(d_state, &zero, sizeof(zero), hipMemcpyHostToDevice));
                CHECK(hipLaunchCooperativeKernel((const void*)bench_atomic_block_barrier<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());

                std::vector<uint64_t> cycles(n_blocks);
                CHECK(hipMemcpy(cycles.data(), d_cycles, n_blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
                std::sort(cycles.begin(), cycles.end());
                double median = (double)cycles[n_blocks / 2] / ITER;
                double sum = 0; for (auto c : cycles) sum += (double)c;
                double mean = sum / n_blocks / ITER;
                printf("  %-30s  %-8d  %-12.2f  %-12.2f\n", "atomic_block_barrier", n_blocks, median, mean);
                CHECK(hipFree(d_state));
            }

            // ---- asm_block_barrier ----
            // NOTE: WIP — empirical wedge in some configurations. The C-level
            // atomic_block_barrier already produces near-optimal codegen
            // (single global_atomic_add + s_waitcnt + s_sleep loop, see
            // rdna3_sync_bench-hip-amdgcn-amd-amdhsa-gfx1100.s). The ASM
            // variant offers no measurable upside in these tests; keep it in
            // the header for documentation but leave commented out here.
            #if 0
            {
                GridBarrierState* d_state = nullptr;
                CHECK(hipMalloc(&d_state, sizeof(GridBarrierState)));
                GridBarrierState zero = {0u, 0u};
                CHECK(hipMemcpy(d_state, &zero, sizeof(zero), hipMemcpyHostToDevice));

                void* params[] = { (void*)&d_state, (void*)&d_cycles };
                dim3 g(n_blocks), b(256);
                CHECK(hipLaunchCooperativeKernel((const void*)bench_asm_block_barrier<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(d_state, &zero, sizeof(zero), hipMemcpyHostToDevice));
                CHECK(hipLaunchCooperativeKernel((const void*)bench_asm_block_barrier<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());

                std::vector<uint64_t> cycles(n_blocks);
                CHECK(hipMemcpy(cycles.data(), d_cycles, n_blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
                std::sort(cycles.begin(), cycles.end());
                double median = (double)cycles[n_blocks / 2] / ITER;
                double sum = 0; for (auto c : cycles) sum += (double)c;
                double mean = sum / n_blocks / ITER;
                printf("  %-30s  %-8d  %-12.2f  %-12.2f\n", "asm_block_barrier", n_blocks, median, mean);
                CHECK(hipFree(d_state));
            }
            #endif

            // ---- atomic_block_barrier_double ----
            {
                GridBarrierState* d_a = nullptr; GridBarrierState* d_b = nullptr;
                CHECK(hipMalloc(&d_a, sizeof(GridBarrierState)));
                CHECK(hipMalloc(&d_b, sizeof(GridBarrierState)));
                GridBarrierState zero = {0u, 0u};
                CHECK(hipMemcpy(d_a, &zero, sizeof(zero), hipMemcpyHostToDevice));
                CHECK(hipMemcpy(d_b, &zero, sizeof(zero), hipMemcpyHostToDevice));

                void* params[] = { (void*)&d_a, (void*)&d_b, (void*)&d_cycles };
                dim3 g(n_blocks), b(256);
                CHECK(hipLaunchCooperativeKernel((const void*)bench_atomic_block_barrier_double<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(d_a, &zero, sizeof(zero), hipMemcpyHostToDevice));
                CHECK(hipMemcpy(d_b, &zero, sizeof(zero), hipMemcpyHostToDevice));
                CHECK(hipLaunchCooperativeKernel((const void*)bench_atomic_block_barrier_double<ITER>, g, b, params, 0, 0));
                CHECK(hipDeviceSynchronize());

                std::vector<uint64_t> cycles(n_blocks);
                CHECK(hipMemcpy(cycles.data(), d_cycles, n_blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
                std::sort(cycles.begin(), cycles.end());
                double median = (double)cycles[n_blocks / 2] / ITER;
                double sum = 0; for (auto c : cycles) sum += (double)c;
                double mean = sum / n_blocks / ITER;
                printf("  %-30s  %-8d  %-12.2f  %-12.2f\n", "atomic_block_barrier_2state", n_blocks, median, mean);
                CHECK(hipFree(d_a)); CHECK(hipFree(d_b));
            }

            CHECK(hipFree(d_cycles));
        }
    }

    // ------------------------------------------------------------------
    // 3. Coarse barrier (every-N) — informs gs1 fusion ROI
    // ------------------------------------------------------------------
    {
        constexpr int ITER = 4096;
        const int n_blocks = 192;

        printf("\n=== Coarse barrier: cg::grid.sync() every Nth iteration (192 blocks) ===\n");
        printf("This informs the gs1 fusion ROI: if cost is per-barrier, fusion linear; if mostly\n"
               "per-loop-overhead, fusion has lower payoff.\n");
        printf("  %-30s  %-12s\n", "every-N", "cyc/iter");
        printf("  %-30s  %-12s\n", "------------------------------", "----------");

        uint64_t* d_cycles = nullptr;
        CHECK(hipMalloc(&d_cycles, n_blocks * sizeof(uint64_t)));

        auto run_every = [&](const char* name, auto kernel_fn) {
            void* params[] = { (void*)&d_cycles };
            dim3 g(n_blocks), b(256);
            CHECK(hipLaunchCooperativeKernel((const void*)kernel_fn, g, b, params, 0, 0));
            CHECK(hipDeviceSynchronize());
            CHECK(hipLaunchCooperativeKernel((const void*)kernel_fn, g, b, params, 0, 0));
            CHECK(hipDeviceSynchronize());
            std::vector<uint64_t> cycles(n_blocks);
            CHECK(hipMemcpy(cycles.data(), d_cycles, n_blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
            std::sort(cycles.begin(), cycles.end());
            double median = (double)cycles[n_blocks / 2] / ITER;
            printf("  %-30s  %-12.3f\n", name, median);
        };

        run_every("every 1 (baseline)", bench_cg_grid_sync_every<ITER, 1>);
        run_every("every 2",            bench_cg_grid_sync_every<ITER, 2>);
        run_every("every 4",            bench_cg_grid_sync_every<ITER, 4>);
        run_every("every 8",            bench_cg_grid_sync_every<ITER, 8>);
        run_every("every 16",           bench_cg_grid_sync_every<ITER, 16>);

        CHECK(hipFree(d_cycles));
    }

    // Poller-priority cross-block latency test omitted: requires reader and
    // writer kernels on independent streams, which races against the
    // cooperative_kernel_launch path used elsewhere in this benchmark. The
    // s_setprio instruction itself is well-documented (RDNA3 ISA §3.3) and
    // should be tested in the actual persistent_worker context, not in a
    // microbench. See kernels/persistent_worker.hip poll loop (line ~52).

    return 0;
}
