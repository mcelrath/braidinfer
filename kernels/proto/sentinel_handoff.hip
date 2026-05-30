// braidinfer-g1wv PROTOTYPE — sentinel-based producer→consumer handoff on gfx1100.
//
// Validates the Kog-monokernel idea (https://blog.kog.ai/...): replace a grid
// barrier between producer and consumer stages with a NaN sentinel written into
// the DATA buffer — consumers poll the data location until the sentinel is gone
// ("readiness encoded in the data path"). On gfx1100 the cross-CU coherence point
// is GL2; we use agent-scope atomics (the same scope braidinfer's
// atomic_block_barrier arrival counter relies on), NOT Kog's CDNA3 sc1.
//
// Ring topology: block i is BOTH a producer (writes slots[i]) and a consumer
// (reads its neighbor slots[(i+1)%N]). Correct result[i] == neighbor value.
//
//   BASELINE   : write slots[i]; cg::this_grid().sync(); read neighbor.
//   SENTINEL   : slots[] pre-init to qNaN; atomic-store slots[i]; spin atomic-load
//                neighbor until != qNaN; read it. NO grid barrier.
//
// Build:  hipcc --offload-arch=gfx1100 -O3 kernels/proto/sentinel_handoff.hip -o /tmp/sentinel_handoff
// Run:    launch-gpu -g 1 -- /tmp/sentinel_handoff
//
// Primary validation: correctness over N COLD trials (0 errors) — does a consumer
// reliably observe a producer's write cross-CU without a barrier? Secondary:
// per-block clock64 handoff span, sentinel vs barrier.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;

// braidinfer-g1wv A/B: also benchmark the PRODUCTION barrier (atomic_block_barrier,
// ~96 cyc) — NOT just grid.sync (~5047) — so we compare the sentinel against what
// braidinfer actually uses, not the slow cg path.
#include "../rdna3/rdna3_barrier.h"  // braidinfer::rdna3::atomic_block_barrier + GridBarrierState
__device__ braidinfer::rdna3::GridBarrierState g_proto_gbs;

static const unsigned int SENTINEL = 0x7fc00000u; // canonical qNaN bit pattern

__device__ __forceinline__ unsigned int load_agent(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void store_agent(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// Baseline: producer write → grid barrier → consumer read neighbor.
__global__ void handoff_barrier(float* slots, float* results, int n,
                                unsigned long long* span) {
    cg::grid_group grid = cg::this_grid();
    int b = blockIdx.x;
    unsigned long long t0 = wall_clock64();
    if (threadIdx.x == 0) ((volatile float*)slots)[b] = (float)(b + 1);
    grid.sync();
    if (threadIdx.x == 0) {
        results[b] = ((volatile float*)slots)[(b + 1) % n];
        span[b] = wall_clock64() - t0;
    }
}

// Sentinel: slots[] start as qNaN. Producer atomic-stores its value; consumer
// spins on the neighbor's data location until the qNaN is gone. No barrier.
__global__ void handoff_sentinel(float* slots, float* results, int n,
                                 unsigned long long* span) {
    int b = blockIdx.x;
    unsigned int* uslots = reinterpret_cast<unsigned int*>(slots);
    unsigned long long t0 = wall_clock64();
    if (threadIdx.x == 0) {
        float val = (float)(b + 1);
        unsigned int bits;
        __builtin_memcpy(&bits, &val, 4);
        store_agent(&uslots[b], bits);          // publish into the data path
        unsigned int nb = (b + 1) % n;
        unsigned int v;
        do { v = load_agent(&uslots[nb]); } while (v == SENTINEL); // poll the data
        float fv;
        __builtin_memcpy(&fv, &v, 4);
        results[b] = fv;
        span[b] = wall_clock64() - t0;
    }
}

// braidinfer's PRODUCTION barrier — the real A/B baseline (NOT grid.sync).
// atomic_block_barrier is the per-instruction barrier dispatch_opcode emits between
// ops, including QKV->attention. write slots[i] -> atomic_block_barrier -> read neighbor.
__global__ void handoff_atomic_barrier(float* slots, float* results, int n,
                                       unsigned long long* span) {
    int b = blockIdx.x;
    unsigned long long t0 = wall_clock64();
    if (threadIdx.x == 0) ((volatile float*)slots)[b] = (float)(b + 1);
    braidinfer::rdna3::atomic_block_barrier(&g_proto_gbs);
    if (threadIdx.x == 0) {
        results[b] = ((volatile float*)slots)[(b + 1) % n];
        span[b] = wall_clock64() - t0;
    }
}

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 96;   // blocks (≤ co-residency cap)
    int TRIALS = (argc > 2) ? atoi(argv[2]) : 20;
    const int THREADS = 64;

    int dev = 0; CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, dev));
    int coop = 0; hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, dev);
    printf("device=%s CUs=%d cooperativeLaunch=%d blocks=%d threads=%d trials=%d\n",
           prop.name, prop.multiProcessorCount, coop, N, THREADS, TRIALS);
    if (!coop) { printf("FAIL: cooperative launch unsupported\n"); return 1; }

    float *d_slots, *d_results; unsigned long long *d_span;
    CHECK(hipMalloc(&d_slots, N * sizeof(float)));
    CHECK(hipMalloc(&d_results, N * sizeof(float)));
    CHECK(hipMalloc(&d_span, N * sizeof(unsigned long long)));

    std::vector<float> h_nan(N), h_zero(N, 0.0f), h_results(N);
    std::vector<unsigned long long> h_span(N);
    { float f; __builtin_memcpy(&f, &SENTINEL, 4); std::fill(h_nan.begin(), h_nan.end(), f); }

    auto run = [&](void* kern, bool sentinel, const char* name) -> int {
        int total_err = 0;
        std::vector<unsigned long long> means; means.reserve(TRIALS);
        unsigned long long worst = 0;
        for (int t = 0; t < TRIALS; ++t) {
            // Cold re-init each trial: sentinel→qNaN, baseline→0.
            CHECK(hipMemcpy(d_slots, sentinel ? h_nan.data() : h_zero.data(),
                            N * sizeof(float), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(d_results, h_zero.data(), N * sizeof(float), hipMemcpyHostToDevice));
            void* args[] = { &d_slots, &d_results, &N, &d_span };
            CHECK(hipLaunchCooperativeKernel(
                kern,
                dim3(N), dim3(THREADS), args, 0, 0));
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(h_results.data(), d_results, N * sizeof(float), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(h_span.data(), d_span, N * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            int err = 0;
            for (int b = 0; b < N; ++b) {
                float expect = (float)(((b + 1) % N) + 1);
                if (h_results[b] != expect) err++;
            }
            total_err += err;
            unsigned long long sum = 0, mx = 0;
            for (int b = 0; b < N; ++b) { sum += h_span[b]; mx = std::max(mx, h_span[b]); }
            means.push_back(sum / N);
            worst = std::max(worst, mx);
        }
        std::sort(means.begin(), means.end());
        printf("  %-9s : errors=%d/%d trials | median per-block span=%llu wclk | worst-block=%llu wclk\n",
               name, total_err, TRIALS, means[means.size()/2], worst);
        return total_err;
    };

    printf("--- handoff: ring producer→consumer, N cold trials ---\n");
    int eb = run((void*)handoff_barrier,        false, "grid.sync");
    int ea = run((void*)handoff_atomic_barrier, false, "atomic_barr");  // braidinfer's real barrier
    int es = run((void*)handoff_sentinel,       true,  "sentinel");
    (void)ea;

    hipFree(d_slots); hipFree(d_results); hipFree(d_span);
    if (es == 0 && eb == 0) { printf("PASS — both correct; sentinel handoff works without a barrier on gfx1100\n"); return 0; }
    printf("RESULT — barrier_err=%d sentinel_err=%d\n", eb, es);
    return (es == 0) ? 0 : 1;
}
