// Lane-primitive microbenchmark for gfx1100 (RDNA3, wave32).
//
// Companion to reduce_bench. Measures the non-reduction lane primitives in
// kernels/rdna3_lane.h:
//
//   wave_ballot / wave_any / wave_all / wave_count_predicate
//   lane_broadcast<src>           (v_readlane, compile-time)
//   lane_broadcast(v, src)        (v_readlane, dynamic uniform src)
//   lane_read_first               (v_readfirstlane)
//   lane_write<dst>               (v_writelane via inline asm)
//   wave_xor_butterfly<N>         (ds_swizzle_b32 BitMaskPerm XOR)
//   wave_quad_swap<N>             (ds_swizzle_b32 QDMode)
//   subwave_tile<W>::shfl_xor<N>  (DPP row_xmask)
//   subwave_tile<W>::broadcast<R> (ds_swizzle BitMaskPerm)
//   wave32_reduce_max             (DPP+permlanex16, max)
//   subwave_reduce_max<W>         (DPP, max)
//
// And cross-checks against the slow paths (HIP __ballot, __shfl_sync) to
// confirm correctness.
//
// Build:
//   ./build_lane_bench.sh
//
// Run (always via launch-gpu.py):
//   python3 scripts/launch-gpu.py --timeout 600 -- \
//     /home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/lane_bench/rdna3_lane_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

// Path is relative; build script runs from the kernels/ root via -I include.
#include "rdna3_lane.h"

using namespace braidinfer::rdna3;

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// ============================================================================
// Reference (slow / CUDA-shaped) primitives
// ============================================================================

// HIP __ballot returns u64 even on wave32. Used as the reference for
// wave_ballot's u32 form.
__device__ __forceinline__ uint32_t wave_ballot_hip(bool pred) {
    return (uint32_t)__ballot((int)pred);
}

// __shfl-based broadcast (lowers to ds_bpermute_b32 on RDNA3).
__device__ __forceinline__ float lane_broadcast_shfl(float v, int src) {
    return __shfl(v, src);
}

// __shfl_xor — for cross-checking subwave_tile<W>::shfl_xor.
__device__ __forceinline__ float subwave_shfl_xor_ref(float v, int N, int W) {
    return __shfl_xor(v, N, W);
}

// ============================================================================
// Cycle-count helper
// ============================================================================

__device__ __forceinline__ uint64_t rdtimer() {
    return (uint64_t)wall_clock64();
}

// ============================================================================
// Performance benchmark kernels
// ============================================================================
//
// Pattern: each block runs ITER iterations of the primitive, accumulating into
// a result that depends on input so the optimizer can't dead-code it.

template<int ITER>
__global__ void bench_ballot_hip(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    uint32_t acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        bool p = (v + (float)i) > (float)(i & 31);
        acc ^= wave_ballot_hip(p);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = (float)acc;
    }
}

template<int ITER>
__global__ void bench_ballot_w32(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    uint32_t acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        bool p = (v + (float)i) > (float)(i & 31);
        acc ^= wave_ballot(p);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = (float)acc;
    }
}

template<int ITER>
__global__ void bench_lane_bcast_shfl(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        // src lane is uniform (i&31).
        acc += lane_broadcast_shfl(v + (float)i, i & 31);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_lane_bcast_readlane(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += lane_broadcast(v + (float)i, i & 31);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_lane_bcast_const0(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        // lane_broadcast<0> -> v_readlane_b32 vDst, vSrc, 0
        acc += lane_broadcast<0>(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_lane_read_first(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += lane_read_first(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_xor_butterfly_shfl(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += __shfl_xor(v + (float)i, 4, 32);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_xor_butterfly_swizzle(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += wave_xor_butterfly<4>(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_quad_swap_swizzle(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += wave_quad_swap<1>(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int W, int ITER>
__global__ void bench_subwave_bcast(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += subwave_tile<W>::template broadcast<0>(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int W, int ITER>
__global__ void bench_subwave_shfl_xor_shfl(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += __shfl_xor(v + (float)i, W >> 1, W);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int W, int ITER>
__global__ void bench_subwave_shfl_xor_dpp(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += subwave_tile<W>::template shfl_xor<(W >> 1)>(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_wave32_max(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        acc += wave32_reduce_max(v + (float)i);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_block_max_lds(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float sh[256];
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        sh[tid] = v + (float)i;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
            __syncthreads();
        }
        acc += sh[0];
        __syncthreads();
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_block_max_dpp_lds(const float* in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    __shared__ float sh[8];
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = wave32_reduce_max(x);
        if (lane == 0) sh[warp] = x;
        __syncthreads();
        if (tid == 0) {
            float m = sh[0];
            for (int j = 1; j < 8; j++) m = fmaxf(m, sh[j]);
            sh[0] = m;
        }
        __syncthreads();
        acc += sh[0];
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

// ============================================================================
// Correctness kernels
// ============================================================================

__global__ void corr_ballot(const float* in, uint32_t* out_w32, uint64_t* out_w64) {
    int tid = threadIdx.x;
    float v = in[tid];
    bool p = v > 0.5f;
    if (tid == 0) {
        out_w32[blockIdx.x] = wave_ballot(p);
        out_w64[blockIdx.x] = wave_ballot_hip(p);
    } else {
        // Force per-lane evaluation
        (void)wave_ballot(p);
    }
}

__global__ void corr_lane_broadcast(const float* in, float* out_shfl, float* out_readlane,
                                    int src_lane) {
    int tid = threadIdx.x;
    float v = in[tid];
    out_shfl[blockIdx.x * 32 + tid]     = lane_broadcast_shfl(v, src_lane);
    out_readlane[blockIdx.x * 32 + tid] = lane_broadcast(v, src_lane);
}

__global__ void corr_xor_butterfly(const float* in, float* out_shfl, float* out_swizzle, int N) {
    int tid = threadIdx.x;
    float v = in[tid];
    out_shfl[blockIdx.x * 32 + tid] = __shfl_xor(v, N, 32);
    // We can only template the swizzle on a const N; sweep N = {1,2,4,8,16}.
    float sw;
    switch (N) {
        case 1:  sw = wave_xor_butterfly<1>(v);  break;
        case 2:  sw = wave_xor_butterfly<2>(v);  break;
        case 4:  sw = wave_xor_butterfly<4>(v);  break;
        case 8:  sw = wave_xor_butterfly<8>(v);  break;
        case 16: sw = wave_xor_butterfly<16>(v); break;
        default: sw = v; break;
    }
    out_swizzle[blockIdx.x * 32 + tid] = sw;
}

__global__ void corr_subwave_bcast_w8(const float* in, float* out_shfl, float* out_swiz) {
    int tid = threadIdx.x;
    float v = in[tid];
    // Broadcast lane 0 of each 8-lane group via __shfl(_, 0, 8) and via swizzle.
    out_shfl[blockIdx.x * 32 + tid] = __shfl(v, 0, 8);
    out_swiz[blockIdx.x * 32 + tid] = subwave_tile<8>::broadcast<0>(v);
}

__global__ void corr_lane_write(const float* in, float* out, int dst_lane) {
    int tid = threadIdx.x;
    float v = in[tid];
    // Write `v` (lane 0's value, post readfirstlane uniformization) into
    // lane `dst_lane` of `v`. Other lanes' values unchanged.
    float r;
    switch (dst_lane) {
        case 0:  r = lane_write<0>(v, lane_read_first(v));  break;
        case 7:  r = lane_write<7>(v, lane_read_first(v));  break;
        case 15: r = lane_write<15>(v, lane_read_first(v)); break;
        case 31: r = lane_write<31>(v, lane_read_first(v)); break;
        default: r = v;
    }
    out[blockIdx.x * 32 + tid] = r;
}

__global__ __launch_bounds__(256) void corr_wave32_max(const float* in, float* out) {
    int tid = threadIdx.x;
    float v = in[blockIdx.x * 32 + (tid & 31)];
    if (tid < 32) {
        float m = wave32_reduce_max(v);
        if (tid == 0) out[blockIdx.x] = m;
    }
}

// ============================================================================
// Host driver
// ============================================================================

struct BenchResult {
    const char* name;
    int width;
    int blocks;
    double avg_cycles_per_op;
    double median_cycles_per_op;
};

template<typename Kernel>
BenchResult run_bench(const char* name, Kernel kernel, int blocks, int threads, int iters,
                      const float* d_in, float* d_result, uint64_t* d_cycles) {
    kernel<<<dim3(blocks), dim3(threads)>>>(d_in, d_result, d_cycles);
    CHECK(hipDeviceSynchronize());
    kernel<<<dim3(blocks), dim3(threads)>>>(d_in, d_result, d_cycles);
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cycles(blocks);
    CHECK(hipMemcpy(cycles.data(), d_cycles, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));

    double sum = 0;
    for (auto c : cycles) sum += (double)c;
    std::sort(cycles.begin(), cycles.end());
    double median = (double)cycles[cycles.size() / 2];

    BenchResult r;
    r.name = name;
    r.width = threads;
    r.blocks = blocks;
    r.avg_cycles_per_op    = sum / blocks / iters;
    r.median_cycles_per_op = median / iters;
    return r;
}

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s   warpSize=%d   gcnArch=%s\n",
           prop.name, prop.warpSize, prop.gcnArchName);

    if (prop.warpSize != 32) {
        fprintf(stderr, "ERROR: this bench requires wave32 (got warpSize=%d)\n",
                prop.warpSize);
        return 1;
    }

    constexpr int ITER   = 4096;
    constexpr int BLOCKS = 64;
    constexpr int BLK    = 256;

    std::vector<float> hin(BLK);
    for (int i = 0; i < BLK; i++) hin[i] = (float)(i % 17) * 0.137f + 0.5f - (i & 1);

    float* d_in = nullptr;
    float* d_result = nullptr;
    uint64_t* d_cycles = nullptr;
    CHECK(hipMalloc(&d_in,     BLK * sizeof(float)));
    CHECK(hipMalloc(&d_result, BLOCKS * sizeof(float)));
    CHECK(hipMalloc(&d_cycles, BLOCKS * sizeof(uint64_t)));
    CHECK(hipMemcpy(d_in, hin.data(), BLK * sizeof(float), hipMemcpyHostToDevice));

    printf("\n=== Lane-primitive cycle counts (in-kernel s_memrealtime ticks per op) ===\n");
    printf("Iterations per block: %d, blocks per launch: %d (median over blocks)\n", ITER, BLOCKS);

    std::vector<BenchResult> results;

    // Ballot
    results.push_back(run_bench("ballot HIP-w64",   bench_ballot_hip<ITER>,       BLOCKS, 32,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("ballot w32",       bench_ballot_w32<ITER>,       BLOCKS, 32,  ITER, d_in, d_result, d_cycles));

    // Lane broadcast
    results.push_back(run_bench("bcast __shfl",     bench_lane_bcast_shfl<ITER>,  BLOCKS, 32,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("bcast readlane",   bench_lane_bcast_readlane<ITER>, BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("bcast<0> readlane",bench_lane_bcast_const0<ITER>,BLOCKS, 32,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("readfirstlane",    bench_lane_read_first<ITER>,  BLOCKS, 32,  ITER, d_in, d_result, d_cycles));

    // XOR butterfly (single-stage, N=4)
    results.push_back(run_bench("xor4 __shfl_xor",  bench_xor_butterfly_shfl<ITER>,    BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("xor4 ds_swizzle",  bench_xor_butterfly_swizzle<ITER>, BLOCKS, 32, ITER, d_in, d_result, d_cycles));

    // Quad swap (N=1)
    results.push_back(run_bench("quad_swap<1> swiz",bench_quad_swap_swizzle<ITER>,     BLOCKS, 32, ITER, d_in, d_result, d_cycles));

    // Subwave broadcast and shfl_xor
    results.push_back(run_bench("sub<2> bcast<0>",  bench_subwave_bcast<2,ITER>,        BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("sub<4> bcast<0>",  bench_subwave_bcast<4,ITER>,        BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("sub<8> bcast<0>",  bench_subwave_bcast<8,ITER>,        BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("sub<16> bcast<0>", bench_subwave_bcast<16,ITER>,       BLOCKS, 32, ITER, d_in, d_result, d_cycles));

    results.push_back(run_bench("sub<8> shfl_xor SHFL", bench_subwave_shfl_xor_shfl<8,ITER>, BLOCKS, 32, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("sub<8> shfl_xor DPP",  bench_subwave_shfl_xor_dpp<8,ITER>,  BLOCKS, 32, ITER, d_in, d_result, d_cycles));

    // Wave / block max
    results.push_back(run_bench("wave32 max DPP+P16",  bench_wave32_max<ITER>,        BLOCKS, 32,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("block256 max LDS",    bench_block_max_lds<ITER>,     BLOCKS, 256, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("block256 max DPP+LDS",bench_block_max_dpp_lds<ITER>, BLOCKS, 256, ITER, d_in, d_result, d_cycles));

    printf("\n  %-26s  %-8s  %-12s  %-12s\n", "primitive", "threads", "median_cyc", "mean_cyc");
    printf("  %-26s  %-8s  %-12s  %-12s\n", "--------------------------", "-------", "------------", "------------");
    for (auto& r : results) {
        printf("  %-26s  %-8d  %-12.2f  %-12.2f\n",
               r.name, r.width, r.median_cycles_per_op, r.avg_cycles_per_op);
    }

    CHECK(hipFree(d_result));
    CHECK(hipFree(d_cycles));

    // ========================================================================
    // Correctness checks
    // ========================================================================
    printf("\n=== Correctness checks ===\n");

    // 1. wave_ballot: u32 form must equal low 32 bits of HIP __ballot.
    {
        constexpr int B = 16;
        uint32_t* d_w32; uint64_t* d_w64;
        CHECK(hipMalloc(&d_w32, B * sizeof(uint32_t)));
        CHECK(hipMalloc(&d_w64, B * sizeof(uint64_t)));
        corr_ballot<<<B, 32>>>(d_in, d_w32, d_w64);
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h32(B);
        std::vector<uint64_t> h64(B);
        CHECK(hipMemcpy(h32.data(), d_w32, B * sizeof(uint32_t), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(h64.data(), d_w64, B * sizeof(uint64_t), hipMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < B; i++) {
            if (h32[i] != (uint32_t)h64[i]) {
                printf("  ballot mismatch block %d: w32=0x%08x  w64=0x%016llx\n",
                       i, h32[i], (unsigned long long)h64[i]);
                ok = false;
            }
            if ((h64[i] >> 32) != 0) {
                printf("  ballot upper-32 nonzero in wave32 mode block %d: 0x%016llx\n",
                       i, (unsigned long long)h64[i]);
                ok = false;
            }
        }
        printf("  wave_ballot vs __ballot:  %s\n", ok ? "PASS" : "FAIL");
        CHECK(hipFree(d_w32)); CHECK(hipFree(d_w64));
    }

    // 2. lane_broadcast(v, src) vs __shfl(v, src). Sweep src in 0..31.
    {
        constexpr int B = 16;
        float* d_a; float* d_b;
        CHECK(hipMalloc(&d_a, B * 32 * sizeof(float)));
        CHECK(hipMalloc(&d_b, B * 32 * sizeof(float)));
        bool all_ok = true;
        for (int src = 0; src < 32; src++) {
            corr_lane_broadcast<<<B, 32>>>(d_in, d_a, d_b, src);
            CHECK(hipDeviceSynchronize());
            std::vector<float> a(B*32), b(B*32);
            CHECK(hipMemcpy(a.data(), d_a, B*32*sizeof(float), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), d_b, B*32*sizeof(float), hipMemcpyDeviceToHost));
            for (int j = 0; j < B*32; j++) {
                if (a[j] != b[j]) {
                    printf("  bcast mismatch src=%d j=%d  shfl=%g  readlane=%g\n",
                           src, j, a[j], b[j]);
                    all_ok = false;
                    break;
                }
            }
            if (!all_ok) break;
        }
        printf("  lane_broadcast vs __shfl:  %s\n", all_ok ? "PASS" : "FAIL");
        CHECK(hipFree(d_a)); CHECK(hipFree(d_b));
    }

    // 3. wave_xor_butterfly<N> vs __shfl_xor(v, N, 32) for N in {1,2,4,8,16}.
    {
        constexpr int B = 16;
        float* d_a; float* d_b;
        CHECK(hipMalloc(&d_a, B * 32 * sizeof(float)));
        CHECK(hipMalloc(&d_b, B * 32 * sizeof(float)));
        bool all_ok = true;
        int Ns[] = {1, 2, 4, 8, 16};
        for (int N : Ns) {
            corr_xor_butterfly<<<B, 32>>>(d_in, d_a, d_b, N);
            CHECK(hipDeviceSynchronize());
            std::vector<float> a(B*32), b(B*32);
            CHECK(hipMemcpy(a.data(), d_a, B*32*sizeof(float), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), d_b, B*32*sizeof(float), hipMemcpyDeviceToHost));
            for (int j = 0; j < B*32; j++) {
                if (a[j] != b[j]) {
                    printf("  xor_butterfly mismatch N=%d j=%d  shfl=%g  swiz=%g\n",
                           N, j, a[j], b[j]);
                    all_ok = false;
                    break;
                }
            }
            if (!all_ok) break;
        }
        printf("  wave_xor_butterfly vs __shfl_xor:  %s\n", all_ok ? "PASS" : "FAIL");
        CHECK(hipFree(d_a)); CHECK(hipFree(d_b));
    }

    // 4. subwave_tile<8>::broadcast<0> vs __shfl(v, 0, 8).
    {
        constexpr int B = 16;
        float* d_a; float* d_b;
        CHECK(hipMalloc(&d_a, B * 32 * sizeof(float)));
        CHECK(hipMalloc(&d_b, B * 32 * sizeof(float)));
        corr_subwave_bcast_w8<<<B, 32>>>(d_in, d_a, d_b);
        CHECK(hipDeviceSynchronize());
        std::vector<float> a(B*32), b(B*32);
        CHECK(hipMemcpy(a.data(), d_a, B*32*sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), d_b, B*32*sizeof(float), hipMemcpyDeviceToHost));
        bool ok = true;
        for (int j = 0; j < B*32; j++) {
            if (a[j] != b[j]) {
                printf("  subwave bcast<8> mismatch j=%d  shfl=%g  swiz=%g\n",
                       j, a[j], b[j]);
                ok = false;
                break;
            }
        }
        printf("  subwave_tile<8>::broadcast<0> vs __shfl(_,0,8):  %s\n",
               ok ? "PASS" : "FAIL");
        CHECK(hipFree(d_a)); CHECK(hipFree(d_b));
    }

    // 5. lane_write<DstLane> — writes the readfirstlane'd value into one lane,
    //    leaves all others alone.
    {
        constexpr int B = 16;
        float* d_o;
        CHECK(hipMalloc(&d_o, B * 32 * sizeof(float)));
        bool all_ok = true;
        int dsts[] = {0, 7, 15, 31};
        for (int d : dsts) {
            corr_lane_write<<<B, 32>>>(d_in, d_o, d);
            CHECK(hipDeviceSynchronize());
            std::vector<float> out(B*32);
            CHECK(hipMemcpy(out.data(), d_o, B*32*sizeof(float), hipMemcpyDeviceToHost));
            // Expect: out[block*32 + lane] == in[lane] for lane != d, == in[0] for lane == d.
            for (int blk = 0; blk < B; blk++) {
                for (int lane = 0; lane < 32; lane++) {
                    float expect = (lane == d) ? hin[0] : hin[lane];
                    float got = out[blk*32 + lane];
                    if (got != expect) {
                        printf("  lane_write mismatch dst=%d lane=%d  expect=%g  got=%g\n",
                               d, lane, expect, got);
                        all_ok = false;
                        break;
                    }
                }
                if (!all_ok) break;
            }
            if (!all_ok) break;
        }
        printf("  lane_write<Dst> single-lane insert:  %s\n", all_ok ? "PASS" : "FAIL");
        CHECK(hipFree(d_o));
    }

    // 6. wave32_reduce_max vs CPU max.
    {
        constexpr int B = 256;
        std::vector<float> all(B * 32);
        std::mt19937 rng(0xC0FFEE);
        std::normal_distribution<float> g(0.f, 4.f);
        std::vector<float> ref(B);
        for (int b = 0; b < B; b++) {
            float m = -1e30f;
            for (int i = 0; i < 32; i++) {
                float v = g(rng);
                all[b*32 + i] = v;
                m = std::max(m, v);
            }
            ref[b] = m;
        }
        float* dd; float* dout;
        CHECK(hipMalloc(&dd, all.size() * sizeof(float)));
        CHECK(hipMalloc(&dout, B * sizeof(float)));
        CHECK(hipMemcpy(dd, all.data(), all.size() * sizeof(float), hipMemcpyHostToDevice));
        corr_wave32_max<<<B, 32>>>(dd, dout);
        CHECK(hipDeviceSynchronize());
        std::vector<float> got(B);
        CHECK(hipMemcpy(got.data(), dout, B*sizeof(float), hipMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < B; i++) {
            if (got[i] != ref[i]) {
                printf("  wave32_reduce_max mismatch i=%d  got=%g  ref=%g\n", i, got[i], ref[i]);
                ok = false;
                break;
            }
        }
        printf("  wave32_reduce_max vs CPU:  %s\n", ok ? "PASS" : "FAIL");
        CHECK(hipFree(dd)); CHECK(hipFree(dout));
    }

    CHECK(hipFree(d_in));
    return 0;
}
