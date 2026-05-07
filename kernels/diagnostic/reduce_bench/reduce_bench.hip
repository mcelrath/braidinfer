// Reduction primitive microbenchmark for gfx1100 (RDNA3, wave32).
//
// Compares candidate primitives at the configurations braidinfer uses:
//   - subwave (tpg in {2,4,8,16}) for MoE GEMV intra-group reduction
//   - full wave32 for warp-level reduction
//   - 256-thread block reduction
//
// Primitives tested:
//   SHFL    : __shfl_down               (lowers to ds_bpermute_b32, LDS, ~4 cycle)
//   DPP     : __builtin_amdgcn_update_dpp (VALU, 1-cycle, register-to-register)
//   PERM16  : permlanex16 + permlane16   (VALU, register-to-register; cross-16 only)
//   LDS     : tree reduce in __shared__
//
// Build:
//   ./build_bench.sh
//
// Run (always via launch-gpu.py):
//   python3 scripts/launch-gpu.py --timeout 600 -- \
//     /home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/reduce_bench/reduce_bench
//
// Cycle counts are in-kernel SCC (s_memrealtime) ticks per reduction, averaged
// over many iterations to amortize loop overhead.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// ============================================================================
// Primitives
// ============================================================================

// Wave32 full reduction via __shfl_down (5 stages of ds_bpermute_b32).
__device__ __forceinline__ float wave32_sum_shfl(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

// Wave32 full reduction via DPP + permlanex16.
// row_shr is the "shift right within rows of 16" DPP modifier.
// DPP control encodings (gfx10/11):
//   row_shr:N      = 0x110 | N    (shift each 16-lane row right by N, lanes 0..15-N)
// We use bound_ctrl=0 (out-of-bounds source = 0).
// row_xmask:N — within each 16-lane row, lane i is paired with lane i XOR N.
// After v += dpp_row_xmask_add<N>(v), every lane in the row has the sum of
// itself and its XOR-N partner. Doing this for N = 8,4,2,1 gives every lane
// the sum of all 16 lanes in its row (butterfly reduction).
template<int N>
__device__ __forceinline__ float dpp_row_xmask_add(float v) {
    static_assert(N >= 1 && N <= 15, "row_xmask N must be 1..15");
    union { float f; int i; } a, b;
    a.f = v;
    // dpp_ctrl 0x160|N = row_xmask:N within each 16-lane row (gfx10+).
    constexpr int ctrl = 0x160 | (N & 0xF);
    b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
    return v + b.f;
}

// row_shr:N (downward) — kept for the case where we explicitly want a
// destructive shift-down reduction with the result at lane 15 of the row.
// Note bound_ctrl=0 means lanes that fetch from out-of-row get 0.
template<int N>
__device__ __forceinline__ float dpp_row_shr_add(float v) {
    static_assert(N >= 1 && N <= 15, "row_shr N must be 1..15");
    union { float f; int i; } a, b;
    a.f = v;
    constexpr int ctrl = 0x110 | (N & 0xF);
    b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
    return v + b.f;
}

__device__ __forceinline__ float wave32_sum_dpp(float v) {
    // Butterfly reduction within each 16-lane row.
    v = dpp_row_xmask_add<8>(v);
    v = dpp_row_xmask_add<4>(v);
    v = dpp_row_xmask_add<2>(v);
    v = dpp_row_xmask_add<1>(v);
    // After 4 xmask stages, every lane in row 0 has the sum of lanes 0..15;
    // every lane in row 1 has the sum of lanes 16..31.
    // Combine the two rows using permlanex16 to swap halves.
    // permlanex16(old, src, sel0, sel1, fi, bc): with sel0=sel1=0 and fi=bc=1,
    // each lane in low half receives the value from lane 16 (broadcast),
    // each lane in high half receives the value from lane 0.
    // Since EVERY lane in row 1 has the same value (the row sum), broadcast
    // from lane 16 puts the row-1 sum into row 0. Symmetric for the other side.
    union { float f; int i; } a, b;
    a.f = v;
    b.i = __builtin_amdgcn_permlanex16((unsigned)a.i, (unsigned)a.i, 0u, 0u, true, true);
    return v + b.f;
    // After this every lane in the wave has the full wave sum.
}

// Sub-wave reduction: width threads cooperate, width in {2,4,8,16}.
// For width <= 16, DPP row_shr is sufficient (within a single 16-lane row).
__device__ __forceinline__ float subwave_sum_shfl(float v, int width) {
    for (int o = width >> 1; o > 0; o >>= 1) v += __shfl_down(v, o, width);
    return v;
}

template<int Width>
__device__ __forceinline__ float subwave_sum_dpp(float v) {
    static_assert(Width == 2 || Width == 4 || Width == 8 || Width == 16, "");
    // Use row_xmask so every lane in the sub-group ends up with the partial sum.
    // (Matches __shfl_down(width) semantics that the existing code consumes from lane 0.)
    if constexpr (Width >= 16) v = dpp_row_xmask_add<8>(v);
    if constexpr (Width >=  8) v = dpp_row_xmask_add<4>(v);
    if constexpr (Width >=  4) v = dpp_row_xmask_add<2>(v);
    if constexpr (Width >=  2) v = dpp_row_xmask_add<1>(v);
    return v;
}

// Block (256-thread) reduction: combine wave reduction + LDS exchange.
// Variant SHFL: __shfl_down within wave + LDS exchange.
__device__ __forceinline__ float block_sum_shfl(float v, float* shared) {
    v = wave32_sum_shfl(v);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += shared[i];
        shared[0] = s;
    }
    __syncthreads();
    return shared[0];
}

// Variant DPP: DPP+permlanex16 wave reduction + LDS exchange.
__device__ __forceinline__ float block_sum_dpp(float v, float* shared) {
    v = wave32_sum_dpp(v);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += shared[i];
        shared[0] = s;
    }
    __syncthreads();
    return shared[0];
}

// Variant TREE-LDS: pure shared-memory tree reduction. No shfl/permlane.
// Matches tree_reduce_sum_256 in megakernel_ops.hip.
__device__ __forceinline__ float block_sum_tree_lds(float v, float* shared) {
    shared[threadIdx.x] = v;
    __syncthreads();
    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// ============================================================================
// Cycle-count helpers (s_memrealtime returns 64-bit ticks)
// ============================================================================

__device__ __forceinline__ uint64_t rdtimer() {
    return (uint64_t)wall_clock64();
}

// ============================================================================
// Benchmark kernels
// ============================================================================

// Each block runs ITER iterations of a reduction primitive, accumulating into
// a value that depends on input -> prevents the optimizer from removing it.
// Output: per-block cycle count in cycles[blockIdx.x], result in result[blockIdx.x].

// To prevent constant folding across iterations, we permute the inputs each iter
// using a simple per-thread state.

template<int ITER>
__global__ void bench_wave_shfl(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    __shared__ float sink_shared;
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = wave32_sum_shfl(x);
        acc += x;
        // Force-broadcast acc to all lanes via shared so the compiler can't hoist
        if (i == ITER - 1 && (tid & 31) == 0) sink_shared = acc;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ void bench_wave_dpp(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    __shared__ float sink_shared;
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = wave32_sum_dpp(x);
        acc += x;
        if (i == ITER - 1 && (tid & 31) == 0) sink_shared = acc;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

// Subwave benchmarks: launch with blockDim.x = WIDTH, gridDim.x = many.
// Run ITER iterations of the subwave reduction.
template<int WIDTH, int ITER>
__global__ void bench_sub_shfl(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = subwave_sum_shfl(x, WIDTH);
        acc += x;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int WIDTH, int ITER>
__global__ void bench_sub_dpp(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = subwave_sum_dpp<WIDTH>(x);
        acc += x;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_block_shfl(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float shared[8];
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = block_sum_shfl(x, shared);
        if (tid == 0) acc += x;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_block_dpp(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float shared[8];
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = block_sum_dpp(x, shared);
        if (tid == 0) acc += x;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_block_tree(const float* __restrict__ in, float* result, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float shared[256];
    float v = in[tid];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        float x = v + (float)i;
        x = block_sum_tree_lds(x, shared);
        if (tid == 0) acc += x;
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

// ============================================================================
// Numerical-error kernels
// ============================================================================
//
// Each block reduces 32 (wave) or 256 (block) input elements via the chosen
// primitive. Compare against host-side Kahan-f64.

__global__ void err_wave_shfl(const float* in, float* out) {
    int tid = threadIdx.x;
    float v = in[blockIdx.x * 32 + tid];
    v = wave32_sum_shfl(v);
    if (tid == 0) out[blockIdx.x] = v;
}

__global__ void err_wave_dpp(const float* in, float* out) {
    int tid = threadIdx.x;
    float v = in[blockIdx.x * 32 + tid];
    v = wave32_sum_dpp(v);
    if (tid == 0) out[blockIdx.x] = v;
}

__global__ __launch_bounds__(256) void err_block_shfl(const float* in, float* out) {
    int tid = threadIdx.x;
    __shared__ float shared[8];
    float v = in[blockIdx.x * 256 + tid];
    v = block_sum_shfl(v, shared);
    if (tid == 0) out[blockIdx.x] = v;
}

__global__ __launch_bounds__(256) void err_block_dpp(const float* in, float* out) {
    int tid = threadIdx.x;
    __shared__ float shared[8];
    float v = in[blockIdx.x * 256 + tid];
    v = block_sum_dpp(v, shared);
    if (tid == 0) out[blockIdx.x] = v;
}

__global__ __launch_bounds__(256) void err_block_tree(const float* in, float* out) {
    int tid = threadIdx.x;
    __shared__ float shared[256];
    float v = in[blockIdx.x * 256 + tid];
    v = block_sum_tree_lds(v, shared);
    if (tid == 0) out[blockIdx.x] = v;
}

// ============================================================================
// Host driver
// ============================================================================

struct BenchResult {
    const char* name;
    int width;
    int blocks;
    int iters;
    double avg_cycles_per_reduction;
    double median_cycles_per_reduction;
    float result_sum;
};

// Run a benchmark kernel many times and report mean/median cycles per reduction.
template<typename Kernel>
BenchResult run_bench(const char* name, Kernel kernel, int blocks, int threads, int iters,
                      const float* d_in, float* d_result, uint64_t* d_cycles) {
    // Warmup
    kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(d_in, d_result, d_cycles);
    CHECK(hipDeviceSynchronize());
    // Real run
    kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(d_in, d_result, d_cycles);
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cycles(blocks);
    std::vector<float>    res(blocks);
    CHECK(hipMemcpy(cycles.data(), d_cycles, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(res.data(), d_result,    blocks * sizeof(float),    hipMemcpyDeviceToHost));

    double sum = 0;
    for (auto c : cycles) sum += (double)c;
    std::sort(cycles.begin(), cycles.end());
    double median = (double)cycles[cycles.size() / 2];

    BenchResult r;
    r.name = name;
    r.width = threads;
    r.blocks = blocks;
    r.iters = iters;
    r.avg_cycles_per_reduction    = sum / blocks / iters;
    r.median_cycles_per_reduction = median / iters;
    r.result_sum = res[0];
    return r;
}

// ============================================================================
// Numerical-error analysis (host)
// ============================================================================

// Kahan summation over a vector
double kahan_sum(const std::vector<float>& v) {
    double s = 0.0, c = 0.0;
    for (float x : v) {
        double y = (double)x - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

struct ErrStats {
    double max_abs;
    double max_rel;
    double mean_rel;
    double p99_rel;
};

ErrStats analyze_errors(const std::vector<float>& gpu, const std::vector<double>& ref) {
    int N = (int)gpu.size();
    std::vector<double> rels; rels.reserve(N);
    double max_abs = 0, max_rel = 0, sum_rel = 0;
    for (int i = 0; i < N; i++) {
        double g = (double)gpu[i];
        double r = ref[i];
        double a = std::fabs(g - r);
        double rel = a / (std::fabs(r) + 1e-30);
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, rel);
        sum_rel += rel;
        rels.push_back(rel);
    }
    std::sort(rels.begin(), rels.end());
    return {max_abs, max_rel, sum_rel / N, rels[(int)(N * 0.99)]};
}

void print_err(const char* name, const ErrStats& e) {
    printf("  %-22s  max_abs=%.3e  max_rel=%.3e  p99_rel=%.3e  mean_rel=%.3e\n",
           name, e.max_abs, e.max_rel, e.p99_rel, e.mean_rel);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s   warpSize=%d   gcnArch=%s\n",
           prop.name, prop.warpSize, prop.gcnArchName);

    // ---- Performance benchmarks ----
    constexpr int ITER  = 4096;
    constexpr int BLOCKS = 64;

    // Block-level kernels: blockDim.x = 256
    constexpr int BLK = 256;

    std::vector<float> hin(BLK);
    for (int i = 0; i < BLK; i++) hin[i] = (float)(i % 17) * 0.137f + 0.5f;

    float* d_in = nullptr;
    float* d_result = nullptr;
    uint64_t* d_cycles = nullptr;
    CHECK(hipMalloc(&d_in,     BLK * sizeof(float)));
    CHECK(hipMalloc(&d_result, BLOCKS * sizeof(float)));
    CHECK(hipMalloc(&d_cycles, BLOCKS * sizeof(uint64_t)));
    CHECK(hipMemcpy(d_in, hin.data(), BLK * sizeof(float), hipMemcpyHostToDevice));

    printf("\n=== Performance: cycles per reduction (in-kernel s_memrealtime ticks) ===\n");
    printf("Iterations per block: %d, blocks per launch: %d (median over blocks)\n", ITER, BLOCKS);

    std::vector<BenchResult> results;

    // Wave32 (blockDim=32)
    results.push_back(run_bench("wave32 SHFL",     bench_wave_shfl<ITER>,     BLOCKS, 32,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("wave32 DPP+P16",  bench_wave_dpp<ITER>,      BLOCKS, 32,  ITER, d_in, d_result, d_cycles));

    // Subwave widths {2,4,8,16}: blockDim = WIDTH
    results.push_back(run_bench("subwave_2 SHFL",  bench_sub_shfl<2,  ITER>,  BLOCKS,  2,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_2 DPP",   bench_sub_dpp<2,   ITER>,  BLOCKS,  2,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_4 SHFL",  bench_sub_shfl<4,  ITER>,  BLOCKS,  4,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_4 DPP",   bench_sub_dpp<4,   ITER>,  BLOCKS,  4,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_8 SHFL",  bench_sub_shfl<8,  ITER>,  BLOCKS,  8,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_8 DPP",   bench_sub_dpp<8,   ITER>,  BLOCKS,  8,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_16 SHFL", bench_sub_shfl<16, ITER>,  BLOCKS, 16,  ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("subwave_16 DPP",  bench_sub_dpp<16,  ITER>,  BLOCKS, 16,  ITER, d_in, d_result, d_cycles));

    // Block-level (blockDim=256)
    results.push_back(run_bench("block256 SHFL",     bench_block_shfl<ITER>, BLOCKS, 256, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("block256 DPP+P16",  bench_block_dpp<ITER>,  BLOCKS, 256, ITER, d_in, d_result, d_cycles));
    results.push_back(run_bench("block256 TREE-LDS", bench_block_tree<ITER>, BLOCKS, 256, ITER, d_in, d_result, d_cycles));

    printf("\n  %-22s  %-8s  %-12s  %-12s\n", "primitive", "threads", "median_cyc", "mean_cyc");
    printf("  %-22s  %-8s  %-12s  %-12s\n", "----------------------", "-------", "------------", "------------");
    for (auto& r : results) {
        printf("  %-22s  %-8d  %-12.2f  %-12.2f\n",
               r.name, r.width, r.median_cycles_per_reduction, r.avg_cycles_per_reduction);
    }

    CHECK(hipFree(d_in)); CHECK(hipFree(d_result)); CHECK(hipFree(d_cycles));

    // ---- Numerical-error analysis ----
    printf("\n=== Numerical error vs Kahan-f64 (10000 random vectors) ===\n");
    constexpr int N_VEC = 10000;

    std::mt19937 rng(0xC0FFEE);
    // Several distributions: typical RMS pre-norm (small-ish), large with outliers, mixed signs.
    auto run_err_suite = [&](const char* dist_name,
                             std::function<float()> sample) {
        printf("Distribution: %s\n", dist_name);

        // 32-element wave reductions
        {
            std::vector<float> all(N_VEC * 32);
            std::vector<double> ref(N_VEC);
            for (int v = 0; v < N_VEC; v++) {
                std::vector<float> chunk(32);
                for (int i = 0; i < 32; i++) chunk[i] = sample();
                for (int i = 0; i < 32; i++) all[v * 32 + i] = chunk[i];
                ref[v] = kahan_sum(chunk);
            }
            float* dd; float* dout;
            CHECK(hipMalloc(&dd, all.size() * sizeof(float)));
            CHECK(hipMalloc(&dout, N_VEC * sizeof(float)));
            CHECK(hipMemcpy(dd, all.data(), all.size() * sizeof(float), hipMemcpyHostToDevice));

            std::vector<float> out_shfl(N_VEC), out_dpp(N_VEC);
            err_wave_shfl<<<N_VEC, 32>>>(dd, dout); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(out_shfl.data(), dout, N_VEC * sizeof(float), hipMemcpyDeviceToHost));
            err_wave_dpp <<<N_VEC, 32>>>(dd, dout); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(out_dpp.data(), dout, N_VEC * sizeof(float), hipMemcpyDeviceToHost));

            print_err("wave32 SHFL",    analyze_errors(out_shfl, ref));
            print_err("wave32 DPP+P16", analyze_errors(out_dpp,  ref));

            CHECK(hipFree(dd)); CHECK(hipFree(dout));
        }

        // 256-element block reductions
        {
            std::vector<float> all(N_VEC * 256);
            std::vector<double> ref(N_VEC);
            for (int v = 0; v < N_VEC; v++) {
                std::vector<float> chunk(256);
                for (int i = 0; i < 256; i++) chunk[i] = sample();
                for (int i = 0; i < 256; i++) all[v * 256 + i] = chunk[i];
                ref[v] = kahan_sum(chunk);
            }
            float* dd; float* dout;
            CHECK(hipMalloc(&dd, all.size() * sizeof(float)));
            CHECK(hipMalloc(&dout, N_VEC * sizeof(float)));
            CHECK(hipMemcpy(dd, all.data(), all.size() * sizeof(float), hipMemcpyHostToDevice));

            std::vector<float> out_shfl(N_VEC), out_dpp(N_VEC), out_tree(N_VEC);
            err_block_shfl<<<N_VEC, 256>>>(dd, dout); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(out_shfl.data(), dout, N_VEC * sizeof(float), hipMemcpyDeviceToHost));
            err_block_dpp <<<N_VEC, 256>>>(dd, dout); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(out_dpp.data(), dout, N_VEC * sizeof(float), hipMemcpyDeviceToHost));
            err_block_tree<<<N_VEC, 256>>>(dd, dout); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(out_tree.data(), dout, N_VEC * sizeof(float), hipMemcpyDeviceToHost));

            print_err("block256 SHFL",     analyze_errors(out_shfl, ref));
            print_err("block256 DPP+P16",  analyze_errors(out_dpp,  ref));
            print_err("block256 TREE-LDS", analyze_errors(out_tree, ref));

            CHECK(hipFree(dd)); CHECK(hipFree(dout));
        }
        printf("\n");
    };

    // RMS-norm-like input: standard normal scaled small. Squared values are non-negative.
    {
        std::normal_distribution<float> g(0.f, 1.f);
        run_err_suite("standard normal x*x (RMS-input-like, all positive)",
            [&]() { float v = g(rng); return v * v; });
    }
    // Mixed-sign input: signed dot-product-like (Q*K terms before scaling)
    {
        std::normal_distribution<float> g(0.f, 1.f);
        run_err_suite("mixed-sign N(0,1) (dot-product-like)",
            [&]() { return g(rng); });
    }
    // Heavy-tail: Cauchy-like — stresses cancellation
    {
        std::normal_distribution<float> g(0.f, 1.f);
        run_err_suite("heavy-tail x*100 outliers (1% large positive)",
            [&]() { float v = g(rng);
                    if ((rng() & 0xFF) == 0) v *= 100.f;
                    return v * v; });
    }

    return 0;
}
