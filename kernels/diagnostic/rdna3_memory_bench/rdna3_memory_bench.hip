// RDNA3 memory primitive microbenchmark for gfx1100 (wave32).
//
// Measures the levers documented in kernels/rdna3_memory.h:
//   1. LDS bank-conflict patterns at tpg ∈ {2,4,8,16}
//   2. Global → LDS staging width: scalar vs vec2 vs vec4 (cp.async is NOT
//      available on gfx1100 — see rdna3_memory.h §1)
//   3. atomicAdd_f32  vs  atomicAdd_bf16_safe (CAS)  at varying contention
//   4. buffer_gl0_inv / buffer_gl1_inv cost
//
// Build:
//   ./build_bench.sh
//
// Run (always via launch-gpu.py):
//   python3 scripts/launch-gpu.py --timeout 600 -- \
//     /home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/rdna3_memory_bench/rdna3_memory_bench
//
// Cycle counts use wall_clock64() (s_memrealtime). Each block runs many iters
// and reports cyc / op (median across blocks).

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include "../../rdna3_memory.h"

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

using namespace braidinfer::rdna3;

__device__ __forceinline__ uint64_t rdtimer() {
    return (uint64_t)wall_clock64();
}

// ============================================================================
// Block 1: LDS bank-conflict scan
// ============================================================================
//
// We build LDS tiles in three layouts and measure cyc/access:
//   PACKED_1D : float[256] strided by lane → each lane reads its own slot.
//   PACKED_2D : float[ROWS][COLS] with COLS == 64 — every lane in a wave
//               accesses ROW=warp, COL=lane → all lanes hit the same bank
//               in different rows; this is the worst-case "stride 64" hit.
//   PADDED_2D : float[ROWS][COLS + 1] — same access, but +1 col padding
//               breaks the 64-bank period.
//
// Each kernel does ITER reads from LDS and stores the running sum to a sink
// to defeat the optimizer.

template<int ITER>
__global__ __launch_bounds__(256) void bench_lds_packed_1d(float* sink, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[256];
    lds[tid] = (float)tid;
    __syncthreads();
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        // Each lane reads its own slot; shifted by i to defeat reuse.
        int idx = (tid + i) & 0xFF;
        acc += lds[idx];
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    if (tid == 0) sink[blockIdx.x]   = acc;
}

// Stride-conflict test: 32 lanes in a wave access (row=lane_in_wave, col=fixed),
// where the row stride is COLS. With COLS == 32 (RDNA3 CU mode) or COLS == 64
// (RDNA3 WGP mode default) we expect every lane to hit the same bank → 32-way
// serialization (worst case for 32 banks; still 32-way for 64 banks because
// the period divides). Padded variant breaks the period with +1 col.
template<int ITER, int COLS>
__global__ __launch_bounds__(256) void bench_lds_2d_packed(float* sink, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[32][COLS];   // 32 rows × COLS cols
    // Fill: each thread writes one slot (lane spans rows in its warp)
    int row = tid & 31;
    int warp = tid >> 5;
    if (warp == 0) lds[row][0] = (float)tid;
    __syncthreads();
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        // All 32 lanes of warp 0 read [lane][i % COLS] — same column,
        // different rows, stride COLS. With COLS multiple of bank count,
        // every lane hits the same bank → serialization.
        int r = row;
        int c = i & (COLS - 1);
        acc += lds[r][c];
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    if (tid == 0) sink[blockIdx.x]   = acc;
}

template<int ITER, int COLS>
__global__ __launch_bounds__(256) void bench_lds_2d_padded(float* sink, uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[32][COLS + 1];   // +1 col padding breaks the bank period
    int row = tid & 31;
    int warp = tid >> 5;
    if (warp == 0) lds[row][0] = (float)tid;
    __syncthreads();
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        int r = row;
        int c = i & (COLS - 1);
        acc += lds[r][c];
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    if (tid == 0) sink[blockIdx.x]   = acc;
}

// tpg-style scan: groups_per_block consecutive floats, each group lane-0
// reads them. This mirrors the `extern __shared__ float shared[]` pattern
// in kernels/moe_expert_ops.h.
template<int ITER, int TPG>
__global__ __launch_bounds__(256) void bench_lds_tpg_packed(float* sink, uint64_t* cycles) {
    int tid = threadIdx.x;
    constexpr int GPB = 256 / TPG;       // groups_per_block
    __shared__ float lds[GPB];
    int g = tid / TPG;
    int lane = tid % TPG;
    if (lane == 0) lds[g] = (float)tid;
    __syncthreads();
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    if (tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < ITER; i++) {
            float s = 0.f;
            #pragma unroll
            for (int gg = 0; gg < GPB; gg++) s += lds[gg];
            acc += s + (float)i * 1e-30f;
        }
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        sink[blockIdx.x]   = acc;
    }
}

// ============================================================================
// Block 2: Global→LDS staging width (scalar vs vec2 vs vec4)
// ============================================================================
//
// gfx1100 does not have a direct VMEM→LDS DMA path (see rdna3_memory.h §1),
// so all global → LDS staging is regular `global_load_b{32,64,128}` →
// VGPR → `ds_write_b{32,64,128}`. The compiler may or may not vectorize
// this depending on alignment hints. Compare three patterns:
//   B32  : scalar float load, scalar ds_write_b32
//   B64  : float2 load (vec2), ds_write_b64
//   B128 : float4 load (vec4), ds_write_b128
// Same total bytes per kernel, different number of memory transactions.

constexpr int IN_DIM       = 2048;
constexpr int LDS_TILE     = 1024;
constexpr int N_WEIGHT_ROW = 32;

__global__ __launch_bounds__(256) void bench_load_b32(const float* __restrict__ weight,
                                                     float* result,
                                                     uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[LDS_TILE];
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    for (int row = 0; row < N_WEIGHT_ROW; row++) {
        const float* g = weight + row * IN_DIM;
        #pragma unroll 1
        for (int i = tid; i < LDS_TILE; i += 256) {
            lds[i] = g[i];
        }
        __syncthreads();
        acc += lds[tid];
        __syncthreads();
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

__global__ __launch_bounds__(256) void bench_load_b64(const float* __restrict__ weight,
                                                     float* result,
                                                     uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[LDS_TILE];
    const float2* w2 = (const float2*)weight;
    float2*       l2 = (float2*)lds;
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    for (int row = 0; row < N_WEIGHT_ROW; row++) {
        const float2* g = w2 + row * (IN_DIM / 2);
        #pragma unroll 1
        for (int i = tid; i < LDS_TILE / 2; i += 256) {
            l2[i] = g[i];
        }
        __syncthreads();
        acc += lds[tid];
        __syncthreads();
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

__global__ __launch_bounds__(256) void bench_load_b128(const float* __restrict__ weight,
                                                      float* result,
                                                      uint64_t* cycles) {
    int tid = threadIdx.x;
    __shared__ float lds[LDS_TILE];
    const float4* w4 = (const float4*)weight;
    float4*       l4 = (float4*)lds;
    uint64_t t0 = rdtimer();
    float acc = 0.f;
    for (int row = 0; row < N_WEIGHT_ROW; row++) {
        const float4* g = w4 + row * (IN_DIM / 4);
        #pragma unroll 1
        for (int i = tid; i < LDS_TILE / 4; i += 256) {
            l4[i] = g[i];
        }
        __syncthreads();
        acc += lds[tid];
        __syncthreads();
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) {
        cycles[blockIdx.x] = t1 - t0;
        result[blockIdx.x] = acc;
    }
}

// ============================================================================
// Block 3: atomicAdd contention scan (f32 HW vs bf16 software-emulated)
// ============================================================================
//
// Each thread issues ITER atomic adds to one of `n_slots` distinct addresses.
// We sweep n_slots ∈ {1, 4, 16, 64, all-distinct} to scan the contention curve.

// Default `atomicAdd(float*, float)` — lowers to global_atomic_cmpswap_b32 loop.
template<int ITER>
__global__ void bench_atomic_f32_cas(int n_slots, float* slots, uint64_t* cycles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = tid % n_slots;
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        atomic_add_f32_cas(&slots[slot], 1.0f);
    }
    uint64_t t1 = rdtimer();
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// HW path via unsafeAtomicAdd → global_atomic_add_f32.
template<int ITER>
__global__ void bench_atomic_f32_hw(int n_slots, float* slots, uint64_t* cycles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = tid % n_slots;
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        atomic_add_f32_hw(&slots[slot], 1.0f);
    }
    uint64_t t1 = rdtimer();
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

template<int ITER>
__global__ void bench_atomic_bf16(int n_slots, unsigned short* slots, uint64_t* cycles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int slot = tid % n_slots;
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        (void)atomic_add_bf16_safe(&slots[slot], 1.0f);
    }
    uint64_t t1 = rdtimer();
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

// ============================================================================
// Block 4: buffer_gl0_inv / buffer_gl1_inv cost
// ============================================================================
//
// A wave issues N invalidation ops back-to-back from lane 0. We measure
// total cycles and divide by N. This is *upper bound* (the inv has to
// complete before s_waitcnt vmcnt drains; we don't insert that here, so
// the measured cost is just the issue latency, not the round-trip).

template<int ITER>
__global__ __launch_bounds__(256) void bench_inv_baseline(uint64_t* cycles) {
    int tid = threadIdx.x;
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int i = 0; i < ITER; i++) {
        // Empty body: just measure loop overhead.
        asm volatile("" ::: "memory");
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_inv_gl0(uint64_t* cycles) {
    int tid = threadIdx.x;
    uint64_t t0 = rdtimer();
    if (tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < ITER; i++) {
            gl0_invalidate();
        }
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_inv_gl1(uint64_t* cycles) {
    int tid = threadIdx.x;
    uint64_t t0 = rdtimer();
    if (tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < ITER; i++) {
            gl1_invalidate();
        }
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
}

template<int ITER>
__global__ __launch_bounds__(256) void bench_inv_gl01(uint64_t* cycles) {
    int tid = threadIdx.x;
    uint64_t t0 = rdtimer();
    if (tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < ITER; i++) {
            gl01_invalidate();
        }
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
}

// ============================================================================
// Driver
// ============================================================================

struct Result {
    const char* name;
    int         param;     // tpg / n_slots / etc.
    double      med_cyc;
    double      mean_cyc;
};

template<typename K, typename... Args>
double run_med_cycles(K kernel, int blocks, int threads, int divisor,
                      uint64_t* d_cycles, Args... args) {
    // Warmup
    kernel<<<dim3(blocks), dim3(threads)>>>(args..., d_cycles);
    CHECK(hipDeviceSynchronize());
    // Real
    kernel<<<dim3(blocks), dim3(threads)>>>(args..., d_cycles);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> c(blocks);
    CHECK(hipMemcpy(c.data(), d_cycles, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(c.begin(), c.end());
    return (double)c[c.size()/2] / (double)divisor;
}

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s   warpSize=%d   gcnArch=%s\n",
           prop.name, prop.warpSize, prop.gcnArchName);
    printf("LDS per block: %zu   shared mem per block: %zu\n",
           (size_t)prop.sharedMemPerBlock, (size_t)prop.sharedMemPerBlock);

    constexpr int BLOCKS = 64;

    uint64_t* d_cycles = nullptr;
    float*    d_sink   = nullptr;
    CHECK(hipMalloc(&d_cycles, BLOCKS * sizeof(uint64_t)));
    CHECK(hipMalloc(&d_sink,   BLOCKS * sizeof(float)));

    // --------------------------------------------------------------
    // Block 1: LDS bank-conflict scan
    // --------------------------------------------------------------
    {
        printf("\n=== LDS bank-conflict scan (cyc/access, median across %d blocks) ===\n", BLOCKS);
        constexpr int ITER = 4096;

        printf("  %-32s  %-10s\n", "pattern", "cyc/acc");
        printf("  %-32s  %-10s\n", "--------------------------------", "----------");

        // PACKED 1D — each lane reads its own slot
        {
            double m = run_med_cycles(bench_lds_packed_1d<ITER>, BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "packed_1d  shared[256]", m);
        }
        // PACKED 2D, COLS=32 (no bank conflict expected: row-major COLS=32 fits 32 banks)
        {
            double m = run_med_cycles(bench_lds_2d_packed<ITER, 32>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "packed_2d  [8][32]", m);
        }
        // PACKED 2D, COLS=64 (worst case: stride matches bank period exactly)
        {
            double m = run_med_cycles(bench_lds_2d_packed<ITER, 64>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "packed_2d  [8][64]  (bank-hit)", m);
        }
        // PACKED 2D, COLS=128 (still hits 64-bank period since 128 mod 64 = 0)
        {
            double m = run_med_cycles(bench_lds_2d_packed<ITER, 128>, BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "packed_2d  [8][128] (bank-hit)", m);
        }
        // PADDED variants: +1 col breaks the period
        {
            double m = run_med_cycles(bench_lds_2d_padded<ITER, 32>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "padded_2d  [8][32+1]", m);
        }
        {
            double m = run_med_cycles(bench_lds_2d_padded<ITER, 64>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "padded_2d  [8][64+1]", m);
        }
        {
            double m = run_med_cycles(bench_lds_2d_padded<ITER, 128>, BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "padded_2d  [8][128+1]", m);
        }

        // tpg patterns (mirror moe_expert_ops.h)
        {
            double m = run_med_cycles(bench_lds_tpg_packed<ITER, 2>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "tpg=2   gpb=128 (1 thread reads)", m);
        }
        {
            double m = run_med_cycles(bench_lds_tpg_packed<ITER, 4>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "tpg=4   gpb=64  (1 thread reads)", m);
        }
        {
            double m = run_med_cycles(bench_lds_tpg_packed<ITER, 8>,  BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "tpg=8   gpb=32  (1 thread reads)", m);
        }
        {
            double m = run_med_cycles(bench_lds_tpg_packed<ITER, 16>, BLOCKS, 256, ITER, d_cycles, d_sink);
            printf("  %-32s  %-10.3f\n", "tpg=16  gpb=16  (1 thread reads)", m);
        }
    }

    // --------------------------------------------------------------
    // Block 2: global→LDS staging width (b32 vs b64 vs b128)
    // --------------------------------------------------------------
    {
        printf("\n=== Global→LDS staging width (regular load + ds_write) ===\n");
        printf("  IN_DIM=%d  LDS_TILE=%d  N_WEIGHT_ROW=%d\n",
               IN_DIM, LDS_TILE, N_WEIGHT_ROW);

        size_t W_FLOATS = (size_t)IN_DIM * (size_t)N_WEIGHT_ROW;
        float* d_w = nullptr;
        CHECK(hipMalloc(&d_w, W_FLOATS * sizeof(float)));
        std::vector<float> h_w(W_FLOATS);
        for (size_t i = 0; i < W_FLOATS; i++) h_w[i] = (float)((i * 1664525u + 1013904223u) & 0xFF) / 256.f;
        CHECK(hipMemcpy(d_w, h_w.data(), W_FLOATS * sizeof(float), hipMemcpyHostToDevice));

        printf("  %-32s  %-12s  %-12s\n", "variant", "cyc/elem", "rel_speed");
        printf("  %-32s  %-12s  %-12s\n", "--------------------------------",
               "------------", "------------");
        const int N_ELEMS = LDS_TILE * N_WEIGHT_ROW;

        double b32_cyc = run_med_cycles(bench_load_b32, BLOCKS, 256, N_ELEMS, d_cycles, d_w, d_sink);
        printf("  %-32s  %-12.3f  %-12s\n", "b32  scalar load + ds_write_b32", b32_cyc, "1.00x");

        double b64_cyc = run_med_cycles(bench_load_b64, BLOCKS, 256, N_ELEMS, d_cycles, d_w, d_sink);
        printf("  %-32s  %-12.3f  %.2fx\n", "b64  vec2 load + ds_write_b64", b64_cyc, b32_cyc / b64_cyc);

        double b128_cyc = run_med_cycles(bench_load_b128, BLOCKS, 256, N_ELEMS, d_cycles, d_w, d_sink);
        printf("  %-32s  %-12.3f  %.2fx\n", "b128 vec4 load + ds_write_b128", b128_cyc, b32_cyc / b128_cyc);

        CHECK(hipFree(d_w));
    }

    // --------------------------------------------------------------
    // Block 3: atomicAdd contention
    // --------------------------------------------------------------
    {
        printf("\n=== atomicAdd contention (f32 HW vs bf16 CAS-emulated) ===\n");
        constexpr int ITER = 4096;
        constexpr int N_BLK = 8;
        constexpr int N_THR = 256;
        constexpr int TOTAL = N_BLK * N_THR;

        // Allocate enough slots for the all-distinct case.
        float* d_f32 = nullptr;
        unsigned short* d_bf16 = nullptr;
        CHECK(hipMalloc(&d_f32,  TOTAL * sizeof(float)));
        CHECK(hipMalloc(&d_bf16, TOTAL * sizeof(unsigned short)));

        printf("  %-12s  %-12s  %-12s  %-12s  %-10s\n",
               "n_slots", "f32_cas cyc", "f32_hw cyc", "bf16 cyc", "bf16/hw");
        printf("  %-12s  %-12s  %-12s  %-12s  %-10s\n",
               "------------", "------------", "------------", "------------", "----------");

        int slots_to_test[] = {1, 4, 16, 64, TOTAL};
        const char* names[]  = {"1 (hot)", "4", "16", "64", "all-distinct"};
        for (int s = 0; s < 5; s++) {
            int n = slots_to_test[s];
            CHECK(hipMemset(d_f32,  0, TOTAL * sizeof(float)));
            double cas_cyc = run_med_cycles(bench_atomic_f32_cas<ITER>, N_BLK, N_THR, ITER, d_cycles, n, d_f32);
            CHECK(hipMemset(d_f32,  0, TOTAL * sizeof(float)));
            double hw_cyc  = run_med_cycles(bench_atomic_f32_hw<ITER>,  N_BLK, N_THR, ITER, d_cycles, n, d_f32);
            CHECK(hipMemset(d_bf16, 0, TOTAL * sizeof(unsigned short)));
            double bf16_cyc = run_med_cycles(bench_atomic_bf16<ITER>,   N_BLK, N_THR, ITER, d_cycles, n, d_bf16);
            printf("  %-12s  %-12.2f  %-12.2f  %-12.2f  %-10.2fx\n",
                   names[s], cas_cyc, hw_cyc, bf16_cyc, bf16_cyc / hw_cyc);
        }

        CHECK(hipFree(d_f32)); CHECK(hipFree(d_bf16));
    }

    // --------------------------------------------------------------
    // Block 4: buffer_gl0_inv / buffer_gl1_inv cost
    // --------------------------------------------------------------
    {
        printf("\n=== Cache invalidation cost (cyc / inv, lane-0 issue, 1-block) ===\n");
        constexpr int ITER = 1024;

        printf("  %-32s  %-12s\n", "variant", "cyc/inv");
        printf("  %-32s  %-12s\n", "--------------------------------", "------------");

        double base = run_med_cycles(bench_inv_baseline<ITER>, BLOCKS, 256, ITER, d_cycles);
        printf("  %-32s  %-12.3f  (loop-overhead floor)\n", "baseline", base);

        double gl0 = run_med_cycles(bench_inv_gl0<ITER>, BLOCKS, 256, ITER, d_cycles);
        printf("  %-32s  %-12.3f\n", "buffer_gl0_inv", gl0);

        double gl1 = run_med_cycles(bench_inv_gl1<ITER>, BLOCKS, 256, ITER, d_cycles);
        printf("  %-32s  %-12.3f\n", "buffer_gl1_inv", gl1);

        double gl01 = run_med_cycles(bench_inv_gl01<ITER>, BLOCKS, 256, ITER, d_cycles);
        printf("  %-32s  %-12.3f\n", "buffer_gl0_inv + buffer_gl1_inv", gl01);

        double clk = (double)prop.clockRate * 1e3; // kHz → Hz; clockRate is in kHz
        printf("  (gpu clock %.0f MHz; 1 cycle ≈ %.2f ns)\n", clk * 1e-6, 1e9 / clk);
    }

    CHECK(hipFree(d_cycles));
    CHECK(hipFree(d_sink));
    printf("\nDone.\n");
    return 0;
}
