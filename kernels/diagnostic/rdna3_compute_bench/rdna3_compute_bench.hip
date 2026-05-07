// Microbenchmark for kernels/rdna3_compute.h on gfx1100 (RDNA3, wave32).
//
// Measures three things that drive header design:
//
//   1. WMMA fragment-load strategy comparison
//      - "RDNA3-native pre-pack": every lane reads a full K-stripe directly
//        from global memory (lanes 0-15 and 16-31 read same address; cache
//        coalesces). This is what load_a_bf16 / load_b_bf16 emit.
//      - "CUDA-style pre-pack": load uniquely in lanes 0-15, broadcast into
//        16-31 via cross-lane permute (mimicking how someone porting from
//        CUDA mma.sync would naturally write it).
//      - "Block-staged": stage the tile into LDS first, then load fragments
//        from LDS. (Realistic shared-memory pipeline; useful baseline.)
//
//   2. global_atomic_add_f32 throughput at varying contention
//      - Uncontended:  every thread writes a unique address.
//      - 16-way:       16 threads contend on 1 address, 1 atomic per group.
//      - 256-way:      whole block contends on 1 address (worst realistic).
//
//   3. split-K-style GEMV vs block-cooperative reduction
//      - Sizes representative of braidinfer ops:
//          N=512   (small projection, e.g. q_proj for non-tied dims)
//          N=2048  (moderate FFN intermediate)
//          N=4096  (LLaMA-style ffn intermediate / lm head proxy)
//          N=248320 (Qwen3.5 LM head — the workload that motivates split-K)
//      - K=1024 fixed (matches qwen35 hidden dim).
//      - Block-coop: one block per output row, 256-thread cooperative dot.
//      - Split-K(k):  k blocks per output row, atomic accumulate.
//
// Build:
//   ./build_bench.sh
//
// Run (always via launch-gpu.py):
//   python3 scripts/launch-gpu.py --timeout 600 -- \
//     /home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/rdna3_compute_bench/rdna3_compute_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include "../../rdna3_reduce.h"
#include "../../rdna3_compute.h"

using namespace braidinfer::rdna3;

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

typedef unsigned short u16;

// f32 -> bf16 bit-truncation (round toward zero; consistent with existing kernels).
static inline u16 f32_to_bf16(float v) {
    unsigned int bits;
    std::memcpy(&bits, &v, 4);
    return (u16)(bits >> 16);
}
static inline float bf16_to_f32_host(u16 v) {
    unsigned int bits = (unsigned int)v << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

__device__ __forceinline__ uint64_t rdtimer() {
    return (uint64_t)wall_clock64();
}

// ============================================================================
// 1. WMMA fragment load strategy
// ============================================================================
//
// All three kernels compute the same C = A @ B^T with M=N=K=16, just to
// measure the per-fragment-load + WMMA cost. We loop ITER WMMAs back-to-back
// to amortize loop overhead.

constexpr int WMMA_ITER = 4096;

// Strategy A: RDNA3-native pre-pack (each lane reads its own K-stripe;
// half-wave replication is automatic via tid&15 row index).
__global__ __launch_bounds__(32, 4) void
bench_wmma_native(const u16* __restrict__ A,
                  const u16* __restrict__ B,
                  float*    __restrict__ C,
                  uint64_t* __restrict__ cycles) {
    const int tid = threadIdx.x;
    fragment_a_bf16 a = load_a_bf16(A, /*M=*/16, /*K=*/16, 0, 0);
    fragment_b_bf16 b = load_b_bf16(B, /*N=*/16, /*K=*/16, 0, 0);
    fragment_c_f32  c = fill_c_zero();

    uint64_t t0 = rdtimer();
    #pragma unroll 4
    for (int i = 0; i < WMMA_ITER; i++) {
        c = mma_sync_bf16(a, b, c);
        // Mutate `a` slightly to prevent the compiler hoisting the WMMA out
        // of the loop. XOR with a per-iteration constant lands in 1 VALU op.
        a.r[0] = a.r[0] ^ (u16)(i & 0xFF);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;

    // Stash output to keep WMMA live.
    store_c_f32_unchecked(C + blockIdx.x * 256, /*N=*/16, 0, 0, c);
}

// Strategy B: CUDA-style pre-pack. Load uniquely in lanes 0-15 and broadcast
// into 16-31 with cross-lane permute. This is what someone porting m16n16k16
// from CUDA mma.sync would write before reading the RDNA3 ISA.
__device__ __forceinline__ fragment_a_bf16
load_a_bf16_cudastyle(const u16* __restrict__ A, int K, int row_tile, int k_tile) {
    fragment_a_bf16 frag;
    const int tid = (int)threadIdx.x;
    const int lane_lo = tid & 15;
    // Only the low half loads:
    if (tid < 16) {
        const int my_row = row_tile + lane_lo;
        const u16* row_ptr = A + (long long)my_row * K + k_tile;
        #pragma unroll
        for (int i = 0; i < 16; i++) frag.r[i] = row_ptr[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) frag.r[i] = 0;
    }
    // Now broadcast lanes 0-15 -> 16-31 via permlanex16.
    // permlanex16(old=0, src=v, sel0=0, sel1=0, fi=true, bc=true): each lane
    // in low half receives the value from lane 16, each lane in high half
    // receives the value from lane 0. Used twice (once for each half-pair),
    // but here we want a different semantic: replicate lo->hi.
    // We do it elementwise on each u16 pair.
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        unsigned packed = ((unsigned)frag.r[i+1] << 16) | (unsigned)frag.r[i];
        // Broadcast lanes 0-15 to lanes 16-31. `__shfl(packed, lane & 15)`
        // does this on AMD: every lane receives packed from `lane & 15`,
        // which means lanes 16-31 receive the value held by lane (lane-16).
        unsigned recv = __shfl(packed, (int)threadIdx.x & 15);
        frag.r[i]   = (u16)(recv & 0xFFFF);
        frag.r[i+1] = (u16)(recv >> 16);
    }
    return frag;
}

__device__ __forceinline__ fragment_b_bf16
load_b_bf16_cudastyle(const u16* __restrict__ B, int K, int col_tile, int k_tile) {
    fragment_b_bf16 frag;
    const int tid = (int)threadIdx.x;
    const int lane_lo = tid & 15;
    if (tid < 16) {
        const int my_col = col_tile + lane_lo;
        const u16* col_ptr = B + (long long)my_col * K + k_tile;
        #pragma unroll
        for (int i = 0; i < 16; i++) frag.r[i] = col_ptr[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) frag.r[i] = 0;
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        unsigned packed = ((unsigned)frag.r[i+1] << 16) | (unsigned)frag.r[i];
        unsigned recv = __shfl(packed, (int)threadIdx.x & 15);
        frag.r[i]   = (u16)(recv & 0xFFFF);
        frag.r[i+1] = (u16)(recv >> 16);
    }
    return frag;
}

__global__ __launch_bounds__(32, 4) void
bench_wmma_cudastyle(const u16* __restrict__ A,
                     const u16* __restrict__ B,
                     float*    __restrict__ C,
                     uint64_t* __restrict__ cycles) {
    const int tid = threadIdx.x;
    fragment_a_bf16 a = load_a_bf16_cudastyle(A, 16, 0, 0);
    fragment_b_bf16 b = load_b_bf16_cudastyle(B, 16, 0, 0);
    fragment_c_f32  c = fill_c_zero();

    uint64_t t0 = rdtimer();
    #pragma unroll 4
    for (int i = 0; i < WMMA_ITER; i++) {
        c = mma_sync_bf16(a, b, c);
        a.r[0] = a.r[0] ^ (u16)(i & 0xFF);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    store_c_f32_unchecked(C + blockIdx.x * 256, 16, 0, 0, c);
}

// Strategy C: load through LDS staging (realistic shared-memory pipeline).
// One wave32 stages a 16x16 tile into __shared__, syncs, then reads.
__global__ __launch_bounds__(32, 4) void
bench_wmma_lds(const u16* __restrict__ A,
               const u16* __restrict__ B,
               float*    __restrict__ C,
               uint64_t* __restrict__ cycles) {
    __shared__ u16 sA[16][16];
    __shared__ u16 sB[16][16];
    const int tid = threadIdx.x;
    const int lane_lo = tid & 15;
    // Cooperative load (32 threads -> 256 elements; 8 each)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 32;
        if (idx < 256) {
            sA[idx >> 4][idx & 15] = A[idx];
            sB[idx >> 4][idx & 15] = B[idx];
        }
    }
    __syncthreads();

    fragment_a_bf16 a;
    fragment_b_bf16 b;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        a.r[i] = sA[lane_lo][i];
        b.r[i] = sB[lane_lo][i];
    }
    fragment_c_f32 c = fill_c_zero();

    uint64_t t0 = rdtimer();
    #pragma unroll 4
    for (int i = 0; i < WMMA_ITER; i++) {
        c = mma_sync_bf16(a, b, c);
        a.r[0] = a.r[0] ^ (u16)(i & 0xFF);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    store_c_f32_unchecked(C + blockIdx.x * 256, 16, 0, 0, c);
}

// ----------------------------------------------------------------------------
// Strategy A2/B2/C2: load + MMA per iteration (loads NOT hoistable).
// These are the realistic "WMMA inner loop" tests — each iteration loads a
// fresh K-tile from a different K offset, so the loader cost is on the
// critical path. K_LOOP_TILES * 16 = K stripe walked per kernel.
// ----------------------------------------------------------------------------

constexpr int WMMA_LOOP_TILES = 64;   // 1024 K elements per kernel call

__global__ __launch_bounds__(32, 4) void
bench_wmma_native_loaded(const u16* __restrict__ A,
                         const u16* __restrict__ B,
                         float*    __restrict__ C,
                         int M, int N, int K,
                         uint64_t* __restrict__ cycles) {
    const int tid = threadIdx.x;
    fragment_c_f32 c = fill_c_zero();
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int kt = 0; kt < WMMA_LOOP_TILES; kt++) {
        fragment_a_bf16 a = load_a_bf16(A, M, K, 0, kt * 16);
        fragment_b_bf16 b = load_b_bf16(B, N, K, 0, kt * 16);
        c = mma_sync_bf16(a, b, c);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    store_c_f32_unchecked(C + blockIdx.x * 256, N, 0, 0, c);
}

__global__ __launch_bounds__(32, 4) void
bench_wmma_cudastyle_loaded(const u16* __restrict__ A,
                            const u16* __restrict__ B,
                            float*    __restrict__ C,
                            int M, int N, int K,
                            uint64_t* __restrict__ cycles) {
    const int tid = threadIdx.x;
    (void)M; (void)N;
    fragment_c_f32 c = fill_c_zero();
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int kt = 0; kt < WMMA_LOOP_TILES; kt++) {
        fragment_a_bf16 a = load_a_bf16_cudastyle(A, K, 0, kt * 16);
        fragment_b_bf16 b = load_b_bf16_cudastyle(B, K, 0, kt * 16);
        c = mma_sync_bf16(a, b, c);
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    store_c_f32_unchecked(C + blockIdx.x * 256, N, 0, 0, c);
}

__global__ __launch_bounds__(32, 4) void
bench_wmma_lds_loaded(const u16* __restrict__ A,
                      const u16* __restrict__ B,
                      float*    __restrict__ C,
                      int M, int N, int K,
                      uint64_t* __restrict__ cycles) {
    __shared__ u16 sA[16][16];
    __shared__ u16 sB[16][16];
    const int tid = threadIdx.x;
    const int lane_lo = tid & 15;
    (void)M; (void)N;
    fragment_c_f32 c = fill_c_zero();
    uint64_t t0 = rdtimer();
    #pragma unroll 1
    for (int kt = 0; kt < WMMA_LOOP_TILES; kt++) {
        // Cooperative load 16x16 A and 16x16 B into shared
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 32;
            if (idx < 256) {
                int row = idx >> 4;
                int col = idx & 15;
                sA[row][col] = A[(long long)row * K + kt * 16 + col];
                sB[row][col] = B[(long long)row * K + kt * 16 + col];
            }
        }
        __syncthreads();
        fragment_a_bf16 a;
        fragment_b_bf16 b;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a.r[i] = sA[lane_lo][i];
            b.r[i] = sB[lane_lo][i];
        }
        c = mma_sync_bf16(a, b, c);
        __syncthreads();
    }
    uint64_t t1 = rdtimer();
    if (tid == 0) cycles[blockIdx.x] = t1 - t0;
    store_c_f32_unchecked(C + blockIdx.x * 256, N, 0, 0, c);
}

// ============================================================================
// 2. atomic_add_f32 throughput
// ============================================================================
//
// Each block runs ITER atomic adds. Three contention regimes:
//   uncontended : thread t in block b -> address blockIdx.x * blockDim.x + t
//   16-way      : thread t in block b -> address blockIdx.x * 16 + (t & 15)
//   256-way     : thread t in block b -> address blockIdx.x
//
// CAS-emulated (atomicAdd default) vs hardware (unsafeAtomicAdd) are both
// measured because the default lowers to a CAS loop (verified by objdump).

constexpr int ATOMIC_ITER = 4096;

#define MAKE_ATOMIC_BENCH(NAME, FN, IDX_EXPR)                                   \
__global__ __launch_bounds__(256) void                                          \
NAME(float* __restrict__ y, uint64_t* cycles) {                                 \
    int tid = threadIdx.x;                                                      \
    int bid = blockIdx.x;                                                       \
    int idx = (IDX_EXPR);                                                       \
    uint64_t t0 = rdtimer();                                                    \
    _Pragma("unroll 4")                                                          \
    for (int i = 0; i < ATOMIC_ITER; i++) {                                     \
        FN(&y[idx], 1.f + (float)(i & 0x3F));                                   \
    }                                                                            \
    uint64_t t1 = rdtimer();                                                    \
    if (tid == 0) cycles[bid] = t1 - t0;                                        \
}

// CAS-loop variants (default atomicAdd)
MAKE_ATOMIC_BENCH(bench_atomic_uncontended_cas,    atomicAdd,       bid * 256 + tid)
MAKE_ATOMIC_BENCH(bench_atomic_16way_cas,          atomicAdd,       bid * 16 + (tid & 15))
MAKE_ATOMIC_BENCH(bench_atomic_256way_cas,         atomicAdd,       bid)

// Hardware-add variants (unsafeAtomicAdd -> global_atomic_add_f32)
MAKE_ATOMIC_BENCH(bench_atomic_uncontended_hw,     unsafeAtomicAdd, bid * 256 + tid)
MAKE_ATOMIC_BENCH(bench_atomic_16way_hw,           unsafeAtomicAdd, bid * 16 + (tid & 15))
MAKE_ATOMIC_BENCH(bench_atomic_256way_hw,          unsafeAtomicAdd, bid)
#undef MAKE_ATOMIC_BENCH

// ============================================================================
// 3. GEMV: split-K atomic vs block-coop reduction
// ============================================================================
//
// y[n] = sum over k in [0, K) of A[n, k] * x[k]
//
// Variant 1: block-coop. One block per row n. 256 threads cooperate on the
// dot product, reduce via block_reduce_sum_256, thread 0 writes y[n].

__global__ __launch_bounds__(256) void
gemv_blockcoop(const float* __restrict__ A,
               const float* __restrict__ x,
               float* __restrict__ y,
               int N, int K) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;
    __shared__ float scratch[8];
    float acc = 0.f;
    #pragma unroll 4
    for (int k = tid; k < K; k += 256) {
        acc += A[(long long)n * K + k] * x[k];
    }
    float s = block_reduce_sum_256(acc, scratch);
    if (tid == 0) y[n] = s;
}

// Variant 2: split-K with atomic. Grid = (k_blocks, N). Caller must zero y.

template<int K_PER_BLOCK, bool UseHardwareAtomic>
__global__ __launch_bounds__(256) void
gemv_splitk_atomic(const float* __restrict__ A,
                   const float* __restrict__ x,
                   float* __restrict__ y,
                   int N, int K) {
    int n  = blockIdx.y;
    int kb = blockIdx.x;
    if (n >= N) return;
    int k_start = kb * K_PER_BLOCK;
    if (k_start >= K) return;
    int k_end = (k_start + K_PER_BLOCK < K) ? k_start + K_PER_BLOCK : K;

    int tid = threadIdx.x;
    __shared__ float scratch[8];
    float acc = 0.f;
    #pragma unroll 4
    for (int k = k_start + tid; k < k_end; k += 256) {
        acc += A[(long long)n * K + k] * x[k];
    }
    float s = block_reduce_sum_256(acc, scratch);
    if (tid == 0) {
        if constexpr (UseHardwareAtomic) {
            unsafeAtomicAdd(&y[n], s);
        } else {
            atomicAdd(&y[n], s);
        }
    }
}

// ============================================================================
// Host driver
// ============================================================================

struct WmmaResult {
    const char* name;
    double median_cyc_per_mma;
    float  result_sentinel;
};

WmmaResult run_wmma_bench(const char* name,
                          void (*kernel)(const u16*, const u16*, float*, uint64_t*),
                          int blocks,
                          const u16* dA, const u16* dB, float* dC, uint64_t* dCyc) {
    // warmup
    kernel<<<blocks, 32>>>(dA, dB, dC, dCyc);
    CHECK(hipDeviceSynchronize());
    kernel<<<blocks, 32>>>(dA, dB, dC, dCyc);
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cyc(blocks);
    CHECK(hipMemcpy(cyc.data(), dCyc, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(cyc.begin(), cyc.end());
    float h_sentinel;
    CHECK(hipMemcpy(&h_sentinel, dC, sizeof(float), hipMemcpyDeviceToHost));

    WmmaResult r;
    r.name = name;
    r.median_cyc_per_mma = (double)cyc[blocks / 2] / (double)WMMA_ITER;
    r.result_sentinel    = h_sentinel;
    return r;
}

struct AtomicResult {
    const char* name;
    double median_cyc_per_atomic;
};

AtomicResult run_atomic_bench(const char* name,
                              void (*kernel)(float*, uint64_t*),
                              int blocks, int threads_per_block,
                              float* dy, uint64_t* dCyc) {
    kernel<<<blocks, threads_per_block>>>(dy, dCyc);
    CHECK(hipDeviceSynchronize());
    kernel<<<blocks, threads_per_block>>>(dy, dCyc);
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> cyc(blocks);
    CHECK(hipMemcpy(cyc.data(), dCyc, blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(cyc.begin(), cyc.end());

    AtomicResult r;
    r.name = name;
    r.median_cyc_per_atomic = (double)cyc[blocks / 2] / (double)ATOMIC_ITER;
    return r;
}

double time_gemv_blockcoop(const float* dA, const float* dx, float* dy, int N, int K, int reps) {
    // warmup
    gemv_blockcoop<<<N, 256>>>(dA, dx, dy, N, K);
    CHECK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < reps; i++) {
        gemv_blockcoop<<<N, 256>>>(dA, dx, dy, N, K);
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return (double)ms / reps;
}

template<int KPB, bool HW>
double time_gemv_splitk(const float* dA, const float* dx, float* dy, int N, int K, int reps) {
    int k_blocks = (K + KPB - 1) / KPB;
    dim3 grid(k_blocks, N);
    // warmup (also includes the memset)
    CHECK(hipMemsetAsync(dy, 0, N * sizeof(float)));
    gemv_splitk_atomic<KPB, HW><<<grid, 256>>>(dA, dx, dy, N, K);
    CHECK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < reps; i++) {
        // memset must happen each iter for correctness; we time it as part of
        // the operation since real callers also need it.
        CHECK(hipMemsetAsync(dy, 0, N * sizeof(float)));
        gemv_splitk_atomic<KPB, HW><<<grid, 256>>>(dA, dx, dy, N, K);
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return (double)ms / reps;
}

// Verify split-K and block-coop agree on a small case.
void verify_gemv() {
    const int N = 64, K = 256;
    std::vector<float> hA(N * K), hx(K);
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> g(0.f, 1.f);
    for (auto& v : hA) v = g(rng);
    for (auto& v : hx) v = g(rng);

    std::vector<float> ref(N, 0.f);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            ref[n] += hA[n * K + k] * hx[k];

    float* dA; float* dx; float* dy;
    CHECK(hipMalloc(&dA, N * K * sizeof(float)));
    CHECK(hipMalloc(&dx, K * sizeof(float)));
    CHECK(hipMalloc(&dy, N * sizeof(float)));
    CHECK(hipMemcpy(dA, hA.data(), N * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, hx.data(), K * sizeof(float), hipMemcpyHostToDevice));

    std::vector<float> hy(N);
    // block-coop
    gemv_blockcoop<<<N, 256>>>(dA, dx, dy, N, K);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hy.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));
    double max_err_bc = 0;
    for (int n = 0; n < N; n++) max_err_bc = std::max(max_err_bc, (double)std::fabs(hy[n] - ref[n]));

    // split-K (4-way) hardware variant
    CHECK(hipMemset(dy, 0, N * sizeof(float)));
    dim3 grid(4, N);
    gemv_splitk_atomic<64, true><<<grid, 256>>>(dA, dx, dy, N, K);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hy.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));
    double max_err_sk_hw = 0;
    for (int n = 0; n < N; n++) max_err_sk_hw = std::max(max_err_sk_hw, (double)std::fabs(hy[n] - ref[n]));

    // split-K (4-way) CAS variant
    CHECK(hipMemset(dy, 0, N * sizeof(float)));
    gemv_splitk_atomic<64, false><<<grid, 256>>>(dA, dx, dy, N, K);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hy.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));
    double max_err_sk_cas = 0;
    for (int n = 0; n < N; n++) max_err_sk_cas = std::max(max_err_sk_cas, (double)std::fabs(hy[n] - ref[n]));

    printf("GEMV correctness check (N=%d, K=%d):\n", N, K);
    printf("  block-coop max abs err vs f64 host ref:           %.3e  %s\n",
           max_err_bc, max_err_bc < 1e-3 ? "OK" : "FAIL");
    printf("  split-K(4) hw-atomic max abs err vs f64 host ref: %.3e  %s\n",
           max_err_sk_hw, max_err_sk_hw < 1e-3 ? "OK" : "FAIL");
    printf("  split-K(4) cas-loop  max abs err vs f64 host ref: %.3e  %s\n",
           max_err_sk_cas, max_err_sk_cas < 1e-3 ? "OK" : "FAIL");
    CHECK(hipFree(dA)); CHECK(hipFree(dx)); CHECK(hipFree(dy));
}

int main() {
    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s   warpSize=%d   gcnArch=%s\n",
           prop.name, prop.warpSize, prop.gcnArchName);

    // ---- Correctness probes ----
    verify_gemv();

    // ---- 1. WMMA strategy comparison ----
    printf("\n=== 1. WMMA fragment-load strategy (cycles/MMA, M=N=K=16, %d back-to-back) ===\n",
           WMMA_ITER);

    constexpr int WMMA_BLOCKS = 64;
    std::vector<u16> hA(256), hB(256);
    std::mt19937 rng(0xDEADBEEF);
    std::normal_distribution<float> g(0.f, 0.5f);
    for (auto& v : hA) v = f32_to_bf16(g(rng));
    for (auto& v : hB) v = f32_to_bf16(g(rng));

    u16* dA; u16* dB; float* dC; uint64_t* dCyc;
    CHECK(hipMalloc(&dA, 256 * sizeof(u16)));
    CHECK(hipMalloc(&dB, 256 * sizeof(u16)));
    CHECK(hipMalloc(&dC, WMMA_BLOCKS * 256 * sizeof(float)));
    CHECK(hipMalloc(&dCyc, WMMA_BLOCKS * sizeof(uint64_t)));
    CHECK(hipMemcpy(dA, hA.data(), 256 * sizeof(u16), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB.data(), 256 * sizeof(u16), hipMemcpyHostToDevice));

    auto r_native    = run_wmma_bench("RDNA3-native pre-pack (load hoisted)", bench_wmma_native,    WMMA_BLOCKS, dA, dB, dC, dCyc);
    auto r_cudastyle = run_wmma_bench("CUDA-style pre-pack   (load hoisted)", bench_wmma_cudastyle, WMMA_BLOCKS, dA, dB, dC, dCyc);
    auto r_lds       = run_wmma_bench("LDS-staged pre-pack   (load hoisted)", bench_wmma_lds,       WMMA_BLOCKS, dA, dB, dC, dCyc);

    printf("  %-40s  %-16s  %s\n", "strategy", "median_cyc/MMA", "ratio_to_native");
    printf("  %-40s  %-16s  %s\n", "--------", "--------------", "---------------");
    auto print_wmma = [&](const WmmaResult& r) {
        printf("  %-40s  %-16.2f  %.2fx\n", r.name, r.median_cyc_per_mma,
               r.median_cyc_per_mma / r_native.median_cyc_per_mma);
    };
    print_wmma(r_native);
    print_wmma(r_cudastyle);
    print_wmma(r_lds);
    printf("  (sentinel C[0,0] native=%.3f cuda=%.3f lds=%.3f — should agree)\n",
           r_native.result_sentinel, r_cudastyle.result_sentinel, r_lds.result_sentinel);

    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC)); CHECK(hipFree(dCyc));

    // ---- 1b. Loaded WMMA strategies (loads NOT hoistable) ----
    printf("\n=== 1b. WMMA fragment-load strategy (load+MMA per iter, %d K-tiles) ===\n",
           WMMA_LOOP_TILES);
    constexpr int WMMA_M = 16, WMMA_N = 16;
    int WMMA_K = WMMA_LOOP_TILES * 16;
    std::vector<u16> hA2((size_t)WMMA_M * WMMA_K), hB2((size_t)WMMA_N * WMMA_K);
    for (auto& v : hA2) v = f32_to_bf16(g(rng) * 0.25f);
    for (auto& v : hB2) v = f32_to_bf16(g(rng) * 0.25f);

    u16* dA2; u16* dB2; float* dC2; uint64_t* dCyc3;
    CHECK(hipMalloc(&dA2, hA2.size() * sizeof(u16)));
    CHECK(hipMalloc(&dB2, hB2.size() * sizeof(u16)));
    CHECK(hipMalloc(&dC2, WMMA_BLOCKS * 256 * sizeof(float)));
    CHECK(hipMalloc(&dCyc3, WMMA_BLOCKS * sizeof(uint64_t)));
    CHECK(hipMemcpy(dA2, hA2.data(), hA2.size() * sizeof(u16), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB2, hB2.data(), hB2.size() * sizeof(u16), hipMemcpyHostToDevice));

    auto run_loaded = [&](const char* name,
                          void (*kernel)(const u16*, const u16*, float*, int, int, int, uint64_t*)) {
        kernel<<<WMMA_BLOCKS, 32>>>(dA2, dB2, dC2, WMMA_M, WMMA_N, WMMA_K, dCyc3);
        CHECK(hipDeviceSynchronize());
        kernel<<<WMMA_BLOCKS, 32>>>(dA2, dB2, dC2, WMMA_M, WMMA_N, WMMA_K, dCyc3);
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> cyc(WMMA_BLOCKS);
        CHECK(hipMemcpy(cyc.data(), dCyc3, WMMA_BLOCKS * sizeof(uint64_t), hipMemcpyDeviceToHost));
        std::sort(cyc.begin(), cyc.end());
        double cyc_per_mma = (double)cyc[WMMA_BLOCKS / 2] / (double)WMMA_LOOP_TILES;
        return cyc_per_mma;
    };

    double cyc_native_l    = run_loaded("native",    bench_wmma_native_loaded);
    double cyc_cudastyle_l = run_loaded("cudastyle", bench_wmma_cudastyle_loaded);
    double cyc_lds_l       = run_loaded("lds",       bench_wmma_lds_loaded);

    printf("  %-40s  %-22s  %s\n", "strategy (load+MMA per iter)", "median_cyc/MMA", "ratio_to_native");
    printf("  %-40s  %-22s  %s\n", "----------------------------", "--------------", "---------------");
    printf("  %-40s  %-22.2f  %.2fx\n", "RDNA3-native pre-pack",  cyc_native_l,    1.0);
    printf("  %-40s  %-22.2f  %.2fx\n", "CUDA-style pre-pack",    cyc_cudastyle_l, cyc_cudastyle_l / cyc_native_l);
    printf("  %-40s  %-22.2f  %.2fx\n", "LDS-staged pre-pack",    cyc_lds_l,       cyc_lds_l       / cyc_native_l);

    CHECK(hipFree(dA2)); CHECK(hipFree(dB2)); CHECK(hipFree(dC2)); CHECK(hipFree(dCyc3));

    // ---- 2. Atomic throughput ----
    printf("\n=== 2. atomic_add_f32 throughput (cycles/atomic, %d adds/block) ===\n",
           ATOMIC_ITER);
    constexpr int ATOMIC_BLOCKS = 64;

    float* dy; uint64_t* dCyc2;
    CHECK(hipMalloc(&dy,   ATOMIC_BLOCKS * 256 * sizeof(float)));  // enough for uncontended
    CHECK(hipMalloc(&dCyc2, ATOMIC_BLOCKS * sizeof(uint64_t)));
    CHECK(hipMemset(dy, 0, ATOMIC_BLOCKS * 256 * sizeof(float)));

    auto a_unc_cas = run_atomic_bench("CAS  uncontended",  bench_atomic_uncontended_cas, ATOMIC_BLOCKS, 256, dy, dCyc2);
    auto a_16_cas  = run_atomic_bench("CAS  16-way",       bench_atomic_16way_cas,       ATOMIC_BLOCKS, 256, dy, dCyc2);
    auto a_256_cas = run_atomic_bench("CAS  256-way",      bench_atomic_256way_cas,      ATOMIC_BLOCKS, 256, dy, dCyc2);
    auto a_unc_hw  = run_atomic_bench("HW   uncontended",  bench_atomic_uncontended_hw,  ATOMIC_BLOCKS, 256, dy, dCyc2);
    auto a_16_hw   = run_atomic_bench("HW   16-way",       bench_atomic_16way_hw,        ATOMIC_BLOCKS, 256, dy, dCyc2);
    auto a_256_hw  = run_atomic_bench("HW   256-way",      bench_atomic_256way_hw,       ATOMIC_BLOCKS, 256, dy, dCyc2);

    printf("  %-36s  %-18s  %s\n", "regime", "median_cyc/atomic", "speedup_hw_over_cas");
    printf("  %-36s  %-18s  %s\n", "------", "-----------------", "-------------------");
    printf("  %-36s  %-18.2f  %.2fx\n", a_unc_cas.name, a_unc_cas.median_cyc_per_atomic, 1.0);
    printf("  %-36s  %-18.2f  %.2fx\n", a_unc_hw.name,  a_unc_hw.median_cyc_per_atomic,
           a_unc_cas.median_cyc_per_atomic / a_unc_hw.median_cyc_per_atomic);
    printf("  %-36s  %-18.2f  %.2fx\n", a_16_cas.name,  a_16_cas.median_cyc_per_atomic, 1.0);
    printf("  %-36s  %-18.2f  %.2fx\n", a_16_hw.name,   a_16_hw.median_cyc_per_atomic,
           a_16_cas.median_cyc_per_atomic / a_16_hw.median_cyc_per_atomic);
    printf("  %-36s  %-18.2f  %.2fx\n", a_256_cas.name, a_256_cas.median_cyc_per_atomic, 1.0);
    printf("  %-36s  %-18.2f  %.2fx\n", a_256_hw.name,  a_256_hw.median_cyc_per_atomic,
           a_256_cas.median_cyc_per_atomic / a_256_hw.median_cyc_per_atomic);

    CHECK(hipFree(dy)); CHECK(hipFree(dCyc2));

    // ---- 3. GEMV split-K vs block-coop ----
    printf("\n=== 3. GEMV: split-K-atomic vs block-cooperative reduction ===\n");
    printf("    K=1024 fixed (qwen35 hidden); reps=20 each; ms median over reps.\n");

    const int K = 1024;
    int Ns[] = {512, 2048, 4096, 248320};

    for (int N : Ns) {
        size_t a_bytes = (size_t)N * K * sizeof(float);
        if (a_bytes > 4ull * 1024ull * 1024ull * 1024ull) {
            printf("\n  N=%d skipped (allocation %.2f GB exceeds 4 GB safety limit)\n",
                   N, a_bytes / 1e9);
            continue;
        }
        std::vector<float> hAv(N * K);
        std::vector<float> hxv(K);
        for (auto& v : hAv) v = g(rng) * 0.1f;
        for (auto& v : hxv) v = g(rng);

        float* dAA; float* dxx; float* dyy;
        CHECK(hipMalloc(&dAA, a_bytes));
        CHECK(hipMalloc(&dxx, K * sizeof(float)));
        CHECK(hipMalloc(&dyy, N * sizeof(float)));
        CHECK(hipMemcpy(dAA, hAv.data(), a_bytes, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dxx, hxv.data(), K * sizeof(float), hipMemcpyHostToDevice));

        const int reps = (N >= 100000) ? 5 : 20;

        double ms_bc       = time_gemv_blockcoop(dAA, dxx, dyy, N, K, reps);
        double ms_sk2_hw   = time_gemv_splitk<512, true >(dAA, dxx, dyy, N, K, reps);
        double ms_sk4_hw   = time_gemv_splitk<256, true >(dAA, dxx, dyy, N, K, reps);
        double ms_sk8_hw   = time_gemv_splitk<128, true >(dAA, dxx, dyy, N, K, reps);
        double ms_sk16_hw  = time_gemv_splitk<64,  true >(dAA, dxx, dyy, N, K, reps);
        double ms_sk4_cas  = time_gemv_splitk<256, false>(dAA, dxx, dyy, N, K, reps);

        printf("\n  N = %-7d (A = %.1f MB)\n", N, a_bytes / 1e6);
        printf("    %-32s %-10s %-10s\n", "variant", "ms/call", "speedup");
        printf("    %-32s %-10s %-10s\n", "-------", "-------", "-------");
        printf("    %-32s %-10.4f %-10s\n", "block-coop (1 k-block)",            ms_bc, "1.00x");
        printf("    %-32s %-10.4f %.2fx\n", "split-K hw  (2 k-blocks)",          ms_sk2_hw,  ms_bc / ms_sk2_hw);
        printf("    %-32s %-10.4f %.2fx\n", "split-K hw  (4 k-blocks)",          ms_sk4_hw,  ms_bc / ms_sk4_hw);
        printf("    %-32s %-10.4f %.2fx\n", "split-K hw  (8 k-blocks)",          ms_sk8_hw,  ms_bc / ms_sk8_hw);
        printf("    %-32s %-10.4f %.2fx\n", "split-K hw  (16 k-blocks)",         ms_sk16_hw, ms_bc / ms_sk16_hw);
        printf("    %-32s %-10.4f %.2fx\n", "split-K CAS (4 k-blocks; comparison)", ms_sk4_cas, ms_bc / ms_sk4_cas);

        CHECK(hipFree(dAA)); CHECK(hipFree(dxx)); CHECK(hipFree(dyy));
    }

    return 0;
}
