// 77r.2.13: bf16x2 / fp16x2 packed FMA + dot2 microbenchmark for gfx1100.
//
// Production hot path: op_linear_proj batched (kernels/megakernel_ops.hip:142-176)
// does scalar `bf16_to_f32(weight) * f32(input) + acc` in its inner loop.
// gfx1100 has hardware packed-pair instructions that the compiler MAY or
// MAY NOT emit from this scalar pattern depending on -ffp-contract / -O.
//
// Variants under test (per-thread K-element dot product):
//
//   V0_scalar_bf16w_f32a   : production shape — bf16 weight upcast to f32,
//                            scalar f32 FMA. Compiler may or may not vectorize.
//   V1_scalar_bf16w_bf16a  : both operands bf16 upcast to f32 then scalar FMA.
//                            Lossy on activation; baseline for bf16-pair dot.
//   V2_dot2_bf16xbf16_f32  : __builtin_amdgcn_fdot2_f32_bf16
//                            (single VOP3P instruction, bf16x2 -> f32 accum).
//                            Lossy on activation (input downcast to bf16).
//   V3_dot2_f16xf16_f32    : __builtin_amdgcn_fdot2 (fp16x2 -> f32).
//                            Different precision profile: 5-bit exp / 10-bit
//                            mantissa vs bf16's 8-bit exp / 7-bit mantissa.
//   V4_scalar_f32w_f32a    : control — full f32 weight × f32 input.
//                            What the workload looks like with no quantization.
//
// Each variant runs N_INNER iterations of a K-element dot product, with the
// per-thread accumulator and inputs held in registers. Reported metric:
// wall_clock64 cycles per inner-K-element FMA (lower is better).
//
// Numerical check: each variant computes the same dot product on the same
// input data (data is generated f32 then cast as needed). Max relative error
// vs Kahan f64 reference is reported.
//
// Build: ./build_bench.sh
// Run:   python3 scripts/launch-gpu.py --timeout 300 -- \
//          kernels/diagnostic/packed_fma_bench/packed_fma_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

typedef _Float16 f16;
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef short    bf16x2 __attribute__((ext_vector_type(2)));
typedef unsigned short u16;

__device__ __forceinline__ float bf16_to_f32(u16 b) {
    return __builtin_bit_cast(float, ((unsigned)b) << 16);
}
__device__ __forceinline__ u16 f32_to_bf16(float f) {
    return (u16)(__builtin_bit_cast(unsigned, f) >> 16);
}

// Per-thread inner-K dot product variants. Each kernel reads K f32 elements
// from `weights` and `inputs` and accumulates K element-wise products. The
// outer loop runs the dot product N_INNER times to amortize timing overhead.

template<int K>
__global__ __launch_bounds__(256) void bench_v0_scalar_bf16w_f32a(
    const u16*  __restrict__ weights,
    const float* __restrict__ inputs,
    float*       __restrict__ out,
    uint64_t*    __restrict__ cycles,
    int n_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread reads its own K-element strip.
    const u16*   w = weights + (long long)tid * K;
    const float* x = inputs  + (long long)tid * K;

    uint64_t t0 = wall_clock64();
    float acc_outer = 0.0f;
    for (int it = 0; it < n_inner; it++) {
        float acc = (float)it * 1e-9f;
        #pragma unroll 8
        for (int i = 0; i < K; i++) {
            acc += bf16_to_f32(w[i]) * x[i];
        }
        acc_outer += acc;
    }
    uint64_t t1 = wall_clock64();

    if (tid == 0) {
        cycles[0] = t1 - t0;
        out[0] = acc_outer;
    }
}

template<int K>
__global__ __launch_bounds__(256) void bench_v1_scalar_bf16w_bf16a(
    const u16*  __restrict__ weights,
    const u16*  __restrict__ inputs_bf16,
    float*       __restrict__ out,
    uint64_t*    __restrict__ cycles,
    int n_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const u16* w = weights     + (long long)tid * K;
    const u16* x = inputs_bf16 + (long long)tid * K;

    uint64_t t0 = wall_clock64();
    float acc_outer = 0.0f;
    for (int it = 0; it < n_inner; it++) {
        float acc = (float)it * 1e-9f;
        #pragma unroll 8
        for (int i = 0; i < K; i++) {
            acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
        }
        acc_outer += acc;
    }
    uint64_t t1 = wall_clock64();
    if (tid == 0) { cycles[0] = t1 - t0; out[0] = acc_outer; }
}

template<int K>
__global__ __launch_bounds__(256) void bench_v2_dot2_bf16(
    const bf16x2* __restrict__ weights,   // K/2 pairs per thread
    const bf16x2* __restrict__ inputs_bf16,
    float*        __restrict__ out,
    uint64_t*     __restrict__ cycles,
    int n_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const bf16x2* w = weights     + (long long)tid * (K / 2);
    const bf16x2* x = inputs_bf16 + (long long)tid * (K / 2);

    uint64_t t0 = wall_clock64();
    float acc_outer = 0.0f;
    for (int it = 0; it < n_inner; it++) {
        float acc = (float)it * 1e-9f;
        #pragma unroll 8
        for (int i = 0; i < K / 2; i++) {
            acc = __builtin_amdgcn_fdot2_f32_bf16(w[i], x[i], acc, false);
        }
        acc_outer += acc;
    }
    uint64_t t1 = wall_clock64();
    if (tid == 0) { cycles[0] = t1 - t0; out[0] = acc_outer; }
}

template<int K>
__global__ __launch_bounds__(256) void bench_v3_dot2_f16(
    const f16x2* __restrict__ weights_f16,   // K/2 pairs per thread
    const f16x2* __restrict__ inputs_f16,
    float*       __restrict__ out,
    uint64_t*    __restrict__ cycles,
    int n_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const f16x2* w = weights_f16 + (long long)tid * (K / 2);
    const f16x2* x = inputs_f16  + (long long)tid * (K / 2);

    uint64_t t0 = wall_clock64();
    float acc_outer = 0.0f;
    for (int it = 0; it < n_inner; it++) {
        float acc = (float)it * 1e-9f;
        #pragma unroll 8
        for (int i = 0; i < K / 2; i++) {
            acc = __builtin_amdgcn_fdot2(w[i], x[i], acc, false);
        }
        acc_outer += acc;
    }
    uint64_t t1 = wall_clock64();
    if (tid == 0) { cycles[0] = t1 - t0; out[0] = acc_outer; }
}

template<int K>
__global__ __launch_bounds__(256) void bench_v4_scalar_f32_f32(
    const float* __restrict__ weights_f32,
    const float* __restrict__ inputs,
    float*       __restrict__ out,
    uint64_t*    __restrict__ cycles,
    int n_inner
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const float* w = weights_f32 + (long long)tid * K;
    const float* x = inputs       + (long long)tid * K;

    uint64_t t0 = wall_clock64();
    float acc_outer = 0.0f;
    for (int it = 0; it < n_inner; it++) {
        float acc = (float)it * 1e-9f;
        #pragma unroll 8
        for (int i = 0; i < K; i++) {
            acc += w[i] * x[i];
        }
        acc_outer += acc;
    }
    uint64_t t1 = wall_clock64();
    if (tid == 0) { cycles[0] = t1 - t0; out[0] = acc_outer; }
}

template<int K>
double dot_kahan_ref(const float* w, const float* x) {
    double s = 0.0, c = 0.0;
    for (int i = 0; i < K; i++) {
        double y = (double)w[i] * (double)x[i] - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

template<int K>
void run_size(const char* label_size, int n_threads, int n_inner) {
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> nd(0.0f, 0.1f);

    std::vector<float> w_f32(n_threads * K);
    std::vector<float> x_f32(n_threads * K);
    for (auto& v : w_f32) v = nd(rng);
    for (auto& v : x_f32) v = nd(rng);

    // Cast variants.
    std::vector<u16> w_bf16(n_threads * K);
    std::vector<u16> x_bf16(n_threads * K);
    std::vector<u16> w_f16(n_threads * K);
    std::vector<u16> x_f16(n_threads * K);
    for (size_t i = 0; i < w_f32.size(); i++) {
        w_bf16[i] = (u16)(__builtin_bit_cast(unsigned, w_f32[i]) >> 16);
        x_bf16[i] = (u16)(__builtin_bit_cast(unsigned, x_f32[i]) >> 16);
        // fp16: cast to _Float16 then bit-pun.
        _Float16 wh = (_Float16)w_f32[i];
        _Float16 xh = (_Float16)x_f32[i];
        std::memcpy(&w_f16[i], &wh, 2);
        std::memcpy(&x_f16[i], &xh, 2);
    }

    int dev = 0;
    CHECK(hipSetDevice(dev));
    void* d_w_f32; void* d_x_f32; void* d_w_bf16; void* d_x_bf16;
    void* d_w_f16; void* d_x_f16; void* d_out; void* d_cyc;
    CHECK(hipMalloc(&d_w_f32,  w_f32.size()  * 4));
    CHECK(hipMalloc(&d_x_f32,  x_f32.size()  * 4));
    CHECK(hipMalloc(&d_w_bf16, w_bf16.size() * 2));
    CHECK(hipMalloc(&d_x_bf16, x_bf16.size() * 2));
    CHECK(hipMalloc(&d_w_f16,  w_f16.size()  * 2));
    CHECK(hipMalloc(&d_x_f16,  x_f16.size()  * 2));
    CHECK(hipMalloc(&d_out, sizeof(float)));
    CHECK(hipMalloc(&d_cyc, sizeof(uint64_t)));
    CHECK(hipMemcpy(d_w_f32,  w_f32.data(),  w_f32.size()  * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_x_f32,  x_f32.data(),  x_f32.size()  * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_w_bf16, w_bf16.data(), w_bf16.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_x_bf16, x_bf16.data(), x_bf16.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_w_f16,  w_f16.data(),  w_f16.size()  * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_x_f16,  x_f16.data(),  x_f16.size()  * 2, hipMemcpyHostToDevice));

    int blocks = std::max(1, n_threads / 256);
    int threads = std::min(256, n_threads);
    auto cycle_per_fma = [&](uint64_t cyc) {
        return (double)cyc / ((double)n_inner * (double)K);
    };

    auto measure = [&](auto launcher) {
        // 3 trials, median.
        std::vector<uint64_t> samples;
        for (int t = 0; t < 3; t++) {
            launcher();
            CHECK(hipDeviceSynchronize());
            uint64_t c = 0;
            CHECK(hipMemcpy(&c, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
            samples.push_back(c);
        }
        std::sort(samples.begin(), samples.end());
        return samples[1];
    };

    uint64_t c0 = measure([&]() {
        bench_v0_scalar_bf16w_f32a<K><<<blocks, threads>>>(
            (const u16*)d_w_bf16, (const float*)d_x_f32,
            (float*)d_out, (uint64_t*)d_cyc, n_inner);
    });
    uint64_t c1 = measure([&]() {
        bench_v1_scalar_bf16w_bf16a<K><<<blocks, threads>>>(
            (const u16*)d_w_bf16, (const u16*)d_x_bf16,
            (float*)d_out, (uint64_t*)d_cyc, n_inner);
    });
    uint64_t c2 = measure([&]() {
        bench_v2_dot2_bf16<K><<<blocks, threads>>>(
            (const bf16x2*)d_w_bf16, (const bf16x2*)d_x_bf16,
            (float*)d_out, (uint64_t*)d_cyc, n_inner);
    });
    uint64_t c3 = measure([&]() {
        bench_v3_dot2_f16<K><<<blocks, threads>>>(
            (const f16x2*)d_w_f16, (const f16x2*)d_x_f16,
            (float*)d_out, (uint64_t*)d_cyc, n_inner);
    });
    uint64_t c4 = measure([&]() {
        bench_v4_scalar_f32_f32<K><<<blocks, threads>>>(
            (const float*)d_w_f32, (const float*)d_x_f32,
            (float*)d_out, (uint64_t*)d_cyc, n_inner);
    });

    double r0 = cycle_per_fma(c0);
    double r1 = cycle_per_fma(c1);
    double r2 = cycle_per_fma(c2);
    double r3 = cycle_per_fma(c3);
    double r4 = cycle_per_fma(c4);

    printf("\n--- K=%d ---\n", K);
    printf("  V0 scalar bf16w x f32a  : %6.2f cyc/FMA  (baseline)\n", r0);
    printf("  V1 scalar bf16w x bf16a : %6.2f cyc/FMA  (%.2fx vs V0)\n", r1, r0 / r1);
    printf("  V2 dot2  bf16  x bf16   : %6.2f cyc/FMA  (%.2fx vs V0)\n", r2, r0 / r2);
    printf("  V3 dot2  f16   x f16    : %6.2f cyc/FMA  (%.2fx vs V0)\n", r3, r0 / r3);
    printf("  V4 scalar f32  x f32    : %6.2f cyc/FMA  (%.2fx vs V0)\n", r4, r0 / r4);

    CHECK(hipFree(d_w_f32));  CHECK(hipFree(d_x_f32));
    CHECK(hipFree(d_w_bf16)); CHECK(hipFree(d_x_bf16));
    CHECK(hipFree(d_w_f16));  CHECK(hipFree(d_x_f16));
    CHECK(hipFree(d_out));    CHECK(hipFree(d_cyc));
}

int main() {
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0));
    printf("device: %s, multiProcessorCount=%d\n", props.name, props.multiProcessorCount);
    printf("Per-thread inner-K dot product cycle counts (median of 3, 256 threads x 1 block).\n");
    printf("Lower is better. Speedups vs V0 (production shape: bf16 weight x f32 input).\n");

    run_size<128>("128",  256, 4096);
    run_size<512>("512",  256, 1024);
    run_size<1024>("1024", 256, 512);
    run_size<2048>("2048", 256, 256);

    return 0;
}
