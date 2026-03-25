#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

__device__ __forceinline__ float bf16_to_f32(unsigned short val) {
    unsigned int bits = ((unsigned int)val) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

// Baseline: 128-bit vectorized loads (no cache hints)
extern "C" __global__ void gemv_vec128(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim, const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= out_dim) return;

    const unsigned short* w = weight + (long long)row * in_dim;
    float acc = 0.0f;

    const int vec_count = in_dim / 8;
    const ulong2* w_vec = (const ulong2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ulong2 wv = w_vec[vi];
        unsigned short ws[8];
        __builtin_memcpy(ws, &wv, 16);
        int base = vi * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc += bf16_to_f32(ws[j]) * input[base + j];
    }

    __shared__ float shared[256];
    shared[tid] = acc;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[row] = shared[0];
}

// Variant: nontemporal loads for weights (GLC+SLC bypass L2)
extern "C" __global__ void gemv_vec128_nt(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim, const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= out_dim) return;

    const unsigned short* w = weight + (long long)row * in_dim;
    float acc = 0.0f;

    typedef unsigned long long ull2 __attribute__((ext_vector_type(2)));
    const int vec_count = in_dim / 8;
    const ull2* w_vec = (const ull2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ull2 wv = __builtin_nontemporal_load(&w_vec[vi]);
        unsigned short ws[8];
        __builtin_memcpy(ws, &wv, 16);
        int base = vi * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc += bf16_to_f32(ws[j]) * input[base + j];
    }

    __shared__ float shared[256];
    shared[tid] = acc;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[row] = shared[0];
}

float bench_ms(hipEvent_t start, hipEvent_t stop) {
    float ms;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

void run_benchmarks(int M, int N, int warmup, int iters) {
    printf("GEMV %d x %d:\n", M, N);

    std::vector<unsigned short> h_w((long long)M * N);
    std::vector<float> h_in(N);
    for (long long i = 0; i < (long long)M * N; i++) {
        float v = (float)(rand() % 1000 - 500) / 1000.0f;
        unsigned int bits;
        memcpy(&bits, &v, 4);
        h_w[i] = (unsigned short)(bits >> 16);
    }
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000 - 500) / 1000.0f;

    unsigned short *d_w; float *d_in, *d_out;
    CHECK_HIP(hipMalloc(&d_w, (long long)M * N * 2));
    CHECK_HIP(hipMalloc(&d_in, N * 4));
    CHECK_HIP(hipMalloc(&d_out, M * 4));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), (long long)M * N * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_in, h_in.data(), N * 4, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    double total_bytes = (double)M * N * 2.0 + N * 4.0 + M * 4.0;

    // Reference
    gemv_vec128<<<M, 256>>>(d_out, d_w, d_in, M, N);
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<float> ref(M);
    CHECK_HIP(hipMemcpy(ref.data(), d_out, M * 4, hipMemcpyDeviceToHost));

    auto bench = [&](const char* name, auto fn) {
        for (int i = 0; i < warmup; i++) fn();
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipEventRecord(start));
        for (int i = 0; i < iters; i++) fn();
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float ms = bench_ms(start, stop) / iters;
        double gbps = total_bytes / (ms * 1e6);

        std::vector<float> out(M);
        CHECK_HIP(hipMemcpy(out.data(), d_out, M * 4, hipMemcpyDeviceToHost));
        float maxd = 0;
        for (int i = 0; i < M; i++) maxd = fmax(maxd, fabs(ref[i] - out[i]));
        printf("  %-24s %7.3f ms  %6.1f GB/s  (diff=%.6f)\n", name, ms, gbps, maxd);
    };

    bench("vec128 (baseline)", [&]{ gemv_vec128<<<M, 256>>>(d_out, d_w, d_in, M, N); });
    bench("vec128 + nontemporal", [&]{ gemv_vec128_nt<<<M, 256>>>(d_out, d_w, d_in, M, N); });

    printf("\n");
    CHECK_HIP(hipFree(d_w)); CHECK_HIP(hipFree(d_in)); CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipEventDestroy(start)); CHECK_HIP(hipEventDestroy(stop));
}

int main() {
    CHECK_HIP(hipSetDevice(0));
    printf("=== Cache Hint Benchmark (BF16 weights) ===\n\n");
    int warmup = 50, iters = 200;
    run_benchmarks(6144, 1024, warmup, iters);
    run_benchmarks(3584, 1024, warmup, iters);
    run_benchmarks(1024, 3584, warmup, iters);
    run_benchmarks(248320, 1024, warmup, iters);
    return 0;
}
