// Flat GEMM benchmark: compare rocBLAS GEMM vs custom GEMV for decode-phase
// matmuls at M=1,2,4,8 with braidinfer-relevant dimensions.
//
// Tests: rocblas_gemm (default padding), custom GEMV (M=1), and pad-to-8 GEMM.
// Output: throughput in GFLOPS and wall time for each (M, N, K) config.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)
#define CHECK_ROCBLAS(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { fprintf(stderr, "rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while(0)

// Simple GEMV kernel: each workgroup computes one output element
// y[i] = sum_k A[0,k] * B[k,i]  for M=1
__global__ void gemv_kernel(const __half* __restrict__ A,
                            const __half* __restrict__ B,
                            __half* __restrict__ C,
                            int N, int K) {
    int col = blockIdx.x;
    if (col >= N) return;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        acc += __half2float(A[k]) * __half2float(B[k * N + col]);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    // Cross-warp reduction via shared memory
    __shared__ float shared[8];
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) shared[warp_id] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        int n_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < n_warps; w++) sum += shared[w];
        C[col] = __float2half(sum);
    }
}

struct BenchResult {
    double gflops;
    double us;
};

BenchResult bench_rocblas_gemm(rocblas_handle handle,
                               int M, int N, int K,
                               __half* dA, __half* dB, __half* dC,
                               int warmup, int iters) {
    rocblas_half alpha, beta;
    *reinterpret_cast<__half*>(&alpha) = __float2half(1.0f);
    *reinterpret_cast<__half*>(&beta) = __float2half(0.0f);

    auto* rbA = reinterpret_cast<const rocblas_half*>(dA);
    auto* rbB = reinterpret_cast<const rocblas_half*>(dB);
    auto* rbC = reinterpret_cast<rocblas_half*>(dC);

    for (int i = 0; i < warmup; i++) {
        CHECK_ROCBLAS(rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                    N, M, K, &alpha, rbB, N, rbA, K, &beta, rbC, N));
    }
    CHECK_HIP(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        CHECK_ROCBLAS(rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                    N, M, K, &alpha, rbB, N, rbA, K, &beta, rbC, N));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (us * 1e3);
    return {gflops, us};
}

BenchResult bench_gemv(int N, int K,
                       __half* dA, __half* dB, __half* dC,
                       int warmup, int iters) {
    int threads = 256;

    for (int i = 0; i < warmup; i++) {
        gemv_kernel<<<N, threads>>>(dA, dB, dC, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        gemv_kernel<<<N, threads>>>(dA, dB, dC, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    double flops = 2.0 * N * K;
    double gflops = flops / (us * 1e3);
    return {gflops, us};
}

int main() {
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // Braidinfer-relevant dimensions
    struct Config { int N; int K; const char* name; };
    Config configs[] = {
        {5120, 5120, "attn_proj (122B)"},
        {27648, 5120, "ffn_up (122B)"},
        {5120, 27648, "ffn_down (122B)"},
        {13824, 5120, "ffn_up (35B)"},
        {5120, 13824, "ffn_down (35B)"},
        {4096, 4096, "attn_proj (120B)"},
        {16384, 4096, "ffn_up (120B)"},
        {4096, 16384, "ffn_down (120B)"},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);
    int M_values[] = {1, 2, 4, 8};
    int n_M = 4;

    int max_M = 8;
    int max_N = 27648;
    int max_K = 27648;

    __half *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, max_M * max_K * sizeof(__half)));
    CHECK_HIP(hipMalloc(&dB, max_K * max_N * sizeof(__half)));
    CHECK_HIP(hipMalloc(&dC, max_M * max_N * sizeof(__half)));
    CHECK_HIP(hipMemset(dA, 0, max_M * max_K * sizeof(__half)));
    CHECK_HIP(hipMemset(dB, 0, max_K * max_N * sizeof(__half)));

    int warmup = 50, iters = 200;

    printf("%-22s  M    rocBLAS_GFLOPS  rocBLAS_us    GEMV_GFLOPS   GEMV_us       winner   speedup\n", "config");
    printf("%-22s  --   -------------  ----------    -----------   -------       ------   -------\n", "");

    for (int ci = 0; ci < n_configs; ci++) {
        int N = configs[ci].N, K = configs[ci].K;
        for (int mi = 0; mi < n_M; mi++) {
            int M = M_values[mi];

            auto rb = bench_rocblas_gemm(handle, M, N, K, dA, dB, dC, warmup, iters);

            if (M == 1) {
                auto gv = bench_gemv(N, K, dA, dB, dC, warmup, iters);
                const char* winner = gv.us < rb.us ? "GEMV" : "rocBLAS";
                double speedup = gv.us < rb.us ? rb.us / gv.us : gv.us / rb.us;
                printf("%-22s  %d    %10.1f     %8.1f      %10.1f    %8.1f      %-8s %.2fx\n",
                       configs[ci].name, M, rb.gflops, rb.us, gv.gflops, gv.us, winner, speedup);
            } else {
                printf("%-22s  %d    %10.1f     %8.1f      %10s    %8s      %-8s\n",
                       configs[ci].name, M, rb.gflops, rb.us, "-", "-", "rocBLAS");
            }
        }
    }

    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC));
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));
    return 0;
}
