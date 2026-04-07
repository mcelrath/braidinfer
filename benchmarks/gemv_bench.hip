#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "bf16_utils.h"

// Our naive BF16-weight matvec (same as linear_proj_f32)
extern "C" __global__ void naive_gemv_bf16w(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= out_dim) return;

    const unsigned short* w = weight + row * in_dim;
    float acc = 0.0f;
    for (int i = tid; i < in_dim; i += stride) {
        acc += bf16_to_f32(w[i]) * input[i];
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

#define CHECK_HIP(x) do { hipError_t e = (x); if (e) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)
#define CHECK_ROCBLAS(x) do { rocblas_status s = (x); if (s) { printf("rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while(0)

float bench_ms(hipEvent_t start, hipEvent_t stop) {
    float ms;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

void benchmark_gemv(int M, int N, int warmup, int iters) {
    printf("GEMV %d x %d (M=%d rows, N=%d cols):\n", M, N, M, N);

    // Allocate host data
    std::vector<float> h_weight_f32(M * N);
    std::vector<unsigned short> h_weight_bf16(M * N);
    std::vector<float> h_input(N);
    for (int i = 0; i < M * N; i++) {
        float v = (float)(rand() % 1000 - 500) / 1000.0f;
        h_weight_f32[i] = v;
        // f32 -> bf16 truncation
        unsigned int bits;
        memcpy(&bits, &v, 4);
        h_weight_bf16[i] = (unsigned short)(bits >> 16);
    }
    for (int i = 0; i < N; i++) h_input[i] = (float)(rand() % 1000 - 500) / 1000.0f;

    // Device allocations
    float *d_weight_f32, *d_input, *d_output_rocblas, *d_output_naive;
    unsigned short *d_weight_bf16;
    CHECK_HIP(hipMalloc(&d_weight_f32, M * N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_weight_bf16, M * N * sizeof(unsigned short)));
    CHECK_HIP(hipMalloc(&d_input, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_output_rocblas, M * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_output_naive, M * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_weight_f32, h_weight_f32.data(), M * N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_weight_bf16, h_weight_bf16.data(), M * N * sizeof(unsigned short), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // --- rocBLAS sgemv (f32 weights) ---
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));
    // rocblas_sgemv: y = alpha * A * x + beta * y
    // A is column-major, but our weights are row-major [M, N]
    // rocblas_operation_transpose on col-major A[N,M] = row-major A[M,N]
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup; i++) {
        CHECK_ROCBLAS(rocblas_sgemv(handle, rocblas_operation_transpose,
            N, M, &alpha, d_weight_f32, N, d_input, 1, &beta, d_output_rocblas, 1));
    }
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_ROCBLAS(rocblas_sgemv(handle, rocblas_operation_transpose,
            N, M, &alpha, d_weight_f32, N, d_input, 1, &beta, d_output_rocblas, 1));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float rocblas_ms = bench_ms(start, stop) / iters;

    // --- Our naive BF16-weight kernel ---
    for (int i = 0; i < warmup; i++) {
        naive_gemv_bf16w<<<M, 256>>>(d_output_naive, d_weight_bf16, d_input, M, N);
    }
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; i++) {
        naive_gemv_bf16w<<<M, 256>>>(d_output_naive, d_weight_bf16, d_input, M, N);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float naive_ms = bench_ms(start, stop) / iters;

    // Memory bandwidth calculations
    // rocBLAS: reads M*N*4 bytes (f32 weights) + N*4 (input) + writes M*4 (output)
    // Naive: reads M*N*2 bytes (bf16 weights) + N*4 (input) + writes M*4 (output)
    double rocblas_bytes = (double)M * N * 4.0 + N * 4.0 + M * 4.0;
    double naive_bytes = (double)M * N * 2.0 + N * 4.0 + M * 4.0;
    double rocblas_gbps = rocblas_bytes / (rocblas_ms * 1e6);
    double naive_gbps = naive_bytes / (naive_ms * 1e6);

    printf("  rocBLAS sgemv (f32):     %.3f ms  (%.1f GB/s)\n", rocblas_ms, rocblas_gbps);
    printf("  Naive bf16w kernel:      %.3f ms  (%.1f GB/s)\n", naive_ms, naive_gbps);
    printf("  Speedup (naive/rocblas): %.2fx\n", rocblas_ms / naive_ms);

    // Verify correctness (approximate, bf16 truncation differs)
    std::vector<float> out_rb(M), out_nv(M);
    CHECK_HIP(hipMemcpy(out_rb.data(), d_output_rocblas, M * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(out_nv.data(), d_output_naive, M * sizeof(float), hipMemcpyDeviceToHost));
    float max_diff = 0;
    for (int i = 0; i < M; i++) max_diff = fmax(max_diff, fabs(out_rb[i] - out_nv[i]));
    printf("  Max diff (rocblas vs naive): %.6f\n\n", max_diff);

    rocblas_destroy_handle(handle);
    CHECK_HIP(hipFree(d_weight_f32));
    CHECK_HIP(hipFree(d_weight_bf16));
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output_rocblas));
    CHECK_HIP(hipFree(d_output_naive));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}

int main() {
    CHECK_HIP(hipSetDevice(0));

    printf("=== GEMV Benchmark: rocBLAS sgemv (f32) vs Naive (bf16 weights) ===\n");
    printf("GPU: gfx1100 (7900 XTX), Peak BW: ~960 GB/s\n\n");

    int warmup = 50, iters = 200;

    // Qwen3.5-0.8B representative sizes
    benchmark_gemv(6144, 1024, warmup, iters);    // GDN QKV projection
    benchmark_gemv(3584, 1024, warmup, iters);    // FFN gate/up projection
    benchmark_gemv(1024, 3584, warmup, iters);    // FFN down projection
    benchmark_gemv(1024, 2048, warmup, iters);    // GDN output projection
    benchmark_gemv(4096, 1024, warmup, iters);    // Attention Q+gate projection
    benchmark_gemv(248320, 1024, warmup, iters);  // LM head (largest)

    return 0;
}
