#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================
// Kernel 1: Current naive BF16-weight GEMV (baseline)
// One block per output row, 256 threads, scalar bf16->f32 + fma
// ============================================================
__device__ __forceinline__ float bf16_to_f32(unsigned short val) {
    unsigned int bits = ((unsigned int)val) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

extern "C" __global__ void naive_gemv_bf16(
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

    const unsigned short* w = weight + (long long)row * in_dim;
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

// ============================================================
// Kernel 2: Vectorized BF16 GEMV with 128-bit loads
// Same structure but uses ulong2 (128-bit) loads for weight reads
// ============================================================
extern "C" __global__ void vec_gemv_bf16(
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

    const unsigned short* w = weight + (long long)row * in_dim;
    float acc = 0.0f;

    // Process 8 bf16 values at a time via 128-bit loads (8 x 16-bit = 128 bits)
    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ulong2* w_vec = (const ulong2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ulong2 wv = w_vec[vi];
        unsigned short ws[8];
        memcpy(ws, &wv, 16);

        int base = vi * vec_elements;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += bf16_to_f32(ws[j]) * input[base + j];
        }
    }

    // Handle remainder
    int remainder_start = vec_count * vec_elements;
    for (int i = remainder_start + tid; i < in_dim; i += stride) {
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

// ============================================================
// Kernel 3: Vectorized + input in shared memory
// Loads input vector to shared mem first (reused by all iterations)
// ============================================================
extern "C" __global__ void vec_smem_gemv_bf16(
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

    // Load input to shared memory (cooperative across block)
    extern __shared__ float smem[];
    for (int i = tid; i < in_dim; i += stride) {
        smem[i] = input[i];
    }
    __syncthreads();

    const unsigned short* w = weight + (long long)row * in_dim;
    float acc = 0.0f;

    const int vec_elements = 8;
    const int vec_count = in_dim / vec_elements;
    const ulong2* w_vec = (const ulong2*)w;

    for (int vi = tid; vi < vec_count; vi += stride) {
        ulong2 wv = w_vec[vi];
        unsigned short ws[8];
        memcpy(ws, &wv, 16);

        int base = vi * vec_elements;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += bf16_to_f32(ws[j]) * smem[base + j];
        }
    }

    // Reuse smem for reduction (input no longer needed since 1 row per block)
    __syncthreads();
    smem[tid] = acc;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[row] = smem[0];
}

// ============================================================
// Kernel 4: Multi-row per block (better occupancy for small in_dim)
// Each block handles ROWS_PER_BLOCK rows, each warp handles one row
// ============================================================
#define ROWS_PER_BLOCK 8
extern "C" __global__ void multirow_gemv_bf16(
    float* __restrict__ output,
    const unsigned short* __restrict__ weight,
    const float* __restrict__ input,
    const int out_dim,
    const int in_dim
) {
    const int warp_id = threadIdx.x / 64;  // wave64
    const int lane_id = threadIdx.x % 64;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const unsigned short* w = weight + (long long)row * in_dim;
    float acc = 0.0f;

    // Each lane in warp processes strided elements
    for (int i = lane_id; i < in_dim; i += 64) {
        acc += bf16_to_f32(w[i]) * input[i];
    }

    // Warp reduction via cross-lane shuffles (wave64)
    for (int offset = 32; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset, 64);
    }

    if (lane_id == 0) output[row] = acc;
}

float bench_ms(hipEvent_t start, hipEvent_t stop) {
    float ms;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

void benchmark(const char* name,
               void(*launcher)(float*, const unsigned short*, const float*, int, int,
                               float*, unsigned short*, float*, hipEvent_t, hipEvent_t, int, int),
               float* d_out, unsigned short* d_w, float* d_in,
               int M, int N, int warmup, int iters,
               hipEvent_t start, hipEvent_t stop) {
    // Defined below
}

void run_benchmarks(int M, int N, int warmup, int iters) {
    printf("GEMV %d x %d:\n", M, N);

    std::vector<unsigned short> h_w(M * (long long)N);
    std::vector<float> h_in(N);
    for (long long i = 0; i < M * (long long)N; i++) {
        float v = (float)(rand() % 1000 - 500) / 1000.0f;
        unsigned int bits;
        memcpy(&bits, &v, 4);
        h_w[i] = (unsigned short)(bits >> 16);
    }
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000 - 500) / 1000.0f;

    unsigned short *d_w;
    float *d_in, *d_out;
    CHECK_HIP(hipMalloc(&d_w, M * (long long)N * sizeof(unsigned short)));
    CHECK_HIP(hipMalloc(&d_in, N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out, M * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), M * (long long)N * sizeof(unsigned short), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    double weight_bytes = (double)M * N * 2.0;
    double input_bytes = N * 4.0;
    double output_bytes = M * 4.0;
    double total_bytes = weight_bytes + input_bytes + output_bytes;

    // Reference output from naive kernel
    naive_gemv_bf16<<<M, 256>>>(d_out, d_w, d_in, M, N);
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<float> ref_out(M);
    CHECK_HIP(hipMemcpy(ref_out.data(), d_out, M * sizeof(float), hipMemcpyDeviceToHost));

    auto run_kernel = [&](const char* kname, auto kernel_fn, int blocks, int threads, int smem_bytes) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            kernel_fn(d_out, d_w, d_in, M, N, blocks, threads, smem_bytes);
        }
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipEventRecord(start));
        for (int i = 0; i < iters; i++) {
            kernel_fn(d_out, d_w, d_in, M, N, blocks, threads, smem_bytes);
        }
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float ms = bench_ms(start, stop) / iters;
        double gbps = total_bytes / (ms * 1e6);

        // Check correctness
        std::vector<float> test_out(M);
        CHECK_HIP(hipMemcpy(test_out.data(), d_out, M * sizeof(float), hipMemcpyDeviceToHost));
        float max_diff = 0;
        for (int i = 0; i < M; i++) max_diff = fmax(max_diff, fabs(ref_out[i] - test_out[i]));

        printf("  %-28s %7.3f ms  %6.1f GB/s  (diff=%.6f)\n", kname, ms, gbps, max_diff);
    };

    // Kernel 1: Naive
    run_kernel("naive (baseline)", [](float* o, const unsigned short* w, const float* in, int M, int N, int, int, int) {
        naive_gemv_bf16<<<M, 256>>>(o, w, in, M, N);
    }, M, 256, 0);

    // Kernel 2: Vectorized 128-bit loads
    run_kernel("vec 128-bit loads", [](float* o, const unsigned short* w, const float* in, int M, int N, int, int, int) {
        vec_gemv_bf16<<<M, 256>>>(o, w, in, M, N);
    }, M, 256, 0);

    // Kernel 3: Vec + shared mem input (only if in_dim fits in shared mem: 48KB max)
    if (N * 4 <= 49152) {
        run_kernel("vec + smem input", [](float* o, const unsigned short* w, const float* in, int M, int N, int, int, int smem) {
            vec_smem_gemv_bf16<<<M, 256, smem>>>(o, w, in, M, N);
        }, M, 256, N * sizeof(float));
    }

    // Kernel 4: Multi-row (wave64 warp reduction)
    int blocks4 = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    run_kernel("multi-row (8 rows/blk)", [](float* o, const unsigned short* w, const float* in, int M, int N, int blocks, int, int) {
        multirow_gemv_bf16<<<blocks, 512>>>(o, w, in, M, N);  // 8 warps * 64 = 512
    }, blocks4, 512, 0);

    printf("\n");

    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_in));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}

int main() {
    CHECK_HIP(hipSetDevice(0));
    printf("=== GEMV Optimization Benchmark (BF16 weights, F32 input/output) ===\n");
    printf("GPU: gfx1100 (7900 XTX), Peak BW: ~960 GB/s\n\n");

    int warmup = 50, iters = 200;

    run_benchmarks(6144, 1024, warmup, iters);
    run_benchmarks(3584, 1024, warmup, iters);
    run_benchmarks(1024, 3584, warmup, iters);
    run_benchmarks(1024, 2048, warmup, iters);
    run_benchmarks(4096, 1024, warmup, iters);
    run_benchmarks(248320, 1024, warmup, iters);

    return 0;
}
