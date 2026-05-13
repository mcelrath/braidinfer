// Mini-coop MROPE with atomic_block_barrier instead of cg::grid.sync.
// Tests whether production's custom barrier has different cache coherence
// than cg::grid_group::sync(), as the differential vs prior repros.
#include <hip/hip_runtime.h>
#include "../../rdna3/rdna3_barrier.h"
#include <cstdio>
#include <cstdint>
#include <vector>

#define HIPCK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s\n", hipGetErrorString(e)); std::abort(); } } while(0)

constexpr int HEAD_DIM   = 128;
constexpr int ROPE_DIM   = 64;
constexpr int N_HEADS    = 8;
constexpr int N_ITER     = 100;
constexpr int N_BLOCKS   = 96;
constexpr int N_THREADS  = 256;

__device__ static braidinfer::rdna3::GridBarrierState g_gbs;

__global__ void mini_coop_atomic(float* k, float* k_out, const float* inv_freq, const int* pos_ids,
                                 int s0, int s1) {
    int total_pairs = ROPE_DIM / 2;
    for (int iter = 0; iter < N_ITER; iter++) {
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                uint32_t u = (uint32_t)((i * 2654435761u) ^ (head * 0xa5a5a5a5u));
                head_ptr[i] = ((float)(u & 0xffff) / 65536.0f) - 0.5f;
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
                int section = (pair < s0) ? 0 : (pair < s0 + s1) ? 1 : 2;
                int pos = pos_ids[section];
                float theta = (float)pos * inv_freq[pair];
                float cos_t = cosf(theta), sin_t = sinf(theta);
                int i0 = pair * 2, i1 = pair * 2 + 1;
                float x0 = head_ptr[i0], x1 = head_ptr[i1];
                head_ptr[i0] = x0 * cos_t - x1 * sin_t;
                head_ptr[i1] = x0 * sin_t + x1 * cos_t;
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            float* out_ptr  = k_out + iter * (N_HEADS * HEAD_DIM) + head * HEAD_DIM;
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                out_ptr[i] = head_ptr[i];
            }
        }
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);
    }
}

static uint64_t hash_bytes(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static int count_distinct(const std::vector<uint64_t>& v) {
    std::vector<uint64_t> s = v; std::sort(s.begin(), s.end());
    int d = 0; for (size_t i = 0; i < s.size(); i++) if (i == 0 || s[i] != s[i-1]) d++;
    return d;
}

int main() {
    int n_dev = 0; HIPCK(hipGetDeviceCount(&n_dev));
    printf("Detected %d GPUs. Using atomic_block_barrier (not grid.sync)\n", n_dev);

    HIPCK(hipSetDevice(0));
    float *k=nullptr, *k_out=nullptr, *inv_freq=nullptr; int* pos_ids=nullptr;
    size_t k_sz = sizeof(float)*N_HEADS*HEAD_DIM;
    size_t k_out_sz = sizeof(float)*N_ITER*N_HEADS*HEAD_DIM;
    HIPCK(hipMalloc(&k, k_sz));
    HIPCK(hipMalloc(&k_out, k_out_sz));
    HIPCK(hipMalloc(&inv_freq, sizeof(float)*(ROPE_DIM/2)));
    HIPCK(hipMalloc(&pos_ids, sizeof(int)*3));
    std::vector<float> inv(ROPE_DIM/2);
    for (int i = 0; i < ROPE_DIM/2; i++) inv[i] = 1.0f/powf(10000.0f, (float)(2*i)/(float)ROPE_DIM);
    int pos[3] = {0,0,0};
    HIPCK(hipMemcpy(inv_freq, inv.data(), sizeof(float)*inv.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(pos_ids, pos, sizeof(pos), hipMemcpyHostToDevice));

    int s0=16, s1=8;
    void* args[] = {&k, &k_out, (void*)&inv_freq, (void*)&pos_ids, &s0, &s1};

    // 5 trials, single-GPU coop with atomic_block_barrier
    printf("\n=== Single-GPU coop, atomic_block_barrier × 5 trials ===\n");
    int divergent_trials = 0;
    for (int trial = 0; trial < 5; trial++) {
        HIPCK(hipLaunchCooperativeKernel((void*)mini_coop_atomic, dim3(N_BLOCKS), dim3(N_THREADS), args, 0, 0));
        HIPCK(hipDeviceSynchronize());
        std::vector<float> out(N_ITER*N_HEADS*HEAD_DIM);
        HIPCK(hipMemcpy(out.data(), k_out, sizeof(float)*out.size(), hipMemcpyDeviceToHost));
        std::vector<uint64_t> h(N_ITER);
        for (int i = 0; i < N_ITER; i++) h[i] = hash_bytes(&out[i*N_HEADS*HEAD_DIM], sizeof(float)*N_HEADS*HEAD_DIM);
        int dd = count_distinct(h);
        if (dd > 1) divergent_trials++;
        printf("  trial %d: %d/%d distinct (first=%016lx)\n", trial+1, dd, N_ITER, h.front());
    }
    printf("trials with divergence: %d/5\n", divergent_trials);
    return 0;
}
