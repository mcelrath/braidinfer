// Standalone op_mrope determinism harness.
// Mirrors kernels/megakernel_ops.hip op_mrope inner loop (lines 1700-1724).
// Tests determinism single-GPU, multi-GPU concurrent, and multi-GPU stress.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>

#define HIPCK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while(0)

constexpr int HEAD_DIM = 128;
constexpr int ROPE_DIM = 64;       // total_pairs = 32
constexpr int NUM_Q_HEADS = 8;
constexpr int NUM_KV_HEADS = 2;
constexpr int NUM_TOKENS = 1;
constexpr int HEADS_PER_TOKEN = NUM_Q_HEADS + NUM_KV_HEADS;
constexpr int N_ITER = 100;

__global__ void mrope_iso(float* q, float* k, const float* inv_freq, const int* pos_ids,
                          int s0, int s1) {
    int vblock = blockIdx.x;
    int token_idx = vblock / HEADS_PER_TOKEN;
    int head_idx = vblock % HEADS_PER_TOKEN;
    float* head_ptr = (head_idx < NUM_Q_HEADS)
        ? q + (token_idx * NUM_Q_HEADS + head_idx) * HEAD_DIM
        : k + (token_idx * NUM_KV_HEADS + (head_idx - NUM_Q_HEADS)) * HEAD_DIM;
    const int* tok_pos = pos_ids + token_idx * 3;
    int total_pairs = ROPE_DIM / 2;
    for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
        int section = (pair < s0) ? 0 : (pair < s0 + s1) ? 1 : 2;
        int pos = tok_pos[section];
        float theta = (float)pos * inv_freq[pair];
        float cos_t = cosf(theta), sin_t = sinf(theta);
        int i0 = pair * 2, i1 = pair * 2 + 1;
        float x0 = head_ptr[i0], x1 = head_ptr[i1];
        head_ptr[i0] = x0 * cos_t - x1 * sin_t;
        head_ptr[i1] = x0 * sin_t + x1 * cos_t;
    }
}

static uint64_t hash_bytes(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Buffers { float *q, *k, *inv_freq; int* pos_ids; float *q_init, *k_init; };

static Buffers alloc_init(int dev) {
    HIPCK(hipSetDevice(dev));
    Buffers b{};
    size_t q_sz = sizeof(float) * NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM;
    size_t k_sz = sizeof(float) * NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM;
    HIPCK(hipMalloc(&b.q, q_sz));
    HIPCK(hipMalloc(&b.k, k_sz));
    HIPCK(hipMalloc(&b.inv_freq, sizeof(float) * (ROPE_DIM/2)));
    HIPCK(hipMalloc(&b.pos_ids, sizeof(int) * NUM_TOKENS * 3));
    b.q_init = (float*)malloc(q_sz); b.k_init = (float*)malloc(k_sz);
    // Deterministic init: indices XOR mixed to floats
    size_t qn = q_sz/4, kn = k_sz/4;
    for (size_t i = 0; i < qn; i++) { uint32_t u = (uint32_t)(i*2654435761u) ^ 0xa5a5a5a5u; b.q_init[i] = ((float)(u & 0xffff) / 65536.0f) - 0.5f; }
    for (size_t i = 0; i < kn; i++) { uint32_t u = (uint32_t)(i*2246822519u) ^ 0x5a5a5a5au; b.k_init[i] = ((float)(u & 0xffff) / 65536.0f) - 0.5f; }
    std::vector<float> inv(ROPE_DIM/2);
    for (int i = 0; i < ROPE_DIM/2; i++) inv[i] = 1.0f / powf(10000.0f, (float)(2*i) / (float)ROPE_DIM);
    int pos[3] = {0, 0, 0};
    HIPCK(hipMemcpy(b.inv_freq, inv.data(), sizeof(float)*inv.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(b.pos_ids, pos, sizeof(pos), hipMemcpyHostToDevice));
    return b;
}

static uint64_t run_once(int dev, Buffers& b, float* k_host) {
    HIPCK(hipSetDevice(dev));
    size_t q_sz = sizeof(float)*NUM_TOKENS*NUM_Q_HEADS*HEAD_DIM;
    size_t k_sz = sizeof(float)*NUM_TOKENS*NUM_KV_HEADS*HEAD_DIM;
    HIPCK(hipMemcpy(b.q, b.q_init, q_sz, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(b.k, b.k_init, k_sz, hipMemcpyHostToDevice));
    int s0 = 16, s1 = 8; // arbitrary sections; pos=0 so section choice is irrelevant numerically
    hipLaunchKernelGGL(mrope_iso, dim3(NUM_TOKENS*HEADS_PER_TOKEN), dim3(256), 0, 0,
                       b.q, b.k, b.inv_freq, b.pos_ids, s0, s1);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(k_host, b.k, k_sz, hipMemcpyDeviceToHost));
    return hash_bytes(k_host, k_sz);
}

static int count_distinct(const std::vector<uint64_t>& v) {
    std::vector<uint64_t> s = v; std::sort(s.begin(), s.end());
    int d = 0; for (size_t i = 0; i < s.size(); i++) if (i == 0 || s[i] != s[i-1]) d++;
    return d;
}

static void run_dev_loop(int dev, std::vector<uint64_t>* out) {
    Buffers b = alloc_init(dev);
    size_t k_sz = sizeof(float)*NUM_TOKENS*NUM_KV_HEADS*HEAD_DIM;
    float* k_host = (float*)malloc(k_sz);
    out->resize(N_ITER);
    for (int i = 0; i < N_ITER; i++) (*out)[i] = run_once(dev, b, k_host);
    free(k_host); free(b.q_init); free(b.k_init);
    hipFree(b.q); hipFree(b.k); hipFree(b.inv_freq); hipFree(b.pos_ids);
}

int main() {
    int n_dev = 0; HIPCK(hipGetDeviceCount(&n_dev));
    printf("Detected %d GPUs\n", n_dev);

    // 1) Single-GPU determinism
    printf("\n=== Single-GPU (dev 0) ===\n");
    std::vector<uint64_t> h0; run_dev_loop(0, &h0);
    int d0 = count_distinct(h0);
    printf("dev0: %d/%d distinct hashes (first=%016lx last=%016lx)\n", d0, N_ITER, h0.front(), h0.back());

    // 2) Multi-GPU concurrent
    if (n_dev >= 2) {
        printf("\n=== Multi-GPU concurrent (%d GPUs, N=%d each) ===\n", n_dev, N_ITER);
        std::vector<std::vector<uint64_t>> hs(n_dev);
        std::vector<std::thread> ts;
        for (int d = 0; d < n_dev; d++) ts.emplace_back(run_dev_loop, d, &hs[d]);
        for (auto& t : ts) t.join();
        for (int d = 0; d < n_dev; d++) {
            int dd = count_distinct(hs[d]);
            printf("dev%d: %d/%d distinct (first=%016lx)\n", d, dd, N_ITER, hs[d].front());
        }
        bool cross_same = true; uint64_t r0 = hs[0].front();
        for (int d = 1; d < n_dev; d++) if (hs[d].front() != r0) cross_same = false;
        printf("cross-GPU first-hash identical: %s\n", cross_same ? "YES" : "NO");
    }

    // 3) Stress: concurrent + extra PCIe host->device traffic
    if (n_dev >= 2) {
        printf("\n=== Stress (concurrent + PCIe contention) ===\n");
        std::vector<std::vector<uint64_t>> hs(n_dev);
        std::vector<std::thread> ts;
        volatile bool stop = false;
        std::thread pcie([&]() {
            std::vector<float> junk(1<<20, 1.0f);
            std::vector<float*> dst(n_dev, nullptr);
            for (int d = 0; d < n_dev; d++) { hipSetDevice(d); hipMalloc(&dst[d], sizeof(float)*junk.size()); }
            while (!stop) for (int d = 0; d < n_dev; d++) { hipSetDevice(d); hipMemcpy(dst[d], junk.data(), sizeof(float)*junk.size(), hipMemcpyHostToDevice); }
            for (int d = 0; d < n_dev; d++) { hipSetDevice(d); hipFree(dst[d]); }
        });
        for (int d = 0; d < n_dev; d++) ts.emplace_back(run_dev_loop, d, &hs[d]);
        for (auto& t : ts) t.join();
        stop = true; pcie.join();
        for (int d = 0; d < n_dev; d++) {
            int dd = count_distinct(hs[d]);
            printf("dev%d (stress): %d/%d distinct (first=%016lx)\n", d, dd, N_ITER, hs[d].front());
        }
    }
    return 0;
}
