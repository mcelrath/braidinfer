// Mini-cooperative MROPE repro: tests whether the kb 5ax non-determinism
// reproduces when op_mrope is sandwiched between adjacent ops inside a
// cooperative kernel (with grid.sync() between ops), as it is in production.
//
// Per Agent B's standalone harness (mrope_repro.hip): op_mrope's math is
// fully deterministic in isolation. The 5ax bug must come from the
// cooperative-kernel context (megakernel/persistent_worker, grid.sync,
// neighboring ops).
//
// This harness reproduces that context minimally:
//   1. Producer op: writes deterministic K (mimics op_qk_norm output)
//   2. grid.sync()
//   3. op_mrope: in-place rotation on K (lines 1700-1724 of megakernel_ops.hip)
//   4. grid.sync()
//   5. Consumer op: copies K to per-iteration output slot (mimics op_d2d_copy)
//
// Loop N_ITER times inside the SAME cooperative kernel invocation. Compare
// all iteration outputs — if they differ, the cooperative-context cache
// coherence bug is reproduced. If they match, the bug requires more than
// these 3 ops (e.g., full megakernel state).

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>

#define HIPCK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while(0)

constexpr int HEAD_DIM   = 128;
constexpr int ROPE_DIM   = 64;
constexpr int N_HEADS    = 8;
constexpr int N_ITER     = 100;
constexpr int N_BLOCKS   = 96;
constexpr int N_THREADS  = 256;

namespace cg = cooperative_groups;

__global__ void mini_coop(float* k, float* k_out, const float* inv_freq, const int* pos_ids,
                          int s0, int s1) {
    cg::grid_group grid = cg::this_grid();
    int total_pairs = ROPE_DIM / 2;

    for (int iter = 0; iter < N_ITER; iter++) {
        // Op 1: Producer — write deterministic K (per-head, per-element).
        // Same pattern every iteration so MROPE input is constant across runs.
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                // Deterministic synthetic value; same each iteration
                uint32_t u = (uint32_t)((i * 2654435761u) ^ (head * 0xa5a5a5a5u));
                head_ptr[i] = ((float)(u & 0xffff) / 65536.0f) - 0.5f;
            }
        }
        grid.sync();

        // Op 2: MROPE — mirrors kernels/megakernel_ops.hip op_mrope (lines 1700-1724)
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            const int* tok_pos = pos_ids;
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
        grid.sync();

        // Op 3: Consumer — copy K to per-iter output slot
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            float* out_ptr  = k_out + iter * (N_HEADS * HEAD_DIM) + head * HEAD_DIM;
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                out_ptr[i] = head_ptr[i];
            }
        }
        grid.sync();
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

struct Buffers { float *k, *k_out, *inv_freq; int* pos_ids; };

static Buffers alloc_dev(int dev) {
    HIPCK(hipSetDevice(dev));
    Buffers b{};
    size_t k_sz     = sizeof(float) * N_HEADS * HEAD_DIM;
    size_t k_out_sz = sizeof(float) * N_ITER * N_HEADS * HEAD_DIM;
    HIPCK(hipMalloc(&b.k, k_sz));
    HIPCK(hipMalloc(&b.k_out, k_out_sz));
    HIPCK(hipMalloc(&b.inv_freq, sizeof(float) * (ROPE_DIM/2)));
    HIPCK(hipMalloc(&b.pos_ids, sizeof(int) * 3));
    std::vector<float> inv(ROPE_DIM/2);
    for (int i = 0; i < ROPE_DIM/2; i++) inv[i] = 1.0f / powf(10000.0f, (float)(2*i)/(float)ROPE_DIM);
    int pos[3] = {0, 0, 0};
    HIPCK(hipMemcpy(b.inv_freq, inv.data(), sizeof(float)*inv.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(b.pos_ids, pos, sizeof(pos), hipMemcpyHostToDevice));
    return b;
}

static std::vector<uint64_t> run_dev(int dev) {
    Buffers b = alloc_dev(dev);
    int s0 = 16, s1 = 8;
    void* args[] = {&b.k, &b.k_out, (void*)&b.inv_freq, (void*)&b.pos_ids, &s0, &s1};
    HIPCK(hipLaunchCooperativeKernel((void*)mini_coop,
                                     dim3(N_BLOCKS), dim3(N_THREADS), args, 0, 0));
    HIPCK(hipDeviceSynchronize());
    size_t k_per_iter = N_HEADS * HEAD_DIM;
    std::vector<float> out(N_ITER * k_per_iter);
    HIPCK(hipMemcpy(out.data(), b.k_out, sizeof(float)*out.size(), hipMemcpyDeviceToHost));
    std::vector<uint64_t> hashes(N_ITER);
    for (int i = 0; i < N_ITER; i++) hashes[i] = hash_bytes(&out[i*k_per_iter], sizeof(float)*k_per_iter);
    hipFree(b.k); hipFree(b.k_out); hipFree(b.inv_freq); hipFree(b.pos_ids);
    return hashes;
}

int main() {
    int n_dev = 0; HIPCK(hipGetDeviceCount(&n_dev));
    printf("Detected %d GPUs\n", n_dev);
    printf("Mini-cooperative MROPE repro: %d iters per launch, %d blocks × %d threads\n",
           N_ITER, N_BLOCKS, N_THREADS);

    // Single-GPU
    printf("\n=== Single-GPU coop (dev 0) ===\n");
    auto h0 = run_dev(0);
    int d0 = count_distinct(h0);
    printf("dev0: %d/%d distinct hashes (first=%016lx last=%016lx)\n",
           d0, N_ITER, h0.front(), h0.back());
    if (d0 > 1) {
        printf("Divergent iterations:\n");
        for (int i = 1; i < N_ITER && i < 5; i++) {
            if (h0[i] != h0[0]) printf("  iter %d: %016lx (diff from iter 0)\n", i, h0[i]);
        }
    }

    // Multi-GPU concurrent
    if (n_dev >= 2) {
        printf("\n=== Multi-GPU concurrent coop (%d GPUs, %d iters each) ===\n", n_dev, N_ITER);
        std::vector<std::vector<uint64_t>> hs(n_dev);
        std::vector<std::thread> ts;
        for (int d = 0; d < n_dev; d++) ts.emplace_back([d, &hs](){ hs[d] = run_dev(d); });
        for (auto& t : ts) t.join();
        for (int d = 0; d < n_dev; d++) {
            int dd = count_distinct(hs[d]);
            printf("dev%d: %d/%d distinct (first=%016lx)\n", d, dd, N_ITER, hs[d].front());
        }
        bool cross_same = true; uint64_t r0 = hs[0].front();
        for (int d = 1; d < n_dev; d++) if (hs[d].front() != r0) cross_same = false;
        printf("cross-GPU first-iter hash identical: %s\n", cross_same ? "YES" : "NO");
    }
    return 0;
}
