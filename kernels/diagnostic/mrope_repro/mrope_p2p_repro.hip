// MROPE + concurrent P2P traffic repro.
//
// Mini-cooperative MROPE passed (1/100 distinct, 4-GPU). The 5ax bug must
// require an additional production-specific ingredient. The leading suspect:
// in production, GPU 0's megakernel runs prefill MROPE while workers on
// GPUs 1-3 are concurrently P2P-writing to GPU 0's VRAM (MoE FFN output
// activations being collated). That cross-GPU PCIe traffic may perturb
// GPU 0's cache state during MROPE.
//
// This harness reproduces that:
//   - GPU 0 runs the cooperative MROPE chain (100 iters)
//   - GPUs 1-3 simultaneously launch peer-write kernels that hammer GPU 0's
//     VRAM with P2P writes (mimicking MoE FFN output activations)
//   - Compare GPU 0's K-cache hashes across iterations

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>

#define HIPCK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while(0)

constexpr int HEAD_DIM   = 128;
constexpr int ROPE_DIM   = 64;
constexpr int N_HEADS    = 8;
constexpr int N_ITER     = 100;
constexpr int N_BLOCKS   = 96;
constexpr int N_THREADS  = 256;

// Size of the P2P target buffer on GPU 0 — large enough to spread cache pressure
constexpr int P2P_BUF_SZ = 64 * 1024 * 1024 / 4; // 64 MB in floats

namespace cg = cooperative_groups;

// Same as mini-coop: producer → grid.sync → MROPE → grid.sync → consumer
__global__ void mini_coop(float* k, float* k_out, const float* inv_freq, const int* pos_ids,
                          int s0, int s1) {
    cg::grid_group grid = cg::this_grid();
    int total_pairs = ROPE_DIM / 2;
    for (int iter = 0; iter < N_ITER; iter++) {
        for (int head = blockIdx.x; head < N_HEADS; head += gridDim.x) {
            float* head_ptr = k + head * HEAD_DIM;
            for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
                uint32_t u = (uint32_t)((i * 2654435761u) ^ (head * 0xa5a5a5a5u));
                head_ptr[i] = ((float)(u & 0xffff) / 65536.0f) - 0.5f;
            }
        }
        grid.sync();
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
        grid.sync();
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

// Peer GPU writes to GPU 0's buffer continuously. Runs as a long-running kernel
// to maximize concurrent PCIe pressure during GPU 0's MROPE execution.
__global__ void peer_hammer(float* target_on_gpu0, int n_elems, int n_iters, volatile uint32_t* stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int iter = 0; iter < n_iters; iter++) {
        if (*stop) return;
        for (int i = tid; i < n_elems; i += stride) {
            target_on_gpu0[i] = (float)(iter * 1664525 + i * 1013904223) * 1e-6f;
        }
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
    printf("Detected %d GPUs\n", n_dev);
    if (n_dev < 2) { printf("Need 2+ GPUs for P2P test\n"); return 0; }

    // Enable P2P from all peers to GPU 0
    HIPCK(hipSetDevice(0));
    for (int d = 1; d < n_dev; d++) {
        HIPCK(hipSetDevice(d));
        int can = 0; hipDeviceCanAccessPeer(&can, d, 0);
        if (can) HIPCK(hipDeviceEnablePeerAccess(0, 0));
    }
    HIPCK(hipSetDevice(0));

    // GPU 0: allocate K + k_out + inv_freq + pos_ids + P2P target buffer
    float* k = nullptr;
    float* k_out = nullptr;
    float* inv_freq = nullptr;
    int* pos_ids = nullptr;
    float* p2p_target = nullptr;  // peer GPUs write here
    uint32_t* stop_flag = nullptr;

    size_t k_sz     = sizeof(float) * N_HEADS * HEAD_DIM;
    size_t k_out_sz = sizeof(float) * N_ITER * N_HEADS * HEAD_DIM;
    HIPCK(hipMalloc(&k, k_sz));
    HIPCK(hipMalloc(&k_out, k_out_sz));
    HIPCK(hipMalloc(&inv_freq, sizeof(float) * (ROPE_DIM/2)));
    HIPCK(hipMalloc(&pos_ids, sizeof(int) * 3));
    HIPCK(hipMalloc(&p2p_target, sizeof(float) * P2P_BUF_SZ));
    HIPCK(hipMallocManaged(&stop_flag, sizeof(uint32_t)));
    *stop_flag = 0;

    std::vector<float> inv(ROPE_DIM/2);
    for (int i = 0; i < ROPE_DIM/2; i++) inv[i] = 1.0f / powf(10000.0f, (float)(2*i)/(float)ROPE_DIM);
    int pos[3] = {0, 0, 0};
    HIPCK(hipMemcpy(inv_freq, inv.data(), sizeof(float)*inv.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(pos_ids, pos, sizeof(pos), hipMemcpyHostToDevice));

    // Test 1: baseline (no P2P traffic)
    printf("\n=== Test 1: GPU 0 alone (no peer traffic) ===\n");
    int s0 = 16, s1 = 8;
    void* args[] = {&k, &k_out, (void*)&inv_freq, (void*)&pos_ids, &s0, &s1};
    HIPCK(hipLaunchCooperativeKernel((void*)mini_coop, dim3(N_BLOCKS), dim3(N_THREADS), args, 0, 0));
    HIPCK(hipDeviceSynchronize());
    std::vector<float> out(N_ITER * N_HEADS * HEAD_DIM);
    HIPCK(hipMemcpy(out.data(), k_out, sizeof(float)*out.size(), hipMemcpyDeviceToHost));
    std::vector<uint64_t> hashes(N_ITER);
    for (int i = 0; i < N_ITER; i++) hashes[i] = hash_bytes(&out[i*N_HEADS*HEAD_DIM], sizeof(float)*N_HEADS*HEAD_DIM);
    int d_base = count_distinct(hashes);
    printf("baseline: %d/%d distinct (first=%016lx)\n", d_base, N_ITER, hashes.front());

    // Test 2: launch peer-hammer kernels on GPUs 1..n-1 first (long-running),
    // then run GPU 0's cooperative MROPE.
    printf("\n=== Test 2: GPU 0 cooperative MROPE under peer P2P pressure ===\n");
    *stop_flag = 0;
    std::vector<hipStream_t> streams(n_dev - 1, nullptr);
    for (int d = 1; d < n_dev; d++) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipStreamCreate(&streams[d-1]));
        // Launch peer hammer to write to GPU 0's p2p_target buffer
        // n_iters=large so it runs throughout GPU 0's work
        const int hammer_iters = 100000;
        hipLaunchKernelGGL(peer_hammer, dim3(48), dim3(256), 0, streams[d-1],
                           p2p_target, P2P_BUF_SZ, hammer_iters, stop_flag);
    }
    // Now launch GPU 0's coop MROPE
    HIPCK(hipSetDevice(0));
    HIPCK(hipLaunchCooperativeKernel((void*)mini_coop, dim3(N_BLOCKS), dim3(N_THREADS), args, 0, 0));
    HIPCK(hipDeviceSynchronize());
    *stop_flag = 1;  // tell peers to stop
    for (int d = 1; d < n_dev; d++) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipStreamSynchronize(streams[d-1]));
        HIPCK(hipStreamDestroy(streams[d-1]));
    }
    HIPCK(hipSetDevice(0));
    HIPCK(hipMemcpy(out.data(), k_out, sizeof(float)*out.size(), hipMemcpyDeviceToHost));
    std::vector<uint64_t> hashes2(N_ITER);
    for (int i = 0; i < N_ITER; i++) hashes2[i] = hash_bytes(&out[i*N_HEADS*HEAD_DIM], sizeof(float)*N_HEADS*HEAD_DIM);
    int d_pressed = count_distinct(hashes2);
    printf("under-pressure: %d/%d distinct (first=%016lx)\n", d_pressed, N_ITER, hashes2.front());
    if (d_pressed > 1) {
        printf("Divergent iterations under pressure:\n");
        int reported = 0;
        for (int i = 1; i < N_ITER && reported < 5; i++) {
            if (hashes2[i] != hashes2[0]) {
                printf("  iter %d: %016lx (diff from iter 0)\n", i, hashes2[i]);
                reported++;
            }
        }
    }

    // Test 3: repeat 5x to check if test 2 is stable
    printf("\n=== Test 3: under-pressure × 5 trials ===\n");
    int total_distinct = 0;
    for (int trial = 0; trial < 5; trial++) {
        *stop_flag = 0;
        std::vector<hipStream_t> s(n_dev - 1);
        for (int d = 1; d < n_dev; d++) {
            HIPCK(hipSetDevice(d));
            HIPCK(hipStreamCreate(&s[d-1]));
            hipLaunchKernelGGL(peer_hammer, dim3(48), dim3(256), 0, s[d-1],
                               p2p_target, P2P_BUF_SZ, 100000, stop_flag);
        }
        HIPCK(hipSetDevice(0));
        HIPCK(hipLaunchCooperativeKernel((void*)mini_coop, dim3(N_BLOCKS), dim3(N_THREADS), args, 0, 0));
        HIPCK(hipDeviceSynchronize());
        *stop_flag = 1;
        for (int d = 1; d < n_dev; d++) {
            HIPCK(hipSetDevice(d));
            HIPCK(hipStreamSynchronize(s[d-1]));
            HIPCK(hipStreamDestroy(s[d-1]));
        }
        HIPCK(hipSetDevice(0));
        HIPCK(hipMemcpy(out.data(), k_out, sizeof(float)*out.size(), hipMemcpyDeviceToHost));
        std::vector<uint64_t> h(N_ITER);
        for (int i = 0; i < N_ITER; i++) h[i] = hash_bytes(&out[i*N_HEADS*HEAD_DIM], sizeof(float)*N_HEADS*HEAD_DIM);
        int dd = count_distinct(h);
        total_distinct += (dd > 1) ? 1 : 0;
        printf("  trial %d: %d/%d distinct\n", trial+1, dd, N_ITER);
    }
    printf("trials with divergence: %d/5\n", total_distinct);

    return 0;
}
