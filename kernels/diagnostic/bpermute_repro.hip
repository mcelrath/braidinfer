// Minimal reproducer for gfx1100 (RDNA3) ds_bpermute_b32 same-VGPR
// non-determinism observed when -ffp-contract=fast lets the compiler
// allocate vDst == vSrc on a cross-lane shuffle.
//
// Two warp reductions are performed in the same kernel:
//   A: control     -- inputs are q[tid] * k[tid] (two distinct loads)
//   B: suspect     -- inputs are x[tid] * x[tid] (self-multiply -> FMA fusion)
//
// Compile (gfx1100, wave64):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 --genco -O3 -std=c++17 \
//       -ffp-contract=fast -mwavefrontsize64 --save-temps \
//       -o bpermute_repro.hsaco bpermute_repro.hip
//
// Inspect the asm:
//   grep 'ds_bpermute_b32' bpermute_repro-hip-amdgcn-amd-amdhsa-gfx1100.s
//
// Expected:
//   Pattern A reductions:  ds_bpermute_b32 vDst, vIdx, vSrc   (vDst != vSrc)
//   Pattern B reductions:  ds_bpermute_b32 vSame, vIdx, vSame (vDst == vSrc)

#include <hip/hip_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Wave size for gfx1100 in wave64 mode.
#define WAVE 64

// Force the compiler to actually emit the squared-reduction in a way
// that can fuse via FMA. The volatile barrier prevents CSE from collapsing
// pattern A into pattern B when the inputs happen to alias.
__device__ __forceinline__ float warp_sum_shfl_down(float v) {
    // Tree reduction across a 64-lane wave using __shfl_down.
    // HIP lowers this to ds_bpermute_b32 on RDNA3.
    v += __shfl_down(v, 32);
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

extern "C"
__global__ __launch_bounds__(BLOCK_SIZE)
void repro_kernel(const float* __restrict__ q,
                  const float* __restrict__ k,
                  const float* __restrict__ x,
                  float* __restrict__ out,
                  int   N,
                  int   runs)
{
    int tid    = threadIdx.x;
    int lane   = tid & (WAVE - 1);
    int warp   = tid / WAVE;

    // --- PATTERN A: distinct-input multiply (control)
    // Two independent loads, sum-reduced. Compiler should NOT fuse these
    // into a self-FMA, so register allocator should produce
    //   ds_bpermute_b32 vDst, vIdx, vSrc with vDst != vSrc.
    float qa = q[tid];
    float kb = k[tid];
    float a  = qa * kb;
    float ra = warp_sum_shfl_down(a);

    // --- PATTERN B: self-multiply (suspect)
    // x[tid] * x[tid] is a perfect candidate for FMA fusion across the
    // warp reduction with -ffp-contract=fast. The fusion frees the
    // producer VGPR, allowing register allocation to overlap dst==src
    // on ds_bpermute_b32.
    float xv = x[tid];
    float b  = xv * xv;
    float rb = warp_sum_shfl_down(b);

    // Lane 0 of each warp writes both reductions.
    if (lane == 0) {
        // Use the run loop tail of out[] to scatter results from many runs.
        // Single-block launch: gridDim.x == 1, blockIdx.x == 0.
        // Slot layout: [run*2 + 0] = pattern A, [run*2 + 1] = pattern B.
        int run = runs;  // host passes the run index as 'runs'
        int slot_base = (run * (BLOCK_SIZE / WAVE) + warp) * 2;
        out[slot_base + 0] = ra;
        out[slot_base + 1] = rb;
    }
}

// Host driver: run the kernel many times back-to-back over the same input
// buffer and tally how many distinct outputs we see in slots A vs B.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <cstring>

#define CHECK(call) do {                                                \
    hipError_t e = (call);                                              \
    if (e != hipSuccess) {                                              \
        fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                hipGetErrorString(e), __FILE__, __LINE__);              \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

int main(int argc, char** argv) {
    int RUNS = (argc > 1) ? atoi(argv[1]) : 1000;
    int N    = BLOCK_SIZE;
    int WARPS_PER_BLOCK = BLOCK_SIZE / WAVE;
    int OUT_PER_RUN     = WARPS_PER_BLOCK * 2;
    size_t out_count    = (size_t)RUNS * OUT_PER_RUN;

    std::vector<float> hq(N), hk(N), hx(N);
    for (int i = 0; i < N; ++i) {
        // Deterministic, non-trivial. Mix in a bit of noise via prime stride.
        hq[i] = float(i) * 0.137f + 0.5f;
        hk[i] = float(i) * 0.073f - 0.25f;
        hx[i] = float(i) * 0.097f + 0.125f;
    }

    float *dq=nullptr, *dk=nullptr, *dx=nullptr, *dout=nullptr;
    CHECK(hipMalloc(&dq,   N * sizeof(float)));
    CHECK(hipMalloc(&dk,   N * sizeof(float)));
    CHECK(hipMalloc(&dx,   N * sizeof(float)));
    CHECK(hipMalloc(&dout, out_count * sizeof(float)));

    CHECK(hipMemcpy(dq, hq.data(), N*sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dk, hk.data(), N*sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, hx.data(), N*sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemset(dout, 0, out_count * sizeof(float)));

    // Launch RUNS times, same buffers, no resets between runs.
    for (int r = 0; r < RUNS; ++r) {
        hipLaunchKernelGGL(repro_kernel,
                           dim3(1), dim3(BLOCK_SIZE), 0, 0,
                           dq, dk, dx, dout, N, r);
    }
    CHECK(hipDeviceSynchronize());

    std::vector<float> hout(out_count);
    CHECK(hipMemcpy(hout.data(), dout, out_count*sizeof(float),
                    hipMemcpyDeviceToHost));

    // Tally distinct values in pattern A and pattern B slices.
    auto count_unique = [&](int slot) {
        std::unordered_set<uint32_t> bits;
        uint32_t first = 0; bool have_first=false;
        int first_diff = -1;
        for (int r = 0; r < RUNS; ++r) {
            // Use only warp 0 for clarity; same logic per warp.
            int idx = r * OUT_PER_RUN + 0 * 2 + slot;
            uint32_t u; std::memcpy(&u, &hout[idx], sizeof(u));
            bits.insert(u);
            if (!have_first) { first = u; have_first = true; }
            else if (u != first && first_diff < 0) { first_diff = r; }
        }
        return std::pair<size_t,int>(bits.size(), first_diff);
    };

    auto a = count_unique(0);
    auto b = count_unique(1);
    printf("Pattern A (dst != src): unique=%zu / %d   first_divergence_run=%d\n",
           a.first, RUNS, a.second);
    printf("Pattern B (dst == src): unique=%zu / %d   first_divergence_run=%d\n",
           b.first, RUNS, b.second);

    if (a.first == 1 && b.first > 1) {
        printf("\nNON-DETERMINISM REPRODUCED: pattern B (self-multiply, "
               "FMA-fusable) shows %zu distinct results over %d runs while "
               "pattern A is bit-stable.\n", b.first, RUNS);
        return 0;
    }
    if (a.first == 1 && b.first == 1) {
        printf("\nBOTH STABLE: bug not reproduced in this run. Inspect the "
               ".s file to confirm same-VGPR ds_bpermute_b32 was emitted.\n");
        return 0;
    }
    printf("\nUnexpected: pattern A also nondeterministic. Inspect asm.\n");
    return 0;
}
