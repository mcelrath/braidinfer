// braidinfer-g1wv latency-hiding investigation.
//
// Question (from the user): replacing a grid barrier with a sentinel isn't only
// about the single-op barrier TIME — does it HIDE LATENCY?
//
// Mechanism under test: a barrier is bulk-synchronous — every producer must finish
// before any consumer starts, so (1) consumer memory-load latency is EXPOSED after
// the barrier instead of overlapping producer compute, and (2) across M sequential
// stages the per-stage SLOWEST block gates the whole grid, so stragglers ACCUMULATE
// (total ≈ Σ_m max_block(stage_m)). A sentinel handoff lets each consumer flow the
// instant ITS specific input is ready, so the critical path is the longest
// dependency CHAIN through the DAG, not the sum of per-stage maxima — and a
// consumer's load latency overlaps other blocks' producer compute.
//
// Demonstrator: M-stage ring pipeline. Block i at stage m computes for T(i,m) cycles
// then publishes out[m][i]; stage m reads its NEIGHBOR's previous stage out[m-1][(i+1)%N]
// plus a LATENCY chain (dependent loads) to model attention's K/V load latency.
// ROTATING straggler: a different block is the slow one each stage (T big when
// i == (m % N)). With a barrier the grid pays max each stage → M*SLOW. With sentinels
// the slow block rotates off the critical path → ~SLOW + (M-1)*FAST + latency overlap.
//
// Build: hipcc --offload-arch=gfx1100 -O3 -std=c++17 kernels/proto/latency_hiding.hip -o /tmp/latency_hiding
// Run:   launch-gpu -g 1 -- /tmp/latency_hiding [N] [M] [SLOW] [FAST] [LAT] [TRIALS]

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;
#include "../rdna3/rdna3_barrier.h"
__device__ braidinfer::rdna3::GridBarrierState g_lh_gbs;

static const unsigned int SENT = 0x7fc00000u; // qNaN sentinel

__device__ __forceinline__ unsigned int la(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void sa(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void spin(unsigned long long cyc) {
    unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < cyc) { __builtin_amdgcn_s_sleep(0); }
}
// dependent-load latency chain over a scratch line (models memory-bound consumer)
__device__ __forceinline__ float lat_chain(const volatile float* scratch, int lat, float seed) {
    float a = seed;
    for (int k = 0; k < lat; ++k) a = scratch[((int)a) & 1023] + 1.0f;
    return a;
}

// BARRIER: atomic_block_barrier between every stage (braidinfer's per-instruction barrier).
__global__ void pipe_barrier(float* out, const volatile float* scratch, int N, int M,
                             unsigned long long SLOW, unsigned long long FAST, int LAT,
                             unsigned long long* total) {
    int i = blockIdx.x;
    unsigned long long t0 = wall_clock64();
    for (int m = 0; m < M; ++m) {
        unsigned long long T = (i == (m % N)) ? SLOW : FAST;   // rotating straggler
        if (threadIdx.x == 0) {
            spin(T);
            float in = (m == 0) ? 0.0f : out[(m - 1) * N + (i + 1) % N];   // neighbor prev stage
            float v = lat_chain(scratch, (m == 0) ? 0 : LAT, in) + (float)(i + 1);
            ((volatile float*)out)[m * N + i] = v;
        }
        braidinfer::rdna3::atomic_block_barrier(&g_lh_gbs);
    }
    if (i == 0 && threadIdx.x == 0) *total = wall_clock64() - t0;
}

// SENTINEL: out[] pre-init qNaN; consumer spins on neighbor's prev-stage slot. No barrier.
__global__ void pipe_sentinel(float* out, const volatile float* scratch, int N, int M,
                              unsigned long long SLOW, unsigned long long FAST, int LAT,
                              unsigned long long* total) {
    int i = blockIdx.x;
    unsigned int* uo = reinterpret_cast<unsigned int*>(out);
    unsigned long long t0 = wall_clock64();
    if (threadIdx.x == 0) {
        for (int m = 0; m < M; ++m) {
            unsigned long long T = (i == (m % N)) ? SLOW : FAST;
            spin(T);
            float in = 0.0f;
            if (m > 0) {
                unsigned int slot = (m - 1) * N + (i + 1) % N, v;
                do { v = la(&uo[slot]); } while (v == SENT);   // poll the data path
                __builtin_memcpy(&in, &v, 4);
            }
            float fv = lat_chain(scratch, (m == 0) ? 0 : LAT, in) + (float)(i + 1);
            unsigned int bits; __builtin_memcpy(&bits, &fv, 4);
            sa(&uo[m * N + i], bits);
        }
    }
    if (i == 0 && threadIdx.x == 0) *total = wall_clock64() - t0;
}

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 2;} } while(0)

int main(int argc, char** argv) {
    int N=(argc>1)?atoi(argv[1]):96, M=(argc>2)?atoi(argv[2]):32;
    unsigned long long SLOW=(argc>3)?atoll(argv[3]):4000, FAST=(argc>4)?atoll(argv[4]):400;
    int LAT=(argc>5)?atoi(argv[5]):64, TRIALS=(argc>6)?atoi(argv[6]):20;
    CK(hipSetDevice(0));
    int coop=0; hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0);
    printf("N=%d M=%d SLOW=%llu FAST=%llu LAT=%d trials=%d coop=%d\n", N,M,SLOW,FAST,LAT,TRIALS,coop);
    if(!coop){printf("FAIL coop\n");return 1;}
    float *out, *scratch; unsigned long long *d_total;
    CK(hipMalloc(&out, (size_t)M*N*sizeof(float)));
    CK(hipMalloc(&scratch, 1024*sizeof(float)));
    CK(hipMalloc(&d_total, sizeof(unsigned long long)));
    std::vector<float> hnan((size_t)M*N), hz(1024,1.0f);
    { float f; __builtin_memcpy(&f,&SENT,4); std::fill(hnan.begin(),hnan.end(),f); }
    CK(hipMemcpy(scratch, hz.data(), 1024*sizeof(float), hipMemcpyHostToDevice));
    auto run=[&](void* k, bool nan, const char* name)->unsigned long long {
        std::vector<unsigned long long> ts;
        for(int t=0;t<TRIALS;++t){
            if(nan) CK(hipMemcpy(out, hnan.data(), (size_t)M*N*sizeof(float), hipMemcpyHostToDevice));
            else CK(hipMemset(out, 0, (size_t)M*N*sizeof(float)));
            void* args[]={&out,&scratch,&N,&M,&SLOW,&FAST,&LAT,&d_total};
            CK(hipLaunchCooperativeKernel(k, dim3(N), dim3(64), args, 0, 0));
            CK(hipDeviceSynchronize());
            unsigned long long tot; CK(hipMemcpy(&tot,d_total,sizeof(tot),hipMemcpyDeviceToHost));
            ts.push_back(tot);
        }
        std::sort(ts.begin(),ts.end());
        printf("  %-9s : median total = %llu wclk\n", name, ts[ts.size()/2]);
        return ts[ts.size()/2];
    };
    printf("--- %d-stage ring pipeline, rotating straggler, latency-%d consumer ---\n", M, LAT);
    printf("  ideal barrier  = M*SLOW = %llu (sum of per-stage maxima)\n", (unsigned long long)M*SLOW);
    unsigned long long b=run((void*)pipe_barrier,false,"barrier");
    unsigned long long s=run((void*)pipe_sentinel,true,"sentinel");
    printf("RESULT: barrier=%llu sentinel=%llu  speedup=%.2fx  (latency hidden + straggler overlap)\n",
           b, s, b>0 ? (double)b/(double)s : 0.0);
    hipFree(out); hipFree(scratch); hipFree(d_total);
    return 0;
}
