// Smoke test for kernels/rdna3_reduce.h. Confirms the header compiles
// stand-alone and produces correct sums.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../../rdna3_reduce.h"

#define CHECK(c) do{auto e=(c); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));std::exit(1);}}while(0)

using namespace braidinfer::rdna3;

__global__ void test_wave32(float* out) {
    int lane = threadIdx.x;
    float v = (float)lane;
    v = wave32_reduce_sum(v);
    out[lane] = v;
}

template<int W>
__global__ void test_subwave(float* out) {
    int lane = threadIdx.x;
    float v = (float)lane;
    v = subwave_reduce_sum<W>(v);
    out[lane] = v;
}

__global__ __launch_bounds__(256) void test_block(float* out) {
    int tid = threadIdx.x;
    __shared__ float s[8];
    float v = (float)tid;
    v = block_reduce_sum_256(v, s);
    out[tid] = v;
}

int main() {
    float h[256];
    float* d; CHECK(hipMalloc(&d, 256 * sizeof(float)));

    test_wave32<<<1, 32>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 32 * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < 32; i++) if (h[i] != 496.0f) { ok = false; break; }
    printf("wave32_reduce_sum lane0..31 (expect 496): %.0f .. %.0f  %s\n",
           h[0], h[31], ok ? "PASS" : "FAIL");

    test_subwave<16><<<1, 32>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 32 * sizeof(float), hipMemcpyDeviceToHost));
    // Each 16-lane group should sum to 0+1+..+15=120 and 16+..+31=376
    printf("subwave<16> lane0=%.0f (expect 120), lane16=%.0f (expect 376)  %s\n",
           h[0], h[16],
           (h[0] == 120.0f && h[16] == 376.0f) ? "PASS" : "FAIL");

    test_subwave<8><<<1, 32>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 32 * sizeof(float), hipMemcpyDeviceToHost));
    // 8-lane groups: 0+..+7=28, 8+..+15=92, 16+..+23=156, 24+..+31=220
    printf("subwave<8> lanes 0,8,16,24 = %.0f, %.0f, %.0f, %.0f (expect 28,92,156,220)  %s\n",
           h[0], h[8], h[16], h[24],
           (h[0]==28.f && h[8]==92.f && h[16]==156.f && h[24]==220.f) ? "PASS" : "FAIL");

    test_subwave<4><<<1, 32>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 32 * sizeof(float), hipMemcpyDeviceToHost));
    // 4-lane: 0+1+2+3=6, 4+5+6+7=22, ...
    printf("subwave<4> lanes 0,4,8 = %.0f, %.0f, %.0f (expect 6,22,38)  %s\n",
           h[0], h[4], h[8],
           (h[0]==6.f && h[4]==22.f && h[8]==38.f) ? "PASS" : "FAIL");

    test_subwave<2><<<1, 32>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 32 * sizeof(float), hipMemcpyDeviceToHost));
    // 2-lane: 0+1=1, 2+3=5, 4+5=9, ...
    printf("subwave<2> lanes 0,2,4 = %.0f, %.0f, %.0f (expect 1,5,9)  %s\n",
           h[0], h[2], h[4],
           (h[0]==1.f && h[2]==5.f && h[4]==9.f) ? "PASS" : "FAIL");

    test_block<<<1, 256>>>(d); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h, d, 256 * sizeof(float), hipMemcpyDeviceToHost));
    // 0+1+...+255 = 255*256/2 = 32640
    bool block_ok = true;
    for (int i = 0; i < 256; i++) if (h[i] != 32640.0f) { block_ok = false; break; }
    printf("block_reduce_sum_256 lane0=%.0f, lane255=%.0f (expect 32640)  %s\n",
           h[0], h[255], block_ok ? "PASS" : "FAIL");

    CHECK(hipFree(d));
    return 0;
}
