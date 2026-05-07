// Tiny correctness probe: 32-lane wave reduction with permlanex16.
// Each lane starts with its own lane ID as a float. Expected sum = 0+1+...+31 = 496.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(c) do{auto e=(c); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));std::exit(1);}}while(0)

template<int N>
__device__ __forceinline__ float dpp_row_xmask_add(float v) {
    union { float f; int i; } a, b;
    a.f = v;
    constexpr int ctrl = 0x160 | (N & 0xF);
    b.i = __builtin_amdgcn_update_dpp(0, a.i, ctrl, 0xF, 0xF, 0);
    return v + b.f;
}

__global__ void k_test(float* out, int* lane_dump) {
    int lane = threadIdx.x;
    float v = (float)lane;
    // Stage A: row_xmask butterfly reductions
    v = dpp_row_xmask_add<8>(v);
    v = dpp_row_xmask_add<4>(v);
    v = dpp_row_xmask_add<2>(v);
    v = dpp_row_xmask_add<1>(v);
    // After A, lane 0 of row 0 should have 0+1+..+15 = 120
    //          lane 0 of row 1 should have 16+17+..+31 = 376
    out[lane] = v;
    lane_dump[lane] = lane;

    // Stage B: combine halves
    union { float f; int i; } a, b;
    a.f = v;
    b.i = __builtin_amdgcn_permlanex16((unsigned)a.i, (unsigned)a.i, 0u, 0u, true, true);
    out[32 + lane] = b.f;          // what came from the other half?
    out[64 + lane] = v + b.f;      // final sum (expected 496 in lane 0)
}

int main() {
    float* d_out; int* d_lane;
    CHECK(hipMalloc(&d_out, 96 * sizeof(float)));
    CHECK(hipMalloc(&d_lane, 32 * sizeof(int)));
    k_test<<<1, 32>>>(d_out, d_lane);
    CHECK(hipDeviceSynchronize());
    float h[96];
    CHECK(hipMemcpy(h, d_out, 96 * sizeof(float), hipMemcpyDeviceToHost));
    printf("After dpp_row_shr (per-lane v):\n");
    for (int i = 0; i < 32; i++) printf("  lane %2d: %.1f\n", i, h[i]);
    printf("\nAfter permlanex16 (the swapped value):\n");
    for (int i = 0; i < 32; i++) printf("  lane %2d: %.1f\n", i, h[32 + i]);
    printf("\nFinal sum (v + b.f):\n");
    for (int i = 0; i < 32; i++) printf("  lane %2d: %.1f\n", i, h[64 + i]);
    printf("\nExpected: lane 0..15 should each see %.0f after adding the swapped 16..31 sum.\n", 496.0f);
    return 0;
}
