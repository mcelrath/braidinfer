// Compute-path GPU-to-GPU memcpy kernel.
// Replaces hipMemcpyPeerAsync (SDMA) which PERMISSION_FAULTs on RDNA3 PCIe.
// Proven approach: GPU compute writes to remote VRAM work 100% reliably on gfx1100.
#include <hip/hip_runtime.h>

extern "C" __global__ void peer_copy_kernel(uint8_t* __restrict__ dst,
                                             const uint8_t* __restrict__ src,
                                             uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
