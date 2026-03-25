#pragma once
#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_to_f32(unsigned short val) {
    unsigned int bits = ((unsigned int)val) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
