#pragma once
#include <hip/hip_runtime.h>

// 128-bit vector type for nontemporal loads (ext_vector_type works with __builtin_nontemporal_load)
typedef unsigned long long ull2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf16_to_f32(unsigned short val) {
    unsigned int bits = ((unsigned int)val) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
