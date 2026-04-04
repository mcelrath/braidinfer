// Common definitions shared between megakernel.hip and persistent_worker.hip.
// Instruction layout, helper functions, type aliases.
#pragma once
#include "opcodes.h"
#include "bf16_utils.h"
#include "quant_consts.h"

#define FLAG_NO_SYNC   0x80000000u  // bit 31: skip grid.sync() after this instruction
#define INST_SIZE_WORDS 17          // 17 u64s per instruction = 136 bytes

typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u32 inst_opcode(const u64* inst) {
    return (u32)(inst[0] & 0x7FFFFFFF);
}
__device__ __forceinline__ bool inst_no_sync(const u64* inst) {
    return (inst[0] & FLAG_NO_SYNC) != 0;
}
__device__ __forceinline__ u32 inst_grid_x(const u64* inst) {
    return (u32)(inst[0] >> 32);
}
__device__ __forceinline__ float* inst_fptr(const u64* inst, int idx) {
    return (float*)(inst[idx]);
}
__device__ __forceinline__ const unsigned short* inst_u16ptr(const u64* inst, int idx) {
    return (const unsigned short*)(inst[idx]);
}
__device__ __forceinline__ const void* inst_ptr(const u64* inst, int idx) {
    return (const void*)(inst[idx]);
}
__device__ __forceinline__ int inst_int(const u64* inst, int idx) {
    return (int)(inst[idx]);
}
__device__ __forceinline__ float inst_float(const u64* inst, int idx) {
    u32 bits = (u32)inst[idx];
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}
