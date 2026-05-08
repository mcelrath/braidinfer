// Per-op cycle profiling primitive for the persistent megakernel.
//
// Plan: ~/.claude/plans/PLAN-op-profile.md (epic braidinfer-xiu).
//
// When BRAIDINFER_OP_PROFILE is defined, each call site wraps the
// dispatch of one opcode with OP_PROFILE_BEGIN(op) / OP_PROFILE_END(op).
// Block 0 thread 0 ("leader" thread) samples wall_clock64() at start and
// end, atomicAdds the elapsed ticks and a unit call count to the per-
// opcode slot in the profile buffer. The other threads in the block do
// nothing — keeps atomic contention to one add per block per op.
//
// "One call" = one opcode dispatch including its post-op grid barrier.
// (See Macro Placement section of the plan.)
//
// When the flag is unset, both macros expand to no-ops; the field
// `q->op_profile` exists in WorkerQueue but is null. Zero production cost.
//
// Counter buffer: GPU-resident DeviceBuffer<u64> of size
// 2 * NUM_OPCODES_PROFILED. Slots are [cycles_total, call_count] per op.
// Allocated by Rust (op_profile.rs) BEFORE PersistentDispatch::init.
//
// Read-out: ONLY safe after the persistent worker has been shut down
// (Drop on PersistentDispatch). hipMemcpy D2H during cooperative kernel
// life deadlocks per kb 77r-2-1-dma-under-persistent-deadlocks-all-paths-
// 2026-05-07.
//
// `wall_clock64()` returns memrealtime ticks (s_memrealtime), NOT shader
// cycles. Convert to ns via hipDeviceAttributeWallClockRate.
//
// Native u64 HW atomic: verified hipcc emits `global_atomic_add_u64` (one
// instruction) for `atomicAdd((unsigned long long*) ptr, val)` on
// gfx1100/ROCm 7.x — no CAS-loop emulation.

#ifndef BRAIDINFER_OP_PROFILE_H
#define BRAIDINFER_OP_PROFILE_H

#include <hip/hip_runtime.h>
#include <stdint.h>

// Number of opcode slots reserved in the counter buffer. Sized large
// enough to cover the OP_* enum in kernels/opcodes.h with headroom.
// Layout: counters[2 * opcode + 0] = cycles_total
//         counters[2 * opcode + 1] = call_count
#define BRAIDINFER_OP_PROFILE_NUM_SLOTS 64

#ifdef BRAIDINFER_OP_PROFILE

// `wall_clock64()` (the documented HIP API) inlines to
// `__ockl_steadyctr_u64()` which the compiler treats as effectively pure.
// Use `__builtin_amdgcn_s_sendmsg_rtnl(0x83)` (the gfx11 raw intrinsic
// for MSG_RTN_GET_REALTIME) directly — verified via --save-temps on a
// standalone 5-line probe that two consecutive intrinsic calls produce
// two distinct sendmsg instructions (the intrinsic is NOT marked pure).
//
// Split-atomic accumulation: BEGIN does `atomicAdd(cycles_total, -t0)`;
// END does `atomicAdd(cycles_total, +t1)`. Net effect per call:
// (+t1 - t0) = dt. The two atomicAdds have visible memory side effects
// that depend on each sendmsg result, so the compiler cannot hoist or
// CSE either of them out of the enclosing loop. Verified via
// --save-temps on the persistent_worker.hip build: 2 sendmsg calls + 2
// `flat_atomic_add_u64` (cycles_total, no offset suffix = offset 0) +
// 5 `(global|flat)_atomic_add_u64 ... offset:8` (call_count, multiple
// clones from if/else branch cloning) = matches plan.
//
// Caveat: the cycles_total counter becomes briefly unsigned-wrapping
// between BEGIN and END. Host readback must only happen after kernel
// shutdown (no atomic ops in flight); see PLAN-op-profile.md §R4.
//
// (gfx1100 has NO `s_memrealtime` or `s_memtime` instruction — verified;
// assembler rejects them. The sendmsg-rtn path is the only way to read
// the realtime counter on this arch.)

#define __OP_PROFILE_TICKS()  ((uint64_t) __builtin_amdgcn_s_sendmsg_rtnl(0x83))

#define OP_PROFILE_BEGIN(opcode_id, profile_buf)                                 \
    if (threadIdx.x == 0 && (profile_buf) != nullptr) {                          \
        uint64_t __op_t0 = __OP_PROFILE_TICKS();                                 \
        atomicAdd((unsigned long long*) &(profile_buf)[2 * (opcode_id)    ],     \
                  (unsigned long long) (0ULL - __op_t0));                        \
    }

#define OP_PROFILE_END(opcode_id, profile_buf)                                   \
    if (threadIdx.x == 0 && (profile_buf) != nullptr) {                          \
        uint64_t __op_t1 = __OP_PROFILE_TICKS();                                 \
        atomicAdd((unsigned long long*) &(profile_buf)[2 * (opcode_id)    ],     \
                  (unsigned long long) __op_t1);                                 \
        atomicAdd((unsigned long long*) &(profile_buf)[2 * (opcode_id) + 1],     \
                  (unsigned long long) 1ULL);                                    \
    }

#else

#define OP_PROFILE_BEGIN(opcode_id, profile_buf) do { } while (0)
#define OP_PROFILE_END(opcode_id, profile_buf)   do { } while (0)

#endif  // BRAIDINFER_OP_PROFILE

#endif  // BRAIDINFER_OP_PROFILE_H
