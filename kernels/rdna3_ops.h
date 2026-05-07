// RDNA3 (gfx1100, wave32) primitive umbrella header.
//
// Single include for all braidinfer::rdna3 primitives. Production callers
// should include THIS header instead of the individual sub-headers; the
// sub-headers exist for reference and for the microbenchmarks that produced
// the cycle counts cited in their docstrings.
//
// Sub-headers (all under namespace braidinfer::rdna3):
//
//   rdna3_memory.h   atomic_add_f32_hw / _cas, atomic_add_bf16_safe,
//                    gl0_invalidate / gl1_invalidate / gl01_invalidate,
//                    lds_pad_for_tpg<>. Notes on cp.async (NOT available
//                    on gfx1100) and per-callsite cache-invalidation rules.
//
//   rdna3_lane.h     wave_ballot, wave_any/all/count_predicate, lane_id,
//                    wave_id_in_block, lane_broadcast<>/lane_broadcast,
//                    lane_read_first, lane_write<>, wave_swizzle<>,
//                    wave_xor_butterfly<>, wave_quad_swap<>, wave_quad_reverse,
//                    wave32_reduce_max, subwave_reduce_max<>, subwave_tile<>.
//
//   rdna3_reduce.h   wave32_reduce_sum, subwave_reduce_sum<>,
//                    block_reduce_sum_256, dpp_row_xmask_add<>,
//                    permlanex16_swap_add. Foundation for compute.h.
//
//   rdna3_compute.h  WMMA fragment types (fragment_a_bf16/f16, fragment_b_*,
//                    fragment_c_f32), load_a_*/load_b_*, mma_sync_bf16/f16,
//                    store_c_f32 / store_c_atomic_f32, block_dot_f32<>,
//                    split_k_gemv_atomic_kernel<>. Includes rdna3_reduce.h.
//
//   rdna3_sync.h     fence_block / fence_device / fence_system_uc and
//                    release/acquire-only variants, barrier_workgroup /
//                    barrier_within_wave, atomic_block_barrier (~115-155x
//                    faster than cg::grid_group::sync at typical block
//                    counts), fast_grid_sync (drop-in safer wrapper).
//
// All measurements made on RX 7900 XTX (gfx1100, wave32, ROCm 7.1.x). The
// per-primitive docstrings cite the bench file under
// kernels/diagnostic/{reduce,lane,rdna3_compute,rdna3_memory,rdna3_sync}_bench/.
//
// Compile constraints (apply to ALL primitives in this library):
//   - Wave32 only. Mixing with -mwavefrontsize64 will silently miscompile
//     (e.g. wave_ballot's u32 width, DPP row mask layout).
//   - gfx1100 only. gfx1030 also has wave32 + WMMA32 path but lane mappings
//     differ; do not assume binary compatibility.
//   - Requires ROCm 7.1.x or newer LLVM that supports v_permlanex16,
//     row_xmask DPP modifier, ds_swizzle bitmask-perm. (gfx1100 = OK.)
//   - All DPP / swizzle ctrl values are compile-time constants because the
//     intrinsics require it; that's why the helpers are templated on N.

#pragma once

// Order: independent headers first, then dependents.
#include "rdna3_memory.h"
#include "rdna3_lane.h"
#include "rdna3_sync.h"
#include "rdna3_reduce.h"
#include "rdna3_compute.h"   // depends on rdna3_reduce.h
