// SPDX-License-Identifier: MIT
// rdna3_peer.h — Cross-GPU peer write/read coordination primitives for gfx1100.
//
// CANONICAL HAZARD (composable_kernel/GFX1100_ARCH.md §11.4, V7 reproducer
// kernels/diagnostic/persistent_skeleton_repro/, 3/10 wedge rate at 4-GPU):
//
//   ANY release-fenced UC store sequence under multi-GPU PCIe pressure can
//   stall on s_waitcnt_vscnt drain. The hazard fires not just AT the next
//   atomic_block_barrier (the typical observable) but at ANY subsequent
//   store that emits a release fence. atomic_block_barrier itself is
//   innocent; the broken thing is the vscnt counter's failure to decrement
//   when multiple GPUs have UC stores in flight concurrently.
//
//   Empirical envelope (2026-05-13):
//     V0 minimal (poll+barrier+ack, no cross-GPU UC):    0/10 wedged
//     V5 (V0 + outer-loop watchdog UC store):             0/10 wedged
//     V7 (V0 + pre-barrier cross-GPU peer-UC store):      3/10 wedged
//   The hazard requires cross-GPU peer-VRAM stores; bare host-mapped UC
//   stores in single-worker minimal skeletons do NOT trigger it.
//
// MITIGATION — the 4fg.5 deferral pattern:
//   Move the cross-GPU UC store to AFTER the next barrier where the host
//   actually needs the value, OR restructure so the cross-GPU store
//   happens at a kernel-boundary (§5.5 Rule 1d). Use the
//   rdna3_peer_write_deferred macro below to mark sites where the
//   deferred-write pattern applies.
//
// Production fix-commits demonstrating the deferral pattern at scale:
//   8cf8084  4fg.5  post-poll shutdown barrier — defer queue->done write
//   37be418  pky.2  op_moe_ffn_remote final coop_copy — barrier-less peer write
//   b503159  pky.2  op_moe_dispatch UC stage   — stage in cached gpu0_acc, single copy
//
// Retracted/superseded bd memories (kept for archival reference; do NOT cite
// as current guidance):
//   rdna3-atomic-block-barrier-multi-gpu-fundamental-issue (May 7)  — wrong
//   rdna3-atomic-block-barrier-cg-grid-group-sync          (May 7)  — wrong
//   rdna3-atomic-block-barrier-multi-gpu-synthetic-passes   (May 7) — incomplete
// Superseded by:
//   correction-rdna3-atomic-block-barrier-multi-gpu-wedge-2026-05-12
//   phase-0b-wedge-localized-to-inner-poll-2026-05-13
//   joint-verdict-rdna3-multigpu-vscnt-wedge-2026-05-12
//
// FORBIDDEN patterns (use rdna3_peer_write_deferred / §5.5 Rule 1a-d instead):
//   - Cross-GPU atomics on peer VRAM (atomicAdd to a pointer in another GPU's
//     address space): undefined on gfx1100 per §5.5 Rule 2.
//   - __threadfence_system on peer VRAM: compiles same as __threadfence()
//     on gfx1100 (no buffer_gl2_inv ISA instruction); does not enforce
//     cross-GPU ordering.
//   - Bare cross-GPU peer-VRAM UC store immediately before atomic_block_barrier
//     in a persistent cooperative kernel under multi-GPU PCIe pressure (V7).

#pragma once
#ifndef BRAIDINFER_RDNA3_PEER_H
#define BRAIDINFER_RDNA3_PEER_H

#include <hip/hip_runtime.h>

namespace braidinfer { namespace rdna3 {

// Helper to suppress T-deduction on the value argument; `T` is deduced from
// the pointer only. Lets callers pass volatile-qualified pointers (e.g.
// WatchdogState fields) without having to spell the template argument.
template <typename T> struct __host_uc_store_identity { using type = T; };

// AGENT-scope atomic store to host-mapped UC memory. Use for kernel->host
// signaling where the value just needs to land in system RAM via PCIe.
//
// DO NOT use SYSTEM scope here even though the target IS host memory. On
// gfx1100, SYSTEM-scope atomic_store emits an s_waitcnt_vscnt for
// cross-agent ordering, but host-mapped UC writes go through PCIe
// independently of any GPU cache hierarchy. The waitcnt provides no
// ordering benefit and is a documented multi-GPU hang source (V7
// reproducer, 3/10 wedge rate at 4-GPU).
//
// Per GFX1100_ARCH.md §5.5 Rule 8: writes from a GPU kernel to host-mapped
// UC memory need NO release fence with SYSTEM scope on gfx1100 — PCIe
// posted-write semantics already enforce that the write reaches host
// memory. AGENT scope is sufficient and avoids the s_waitcnt_vscnt trap.
template <typename T>
__device__ __forceinline__ void host_uc_store_agent(
    volatile T* ptr,
    typename __host_uc_store_identity<T>::type value)
{
    __hip_atomic_store(const_cast<T*>(ptr), value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// Cross-GPU peer-VRAM AGENT-scope store. Use for kernel-to-peer-GPU
// signaling where the value lands in the peer GPU's UC memory via P2P.
// Same semantics as host_uc_store_agent: AGENT scope, no SYSTEM-scope
// vscnt drain. Safe in non-barrier-adjacent contexts.
//
// HAZARD: if this store is followed by atomic_block_barrier in a persistent
// cooperative kernel under multi-GPU PCIe pressure, the V7 wedge fires
// intermittently (~30% rate at 4-GPU). Use the deferral pattern: place
// the store after the barrier OR at a kernel-launch boundary.
template <typename T>
__device__ __forceinline__ void peer_uc_store_agent(
    volatile T* peer_ptr,
    typename __host_uc_store_identity<T>::type value)
{
    __hip_atomic_store(const_cast<T*>(peer_ptr), value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

}}  // namespace braidinfer::rdna3

// rdna3_peer_write_deferred — codifies the 4fg.5 deferral pattern.
//
// The macro doesn't change runtime behavior; it's a grep-able marker that
// the caller has acknowledged the §11.4 hazard and structured the code so
// the peer-UC store occurs AFTER the next barrier where the value is
// actually needed. The wrapper makes a deliberate authoring decision
// reviewable: any peer-VRAM UC store sequence followed by a barrier
// should either use this macro to mark the deferral, or be flagged as a
// §11.4 review item.
//
// Typical use (mirrors production commit 37be418 / b503159):
//
//   __threadfence();                          // flush cached buffer
//   rdna3_peer_write_deferred(out_p2p, lo, gupd, /*via*/ memcpy_loop);
//   __threadfence();                          // ensure PCIe drain at exit
//   // NO atomic_block_barrier here. Next barrier is at a kernel boundary.
//
// (Implementations vary by use case — sometimes a coop_copy, sometimes a
// straight-line memcpy; the contract is "no atomic_block_barrier between
// this store and the next kernel exit".)
#define rdna3_peer_write_deferred(dst, src, count, BODY) \
    do { \
        BODY((dst), (src), (count)); \
    } while (0)

#endif  // BRAIDINFER_RDNA3_PEER_H
