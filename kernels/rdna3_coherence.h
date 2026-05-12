// kernels/rdna3_coherence.h — Cross-agent coherence primitives for gfx1100.
//
// Documented in composable_kernel/GFX1100_ARCH.md §5.5.
#ifndef BRAIDINFER_RDNA3_COHERENCE_H
#define BRAIDINFER_RDNA3_COHERENCE_H

#include <hip/hip_runtime.h>

namespace braidinfer { namespace rdna3 {

// AGENT-scope atomic store to host-mapped UC memory. Use for kernel->host
// signaling where the value just needs to land in system RAM via PCIe.
//
// IMPORTANT — do not use SYSTEM scope here even though the target IS host
// memory. On gfx1100, SYSTEM-scope atomic_store emits an s_waitcnt_vscnt
// for cross-agent ordering, but host-mapped UC writes go through PCIe
// independently of any GPU cache hierarchy. The waitcnt provides no
// ordering benefit and is a documented multi-GPU hang source (per kb
// braidinfer-multigpu-persistent-worker-shutdown-wedge-analysis: stalls
// indefinitely if another PCIe write is in flight before an
// atomic_block_barrier).
//
// Per GFX1100_ARCH.md §5.5 Rule 8: writes from a GPU kernel to host-mapped
// UC memory need NO release fence with SYSTEM scope on gfx1100 — PCIe
// posted-write semantics already enforce that the write reaches host
// memory. AGENT scope is sufficient and avoids the s_waitcnt_vscnt trap.
// Helper to suppress T-deduction on the value argument; `T` is deduced from
// the pointer only. Lets callers pass volatile-qualified pointers (e.g.
// WatchdogState fields) without having to spell the template argument.
template <typename T> struct __host_uc_store_identity { using type = T; };

template <typename T>
__device__ __forceinline__ void host_uc_store_agent(
    volatile T* ptr,
    typename __host_uc_store_identity<T>::type value)
{
    __hip_atomic_store(const_cast<T*>(ptr), value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

}}  // namespace braidinfer::rdna3

#endif  // BRAIDINFER_RDNA3_COHERENCE_H
