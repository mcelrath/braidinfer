// SPDX-License-Identifier: MIT
// rdna3_signal_host_mapped.h — Host-mapped signal mailbox allocation for
// gfx1100 kernel<->host messaging (cooperative-worker progress counters,
// force_exit flags, watchdog telemetry, dispatch queue mailboxes).
//
// CANONICAL PATTERN (GFX1100_ARCH.md §5.5 Rule 1b — host-mapped UC):
//   Allocate via hipHostMalloc with the
//   hipHostMallocMapped | hipHostMallocCoherent flags. The kernel polls
//   the mapped pointer via VOLATILE reads (NOT __hip_atomic_load —
//   SYSTEM-scope atomic loads hang on gfx1100 host-mapped UC) and writes
//   it via AGENT-scope host_uc_store_agent (rdna3/rdna3_peer.h), NOT
//   SYSTEM scope (s_waitcnt_vscnt drain is the V7 wedge trigger).
//
// HAZARD ENVELOPE (advisory only — not strictly enforced):
//   Host-mapped UC polling in single-worker / 2-GPU configurations is
//   safe (V0/V5 persistent_skeleton_repro negative at n=10, 2026-05-13).
//   The §11.4 wedge requires cross-GPU peer-VRAM stores under multi-GPU
//   PCIe pressure (see rdna3/rdna3_peer.h). Host-mapped UC alone has
//   never reproduced the wedge in our test envelope.
//
//   Therefore this header documents the canonical allocation +
//   discipline but does NOT gate on a confirmation macro — accidental
//   misuse here is recoverable. The strict gate lives in
//   rdna3_persistent.h (the polling-loop wrapper) and rdna3_peer.h (the
//   cross-GPU primitives).

#pragma once
#ifndef BRAIDINFER_RDNA3_SIGNAL_HOST_MAPPED_H
#define BRAIDINFER_RDNA3_SIGNAL_HOST_MAPPED_H

#include <hip/hip_runtime.h>
#include <cstddef>

namespace braidinfer { namespace rdna3 {

// Allocate a host-mapped UC mailbox visible to both CPU and GPU.
//
//   host_ptr    — output: CPU virtual pointer (CPU writes/reads through this).
//   device_ptr  — output: GPU virtual pointer (kernel writes/reads through this).
//   bytes       — size in bytes; rounded up to host page size by HIP.
//
// Returns hipError_t directly; caller checks. On success, the buffer is
// zero-initialized via memset for predictable poll-from-zero semantics.
//
// Memory is automatically coherent — host writes are visible to the GPU
// without any cache-flush API call, and GPU AGENT-scope stores reach the
// host via PCIe posted-write semantics with no host-side invalidate
// required. See GFX1100_ARCH.md §5.5 Rule 1b for the underlying
// rationale.
inline hipError_t signal_host_mapped_alloc(
    void** host_ptr,
    void** device_ptr,
    size_t bytes)
{
    hipError_t err = hipHostMalloc(
        host_ptr, bytes,
        hipHostMallocMapped | hipHostMallocCoherent);
    if (err != hipSuccess) return err;

    err = hipHostGetDevicePointer(device_ptr, *host_ptr, /*flags=*/0u);
    if (err != hipSuccess) {
        (void)hipHostFree(*host_ptr);
        *host_ptr = nullptr;
        *device_ptr = nullptr;
        return err;
    }
    // Zero-init so volatile poll-from-zero is well-defined on first launch.
    for (size_t i = 0; i < bytes; ++i)
        reinterpret_cast<volatile unsigned char*>(*host_ptr)[i] = 0;
    return hipSuccess;
}

// Free a buffer allocated by signal_host_mapped_alloc. Pass the host
// pointer (the device pointer is just a mapping of the same allocation).
inline hipError_t signal_host_mapped_free(void* host_ptr) {
    if (host_ptr == nullptr) return hipSuccess;
    return hipHostFree(host_ptr);
}

}}  // namespace braidinfer::rdna3

#endif  // BRAIDINFER_RDNA3_SIGNAL_HOST_MAPPED_H
