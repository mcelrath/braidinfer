// SPDX-License-Identifier: MIT
// rdna3_signal_uc_device.h — UC device-memory buffer allocation for
// cross-GPU peer signaling on gfx1100.
//
// CANONICAL PATTERN (GFX1100_ARCH.md §5.5 Rule 1a — UC device buffer):
//   Allocate via hipExtMallocWithFlags(hipDeviceMallocUncached). The
//   resulting buffer lives in the source GPU's VRAM but is mapped
//   MTYPE=UC (uncached), so reads/writes bypass GPU L2 and propagate
//   directly through HBM. After hipDeviceEnablePeerAccess, peer GPUs
//   can address the buffer via P2P; AGENT-scope stores from the peer
//   land in the source GPU's UC memory without any cache invalidate
//   required.
//
// CANONICAL HAZARD (§11.4, V7 reproducer):
//   Cross-GPU peer-VRAM UC stores immediately before atomic_block_barrier
//   in a persistent cooperative kernel under multi-GPU PCIe pressure
//   trigger the vscnt-drain wedge (3/10 rate at 4-GPU, 2026-05-13).
//   This header allocates the storage; the discipline for *how* to write
//   it (4fg.5 deferral pattern via rdna3_peer_write_deferred) lives in
//   rdna3/rdna3_peer.h.
//
// FORBIDDEN — do not allocate cross-agent signal storage via:
//   - plain hipMalloc       (cached; UC store ordering disciplines don't apply)
//   - hipMallocManaged       (managed memory; gfx1100 P2P semantics not
//                             validated for managed pages)
//   - hipHostMalloc          (host-mapped; that's the SEPARATE §5.5 Rule 1b
//                             path covered by rdna3_signal_host_mapped.h)

#pragma once
#ifndef BRAIDINFER_RDNA3_SIGNAL_UC_DEVICE_H
#define BRAIDINFER_RDNA3_SIGNAL_UC_DEVICE_H

#include <hip/hip_runtime.h>
#include <cstddef>

namespace braidinfer { namespace rdna3 {

// Allocate a UC-mapped buffer in the current device's VRAM.
//
//   device_ptr  — output: device virtual pointer in this GPU's address
//                 space. Peers gain access via hipDeviceEnablePeerAccess
//                 (caller's responsibility — typically once per
//                 (src, dst) pair at init time).
//   bytes       — size in bytes; HIP rounds up to the device page size.
//
// On success, the buffer is zero-initialized via hipMemset so peer
// polls from zero are well-defined.
//
// Returns hipError_t directly. On failure, *device_ptr is left null.
inline hipError_t signal_uc_device_alloc(
    void** device_ptr,
    size_t bytes)
{
    *device_ptr = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        device_ptr, bytes, hipDeviceMallocUncached);
    if (err != hipSuccess) {
        *device_ptr = nullptr;
        return err;
    }
    err = hipMemset(*device_ptr, 0, bytes);
    if (err != hipSuccess) {
        (void)hipFree(*device_ptr);
        *device_ptr = nullptr;
        return err;
    }
    return hipSuccess;
}

// Free a buffer allocated by signal_uc_device_alloc.
inline hipError_t signal_uc_device_free(void* device_ptr) {
    if (device_ptr == nullptr) return hipSuccess;
    return hipFree(device_ptr);
}

}}  // namespace braidinfer::rdna3

#endif  // BRAIDINFER_RDNA3_SIGNAL_UC_DEVICE_H
