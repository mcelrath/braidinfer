// SPDX-License-Identifier: MIT
// rdna3_coord.h — Umbrella header for RDNA3 GPU-side coordination primitives.
//
// Pulls in the full set of coordination headers needed by device kernels that
// perform cross-CU, cross-wave, or cross-GPU synchronization and signaling.
// Include this instead of individual headers when you need two or more of the
// sub-components.
//
// EXCLUDED intentionally:
//   rdna3_persistent.h     — requires BRAIDINFER_PERSISTENT_WORKER gate macro
//                            (single-entry-point cooperative kernel contract);
//                            include it explicitly after defining the macro.
//   rdna3_persistent_protocol.h — companion to rdna3_persistent.h; same gate.
//   rdna3_compat.h         — host C, not device HIP.
//   rdna3_timing.h         — host C instrumentation, not device HIP.
//
// Usage:
//   #include "rdna3/rdna3_coord.h"
//
// Or with an explicit -I path set to the kernels/ directory:
//   #include "rdna3/rdna3_coord.h"
//
// Two-tier library structure (preserved):
//   Tier 1 — synchronization primitives (barrier, peer)
//   Tier 2 — signaling / messaging (signal_host_mapped, signal_uc_device, sdma)
//   Tier 3 — performance envelope (perf_envelope — advisory, no code-gen)
//
// See individual headers for CANONICAL HAZARD notes (GFX1100_ARCH.md §11.4).

#ifndef BRAIDINFER_RDNA3_COORD_H
#define BRAIDINFER_RDNA3_COORD_H

// Tier 1 — cross-CU / cross-wave synchronization
#include "rdna3_barrier.h"
#include "rdna3_peer.h"

// Tier 2 — kernel<->host and kernel<->kernel signaling
#include "rdna3_signal_host_mapped.h"
#include "rdna3_signal_uc_device.h"
#include "rdna3_sdma.h"

// Tier 3 — performance envelope (advisory constants, no side effects)
#include "rdna3_perf_envelope.h"

#endif  // BRAIDINFER_RDNA3_COORD_H
