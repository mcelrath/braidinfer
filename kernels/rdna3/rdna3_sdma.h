// SPDX-License-Identifier: MIT
// rdna3_sdma.h — SDMA threshold awareness and path-choice for P2P transfers
// on gfx1100 / ROCm 7.2.x.
//
// Empirical basis (exterior_algebra results/sdma_latency_curve.json, n=200 iters/size,
// 2026-05-12, gfx1100 RX 7900 XTX x2):
//
//   transfer_size  p2p_latency_us_median  p2p_throughput_GB/s  path
//   64 B           14.38                  0.004                blit kernel
//   256 B          14.84                  0.017                blit kernel
//   1 KB           14.84                  0.069                blit kernel
//   4 KB           14.92                  0.275                blit kernel
//   16 KB          16.36                  1.001                blit kernel (BW onset)
//   64 KB          24.08                  2.722                blit kernel
//   256 KB         54.16                  4.840                blit kernel
//   1 MB           176.28                 5.948                SDMA threshold
//   4 MB           664.44                 6.313                SDMA
//
// Threshold source: rocclr/utils/flags.hpp:220 ROC_P2P_SDMA_SIZE = 1048576.
// Below threshold: blit kernel (compute queue, ~15 µs floor for any size).
// At/above threshold: SDMA (separate hardware queue, concurrent with compute).
//
// For MoE expert dispatch (typical tile 1-64 KB), P2P transfers are 14.9-24 µs,
// blit-kernel-bound NOT SDMA-bound. SDMA only helps at ≥1 MB transfers.
//
// Cross-reference: results/sdma_verify.json confirmed SDMA actually runs
// concurrent with compute (overlap_ratio=1.007 at 512 MB transfer).
//
// Kernel-initiated SDMA: not user-accessible without amdkfd patch (the SDMA
// doorbell BAR is mapped into a queue's doorbell_signal, not into device VA).
// See bd memory `phase-1-amdkfd-doorbell-mmap-feasibility`.

#pragma once

#include <stddef.h>

#define RDNA3_SDMA_THRESHOLD_BYTES   ((size_t)1048576)   // 1 MB; source: rocclr flags.hpp:220
#define RDNA3_BLIT_KERNEL_FLOOR_US   15                  // floor for any sub-threshold transfer
#define RDNA3_SDMA_MIN_USEFUL_BYTES  ((size_t)1048576)   // below this, blit kernel always wins on latency

typedef enum {
    RDNA3_P2P_PATH_DIRECT_WRITE,     // < 256 B: just write through MTYPE_UC peer mapping
    RDNA3_P2P_PATH_BLIT_KERNEL,      // 256 B .. 1 MB: hipMemcpyPeerAsync uses blit kernel
    RDNA3_P2P_PATH_SDMA              // ≥ 1 MB: hipMemcpyPeerAsync uses SDMA (concurrent with compute)
} rdna3_p2p_path_t;

// Recommend the lowest-latency path for a P2P transfer of `bytes`.
// Not a runtime dispatch — caller chooses primitives accordingly.
static inline rdna3_p2p_path_t rdna3_choose_p2p_path(size_t bytes) {
    if (bytes < 256)                            return RDNA3_P2P_PATH_DIRECT_WRITE;
    if (bytes < RDNA3_SDMA_THRESHOLD_BYTES)     return RDNA3_P2P_PATH_BLIT_KERNEL;
    return RDNA3_P2P_PATH_SDMA;
}

// Estimated latency (microseconds) for a P2P transfer of `bytes`, derived
// from the curve above. For decision-making, not for precise budgeting.
static inline double rdna3_estimate_p2p_latency_us(size_t bytes) {
    if (bytes < 256)                            return 1.2;    // peer-write hw floor (results/cross_gpu_write_latency.json)
    if (bytes < 16384)                          return (double)RDNA3_BLIT_KERNEL_FLOOR_US;
    if (bytes < RDNA3_SDMA_THRESHOLD_BYTES)     return RDNA3_BLIT_KERNEL_FLOOR_US + bytes / 6300.0; // ~6.3 GB/s
    return bytes / 6300.0;                                     // SDMA scales with size
}
