// SPDX-License-Identifier: MIT
// rdna3_timing.h — measurement primitives for the rdna3 library.
//
// Two services:
//   (1) CPU-bracketed timing with affinity guards. Refuses to measure unless
//       the caller is on CPU 55 SCHED_FIFO (configurable). Without this guard
//       reported latencies drift up to 2x (see exterior_algebra/results/
//       megakernel_under_load_rt.json vs the prior no-pinning run).
//   (2) Percentile-vector emitter. Library policy: never report a single number;
//       always emit min/p50/p90/p99/max/n to standardize comparisons across
//       investigations.
//
// GPU-clock note: gfx1100 clock64() is unreliable for spin-relative timing —
// SCLK throttles when the GPU is mostly idle, and a calibration kernel using
// s_sleep reports ~18 MHz instead of the real ~2.5 GHz (see
// exterior_algebra/results/dual_megakernel_gpuclock.json side finding).
// Use clock64() only for snapshot reads at known compute-busy points, never
// for tight-loop calibration. The CPU bracket is canonical for sub-microsecond
// RTT measurements.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Verify caller's CPU affinity + scheduler. Returns 0 if ready to measure,
// nonzero with stderr warning otherwise.
// expected_cpu = -1 to skip CPU check; require_fifo = 0 to skip scheduler check.
int rdna3_timing_check_affinity(int expected_cpu /* 55 */, int require_fifo /* 1 */);

// Tight CPU bracket: returns elapsed microseconds between begin/end.
static inline uint64_t rdna3_timing_begin(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline double rdna3_timing_end_us(uint64_t begin_ns) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    return (double)(now - begin_ns) / 1000.0;
}

// Percentile vector. Library callers fill samples[], then call emit.
typedef struct {
    double min_us, p50_us, p90_us, p99_us, max_us;
    size_t n;
} rdna3_timing_envelope_t;

// Computes the envelope from samples_us[]. Sorts samples_us[] in place.
void rdna3_timing_compute(double* samples_us, size_t n, rdna3_timing_envelope_t* out);

// Emits the envelope as one JSON line to fp. Caller supplies the wrapping label/keys.
static inline void rdna3_timing_emit_json(FILE* fp, const char* name,
                                          const rdna3_timing_envelope_t* env) {
    fprintf(fp,
        "{\"name\":\"%s\",\"min_us\":%.3f,\"p50_us\":%.3f,\"p90_us\":%.3f,"
        "\"p99_us\":%.3f,\"max_us\":%.3f,\"n\":%zu}\n",
        name, env->min_us, env->p50_us, env->p90_us, env->p99_us,
        env->max_us, env->n);
}

#ifdef __cplusplus
}
#endif
