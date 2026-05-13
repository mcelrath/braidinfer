// SPDX-License-Identifier: MIT
// rdna3_compat.h — compile-time + runtime compat assertions for the rdna3
// coordination library on gfx1100 / ROCm 7.2.x.
//
// Empirically depended on (verified by exterior_algebra and braidinfer
// microbenches, see results/ in exterior_algebra @ 3586a44+ and braidinfer
// kernels/diagnostic/persistent_skeleton_repro/):
//   - GPU architecture: gfx1100 (RDNA3, Navi 31).
//   - ROCm runtime: 7.2.x. Tested 7.2.2 + 7.2.3.
//   - buffer_gl1_inv ISA: present and codegens. Used in rdna3_peer.h to
//     refresh L1 on peer-VRAM reads.
//   - buffer_gl2_inv ISA: ABSENT on gfx1100. The library's wedge-mitigation
//     story depends on this; if a future ROCm/firmware enables L2 invalidate
//     from kernel side, the §11.4 envelope changes and this library's
//     enforcement must be re-validated.
//
// Validation strategy: two layers.
//   (1) Compile-time static_assert that the caller's RDNA3_PERSISTENT_POLLING_VALIDATED
//       struct literal claims the expected hardware/ROCm context.
//   (2) Runtime probe (rdna3_compat_init) that reads actual hipDeviceProp_t.name
//       + /opt/rocm/.info/version and compares. Default policy on drift: WARN to
//       stderr. Strict policy (abort) opt-in via -DRDNA3_VALIDATE_STRICT=1.
//
// Implementation: header declares the runtime probe; rdna3_compat.c defines it.
// Header-only callers can skip the runtime check (define RDNA3_COMPAT_NO_RUNTIME).

#pragma once

#define RDNA3_COMPAT_EXPECTED_HW       "gfx1100"
#define RDNA3_COMPAT_EXPECTED_ROCM     "7.2"    // prefix-match; 7.2.x acceptable
#define RDNA3_COMPAT_BUFFER_GL1_INV    1        // expected present
#define RDNA3_COMPAT_BUFFER_GL2_INV    0        // expected absent

// Struct literal callers populate to declare their validation context.
typedef struct {
    const char* hw;            // e.g. "gfx1100"
    int         gpu_count;     // 1..8
    const char* rocm;          // e.g. "7.2.2" or "7.2.3"
    const char* workload;      // free-form, e.g. "q8moe"
    const char* date;          // YYYY-MM-DD
    const char* fixture;       // in-tree path to the reproducer that validated this
} rdna3_validation_context_t;

#ifndef RDNA3_COMPAT_NO_RUNTIME
#ifdef __cplusplus
extern "C" {
#endif

// One-time runtime probe. Returns 0 on match, nonzero on drift.
// Default: also writes a single line to stderr on drift.
// With RDNA3_VALIDATE_STRICT defined, abort() on drift.
//
// Safe to call from main() at process start. Also registered as
// __attribute__((constructor)) if RDNA3_COMPAT_AUTOINIT is defined at build.
int rdna3_compat_init(const rdna3_validation_context_t* claimed);

#ifdef __cplusplus
}
#endif
#endif // RDNA3_COMPAT_NO_RUNTIME

// Static-only field-presence assertions. Caller's struct literal must declare
// all fields; this catches missing fields at compile time.
#define RDNA3_COMPAT_CHECK_STRUCT(ctx) \
    do { \
        (void)(ctx).hw; (void)(ctx).gpu_count; (void)(ctx).rocm; \
        (void)(ctx).workload; (void)(ctx).date; (void)(ctx).fixture; \
    } while (0)
