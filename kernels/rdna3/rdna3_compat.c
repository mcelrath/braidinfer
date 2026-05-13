// SPDX-License-Identifier: MIT
// rdna3_compat.c — runtime probe for hw/ROCm drift.

#include "rdna3_compat.h"

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_rocm_version(char* buf, size_t buflen) {
    FILE* f = fopen("/opt/rocm/.info/version", "r");
    if (!f) f = fopen("/opt/rocm/.info/version-dev", "r");
    if (!f) return 1;
    if (!fgets(buf, (int)buflen, f)) { fclose(f); return 1; }
    fclose(f);
    // Strip trailing newline
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = 0;
    return 0;
}

int rdna3_compat_init(const rdna3_validation_context_t* claimed) {
    int drift = 0;

    hipDeviceProp_t prop;
    hipError_t herr = hipGetDeviceProperties(&prop, 0);
    if (herr != hipSuccess) {
        fprintf(stderr, "rdna3_compat: hipGetDeviceProperties failed (%d)\n", herr);
        drift = 1;
    } else {
        // gfx name is in prop.gcnArchName on AMD HIP.
        if (strncmp(prop.gcnArchName, RDNA3_COMPAT_EXPECTED_HW, strlen(RDNA3_COMPAT_EXPECTED_HW)) != 0) {
            fprintf(stderr,
                "rdna3_compat: HW DRIFT — detected '%s', expected '%s', claimed '%s'\n",
                prop.gcnArchName, RDNA3_COMPAT_EXPECTED_HW,
                claimed ? claimed->hw : "(null)");
            drift = 1;
        }
        if (claimed && strncmp(prop.gcnArchName, claimed->hw, strlen(claimed->hw)) != 0) {
            fprintf(stderr,
                "rdna3_compat: HW vs CLAIMED MISMATCH — detected '%s', claimed '%s'\n",
                prop.gcnArchName, claimed->hw);
            drift = 1;
        }
    }

    char rocm_buf[64];
    if (read_rocm_version(rocm_buf, sizeof(rocm_buf)) == 0) {
        if (strncmp(rocm_buf, RDNA3_COMPAT_EXPECTED_ROCM, strlen(RDNA3_COMPAT_EXPECTED_ROCM)) != 0) {
            fprintf(stderr,
                "rdna3_compat: ROCm DRIFT — detected '%s', expected prefix '%s', claimed '%s'\n",
                rocm_buf, RDNA3_COMPAT_EXPECTED_ROCM,
                claimed ? claimed->rocm : "(null)");
            drift = 1;
        }
    } else {
        fprintf(stderr, "rdna3_compat: WARN — could not read /opt/rocm/.info/version\n");
    }

#ifdef RDNA3_VALIDATE_STRICT
    if (drift) {
        fprintf(stderr, "rdna3_compat: STRICT mode — aborting due to compat drift\n");
        abort();
    }
#endif

    return drift;
}

#ifdef RDNA3_COMPAT_AUTOINIT
__attribute__((constructor))
static void rdna3_compat_autoinit(void) {
    rdna3_compat_init(NULL);
}
#endif
