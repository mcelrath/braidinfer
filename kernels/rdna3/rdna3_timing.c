// SPDX-License-Identifier: MIT
// rdna3_timing.c — implementation of the affinity check and percentile compute.

#define _GNU_SOURCE

#include "rdna3_timing.h"

#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int compare_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

void rdna3_timing_compute(double* samples_us, size_t n, rdna3_timing_envelope_t* out) {
    if (n == 0) {
        out->min_us = out->p50_us = out->p90_us = out->p99_us = out->max_us = 0.0;
        out->n = 0;
        return;
    }
    qsort(samples_us, n, sizeof(double), compare_double);
    out->min_us = samples_us[0];
    out->p50_us = samples_us[n / 2];
    out->p90_us = samples_us[(size_t)((n - 1) * 0.90)];
    out->p99_us = samples_us[(size_t)((n - 1) * 0.99)];
    out->max_us = samples_us[n - 1];
    out->n = n;
}

int rdna3_timing_check_affinity(int expected_cpu, int require_fifo) {
    int warns = 0;

    if (expected_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            if (!CPU_ISSET(expected_cpu, &set)) {
                fprintf(stderr, "rdna3_timing: WARN — not pinned to CPU %d; "
                        "latencies may drift >2x. Suggest: chrt -f 50 taskset -c %d ...\n",
                        expected_cpu, expected_cpu);
                warns++;
            }
            int n_allowed = CPU_COUNT(&set);
            if (n_allowed > 1) {
                fprintf(stderr, "rdna3_timing: WARN — affinity spans %d CPUs; pin to one for tight latency.\n",
                        n_allowed);
                warns++;
            }
        }
    }

    if (require_fifo) {
        int policy = sched_getscheduler(0);
        if (policy != SCHED_FIFO) {
            fprintf(stderr, "rdna3_timing: WARN — scheduler is %d, not SCHED_FIFO. "
                    "Suggest: chrt -f 50 ...\n", policy);
            warns++;
        }
    }

    return warns;
}
