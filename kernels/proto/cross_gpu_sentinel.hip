// braidinfer-yef5 Phase 0 GATE — Cross-GPU sentinel handoff microbench.
//
// Validates whether a GPU-side NaN-sentinel signal can cross the PCIe link
// from GPU0 (writer) to GPU1 (reader) on gfx1100, given three memory variants:
//
//  A (el1f-repro / harness sanity):
//    Sentinel slot in GPU0's own VRAM (hipMalloc, default MTYPE_NC cached).
//    GPU0: AGENT-RELEASE atomic store.
//    GPU1: spin-load via peer access until != qNaN, with wall_clock64 timeout.
//    EXPECTED: TIMEOUT/MISS — GPU0's L2-cached write is NOT seen by GPU1's
//    MTYPE_UC peer read (el1f hazard; L2 not flushed cross-link on gfx1100).
//
//  B (UC-write-bypass / production candidate):
//    Sentinel slot in host-mapped UC (hipHostMalloc Portable+Coherent).
//    Both GPUs map via hipHostGetDevicePointer.
//    GPU0: AGENT-RELEASE store.
//    GPU1: spin-load until != qNaN, same timeout.
//    EXPECTED: SEE — host UC memory bypasses GPU L2 on both sides.
//
//  C (cached + drain / cost gate):
//    GPU0's cached VRAM slot as in A. GPU0 writes then does a producer-side
//    drain (fence_device() + scalar readback of the slot to force L2 flush).
//    GPU1 should SEE after the drain. Report per-signal drain cost in cycles.
//
// Each variant runs N=100 COLD trials. Output:
//   variant | see-rate | median latency (cycles) | drain cost (C only)
//
// Build:
//   hipcc --offload-arch=gfx1100 -O3 -std=c++17 \
//     -I kernels \
//     kernels/proto/cross_gpu_sentinel.hip -o /tmp/cross_gpu_sentinel
// Run:
//   python3 scripts/launch-gpu.py -g 2 --timeout 600 -- /tmp/cross_gpu_sentinel

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unistd.h>   // usleep

#include "../rdna3/rdna3_barrier.h"

static const unsigned int SENTINEL = 0x7fc00000u; // canonical qNaN

// Timeout in wall-clock cycles (~1e8 ≈ 100ms on ~1GHz shader clock)
static const unsigned long long TIMEOUT_CYC = 100000000ULL;

// ============================================================
// VARIANT A — GPU0 cached VRAM, peer access read from GPU1
// ============================================================

// GPU0 writer kernel (single block, single thread)
__global__ void varA_writer(unsigned int* slot, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __hip_atomic_store(slot, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// GPU1 reader kernel: spin-poll the peer slot with timeout
// Returns: cycles until observed (or TIMEOUT_CYC if timed out), and 0/1 seen flag
__global__ void varA_reader(const unsigned int* peer_slot,
                             unsigned long long* out_latency,
                             unsigned int* out_seen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0 = wall_clock64();
        unsigned int v;
        for (;;) {
            // volatile load — ensures we go to memory (peer UC mapping)
            v = __hip_atomic_load(peer_slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            if (v != SENTINEL) break;
            if (wall_clock64() - t0 >= TIMEOUT_CYC) break;
            __builtin_amdgcn_s_sleep(1);
        }
        *out_latency = wall_clock64() - t0;
        *out_seen    = (v != SENTINEL) ? 1u : 0u;
    }
}

// ============================================================
// VARIANT B — host-mapped UC, both GPUs use their own mapped pointer
// ============================================================

// GPU0 writer to host-UC mapped pointer
__global__ void varB_writer(unsigned int* uc_slot, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // AGENT-RELEASE store. For host UC memory the write bypasses GPU L2
        // and goes directly to the coherent host memory path.
        __hip_atomic_store(uc_slot, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// GPU1 reader from host-UC mapped pointer
__global__ void varB_reader(const unsigned int* uc_slot,
                             unsigned long long* out_latency,
                             unsigned int* out_seen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0 = wall_clock64();
        unsigned int v;
        for (;;) {
            v = __hip_atomic_load(uc_slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            if (v != SENTINEL) break;
            if (wall_clock64() - t0 >= TIMEOUT_CYC) break;
            __builtin_amdgcn_s_sleep(1);
        }
        *out_latency = wall_clock64() - t0;
        *out_seen    = (v != SENTINEL) ? 1u : 0u;
    }
}

// ============================================================
// VARIANT C — GPU0 cached VRAM + producer-side drain
// ============================================================

// GPU0 writer: write sentinel value, then fence+readback (drain) to force L2 flush
// out_drain_cost: cycles spent on the drain operation
__global__ void varC_writer(unsigned int* slot, unsigned int value,
                             unsigned long long* out_drain_cost) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Step 1: AGENT-RELEASE store (same as variant A)
        __hip_atomic_store(slot, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

        // Step 2: drain — fence to flush stores, then readback to confirm
        unsigned long long drain_t0 = wall_clock64();

        // fence_device() = __threadfence() = agent-scope acq_rel fence
        // Flushes this thread's stores to L2.
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");

        // Readback: load the slot back to confirm it reached L2 (forces
        // the write to complete from L2's perspective). Use acquire so the
        // compiler cannot reorder past it.
        unsigned int confirm = __hip_atomic_load(slot, __ATOMIC_ACQUIRE,
                                                  __HIP_MEMORY_SCOPE_AGENT);
        (void)confirm; // prevent dead-code elimination

        *out_drain_cost = wall_clock64() - drain_t0;
    }
}

// GPU1 reader for variant C (same as A)
__global__ void varC_reader(const unsigned int* peer_slot,
                             unsigned long long* out_latency,
                             unsigned int* out_seen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0 = wall_clock64();
        unsigned int v;
        for (;;) {
            v = __hip_atomic_load(peer_slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            if (v != SENTINEL) break;
            if (wall_clock64() - t0 >= TIMEOUT_CYC) break;
            __builtin_amdgcn_s_sleep(1);
        }
        *out_latency = wall_clock64() - t0;
        *out_seen    = (v != SENTINEL) ? 1u : 0u;
    }
}

// ============================================================
// ROUND 2 — multi-word handoff safety on the variant-B (host-UC) path
//
// The real handoff is multi-word (an activation vector, not one flag word).
// Round 1 only proved a single word eventually flips != qNaN. Round 2 tests:
//   R2.1 WRITE-ORDERING   (§5.5 Rule 1c addendum): payload-first vs sentinel-first
//   R2.2 POSTED-WRITE     (§11.20 candidate #2): does a plain flat_store land
//                          on its own, or does it need a drain (fence+readback)?
//   R2.3 UNDER PCIE LOAD  (§11.4 class): does ordering/visibility survive a
//                          concurrent P2P-copy stream hammering the link?
//
// PAYLOAD_N words live in a host-UC buffer pre-init to qNaN. A SEPARATE host-UC
// word is the sentinel/trigger. The reader spins on the sentinel, then reads
// ALL payload words and checks they are the CORRECT current values (value+idx),
// NOT stale (qNaN) and NOT wrong.
// ============================================================

static const int PAYLOAD_N = 256;  // 256 floats = 1 KiB, 8 cache lines (128 B)

// --- R2.1 / R2.3: payload-first-then-sentinel (the SAFE ordering) ---
// Writes every payload word with a __threadfence() between the payload block
// and the sentinel, so the payload is drained before the trigger flips.
__global__ void r2_writer_payload_first(unsigned int* payload, unsigned int* sentinel,
                                        unsigned int base_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < PAYLOAD_N; ++i) {
            unsigned int v = base_value + (unsigned int)i;
            __hip_atomic_store(&payload[i], v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        // Drain the payload writes before publishing the trigger.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __hip_atomic_store(sentinel, base_value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// --- R2.1: sentinel-first-then-payload (the HAZARD ordering) ---
// Flips the trigger BEFORE writing the payload — the §5.5 Rule 1c addendum
// hazard. A reader that wakes on the sentinel may read stale payload.
//
// A deliberate ~20000-cycle spin gap is inserted AFTER the sentinel store and
// BEFORE the payload writes. This WIDENS the race window so the harness can
// PROVE it catches staleness when present (a negative control). Without the
// gap, the payload writes follow the sentinel so fast that a reader waking on
// the same PCIe round-trip can't slot in between — making stale=0 ambiguous
// ("no hazard" vs "harness too slow to see it"). With the gap, a reader that
// wakes on the sentinel during the gap and reads qNaN payload is a genuine
// observable ordering violation.
__global__ void r2_writer_sentinel_first(unsigned int* payload, unsigned int* sentinel,
                                         unsigned int base_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __hip_atomic_store(sentinel, base_value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        // Deliberate gap so a reader can wake on the sentinel mid-handoff.
        unsigned long long g0 = wall_clock64();
        while (wall_clock64() - g0 < 20000ULL) { __builtin_amdgcn_s_sleep(0); }
        for (int i = 0; i < PAYLOAD_N; ++i) {
            unsigned int v = base_value + (unsigned int)i;
            __hip_atomic_store(&payload[i], v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// --- R2.1 / R2.3 reader: spin on sentinel, then verify ALL payload words ---
// out_seen      : 1 if sentinel observed != qNaN within timeout
// out_stale     : count of payload words still == qNaN (never landed)
// out_wrong     : count of payload words != qNaN but != expected (torn/wrong)
// out_latency   : cycles to observe sentinel
__global__ void r2_reader_verify(const unsigned int* payload, const unsigned int* sentinel,
                                 unsigned int expect_base,
                                 unsigned long long* out_latency,
                                 unsigned int* out_seen,
                                 unsigned int* out_stale,
                                 unsigned int* out_wrong) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0 = wall_clock64();
        unsigned int s;
        for (;;) {
            s = __hip_atomic_load(sentinel, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            if (s != SENTINEL) break;
            if (wall_clock64() - t0 >= TIMEOUT_CYC) break;
            __builtin_amdgcn_s_sleep(1);
        }
        *out_latency = wall_clock64() - t0;
        unsigned int seen = (s != SENTINEL) ? 1u : 0u;
        *out_seen = seen;

        // Read ALL payload words immediately after observing the sentinel.
        unsigned int stale = 0, wrong = 0;
        for (int i = 0; i < PAYLOAD_N; ++i) {
            unsigned int v = __hip_atomic_load(&payload[i], __ATOMIC_ACQUIRE,
                                               __HIP_MEMORY_SCOPE_AGENT);
            unsigned int expect = expect_base + (unsigned int)i;
            if (v == SENTINEL)      stale++;
            else if (v != expect)   wrong++;
        }
        *out_stale = stale;
        *out_wrong = wrong;
    }
}

// --- R2.2: posted-write completion test ---
// Two writer flavors for a single sentinel word on host-UC.
//   no-drain : plain release store, nothing waits for it to land.
//   drain    : release store + __threadfence() + self-readback (forces it out).
__global__ void r2_writer_nodrain(unsigned int* sentinel, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __hip_atomic_store(sentinel, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        // NO fence, NO readback — a bare posted write.
    }
}
__global__ void r2_writer_drain(unsigned int* sentinel, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __hip_atomic_store(sentinel, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
        volatile unsigned int confirm =
            __hip_atomic_load(sentinel, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        (void)confirm;
    }
}

// ============================================================
// Helpers
// ============================================================

#define CHECK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

static unsigned long long median(std::vector<unsigned long long>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ============================================================
// VARIANT A runner
// ============================================================
static bool run_variant_A(int trials,
                           int& out_seen_count,
                           unsigned long long& out_median_lat) {
    // GPU0: allocate cached VRAM slot
    unsigned int* d0_slot = nullptr;
    CHECK(hipSetDevice(0));
    CHECK(hipMalloc(&d0_slot, sizeof(unsigned int)));

    // GPU1: allocate output buffers
    unsigned long long* d1_latency = nullptr;
    unsigned int* d1_seen = nullptr;
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&d1_latency, sizeof(unsigned long long)));
    CHECK(hipMalloc(&d1_seen, sizeof(unsigned int)));

    // Enable peer access GPU1 -> GPU0
    CHECK(hipSetDevice(1));
    int can_access = 0;
    hipDeviceCanAccessPeer(&can_access, 1, 0);
    if (!can_access) {
        printf("  SKIP: GPU1 cannot peer-access GPU0\n");
        hipFree(d0_slot);
        hipSetDevice(1); hipFree(d1_latency); hipFree(d1_seen);
        out_seen_count = -1;
        out_median_lat = 0;
        return true;
    }
    hipError_t peer_err = hipDeviceEnablePeerAccess(0, 0);
    if (peer_err != hipSuccess && peer_err != hipErrorPeerAccessAlreadyEnabled) {
        printf("  WARN: peer access enable returned %s — continuing\n", hipGetErrorString(peer_err));
    }

    std::vector<unsigned long long> lats;
    lats.reserve(trials);
    int seen = 0;

    // Use separate streams for GPU0/GPU1 to serialize writer→reader
    hipStream_t s0, s1;
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&s0));
    CHECK(hipSetDevice(1)); CHECK(hipStreamCreate(&s1));

    for (int t = 0; t < trials; t++) {
        // Init slot to SENTINEL on GPU0
        CHECK(hipSetDevice(0));
        CHECK(hipMemcpy(d0_slot, &SENTINEL, sizeof(unsigned int), hipMemcpyHostToDevice));

        // Reset reader outputs
        CHECK(hipSetDevice(1));
        const unsigned long long zero_lat = 0;
        const unsigned int zero_seen = 0;
        CHECK(hipMemcpy(d1_latency, &zero_lat, sizeof(zero_lat), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_seen, &zero_seen, sizeof(zero_seen), hipMemcpyHostToDevice));

        // Launch GPU1 reader FIRST (starts spinning), then GPU0 writer.
        // 10ms host sleep ensures the reader kernel is dispatched and in its
        // spin loop before the writer fires — critical for A's el1f repro.
        CHECK(hipSetDevice(1));
        hipLaunchKernelGGL(varA_reader, dim3(1), dim3(1), 0, s1,
                           (const unsigned int*)d0_slot, d1_latency, d1_seen);
        usleep(10000); // 10ms: reader is spinning before writer fires

        CHECK(hipSetDevice(0));
        unsigned int value = t + 1u; // non-sentinel
        hipLaunchKernelGGL(varA_writer, dim3(1), dim3(1), 0, s0, d0_slot, value);
        CHECK(hipStreamSynchronize(s0));

        // Wait for reader to finish (timeout or observe)
        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));

        unsigned long long lat;
        unsigned int sv;
        CHECK(hipMemcpy(&lat, d1_latency, sizeof(lat), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&sv, d1_seen, sizeof(sv), hipMemcpyDeviceToHost));

        if (sv) { seen++; lats.push_back(lat); }
        else     { lats.push_back(TIMEOUT_CYC); }
    }

    CHECK(hipSetDevice(0)); hipStreamDestroy(s0); hipFree(d0_slot);
    CHECK(hipSetDevice(1)); hipStreamDestroy(s1); hipFree(d1_latency); hipFree(d1_seen);

    out_seen_count = seen;
    out_median_lat = median(lats);
    return true;
}

// ============================================================
// VARIANT B runner
// ============================================================
static bool run_variant_B(int trials,
                           int& out_seen_count,
                           unsigned long long& out_median_lat) {
    // Allocate host-mapped UC coherent buffer
    unsigned int* h_slot = nullptr;
    CHECK(hipHostMalloc((void**)&h_slot, sizeof(unsigned int),
                        hipHostMallocPortable | hipHostMallocCoherent));

    // Get device pointers for each GPU
    unsigned int* d0_slot = nullptr;
    unsigned int* d1_slot = nullptr;
    CHECK(hipSetDevice(0));
    CHECK(hipHostGetDevicePointer((void**)&d0_slot, h_slot, 0));
    CHECK(hipSetDevice(1));
    CHECK(hipHostGetDevicePointer((void**)&d1_slot, h_slot, 0));

    // GPU1 output buffers
    unsigned long long* d1_latency = nullptr;
    unsigned int* d1_seen = nullptr;
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&d1_latency, sizeof(unsigned long long)));
    CHECK(hipMalloc(&d1_seen, sizeof(unsigned int)));

    std::vector<unsigned long long> lats;
    lats.reserve(trials);
    int seen = 0;

    hipStream_t s0, s1;
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&s0));
    CHECK(hipSetDevice(1)); CHECK(hipStreamCreate(&s1));

    for (int t = 0; t < trials; t++) {
        // Init slot to SENTINEL on host side
        *h_slot = SENTINEL;
        // Ensure host write is visible before GPU kernels launch
        __sync_synchronize();

        // Reset reader outputs
        CHECK(hipSetDevice(1));
        const unsigned long long zero_lat = 0;
        const unsigned int zero_seen = 0;
        CHECK(hipMemcpy(d1_latency, &zero_lat, sizeof(zero_lat), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_seen, &zero_seen, sizeof(zero_seen), hipMemcpyHostToDevice));

        // Launch reader on GPU1 first, then writer on GPU0
        CHECK(hipSetDevice(1));
        hipLaunchKernelGGL(varB_reader, dim3(1), dim3(1), 0, s1,
                           (const unsigned int*)d1_slot, d1_latency, d1_seen);
        usleep(10000); // 10ms: reader is spinning before writer fires

        CHECK(hipSetDevice(0));
        unsigned int value = t + 1u;
        hipLaunchKernelGGL(varB_writer, dim3(1), dim3(1), 0, s0, d0_slot, value);
        CHECK(hipStreamSynchronize(s0));

        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));

        unsigned long long lat;
        unsigned int sv;
        CHECK(hipMemcpy(&lat, d1_latency, sizeof(lat), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&sv, d1_seen, sizeof(sv), hipMemcpyDeviceToHost));

        if (sv) { seen++; lats.push_back(lat); }
        else     { lats.push_back(TIMEOUT_CYC); }
    }

    CHECK(hipSetDevice(0)); hipStreamDestroy(s0);
    CHECK(hipSetDevice(1)); hipStreamDestroy(s1);
    hipFree(d1_latency); hipFree(d1_seen);
    hipHostFree(h_slot);

    out_seen_count = seen;
    out_median_lat = median(lats);
    return true;
}

// ============================================================
// VARIANT C runner
// ============================================================
static bool run_variant_C(int trials,
                           int& out_seen_count,
                           unsigned long long& out_median_lat,
                           unsigned long long& out_median_drain) {
    // GPU0: cached VRAM slot + drain cost output
    unsigned int* d0_slot = nullptr;
    unsigned long long* d0_drain = nullptr;
    CHECK(hipSetDevice(0));
    CHECK(hipMalloc(&d0_slot, sizeof(unsigned int)));
    CHECK(hipMalloc(&d0_drain, sizeof(unsigned long long)));

    // GPU1 output buffers
    unsigned long long* d1_latency = nullptr;
    unsigned int* d1_seen = nullptr;
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&d1_latency, sizeof(unsigned long long)));
    CHECK(hipMalloc(&d1_seen, sizeof(unsigned int)));

    // Enable peer access GPU1 -> GPU0
    CHECK(hipSetDevice(1));
    int can_access = 0;
    hipDeviceCanAccessPeer(&can_access, 1, 0);
    if (!can_access) {
        printf("  SKIP: GPU1 cannot peer-access GPU0\n");
        out_seen_count = -1; out_median_lat = 0; out_median_drain = 0;
        hipSetDevice(0); hipFree(d0_slot); hipFree(d0_drain);
        hipSetDevice(1); hipFree(d1_latency); hipFree(d1_seen);
        return true;
    }
    hipError_t peer_err = hipDeviceEnablePeerAccess(0, 0);
    if (peer_err != hipSuccess && peer_err != hipErrorPeerAccessAlreadyEnabled) {
        printf("  WARN: peer access enable returned %s — continuing\n", hipGetErrorString(peer_err));
    }

    std::vector<unsigned long long> lats, drains;
    lats.reserve(trials); drains.reserve(trials);
    int seen = 0;

    hipStream_t s0, s1;
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&s0));
    CHECK(hipSetDevice(1)); CHECK(hipStreamCreate(&s1));

    for (int t = 0; t < trials; t++) {
        // Init slot to SENTINEL on GPU0
        CHECK(hipSetDevice(0));
        CHECK(hipMemcpy(d0_slot, &SENTINEL, sizeof(unsigned int), hipMemcpyHostToDevice));

        // Reset outputs
        CHECK(hipSetDevice(1));
        const unsigned long long zero_lat = 0;
        const unsigned int zero_seen = 0;
        CHECK(hipMemcpy(d1_latency, &zero_lat, sizeof(zero_lat), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_seen, &zero_seen, sizeof(zero_seen), hipMemcpyHostToDevice));

        // Launch reader first, then writer (with drain)
        CHECK(hipSetDevice(1));
        hipLaunchKernelGGL(varC_reader, dim3(1), dim3(1), 0, s1,
                           (const unsigned int*)d0_slot, d1_latency, d1_seen);
        usleep(10000); // 10ms: reader spinning before writer fires

        CHECK(hipSetDevice(0));
        unsigned int value = t + 1u;
        hipLaunchKernelGGL(varC_writer, dim3(1), dim3(1), 0, s0,
                           d0_slot, value, d0_drain);
        CHECK(hipStreamSynchronize(s0));

        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));

        unsigned long long lat, drain;
        unsigned int sv;
        CHECK(hipMemcpy(&lat, d1_latency, sizeof(lat), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&sv, d1_seen, sizeof(sv), hipMemcpyDeviceToHost));
        CHECK(hipSetDevice(0));
        CHECK(hipMemcpy(&drain, d0_drain, sizeof(drain), hipMemcpyDeviceToHost));

        if (sv) { seen++; lats.push_back(lat); }
        else     { lats.push_back(TIMEOUT_CYC); }
        drains.push_back(drain);
    }

    CHECK(hipSetDevice(0)); hipStreamDestroy(s0); hipFree(d0_slot); hipFree(d0_drain);
    CHECK(hipSetDevice(1)); hipStreamDestroy(s1); hipFree(d1_latency); hipFree(d1_seen);

    out_seen_count = seen;
    out_median_lat = median(lats);
    out_median_drain = median(drains);
    return true;
}

// ============================================================
// ROUND 2 results struct + runners
// ============================================================
struct R2Result {
    int   seen;          // # trials sentinel observed within timeout
    int   trials_stale;  // # trials with >=1 stale payload word
    int   trials_wrong;  // # trials with >=1 wrong (non-stale, non-expected) word
    int   total_stale_words;
    int   total_wrong_words;
    unsigned long long median_lat;
};

// R2.1 / R2.3: multi-word handoff verification on the host-UC path.
// writer_payload_first=true -> SAFE ordering; false -> HAZARD (sentinel-first).
// pressure=true -> launch a continuous background hipMemcpyPeerAsync P2P stream
//   that hammers the PCIe link (GPU0<->GPU1) concurrently with the handoff.
static bool run_round2_multiword(int trials, bool writer_payload_first, bool pressure,
                                 R2Result& out) {
    out = R2Result{0,0,0,0,0,0};

    // Host-UC payload + sentinel (variant-B allocation: Rule 1b).
    unsigned int* h_payload = nullptr;
    unsigned int* h_sentinel = nullptr;
    CHECK(hipHostMalloc((void**)&h_payload, PAYLOAD_N * sizeof(unsigned int),
                        hipHostMallocPortable | hipHostMallocCoherent));
    CHECK(hipHostMalloc((void**)&h_sentinel, sizeof(unsigned int),
                        hipHostMallocPortable | hipHostMallocCoherent));

    unsigned int *d0_payload=nullptr, *d0_sentinel=nullptr;  // GPU0 writer view
    unsigned int *d1_payload=nullptr, *d1_sentinel=nullptr;  // GPU1 reader view
    CHECK(hipSetDevice(0));
    CHECK(hipHostGetDevicePointer((void**)&d0_payload, h_payload, 0));
    CHECK(hipHostGetDevicePointer((void**)&d0_sentinel, h_sentinel, 0));
    CHECK(hipSetDevice(1));
    CHECK(hipHostGetDevicePointer((void**)&d1_payload, h_payload, 0));
    CHECK(hipHostGetDevicePointer((void**)&d1_sentinel, h_sentinel, 0));

    // GPU1 reader outputs.
    unsigned long long* d1_lat=nullptr; unsigned int *d1_seen=nullptr,*d1_stale=nullptr,*d1_wrong=nullptr;
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&d1_lat, sizeof(unsigned long long)));
    CHECK(hipMalloc(&d1_seen, sizeof(unsigned int)));
    CHECK(hipMalloc(&d1_stale, sizeof(unsigned int)));
    CHECK(hipMalloc(&d1_wrong, sizeof(unsigned int)));

    // Background PCIe-pressure buffers (a big P2P copy GPU0<->GPU1 in a loop).
    const size_t LOAD_BYTES = 16 * 1024 * 1024; // 16 MiB > ROC_P2P_SDMA_SIZE (1 MiB)
    unsigned char *d0_load=nullptr, *d1_load=nullptr;
    hipStream_t load_stream = nullptr;
    if (pressure) {
        CHECK(hipSetDevice(0)); CHECK(hipMalloc(&d0_load, LOAD_BYTES));
        CHECK(hipSetDevice(1)); CHECK(hipMalloc(&d1_load, LOAD_BYTES));
        CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&load_stream));
        // Ensure peer access for the P2P copy.
        int ca=0; hipDeviceCanAccessPeer(&ca,0,1);
        if (ca) { hipError_t pe=hipDeviceEnablePeerAccess(1,0);
                  if (pe!=hipSuccess && pe!=hipErrorPeerAccessAlreadyEnabled)
                      printf("  WARN: load peer-enable %s\n", hipGetErrorString(pe)); }
    }

    hipStream_t s0, s1;
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&s0));
    CHECK(hipSetDevice(1)); CHECK(hipStreamCreate(&s1));

    std::vector<unsigned long long> lats; lats.reserve(trials);

    void* writer_kern = writer_payload_first ? (void*)r2_writer_payload_first
                                             : (void*)r2_writer_sentinel_first;

    for (int t = 0; t < trials; ++t) {
        unsigned int base = (t + 1u) * 1000u; // non-qNaN base for this trial
        // Init payload + sentinel to qNaN (host side, then fence).
        for (int i = 0; i < PAYLOAD_N; ++i) h_payload[i] = SENTINEL;
        *h_sentinel = SENTINEL;
        __sync_synchronize();

        // Reset reader outputs.
        CHECK(hipSetDevice(1));
        const unsigned long long z64 = 0; const unsigned int z = 0;
        CHECK(hipMemcpy(d1_lat,&z64,sizeof(z64),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_seen,&z,sizeof(z),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_stale,&z,sizeof(z),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_wrong,&z,sizeof(z),hipMemcpyHostToDevice));

        // Kick off background PCIe pressure (fire several async P2P copies).
        if (pressure) {
            CHECK(hipSetDevice(0));
            for (int k = 0; k < 8; ++k)
                CHECK(hipMemcpyPeerAsync(d1_load,1,d0_load,0,LOAD_BYTES,load_stream));
        }

        // Launch reader (spins), then writer after a SHORT 200us settle — just
        // enough for the reader kernel to dispatch and enter its spin loop, but
        // far shorter than the writer's 20000-cyc sentinel->payload gap. So the
        // reader is ALREADY SPINNING when the writer flips the sentinel, and it
        // observes the sentinel + reads the payload DURING the writer's gap —
        // catching an ordering violation if one exists. (The old 10ms settle let
        // the writer fully complete first, masking the race — med_lat ~1e6 cyc.)
        CHECK(hipSetDevice(1));
        hipLaunchKernelGGL(r2_reader_verify, dim3(1),dim3(1),0,s1,
                           (const unsigned int*)d1_payload,(const unsigned int*)d1_sentinel,
                           base, d1_lat, d1_seen, d1_stale, d1_wrong);
        usleep(200);

        CHECK(hipSetDevice(0));
        void* wargs[] = { &d0_payload, &d0_sentinel, &base };
        CHECK(hipLaunchKernel(writer_kern, dim3(1),dim3(1), wargs, 0, s0));
        CHECK(hipStreamSynchronize(s0));

        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));

        unsigned long long lat; unsigned int sv,stale,wrong;
        CHECK(hipMemcpy(&lat,d1_lat,sizeof(lat),hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&sv,d1_seen,sizeof(sv),hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&stale,d1_stale,sizeof(stale),hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&wrong,d1_wrong,sizeof(wrong),hipMemcpyDeviceToHost));

        if (sv) { out.seen++; lats.push_back(lat); } else lats.push_back(TIMEOUT_CYC);
        if (stale) out.trials_stale++;
        if (wrong) out.trials_wrong++;
        out.total_stale_words += (int)stale;
        out.total_wrong_words += (int)wrong;
    }

    out.median_lat = median(lats);

    CHECK(hipSetDevice(0)); hipStreamDestroy(s0);
    CHECK(hipSetDevice(1)); hipStreamDestroy(s1);
    if (pressure) {
        CHECK(hipSetDevice(0)); hipStreamSynchronize(load_stream); hipStreamDestroy(load_stream);
        hipFree(d0_load); CHECK(hipSetDevice(1)); hipFree(d1_load);
    }
    CHECK(hipSetDevice(1));
    hipFree(d1_lat); hipFree(d1_seen); hipFree(d1_stale); hipFree(d1_wrong);
    hipHostFree(h_payload); hipHostFree(h_sentinel);
    return true;
}

// R2.2: posted-write completion. Does a bare flat_store to host-UC land within
// a bounded time WITHOUT a producer drain? Compare no-drain vs drain.
static bool run_round2_posted(int trials, bool drain, int& out_seen,
                              unsigned long long& out_median_lat) {
    out_seen = 0;
    unsigned int* h_sentinel = nullptr;
    CHECK(hipHostMalloc((void**)&h_sentinel, sizeof(unsigned int),
                        hipHostMallocPortable | hipHostMallocCoherent));
    unsigned int *d0_s=nullptr,*d1_s=nullptr;
    CHECK(hipSetDevice(0)); CHECK(hipHostGetDevicePointer((void**)&d0_s,h_sentinel,0));
    CHECK(hipSetDevice(1)); CHECK(hipHostGetDevicePointer((void**)&d1_s,h_sentinel,0));

    unsigned long long* d1_lat=nullptr; unsigned int* d1_seen=nullptr;
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&d1_lat,sizeof(unsigned long long)));
    CHECK(hipMalloc(&d1_seen,sizeof(unsigned int)));

    hipStream_t s0,s1;
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreate(&s0));
    CHECK(hipSetDevice(1)); CHECK(hipStreamCreate(&s1));

    std::vector<unsigned long long> lats; lats.reserve(trials);
    void* wk = drain ? (void*)r2_writer_drain : (void*)r2_writer_nodrain;

    for (int t=0;t<trials;++t) {
        *h_sentinel = SENTINEL; __sync_synchronize();
        CHECK(hipSetDevice(1));
        const unsigned long long z64=0; const unsigned int z=0;
        CHECK(hipMemcpy(d1_lat,&z64,sizeof(z64),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1_seen,&z,sizeof(z),hipMemcpyHostToDevice));

        // Reader spins (reuse varB_reader semantics: spin until != qNaN).
        // 200us settle so the reader is spinning before the writer fires; the
        // reader's own TIMEOUT_CYC bound still catches a posted write that never
        // lands, so this delay does not mask a non-landing store.
        CHECK(hipSetDevice(1));
        hipLaunchKernelGGL(varB_reader, dim3(1),dim3(1),0,s1,
                           (const unsigned int*)d1_s, d1_lat, d1_seen);
        usleep(200);

        CHECK(hipSetDevice(0));
        unsigned int value = t + 1u;
        void* wargs[] = { &d0_s, &value };
        CHECK(hipLaunchKernel(wk, dim3(1),dim3(1), wargs, 0, s0));
        CHECK(hipStreamSynchronize(s0));

        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));

        unsigned long long lat; unsigned int sv;
        CHECK(hipMemcpy(&lat,d1_lat,sizeof(lat),hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&sv,d1_seen,sizeof(sv),hipMemcpyDeviceToHost));
        if (sv) { out_seen++; lats.push_back(lat); } else lats.push_back(TIMEOUT_CYC);
    }
    out_median_lat = median(lats);

    CHECK(hipSetDevice(0)); hipStreamDestroy(s0);
    CHECK(hipSetDevice(1)); hipStreamDestroy(s1);
    hipFree(d1_lat); hipFree(d1_seen); hipHostFree(h_sentinel);
    return true;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    int trials = (argc > 1) ? atoi(argv[1]) : 100;

    // Check we have at least 2 GPUs
    int ndev = 0;
    hipGetDeviceCount(&ndev);
    if (ndev < 2) {
        printf("FAIL: need at least 2 GPUs, found %d\n", ndev);
        return 1;
    }

    hipDeviceProp_t p0, p1;
    hipSetDevice(0); hipGetDeviceProperties(&p0, 0);
    hipSetDevice(1); hipGetDeviceProperties(&p1, 1);
    printf("GPU0: %s  GPU1: %s\n", p0.name, p1.name);
    printf("trials=%d  timeout=%.0e cyc\n", trials, (double)TIMEOUT_CYC);
    printf("\n");

    // ---- Variant A ----
    printf("=== Variant A: GPU0 cached VRAM, GPU1 peer spin-read ===\n");
    printf("    EXPECTED: MISS (GPU0 L2 not flushed to GPU1 peer)\n");
    int a_seen = 0; unsigned long long a_lat = 0;
    if (!run_variant_A(trials, a_seen, a_lat)) return 1;
    if (a_seen >= 0) {
        printf("  see-rate: %d/%d (%.0f%%)\n", a_seen, trials, 100.0*a_seen/trials);
        printf("  median latency: %llu cyc\n", a_lat);
    }
    printf("\n");

    // ---- Variant B ----
    printf("=== Variant B: host-mapped UC, both GPUs mapped ===\n");
    printf("    EXPECTED: SEE (UC bypasses GPU L2)\n");
    int b_seen = 0; unsigned long long b_lat = 0;
    if (!run_variant_B(trials, b_seen, b_lat)) return 1;
    if (b_seen >= 0) {
        printf("  see-rate: %d/%d (%.0f%%)\n", b_seen, trials, 100.0*b_seen/trials);
        printf("  median latency: %llu cyc\n", b_lat);
    }
    printf("\n");

    // ---- Variant C ----
    printf("=== Variant C: GPU0 cached VRAM + producer drain ===\n");
    printf("    EXPECTED: SEE after drain; report drain cost\n");
    int c_seen = 0; unsigned long long c_lat = 0, c_drain = 0;
    if (!run_variant_C(trials, c_seen, c_lat, c_drain)) return 1;
    if (c_seen >= 0) {
        printf("  see-rate: %d/%d (%.0f%%)\n", c_seen, trials, 100.0*c_seen/trials);
        printf("  median latency: %llu cyc\n", c_lat);
        printf("  median drain cost: %llu cyc\n", c_drain);
    }
    printf("\n");

    // ---- Summary table ----
    printf("=== SUMMARY ===\n");
    printf("  variant | see-rate        | median latency (cyc) | drain cost (cyc)\n");
    printf("  --------|-----------------|----------------------|------------------\n");
    if (a_seen >= 0)
        printf("  A (cach)| %3d/%-3d (%3.0f%%) | %20llu |        N/A\n",
               a_seen, trials, 100.0*a_seen/trials, a_lat);
    if (b_seen >= 0)
        printf("  B (UC)  | %3d/%-3d (%3.0f%%) | %20llu |        N/A\n",
               b_seen, trials, 100.0*b_seen/trials, b_lat);
    if (c_seen >= 0)
        printf("  C (drn) | %3d/%-3d (%3.0f%%) | %20llu | %16llu\n",
               c_seen, trials, 100.0*c_seen/trials, c_lat, c_drain);
    printf("\n");

    // ---- Interpretation ----
    bool a_miss  = (a_seen < trials/2);  // <50% = harness valid, el1f confirmed
    bool b_see   = (b_seen > trials*9/10); // >90% = production viable
    bool c_works = (c_seen > trials*9/10); // >90% = drain approach works

    printf("=== INTERPRETATION ===\n");
    printf("  A harness sanity (el1f confirmed): %s\n", a_miss  ? "PASS (miss as expected)" : "FAIL (unexpected see)");
    printf("  B production viable (UC sees):     %s\n", b_see   ? "PASS" : "FAIL");
    printf("  C drain approach viable:            %s", c_works ? "PASS" : "FAIL");
    if (c_works && c_drain > 0)
        printf(" (drain cost: %llu cyc = ~%.1f us at 1GHz)", c_drain, c_drain / 1000.0);
    printf("\n\n");

    // ========================================================
    // ROUND 2 — multi-word handoff safety on the variant-B (host-UC) path
    // ========================================================
    printf("################ ROUND 2 (variant-B / host-UC) ################\n");
    printf("PAYLOAD_N=%d words (%d B, %d cache lines)\n",
           PAYLOAD_N, PAYLOAD_N*4, PAYLOAD_N*4/128);
    printf("\n");

    // ---- R2.1 WRITE-ORDERING (idle) ----
    printf("=== R2.1 write-ordering (idle) ===\n");
    R2Result pf_idle, sf_idle;
    if (!run_round2_multiword(trials, /*payload_first=*/true,  /*pressure=*/false, pf_idle)) return 1;
    if (!run_round2_multiword(trials, /*payload_first=*/false, /*pressure=*/false, sf_idle)) return 1;
    printf("  payload-first (SAFE)   : seen=%d/%d  stale-trials=%d  wrong-trials=%d  (stale_words=%d wrong_words=%d)  med_lat=%llu cyc\n",
           pf_idle.seen, trials, pf_idle.trials_stale, pf_idle.trials_wrong,
           pf_idle.total_stale_words, pf_idle.total_wrong_words, pf_idle.median_lat);
    printf("  sentinel-first (HAZARD): seen=%d/%d  stale-trials=%d  wrong-trials=%d  (stale_words=%d wrong_words=%d)  med_lat=%llu cyc\n",
           sf_idle.seen, trials, sf_idle.trials_stale, sf_idle.trials_wrong,
           sf_idle.total_stale_words, sf_idle.total_wrong_words, sf_idle.median_lat);
    printf("\n");

    // ---- R2.2 POSTED-WRITE COMPLETION (idle) ----
    printf("=== R2.2 posted-write completion (idle) ===\n");
    int nd_seen=0, dr_seen=0; unsigned long long nd_lat=0, dr_lat=0;
    if (!run_round2_posted(trials, /*drain=*/false, nd_seen, nd_lat)) return 1;
    if (!run_round2_posted(trials, /*drain=*/true,  dr_seen, dr_lat)) return 1;
    printf("  no-drain (bare posted store): bounded-observe=%d/%d  med_lat=%llu cyc\n",
           nd_seen, trials, nd_lat);
    printf("  with-drain (fence+readback) : bounded-observe=%d/%d  med_lat=%llu cyc\n",
           dr_seen, trials, dr_lat);
    printf("\n");

    // ---- R2.3 UNDER PCIE PRESSURE ----
    printf("=== R2.3 under PCIe pressure (concurrent P2P-copy stream) ===\n");
    R2Result pf_load, sf_load;
    if (!run_round2_multiword(trials, /*payload_first=*/true,  /*pressure=*/true, pf_load)) return 1;
    if (!run_round2_multiword(trials, /*payload_first=*/false, /*pressure=*/true, sf_load)) return 1;
    printf("  payload-first (SAFE)   : seen=%d/%d  stale-trials=%d  wrong-trials=%d  (stale_words=%d wrong_words=%d)  med_lat=%llu cyc\n",
           pf_load.seen, trials, pf_load.trials_stale, pf_load.trials_wrong,
           pf_load.total_stale_words, pf_load.total_wrong_words, pf_load.median_lat);
    printf("  sentinel-first (HAZARD): seen=%d/%d  stale-trials=%d  wrong-trials=%d  (stale_words=%d wrong_words=%d)  med_lat=%llu cyc\n",
           sf_load.seen, trials, sf_load.trials_stale, sf_load.trials_wrong,
           sf_load.total_stale_words, sf_load.total_wrong_words, sf_load.median_lat);
    printf("\n");

    // ---- Round 2 summary table ----
    printf("=== ROUND 2 SUMMARY [test | idle | under-pressure] ===\n");
    printf("  test                          | idle                         | under-pressure\n");
    printf("  ------------------------------|------------------------------|------------------------------\n");
    printf("  R2.1 payload-first stale-rate | %3d/%-3d trials               | %3d/%-3d trials\n",
           pf_idle.trials_stale, trials, pf_load.trials_stale, trials);
    printf("  R2.1 sentinel-first stale-rate| %3d/%-3d trials               | %3d/%-3d trials\n",
           sf_idle.trials_stale, trials, sf_load.trials_stale, trials);
    printf("  R2.1 payload-first see-rate   | %3d/%-3d                      | %3d/%-3d\n",
           pf_idle.seen, trials, pf_load.seen, trials);
    printf("  R2.2 posted no-drain observe  | %3d/%-3d                      | (idle only)\n",
           nd_seen, trials);
    printf("  R2.2 posted with-drain observe| %3d/%-3d                      | (idle only)\n",
           dr_seen, trials);
    printf("\n");

    // ---- Round 2 interpretation ----
    bool ordering_safe   = (pf_idle.trials_stale == 0 && pf_idle.trials_wrong == 0);
    bool ordering_hazard = (sf_idle.trials_stale > 0 || sf_idle.trials_wrong > 0);
    bool posted_lands    = (nd_seen > trials*9/10);
    bool pressure_safe   = (pf_load.seen > trials*9/10 &&
                            pf_load.trials_stale == 0 && pf_load.trials_wrong == 0);

    printf("=== ROUND 2 INTERPRETATION ===\n");
    printf("  Ordering: payload-first is %s (stale=%d wrong=%d) — must precede sentinel.\n",
           ordering_safe ? "SAFE" : "UNSAFE", pf_idle.total_stale_words, pf_idle.total_wrong_words);
    printf("  Ordering: sentinel-first %s the §5.5 Rule-1c hazard (stale=%d wrong=%d words over %d trials).\n",
           ordering_hazard ? "REPRODUCES" : "did NOT reproduce", sf_idle.total_stale_words, sf_idle.total_wrong_words, trials);
    printf("  Posted-write: bare store %s on its own within timeout (%d/%d) — drain %s.\n",
           posted_lands ? "LANDS" : "does NOT reliably land", nd_seen, trials,
           posted_lands ? "NOT required for visibility" : "REQUIRED");
    printf("  Under pressure: payload-first handoff is %s (seen=%d/%d stale=%d wrong=%d).\n",
           pressure_safe ? "SAFE" : "EXPOSED", pf_load.seen, trials,
           pf_load.total_stale_words, pf_load.total_wrong_words);
    printf("  => B PRODUCTION-SAFE for multi-word sentinel handoff: %s\n",
           (ordering_safe && posted_lands && pressure_safe)
             ? "YES, IF writer writes payload-before-sentinel with a fence between"
             : "NO / CONDITIONAL — see which check failed above");
    printf("\n");

    return 0;
}
