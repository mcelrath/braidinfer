// Failure-envelope test for hardware global_atomic_add_f32 on gfx1100.
//
// Hardware atomic-add (`unsafeAtomicAdd`, `__builtin_amdgcn_global_atomic_fadd_f32`)
// is dramatically faster than the default CAS-loop atomicAdd, but it is NOT
// safe on every memory type. This program enumerates the memory types we use
// in braidinfer and tests each combination, comparing GPU-atomic results
// against:
//   (a) host-CPU readback
//   (b) peer-GPU readback (single-pass kernel that reads then writes a sentinel)
// and against a known reference sum.
//
// Memory types tested:
//   - hipMalloc                                   (coarse-grained device mem)
//   - hipExtMallocWithFlags(hipDeviceMallocUncached) (MTYPE=UC, what
//                                                  braidinfer's
//                                                  alloc_uncached uses)
//   - hipExtMallocWithFlags(hipDeviceMallocFinegrained) (MTYPE=CC fine-grained)
//   - hipHostMalloc(hipHostMallocMapped)          (host-mapped, GPU-visible)
//   - hipHostMalloc(hipHostMallocCoherent | Mapped)
//   - hipHostMalloc(hipHostMallocNonCoherent | Mapped)
//   - peer-mapped buffer (allocated on GPU 1, accessed from GPU 0)
//
// Operations tested per memory type:
//   - unsafeAtomicAdd (HW global_atomic_add_f32)
//   - atomicAdd       (CAS-loop reference)
//   - hand-rolled CAS via atomicCAS (control)
//
// Pathology probes:
//   - Inject NaN mid-update — does HW atomic produce the same final value
//     as CAS would?
//   - Inject denormal mid-update — same question.
//
// Each test is repeated multiple times to detect non-determinism.
// Tabulated output: which memory types give CORRECT, WRONG, or NON-DETERMINISTIC
// results for the HW atomic.

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_unsafe_atomics.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#define CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while(0)

// ============================================================================
// Test kernels
// ============================================================================

// Each thread issues N adds of 1.0f to slots[tid % n_slots]. Total adds =
// blocks * threads * N. Expected per slot = (blocks * threads * N) / n_slots
// (assuming threads divides slots evenly).
__global__ void k_atomic_hw(float* slots, int n_slots, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s   = tid % n_slots;
    for (int i = 0; i < n_iter; i++) {
        unsafeAtomicAdd(&slots[s], 1.0f);
    }
}

__global__ void k_atomic_cas(float* slots, int n_slots, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s   = tid % n_slots;
    for (int i = 0; i < n_iter; i++) {
        atomicAdd(&slots[s], 1.0f);
    }
}

// Hand-rolled CAS — explicit reference for "what the safe path does".
__global__ void k_atomic_handcas(float* slots, int n_slots, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s   = tid % n_slots;
    uint32_t* p = (uint32_t*)&slots[s];
    for (int i = 0; i < n_iter; i++) {
        uint32_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
        while (true) {
            union { float f; uint32_t u; } cur; cur.u = old;
            float new_f = cur.f + 1.0f;
            union { float f; uint32_t u; } nf; nf.f = new_f;
            uint32_t prev = atomicCAS(p, old, nf.u);
            if (prev == old) break;
            old = prev;
        }
    }
}

// Peer-GPU readback kernel: one thread on GPU B reads slots[i] and writes to
// out[i]. If the source is uncached or peer-mapped, the read goes over PCIe.
__global__ void k_peer_readback(const float* src, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = src[tid];
}

// NaN-pathology probe: half of threads atomic-add 1.0f, the other half add
// NaN. CAS would absorb the NaN deterministically (NaN + x = NaN, comparisons
// fail forever); HW atomic add depends on what the FPU does with NaN
// arithmetic. Run on a single slot and check whether the final value is NaN
// (or some other deterministic answer).
__global__ void k_nan_inject(float* slots, int n_threads_with_nan,
                             int n_threads_total, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads_total) return;
    bool inject_nan = (tid < n_threads_with_nan);
    float val = inject_nan ? __builtin_nanf("") : 1.0f;
    for (int i = 0; i < n_iter; i++) {
        unsafeAtomicAdd(&slots[0], val);
    }
}

__global__ void k_nan_inject_cas(float* slots, int n_threads_with_nan,
                                 int n_threads_total, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads_total) return;
    bool inject_nan = (tid < n_threads_with_nan);
    float val = inject_nan ? __builtin_nanf("") : 1.0f;
    for (int i = 0; i < n_iter; i++) {
        atomicAdd(&slots[0], val);
    }
}

// Denormal-pathology probe: each thread adds a tiny denormal (~1e-40f) repeatedly.
// FTZ/DAZ behavior of HW atomic vs CAS may differ.
__global__ void k_denormal_probe(float* slots, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 1e-40f is below normal range (smallest normal ~1.18e-38)
    float denorm = 1.0e-40f;
    for (int i = 0; i < n_iter; i++) {
        unsafeAtomicAdd(&slots[0], denorm);
    }
}

__global__ void k_denormal_probe_cas(float* slots, int n_iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float denorm = 1.0e-40f;
    for (int i = 0; i < n_iter; i++) {
        atomicAdd(&slots[0], denorm);
    }
}

// ============================================================================
// Memory-type allocators
// ============================================================================

enum class MemType {
    DEV_COARSE,        // hipMalloc — coarse-grained device memory (default)
    DEV_UNCACHED,      // hipExtMallocWithFlags(hipDeviceMallocUncached)
    DEV_FINEGRAINED,   // hipExtMallocWithFlags(hipDeviceMallocFinegrained)
    HOST_MAPPED,       // hipHostMalloc(hipHostMallocMapped)
    HOST_COHERENT,     // hipHostMalloc(hipHostMallocCoherent | Mapped)
    HOST_NONCOHERENT,  // hipHostMalloc(hipHostMallocNonCoherent | Mapped)
    PEER_COARSE,       // alloced on GPU 1 (peer of GPU 0), used from GPU 0
    PEER_UNCACHED,     // uncached on GPU 1, used from GPU 0
};

const char* memtype_name(MemType m) {
    switch (m) {
        case MemType::DEV_COARSE:       return "dev_coarse (hipMalloc)";
        case MemType::DEV_UNCACHED:     return "dev_uncached (DeviceMallocUncached)";
        case MemType::DEV_FINEGRAINED:  return "dev_finegrained (DeviceMallocFinegrained)";
        case MemType::HOST_MAPPED:      return "host_mapped (HostMallocMapped)";
        case MemType::HOST_COHERENT:    return "host_coherent (HostMalloc Coherent|Mapped)";
        case MemType::HOST_NONCOHERENT: return "host_noncoherent (HostMalloc NonCoherent|Mapped)";
        case MemType::PEER_COARSE:      return "peer_coarse (hipMalloc on GPU1)";
        case MemType::PEER_UNCACHED:    return "peer_uncached (Uncached on GPU1)";
    }
    return "?";
}

struct Buffer {
    float* device_ptr;     // pointer the kernel sees
    float* host_ptr;       // pointer for CPU readback (if mapped) or scratch
    void*  raw_ptr;        // for free()
    size_t bytes;
    MemType mt;
    int     home_gpu;      // device that owns the allocation
    bool    owns_host;     // whether host_ptr was separately allocated
};

Buffer alloc_buffer(MemType mt, size_t bytes, int active_gpu) {
    Buffer b{};
    b.bytes  = bytes;
    b.mt     = mt;
    b.home_gpu = active_gpu;
    b.owns_host = false;

    void* p = nullptr;
    switch (mt) {
        case MemType::DEV_COARSE: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipMalloc(&p, bytes));
            b.device_ptr = (float*)p;
            b.host_ptr   = (float*)std::malloc(bytes);
            b.owns_host  = true;
            b.raw_ptr    = p;
            break;
        }
        case MemType::DEV_UNCACHED: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
            b.device_ptr = (float*)p;
            b.host_ptr   = (float*)std::malloc(bytes);
            b.owns_host  = true;
            b.raw_ptr    = p;
            break;
        }
        case MemType::DEV_FINEGRAINED: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained));
            b.device_ptr = (float*)p;
            b.host_ptr   = (float*)std::malloc(bytes);
            b.owns_host  = true;
            b.raw_ptr    = p;
            break;
        }
        case MemType::HOST_MAPPED: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipHostMalloc(&p, bytes, hipHostMallocMapped));
            b.host_ptr = (float*)p;
            void* dp = nullptr;
            CHECK(hipHostGetDevicePointer(&dp, p, 0));
            b.device_ptr = (float*)dp;
            b.raw_ptr    = p;
            break;
        }
        case MemType::HOST_COHERENT: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipHostMalloc(&p, bytes,
                                hipHostMallocCoherent | hipHostMallocMapped));
            b.host_ptr = (float*)p;
            void* dp = nullptr;
            CHECK(hipHostGetDevicePointer(&dp, p, 0));
            b.device_ptr = (float*)dp;
            b.raw_ptr    = p;
            break;
        }
        case MemType::HOST_NONCOHERENT: {
            CHECK(hipSetDevice(active_gpu));
            CHECK(hipHostMalloc(&p, bytes,
                                hipHostMallocNonCoherent | hipHostMallocMapped));
            b.host_ptr = (float*)p;
            void* dp = nullptr;
            CHECK(hipHostGetDevicePointer(&dp, p, 0));
            b.device_ptr = (float*)dp;
            b.raw_ptr    = p;
            break;
        }
        case MemType::PEER_COARSE: {
            // Allocate on a different GPU; use from active_gpu.
            int peer_gpu = (active_gpu == 0) ? 1 : 0;
            CHECK(hipSetDevice(peer_gpu));
            CHECK(hipMalloc(&p, bytes));
            // Enable peer access from active_gpu → peer_gpu.
            CHECK(hipSetDevice(active_gpu));
            int can = 0;
            CHECK(hipDeviceCanAccessPeer(&can, active_gpu, peer_gpu));
            if (can) {
                hipError_t e = hipDeviceEnablePeerAccess(peer_gpu, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "EnablePeerAccess failed: %s\n", hipGetErrorString(e));
                }
            }
            b.device_ptr = (float*)p;
            b.host_ptr   = (float*)std::malloc(bytes);
            b.owns_host  = true;
            b.home_gpu   = peer_gpu;
            b.raw_ptr    = p;
            break;
        }
        case MemType::PEER_UNCACHED: {
            int peer_gpu = (active_gpu == 0) ? 1 : 0;
            CHECK(hipSetDevice(peer_gpu));
            CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
            CHECK(hipSetDevice(active_gpu));
            int can = 0;
            CHECK(hipDeviceCanAccessPeer(&can, active_gpu, peer_gpu));
            if (can) {
                hipError_t e = hipDeviceEnablePeerAccess(peer_gpu, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "EnablePeerAccess failed: %s\n", hipGetErrorString(e));
                }
            }
            b.device_ptr = (float*)p;
            b.host_ptr   = (float*)std::malloc(bytes);
            b.owns_host  = true;
            b.home_gpu   = peer_gpu;
            b.raw_ptr    = p;
            break;
        }
    }
    return b;
}

void free_buffer(Buffer& b) {
    switch (b.mt) {
        case MemType::DEV_COARSE:
        case MemType::DEV_UNCACHED:
        case MemType::DEV_FINEGRAINED:
        case MemType::PEER_COARSE:
        case MemType::PEER_UNCACHED:
            CHECK(hipSetDevice(b.home_gpu));
            CHECK(hipFree(b.raw_ptr));
            if (b.owns_host) std::free(b.host_ptr);
            break;
        case MemType::HOST_MAPPED:
        case MemType::HOST_COHERENT:
        case MemType::HOST_NONCOHERENT:
            CHECK(hipHostFree(b.raw_ptr));
            break;
    }
    b.raw_ptr = nullptr;
    b.device_ptr = nullptr;
    b.host_ptr = nullptr;
}

void zero_buffer(Buffer& b, int active_gpu) {
    if (b.mt == MemType::HOST_MAPPED ||
        b.mt == MemType::HOST_COHERENT ||
        b.mt == MemType::HOST_NONCOHERENT) {
        std::memset(b.host_ptr, 0, b.bytes);
        // No GPU-side op needed; the GPU will see zeros via the mapping.
    } else {
        CHECK(hipSetDevice(b.home_gpu));
        CHECK(hipMemset(b.raw_ptr, 0, b.bytes));
        CHECK(hipDeviceSynchronize());
        CHECK(hipSetDevice(active_gpu));
    }
}

void readback_cpu(Buffer& b, int active_gpu) {
    if (b.mt == MemType::HOST_MAPPED ||
        b.mt == MemType::HOST_COHERENT ||
        b.mt == MemType::HOST_NONCOHERENT) {
        // host_ptr already CPU-visible. No copy needed (assumes kernel sync'd).
        // For NonCoherent, would technically need flush; we assume hipDeviceSynchronize
        // covers it for this test.
    } else {
        CHECK(hipSetDevice(b.home_gpu));
        CHECK(hipMemcpy(b.host_ptr, b.raw_ptr, b.bytes, hipMemcpyDeviceToHost));
        CHECK(hipDeviceSynchronize());
        CHECK(hipSetDevice(active_gpu));
    }
}

// ============================================================================
// Test driver
// ============================================================================

struct TestResult {
    const char* op_name;
    const char* mem_name;
    int   n_slots;
    float expected_per_slot;
    float observed_per_slot[3];   // 3 trial runs to detect non-determinism
    bool  correct;
    bool  deterministic;
    bool  ran;                    // false if memtype unsupported / alloc failed
    const char* note;
};

void print_table_header() {
    printf("\n  %-26s  %-44s  %-7s  %-12s  %-12s  %-12s  %-12s  %-7s  %s\n",
           "op", "memtype", "slots", "expected", "trial1", "trial2", "trial3",
           "verdict", "note");
    printf("  %-26s  %-44s  %-7s  %-12s  %-12s  %-12s  %-12s  %-7s  %s\n",
           "--------------------------",
           "--------------------------------------------",
           "-------",
           "------------",
           "------------",
           "------------",
           "------------",
           "-------",
           "----");
}

void print_row(const TestResult& r) {
    if (!r.ran) {
        printf("  %-26s  %-44s  SKIP   %s\n", r.op_name, r.mem_name, r.note ? r.note : "");
        return;
    }
    const char* verdict;
    if (!r.deterministic) verdict = "NONDET";
    else if (r.correct)   verdict = "OK";
    else                  verdict = "WRONG";

    printf("  %-26s  %-44s  %-7d  %-12.1f  %-12.1f  %-12.1f  %-12.1f  %-7s  %s\n",
           r.op_name, r.mem_name, r.n_slots, r.expected_per_slot,
           r.observed_per_slot[0], r.observed_per_slot[1], r.observed_per_slot[2],
           verdict, r.note ? r.note : "");
}

// Run one (op, memtype) combination and return result.
TestResult run_atomic_test(const char* op_name,
                           void (*kernel)(float*, int, int),
                           MemType mt,
                           int active_gpu,
                           int n_slots, int n_blocks, int n_threads, int n_iter) {
    TestResult r{};
    r.op_name  = op_name;
    r.mem_name = memtype_name(mt);
    r.n_slots  = n_slots;
    r.expected_per_slot = (float)((long)n_blocks * n_threads * n_iter) / (float)n_slots;
    r.ran = false;
    r.note = "";

    // Try to allocate; some types may not be supported on this system.
    Buffer b{};
    bool alloc_ok = true;
    {
        // Catch allocation errors for unsupported memory types (e.g. peer if
        // only one GPU is available).
        if (mt == MemType::PEER_COARSE || mt == MemType::PEER_UNCACHED) {
            int n_dev = 0;
            CHECK(hipGetDeviceCount(&n_dev));
            if (n_dev < 2) {
                r.note = "needs >=2 GPUs";
                return r;
            }
        }
        // alloc_buffer aborts on error; protect callers by guarding above.
        b = alloc_buffer(mt, n_slots * sizeof(float), active_gpu);
    }
    if (!alloc_ok) return r;

    r.ran = true;
    r.deterministic = true;
    bool first_set = false;
    float first_val = 0.f;

    // Run 3 trials.
    for (int trial = 0; trial < 3; trial++) {
        zero_buffer(b, active_gpu);
        CHECK(hipSetDevice(active_gpu));
        kernel<<<n_blocks, n_threads>>>(b.device_ptr, n_slots, n_iter);
        CHECK(hipDeviceSynchronize());
        readback_cpu(b, active_gpu);

        // Use slot 0 as the canonical sample (all slots should equal expected).
        float observed = b.host_ptr[0];
        r.observed_per_slot[trial] = observed;
        if (!first_set) {
            first_set = true;
            first_val = observed;
        } else {
            // Compare bit-pattern (NaN-safe): non-determinism flagged if any pair differs.
            uint32_t a, c;
            std::memcpy(&a, &first_val, 4);
            std::memcpy(&c, &observed, 4);
            if (a != c) r.deterministic = false;
        }
    }

    // Correctness: average of trials should equal expected within 0.5 ulp
    // (we used integer 1.0 increments so the result is exactly representable
    // up to 2^24 ≈ 16M).
    float observed = r.observed_per_slot[0];
    float diff = std::fabs(observed - r.expected_per_slot);
    r.correct = (diff <= 0.0f);   // strict bit-equal expected

    // Annotate "interesting" cases.
    if (!r.correct) {
        if (observed == 0.f) r.note = "no-op (atomic silently dropped)";
        else                 r.note = "value differs from CAS reference";
    }

    free_buffer(b);
    return r;
}

// Peer-readback test: writer kernel runs on GPU 0 (HW atomic add to a buffer
// allocated on GPU 1); reader kernel runs on GPU 1 to read the result.
// Compares against host-CPU readback (which goes through L2-flush via
// hipMemcpy + sync).
struct PeerReadbackResult {
    const char* op_name;
    const char* mem_name;
    float expected;
    float cpu_observed;
    float peer_observed;
    bool  cpu_ok;
    bool  peer_ok;
    bool  ran;
    const char* note;
};

PeerReadbackResult run_peer_readback_test(const char* op_name,
                                          void (*writer_kernel)(float*, int, int),
                                          MemType mt,
                                          int writer_gpu,
                                          int n_slots, int n_blocks, int n_threads, int n_iter) {
    PeerReadbackResult r{};
    r.op_name  = op_name;
    r.mem_name = memtype_name(mt);
    r.expected = (float)((long)n_blocks * n_threads * n_iter) / (float)n_slots;
    r.ran = false;
    r.note = "";

    if (mt != MemType::PEER_COARSE && mt != MemType::PEER_UNCACHED) {
        // Only meaningful for peer-mapped buffers.
        return r;
    }
    int n_dev = 0;
    CHECK(hipGetDeviceCount(&n_dev));
    if (n_dev < 2) { r.note = "needs >=2 GPUs"; return r; }

    Buffer b = alloc_buffer(mt, n_slots * sizeof(float), writer_gpu);
    r.ran = true;

    zero_buffer(b, writer_gpu);

    // Writer on writer_gpu (GPU 0), buffer lives on home_gpu (GPU 1).
    CHECK(hipSetDevice(writer_gpu));
    writer_kernel<<<n_blocks, n_threads>>>(b.device_ptr, n_slots, n_iter);
    CHECK(hipDeviceSynchronize());

    // CPU readback (goes through KMD-driven D2H copy, which flushes L2).
    readback_cpu(b, writer_gpu);
    r.cpu_observed = b.host_ptr[0];

    // Peer readback: kernel on home_gpu (GPU 1) reads slot 0 → small device buffer →
    // hipMemcpy back. The interesting question is: can GPU 1 "see" GPU 0's HW
    // atomic writes through its own L2?
    int home_gpu = b.home_gpu;
    CHECK(hipSetDevice(home_gpu));
    float* d_out = nullptr;
    CHECK(hipMalloc(&d_out, sizeof(float)));
    CHECK(hipMemset(d_out, 0xff, sizeof(float)));
    k_peer_readback<<<1, 32>>>(b.device_ptr, d_out, 1);
    CHECK(hipDeviceSynchronize());
    float peer_val = 0.f;
    CHECK(hipMemcpy(&peer_val, d_out, sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipFree(d_out));
    CHECK(hipSetDevice(writer_gpu));
    r.peer_observed = peer_val;

    r.cpu_ok  = (r.cpu_observed  == r.expected);
    r.peer_ok = (r.peer_observed == r.expected);

    if (!r.peer_ok && r.cpu_ok) r.note = "CPU sees correct, peer-GPU stale";
    else if (!r.cpu_ok)         r.note = "writer didn't even reach VRAM";

    free_buffer(b);
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int n_dev = 0;
    CHECK(hipGetDeviceCount(&n_dev));
    int active_gpu = 0;
    CHECK(hipSetDevice(active_gpu));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, active_gpu));
    printf("Device: %s   gcnArch=%s   n_dev=%d\n",
           prop.name, prop.gcnArchName, n_dev);

    // Workload: small enough that even single-slot CAS finishes in seconds.
    // Total adds = N_BLOCKS * N_THREADS * N_ITER must be exactly representable
    // in f32 (≤ 2^24). We use small numbers because the CAS-loop kernel at a
    // single hot slot serializes and runs in O(total_adds) time.
    constexpr int N_BLOCKS  = 4;
    constexpr int N_THREADS = 64;
    constexpr int N_ITER    = 16;
    constexpr int N_SLOTS_HOT = 1;     // single hot slot
    constexpr int N_SLOTS_FAN = 16;    // distributed (N_SLOTS_FAN ≤ N_BLOCKS*N_THREADS)
    // total adds = 4*64*16 = 4096 — small, fast, exactly representable.

    MemType mts[] = {
        MemType::DEV_COARSE,
        MemType::DEV_UNCACHED,
        MemType::DEV_FINEGRAINED,
        MemType::HOST_MAPPED,
        MemType::HOST_COHERENT,
        MemType::HOST_NONCOHERENT,
        MemType::PEER_COARSE,
        MemType::PEER_UNCACHED,
    };

    // ---------- Block A: HW atomic vs CAS vs hand-CAS, hot-slot ----------
    printf("\n=== Block A: HW vs CAS vs hand-CAS, single hot slot, %d blocks × %d threads × %d iters ===\n",
           N_BLOCKS, N_THREADS, N_ITER);
    print_table_header();
    for (auto mt : mts) {
        auto r_hw   = run_atomic_test("unsafeAtomicAdd (HW)", k_atomic_hw,
                                      mt, active_gpu, N_SLOTS_HOT, N_BLOCKS, N_THREADS, N_ITER);
        auto r_cas  = run_atomic_test("atomicAdd (CAS)",     k_atomic_cas,
                                      mt, active_gpu, N_SLOTS_HOT, N_BLOCKS, N_THREADS, N_ITER);
        auto r_hand = run_atomic_test("atomicCAS (hand)",    k_atomic_handcas,
                                      mt, active_gpu, N_SLOTS_HOT, N_BLOCKS, N_THREADS, N_ITER);
        print_row(r_hw); print_row(r_cas); print_row(r_hand);
    }

    // ---------- Block B: HW atomic vs CAS, distributed slots ----------
    printf("\n=== Block B: HW vs CAS, %d distributed slots ===\n", N_SLOTS_FAN);
    print_table_header();
    for (auto mt : mts) {
        auto r_hw  = run_atomic_test("unsafeAtomicAdd (HW)", k_atomic_hw,
                                     mt, active_gpu, N_SLOTS_FAN, N_BLOCKS, N_THREADS, N_ITER);
        auto r_cas = run_atomic_test("atomicAdd (CAS)",      k_atomic_cas,
                                     mt, active_gpu, N_SLOTS_FAN, N_BLOCKS, N_THREADS, N_ITER);
        print_row(r_hw); print_row(r_cas);
    }

    // ---------- Block C: peer-GPU readback ----------
    printf("\n=== Block C: peer-GPU readback (writer on GPU 0, reader on GPU 1) ===\n");
    if (n_dev < 2) {
        printf("  SKIPPED — only %d GPU available.\n", n_dev);
    } else {
        printf("  %-26s  %-44s  %-12s  %-12s  %-12s  %-7s  %-7s  %s\n",
               "op", "memtype", "expected", "cpu_observed", "peer_observed",
               "cpu", "peer", "note");
        printf("  %-26s  %-44s  %-12s  %-12s  %-12s  %-7s  %-7s  %s\n",
               "--------------------------",
               "--------------------------------------------",
               "------------", "------------", "------------",
               "-------", "-------", "----");
        for (auto mt : { MemType::PEER_COARSE, MemType::PEER_UNCACHED }) {
            auto r_hw  = run_peer_readback_test("unsafeAtomicAdd (HW)", k_atomic_hw,
                                                mt, active_gpu,
                                                N_SLOTS_HOT, N_BLOCKS, N_THREADS, N_ITER);
            auto r_cas = run_peer_readback_test("atomicAdd (CAS)",      k_atomic_cas,
                                                mt, active_gpu,
                                                N_SLOTS_HOT, N_BLOCKS, N_THREADS, N_ITER);
            for (auto& r : { r_hw, r_cas }) {
                if (!r.ran) {
                    printf("  %-26s  %-44s  SKIP   %s\n", r.op_name, r.mem_name, r.note ? r.note : "");
                    continue;
                }
                printf("  %-26s  %-44s  %-12.1f  %-12.1f  %-12.1f  %-7s  %-7s  %s\n",
                       r.op_name, r.mem_name, r.expected, r.cpu_observed, r.peer_observed,
                       r.cpu_ok ? "OK" : "WRONG", r.peer_ok ? "OK" : "WRONG",
                       r.note ? r.note : "");
            }
        }
    }

    // ---------- Block D: NaN injection pathology ----------
    printf("\n=== Block D: NaN injection (half threads add NaN, half add 1.0) on dev_coarse ===\n");
    printf("  CAS-loop semantics: NaN immediately propagates → final value is NaN.\n");
    printf("  HW global_atomic_add_f32: depends on FPU NaN handling.\n");
    {
        // Use a fresh, larger workload for NaN test so we have meaningful counts.
        constexpr int N_NAN_BLOCKS  = 8;
        constexpr int N_NAN_THREADS = 64;
        constexpr int N_NAN_ITER    = 16;
        Buffer b = alloc_buffer(MemType::DEV_COARSE, sizeof(float), active_gpu);
        constexpr int TT = N_NAN_BLOCKS * N_NAN_THREADS;
        constexpr int N_NAN = TT / 2;

        zero_buffer(b, active_gpu);
        k_nan_inject<<<N_NAN_BLOCKS, N_NAN_THREADS>>>(b.device_ptr, N_NAN, TT, N_NAN_ITER);
        CHECK(hipDeviceSynchronize());
        readback_cpu(b, active_gpu);
        float hw_v = b.host_ptr[0];

        zero_buffer(b, active_gpu);
        k_nan_inject_cas<<<N_NAN_BLOCKS, N_NAN_THREADS>>>(b.device_ptr, N_NAN, TT, N_NAN_ITER);
        CHECK(hipDeviceSynchronize());
        readback_cpu(b, active_gpu);
        float cas_v = b.host_ptr[0];

        bool hw_nan  = std::isnan(hw_v);
        bool cas_nan = std::isnan(cas_v);
        printf("  HW  result: %s (raw bits %08x)\n", hw_nan  ? "NaN" : "non-NaN",
               *(uint32_t*)&hw_v);
        printf("  CAS result: %s (raw bits %08x)\n", cas_nan ? "NaN" : "non-NaN",
               *(uint32_t*)&cas_v);
        if (hw_nan == cas_nan) printf("  → Equal NaN-handling behavior.\n");
        else                   printf("  → DIVERGENT: HW and CAS disagree on NaN propagation.\n");

        free_buffer(b);
    }

    // ---------- Block E: Denormal pathology ----------
    printf("\n=== Block E: Denormal injection (each thread adds 1e-40f) on dev_coarse ===\n");
    printf("  IEEE-754 (no FTZ): sum approaches expected.\n");
    printf("  FTZ behavior: result is 0 (denormals flushed to zero).\n");
    {
        constexpr int N_DN_BLOCKS  = 8;
        constexpr int N_DN_THREADS = 64;
        constexpr int N_DN_ITER    = 16;
        Buffer b = alloc_buffer(MemType::DEV_COARSE, sizeof(float), active_gpu);

        zero_buffer(b, active_gpu);
        k_denormal_probe<<<N_DN_BLOCKS, N_DN_THREADS>>>(b.device_ptr, N_DN_ITER);
        CHECK(hipDeviceSynchronize());
        readback_cpu(b, active_gpu);
        float hw_v = b.host_ptr[0];

        zero_buffer(b, active_gpu);
        k_denormal_probe_cas<<<N_DN_BLOCKS, N_DN_THREADS>>>(b.device_ptr, N_DN_ITER);
        CHECK(hipDeviceSynchronize());
        readback_cpu(b, active_gpu);
        float cas_v = b.host_ptr[0];

        // Expected (idealized non-FTZ).
        float expected = (float)(N_DN_BLOCKS * N_DN_THREADS * N_DN_ITER) * 1.0e-40f;
        printf("  HW  result: %.6e  (bits %08x)\n", hw_v,  *(uint32_t*)&hw_v);
        printf("  CAS result: %.6e  (bits %08x)\n", cas_v, *(uint32_t*)&cas_v);
        printf("  Idealized non-FTZ: %.6e\n", expected);
        if (hw_v == cas_v)
            printf("  → HW and CAS produce same result (likely both FTZ or both non-FTZ).\n");
        else
            printf("  → DIVERGENT: HW and CAS differ by %g\n", std::fabs(hw_v - cas_v));

        free_buffer(b);
    }

    printf("\nDone.\n");
    return 0;
}
