// Tiny <<<1,1>>> kernels for stream-side mailbox signalling without HIP API
// sync calls. Pattern adapted from llama.cpp ep_sync_signal_kernel /
// ep_sync_wait_kernel (commits 8277eac83, 45f564e6d).
//
// Use case: when one device has a running cooperative kernel, ANY HIP
// runtime sync API (hipStreamSynchronize, hipEventSynchronize, etc.) on
// that device deadlocks because HIP's internal SyncAllStreams waits for
// the cooperative kernel to complete. To wait for a non-cooperative
// kernel on the same device without that deadlock, write a host-mapped
// flag at the end of the kernel chain and have the host poll it directly
// via read_volatile (no HIP API).
//
// set_flag_kernel: launched with <<<1,1>>>; only one CU/wave is busy.
// wait_flag_kernel: stream-side spin (used for GPU-side ordering, with
// timeout cap to avoid TDR if signal never arrives).

#include <hip/hip_runtime.h>

#ifndef SYNC_FLAG_TIMEOUT_TICKS
// ~1s @ sclk=2.5GHz on gfx1100. Far below TDR threshold (~5s), well above
// any realistic wait latency.
#define SYNC_FLAG_TIMEOUT_TICKS 2500000000ULL
#endif

// Set *flag = value with system-scope release fence so a CPU reader (or
// a peer GPU reading via P2P) sees prior writes from this stream before
// the flag flip.
extern "C" __global__ void set_flag_kernel(volatile uint32_t* flag, uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence();
        *flag = value;
    }
}

// Spin until *flag == expect (or timeout). Used to make a stream wait
// for a host-written or peer-GPU-written mailbox before the next kernel
// in the same stream runs. Single thread, single block — minimal CU
// occupancy so the rest of the GPU is free for other work.
extern "C" __global__ void wait_flag_kernel(volatile const uint32_t* flag, uint32_t expect) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint64_t deadline = clock64() + SYNC_FLAG_TIMEOUT_TICKS;
        uint32_t iter_lo = 0;
        while (*flag != expect) {
            __threadfence();
            if ((++iter_lo & 0xFFFF) == 0) {
                if (clock64() > deadline) return; // TDR-safe bailout
            }
        }
        __threadfence();
    }
}
