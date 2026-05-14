// bulk_write_repro_host: launches bulk_repro_worker once and runs N dispatches
// within a single kernel lifetime, configurable bulk-write size and pre-first-
// poll delay. See bulk_write_repro.hip header for layout/mechanism details.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#define INST_SIZE_WORDS 16
#define MAX_BATCH_INSTRUCTIONS 256

struct ReproQueue {
    volatile uint32_t seq_num;
    volatile uint32_t shutdown;
    uint32_t num_instructions;
    volatile uint32_t block_alive_count;
    uint64_t inst[MAX_BATCH_INSTRUCTIONS * INST_SIZE_WORDS];
    volatile uint32_t ack;
    volatile uint32_t progress_pc;
    volatile uint32_t completed_dispatches;
    uint32_t _pad;
};

extern "C" __global__ void bulk_repro_worker(volatile ReproQueue*);

#define CHK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %d at %s\n",(int)e,#x); std::exit(2);} } while(0)

static long env_long(const char* k, long dflt) {
    if (const char* v = std::getenv(k)) return std::atol(v);
    return dflt;
}

int main(int argc, char** argv) {
    long bulk_bytes        = env_long("BULK_BYTES", 32768);
    long n_dispatches      = env_long("N_DISPATCHES", 10);
    long pre_poll_delay_us = env_long("PRE_POLL_DELAY_US", 0);
    long timeout_ms        = env_long("TIMEOUT_MS", 5000);

    if (bulk_bytes < 0) bulk_bytes = 0;
    if (bulk_bytes > (long)sizeof(((ReproQueue*)0)->inst))
        bulk_bytes = sizeof(((ReproQueue*)0)->inst);

    CHK(hipSetDevice(0));

    // udi #167 (c): simulate production state before queue allocation.
    // BRAIDINFER_PREDMA_MB = MB of device memory to hipMemset before queue
    // alloc, mimicking weight-load DMA pressure (production: ~2400 MB).
    long pre_dma_mb = env_long("PREDMA_MB", 0);
    if (pre_dma_mb > 0) {
        void* dev_buf = nullptr;
        size_t bytes = (size_t)pre_dma_mb * 1024 * 1024;
        CHK(hipMalloc(&dev_buf, bytes));
        CHK(hipMemset(dev_buf, 0, bytes));
        CHK(hipDeviceSynchronize());
    }
    long memcpy_mb = env_long("MEMCPY_MB", 0);
    if (memcpy_mb > 0) {
        size_t bytes = (size_t)memcpy_mb * 1024 * 1024;
        void* host_src = std::malloc(bytes);
        std::memset(host_src, 0xAA, bytes);
        void* dev_dst = nullptr;
        CHK(hipMalloc(&dev_dst, bytes));
        CHK(hipMemcpy(dev_dst, host_src, bytes, hipMemcpyHostToDevice));
        CHK(hipDeviceSynchronize());
        std::free(host_src);
    }

    void* host_ptr = nullptr;
    CHK(hipHostMalloc(&host_ptr, sizeof(ReproQueue), hipHostMallocMapped));
    std::memset(host_ptr, 0, sizeof(ReproQueue));
    void* dev_ptr = nullptr;
    CHK(hipHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    auto* q_host = (ReproQueue*)host_ptr;
    auto* q_dev  = (ReproQueue*)dev_ptr;

    hipStream_t stream; CHK(hipStreamCreate(&stream));
    int num_cus=0;
    CHK(hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, 0));
    int bpsm = 2;
    int num_blocks = bpsm * num_cus;

    void* args[1] = {&q_dev};
    if (hipLaunchCooperativeKernel((const void*)bulk_repro_worker,
                                   dim3(num_blocks,1,1), dim3(256,1,1),
                                   args, 0, stream) != hipSuccess) {
        fprintf(stderr, "launch failed\n"); return 3;
    }

    auto t0 = std::chrono::steady_clock::now();
    while (q_host->block_alive_count < (uint32_t)num_blocks) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            fprintf(stderr, "block_alive=%u/%d timeout at launch\n",
                    q_host->block_alive_count, num_blocks);
            return 4;
        }
    }

    if (pre_poll_delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(pre_poll_delay_us));
    }

    printf("{\"bulk_bytes\":%ld,\"n_dispatches\":%ld,\"pre_poll_delay_us\":%ld,\"num_blocks\":%d}\n",
           bulk_bytes, n_dispatches, pre_poll_delay_us, num_blocks);

    int wedges = 0;
    long inst_n = bulk_bytes / (long)sizeof(uint64_t);
    for (long disp = 1; disp <= n_dispatches; disp++) {
        volatile uint64_t* inst_v = (volatile uint64_t*)q_host->inst;
        for (long i = 0; i < inst_n; i++) {
            inst_v[i] = (uint64_t)(disp << 32 | i);
        }
        *(volatile uint32_t*)&q_host->num_instructions = (uint32_t)inst_n;
        *(volatile uint32_t*)&q_host->seq_num = (uint32_t)disp;

        auto d0 = std::chrono::steady_clock::now();
        bool wedged = true;
        while (true) {
            if ((uint32_t)q_host->ack == (uint32_t)disp) { wedged = false; break; }
            if (std::chrono::steady_clock::now() - d0 > std::chrono::milliseconds(timeout_ms))
                break;
        }
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - d0).count();
        printf("{\"disp\":%ld,\"wedged\":%s,\"ack\":%u,\"progress_pc\":\"0x%08x\","
               "\"completed\":%u,\"ms\":%ld}\n",
               disp, wedged?"true":"false", q_host->ack,
               q_host->progress_pc, q_host->completed_dispatches, ms);
        fflush(stdout);
        if (wedged) { wedges++; break; }
    }

    q_host->shutdown = 1u;
    hipStreamSynchronize(stream);
    hipHostFree(host_ptr);
    return wedges ? 1 : 0;
}
