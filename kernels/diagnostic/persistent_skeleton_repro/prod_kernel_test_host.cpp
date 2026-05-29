// prod_kernel_test_host: loads the production megakernel.hsaco and launches
// the production `persistent_worker` symbol with a freshly-allocated
// WorkerQueue. Isolates kernel-vs-context: does the production kernel wedge
// even in the standalone skeleton's context (no weight load, no prior coop
// launches), or is the wedge specific to braidinfer's process state?

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

// Use the REAL canonical ABI — do NOT hardcopy. This harness previously hardcoded
// INST_SIZE_WORDS=18 (kernel's is 19) + MAX_BATCH_INSTRUCTIONS + a WorkerQueue
// struct; the stale 18 silently shifted ack/progress_pc by 256*1*8 = 2048 bytes
// (ack read @36880 vs kernel @38928) → a deterministic FALSE dispatch-1 wedge that
// looked like a kernel/loop bug. Including the single source of truth makes drift
// IMPOSSIBLE: worker_queue.h provides the WorkerQueue struct + MAX_BATCH_INSTRUCTIONS,
// and (via megakernel_common.h) INST_SIZE_WORDS + the static_asserts that
// compile-check every typed Inst struct against INST_SIZE_WORDS*8. If the kernel ABI
// changes, this harness fails to COMPILE instead of silently false-wedging.
#include "../../worker_queue.h"   // -> WorkerQueue, INST_SIZE_WORDS, MAX_BATCH_INSTRUCTIONS

#define CHK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %d at %s\n",(int)e,#x); std::exit(2);} } while(0)

int main(int argc, char** argv) {
    const char* hsaco_path = argc > 1 ? argv[1] :
        "/home/mcelrath/Projects/ai/braidinfer/target/release/build/braidinfer-hip-7a7510fa032f4834/out/megakernel.hsaco";
    long memcpy_mb = std::getenv("MEMCPY_MB") ? std::atol(std::getenv("MEMCPY_MB")) : 0;

    CHK(hipSetDevice(0));

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

    hipModule_t module;
    CHK(hipModuleLoad(&module, hsaco_path));
    hipFunction_t func;
    const char* sym = std::getenv("KERN") ? std::getenv("KERN") : "persistent_worker";
    CHK(hipModuleGetFunction(&func, module, sym));
    printf("{\"phase\":\"loaded\",\"sym\":\"%s\"}\n", sym);

    void* host_ptr = nullptr;
    CHK(hipHostMalloc(&host_ptr, sizeof(WorkerQueue), hipHostMallocMapped));
    std::memset(host_ptr, 0, sizeof(WorkerQueue));
    void* dev_queue_ptr = nullptr;
    CHK(hipHostGetDevicePointer(&dev_queue_ptr, host_ptr, 0));
    auto* q_host = (WorkerQueue*)host_ptr;

    hipStream_t stream;
    CHK(hipStreamCreate(&stream));
    int num_cus = 0;
    CHK(hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, 0));
    int num_blocks = 2 * num_cus;
    if (const char* nb = std::getenv("NUM_BLOCKS")) num_blocks = std::atoi(nb);

    void* queue_arg = dev_queue_ptr;
    void* wd_arg = nullptr;
    void* args[2] = { &queue_arg, &wd_arg };

    uint32_t shared_mem = 31776;
    if (const char* sm = std::getenv("SHARED_MEM")) shared_mem = (uint32_t)std::atoi(sm);
    if (hipModuleLaunchCooperativeKernel(func,
            num_blocks, 1, 1, 256, 1, 1, shared_mem, stream,
            args) != hipSuccess) {
        fprintf(stderr, "cooperative launch failed\n");
        return 3;
    }

    auto t0 = std::chrono::steady_clock::now();
    while (q_host->block_alive_count < (uint32_t)num_blocks) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            fprintf(stderr, "block_alive=%u/%d timeout at launch\n",
                    q_host->block_alive_count, num_blocks);
            return 4;
        }
    }
    printf("{\"phase\":\"launched\",\"num_blocks\":%d,\"memcpy_mb\":%ld}\n",
           num_blocks, memcpy_mb);
    fflush(stdout);

    // Send N dispatches, wait for ack each time.
    long n_disp = std::getenv("N_DISP") ? std::atol(std::getenv("N_DISP")) : 3;
    for (long d = 1; d <= n_disp; d++) {
        q_host->inst[0] = 0;  // OP_NOP (production OP_HALT = 16)
        for (int i = 1; i < INST_SIZE_WORDS; i++) q_host->inst[i] = 0;
        q_host->num_instructions = 1;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        q_host->seq_num = (uint32_t)d;

        auto d0 = std::chrono::steady_clock::now();
        bool wedged = true;
        while (true) {
            uint32_t a = q_host->ack;
            if (a == (uint32_t)d) { wedged = false; break; }
            if (q_host->done == 1u) { wedged = false; break; }
            if (std::chrono::steady_clock::now() - d0 > std::chrono::seconds(5))
                break;
        }
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - d0).count();
        printf("{\"phase\":\"dispatch\",\"seq\":%ld,\"wedged\":%s,\"ack\":%u,"
               "\"done\":%u,\"progress_pc\":\"0x%08x\",\"ms\":%ld}\n",
               d, wedged?"true":"false", q_host->ack, q_host->done,
               q_host->progress_pc, ms);
        fflush(stdout);
        if (wedged) break;
    }

    q_host->shutdown = 1u;
    hipStreamSynchronize(stream);
    return 0;
}
