// WorkerQueue: host-mapped CPU→GPU mailbox for the persistent cooperative
// worker kernel. CPU writes an instruction + increments seq_num; worker
// polls seq_num and runs the dispatched batch.
//
// Layout MUST match crates/braidinfer-runtime/src/persistent_dispatch.rs
// `WorkerQueueLayout` (#[repr(C)]). The Rust side has a
// `static_assert(size_of::<WorkerQueueLayout>() == ...)` to catch any drift.
//
// This header was extracted from kernels/persistent_worker.hip in the
// braidinfer-zqw merge to allow inclusion from both the persistent_worker
// entry point and the megakernel entry point in a single .hip file.
#ifndef BRAIDINFER_WORKER_QUEUE_H
#define BRAIDINFER_WORKER_QUEUE_H

#include <stdint.h>
#include "megakernel_common.h"   // for INST_SIZE_WORDS

// Max instructions per batch dispatch (dense worker).
// Matches MAX_BATCH_INSTRUCTIONS in crates/braidinfer-runtime/src/persistent_dispatch.rs.
#define MAX_BATCH_INSTRUCTIONS 256

struct WorkerQueue {
    volatile uint32_t seq_num;      // monotonic counter, triggers worker
    volatile uint32_t shutdown;     // 1 = exit
    uint32_t num_instructions;      // instructions in this batch (1..MAX_BATCH_INSTRUCTIONS)
    // Diagnostic: each block's thread 0 atomicAdds 1 at kernel entry. Lets the
    // host tell whether the cooperative grid is fully scheduled (count == num_blocks)
    // or only some blocks made it onto the GPU (count < num_blocks). braidinfer-pky.2
    // Phase 0b wedge diagnostic 2026-05-13.
    volatile uint32_t block_alive_count;
    uint64_t inst[MAX_BATCH_INSTRUCTIONS * INST_SIZE_WORDS]; // instruction batch
    volatile uint32_t ack;          // worker writes seq_num when done
    volatile uint32_t done;         // kernel writes 1 when exiting (for Drop polling)
    volatile uint32_t progress_pc;  // worker writes pc before each instruction (for timeout diagnosis)
    uint32_t pc_base;  // ov5m.5: global instruction offset of this batch; trace dump records pc+pc_base
    // op_profile: GPU-resident DeviceBuffer<u64> base pointer, size 2 *
    // BRAIDINFER_OP_PROFILE_NUM_SLOTS. Null when -DBRAIDINFER_OP_PROFILE
    // is unset. See kernels/op_profile.h.
    uint64_t* op_profile;
    // Trace-dump infrastructure (zqw): set by Rust PersistentDispatch::add_device
    // when Model::trace is active. dispatch_opcode reads these to drive
    // dump_instruction_output. Null base = trace disabled.
    // Field order chosen to avoid internal padding (8-byte ptrs first).
    char* dump_base;
    int*  dump_count;
    int   dump_capacity;
    uint32_t _pad3;
};

#endif  // BRAIDINFER_WORKER_QUEUE_H
