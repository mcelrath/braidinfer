// MoE work queue for GPU-initiated expert dispatch.
// Allocated with hipHostMallocMapped — visible to all GPUs and CPU.
// GPU 0 megakernel writes work items; persistent worker kernels poll.
#pragma once
#include <stdint.h>

#define MOE_MAX_ACTIVE_EXPERTS 32
#define MOE_MAX_GPUS 8
// Maximum tokens per batched prefill dispatch (= CHUNK_TOKENS).
// Fixed so expert_ids/expert_weights arrays have static size — no C VLAs.
#define MOE_MAX_PREFILL_BATCH 64

struct MoeWorkItem {
    // Monotonic sequence number. Workers poll this. 0 = no work.
    volatile uint32_t seq_num;
    // Number of tokens in this dispatch. 1 for decode, up to MOE_MAX_PREFILL_BATCH for prefill.
    uint32_t batch_size;
    uint32_t layer_idx;
    uint32_t num_active;          // k (top-k experts selected per token)
    uint32_t hidden_size;
    uint32_t expert_intermediate_size;
    uint32_t has_gate_proj;       // 1 = gate+up (SiLU*up), 0 = up only (ReLU²)
    uint32_t num_workers;
    // Expert input dimension: hidden_size for standard MoE, moe_latent_size for Nemotron-H.
    // Used by workers for gate_up/down projection in_dim and activation copy count.
    uint32_t gate_up_in_dim;

    // Per-token expert routing: layout [batch_size × num_active], indexed [t * num_active + j].
    // Fixed size: MOE_MAX_PREFILL_BATCH × MOE_MAX_ACTIVE_EXPERTS entries.
    int32_t  expert_ids[MOE_MAX_PREFILL_BATCH * MOE_MAX_ACTIVE_EXPERTS];
    float    expert_weights[MOE_MAX_PREFILL_BATCH * MOE_MAX_ACTIVE_EXPERTS];

    // Explicit padding to ensure 8-byte alignment for activation_ptr.
    // After the header (9×uint32 = 36 bytes) + expert_ids (8192 bytes) + expert_weights (8192 bytes)
    // = 16420 bytes, which is 4 mod 8. Four bytes of padding brings to 16424 (8-byte aligned).
    uint32_t _pad_align;

    // GPU 0 VRAM pointer to expert activation [gate_up_in_dim] (single-token, for decode)
    uint64_t activation_ptr;
    // GPU 0 VRAM pointer to per-worker output slots [batch_size × total_gpus × hidden_size]
    // Layout: output_slots[(t * total_gpus + gpu_idx) * hidden_size], total_gpus = num_workers + 1
    uint64_t output_slots_ptr;

    // Per-worker ack flags. Worker writes seq_num here when done (single ack covers full batch).
    volatile uint32_t ack_flags[MOE_MAX_GPUS];

    // Activation cache: GPU 0 writes batch_size × gate_up_in_dim floats here (GART, bypasses L2).
    // Layout: activation_cache[t * gate_up_in_dim + d] for token t, dimension d.
    // Flexible array — allocation is sizeof(MoeWorkItem) + batch_size * gate_up_in_dim * sizeof(float).
    // Each GPU accesses this via its own per-GPU device VA (from hipHostGetDevicePointer).
    float activation_cache[];
};

// Per-expert entry in worker config (device memory on each worker GPU).
struct MoeExpertEntry {
    uint32_t global_expert_id;
    uint32_t _pad;
    uint64_t gate_up_ptr;   // device pointer to packed Q4 gate_up weights
    uint64_t down_ptr;      // device pointer to packed Q4 down weights
};

// Worker configuration (device memory, one per worker GPU).
struct MoeWorkerConfig {
    uint32_t my_gpu_id;
    uint32_t num_experts_local;
    uint32_t gate_up_row_stride;  // bytes per row in Q4 packed format
    uint32_t hidden_size;
    uint32_t expert_intermediate_size;
    uint32_t _pad[3];
    // Map: global_expert_id → local entry (or NULL if not on this GPU).
    // Indexed by global expert ID. Only populated entries have valid pointers.
    MoeExpertEntry entries[512];
};

// Shutdown flag (host-mapped, one per worker).
// Set to 1 to signal worker to exit its polling loop.
struct MoeShutdownFlag {
    volatile uint32_t shutdown;
};
