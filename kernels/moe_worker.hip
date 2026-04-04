// Persistent cooperative worker kernel for GPU-initiated MoE expert dispatch.
// Launched once per worker GPU, polls host-mapped work queue for expert FFN tasks.
// Computes Q4 PcG32 GEMV inline (gate_up + silu_mul + down + weighted accumulate).
// Writes results to GPU 0 VRAM via P2P compute write, acks via host-mapped flag.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "moe_work_queue.h"
#include "bf16_utils.h"

// Q4 PcG32 GEMV: one virtual block per output row, cooperative across all blocks.
// weight: packed Q4 data, group_size=32, 20 bytes/group (16B nibbles + 2B min + 2B scale).
// input: f32[in_dim], output: f32[out_dim].
__device__ void worker_gemv_pcg32(
    float* output, const unsigned char* weight, const float* input,
    int out_dim, int in_dim,
    cooperative_groups::grid_group& grid
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int group_size = 32;
    const int group_bytes = 20;
    const int num_groups = (in_dim + group_size - 1) / group_size;
    extern __shared__ float shared[];

    for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
        const unsigned char* row_data = weight + (long long)row * num_groups * group_bytes;
        float acc = 0.0f;
        for (int g = tid; g < num_groups; g += stride) {
            const unsigned char* gp = row_data + g * group_bytes;
            float mn = bf16_to_f32(*(const unsigned short*)(gp + 16));
            float sc = bf16_to_f32(*(const unsigned short*)(gp + 18));
            int base = g * group_size;
            int count = min(group_size, in_dim - base);
            for (int i = 0; i < count; i += 2) {
                unsigned char byte = gp[i / 2];
                float v0 = (float)(byte & 0xF) * sc + mn;
                float v1 = (float)((byte >> 4) & 0xF) * sc + mn;
                acc += v0 * input[base + i] + v1 * input[base + i + 1];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if ((tid & 31) == 0) shared[tid >> 5] = acc;
        __syncthreads();
        if (tid < 8) {
            acc = shared[tid];
            for (int offset = 4; offset > 0; offset >>= 1)
                acc += __shfl_down(acc, offset);
            if (tid == 0) output[row] = acc;
        }
        __syncthreads();
    }
    grid.sync();
}

// SiLU(gate) * up → output, cooperative across all blocks.
__device__ void worker_silu_mul(
    float* output, const float* gate, const float* up, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        float g = gate[i];
        output[i] = (g / (1.0f + expf(-g))) * up[i];
    }
    grid.sync();
}

// ReLU²(x) = relu(x)², cooperative.
__device__ void worker_relu_squared(
    float* output, const float* input, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        float x = input[i];
        float r = x > 0.0f ? x : 0.0f;
        output[i] = r * r;
    }
    grid.sync();
}

// Weighted accumulate: output[i] += weight * input[i], cooperative.
__device__ void worker_weighted_acc(
    float* output, const float* input, float weight, int size,
    cooperative_groups::grid_group& grid
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        output[i] += weight * input[i];
    }
    grid.sync();
}

// Zero a buffer, cooperative.
__device__ void worker_zero(float* buf, int size, cooperative_groups::grid_group& grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        buf[i] = 0.0f;
    }
    grid.sync();
}

// Cooperative P2P copy: all blocks help copy src → dst.
__device__ void worker_copy(float* dst, const float* src, int count,
                            cooperative_groups::grid_group& grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
    grid.sync();
}

extern "C" __global__ __launch_bounds__(256) void moe_worker_kernel(
    volatile MoeWorkItem* work_queue,
    volatile MoeShutdownFlag* shutdown_flag,
    MoeWorkerConfig* config,
    float* local_activation,  // [hidden_size] local VRAM
    float* scratch_gate,      // [expert_intermediate_size] local VRAM
    float* scratch_up,        // [expert_intermediate_size] local VRAM
    float* scratch_act,       // [expert_intermediate_size] local VRAM
    float* local_output       // [hidden_size] local VRAM — accumulated then P2P written
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const uint32_t my_gpu = config->my_gpu_id;
    const uint32_t hs = config->hidden_size;
    const uint32_t eis = config->expert_intermediate_size;

    uint32_t last_seq = 0;

    while (true) {
        // Check shutdown
        if (shutdown_flag->shutdown) return;

        // Poll for new work
        uint32_t seq = work_queue->seq_num;
        if (seq <= last_seq) {
            __builtin_amdgcn_s_sleep(0);
            continue;
        }

        // New work available
        uint32_t k = work_queue->num_active;
        uint32_t has_gate = work_queue->has_gate_proj;
        const float* activation_src = (const float*)work_queue->activation_ptr;
        float* output_slots = (float*)work_queue->output_slots_ptr;
        float* my_output_slot = output_slots + my_gpu * hs;

        // 1. Cooperative P2P copy activation from GPU 0 VRAM to local buffer
        worker_copy(local_activation, activation_src, hs, grid);

        // 2. Zero local output
        worker_zero(local_output, hs, grid);

        // 3. Process each expert assigned to this GPU
        for (uint32_t j = 0; j < k; j++) {
            int32_t eid = work_queue->expert_ids[j];
            float ew = work_queue->expert_weights[j];
            if (eid < 0 || eid >= 256) continue;

            MoeExpertEntry& entry = config->entries[eid];
            if (entry.gate_up_ptr == 0) continue; // not on this GPU

            const unsigned char* gate_up_data = (const unsigned char*)entry.gate_up_ptr;
            const unsigned char* down_data = (const unsigned char*)entry.down_ptr;

            if (has_gate) {
                // Gate projection: gate_up_data[0..eis] → scratch_gate
                worker_gemv_pcg32(scratch_gate, gate_up_data,
                                  local_activation, eis, hs, grid);

                // Up projection: gate_up_data[eis..2*eis] → scratch_up
                const unsigned char* up_data = gate_up_data +
                    (long long)eis * config->gate_up_row_stride;
                worker_gemv_pcg32(scratch_up, up_data,
                                  local_activation, eis, hs, grid);

                // SiLU(gate) * up → scratch_act
                worker_silu_mul(scratch_act, scratch_gate, scratch_up, eis, grid);
            } else {
                // Up only (ReLU²)
                worker_gemv_pcg32(scratch_up, gate_up_data,
                                  local_activation, eis, hs, grid);
                worker_relu_squared(scratch_act, scratch_up, eis, grid);
            }

            // Down projection: scratch_act → scratch_gate (reuse as temp)
            worker_gemv_pcg32(scratch_gate, down_data,
                              scratch_act, hs, eis, grid);

            // Weighted accumulate into local_output
            worker_weighted_acc(local_output, scratch_gate, ew, hs, grid);
        }

        // 4. P2P write local_output to GPU 0's output_slots[my_gpu * hs]
        worker_copy(my_output_slot, local_output, hs, grid);

        // 5. Ack: write seq_num to host-mapped flag
        __threadfence_system();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            work_queue->ack_flags[my_gpu] = seq;
        }

        last_seq = seq;
        grid.sync();
    }
}
