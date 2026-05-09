#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "megakernel_common.h"
#include "watchdog.h"
#include "rdna3_sync.h"          // braidinfer::rdna3::atomic_block_barrier
#include "op_profile.h"          // OP_PROFILE_BEGIN/END (no-op when -DBRAIDINFER_OP_PROFILE unset)
#include "dump.h"                // dump_instruction_output (used by megakernel_dispatch)

// Megakernel: persistent cooperative kernel that executes a program of instructions.
// Launched once with max cooperative blocks (384 on gfx1100 with 256 threads).
// Each instruction specifies an operation and how many "virtual blocks" it needs.
// Physical blocks loop over virtual blocks: for (vb = blockIdx.x; vb < grid_x; vb += gridDim.x)

// Per-TU GridBarrierState: zero-init __device__ static, cache-coherent VRAM.
// One state suffices — the barrier protocol resets the counter via generation flip.
__device__ static braidinfer::rdna3::GridBarrierState g_gbs;

#include "megakernel_ops.hip"
#include "megakernel_moe.hip"
#include "megakernel_moe_dispatch.hip"
#include "megakernel_dispatch.hip"  // unified dispatch_opcode helper (zqw)

// ============================================================
// Main megakernel entry point
// ============================================================

#define INST_SIZE 18  // 18 u64s per instruction = 144 bytes

extern "C" __global__ __launch_bounds__(256) void megakernel_f32(
    const u64* __restrict__ program,
    int num_instructions,
    WatchdogState* watchdog,  // host-mapped watchdog state; NULL to disable
    uint64_t* op_profile      // op_profile counter buffer (null = disabled)
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // bz0 probe v3: kernel-entry L1 invalidation WITHOUT s_waitcnt (which has hang
    // pathology on gfx1100 — confirmed by PCIe-coherency investigation: vmcnt(0) blocks
    // forever if any vmem op response is delayed). RDNA3 cooperative-launch AQL packets
    // use AGENT scope which flushes/invalidates L2 but NOT per-WGP L1.
    if (threadIdx.x == 0) {
        asm volatile("buffer_gl0_inv\n\t" "buffer_gl1_inv\n\t" ::: "memory");
    }
    __syncthreads();

    // Dump state: encoded in first instruction if OP_NOP with dump_base != 0
    // First instruction words: [0]=opcode(NOP), [1]=dump_base ptr, [2]=dump_capacity, [3]=dump_count ptr
    char* dump_base = nullptr;
    int dump_capacity = 0;
    int* dump_count = nullptr;
    if (num_instructions > 0 && inst_opcode(program) == OP_NOP && ((const NopInst*)program)->dump_buf != nullptr) {
        const NopInst* nop = (const NopInst*)program;
        dump_base = (char*)nop->dump_buf;
        dump_capacity = (int)nop->max_slots;
        dump_count = (int*)nop->dump_counter;
    }

    uint32_t watchdog_beat_ctr = 0;
    // Seed counter from prior value so successive kernel launches against the
    // same WatchdogState produce a monotonically advancing sequence (otherwise
    // every launch stores 100, 200, ... — identical across launches — and the
    // host watchdog spuriously declares no-progress between launches).
    if (watchdog) watchdog_init(watchdog, &watchdog_beat_ctr);

    for (int pc = 0; pc < num_instructions; pc++) {
        // Watchdog: check force_exit between instructions. All blocks reach this point
        // via the grid.sync() at the end of each instruction, so cooperative exit is safe.
        if (watchdog && watchdog_poll_and_check(watchdog, &g_gbs, (uint32_t)pc)) {
            return;  // all blocks return together; no done_flag (megakernel has no persistent done)
        }
        if (watchdog) watchdog_beat(watchdog, &watchdog_beat_ctr);

        const u64* inst = program + pc * INST_SIZE;
        u32 opcode = inst_opcode(inst);
        u32 grid_x = inst_grid_x(inst);

        if (opcode == OP_HALT) break;
        // Per-instruction-dispatch profiling. "One call" = one opcode dispatch
        // including its post-op grid barrier. No-op when -DBRAIDINFER_OP_PROFILE
        // is unset. See PLAN-op-profile.md.
        OP_PROFILE_BEGIN(opcode, op_profile);

        if (opcode == OP_BARRIER) {
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // OP_MOE_FFN manages its own blocks and grid.sync() internally
        if (opcode == OP_MOE_FFN) {
            op_moe_ffn(inst, grid, &g_gbs);
            // op_moe_ffn does its own barrier; skip outer sync
            if (dump_base) {
                dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
                braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            }
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // OP_LINEAR_PROJ_PCG32/RNF4: tiled-LDS optimized GEMV. Manages its own row loop
        // (loads input tile once per block, processes all assigned rows per tile).
        if (opcode == OP_LINEAR_PROJ_PCG32) {
            op_linear_proj_pcg32_full(inst, grid_x);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            if (dump_base) { dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc); braidinfer::rdna3::atomic_block_barrier(&g_gbs); }
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }
        if (opcode == OP_LINEAR_PROJ_RNF4) {
            op_linear_proj_rnf4_full(inst, grid_x);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            if (dump_base) { dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc); braidinfer::rdna3::atomic_block_barrier(&g_gbs); }
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // OP_LM_HEAD (braidinfer-bfd): full-grid bf16 lm_head with LDS-staged
        // input. Bypasses the virtual block loop (rows iterated internally).
        if (opcode == OP_LM_HEAD) {
            op_lm_head(inst, grid_x);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
            if (dump_base) { dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc); braidinfer::rdna3::atomic_block_barrier(&g_gbs); }
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // OP_MOE_DISPATCH: PRE stage — zero GPU 0's output slot + run GPU 0 local experts.
        // Manages its own grid.sync() internally.
        if (opcode == OP_MOE_DISPATCH) {
            op_moe_dispatch(inst, grid, &g_gbs, watchdog);
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // OP_MOE_DISPATCH_POST: sum output_slots across all GPUs into final_output.
        // Fired AFTER CPU wait_ack on workers + after PRE batch completes.
        if (opcode == OP_MOE_DISPATCH_POST) {
            op_moe_dispatch_post(inst, grid, &g_gbs, watchdog);
            OP_PROFILE_END(opcode, op_profile);
            continue;
        }

        // Virtual block loop: each physical block handles multiple virtual blocks
        for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
            switch (opcode) {
                case OP_RMSNORM:      op_rmsnorm(inst, vb); break;
                case OP_LINEAR_PROJ:  op_linear_proj(inst, vb); break;
                case OP_CONV1D:       op_conv1d(inst, vb); break;
                case OP_GDN_GATE:     op_gdn_gate(inst, vb); break;
                case OP_GDN_RECUR:    op_gdn_recurrent(inst, vb); break;
                case OP_RMSNORM_GATE: op_rmsnorm_gated(inst, vb); break;
                case OP_RESIDUAL_ADD: op_residual_add(inst, vb); break;
                case OP_QK_NORM:      op_qk_norm(inst, vb); break;
                case OP_MROPE:        op_mrope(inst, vb); break;
                case OP_GQA_ATTN:     op_gqa_attn(inst, vb); break;
                case OP_OUTPUT_GATE:  op_output_gate(inst, vb); break;
                case OP_FFN_GATE_UP:  op_ffn_gate_up(inst, vb); break;
                case OP_FFN_DOWN_RES: op_ffn_down_res(inst, vb); break;
                case OP_EMBEDDING:    op_embedding(inst, vb); break;
                case OP_D2D_COPY:     op_d2d_copy(inst, vb); break;
                case OP_ATTN_PAGED:   op_attn_paged(inst, vb); break;
                case OP_ATTN_PREFILL: op_attn_prefill(inst, vb); break;
                case OP_DEINTERLEAVE: op_deinterleave(inst, vb); break;
                case OP_KV_QUANTIZE:  op_kv_quantize(inst, vb); break;
                case OP_ATTN_PAGED_Q: op_attn_paged_quant(inst, vb); break;
                case OP_MOE_GATE:    op_moe_gate(inst, vb); break;
                case OP_LINEAR_PROJ_RNF4:  op_linear_proj_rnf4(inst, vb); break;
                case OP_LINEAR_PROJ_PCG32: op_linear_proj_pcg32(inst, vb); break;
                case OP_RMSNORM_WX: op_rmsnorm_wx(inst, vb); break;
                case OP_SILU_MUL:   op_silu_mul(inst, vb); break;
                case OP_FFN_GATE_UP_RNF4:  op_ffn_gate_up_rnf4(inst, vb); break;
                case OP_FFN_DOWN_RES_RNF4: op_ffn_down_res_rnf4(inst, vb); break;
                case OP_SIGMOID_WEIGHTED_ADD: op_sigmoid_weighted_add(inst, vb); break;
                case OP_SCALE_ADD:   op_scale_add(inst, vb); break;
                case OP_RELU_SQ:     op_relu_sq(inst, vb); break;
                case OP_MAMBA2_CONV1D:    op_mamba2_conv1d(inst, vb); break;
                case OP_SSM_UPDATE:       op_ssm_update(inst, vb); break;
                case OP_MAMBA2_NORM_GATED: op_mamba2_norm_gated(inst, vb); break;
                case OP_CONV1D_3X:         op_conv1d_3x(inst, vb); break;
                case OP_FFN_GATE_UP_WX:    op_ffn_gate_up_wx(inst, vb); break;
                case OP_FFN_GATE_UP_RNF4_WX: op_ffn_gate_up_rnf4_wx(inst, vb); break;
                case OP_LINEAR_PROJ_2X:    op_linear_proj_2x(inst, vb); break;
                default:
                    if (threadIdx.x == 0 && blockIdx.x == 0)
                        printf("MEGAKERNEL: unknown opcode %u at pc %d\n", opcode, pc);
                    break;
            }
        }

        // Dump output when enabled (no grid.sync here; caller inserts BarrierInst after dump groups)
        braidinfer::rdna3::atomic_block_barrier(&g_gbs);

        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(&g_gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
    }

    // Signal clean exit so host watchdog pauses its no-progress timer until
    // the next launch (otherwise it spuriously fires on the frozen
    // progress_counter during inter-segment dispatch gaps).
    if (watchdog) watchdog_signal_exited(watchdog);
}
