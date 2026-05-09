// Unified opcode-dispatch body for megakernel.hip and persistent_worker.hip.
//
// One helper, called from both kernels' driver loops. Eliminates the
// parallel-dispatch-table maintenance burden documented in
// PLAN-merge-megakernel-dispatch.md (epic braidinfer-zqw). Each kernel
// keeps its own outer driver loop (one-shot for-loop vs persistent
// mailbox-poll) but they share this opcode router.
//
// MUST be `#include`d AFTER megakernel_ops.hip / megakernel_moe.hip /
// megakernel_moe_dispatch.hip — depends on every per-op symbol they
// define. AND after rdna3_sync.h (atomic_block_barrier) and
// op_profile.h (OP_PROFILE_BEGIN/END).
//
// `__forceinline__` is required: the helper IS the dispatch body, and
// inlining it is what makes this refactor zero-cost. Verify post-build
// with `--save-temps` if perf is suspicious.
#ifndef BRAIDINFER_MEGAKERNEL_DISPATCH_HIP
#define BRAIDINFER_MEGAKERNEL_DISPATCH_HIP

// Run a single opcode dispatch including its post-op grid barrier and
// optional trace dump.
//
// Caller responsibilities:
// - Filter OP_HALT before calling (megakernel breaks the driver loop;
//   persistent_worker should never see it via its mailbox protocol).
// - Provide an instruction pointer in shared/global memory; the helper
//   re-extracts opcode and grid_x.
// - Pass `pc` for trace-dump labelling and op_profile slot mapping.
// - Pass `dump_base = nullptr` (and any of `dump_capacity`/`dump_count`)
//   when trace mode is disabled. The dump path is gated on `dump_base`
//   non-null.
// - Pass `op_profile = nullptr` when profiling is disabled (gated by
//   the OP_PROFILE_BEGIN/END macros' null-pointer guard).
__device__ __forceinline__ void dispatch_opcode(
    const u64* inst,
    cooperative_groups::grid_group& grid,
    braidinfer::rdna3::GridBarrierState* gbs,
    char* dump_base,
    int dump_capacity,
    int* dump_count,
    uint64_t* op_profile,
    int pc,
    WatchdogState* watchdog
) {
    u32 opcode = inst_opcode(inst);
    u32 grid_x = inst_grid_x(inst);

    OP_PROFILE_BEGIN(opcode, op_profile);

    if (opcode == OP_BARRIER) {
        braidinfer::rdna3::atomic_block_barrier(gbs);
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // OP_MOE_FFN manages its own blocks and grid.sync() internally.
    if (opcode == OP_MOE_FFN) {
        op_moe_ffn(inst, grid, gbs);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // OP_MOE_FFN_REMOTE — per-worker P2P FFN dispatch on workers GPUs 1..N-1.
    // Manages its own grid.sync() internally.
    if (opcode == OP_MOE_FFN_REMOTE) {
        op_moe_ffn_remote(inst, grid, gbs);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // OP_MOE_DISPATCH PRE: zero GPU 0 output slot + run GPU 0 local experts.
    // Manages its own grid.sync() internally.
    if (opcode == OP_MOE_DISPATCH) {
        op_moe_dispatch(inst, grid, gbs, watchdog);
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // OP_MOE_DISPATCH_POST: sum output_slots across GPUs into final_output.
    if (opcode == OP_MOE_DISPATCH_POST) {
        op_moe_dispatch_post(inst, grid, gbs, watchdog);
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // Tiled-LDS optimized GEMVs (manage their own row loop internally).
    if (opcode == OP_LINEAR_PROJ_PCG32) {
        op_linear_proj_pcg32_full(inst, grid_x);
        braidinfer::rdna3::atomic_block_barrier(gbs);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }
    if (opcode == OP_LINEAR_PROJ_RNF4) {
        op_linear_proj_rnf4_full(inst, grid_x);
        braidinfer::rdna3::atomic_block_barrier(gbs);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // OP_LM_HEAD (braidinfer-bfd): full-grid bf16 lm_head with LDS-staged
    // input. Bypasses the virtual block loop.
    if (opcode == OP_LM_HEAD) {
        op_lm_head(inst, grid_x);
        braidinfer::rdna3::atomic_block_barrier(gbs);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // Virtual block loop: each physical block handles multiple virtual blocks.
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
            case OP_MOE_GATE:     op_moe_gate(inst, vb); break;
            case OP_LINEAR_PROJ_RNF4:    op_linear_proj_rnf4(inst, vb); break;
            case OP_LINEAR_PROJ_PCG32:   op_linear_proj_pcg32(inst, vb); break;
            case OP_RMSNORM_WX:   op_rmsnorm_wx(inst, vb); break;
            case OP_SILU_MUL:     op_silu_mul(inst, vb); break;
            case OP_FFN_GATE_UP_RNF4:    op_ffn_gate_up_rnf4(inst, vb); break;
            case OP_FFN_DOWN_RES_RNF4:   op_ffn_down_res_rnf4(inst, vb); break;
            case OP_SIGMOID_WEIGHTED_ADD: op_sigmoid_weighted_add(inst, vb); break;
            case OP_SCALE_ADD:    op_scale_add(inst, vb); break;
            case OP_RELU_SQ:      op_relu_sq(inst, vb); break;
            case OP_MAMBA2_CONV1D:        op_mamba2_conv1d(inst, vb); break;
            case OP_SSM_UPDATE:           op_ssm_update(inst, vb); break;
            case OP_MAMBA2_NORM_GATED:    op_mamba2_norm_gated(inst, vb); break;
            case OP_CONV1D_3X:            op_conv1d_3x(inst, vb); break;
            case OP_FFN_GATE_UP_WX:       op_ffn_gate_up_wx(inst, vb); break;
            case OP_FFN_GATE_UP_RNF4_WX:  op_ffn_gate_up_rnf4_wx(inst, vb); break;
            case OP_LINEAR_PROJ_2X:       op_linear_proj_2x(inst, vb); break;
            default:
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    printf("dispatch_opcode: unknown opcode %u at pc %d\n", opcode, pc);
                break;
        }
    }

    braidinfer::rdna3::atomic_block_barrier(gbs);

    if (dump_base) {
        dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
        braidinfer::rdna3::atomic_block_barrier(gbs);
    }
    OP_PROFILE_END(opcode, op_profile);
}

#endif  // BRAIDINFER_MEGAKERNEL_DISPATCH_HIP
