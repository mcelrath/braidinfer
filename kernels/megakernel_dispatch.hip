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
    // yef5.2: local beat counter for sub-op spins (op_moe_ffn_remote wait loop).
    uint32_t local_watchdog_beat_ctr = 0;

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
        op_moe_ffn_remote(inst, grid, gbs, watchdog, &local_watchdog_beat_ctr);
        if (dump_base) {
            dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
            braidinfer::rdna3::atomic_block_barrier(gbs);
        }
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // bd 0hu3-b: OP_MOE_DISPATCH retired. compile_inner_p2p now emits
    // OP_MOE_FFN_REMOTE for GPU 0 (handled above), unifying the per-GPU
    // MoE expert compute path. OP_MOE_DISPATCH_POST (sum step) is kept.

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

    // bd 174k diag: OP_NOP at end of a batch is a drain primitive (Step 1
    // trailing NOP for producer-readback). Acknowledge it explicitly so the
    // default-case "unknown opcode" printf doesn't fire on every batch.
    if (opcode == OP_NOP) {
        braidinfer::rdna3::atomic_block_barrier(gbs);
        // bd yh8z (narrow fix): force reload of any pointer held live across the
        // barrier (dump_base/dump_count/op_profile loaded once per batch) so a
        // register clobbered across the barrier body isn't dereferenced. NARROW —
        // placed at the use site, NOT inside atomic_block_barrier — so it does NOT
        // re-emit s_waitcnt_vscnt(0) on the ack's host-UC store (a global barrier
        // 'memory' clobber did, re-introducing the §11.4 vscnt ack-stall: am-rs c9).
        asm volatile("" ::: "memory");
        OP_PROFILE_END(opcode, op_profile);
        return;
    }

    // Virtual block loop: each physical block handles multiple virtual blocks.
    // yef5.2: OP_D2D_COPY sentinel signal is fired AFTER the outer barrier below
    // (not inside the vblock body) so all blocks' writes are complete first.
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
            case OP_DOT_SIGMOID_SCALE_ADD: op_dot_sigmoid_scale_add(inst, vb); break;
            case OP_SCALE_ADD:    op_scale_add(inst, vb); break;
            case OP_RELU_SQ:      op_relu_sq(inst, vb); break;
            case OP_MAMBA2_CONV1D:        op_mamba2_conv1d(inst, vb); break;
            case OP_SSM_UPDATE:           op_ssm_update(inst, vb); break;
            case OP_MAMBA2_NORM_GATED:    op_mamba2_norm_gated(inst, vb); break;
            case OP_CONV1D_3X:            op_conv1d_3x(inst, vb); break;
            case OP_FFN_GATE_UP_WX:       op_ffn_gate_up_wx(inst, vb); break;
            case OP_FFN_GATE_UP_RNF4_WX:  op_ffn_gate_up_rnf4_wx(inst, vb); break;
            case OP_LINEAR_PROJ_2X:       op_linear_proj_2x(inst, vb); break;
            case OP_KV_WRITE_PAGED_BATCH: op_kv_write_paged_batch(inst, vb); break;
            default:
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    printf("dispatch_opcode: unknown opcode %u at pc %d\n", opcode, pc);
                break;
        }
    }

    braidinfer::rdna3::atomic_block_barrier(gbs);
    asm volatile("" ::: "memory");  // bd yh8z narrow: reload dump_base/op_profile after barrier (not on barrier -> no ack vscnt)

    // yef5.2 fix: OP_D2D_COPY sentinel signal. Fired HERE (after the outer
    // atomic_block_barrier above) so that ALL blocks have completed their vblock
    // stores before the RELEASE sentinel wakes the worker. The signal was previously
    // inside op_d2d_copy's vblock body (vblock==0), which only guaranteed vblock==0's
    // writes drained — leaving dst[blockDim.x..count-1] potentially in-flight when
    // the worker woke up and read act_p2p → intermittent ~2/8 divergence (yef5.2).
    // After this barrier all blocks have ISSUED their UC-VRAM stores (UC bypasses L2 →
    // straight to VRAM; op_d2d_copy no longer calls __threadfence). The drain-probe below
    // confirms the last element. NOTE: formalizing all-blocks-DRAINED (not just issued)
    // before the sentinel — vs the current barrier+last-elem-probe, validated 10/10
    // empirically — is tracked as a yef5.2 follow-up (restore agent-fence or prove sufficiency).
    if (opcode == OP_D2D_COPY) {
        const D2dCopyInst* d2d = (const D2dCopyInst*)inst;
        if (d2d->signal_ptr != 0) {
            // All blocks have completed their vblock stores (outer barrier above).
            // UC-VRAM writes bypass L2, so no additional __threadfence needed here:
            // each block's stores already went directly to VRAM as they executed.
            // A single drain-probe of dst[n_elems-1] confirms the last written
            // element is visible (UC-VRAM read returns only after VRAM is reached).
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                // Drain-probe: read the LAST element to confirm the full payload is in VRAM.
                // With UC-VRAM dst, a read returning correct data means the write reached VRAM.
                volatile float drain_probe = d2d->dst[(int)d2d->n_elems - 1];
                (void)drain_probe;
                __atomic_store_n(
                    (volatile unsigned int*)d2d->signal_ptr,
                    (unsigned int)d2d->signal_seq,
                    __ATOMIC_RELEASE);
            }
            // Single-block barrier: all other blocks wait for block 0 to finish the
            // sentinel store. This prevents a stale signal_ptr read on the next iteration
            // of the mailbox loop (same barrier as the dump_base path below uses).
            braidinfer::rdna3::atomic_block_barrier(gbs);
            asm volatile("" ::: "memory");
        }
    }

    if (dump_base) {
        dump_instruction_output(dump_base, dump_capacity, dump_count, inst, opcode, pc);
        braidinfer::rdna3::atomic_block_barrier(gbs);
        asm volatile("" ::: "memory");  // bd yh8z narrow
    }
    OP_PROFILE_END(opcode, op_profile);
}

#endif  // BRAIDINFER_MEGAKERNEL_DISPATCH_HIP
