// bd w1li P3: per-op megakernel TEST ENTRIES.
//
// Standalone __global__ symbols that each run ONE op via the SAME __device__
// op_core production's dispatch_opcode uses, so am-rs codeobj parse_kernel(symbol)
// can load + validate each op independently on VFIO. One source of truth: these
// entries and dispatch_opcode call the identical op_<name>; there is no second
// implementation to drift (the false-confidence trap w1li closes).
//
// LOCKED ABI (am-rs-dev bridge #4565):
//   - Symbol: op_<name>_test_entry  (extern "C", one per op, distinct so
//     parse_kernel(symbol) loads each standalone).
//   - Args (kernarg segment; native rsrc2, not inlined USER_DATA):
//       (const u64* inst, void* out)
//     inst = ONE packed instruction — INST_SIZE_WORDS u64, byte-identical to a
//            WorkerQueue mailbox slot. The Rust/am-rs side builds it via the same
//            *Inst -> into_inst() encode the mailbox uses.
//     out  = the op output buffer (host-mapped coherent or VRAM). The encoder sets
//            the instruction's output pointer = out, so the op writes its result
//            there; am-rs reads it back byte-exact vs a CPU reference.
//   - Launch: UNIFORM cooperative for ALL entries (one launch path matching
//     persistent_worker). grid_x = num_rows (op-defined). Vblock ops that need no
//     cross-block sync simply never call grid.sync().
//   - Body: decode grid_x -> virtual-block loop -> op_core(inst, vb).
//
// MUST be #included AFTER megakernel_ops.hip (defines op_rmsnorm et al.) and the
// inst-field helpers (inst_grid_x). See megakernel.hip include order.
#ifndef BRAIDINFER_MEGAKERNEL_TEST_ENTRIES_HIP
#define BRAIDINFER_MEGAKERNEL_TEST_ENTRIES_HIP

// op_rmsnorm (OP_RMSNORM, the y = x·rms·(1+w) variant). Self-contained vblock op:
// each virtual block normalizes one row. op_rmsnorm's reduction uses
// `extern __shared__ float[]`, so the launch must provide dynamic shared memory
// >= blockDim.x floats. No grid.sync (per-row independent) — cooperative launch
// only for ABI uniformity with the ops that do need it.
extern "C" __global__ void op_rmsnorm_test_entry(const u64* inst, void* out) {
    (void)out;  // op_rmsnorm writes to inst->output, which the encoder sets = out.
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_rmsnorm(inst, vb);
    }
}

// op_residual_add (OP_RESIDUAL_ADD, out[i] = src[i] + residual[i]). bd w1li.3 /
// am-rs-dev #4572(4): MINIMAL NON-SPILLING isolation entry. op_rmsnorm SPILLS
// (priv_seg=80 — register-cached reduction); residual_add is pure elementwise
// (no reduction, no dynamic LDS, priv_seg≈0). If THIS retires-with-EOP but
// op_rmsnorm hangs -> the bug is am-rs's spill/scratch retire-ABI; if it ALSO
// hangs -> the kernarg/USER_DATA ABI. Same single-source pattern.
extern "C" __global__ void op_residual_add_test_entry(const u64* inst, void* out) {
    (void)out;  // op_residual_add writes to inst->output, which the encoder sets = out.
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_residual_add(inst, vb);
    }
}

// op_silu_mul (OP_SILU_MUL, out[i] = silu(gate[i]) * up[i]; silu(x)=x/(1+exp(-x))). bd w1li.3:
// non-spilling vblock op. The entry ABI is proven end-to-end on vfio (am-rs #4581 — residual_add
// retired byte-exact), so non-spilling entries validate out-of-the-box. Same single-source
// pattern (calls the production op_silu_mul dispatch_opcode uses).
extern "C" __global__ void op_silu_mul_test_entry(const u64* inst, void* out) {
    (void)out;  // op_silu_mul writes to inst->output, which the encoder sets = out.
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_silu_mul(inst, vb);
    }
}

// bd w1li.3 batch (am-rs #4595/#4596): next vblock ops. scale_add + relu_sq are pure elementwise
// (priv_seg=0, validate out-of-the-box); qk_norm has a per-head reduction so MAY spill (priv_seg>0,
// then it waits for am-rs's .11 wave-offset-SGPR spill-retire fix, like op_rmsnorm). All call the
// SAME production op_core dispatch_opcode uses.
extern "C" __global__ void op_scale_add_test_entry(const u64* inst, void* out) {
    (void)out;
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_scale_add(inst, vb);
    }
}

extern "C" __global__ void op_relu_sq_test_entry(const u64* inst, void* out) {
    (void)out;
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_relu_sq(inst, vb);
    }
}

extern "C" __global__ void op_qk_norm_test_entry(const u64* inst, void* out) {
    (void)out;
    u32 grid_x = inst_grid_x(inst);
    for (u32 vb = blockIdx.x; vb < grid_x; vb += gridDim.x) {
        op_qk_norm(inst, vb);
    }
}

#endif  // BRAIDINFER_MEGAKERNEL_TEST_ENTRIES_HIP
