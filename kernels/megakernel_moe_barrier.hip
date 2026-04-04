
// OP_BARRIER: Park megakernel, signal CPU to dispatch multi-GPU MoE, wait for resume.
//
// Instruction layout:
//   inst[1] = barrier_flag_ptr  (mapped host uint32_t: GPU writes layer_idx+1 to signal CPU)
//   inst[2] = resume_flag_ptr   (mapped host uint32_t: CPU writes 1; GPU resets to 0)
//   inst[3] = layer_idx         (which MoE layer; passed to CPU to look up dispatch params)
//
// Only block 0 executes (grid_x=1). Other blocks skip the vb loop and hit grid.sync().
// Block 0 writes barrier_flag, spins on resume_flag, then exits and hits grid.sync().
// Expert FFN results are written by CPU-dispatched kernels before resume is signaled;
// megakernel reads them fresh (no stale L2 — no prior read of those addresses).
__device__ void op_barrier(const u64* inst, u32 vb) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    volatile uint32_t* barrier_flag = (volatile uint32_t*)(uintptr_t)inst[1];
    volatile uint32_t* resume_flag  = (volatile uint32_t*)(uintptr_t)inst[2];
    uint32_t layer_idx = (uint32_t)inst[3];

    *barrier_flag = layer_idx + 1;
    while (*resume_flag == 0) {
        __builtin_amdgcn_s_sleep(7);
    }
    *resume_flag = 0;
}
