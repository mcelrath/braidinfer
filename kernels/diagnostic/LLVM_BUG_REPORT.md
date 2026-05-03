# [AMDGPU][gfx11] `ds_bpermute_b32` with `vdst == vdata` produces non-deterministic results in wave reductions; emitted under `-ffp-contract=fast`

## Summary

On AMD gfx1100 (RDNA3, RX 7900 XTX), `__shfl_down`-based warp reductions can
produce **bit-level non-deterministic results across consecutive identical
runs in the same process** when the compiler emits `ds_bpermute_b32 vDst, vIdx, vSrc`
with `vDst == vSrc`. The same-VGPR emit pattern is reachable from valid HIP
C++ source under `-ffp-contract=fast`.

## Reproducer

Minimal repro, ~80 lines of HIP, attached:
`https://github.com/<your-fork>/bpermute_repro/...` (or paste inline).

```hip
__device__ __forceinline__ float warp_sum_shfl_down(float v) {
    v += __shfl_down(v, 32);
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

__global__ __launch_bounds__(256)
void repro_kernel(const float* q, const float* k, const float* x,
                  float* out, int N, int run) {
    int tid = threadIdx.x;
    // PATTERN A (control): inputs are two distinct loads.
    float a = q[tid] * k[tid];
    float ra = warp_sum_shfl_down(a);

    // PATTERN B (suspect): self-multiply, FMA-fusable.
    float xv = x[tid];
    float b  = xv * xv;
    float rb = warp_sum_shfl_down(b);

    if ((tid & 63) == 0) {
        int slot = (run * 4 + (tid >> 6)) * 2;
        out[slot + 0] = ra;
        out[slot + 1] = rb;
    }
}
```

Build: `hipcc --offload-arch=gfx1100 --genco -O3 -ffp-contract=fast
-mwavefrontsize64 --save-temps -o repro repro.hip`

Run: `./repro 1000` (1000 back-to-back launches, same input buffer).

## Expected vs observed

**Expected**: pattern A and pattern B both produce 1 unique output across all
1000 runs (deterministic GPU compute given identical input).

**Observed (in production reduction kernel feeding an online-softmax max)**:
- Pattern A: bit-stable (1/1000 unique)
- Pattern B: ~hundreds of distinct outputs over 1000 runs, in the same process
- All logits downstream of the softmax-amplified pattern B diverge
  (`max_abs_diff = 4.22` on a model where weights and inputs are byte-identical
  across runs)

The reproducer above demonstrates the asm pattern; manifestation as
non-determinism depends on wave-on-WGP contention which is heavier in the
production cooperative megakernel than in a single-block standalone test. We
do not yet have a self-contained reproducer that triggers the non-determinism
on the standalone test alone — the asm-level smoking gun is the same.

## Smoking-gun asm

`-save-temps` output for the reproducer:

```
27: v_mul_f32_e32 v7, v1, v1        ; pattern B input: x*x
28: ds_bpermute_b32 v6, v5, v6      ; <<< vDst == vData; same VGPR
29: ds_bpermute_b32 v5, v5, v7      ; control: distinct VGPRs
36: v_fmac_f32_e32 v6, v2, v3       ; FMA fuses bpermute_result + x*x;
                                    ; recomputing x*x from v2, v3 frees the
                                    ; producer VGPR and lets regalloc alias
                                    ; vDst == vData on the bpermute above.
```

The same pattern in production (reduced from a 27,882-line megakernel asm,
2 of 221 `ds_bpermute_b32` instructions had `dst == src`):

```
v_mul_f32_e32 v81, v80, v80        ; v81 = my_k * my_k (warp_reduce input)
ds_bpermute_b32 v81, v55, v81      ; <<< vDst == vData
s_waitcnt lgkmcnt(0)
v_fmac_f32_e32 v81, v80, v80       ; FMA: v81 += my_k*my_k (recomputed from v80)
```

The two same-VGPR sites in the production binary were exactly the two
non-deterministic call sites identified by application-level A/B testing
(K-rms reductions in paged attention).

## Hypothesized mechanism

`ds_bpermute_b32` semantically:
1. Each lane writes its `vSrc` register value to a slot in the LDS routing fabric.
2. Each lane reads the value at index `vIdx`.
3. Each lane writes the result to its `vDst` register.

When `vDst == vSrc`, lane L's writeback to its own register may race with
the LDS fabric's read of lane L's vSrc requested by lanes other than L. If
the fabric reads occur out-of-order with the writeback, some other lanes
see the post-write value (already permuted) instead of the pre-write value.

Wave-on-WGP contention (multiple co-resident wave32/wave64s on the same
SIMD32) appears to be required to produce the manifestation. The asm
pattern alone is necessary but not sufficient.

## Why the compiler emits this

`-ffp-contract=fast` allows LLVM to fuse `bpermute(x²) + x²` into a single
`v_fmac_f32`. The FMA recomputes `x²` from its source operand registers
rather than reading the previously-computed `x²` from a temp register.
This makes the producer (`v_mul_f32`) register dead immediately after
the bpermute, so the register allocator picks the producer's VGPR as the
bpermute destination, saving one VGPR.

Without `-ffp-contract=fast`, the FMA fusion does not happen and the
register allocator keeps producer + bpermute on distinct VGPRs.

## Source code evidence in LLVM

1. `llvm/lib/Target/AMDGPU/DSInstructions.td`, class `DS_1A1D_PERMUTE`:
   no tied-operand constraint between `$vdst` and `$data0`. Register
   allocator is free to alias them.

2. `llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp`: handles
   `V_PERMLANE*` family hazards (`isPermlane()`,
   `fixVcmpxPermlaneHazards()`, `checkPermlaneHazards()`) but excludes
   `DS_BPERMUTE_B32` and `DS_PERMUTE_B32` from operand-aliasing /
   wait-state checks.

3. PR https://github.com/llvm/llvm-project/pull/117287 and Phab D127344
   added VALU→permlane hazard fixes for gfx950 (CDNA3) but did not
   extend the fix to gfx11.

## Suggested fix directions

| Option | Where | Tradeoff |
|---|---|---|
| Forbid `vDst == vSrc` in regalloc for `DS_BPERMUTE_B32` / `DS_PERMUTE_B32` on gfx11 | `AMDGPUInstructionSelector` or per-target ISA-info | Costs at most 1 extra VGPR per shfl-reduction; clean. |
| Insert `s_nop`/wait-state in hazard recognizer when `vDst == vSrc` | `GCNHazardRecognizer.cpp` | Needs hardware confirmation of the actual wait-state count. |
| Extend the gfx950 permlane hazard fix to cover `ds_bpermute_b32` on gfx11 | Same PR areas | Most consistent with prior art. |

## Affected versions

- ROCm 7.x (likely earlier too)
- LLVM/clang 22.0.0git (and very likely all prior versions on gfx11)
- gfx1100 (RX 7900 XTX) confirmed; other RDNA3 SKUs likely affected

## System

```
GPU: 8× AMD Radeon RX 7900 XTX (gfx1100), Navi 31, RDNA3
ROCm: 7.x
hipcc: AMD LLVM 22.0.0git
Host: Linux 7.0.3-arch1-2-p2p, AMD CPU
```

## Application-level workaround used in production

```c
// Shared-memory tree reduction, no __shfl_down. Bit-exact across runs.
__device__ __forceinline__ float tree_reduce_sum_256(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    return shared[0];
}
```
