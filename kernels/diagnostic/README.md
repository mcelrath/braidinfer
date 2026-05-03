# bpermute_repro — minimal reproducer for `ds_bpermute_b32` non-determinism on gfx1100

## What this is

A standalone HIP test that exhibits non-deterministic results from a `__shfl_down`
based warp reduction on AMD gfx1100 (RDNA3) when the compiler emits
`ds_bpermute_b32 vDst, vIdx, vSrc` with `vDst == vSrc`.

Discovered while debugging `braidinfer` (LLM inference engine) — the K-rms
reduction in paged attention produced different logits across consecutive
identical decode steps in the same process. Localized to one ISA pattern.

## Files

| File | Role |
|---|---|
| `bpermute_repro.hip` | Kernel + host driver. Two reductions: pattern A (control, `q*k`), pattern B (suspect, `x*x`). Runs each N=1000 times, tallies distinct outputs. |
| `build_repro.sh` | Compile with `hipcc --save-temps` and grep emitted asm for the same-VGPR pattern. |
| `bpermute_repro-hip-amdgcn-amd-amdhsa-gfx1100.s` | Captured device asm, demonstrates the smoking-gun emit. |

## How to build

```bash
./build_repro.sh
```

Builds `bpermute_repro` (host executable) and dumps device asm with `--save-temps`.

## How to run (requires gfx1100 GPU)

```bash
./bpermute_repro 1000
```

Expected output if bug reproduces:

```
Pattern A (dst != src): unique=1 / 1000   first_divergence_run=-1
Pattern B (dst == src): unique=K / 1000   first_divergence_run=R
NON-DETERMINISM REPRODUCED: ...
```

If both stable, regalloc may have happened to allocate distinct VGPRs — re-check
the asm. The hazard requires both: same-VGPR emit AND wave-on-WGP contention.

## Smoking-gun asm

From `bpermute_repro-hip-amdgcn-amd-amdhsa-gfx1100.s`:

```
27: v_mul_f32_e32 v7, v1, v1        ; pattern B (x*x)
28: ds_bpermute_b32 v6, v5, v6      ; <<< dst == src — BUG PATTERN
29: ds_bpermute_b32 v5, v5, v7      ; dst != src — control
36: v_fmac_f32_e32 v6, v2, v3       ; FMA fusion that frees v6 (the alias mechanism)
```

Of 12 `ds_bpermute_b32` instructions in the test, exactly 1 has `dst == src`
(the FMA-fusable self-multiply path). The 11 controls do not.

## Why this happens

`-ffp-contract=fast` allows LLVM to fuse `bpermute(x²) + x²` into a single
`v_fmac_f32`. This recomputes `x²` from the original operand register, freeing
the producer VGPR. The register allocator then picks the same VGPR for both the
`ds_bpermute_b32` destination and its source, since the source is no longer live
past the bpermute.

The hardware behavior of `ds_bpermute_b32 vX, vY, vX` on RDNA3 wave32/wave64 is
not well-defined: the destination writeback to lane L's vX appears to race with
the cross-lane fabric's read of lane L's vX requested by other lanes. Different
runs see different orderings, producing non-deterministic results.

## Filing target

Best filed at https://github.com/llvm/llvm-project (label: `backend:AMDGPU`).
Cross-post to https://github.com/ROCm/ROCm.

Title:
> `[AMDGPU][gfx11] ds_bpermute_b32 with vdst == vdata produces non-deterministic
> results in wave reductions; emitted under -ffp-contract=fast`

## Background notes

- LLVM `GCNHazardRecognizer.cpp` handles `V_PERMLANE*` family hazards (see
  `isPermlane()`, `fixVcmpxPermlaneHazards()`, `checkPermlaneHazards()`) but
  excludes `DS_BPERMUTE_B32` and `DS_PERMUTE_B32` from operand-aliasing /
  wait-state checks.
- `DS_1A1D_PERMUTE` class in `llvm-project/llvm/lib/Target/AMDGPU/DSInstructions.td`
  has no tied-operand constraint between `$vdst` and `$data0`.
- AMD landed VALU→permlane hazard fixes for gfx950 (CDNA3) in PR
  https://github.com/llvm/llvm-project/pull/117287 and Phab D127344, but
  did NOT extend the fix to gfx11.
- AMD GPUOpen tutorial on cross-lane operations uses same-VGPR `ds_bpermute_b32`
  as a legitimate idiom — implies the issue is a hardware/firmware regression on
  RDNA3, not a documented ISA constraint.

## Workaround in production code

Replace `__shfl_down`-based block reductions with a shared-memory tree reduction
at any sum-of-squares site that feeds an online-softmax max comparison. See
`tree_reduce_sum_256` in `kernels/megakernel_ops.hip`.
