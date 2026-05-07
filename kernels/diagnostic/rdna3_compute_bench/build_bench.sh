#!/usr/bin/env bash
# Build rdna3_compute primitive microbenchmark for gfx1100 (RDNA3, wave32).
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-rdna3_compute_bench}

rm -f *.s *.bc *.ll *.o *.hsaco *.hipi *.hipfb "$OUT"

$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    --save-temps \
    -o "$OUT" \
    rdna3_compute_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo
  echo "v_wmma_f32_16x16x16_bf16 instructions emitted:"
  /usr/bin/grep -nE 'v_wmma_f32_16x16x16_bf16' "$ASM" | wc -l
  echo "v_wmma_f32_16x16x16_f16 instructions emitted:"
  /usr/bin/grep -nE 'v_wmma_f32_16x16x16_f16' "$ASM" | wc -l
  echo "global_atomic_add_f32 instructions emitted:"
  /usr/bin/grep -nE 'global_atomic_add_f32' "$ASM" | wc -l
  echo "ds_bpermute_b32 instructions (should be 0 — we use DPP+permlane):"
  /usr/bin/grep -nE 'ds_bpermute_b32' "$ASM" | wc -l
  echo "v_permlanex16_b32 instructions:"
  /usr/bin/grep -nE 'v_permlanex16_b32' "$ASM" | wc -l
  echo "DPP-modified VALU adds (v_add_f32 ... row_xmask):"
  /usr/bin/grep -nE 'v_add_f32_dpp.* row_(shr|xmask)' "$ASM" | wc -l
  echo
  echo "First WMMA-bearing function (50 lines around first v_wmma):"
  FIRST_WMMA_LINE=$(/usr/bin/grep -nE 'v_wmma_f32_16x16x16_bf16' "$ASM" | head -1 | cut -d: -f1 || true)
  if [[ -n "$FIRST_WMMA_LINE" ]]; then
    START=$((FIRST_WMMA_LINE - 25 > 0 ? FIRST_WMMA_LINE - 25 : 1))
    END=$((FIRST_WMMA_LINE + 25))
    /usr/bin/sed -n "${START},${END}p" "$ASM"
  fi
fi
