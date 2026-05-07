#!/usr/bin/env bash
# Build reduction primitive microbenchmark for gfx1100 (RDNA3, wave32).
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-reduce_bench}

rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

# Wave32 build (no -mwavefrontsize64). RDNA3 default for HIP is wave32 anyway.
$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    --save-temps \
    -o "$OUT" \
    reduce_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo
  echo "ds_bpermute_b32 instructions emitted (should appear in shfl variants only):"
  /usr/bin/grep -nE 'ds_bpermute_b32' "$ASM" | wc -l
  echo "v_permlanex16_b32 instructions:"
  /usr/bin/grep -nE 'v_permlanex16_b32' "$ASM" | wc -l
  echo "v_permlane16_b32 instructions:"
  /usr/bin/grep -nE 'v_permlane16_b32' "$ASM" | wc -l
  echo "DPP-modified VALU adds (v_add_f32 ... row_shr / quad_perm):"
  /usr/bin/grep -nE 'v_add_f32_dpp.* row_(shr|xmask)' "$ASM" | wc -l
fi
