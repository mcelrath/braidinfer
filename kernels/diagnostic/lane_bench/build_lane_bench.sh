#!/usr/bin/env bash
# Build the lane-primitive microbenchmark for gfx1100 (RDNA3, wave32).
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-rdna3_lane_bench}

rm -f *.s *.bc *.ll *.o *.hsaco *.hipi *.hipfb *.resolution.txt "$OUT"

# Wave32 build (HIP default for gfx1100). Include path covers ../../<root>/kernels.
$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    -I"$(dirname "$0")/../.." \
    --save-temps \
    -o "$OUT" \
    rdna3_lane_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo
  echo "==== Instruction counts in device asm ===="
  for instr in \
      ds_bpermute_b32 \
      ds_swizzle_b32 \
      v_permlane16_b32 \
      v_permlanex16_b32 \
      v_readlane_b32 \
      v_readfirstlane_b32 \
      v_writelane_b32 \
      v_cmp_lt_f32 \
      v_cndmask_b32 \
      v_cmp_gt_f32 \
      v_cmp_neq_f32 \
      v_max_f32_dpp \
      v_add_f32_dpp \
      v_mov_b32_dpp \
      s_nop ; do
    n=$(/usr/bin/grep -cE "^[[:space:]]*${instr}\b" "$ASM" || true)
    printf "  %-22s  %d\n" "$instr" "$n"
  done
  echo
  echo "==== bench_lane_bcast_const0: should be a v_readlane_b32, NOT ds_bpermute ===="
  /usr/bin/awk '/_Z[0-9]+.*bench_lane_bcast_const0/{p=1} p && /^[[:space:]]*\./{exit} p' "$ASM" | \
    /usr/bin/grep -E "v_readlane_b32|ds_bpermute_b32|v_permlanex16_b32|s_nop|s_endpgm" | head -20 || true
  echo
  echo "==== bench_xor_butterfly_swizzle: should emit ds_swizzle_b32 ===="
  /usr/bin/awk '/_Z[0-9]+.*bench_xor_butterfly_swizzle/{p=1} p && /^[[:space:]]*\./{exit} p' "$ASM" | \
    /usr/bin/grep -E "ds_swizzle_b32|ds_bpermute_b32|s_nop|s_endpgm" | head -20 || true
  echo
  echo "==== bench_ballot_w32: should emit v_cmp_* / s_mov_b32, NO ds_bpermute ===="
  /usr/bin/awk '/_Z[0-9]+.*bench_ballot_w32/{p=1} p && /^[[:space:]]*\./{exit} p' "$ASM" | \
    /usr/bin/grep -E "v_cmp_|s_mov_b32|v_cndmask|s_endpgm" | head -20 || true
  echo
  echo "==== corr_lane_write: should emit v_writelane_b32 ===="
  /usr/bin/awk '/_Z[0-9]+.*corr_lane_write/{p=1} p && /^[[:space:]]*\./{exit} p' "$ASM" | \
    /usr/bin/grep -E "v_writelane_b32|v_readfirstlane_b32|v_mov_b32|s_endpgm" | head -20 || true
  echo
fi
