#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-atomic_failure_envelope}

rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    --save-temps \
    -o "$OUT" \
    atomic_failure_envelope.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo "global_atomic_add_f32:     $(grep -c 'global_atomic_add_f32' "$ASM")"
  echo "global_atomic_cmpswap_b32: $(grep -c 'global_atomic_cmpswap_b32' "$ASM")"
  echo "flat_atomic_cmpswap_b32:   $(grep -c 'flat_atomic_cmpswap_b32' "$ASM")"
fi
