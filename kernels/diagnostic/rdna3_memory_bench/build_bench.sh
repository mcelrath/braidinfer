#!/usr/bin/env bash
# Build memory primitive microbenchmark for gfx1100 (RDNA3, wave32).
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-rdna3_memory_bench}

rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    --save-temps \
    -I../.. \
    -o "$OUT" \
    rdna3_memory_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo
  echo "global_load_lds_* instructions emitted:"
  /usr/bin/grep -nE 'global_load_lds_(b|u)' "$ASM" | wc -l
  echo "buffer_gl0_inv instructions:"
  /usr/bin/grep -nE 'buffer_gl0_inv' "$ASM" | wc -l
  echo "buffer_gl1_inv instructions:"
  /usr/bin/grep -nE 'buffer_gl1_inv' "$ASM" | wc -l
  echo "global_atomic_add_f32 instructions:"
  /usr/bin/grep -nE 'global_atomic_add_f32' "$ASM" | wc -l
  echo "buffer_load_(b|u){32,64,128}:"
  /usr/bin/grep -nE 'buffer_load_(b|u)(32|64|128)' "$ASM" | wc -l
fi
