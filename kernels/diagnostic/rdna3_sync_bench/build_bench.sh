#!/usr/bin/env bash
# Build sync primitive microbenchmark for gfx1100 (RDNA3, wave32).
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-rdna3_sync_bench}

rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    -DHIP_ENABLE_COOPERATIVE_GROUPS=1 \
    --save-temps \
    -o "$OUT" \
    rdna3_sync_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo
  echo "buffer_gl0_inv (workgroup-fence emit):"
  /usr/bin/grep -nE 'buffer_gl0_inv' "$ASM" | wc -l
  echo "buffer_gl1_inv (agent-fence emit):"
  /usr/bin/grep -nE 'buffer_gl1_inv' "$ASM" | wc -l
  echo "s_waitcnt vmcnt(0) (typical fence drain):"
  /usr/bin/grep -nE 's_waitcnt.*vmcnt\(0\)' "$ASM" | wc -l
  echo "s_atomic / global_atomic (counter barrier path):"
  /usr/bin/grep -nE '(s_atomic|global_atomic_add)' "$ASM" | wc -l
fi
