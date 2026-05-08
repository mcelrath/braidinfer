#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-packed_fma_bench}
rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"
$HIPCC --offload-arch=$ARCH -O3 -std=c++17 -ffp-contract=fast --save-temps -I../.. -o "$OUT" packed_fma_bench.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -n "$ASM" ]]; then
  echo
  echo "Device asm: $ASM"
  echo "  v_dot2_f32_bf16 / v_dot2c_f32_bf16:"
  /usr/bin/grep -cE 'v_dot2c?_(f32_bf16|bf16_bf16)' "$ASM" || true
  echo "  v_dot2_f32_f16 / v_dot2c_f32_f16:"
  /usr/bin/grep -cE 'v_dot2c?_(f32_f16|f16_f16)' "$ASM" || true
  echo "  v_pk_fma_f16 / v_pk_fma_bf16:"
  /usr/bin/grep -cE 'v_pk_fma_(f16|bf16)' "$ASM" || true
  echo "  v_fma_f32 (scalar baseline):"
  /usr/bin/grep -cE 'v_fma_f32' "$ASM" || true
fi

echo "built: $OUT"
