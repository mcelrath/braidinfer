#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-dma_under_persistent_bench}

rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    -pthread \
    -I../.. \
    -o "$OUT" \
    dma_under_persistent_bench.hip

echo "built: $OUT"
