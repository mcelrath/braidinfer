#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-p2p_multicast_bench}
rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"
$HIPCC --offload-arch=$ARCH -O3 -std=c++17 -ffp-contract=fast -I../.. -o "$OUT" p2p_multicast_bench.hip
echo "built: $OUT"
