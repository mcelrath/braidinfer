#!/bin/bash
set -e
cd "$(dirname "$0")"
/opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 -std=c++17 \
    -ffp-contract=fast -mwavefrontsize64 \
    mgpu_idle_barrier.hip -I../.. -o mgpu_idle_barrier
