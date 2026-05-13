#!/bin/bash
set -e
cd "$(dirname "$0")"
/opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 -std=c++17 \
    -ffp-contract=fast -mwavefrontsize64 \
    persistent_worker_skeleton.hip host_runner.cpp \
    -I../.. -o host_runner
