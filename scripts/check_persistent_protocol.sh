#!/bin/bash
# CI gate: verify the persistent_worker ack protocol is intact.
# Builds and runs kernels/diagnostic/persistent_skeleton_repro/prod_kernel_test
# which loads megakernel.hsaco's persistent_worker symbol with a fresh
# WorkerQueue and dispatches 3 sequential batches, asserting ack matches
# seq each time. Catches deferred-ack / off-by-one / barrier-misorder
# regressions in seconds, no model load required.
#
# Standing regression for the 2026-05-14 Phase 2' deferred-ack deadlock fix.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/kernels/diagnostic/persistent_skeleton_repro"
HSACO="$ROOT/target/release/build/braidinfer-hip-7a7510fa032f4834/out/megakernel.hsaco"

if [ ! -f "$HSACO" ]; then
    echo "FAIL: megakernel.hsaco not found at $HSACO — run cargo build first" >&2
    exit 2
fi

if [ ! -f "$D/prod_kernel_test" ] || [ "$D/prod_kernel_test_host.cpp" -nt "$D/prod_kernel_test" ]; then
    /opt/rocm/bin/hipcc -O3 -std=c++17 \
        "$D/prod_kernel_test_host.cpp" -I"$D/../.." -o "$D/prod_kernel_test" 2>&1 | grep -i error && {
        echo "FAIL: prod_kernel_test compile failed" >&2
        exit 3
    }
fi

OUT=$(KERN=persistent_worker N_DISP=3 \
    python3 "$ROOT/scripts/launch-gpu.py" --timeout 60 -- "$D/prod_kernel_test" 2>&1)

EXPECTED_PATTERN='"phase":"dispatch","seq":1,"wedged":false,"ack":1'
EXPECTED_2='"phase":"dispatch","seq":2,"wedged":false,"ack":2'
EXPECTED_3='"phase":"dispatch","seq":3,"wedged":false,"ack":3'

if echo "$OUT" | grep -q "$EXPECTED_PATTERN" \
   && echo "$OUT" | grep -q "$EXPECTED_2" \
   && echo "$OUT" | grep -q "$EXPECTED_3"; then
    echo "Persistent protocol OK: 3/3 dispatches acked correctly."
    exit 0
fi

echo "FAIL: persistent protocol regression. Output:" >&2
echo "$OUT" >&2
echo "" >&2
echo "Likely cause: ack-write ordering violated (deferred-ack pattern reintroduced?)." >&2
echo "Reference: kb persistent-wedge-fix-2026-05-14, kernels/rdna3/rdna3_persistent_protocol.h" >&2
exit 1
