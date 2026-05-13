#!/bin/bash
# Joint udi+braidinfer wedge reproducer harness.
# Originated in exterior_algebra/scripts/; designed to be absorbable
# into braidinfer/kernels/diagnostic/persistent_skeleton_repro/ as the
# permanent rdna3 test fixture. All paths are env-var overridable;
# defaults look first in the script's own directory, then in the
# current exterior_algebra layout.
#
# Usage: p2p_wedge_repro_runner.sh [--n-trials N]
# Env overrides:
#   SKELETON_BIN     path to host_runner binary
#   LAUNCH_GPU       command prefix that gates GPU access (e.g. launch-gpu.py)
#   AGGREGATOR       command that produces wedge_repro_matrix.json from JSONL
#   RESULTS_DIR      directory for the output JSON
#   VARIANTS         space-separated variant list (default: V0..V8)
#   N_TRIALS         trials per variant (default: 10)
#
# Pass criterion (applied by aggregator):
#   V0: informative — minimal pattern, does NOT reproduce
#   V1: passes ≥N/N — 2-GPU control, §5.5 envelope
#   V2: passes ≥N/N — no-barrier control
#   V3: passes ≥N/N — non-persistent control
#   V4: informative — peer-GPU UC dispatcher (control-axis)
#   V5: informative — V0 + outer-loop watchdog_alive UC store
#   V7: wedge ≥0.20 — cross-GPU peer-UC + post-poll barrier (the §11.4 reproducer)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Skeleton binary: prefer co-located, fall back to braidinfer-tree default.
if [ -z "${SKELETON_BIN:-}" ]; then
    if [ -x "$SCRIPT_DIR/host_runner" ]; then
        SKELETON_BIN="$SCRIPT_DIR/host_runner"
    else
        SKELETON_BIN="$SCRIPT_DIR/../../braidinfer/kernels/diagnostic/persistent_skeleton_repro/host_runner"
    fi
fi

# Launcher: prefer co-located scripts/launch-gpu.py, fall back to exterior_algebra layout.
if [ -z "${LAUNCH_GPU:-}" ]; then
    if [ -x "$SCRIPT_DIR/launch-gpu.py" ]; then
        LAUNCH_GPU="python $SCRIPT_DIR/launch-gpu.py"
    elif [ -x "$SCRIPT_DIR/../launch-gpu.py" ]; then
        LAUNCH_GPU="python $SCRIPT_DIR/../launch-gpu.py"
    else
        LAUNCH_GPU="python scripts/launch-gpu.py"
    fi
fi

# Aggregator: prefer co-located.
if [ -z "${AGGREGATOR:-}" ]; then
    if [ -x "$SCRIPT_DIR/p2p_wedge_repro_aggregate.py" ]; then
        AGGREGATOR="python $SCRIPT_DIR/p2p_wedge_repro_aggregate.py"
    else
        AGGREGATOR="python scripts/p2p_wedge_repro_aggregate.py"
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
N_TRIALS="${N_TRIALS:-10}"
RESULTS_RAW="/tmp/wedge_repro_raw_$$.jsonl"
RESULTS_FINAL="$RESULTS_DIR/wedge_repro_matrix.json"
mkdir -p "$RESULTS_DIR"

while [ $# -gt 0 ]; do
    case "$1" in
        --n-trials) N_TRIALS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$SKELETON_BIN" ]; then
    python3 - <<PY
import json
print(json.dumps({
    "skeleton_not_built": True,
    "expected_path": "$SKELETON_BIN",
    "hint": "braidinfer-pky.3 must land kernels/diagnostic/persistent_skeleton_repro/host_runner first"
}, indent=2))
PY
    exit 0
fi

declare -A VARIANT_NGPUS=(
    [V0]=4
    [V1]=2
    [V2]=4
    [V3]=4
    [V4]=4
    [V5]=4
    [V6]=4
    [V7]=4
    [V8]=4
)
VARIANTS="${VARIANTS:-V0 V1 V2 V3 V4 V5 V6 V7 V8}"

: > "$RESULTS_RAW"

for variant in $VARIANTS; do
    n_gpus="${VARIANT_NGPUS[$variant]}"
    echo "=== $variant (n_gpus=$n_gpus, trials=$N_TRIALS) ===" >&2
    for trial in $(seq 1 "$N_TRIALS"); do
        $LAUNCH_GPU -g "$n_gpus" --timeout 60 -- \
            chrt -f 50 taskset -c 55 \
            "$SKELETON_BIN" --variant "$variant" --n-gpus "$n_gpus" --n-trials 1 \
            2>>"$RESULTS_RAW.err" >>"$RESULTS_RAW" \
            || echo "{\"event\":\"launch_failed\",\"variant\":\"$variant\",\"trial\":$trial}" >> "$RESULTS_RAW"
    done
done

$AGGREGATOR "$RESULTS_RAW" "$RESULTS_FINAL"
echo "wrote $RESULTS_FINAL"
echo "raw stderr: $RESULTS_RAW.err"
