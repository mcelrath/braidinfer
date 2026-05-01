#!/bin/bash
# Benchmark all models in models/ for decode and prefill tok/s.
# Must be invoked via launch-gpu.py (which sets HIP_VISIBLE_DEVICES):
#   python3 scripts/launch-gpu.py -g 4 --gpu-timeout 43200 --timeout 7200 -- \
#     bash scripts/benchmark_all.sh [output_dir]
set -uo pipefail

OUTDIR="${1:-benchmark_results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
BINARY="target/release/braid_bench"

if [[ ! -x "$BINARY" ]]; then
    echo "Building braid_bench..."
    cargo build --release -p braidinfer-runtime --bin braid_bench 2>&1
fi

echo "Results → $OUTDIR"

for bqnt in models/*.bqnt; do
    name=$(basename "$bqnt" .bqnt)
    outfile="$OUTDIR/${name}.txt"
    echo "=== $name ===" | tee -a "$OUTDIR/summary.txt"
    # Run binary directly — GPUs already reserved by outer launch-gpu.py invocation.
    if MODEL="$bqnt" BENCH_WARMUP=2 BENCH_RUNS=5 \
        timeout 600 "$BINARY" 2>&1 \
        | tee "$outfile" \
        | grep -E "tok/s|PASS|FAIL|positions|Coherence|Multi-GPU"; then
        true
    else
        echo "  SKIPPED (exit $?)" | tee -a "$OUTDIR/summary.txt"
    fi
    echo "" | tee -a "$OUTDIR/summary.txt"
done

echo "Done. Results in $OUTDIR/"
