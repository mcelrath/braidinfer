#!/bin/bash
# Collect rocprofv3 hardware counter data for a single model.
# Usage: MODEL=models/foo.bqnt bash scripts/profile_model.sh [output_dir]
set -euo pipefail

MODEL="${MODEL:-models/qwen35_2b.q4.bqnt}"
OUTDIR="${1:-profile_results/$(basename "$MODEL" .bqnt)}"
mkdir -p "$OUTDIR"

# Key counters for LLM decode analysis:
#   OccupancyPercent / MeanOccupancyPerCU  — wave occupancy (target: >50%)
#   VALUInsts / SALUInsts                  — VALU vs SALU instruction mix
#   SQ_INST_CYCLES_VMEM / SQ_WAIT_ANY      — memory stall cycles
#   MemUnitBusy                            — memory unit utilization
#   LDSBankConflict / SQC_LDS_BANK_CONFLICT — LDS bank conflicts
#   FETCH_SIZE / WRITE_SIZE                — HBM bandwidth (bytes)
#   L2CacheHit                             — L2 hit rate
#   GL2C_HIT / GL2C_MISS                   — L2 hit/miss counts
#   GPUBusy / GRBM_GUI_ACTIVE              — overall GPU utilization
#   SQ_WAVES                               — total wavefronts dispatched
#   ALUStalledByLDS                        — VALU stalls waiting on LDS
# Split into passes (hardware limit: ~8 counters per pass on gfx1100).
# Pass 1: occupancy + utilization
PASS1="OccupancyPercent,MeanOccupancyPerCU,GPUBusy,MemUnitBusy,SQ_WAVES"
# Pass 2: instruction mix
PASS2="VALUInsts,SALUInsts,ALUStalledByLDS,LDSBankConflict"
# Pass 3: memory bandwidth + cache
PASS3="FETCH_SIZE,WRITE_SIZE,L2CacheHit"
# Pass 4: stall analysis
PASS4="SQ_INST_CYCLES_VMEM,SQ_WAIT_ANY"

echo "Profiling: $MODEL → $OUTDIR"

# Note: PERSISTENT=0 required — cooperative persistent kernel crashes rocprofv3 at process exit.
# Hardware counters collected on the paged (non-persistent) decode path, which has per-layer
# kernel dispatches. Timing benchmarks should still use PERSISTENT=1.
for pass in 1 2 3 4; do
    eval "COUNTERS=\$PASS$pass"
    echo "  pass $pass: $COUNTERS"
    PERSISTENT=0 MODEL="$MODEL" BENCH_WARMUP=1 BENCH_RUNS=2 \
        python3 scripts/launch-gpu.py --timeout 300 -- \
        bash -c "rocprofv3 --pmc $COUNTERS -f csv -d '$OUTDIR/pass$pass' -o counters -- target/release/braid_bench" \
        2>&1 | tail -5
done

echo "Counter data: $OUTDIR/"
