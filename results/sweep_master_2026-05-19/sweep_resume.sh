#!/bin/bash
set -u
N=${N:-5}
OUTROOT=/tmp/sweep_master
GS="$OUTROOT/global_summary.txt"
echo "--- RESUME from model 15 ---" | tee -a "$GS"
date | tee -a "$GS"

run_model() {
  local SLOT=$1 LABEL=$2 MODEL=$3 GPUS=$4
  local DIR="$OUTROOT/$LABEL"
  mkdir -p "$DIR"
  local S="$DIR/summary.txt"
  : > "$S"
  echo "--- model $SLOT: $LABEL (-g $GPUS) ---" | tee -a "$GS"
  date >> "$S"
  for i in $(seq 1 $N); do
    local LOG="$DIR/trial_${i}.log"
    sudo dmesg -C 2>/dev/null
    MODEL=$MODEL RAW=1 MAX_TOKENS=20 \
      python3 /home/mcelrath/Projects/ai/braidinfer/scripts/launch-gpu.py -g $GPUS --timeout 1800 -- \
      /home/mcelrath/Projects/ai/braidinfer/target/release/generate "The quick brown fox" \
      >"$LOG" 2>&1
    local rc=$?
    local warn_nan=$(grep -c "WARN: " "$LOG")
    local bang_tail=$(tail -5 "$LOG" | grep -c "!!!!!!")
    local hw_exc=$(grep -c "HW Exception" "$LOG")
    local warmup_line=$(grep -E "warmup-mailbox|warmup-discard" "$LOG" | head -1)
    local mes_errs=$(sudo dmesg 2>/dev/null | grep -ciE "MES failed to respond|MES might be in unrecoverable")
    local decode_ok=0
    if [ "$warn_nan" -eq 0 ] && [ "$bang_tail" -eq 0 ] && [ "$hw_exc" -eq 0 ] && [ "$rc" -eq 0 ]; then decode_ok=1; fi
    echo "T$i rc=$rc decode_ok=$decode_ok warn_nan=$warn_nan bang_tail=$bang_tail hw_exc=$hw_exc mes_errs=$mes_errs | $warmup_line" | tee -a "$S"
  done
  local ok=$(grep -c "decode_ok=1" "$S")
  local rc0=$(grep -c "rc=0 " "$S")
  local hwx=$(grep -c "hw_exc=[1-9]" "$S")
  local mestot=$(grep -oE "mes_errs=[0-9]+" "$S" | awk -F= '{s+=$2} END {print s+0}')
  echo "  $SLOT/$LABEL: decode_ok=$ok/$N rc=0=$rc0 hw_exc_trials=$hwx mes_errs_total=$mestot" | tee -a "$GS"
}

run_model 15 qwen36_35b_a3b-q8         /home/mcelrath/Projects/ai/braidinfer/models/qwen36_35b_a3b.q8.bqnt        4
run_model 16 deepseek-v2-lite-mixed    /home/mcelrath/Projects/ai/braidinfer/models/deepseek-v2-lite-mixed.bqnt   auto
run_model 17 leanstral-mixed           /home/mcelrath/Projects/ai/braidinfer/models/leanstral-mixed.bqnt          4

date | tee -a "$GS"
echo "=== RESUMED MODELS DONE ===" | tee -a "$GS"
