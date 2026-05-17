#!/usr/bin/env bash
# wuf.9 — Qwen3.6 forward-pass divergence: compare HF reference vs braidinfer.
#
# Hypothesis: braidinfer MRoPE uses adjacent-pair rotation
#   (kernels/megakernel_ops.hip:1700, i0=2*pair, i1=2*pair+1)
# while HF Qwen3 uses half-split rotation
#   (rotate_half: i0=pair, i1=pair+rope_dim/2).
#
# This script runs both traces and diffs them with compare_traces.py.
# First diverging checkpoint name + the probe.L0.q_post_rope_{half,adj} blocks
# tell you whether the MRoPE pairing is the culprit.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPT="${PROMPT:-Hello world short test}"
HF_OUT="${HF_OUT:-/tmp/hf_ref.bin}"
BRAID_OUT="${BRAID_OUT:-/tmp/braid.bin}"
MODEL_BQNT="${MODEL_BQNT:-models/qwen36_35b_a3b.q4.bqnt}"
TOL="${TOL:-0.05}"

echo "==> HF reference trace -> $HF_OUT"
python3 scripts/trace_hf_qwen36.py \
    --out "$HF_OUT" \
    --prompt "$PROMPT" \
    --probe-rope

echo "==> Braidinfer trace -> $BRAID_OUT"
TRACE="$BRAID_OUT" RAW=1 MODEL="$MODEL_BQNT" MAX_TOKENS=1 \
  python3 scripts/launch-gpu.py --timeout 43200 -- \
    target/release/generate "$PROMPT"

echo "==> Compare (tol=$TOL)"
python3 scripts/compare_traces.py "$HF_OUT" "$BRAID_OUT" --tolerance "$TOL" || true

echo
echo "First diverging checkpoint above = layer where bug wuf.9 manifests."
echo "If divergence first appears at L0.attn_out (but L0.q_proj/k_proj match),"
echo "and probe.L0.q_post_rope_half matches braidinfer's L0.k_post_mrope while"
echo "probe.L0.q_post_rope_adj does NOT, the MRoPE pairing hypothesis is confirmed."
