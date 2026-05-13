#!/usr/bin/env python3
"""Aggregate JSON-lines output from p2p_wedge_repro_runner.sh into matrix JSON.

Input: one JSON object per line from braidinfer's host_runner. Recognized events:
  {"event": "config", "hw_detect": "gfx1100", "rocm_detect": "7.2.2", ...}
  {"variant": "V0", "trial": N, "wedged": bool,
   "wedge_signature": {seq, ack, progress_pc, block_alive_count, gpu_id, elapsed_ms},
   "completed_dispatches": int}
  {"event": "launch_failed", "variant": "...", "trial": N}

Output: results/wedge_repro_matrix.json
  {"context": {...startup config...},
   "context_mismatch": bool,
   "variants": [{"name": "V0..V4", "n_trials": int, "wedged_count": int,
                 "wedge_rate": float, "median_completion_dispatches": int,
                 "wedge_signatures": [...]}],
   "pass_fail_matrix": {"V0": "PASS|FAIL: ...", ...},
   "joint_validation": "PASS|FAIL"}
"""
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def detect_rocm_version():
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if os.path.exists(path):
            return Path(path).read_text().strip()
    try:
        out = subprocess.check_output(["hipcc", "--version"], stderr=subprocess.STDOUT, text=True)
        for line in out.splitlines():
            if "HIP version" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "unknown"


EXPECTED_HW = "gfx1100"
EXPECTED_ROCM_PREFIX = "7.2"

PASS_RULES = {
    # V0/V5 originally proposed as wedge targets, empirically informative-only
    # (minimal patterns do not reproduce at 4 GPUs; finding documented).
    "V0": ("informative", None),
    "V1": ("pass", 1.0),
    "V2": ("pass", 1.0),
    "V3": ("pass", 1.0),
    "V4": ("informative", None),
    "V5": ("informative", None),
    "V6": ("informative", None),
    # V7 (cross-GPU coop_copy peer-VRAM + post-poll barrier) is the
    # confirmed minimal reproducer; intermittent ~30% rate matches production.
    "V7": ("wedge", 0.2),
    "V8": ("informative", None),
}


def main(raw_path, out_path):
    by_variant = defaultdict(list)
    context = None
    context_mismatch = False
    launch_failures = defaultdict(int)

    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "config":
                # Only hw_detect + rocm_detect are load-bearing for cross-trial
                # comparability. commit field is advisory (reflects CWD git state
                # at invocation, not necessarily binary build state).
                stable_fields = {k: obj.get(k) for k in ("hw_detect", "rocm_detect")}
                if context is None:
                    context = obj
                else:
                    prev_stable = {k: context.get(k) for k in ("hw_detect", "rocm_detect")}
                    if stable_fields != prev_stable:
                        context_mismatch = True
            elif obj.get("event") == "launch_failed":
                launch_failures[obj["variant"]] += 1
            elif "variant" in obj:
                by_variant[obj["variant"]].append(obj)

    if context:
        hw_ok = context.get("hw_detect") == EXPECTED_HW
        rocm_ok = str(context.get("rocm_detect", "")).startswith(EXPECTED_ROCM_PREFIX)
        if not (hw_ok and rocm_ok):
            context_mismatch = True

    variants_out = []
    pass_fail = {}
    overall_pass = True
    for variant, rule in PASS_RULES.items():
        trials = by_variant.get(variant, [])
        n = len(trials)
        wedged_self = sum(1 for t in trials if t.get("wedged"))
        # Canonical metric (braidinfer@eedc254+): seq_completed=False means the
        # dispatched seq was never acked end-to-end. Falls back to
        # completed_dispatches==0 for older binaries that don't emit the field.
        def _is_wedge(t):
            if "seq_completed" in t:
                return not t["seq_completed"]
            return t.get("completed_dispatches", 1) == 0
        wedged_inferred = sum(1 for t in trials if _is_wedge(t))
        wedged = max(wedged_self, wedged_inferred)
        rate = wedged / n if n else 0.0
        completions = [t.get("completed_dispatches", 0) for t in trials]
        median_completions = int(statistics.median(completions)) if completions else 0
        wedge_sigs = [t["wedge_signature"] for t in trials
                      if (t.get("wedged") or _is_wedge(t))
                      and "wedge_signature" in t]

        variants_out.append({
            "name": variant,
            "n_trials": n,
            "launch_failures": launch_failures.get(variant, 0),
            "wedged_count": wedged,
            "wedged_self_reported": wedged_self,
            "wedged_inferred_from_completion": wedged_inferred,
            "wedge_rate": round(rate, 3),
            "median_completion_dispatches": median_completions,
            "wedge_signatures_sample": wedge_sigs[:3],
        })

        kind, threshold = rule
        if kind == "wedge":
            ok = rate >= threshold and n > 0
            pass_fail[variant] = f"{'PASS' if ok else 'FAIL'}: wedge_rate={rate:.2f} (target ≥{threshold:.2f})"
            if not ok:
                overall_pass = False
        elif kind == "pass":
            ok = wedged == 0 and n > 0
            pass_fail[variant] = f"{'PASS' if ok else 'FAIL'}: wedged={wedged}/{n} (target 0)"
            if not ok:
                overall_pass = False
        elif kind == "informative":
            pass_fail[variant] = f"INFORMATIVE: wedge_rate={rate:.2f} wedged={wedged}/{n}"

    out = {
        "context": context or {"missing": True},
        "context_mismatch": context_mismatch,
        "expected_hw": EXPECTED_HW,
        "expected_rocm_prefix": EXPECTED_ROCM_PREFIX,
        "detected_rocm_via_aggregator": detect_rocm_version(),
        "variants": variants_out,
        "pass_fail_matrix": pass_fail,
        "joint_validation": "PASS" if overall_pass and not context_mismatch else "FAIL",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <raw.jsonl> <out.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
