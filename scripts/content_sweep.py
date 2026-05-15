#!/usr/bin/env python3
"""Content-verified model sweep.

Replaces the older hash-only sweep (benchmark_results/regression/
2026-05-14_post_wedge_fix/sweep_17models_3runs.tsv) which produced
false positives — a model that crashed into degenerate <|im_start|>
token-soup on real prompts could still produce a stable sha256 across
3 runs of a short prompt.

Verifies each (model, n_gpu) pair against three prompts (short factual,
short generative, longer reasoning). For each generation, captures:
- the actual generated text (not just hash)
- a quality classification per-prompt: pass | degenerate | nan | hang | load_fail | oom

Classification rules:
- "nan"        → output mentions "NaN in logits" or contains "!!!!!!"
- "degenerate" → >60% of generated tokens are a single special token
                 (<|im_start|>, <think>, <unk>) or whitespace; OR the
                 same token repeats ≥6 times in a row
- "load_fail"  → process exit before generation produced any tokens
                 due to MissingWeight / tokenizer error / etc.
- "oom"        → HipError(2) at load
- "hang"       → process did not return within --timeout
- "pass"       → none of the above; sensible-looking text

Aggregates across prompts: a model PASSes a config iff ALL prompts pass.

Usage:
  python3 scripts/content_sweep.py --gpus 1 --out /tmp/sweep.tsv
  python3 scripts/content_sweep.py --gpus 1,4 --models qwen35_2b.q4
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

PROMPTS = [
    ("paris",     "The capital of France is",                                        20),
    ("write",     "Write a single sentence about the ocean.",                        40),
    ("attention", "Explain the difference between attention and convolution in 3 sentences.", 100),
]


def classify(stdout: str, stderr: str, returncode: int) -> tuple[str, str]:
    """Returns (status, evidence). stdout = generated tokens (print!),
    stderr = all status / banner / NaN-warn lines (eprintln!)."""
    combined = stdout + stderr
    if "Hip(HipError(2)" in combined:
        return "oom", "HipError(2) at load"
    if "MissingWeight" in combined:
        m = re.search(r"MissingWeight\([^)]*\)", combined)
        return "load_fail", m.group(0) if m else "MissingWeight"
    if "failed to load tokenizer" in combined:
        return "load_fail", "tokenizer load failed"
    if "Could not resolve HF cache dir" in combined:
        return "load_fail", "HF cache miss"
    if "no chat_template found" in combined:
        return "load_fail", "no chat_template (base model)"
    if returncode == 124:
        return "hang", "timeout"
    if "NaN in logits" in combined or "!!!!!!" in combined:
        return "nan", "NaN logits"
    # generate.rs prints generated tokens to stdout via print!(); the
    # eprintln!("N tokens in ...") goes to stderr. Use stdout for text.
    text = stdout.strip()
    toks = re.search(r"(\d+) tokens in", stderr)
    if not toks:
        return "load_fail", "no tokens-in marker"
    n_tokens = int(toks.group(1))
    if n_tokens == 0:
        return "stop_early", "0 tokens generated (immediate EOS)"
    # If generation ran but stdout is effectively empty, the tokens were
    # all special / stripped → that is degenerate output, not a load
    # failure.
    if not text or len(text) < 4:
        return "degenerate", f"{n_tokens} tokens but stdout strips to {text!r}"
    # Repetition collapse #1: 6+ identical short tokens in a row, whitespace-separated
    if re.search(r"(\S{1,12})(\s+\1){5,}", text):
        return "degenerate", f"repetition collapse: {text[:80]!r}"
    # Low unique-token ratio across >=10 generated tokens — qwen3.6's
    # <|im_start|>-soup output has ~3 unique tokens across 100 generated.
    tokens = re.findall(r"<\|[^|]+\|>|\S+", text)
    if len(tokens) >= 10:
        uniq = len(set(tokens))
        if uniq / len(tokens) < 0.20:
            return "degenerate", f"low diversity ({uniq}/{len(tokens)} unique): {text[:80]!r}"
        special = sum(1 for t in tokens if t.startswith("<|") or t in ("<think>", "</think>", "<unk>", "<", ">"))
        if special / len(tokens) > 0.6:
            return "degenerate", f"special-token soup ({special}/{len(tokens)}): {text[:80]!r}"
    return "pass", text[:80]


def run_one(model: Path, n_gpu: int, prompt: str, max_tokens: int, timeout: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["MODEL"] = str(model)
    env["RAW"] = "1"
    env["MAX_TOKENS"] = str(max_tokens)
    cmd = [
        "python3", str(ROOT / "scripts" / "launch-gpu.py"),
        "-g", str(n_gpu),
        "--timeout", str(timeout),
        "--",
        str(ROOT / "target" / "release" / "generate"),
        prompt,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout + 30)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1", help="comma-separated GPU counts (e.g. 1,2,4)")
    ap.add_argument("--models", default=None, help="comma-separated model basenames (e.g. qwen35_2b.q4); default = all")
    ap.add_argument("--out", default=None, help="output TSV path")
    ap.add_argument("--timeout", type=int, default=600, help="per-run launcher timeout sec")
    args = ap.parse_args()

    gpu_counts = [int(x) for x in args.gpus.split(",")]
    if args.models:
        models = [MODELS_DIR / f"{m}.bqnt" for m in args.models.split(",")]
    else:
        models = sorted(MODELS_DIR.glob("*.bqnt"))

    out_path = Path(args.out) if args.out else None
    fout = open(out_path, "w") if out_path else sys.stdout
    fout.write("model\tn_gpu\tprompt\tstatus\tevidence\n")

    for model in models:
        name = model.name
        for n_gpu in gpu_counts:
            for prompt_id, prompt, max_tokens in PROMPTS:
                rc, sout, serr = run_one(model, n_gpu, prompt, max_tokens, args.timeout)
                status, evidence = classify(sout, serr, rc)
                fout.write(f"{name}\t{n_gpu}\t{prompt_id}\t{status}\t{evidence}\n")
                fout.flush()
                # If the model couldn't load at all, skip remaining prompts at this GPU count.
                # Per-prompt failures (degenerate, nan, hang, stop_early) keep going so the
                # full failure surface is visible.
                if status in ("load_fail", "oom"):
                    break

    if out_path:
        fout.close()
        print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
