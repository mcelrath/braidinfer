#!/usr/bin/env python3
"""Scoped, fail-fast validation of the known-working model set + multi-GPU MoE.

Unlike the blind sweep_all_models.py (which marched over failed-load/broken
arches with a 600s budget each and wedged cards), this:
  - tests only the MODELS.md "Verified Working" set + the multi-GPU MoE configs
  - avoids degraded cards via BRAIDINFER_AVOID_GPUS (card 47 bad, c6 = forensics)
  - uses per-model process timeouts (fail-fast; the per-instruction watchdog
    aborts real megakernel hangs in ~3s)
  - reports pp (prefill tok/s), tg (decode tok/s), coherence, and a snippet
  - uses chat mode for chat-only models (RAW stops early on them per MODELS.md)
"""
import os, subprocess, time, json, sys
from pathlib import Path
from collections import Counter

REPO = Path("/home/mcelrath/Projects/ai/braidinfer")
LAUNCH = REPO / "scripts/launch-gpu.py"
GEN = ["target/release/generate"]
AVOID = "c6:00.0"  # 47 recovered after amdgpu reload (2026-05-29); leave c6 for mes-researcher forensics
SHORT = "What is the capital of France?"
LONG = "Explain attention vs convolution"

# (file, mode raw|chat, gpus, proc_timeout_s)
MATRIX = [
    ("qwen35_08b.q4.bqnt",            "raw",  1, 120),
    ("qwen35-0.8b-mixed.bqnt",        "raw",  1, 120),
    ("qwen35_2b.q4.bqnt",             "raw",  1, 120),
    ("qwen35_27b.q4.bqnt",            "raw",  1, 220),
    ("qwen35_35b_a3b.q4.bqnt",        "raw",  1, 220),
    ("qwen36_27b.q4.bqnt",            "chat", 1, 220),
    ("mistral-7b-q4.bqnt",            "raw",  1, 150),
    ("mistral-nemo-q4.bqnt",          "raw",  1, 150),
    ("nemotron_cascade_30b.q4.bqnt",  "chat", 1, 220),
    # multi-GPU MoE
    ("qwen35_35b_a3b.q4.bqnt",        "raw",  2, 280),
    ("qwen35_122b_a10b.q4.bqnt",      "raw",  4, 480),
]

def extract_pptg(text):
    for line in reversed(text.splitlines()):
        if line.startswith("PPTG ") or " PPTG " in line:
            try:
                seg = line[line.index("PPTG "):]
                return float(seg.split("pp=")[1].split()[0]), float(seg.split("tg=")[1].split()[0])
            except Exception:
                return None
    return None

def snippet(stdout, limit=240):
    lines = [l for l in stdout.replace("\r", "").split("\n") if l and not l.startswith("[braidinfer]")]
    return " ".join(lines).strip()[:limit]

def coherence(snip, status):
    if status != "OK":
        return "n/a"
    s = snip.strip()
    if not s:
        return "EMPTY(stop_early?)"
    if "NaN" in s:
        return "NaN"
    soup = ("<|im_start|>", "<|im_end|>", "<unk>", "<think>\n<")
    if sum(s.count(t) for t in soup) >= 3:
        return "TOKEN_SOUP"
    # non-Latin heavy (qwen3.6 degeneracy signature)
    nonlatin = sum(1 for c in s if ord(c) > 0x2000 and not c.isspace())
    if nonlatin > len(s) // 3:
        return "NON_LATIN"
    toks = s.split()
    if len(toks) >= 8 and len(set(toks)) <= 2:
        return "REPEAT"
    return "OK?"

def status_of(text, rc):
    if "NaN in logits" in text: return "NaN"
    if rc == 124 or "*** harness-timeout ***" in text: return "TIMEOUT"
    if "Memory access fault" in text: return "MEMFAULT"
    if "MissingWeight" in text: return "MISSING_WEIGHT"
    if "Could not resolve HF" in text: return "NO_HF_DIR"
    if "tokenizer" in text.lower() and "failed to load" in text.lower(): return "NO_TOKENIZER"
    if "OutOfMemory" in text or "HipError(2)" in text: return "OOM"
    if rc != 0: return f"RC={rc}"
    return "OK"

def run(model, mode, g, ptimeout, prompt, max_tok):
    env = os.environ.copy()
    env["MODEL"] = str(REPO / "models" / model)
    env["MAX_TOKENS"] = str(max_tok)
    env["BRAIDINFER_AVOID_GPUS"] = AVOID
    if mode == "raw":
        env["RAW"] = "1"
    else:
        env.pop("RAW", None)
    cmd = ["python3", str(LAUNCH), "-g", str(g), "--timeout", str(ptimeout),
           "--gpu-timeout", "300", "--", *GEN, prompt]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True,
                           text=True, timeout=ptimeout + 90)
        out, err, rc = p.stdout or "", p.stderr or "", p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = ((e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + "\n*** harness-timeout ***"
        rc = 124
    wall = round(time.time() - t0, 1)
    combined = out + "\n" + err
    st = status_of(combined, rc)
    snip = snippet(out)
    pptg = extract_pptg(combined)
    return {
        "model": model, "mode": mode, "g": g, "prompt": "short" if prompt == SHORT else "long",
        "status": st, "wall_s": wall,
        "pp": round(pptg[0], 1) if pptg else None,
        "tg": round(pptg[1], 1) if pptg else None,
        "coherence": coherence(snip, st), "snippet": snip,
    }

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_log = REPO / "scripts/validate_results.jsonl"
    out_log.unlink(missing_ok=True)
    results = []
    for model, mode, g, pt in MATRIX:
        if only and only not in model and only != f"g{g}":
            continue
        tag = f"{model} ({mode}, g={g})"
        print(f"\n=== {tag} ===", flush=True)
        for prompt, mt in [(SHORT, 24), (LONG, 40)]:
            r = run(model, mode, g, pt, prompt, mt)
            results.append(r)
            with out_log.open("a") as f:
                f.write(json.dumps(r) + "\n")
            pp = f"{r['pp']}" if r['pp'] is not None else "-"
            tg = f"{r['tg']}" if r['tg'] is not None else "-"
            print(f"  {r['prompt']:5} status={r['status']:13} wall={r['wall_s']:6}s "
                  f"pp={pp:>6} tg={tg:>6} coh={r['coherence']:16} out={r['snippet']!r}", flush=True)
    print("\n=== SUMMARY (pp=prefill tok/s, tg=decode tok/s) ===")
    print(f"{'model':32}{'mode':5}{'g':>2} {'prompt':6}{'status':13}{'pp':>7}{'tg':>7} coherent")
    for r in results:
        pp = f"{r['pp']}" if r['pp'] is not None else "-"
        tg = f"{r['tg']}" if r['tg'] is not None else "-"
        print(f"{r['model']:32}{r['mode']:5}{r['g']:>2} {r['prompt']:6}{r['status']:13}{pp:>7}{tg:>7} {r['coherence']}")

if __name__ == "__main__":
    main()
