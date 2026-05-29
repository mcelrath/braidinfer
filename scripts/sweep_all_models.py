#!/usr/bin/env python3
import os, subprocess, time, json, sys
from pathlib import Path

REPO = Path("/home/mcelrath/Projects/ai/braidinfer")
MODELS_DIR = REPO / "models"
LAUNCH = REPO / "scripts/launch-gpu.py"
GENERATE = ["cargo", "run", "--release", "-p", "braidinfer-runtime", "--bin", "generate", "--"]

SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "Transformers process token sequences in two phases: prefill processes the prompt "
    "in parallel by running attention over all tokens at once, while decoding emits "
    "one token at a time by attending over the cached key-value tensors. For long "
    "contexts the prefill phase dominates wall-clock time, while for streaming chat "
    "the decode phase dominates. Explain in one paragraph why"
)

PER_RUN_TIMEOUT_S = 600
SHORT_MAX_TOK = 10
LONG_MAX_TOK = 30

# Skip .old / non-bqnt artifacts.
def models():
    out = []
    for p in sorted(MODELS_DIR.glob("*.bqnt")):
        if p.name.endswith(".bqnt.old"):
            continue
        out.append(p)
    return out

def gpu_count(model_path: Path) -> int:
    size_gb = model_path.stat().st_size / 1e9
    # Single 7900 XTX = ~24 GB. >20 GB → multi-GPU.
    return 4 if size_gb > 20 else 1

def classify_output(text: str, returncode: int) -> str:
    if "NaN in logits" in text:
        return "NaN"
    if "TIMEOUT" in text or returncode == 124:
        return "TIMEOUT"
    if "Memory access fault" in text:
        return "MEMFAULT"
    if "MissingWeight" in text:
        return "MISSING_WEIGHT"
    if "no tokenizer.json" in text or "tokenizer not found" in text.lower():
        return "NO_TOKENIZER"
    if returncode != 0:
        return f"RC={returncode}"
    return "OK"

def extract_metric(text: str):
    # Last "N tokens in T.TTTs = X.X tok/s" line on stderr.
    for line in reversed(text.splitlines()):
        if "tok/s" in line and "tokens in" in line:
            try:
                # "10 tokens in 2.299s = 4.4 tok/s"
                parts = line.split()
                ntok = int(parts[0])
                secs = float(parts[3].rstrip("s"))
                tps = float(parts[-2])
                return ntok, secs, tps
            except Exception:
                return None
    return None

def extract_pptg(text: str):
    # Last "PPTG pp=X tok/s (...) tg=Y tok/s (...)" line (per generate.rs).
    for line in reversed(text.splitlines()):
        if line.startswith("PPTG ") or " PPTG " in line:
            try:
                seg = line[line.index("PPTG "):]
                pp = float(seg.split("pp=")[1].split()[0])
                tg = float(seg.split("tg=")[1].split()[0])
                return pp, tg
            except Exception:
                return None
    return None

def classify_coherence(snippet: str, status: str) -> str:
    # Heuristic only; the snippet is reported for human judgment.
    if status != "OK":
        return "n/a"
    s = snippet.strip()
    if not s:
        return "EMPTY"
    if "NaN" in s or "nan nan" in s.lower():
        return "NaN"
    toks = s.split()
    if len(toks) >= 6:
        # repeated single token collapse (argmax-of-NaN signature)
        from collections import Counter
        c = Counter(toks)
        if c.most_common(1)[0][1] >= max(6, len(toks) * 3 // 4):
            return "REPEAT"
        # repeated short phrase
        if len(set(toks)) <= 2:
            return "REPEAT"
    return "OK?"

def extract_output_snippet(stdout: str, limit: int = 200) -> str:
    # generate writes the produced text to stdout via print!/println!.
    # Strip trailing newlines and "[braidinfer] multi-GPU fast-exit ..." which
    # also goes to stdout per the binary's exit path.
    text = stdout.replace("\r", "")
    lines = [l for l in text.split("\n") if l and not l.startswith("[braidinfer]")]
    joined = " ".join(lines).strip()
    return joined[:limit]

def run_one(model: Path, prompt: str, max_tokens: int, g: int) -> dict:
    env = os.environ.copy()
    env["MODEL"] = str(model)
    env["RAW"] = "1"
    env["MAX_TOKENS"] = str(max_tokens)
    cmd = ["python3", str(LAUNCH), "-g", str(g), "--timeout", str(PER_RUN_TIMEOUT_S),
           "--", *GENERATE, prompt]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=PER_RUN_TIMEOUT_S + 60)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", "replace")
        stderr = (e.stderr or b"").decode("utf-8", "replace") + "\n*** harness-timeout ***"
        rc = 124
    elapsed = time.time() - t0
    combined = stdout + "\n" + stderr
    status = classify_output(combined, rc)
    metric = extract_metric(combined)
    snippet = extract_output_snippet(stdout)
    pptg = extract_pptg(combined)
    coherence = classify_coherence(snippet, status)
    return {
        "model": model.name,
        "prompt_kind": "short" if max_tokens == SHORT_MAX_TOK else "long",
        "g": g,
        "status": status,
        "wall_s": round(elapsed, 1),
        "metric": metric,
        "pp": round(pptg[0], 1) if pptg else None,
        "tg": round(pptg[1], 1) if pptg else None,
        "coherence": coherence,
        "snippet": snippet,
    }

def main():
    results = []
    out_log = REPO / "scripts/sweep_results.jsonl"
    out_log.unlink(missing_ok=True)
    for m in models():
        g = gpu_count(m)
        print(f"\n=== {m.name} (g={g}) ===", flush=True)
        for prompt, mt in [(SHORT_PROMPT, SHORT_MAX_TOK), (LONG_PROMPT, LONG_MAX_TOK)]:
            r = run_one(m, prompt, mt, g)
            results.append(r)
            with out_log.open("a") as f:
                f.write(json.dumps(r) + "\n")
            pp = f"{r['pp']:.1f}" if r['pp'] is not None else "-"
            tg = f"{r['tg']:.1f}" if r['tg'] is not None else "-"
            print(f"  {r['prompt_kind']:5} status={r['status']:14} "
                  f"wall={r['wall_s']:5}s pp={pp} tg={tg} coh={r['coherence']} "
                  f"out={r['snippet']!r}", flush=True)
    print("\n=== SUMMARY (pp = prefill tok/s, tg = decode tok/s) ===")
    print(f"{'model':40} {'kind':5} {'g':>2} {'status':12} {'pp':>8} {'tg':>8} {'coherent':9}")
    for r in results:
        pp = f"{r['pp']:.1f}" if r['pp'] is not None else "-"
        tg = f"{r['tg']:.1f}" if r['tg'] is not None else "-"
        print(f"{r['model']:40} {r['prompt_kind']:5} {r['g']:>2} "
              f"{r['status']:12} {pp:>8} {tg:>8} {r['coherence']:9}")

if __name__ == "__main__":
    main()
