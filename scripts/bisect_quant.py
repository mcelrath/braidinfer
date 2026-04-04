#!/usr/bin/env python3
"""Binary search to find which layer's quantization causes output divergence.

Usage:
    # Generate bf16 reference trace:
    TRACE=ref.bin cargo run --release -p braidinfer-runtime --bin generate -- "Hello"

    # Bisect which layer breaks with Q4:
    python3 scripts/bisect_quant.py --ref ref.bin --model qwen35_2b.q4.bqnt \
        --num-layers 36 --prompt "Hello" --tolerance 0.01

The script binary-searches WEIGHT_QUANT_LAYERS ranges, running inference with
each range and comparing traces against the reference.
"""
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from compare_traces import read_trace


def run_inference(model, prompt, quant_layers, trace_path, max_tokens=1):
    env = os.environ.copy()
    env["MODEL"] = model
    env["RAW"] = "1"
    env["MAX_TOKENS"] = str(max_tokens)
    env["TRACE"] = trace_path
    if quant_layers is not None:
        env["WEIGHT_QUANT_LAYERS"] = quant_layers
    cmd = [
        "python3", "scripts/launch-gpu.py", "--timeout", "300", "--",
        "cargo", "run", "--release", "-p", "braidinfer-runtime",
        "--bin", "generate", "--", prompt,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {result.stderr[-200:]}", file=sys.stderr)
        return False
    return True


def compare(ref_path, test_path, tolerance):
    ref_cps = read_trace(ref_path)
    test_cps = read_trace(test_path)
    ref_dict = {name: data for name, data in ref_cps}
    divergent = []
    for name, test_data in test_cps:
        if name not in ref_dict:
            continue
        ref_data = ref_dict[name]
        if len(ref_data) != len(test_data):
            divergent.append((name, float("inf")))
            continue
        max_diff = max(abs(a - b) for a, b in zip(ref_data, test_data))
        if max_diff > tolerance:
            divergent.append((name, max_diff))
    return divergent


def main():
    parser = argparse.ArgumentParser(description="Bisect quantization layer divergence")
    parser.add_argument("--ref", required=True, help="Reference trace file (bf16)")
    parser.add_argument("--model", required=True, help="Q4 bqnt model path")
    parser.add_argument("--num-layers", type=int, required=True, help="Total number of layers")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Max abs diff tolerance")
    parser.add_argument("--max-tokens", type=int, default=1, help="Tokens to generate")
    args = parser.parse_args()

    n = args.num_layers
    print(f"Bisecting {n} layers, tolerance={args.tolerance}")

    # First: test all-quantized to confirm it diverges
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        test_path = f.name
    try:
        print(f"\nAll layers quantized (0-{n-1}):")
        if not run_inference(args.model, args.prompt, f"0-{n-1}", test_path, args.max_tokens):
            print("  Inference failed, aborting")
            return
        divs = compare(args.ref, test_path, args.tolerance)
        if not divs:
            print("  No divergence — Q4 matches bf16 within tolerance!")
            return
        print(f"  {len(divs)} divergent checkpoints, first: {divs[0][0]} (diff={divs[0][1]:.6f})")

        # Binary search: find the first layer that causes divergence
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            range_str = f"0-{mid}"
            print(f"\nTesting WEIGHT_QUANT_LAYERS={range_str} (layers 0..{mid}):")
            if not run_inference(args.model, args.prompt, range_str, test_path, args.max_tokens):
                print("  Inference failed, assuming divergent")
                hi = mid
                continue
            divs = compare(args.ref, test_path, args.tolerance)
            if divs:
                print(f"  DIVERGED at {divs[0][0]} (diff={divs[0][1]:.6f})")
                hi = mid
            else:
                print(f"  OK (within tolerance)")
                lo = mid + 1

        print(f"\n{'='*60}")
        print(f"First divergent layer: {lo}")
        print(f"WEIGHT_QUANT_LAYERS=0-{lo-1} is clean, layer {lo} causes divergence")
        print(f"{'='*60}")

    finally:
        if os.path.exists(test_path):
            os.unlink(test_path)


if __name__ == "__main__":
    main()
