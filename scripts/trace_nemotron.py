#!/usr/bin/env python3
"""Generate per-layer activation traces from HuggingFace Nemotron-H model.

Runs on CPU in float32 for a single token, dumps per-layer hidden states
to a BTRC trace file for comparison with braidinfer output.

Usage:
  python3 scripts/trace_nemotron.py --model /path/to/nemotron --prompt "Hello" --output /tmp/nemotron_ref.btrc
"""

import argparse
import struct
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def write_btrc(path, checkpoints):
    with open(path, "wb") as f:
        f.write(b"BTRC")
        f.write(struct.pack("<I", 1))
        for name, data in checkpoints:
            name_bytes = name.encode()
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            flat = data.float().flatten().cpu().numpy()
            f.write(struct.pack("<I", len(flat)))
            f.write(flat.tobytes())
        f.write(struct.pack("<I", len(checkpoints)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--output", default="/tmp/nemotron_ref.btrc")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    checkpoints = []

    with torch.no_grad():
        backbone = model.backbone if hasattr(model, "backbone") else model.model
        embed = backbone.embeddings if hasattr(backbone, "embeddings") else backbone.embed_tokens
        hidden = embed(input_ids).squeeze(0)
        checkpoints.append(("embed", hidden[-1]))

        for i, layer in enumerate(backbone.layers):
            residual = hidden[-1:].unsqueeze(0)
            normed = layer.norm(residual)
            mixer_out = layer.mixer(normed)
            if isinstance(mixer_out, tuple):
                mixer_out = mixer_out[0]
            hidden_new = residual + mixer_out
            hidden = torch.cat([hidden[:-1], hidden_new.squeeze(0)], dim=0)
            checkpoints.append((f"L{i}.post", hidden[-1]))

        norm_f = backbone.norm_f if hasattr(backbone, "norm_f") else backbone.norm
        final = norm_f(hidden[-1:].unsqueeze(0))
        checkpoints.append(("final_norm", final.squeeze()))

    write_btrc(args.output, checkpoints)
    print(f"Wrote {len(checkpoints)} checkpoints to {args.output}")

if __name__ == "__main__":
    main()
