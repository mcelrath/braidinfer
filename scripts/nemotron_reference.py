#!/usr/bin/env python3
"""
Nemotron-Cascade-2 reference forward pass — no mamba_ssm/causal_conv1d needed.

HF transformers has native NemotronHForCausalLM with pure-torch Mamba2 fallback.
Block the CUDA-only imports to force the fallback path.

Usage:
    python3 scripts/nemotron_reference.py [--dump-hidden] [--max-tokens 32]

Outputs per-layer hidden states for comparison with braidinfer megakernel.
"""
import sys
import argparse

# Block CUDA-only Mamba/DeltaNet dependencies — forces pure-torch fallback
for _m in ['causal_conv1d', 'causal_conv1d.causal_conv1d_varlen_fn',
           'mamba_ssm', 'mamba_ssm.ops', 'mamba_ssm.ops.selective_scan_interface',
           'fla', 'fla.ops', 'fla.ops.gated_delta_rule']:
    sys.modules[_m] = None

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nvidia/Nemotron-Cascade-2-30B-A3B')
parser.add_argument('--prompt', default='The eigenvalues of the Hamiltonian are')
parser.add_argument('--max-tokens', type=int, default=32)
parser.add_argument('--dump-hidden', action='store_true')
parser.add_argument('--output', default=None)
parser.add_argument('--device', default='auto')
args = parser.parse_args()

max_mem = {i: '18GiB' for i in range(4)}
max_mem['cpu'] = '150GiB'

print(f"Loading {args.model}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16,
    device_map=args.device if args.device != 'auto' else 'auto',
    max_memory=max_mem if args.device == 'auto' else None,
    low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

first_dev = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else 'cuda:0'
dev = f'cuda:{first_dev}' if isinstance(first_dev, int) else str(first_dev)

ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(dev)
print(f"Input: {ids.shape[1]} tokens on {dev}", flush=True)

with torch.no_grad():
    out = model(ids, output_hidden_states=args.dump_hidden)

logits = out.logits[0, -1].float()
top5 = logits.topk(5)
print(f"\nTop-5 predictions:")
for i in range(5):
    tok = tokenizer.decode([top5.indices[i]])
    print(f"  {top5.values[i].item():.2f}: {repr(tok)}")

if args.dump_hidden and out.hidden_states:
    output_path = args.output or 'nemotron_hidden_states.npz'
    hidden = {}
    for i, h in enumerate(out.hidden_states):
        hidden[f'layer_{i}'] = h[0].float().cpu().numpy()
    np.savez(output_path, **hidden)
    print(f"\nDumped {len(hidden)} hidden states to {output_path}")

# Generate tokens
if args.max_tokens > 0:
    gen = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id)
    generated = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated: {generated}")

    if args.output:
        token_ids = gen[0].cpu().tolist()
        with open(args.output.replace('.npz', '_tokens.json'), 'w') as f:
            json.dump({'prompt': args.prompt, 'token_ids': token_ids,
                        'generated': generated}, f, indent=2)
