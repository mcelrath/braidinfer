#!/usr/bin/env python3
"""
Dump HF reference activations for Qwen3-30B-A3B (project alias: qwen36_35b_a3b)
in BraidInfer BTRC trace format, for cross-checking braidinfer's forward pass.

Bug context: wuf.9 — Qwen3.6 forward-pass divergence.
Hypothesis: braidinfer's MRoPE uses adjacent-pair rotation
  (i0 = pair*2, i1 = pair*2 + 1, kernels/megakernel_ops.hip:1700)
while HF Qwen3 uses half-split rotation
  (i0 = pair, i1 = pair + rope_dim/2).

BTRC format (see crates/braidinfer-runtime/src/trace.rs):
  magic 'BTRC' | u32 version=1 |
    repeated { u32 name_len | name_bytes | u32 num_elements | f32 * n } |
  u32 count

Usage:
  python scripts/trace_hf_qwen36.py --out /tmp/hf_ref.bin --prompt "Hello world short test"
  python scripts/trace_hf_qwen36.py --out /tmp/hf_ref.bin --probe-rope
"""
import argparse
import struct
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_PROMPT = "Hello world short test"
NUM_TRACED_LAYERS = 4  # first 4 layers — wuf.9 expects divergence at layer 0 or 1


def write_btrc(path, checkpoints):
    """checkpoints: list[(name:str, data:1D float tensor or numpy)]"""
    with open(path, "wb") as f:
        f.write(b"BTRC")
        f.write(struct.pack("<I", 1))
        count = 0
        for name, data in checkpoints:
            t = data
            if isinstance(t, torch.Tensor):
                t = t.detach().to(torch.float32).contiguous().view(-1).cpu().numpy()
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", t.size))
            f.write(t.astype("<f4").tobytes())
            count += 1
        f.write(struct.pack("<I", count))
    print(f"wrote {count} checkpoints to {path}", file=sys.stderr)


class ActivationCapture:
    def __init__(self):
        self.acts = {}
        self.handles = []

    def grab(self, name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.acts[name] = t.detach()
        return hook

    def grab_input(self, name, idx=0):
        def hook(_mod, inp, _out):
            t = inp[idx]
            self.acts[name] = t.detach()
        return hook

    def register(self, mod, name, on="output", idx=0):
        if on == "output":
            h = mod.register_forward_hook(self.grab(name))
        else:
            h = mod.register_forward_pre_hook(
                lambda _m, _i, _name=name, _idx=idx:
                    self.acts.__setitem__(_name, _i[_idx].detach())
            )
        self.handles.append(h)

    def close(self):
        for h in self.handles:
            h.remove()


def describe_rope(model):
    """Print HF rotary embedding config relevant to wuf.9."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    fields = [
        "rope_theta", "rope_scaling", "partial_rotary_factor",
        "head_dim", "hidden_size", "num_attention_heads",
        "num_key_value_heads",
    ]
    print("# HF rope/attn config", file=sys.stderr)
    for f in fields:
        v = getattr(text_cfg, f, None)
        print(f"  {f} = {v}", file=sys.stderr)

    # Try to locate a rotary_emb module and report its inv_freq tensor.
    found = False
    for name, m in model.named_modules():
        if name.endswith("rotary_emb"):
            inv = getattr(m, "inv_freq", None)
            if inv is not None:
                print(f"  {name}.inv_freq.shape = {tuple(inv.shape)}, "
                      f"first8 = {inv[:8].tolist()}", file=sys.stderr)
                found = True
                break
    if not found:
        print("  (no rotary_emb module exposed at top level — "
              "HF Qwen3 builds rope inline per layer)", file=sys.stderr)

    # Note on rotation convention.
    print("# HF Qwen3 rotation convention (transformers/models/qwen3/modeling_qwen3.py):",
          file=sys.stderr)
    print("  apply_rotary_pos_emb() uses rotate_half():", file=sys.stderr)
    print("    x1 = x[..., : x.shape[-1] // 2]", file=sys.stderr)
    print("    x2 = x[..., x.shape[-1] // 2 :]", file=sys.stderr)
    print("    return torch.cat((-x2, x1), dim=-1)", file=sys.stderr)
    print("  => HALF-SPLIT pairing: dim i is rotated with dim i + rope_dim/2.",
          file=sys.stderr)
    print("  Braidinfer kernels/megakernel_ops.hip:1700 uses ADJACENT-PAIR "
          "(i0=2*pair, i1=2*pair+1). MISMATCH ⇒ wuf.9.",
          file=sys.stderr)


def probe_rope_pairing(model, tokenizer, device):
    """
    Capture q_proj output BEFORE rope and q AFTER rope for layer 0,
    so the caller can diff them and infer the pairing empirically.
    """
    cap = ActivationCapture()
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    cap.register(attn.q_proj, "probe.L0.q_pre_rope", on="output")

    # Intercept the q tensor going into apply_rotary_pos_emb by hooking the
    # self_attn forward and snapshotting q after q_norm if present, before rope.
    # Then snapshot q after the attention sub-module's q rotated path.
    # The cleanest portable hook: read q from attn forward's locals via a
    # custom wrapper. We instead capture the full self_attn output and rely
    # on offline kernel inspection. For *empirical* pairing detection we run
    # the rotation by hand below.
    ids = tokenizer(DEFAULT_PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model(input_ids=ids)
    cap.close()

    q_pre = cap.acts["probe.L0.q_pre_rope"][0, 0]  # [hidden -> heads*head_dim]
    cfg = getattr(model.config, "text_config", model.config)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    q_pre = q_pre.view(n_heads, head_dim)
    rope_dim = int(head_dim * getattr(cfg, "partial_rotary_factor", 1.0))

    # Build inv_freq the way HF Qwen3 does.
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, rope_dim, 2, device=q_pre.device).float() / rope_dim
    ))
    pos = torch.tensor([0.0], device=q_pre.device)
    freqs = torch.outer(pos, inv_freq)               # [1, rope_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)          # [1, rope_dim] half-split
    cos_hs, sin_hs = emb.cos(), emb.sin()

    # half-split rotation as HF does:
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rope_part = q_pre[..., :rope_dim].float()
    q_hs = q_rope_part * cos_hs + rotate_half(q_rope_part) * sin_hs

    # adjacent-pair (braidinfer) variant:
    q_ap = q_rope_part.clone()
    half = rope_dim // 2
    cos_ap = torch.cat([freqs, freqs], dim=-1)  # only first half used below
    # Build adjacent-pair cos/sin: dim 2k and 2k+1 share angle freqs[0,k]
    cos_per_pair = freqs[0]  # [rope_dim/2]
    sin_per_pair = freqs[0].sin() if False else freqs[0]  # placeholder
    cos_pair = freqs[0].cos()
    sin_pair = freqs[0].sin()
    even = q_rope_part[..., 0::2]
    odd  = q_rope_part[..., 1::2]
    rot_even = even * cos_pair - odd * sin_pair
    rot_odd  = even * sin_pair + odd * cos_pair
    q_ap = torch.empty_like(q_rope_part)
    q_ap[..., 0::2] = rot_even
    q_ap[..., 1::2] = rot_odd

    return [
        ("probe.L0.q_pre_rope",         q_pre[..., :rope_dim]),
        ("probe.L0.q_post_rope_half",   q_hs),
        ("probe.L0.q_post_rope_adj",    q_ap),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output BTRC path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--probe-rope", action="store_true",
                    help="Also dump q_pre_rope + both pairing variants for L0.")
    ap.add_argument("--num-layers", type=int, default=NUM_TRACED_LAYERS)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"# loading {args.model} dtype={args.dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    describe_rope(model)

    cap = ActivationCapture()
    base = model.model  # Qwen3MoeModel
    cap.register(base.embed_tokens, "embed")

    for i in range(min(args.num_layers, len(base.layers))):
        layer = base.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp
        cap.register(layer.input_layernorm, f"L{i}.input_norm")
        cap.register(attn.q_proj, f"L{i}.q_proj")
        cap.register(attn.k_proj, f"L{i}.k_proj")
        cap.register(attn.v_proj, f"L{i}.v_proj")
        cap.register(attn,        f"L{i}.attn_out")
        cap.register(layer.post_attention_layernorm, f"L{i}.post_attn_norm")
        cap.register(mlp,         f"L{i}.post_ffn")
        cap.register(layer,       f"L{i}.layer_out")

    cap.register(base.norm,     "final_norm")
    cap.register(model.lm_head, "lm_head")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"# prompt='{args.prompt}'  ids={ids[0].tolist()}", file=sys.stderr)
    with torch.no_grad():
        model(input_ids=ids)
    cap.close()

    # Flatten captured activations (use last token of seq for parity with
    # braidinfer's per-step trace; full seq also kept under .seq suffix for
    # debugging).
    ckpts = []
    for name, t in cap.acts.items():
        # t: [B, T, H] typically. Last-token slice + full slice.
        if t.dim() >= 2:
            last = t[..., -1, :] if t.dim() == 3 else t[-1, :]
            ckpts.append((name, last.reshape(-1)))
            ckpts.append((name + ".seq", t.reshape(-1)))
        else:
            ckpts.append((name, t.reshape(-1)))

    if args.probe_rope:
        ckpts.extend(probe_rope_pairing(model, tok, device))

    write_btrc(args.out, ckpts)


if __name__ == "__main__":
    main()
