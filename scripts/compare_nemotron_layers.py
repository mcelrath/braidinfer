#!/usr/bin/env python3
"""Compare braidinfer BTRC trace vs HF NPZ hidden states per layer."""
import struct, sys, numpy as np

def read_btrc(path):
    entries = []
    with open(path, 'rb') as f:
        f.read(8)  # magic + version
        while True:
            h = f.read(4)
            if len(h) < 4: break
            nl = struct.unpack('<I', h)[0]
            nb = f.read(nl)
            if len(nb) < nl: break
            name = nb.decode()
            ne_b = f.read(4)
            if len(ne_b) < 4: break
            ne = struct.unpack('<I', ne_b)[0]
            db = f.read(ne * 4)
            if len(db) < ne * 4: break
            data = np.frombuffer(db, dtype=np.float32).copy()
            entries.append((name, data))
    return entries

btrc = read_btrc(sys.argv[1])
npz = np.load(sys.argv[2])

# Map braidinfer checkpoint names to HF layer indices
# braidinfer: embed, L0.post_mixer, L0.post_ffn, L1.post_mixer, ...
# HF: layer_0 (embed), layer_1 (after L0), layer_2 (after L1), ...
# Note: HF layer_{i+1} = after layer i. Token index 1 = Hello (after BOS).

print(f"BTRC: {len(btrc)} checkpoints")
print(f"NPZ: {len(npz.files)} layers")

for name, bi_data in btrc:
    if name == 'embed':
        hf_key = 'layer_0'
        hf_data = npz[hf_key][1] if npz[hf_key].ndim == 2 else npz[hf_key][0, 1]
    elif '.post_mixer' in name:
        layer_i = int(name.split('.')[0][1:])
        hf_key = f'layer_{layer_i + 1}'
        if hf_key not in npz:
            continue
        hf_data = npz[hf_key][1] if npz[hf_key].ndim == 2 else npz[hf_key][0, 1]
    else:
        continue

    if len(bi_data) != len(hf_data):
        print(f"{name} vs {hf_key}: SIZE MISMATCH {len(bi_data)} vs {len(hf_data)}")
        continue

    diff = np.abs(bi_data - hf_data)
    max_diff = diff.max()
    cos_sim = np.dot(bi_data, hf_data) / (np.linalg.norm(bi_data) * np.linalg.norm(hf_data) + 1e-8)
    bi_max = np.abs(bi_data).max()
    hf_max = np.abs(hf_data).max()

    status = "OK" if cos_sim > 0.99 else ("WARN" if cos_sim > 0.9 else "FAIL")
    print(f"{name:20s} vs {hf_key:10s}: cos={cos_sim:.4f} max_diff={max_diff:.4f} bi_max={bi_max:.2f} hf_max={hf_max:.2f} {status}")

    if status == "FAIL":
        # Print most different elements
        worst = np.argsort(-diff)[:5]
        for idx in worst:
            print(f"  [{idx}] bi={bi_data[idx]:.6f} hf={hf_data[idx]:.6f} diff={diff[idx]:.6f}")
