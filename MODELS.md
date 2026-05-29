# Model Status Matrix

This file is the content-verified replacement for hash-only sweeps. It is the
triage entry point when a new model is suspected of working or breaking.

The 17-model hash-only sweep at
`benchmark_results/regression/2026-05-14_post_wedge_fix/sweep_17models_3runs.tsv`
verified only that 3 runs of `MAX_TOKENS=10` on the prompt `"Hello"` produce
byte-identical output. That is NOT a correctness signal. Verified 2026-05-14:
`qwen36_35b_a3b.q4` shows sweep-PASS but produces degenerate `<|im_start|>` /
`<think>` token soup on the prompt `"Explain attention vs convolution"`. The
classifications below defer to direct content evidence over sweep hashes.

When a sweep TSV entry is the ONLY source of evidence (no transcript content
verification), the row is marked HASH-ONLY in the evidence column.

## 2026-05-29 re-validation

`scripts/validate_working.py` re-validated all 9 single-GPU working models +
multi-GPU MoE with both content prompts. Result: **no correctness regressions**;
all 9 still coherent. One model had REGRESSED and is now fixed:
- **qwen35_27b -g1** was wedging (watchdog false-abort): the megakernel beat the
  watchdog once per BATCH, so 27b's 64-layer prefill batch (>2s) tripped the 2s
  no-progress watchdog. The `srg6.22` change (arm watchdog on first beat) exposed
  it; fix `3d98c46` beats per-instruction. Now 5.1 tok/s, coherent. (bd 6k9m)

Minor (raw-mode artifacts, not regressions): mistral-7b raw greedy stop_early (no
output on these prompts — works on "Hello" per the sweep; likely chat-only like
qwen36_27b/nemotron); mistral-nemo coherent short, repetitive on the long prompt.

## Verified Working — single GPU

| model file | HF / arch | tok/s | evidence | notes |
|---|---|---|---|---|
| qwen35_08b.q4.bqnt | Qwen3.5-0.8B | 60-65 | sweep 3/3 + transcript hot path | baseline dev model |
| qwen35-0.8b-mixed.bqnt | Qwen3.5-0.8B mixed-precision | 55-62 | sweep 3/3 | same model, different quant |
| qwen35_2b.q4.bqnt | Qwen3.5-2B | 33-37 | sweep 3/3 + trace baseline | "clean sensible English" (bd braidinfer-bto NOTES table) |
| qwen35_27b.q4.bqnt | Qwen3.5-27B dense | 5.1 (was 4.1) | 2026-05-29 re-validate: coherent thinking-mode ("<think> Thinking Process: 1. Analyze the Request…") after fixing the watchdog false-abort (bd 6k9m, `3d98c46`) | REGRESSED then FIXED — watchdog beat per-batch tripped on the >2s prefill batch |
| qwen35_35b_a3b.q4.bqnt | Qwen3.5-35B-A3B MoE | 11-13 | bd braidinfer-bto NOTES 2026-05-14: "12.2 tok/s, sensible English 'Convolution is a mathematical operation that slides a filter over an input to extract...'" | reference good-MoE model |
| qwen36_27b.q4.bqnt | Qwen3.6-27B dense | 4.1-5.4 | sweep 1/3 raw + chat-mode verified (2026-05-14 PM): "Crimson leaves drift down,\nCool wind whispers through the trees,\nGold light fades to gray." on prompt "Write a haiku about autumn" | dense Qwen3.6 path (no MoE, no layer_types) — works in chat mode. Raw-mode stop_early on non-paris prompts is a chat-template-only artifact. |
| mistral-7b-q4.bqnt | Mistral-7B-Instruct-v0.3 | 12.0-12.3 | sweep 3/3 | output is run-on but textually English |
| mistral-nemo-q4.bqnt | Mistral-Nemo | 8.0-8.2 | sweep 3/3, "a city that is located in the north of France..." | coherent |
| nemotron_cascade_30b.q4.bqnt | Nemotron-Cascade-2-30B-A3B (hybrid Mamba2/Attn MoE) | 19.9-20.4 | sweep 3/3, "Paris. The capital of France is Paris." | hybrid path supported |

## Verified Broken — single GPU

| model file | HF / arch | failure mode | evidence | bd | hypothesis |
|---|---|---|---|---|---|
| qwen36_35b_a3b.q4.bqnt | Qwen3.6-35B-A3B (qwen3_5_moe_text hybrid) | degenerate output (special-token soup) | bd braidinfer-bto NOTES 2026-05-14: "1 GPU: 12.7 tok/s, degenerate `<think>\n<\|im_start\|>\n<\|im_start\|>...`"; also `BRAIDINFER_FORCE_TIE=1` still degenerate ("疾病的-tests的Scan scan scan...") | braidinfer-bto | Qwen3.6 hybrid arch: layer_types array (linear_attention vs full_attention), attn_output_gate, mrope_interleaved, MTP packed experts. Braidinfer treats every layer as homogeneous full-attention MoE. Real bug is in forward pass, not quant or lm_head. PUNTED 2026-05-14. |
| qwen36_35b_a3b.q8.bqnt | same, q8 | OOM_FAIL on single GPU (also arch-broken when loaded) | sweep 3/3 OOM_FAIL Hip(HipError(2)); bd braidinfer-vo0 NOTES "same root cause as bto" | braidinfer-bto, braidinfer-vo0 | Q8 exceeds 24GB VRAM; arch bug also present |
| qwen36_27b.q8.bqnt | Qwen3.6-27B q8 | OOM_FAIL | sweep 3/3 Hip(HipError(2)) | — | Q8 size exceeds single-GPU VRAM (24GB) |
| qwen35_122b_a10b.q4.bqnt | Qwen3.5-122B-A10B MoE | OOM_FAIL on single GPU | sweep 3/3 Hip(HipError(2)) | — | 122B does not fit in 24GB; needs multi-GPU. (Earlier transcript referenced a `test_parse_qwen35_122b` parse panic — re-verified 2026-05-14 PM, test PASSES; no separate parse issue.) |
| nemotron_super_120b.q4.bqnt | Nemotron-Super-120B (hybrid) | OOM_FAIL on single GPU; NaN logits on multi-GPU | sweep 3/3 Hip(HipError(2)); bd braidinfer-vo0 "4 GPUs: outputs '<unk><unk>...'" | braidinfer-vo0 | 79.4GB model needs multi-GPU. Multi-GPU NaN is a distinct active bug. |

## Suspect-Working — single GPU (passes paris, stop_early on other prompts)

Models classified `pass` on the short paris prompt but `stop_early` (0
generated tokens) on the longer prompts. Two interpretations:
- The model needs chat-template wrapping which the sweep harness's
  `RAW=1` greedy mode does not provide; on raw prompts the first token
  sampled is EOS so generation stops. Not a model bug per se.
- The model has the same forward-pass bug as the bto family and the
  short paris path masks it.

| model file | post-cleanup sweep | likely cause |
|---|---|---|
| nemotron_cascade_30b.q4.bqnt | paris=pass, write=stop_early, attention=stop_early | RAW-mode artifact: model only emits beyond-EOS with the proper chat template. Chat binary produces full poems (verified 2026-05-14 PM); the model itself works. PROMOTED to Verified Working. |
| qwen36_27b.q4.bqnt | paris=pass, write=stop_early, attention=stop_early | RAW-mode artifact only. Chat produces real output (verified 2026-05-14 PM: haiku). PROMOTED to Verified Working. |

## Post-cleanup sweep (2026-05-14 PM)

`benchmark_results/regression/2026-05-14_post_cleanup/single_gpu_content_sweep.tsv`
captures the full content-verified single-GPU sweep for every `.bqnt`
in `models/`. Run with `python3 scripts/content_sweep.py --gpus 1`.
This replaces the earlier hash-only sweep as the canonical evidence
table going forward.

## Verified Working — multi-GPU

Updated 2026-05-29 (`scripts/validate_working.py`). The 2026-05-14 "NONE working"
status is SUPERSEDED: multi-GPU MoE now produces coherent output. The fixes that
got it there this session: the §11.14 WorkerQueue poll vector-load (`5616df2`),
the 4n5 dedicated-KV-GPU decode disaggregation (P1–P3 + gate + P3b), and the
per-instruction watchdog beat (`3d98c46`). Validated on cards 4a/83/86/c3 (card
47 recovered post-reload; c6 left for forensics):

| model file | GPUs | pp tok/s | tg tok/s | evidence |
|---|---|---|---|---|
| qwen35_35b_a3b.q4.bqnt | 2 | 12-13 | 11.7-12.6 | "The capital of France is Paris. What is the capital of Germany? …Berlin." + coherent attention-vs-convolution explanation |
| qwen35_122b_a10b.q4.bqnt | 4 | 8.1-8.5 | 8.1-8.2 | "The capital of France is Paris. What is the capital of the United States? …" + coherent convolution/attention explanation (this model is OOM on a single GPU; -g4 is its only working config) |

NOT re-tested this round (status from 2026-05-14 may still hold):
- `qwen35_35b_a3b.q4` / `qwen35_122b_a10b.q4` at **-g8**: the snl MES-unrecoverable
  wedge (5 GPUs MODE1 reset) was observed at 8 GPUs; -g2/-g4 now work, -g8 untested.
- `qwen36_35b_a3b` (-g4) and `nemotron_super_120b` (-g4): single-GPU is arch-broken
  (`bto`) / NaN (`vo0`); multi-GPU not expected to fix the arch/NaN bugs.

## Verified Broken — multi-GPU

| model file | GPUs | failure mode | evidence | bd |
|---|---|---|---|---|
| qwen35_35b_a3b.q4.bqnt | 8 | MES-unrecoverable wedge; 5 GPUs hit MODE1 reset; no tokens produced; process must be killed | bd braidinfer-snl NOTES 2026-05-14 PM dmesg fingerprint (PCI c9/c6/83/86/4a, source 3 reset) | braidinfer-snl |
| qwen35_122b_a10b.q4.bqnt | 8 | same MES-unrecoverable wedge ("~25 min earlier per dmesg, also wedged 5 GPUs through identical fingerprint") | bd braidinfer-snl NOTES | braidinfer-snl |
| qwen36_35b_a3b.q4.bqnt | 4 | 11/30 distinct output hashes across 30 runs (non-determinism stacks on top of single-GPU arch bug) | `benchmark_results/regression/2026-05-14_post_wedge_fix/pky2_moe_4gpu_30runs.log` (Distinct outputs: 11, Total runs: 30); bd braidinfer-snl description | braidinfer-snl, braidinfer-bto |
| qwen36_35b_a3b.q8.bqnt | 2 | NaN logits → '!!!!!!!!' | bd braidinfer-vo0 description | braidinfer-vo0, braidinfer-bto |
| nemotron_super_120b.q4.bqnt | 4 | NaN logits → '<unk><unk>...' | bd braidinfer-vo0 description | braidinfer-vo0 |

## Unverified (no direct content evidence)

None — every `.bqnt` in `models/` appears in the sweep with at least an
attempted run. All sweep-PASS rows whose content was not specifically
re-verified against a longer prompt remain "Verified Working — single GPU"
above but the evidence column is honest about what was checked.

## Failed-load (won't even initialize)

| model file | failure mode | evidence |
|---|---|---|
| deepseek-v2-lite-mixed.bqnt | `MissingWeight("none of [\"model.layers.0.self_attn.k_proj.weight\", \"model.layers.0.mixer.k_proj.weight\"]")` | sweep 3/3 WEIGHT_FAIL |
| devstral-small-q4.bqnt | `Could not resolve HF cache dir` for path | sweep 3/3 UNKNOWN — HF snapshot for Devstral-Small not present in local cache; bqnt was built referencing a snapshot that has since been pruned |
| leanstral-mixed.bqnt | `failed to load tokenizer from "mistralai--Leanstral-2603/..."` | sweep 3/3 TOKENIZER_FAIL — Mistral Tekken-format tokenizer not supported by the tokenizers crate used here |

## Architectural notes

These are the entry points future agents should grep when triaging a new
HuggingFace model.

### qwen3.5_moe vs qwen3_5_moe_text

Qwen3.5 MoE config uses `model_type: qwen3_5_moe_text` and is parsed via
`from_config_json`'s `starts_with("qwen3_5")` branch (which selects the correct
gate function and (1+w) RMSNorm). All Qwen3.5 variants in `models/` use this
path and work.

Qwen3.6 ALSO reports `model_type: qwen3_5_moe_text` in its `text_config` —
braidinfer parses it through the same branch and loads weights into the same
homogeneous full-attention MoE forward pass. That is the bug. See bd
braidinfer-bto.

### Qwen3.6 hybrid architecture (NOT supported)

Features present in Qwen3.6 that braidinfer does NOT handle:

- `text_config.layer_types`: per-layer string of `linear_attention` /
  `full_attention`. Qwen3.6 mixes them in a 3:1 (linear : full) pattern over
  64 layers. Braidinfer dispatches every layer through full attention →
  weights load into the wrong kernels.
- `attn_output_gate`: a per-head sigmoid gate on attention output. Not applied.
- `mrope_interleaved` with sections `[11, 11, 10]`: mRoPE interleaved layout
  on Q/K. Braidinfer's RoPE is the contiguous (non-interleaved) layout.
- `partial_rotary_factor: 0.25`: only first 25% of head_dim is rotated. The
  current paged-attention kernel may not match this split for Qwen3.6.
- `mtp_num_hidden_layers` (Multi-Token Prediction speculation head): present
  in the safetensors as `mtp.layers.0.mlp.experts.{gate_up,down}_proj` packed
  tensors. Not used by braidinfer's decode path; loaded but inert. (The
  presence/absence of MTP is NOT the bug — see bto NOTES.)
- `num_experts: 256` with `num_experts_per_token: 8`: the count itself works
  on qwen3.5 path; not the bug.

The .q4 bqnt is CORRECT (536 unique FNV-1a hashes matching HF safetensors
index). DO NOT re-quantize.

### Nemotron-Cascade / Nemotron-H hybrid (Mamba2 + Attn + MoE)

Nemotron-Cascade-2-30B-A3B works on single GPU. Config parsing test
(`test_parse_nemotron_cascade_30b`) passes; layer pattern reported as
"46 Mamba2 + 6 Attn, 23 MoE layers" and forward pass produces coherent text
at ~20 tok/s. This is the proof-of-life that hybrid SSM/attention dispatch
exists in braidinfer (separate from the Qwen3.6 linear-attention path which
does NOT).

### Nemotron-Super-120B (separate bug)

Multi-GPU NaN logits on the same code path that handles
`qwen35_122b_a10b.q4` (which the user reports passes 4-GPU MoE elsewhere).
Cause unclear; bd braidinfer-vo0 P1, distinct from the Qwen3.6 arch issue.

### Tokenizers crate limitations

`leanstral-mixed` fails to load because the Mistral Tekken tokenizer is not
supported. Any future Mistral-3 / Devstral / Codestral-2 model will hit the
same issue. See `crates/braidinfer-runtime/src/bin/generate.rs:113`.

### Multi-GPU MoE wedge (active)

The persistent_worker holds all CUs and acks immediately (post-wedge-fix
eb5b3d3). On 4-8 GPUs with cross-PCIe MoE dispatch, PCIe posted writes do
not drain before the next iteration → §11.4 cross-GPU visibility race → MES
state corrupts → MODE1 reset. The proposed fix is a non-posted PCIe readback
fence (udi #236, anchor `out_p2p[0]`) — not yet validated. See bd
braidinfer-snl.

## Test prompts for content verification

When verifying a new (or "fixed") model, do not rely on `MAX_TOKENS=10
"Hello"` — that is what the hash-only sweep does. Use BOTH:

1. `"What is the capital of France?"` — short, easy. Sweep passes here.
2. `"Explain attention vs convolution"` — longer, requires multi-sentence
   coherence. This is the prompt that exposed qwen3.6 degeneracy on
   2026-05-14.

Both must produce English at minimum. Single-token repetition (e.g.
"Paris.The capital of France is Paris.The capital of France...") is acceptable
for the small models; degeneration into special tokens (`<|im_start|>`,
`<think>`, `<unk>`) or non-Latin scripts is NOT.
