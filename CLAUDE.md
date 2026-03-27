# BraidInfer

GPU-native LLM inference engine in Rust + HIP, targeting AMD RDNA3 (gfx1100, 7900XTX).

## GPU Binary Launch Requirements

**MANDATORY**: Always use `scripts/launch-gpu.py` to run any command that uses GPUs. This includes cargo tests, benchmarks, profiling, and any binary that touches HIP. **No exceptions. No workarounds.**

**Why**: Multiple users/sessions share the GPUs. Direct invocation skips GPU reservation (flock), risks VRAM conflicts, and leaves orphaned processes.

### How to use it

The launch script wraps your command. Everything after `--` is the command to run:

```bash
# Run a test:
python3 scripts/launch-gpu.py --timeout 300 -- \
  cargo test -p braidinfer-runtime --test megakernel_test -- --nocapture

# Run a binary with env vars (env vars go BEFORE python3):
MODEL=/path/to/model RAW=1 MAX_TOKENS=20 \
  python3 scripts/launch-gpu.py --timeout 120 -- \
  cargo run --release -p braidinfer-runtime --bin generate -- "Hello"

# Run benchmarks (longer timeout):
python3 scripts/launch-gpu.py --timeout 600 -- \
  cargo test -p braidinfer-runtime --test bench_decode -- --nocapture

# Multiple GPUs:
python3 scripts/launch-gpu.py -g 2 -- cargo test ...

# Status/cleanup:
python3 scripts/launch-gpu.py --status
python3 scripts/launch-gpu.py --cleanup
```

**GPU waiting is automatic**: The script blocks until GPUs are free (polls every 5s). GPUs may be busy for hours — that is expected and correct. Never reduce timeouts to work around busy GPUs.

### Always launch as background Bash with run_in_background

GPUs may not be available immediately. **Always** launch GPU commands as background tasks so you get notified when they complete, rather than blocking your context window:

```bash
# CORRECT: background task, long timeout, notified on completion
python3 scripts/launch-gpu.py --timeout 43200 -- \
  cargo run --release -p braidinfer-runtime --bin generate -- "Hello"
```
Use `Bash run_in_background=true` and `timeout=600000`. The launch script waits for a free GPU (up to 12 hours with `--timeout 43200`), runs the command, then the background task completes and you're notified.

**Do NOT** set short timeouts hoping GPUs free up soon. Do NOT poll or check GPU status. Do NOT try alternative GPUs. The script handles all of this.

### PROHIBITED patterns

| Anti-pattern | Why it's wrong |
|---|---|
| `HIP_VISIBLE_DEVICES=0 cargo run ...` | Bypasses reservation, risks VRAM conflict |
| `bash -c 'cargo run ... & PID=$!; rocm-smi'` | Subshell bypasses reservation |
| `cargo test ... && rocm-smi` | Direct GPU access without reservation |
| Running rocm-smi to check GPU state | launch-gpu.py handles this; manual checks race |
| Short timeouts (`--timeout 60`) when GPUs are busy | GPUs may be busy for hours; use `--timeout 43200` |
| Checking VRAM to see if GPUs are "almost free" | The script polls automatically; don't second-guess it |
| Trying different `HIP_VISIBLE_DEVICES` values | The script selects the best GPU; manual selection conflicts |

**If the script doesn't support what you need**: STOP. Do not bypass. Tell the user.

**If you need in-process measurements** (e.g., VRAM after model load): Add reporting to the binary itself (e.g., print VRAM usage from within Rust), don't try to race external tools against the process.

## Build

```bash
cargo build -p braidinfer-runtime  # builds HIP kernels via build.rs
```

Kernels are in `kernels/*.hip`, compiled by `crates/braidinfer-hip/build.rs` with hipcc.

## Architecture

- `crates/braidinfer-hip/` — Low-level HIP bindings (module loading, kernel launch, memory)
- `crates/braidinfer-core/` — Shared types (DeviceId, etc.)
- `crates/braidinfer-runtime/` — Model loading, kernel dispatch, megakernel
- `kernels/` — HIP kernel source files (.hip)

### Model: Qwen3.5-0.8B
- 24 layers: 18 GDN + 6 full attention (3:1 pattern)
- hidden=1024, vocab=248320
- GDN: 16 heads × 128 key/value dim, conv_dim=6144, kernel_size=4
- Attention: 8 q_heads × 256 head_dim, 2 kv_heads, rope_dim=64
- RMSNorm uses (1+weight) pattern (weights initialized at zeros)
- Weight tying: lm_head shares embed_tokens.weight

### Megakernel
Single persistent cooperative kernel (384 blocks × 256 threads) replacing ~345 individual launches.
128-byte instructions, virtual block loop, grid.sync() between instructions.
2.1x speedup over naive dispatch (6.4ms vs 13.4ms per token).

## Expert Code Review Process

Run full codebase reviews using the `expert-review` agent with a 6-reviewer panel:
Tri Dao (kernels/SSM), Horace He (roofline/fusion), Woosuk Kwon (KV cache/serving),
Jon Gjengset (Rust safety), Software Architect (modularity), Claude (anti-patterns).

**Key technique**: Include a "Known Planned Work — DO NOT flag" section in the review
prompt listing all tracked-but-unimplemented features with their beads issue IDs. This
prevents reviewers from re-flagging intentional incompleteness (no batching, no quantization,
no multi-GPU, etc.) as findings. Also list recent fixes so they verify correctness rather
than re-report.

```
bd list --status=open -n 50   # gather all planned work
# Include in review prompt as numbered list with issue IDs
# Include recent fixes as "verify these, don't re-flag" section
```

Review history:
- 2026-03-25 round 1: 21 findings → epic braidinfer-ji9, 13 fixed, 8 deferred to braidinfer-9ip/l4d
- 2026-03-25 round 2: 3 P1 + 2 P2 (much cleaner), APPROVED 6/6
