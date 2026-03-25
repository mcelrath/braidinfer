# BraidInfer

GPU-native LLM inference engine in Rust + HIP, targeting AMD RDNA3 (gfx1100, 7900XTX).

## GPU Binary Launch Requirements

**MANDATORY**: Always use `scripts/launch-gpu.py` to run any command that uses GPUs. This includes cargo tests, benchmarks, profiling, and any binary that touches HIP. No exceptions.

**Why**: Multiple users/sessions share the GPUs. Direct invocation skips GPU reservation (flock), risks VRAM conflicts, and leaves orphaned processes.

```bash
# Run a test:
python3 scripts/launch-gpu.py --timeout 300 -- \
  cargo test -p braidinfer-runtime --test megakernel_test -- --nocapture

# Run benchmarks:
python3 scripts/launch-gpu.py --timeout 600 -- \
  cargo test -p braidinfer-runtime --test bench_decode -- --nocapture

# Multiple GPUs:
python3 scripts/launch-gpu.py -g 2 -- cargo test ...

# Status/cleanup:
python3 scripts/launch-gpu.py --status
python3 scripts/launch-gpu.py --cleanup
```

**GPU waiting is automatic**: The script blocks until GPUs are free (polls every 5s, 1hr timeout). Do NOT manually check with `rocm-smi`. Set `Bash timeout=600000` to allow wait time.

**If the script doesn't support what you need**: STOP. Do not bypass. Tell the user.

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
