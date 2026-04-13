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
| Killing processes that hold GPU VRAM | Other sessions/services depend on those processes. NEVER kill. |
| Using `fuser` to find GPU process owners | Same as rocm-smi — don't probe GPU state manually |
| Asking the user to kill processes for GPU access | Just queue with launch-gpu.py and wait. GPUs free eventually. |

**If GPUs are busy**: Queue your command with `launch-gpu.py --timeout 43200` and WAIT. Do not investigate what's using them, do not ask to kill processes, do not probe VRAM. The script handles everything. Other users and services share these GPUs and their work is equally important.

**If the script doesn't support what you need**: STOP. Do not bypass. Tell the user.

**If you need in-process measurements** (e.g., VRAM after model load): Add reporting to the binary itself (e.g., print VRAM usage from within Rust), don't try to race external tools against the process.

## What Causes Hangs in Persistent Mode

**Root cause**: The persistent cooperative worker (`persistent_worker.hip`) is a cooperative kernel that holds ALL GPU CUs for its entire lifetime. Any HIP operation that requires launching a kernel or DMA transfer on the SAME GPU will deadlock waiting for free CUs.

### Causes hang (NEVER do while persistent worker is running):

| Operation | Why |
|---|---|
| `hipMemcpy` / `memcpy_d2h` / `memcpy_h2d` | DMA requires CUs to be free |
| Any `hipLaunchKernel` on GPU 0 | Needs CUs — all held by persistent worker |
| `hipMemset` on GPU 0 | Same |
| `hipDeviceSynchronize()` after non-persistent launch on GPU 0 | Deadlocks |
| `peer_copy_async` (GPU-to-GPU copy via kernel on GPU 0) | Needs GPU 0 CUs |

These operations are safe ONLY:
- During model initialization, BEFORE `persistent_worker` is launched
- AFTER the persistent worker has been shut down (never in practice during inference)
- On GPU 1-3 (they run kbk workers, not persistent cooperative kernels)

### Safe while persistent worker is running:

| Operation | Why safe |
|---|---|
| GPU-side `printf()` in kernel code | Runs within existing CUs |
| `grid.sync()` inside the cooperative kernel | Same kernel |
| Reading/writing to host-mapped buffers (`MappedHostBuffer`) | CPU MMIO, no GPU CUs needed |
| `write_volatile` / `read_volatile` on queue memory | CPU-side volatile memory access |
| `dispatch_batch` / `dispatch_batch_fire` (work queue writes) | CPU writes to host-mapped memory |
| P2P DMA from GPU 0 to GPU 1-3 (`peer_copy_async` on GPU 1-3) | Uses GPU 1-3 CUs, not GPU 0 |

### Debugging rule:

**GPU-side `printf()` only.** Add debug prints directly in `.hip` kernel files. Never add `memcpy_d2h`, `hipMemcpy`, or any HIP API call in code that runs during active inference (after worker launch).

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

## Mandatory Review Workflow

Reviews are always approved and expected. Do not ask the user for permission to review.
Run the appropriate review automatically at each checkpoint. Fix trivial items inline;
create a review epic with beads for larger items, expert-review the epic, then implement.

### Review Checkpoints (run these automatically)

| Checkpoint | When | Agent | Action |
|---|---|---|---|
| **Plan review** | After writing any plan in `~/.claude/plans/` | `expert-review` | FULL review. Block implementation until APPROVED. |
| **Post-implementation** | After completing an epic or multi-file change | `implementation-review` | Verify code changes, tests, build. Fix trivial issues inline. |
| **Post-sprint / periodic** | After 5+ commits or when user says "what's next" | `software-architect` | Full codebase review. Creates findings epic if >3 issues found. |
| **Code quality** | After any commit touching >100 lines | `/simplify` | Review changed code for reuse, quality, efficiency. Fix inline. |

### Activation Trace Requirement

**New GPU kernels or megakernel opcodes must pass trace comparison before commit.**

Generate a reference trace, apply the change, generate a test trace, compare:
```bash
# Reference (before change):
TRACE=ref.bin MODEL=qwen35_2b.q4.bqnt RAW=1 MAX_TOKENS=1 \
  python3 scripts/launch-gpu.py --timeout 300 -- target/release/generate "Hello"

# Test (after change):
TRACE=test.bin MODEL=qwen35_2b.q4.bqnt RAW=1 MAX_TOKENS=1 \
  python3 scripts/launch-gpu.py --timeout 300 -- target/release/generate "Hello"

# Compare:
python3 scripts/compare_traces.py ref.bin test.bin
```

For quantization changes, use the bisection tool:
```bash
python3 scripts/bisect_quant.py --ref ref.bin --model model.q4.bqnt \
  --num-layers 36 --prompt "Hello"
```

Per-layer quantization control: `WEIGHT_QUANT_LAYERS=0-11,20-31` restricts Q4 to
those layers (rest load bf16). Useful for isolating which layer diverges.

### Review Process

1. **Plan**: Write plan → `expert-review` (run_in_background=True) → wait for APPROVED
2. **Implement**: Code → commit → `implementation-review` (run_in_background=True)
3. **Simplify**: After implementation-review passes, run `/simplify` on changed files
4. **Periodic architect**: Every 5+ commits, launch `software-architect` to review full codebase
   - Fix trivial items (dead code, naming) inline
   - Create epic with beads for larger items (>30 min work each)
   - `expert-review` the epic
   - Implement the epic

### Review Agent Usage

```
# Plan review (before implementation):
Agent(subagent_type="expert-review", run_in_background=True,
  prompt="FULL REVIEW: epic=<id> plan=<path> project_root=...")

# Post-implementation (after commits):
Agent(subagent_type="implementation-review", run_in_background=True,
  prompt="Review implementation of <epic-id>. project_root=...")

# Periodic codebase review:
Agent(subagent_type="software-architect", run_in_background=True,
  prompt="Full architectural review of <project_root>. Skip <files being edited>...")

# Code quality on changed files:
/simplify
```

### Review Prompt Template

Always include in review prompts:
- "Known Planned Work — DO NOT flag" section with `bd list --status=open -n 50`
- "Recent fixes — verify, don't re-flag" section with recent commit summaries
- Explicit scope (which files to review, which to skip)

### File Size Policy

**model.rs > 1000 lines → split before adding more.** Any source file over 1000 lines
should be split into focused modules before new features are added. This prevents god
objects from accumulating.

### Review History
- 2026-03-25 round 1: 21 findings → epic braidinfer-ji9, 13 fixed, 8 deferred
- 2026-03-25 round 2: 3 P1 + 2 P2, APPROVED 6/6
- 2026-03-26 full codebase review: 15 findings → epic braidinfer-kvn, APPROVED 5/5
