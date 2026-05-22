# BraidInfer

GPU-native LLM inference engine in Rust + HIP, targeting AMD RDNA3 (gfx1100, 7900XTX).

## System Software Deviates From Stock — Patch Manifest

This host runs **custom-patched** kernel and ROCm. Behaviors may differ from upstream documentation. **Check this list before debugging kernel/GPU coherence anomalies.**

> **Maintenance**: kernel/ROCm patch work happens in `../exterior_algebra/` (authoritative source for this section). When a patch is permanently added/removed/falsified there, this section MUST be re-mirrored from `../exterior_algebra/AGENTS.md` (or `CLAUDE.md` — same file via symlink) to keep agents aligned. Do not edit the manifest here directly without also updating the other two project CLAUDE.md files (`../exterior_algebra/` and `../llama.cpp/`).

### Kernel — `linux-p2p` package at `/home/mcelrath/builds/linux-p2p/`

Source tree at `src/linux-7.0.9/`. PKGBUILD lists active `source+=('NNNN-...patch')` entries. Quick map:

- **0001** MTYPE_UC for peer GPU VRAM on gfx11 (`amdgpu_amdkfd_gpuvm.c:kfd_mem_attach`) — workaround for missing `buffer_wbl2` on gfx11. Peer VRAM mapped uncached so writes bypass writer's L2.
- **0002** smu13_0_0 if_version bump to 0x40 — accept newer SMU firmware.
- **0003 / 0007 / 0009** PCIe error handling: coredump + bail, dedup PCI-error coredump, dedup coredump in ASIC reset.
- **0005 / 0008 / 0010** SMU mailbox: skip on channel failure, prefail markers, ratelimit bus errors.
- **0006** Skip `halt_activities` on non-hive frozen recovery to avoid NULL deref.
- **0012** HDP flush after CPU memset of `proc_ctx_bo` (`kfd_process_queue_manager.c:386`) — closes gfx11 multi-GPU coherence gap.
- **0013** HDP flush after CPU memset of `gang_ctx_bo` (same file, per-queue) — companion to 0012.
- **0016** HDP flush on worker adev before MES ADD_QUEUE (`kfd_device_queue_manager.c:add_queue_mes`) — closes page-table-BO coherence gap, eliminates WALKER_ERROR:0x6 / MAPPING_ERROR:0x1 dmesg signature.
- **0017** PTE-update fence synchronous wait in MAP_MEMORY_TO_GPU (`amdgpu_amdkfd_gpuvm.c:2119`, change `unreserve_bo_and_vms(&ctx, false, false)` → `(&ctx, true, false)`) — closes PERMISSION_FAULTS:0x3 race.
- **0018** HDP flush after debug-trap proc_ctx memset (`kfd_debug.c:372`) — mirror of 0012 for debug-trap path.

Reverted/falsified (DO NOT REAPPLY): `0014.REVERTED`, `0015.FALSIFIED`, older `0016-...stb-log....FALSIFIED`, `0019-...peer-adevs....FALSIFIED_DEADLOCK`. Sit in the same directory with `.FALSIFIED*` suffix.

Module is auto-loaded with `amdgpu.mes_log_enable=1` (modprobe.d): exposes per-card MES Scheduler Log at `/sys/kernel/debug/dri/<bdf>/amdgpu_mes_event_log` (intr_history only; api_history NOT exposed by this flag — see `bd show exterior_algebra-6gv.13`).

### HSA Runtime — `hsa-rocr` package at `/home/mcelrath/builds/hsa-rocr/`

- **`hsa-rocr-p2p-mtype-uc-gfx11.patch`** ACTIVE — forces peer VRAM mappings on multi-GPU gfx11 to `HSA_CACHING_NONCACHED`. Workaround for missing `buffer_wbl2` writer-side gap; without it, GPU compute writes to P2P-mapped peer VRAM never reach the remote GPU (stay in writer's L2). Touches `amd_memory_region.cpp:510`.
- **`hsa-rocr-coop-force-destroy-gfx11.patch`** OBSOLETE since 2026-05-12, NOT applied. Patch file retained for history with explanatory header. Removed because it introduced a `delete this` double-free.

### Implications for agent reasoning

- "GPU L2 doesn't flush to peer VRAM on gfx11" is a known issue worked around by 0001 (kernel) + hsa-rocr-p2p-mtype-uc (userspace). Don't rediscover.
- HDP-flush-on-owner-adev-only is the coherence-gap pattern 0012/0013/0016/0018 close. If you see new `WALKER_ERROR` / `MAPPING_ERROR` / `PERMISSION_FAULTS` signatures in dmesg for multi-GPU paths, suspect another instance of this class.
- `s_buffer_gl0_inv` / `s_buffer_gl1_inv` are **silently no-op on gfx11+** for host-mapped UC scalar loads (see braidinfer `GFX1100_ARCH.md §11.14`). Do not assume scalar-cache invalidation works.
- Production cold-start cure: userspace warmup-discard + mailbox-only default. The race is "MES μC private cache / memory-hub state, identified-but-unreachable from userspace" per bd memory `gfx1100-cold-start-race-bd-4e2m-final-state`. Kernel patches 0016/0017 close upstream-visible portions but do NOT fix the userspace symptom.
- `bd 6gv` epic (in exterior_algebra project) is the cumulative reinvestigation record. Search `bd memories 6gv` and `bd list --parent=exterior_algebra-6gv` for current status.
- `rocgdb` package is **`rocm-gdb`** in Arch repos (extra), NOT `rocgdb`.

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

# Kill your OWN stuck processes (session-scoped via CLAUDE_SESSION_ID):
python3 scripts/launch-gpu.py --kill
```

**If YOUR OWN job wedges** (kernel spin-loops, dispatcher deadlock, etc.) and is holding GPUs you need: kill it with `python3 scripts/launch-gpu.py --kill`. The script scopes the kill to this session's processes only — it WILL NOT touch other users or services. This is the only allowed way to kill a GPU-holding process; never `kill`, `kill -9`, or `pkill` directly.

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
| Killing OTHER sessions' processes (via `kill`, `kill -9`, `pkill`, etc.) | Other sessions/services depend on those processes. NEVER touch directly. To kill YOUR OWN stuck process, use `launch-gpu.py --kill` — it's session-scoped via `CLAUDE_SESSION_ID` and won't affect anyone else. |
| Using `fuser` to find GPU process owners | Same as rocm-smi — don't probe GPU state manually |
| Asking the user to kill processes for GPU access | If it's your own process, run `launch-gpu.py --kill`. Otherwise queue with launch-gpu.py and wait. |

**If GPUs are busy**: Queue your command with `launch-gpu.py --timeout 43200` and WAIT. Do not investigate what's using them, do not ask to kill processes, do not probe VRAM. The script handles everything. Other users and services share these GPUs and their work is equally important.

**If the script doesn't support what you need**: STOP. Do not bypass. Tell the user.

**If you need in-process measurements** (e.g., VRAM after model load): Add reporting to the binary itself (e.g., print VRAM usage from within Rust), don't try to race external tools against the process.

## What Causes Hangs in Persistent Mode

**Root cause**: The persistent cooperative worker (`persistent_worker` entry in `kernels/megakernel.hip`) is a cooperative kernel that holds ALL GPU CUs for its entire lifetime. Any HIP operation that requires launching a kernel or DMA transfer on the SAME GPU will deadlock waiting for free CUs.

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

`kernels/megakernel.hip` defines a single `__global__` entry point:

- `persistent_worker(queue, watchdog)` — persistent driver: polls a
  host-mapped `WorkerQueue` mailbox forever, acks each batch. Used by
  prefill and steady-state decode (always-persistent). Holds CUs for the
  model's lifetime.

Op routing lives in the `dispatch_opcode(...)` helper in
`kernels/megakernel_dispatch.hip` — the single routing table for every
opcode the megakernel handles. Adding a new opcode requires editing
exactly one file. Trace dump infrastructure is wired through
`WorkerQueue::dump_*` fields populated by the Rust runtime when
`Model::trace` is active.

`kernels/megakernel.hsaco` carries persistent_worker. The Rust runtime
loads it from `PersistentDispatch` via `Module::get_function("persistent_worker")`.

History: prior to the braidinfer-zqw merge, `persistent_worker` lived in
its own `kernels/persistent_worker.hip` with a parallel (and drifting)
opcode-dispatch chain. That drift caused the OP_LM_HEAD regression
(commit 9c21cf6: 250× slowdown on persistent decode). The merge
(commits b1cb2de…9deeb50) collapsed both chains into one helper. bd 9gmh
Phase 4 (2026-05-21) then deleted the megakernel_f32 one-shot entry
since all production paths route through the mailbox.

## Watchdog for Persistent Kernels

Persistent cooperative kernels (moe_worker, moe_gemv_worker, megakernel.hip's `persistent_worker` entry, megakernel_moe_dispatch) include a host-side watchdog that detects wedged kernels.

### How it works

1. Each kernel receives a `WatchdogState*` pointer (host-mapped memory, visible from CPU).
2. The kernel calls `watchdog_beat()` periodically to increment `progress_counter`.
3. A host thread (`WatchdogThread` in `crates/braidinfer-runtime/src/watchdog.rs`) polls `progress_counter` every `BRAIDINFER_WATCHDOG_POLL_MS` ms (default 100).
4. If no progress for `BRAIDINFER_WATCHDOG_NO_PROGRESS_MS` ms (default 2000): host writes `force_exit=1`.
5. Well-behaved kernels call `watchdog_poll_and_check()` and exit when `force_exit=1`. They have `BRAIDINFER_WATCHDOG_GRACE_MS` ms (default 1000) to exit.
6. If the kernel still hasn't advanced `progress_counter` after grace: host dumps telemetry and calls `std::process::abort()`.

### Recovery model (RDNA3-specific)

**Cooperative exit** (normal): kernel polls `force_exit`, exits via `grid.sync()` + return. Recovery time ~4.7ms (measured).

**Abort escalation** (stuck kernel): `hipDeviceReset` is NOT called — it blocks indefinitely on RDNA3/gfx1100 because ROCm has no GPU TDR preemption for compute kernels. Process abort triggers amdgpu driver context teardown and releases the GPU. Confirmed by `exterior_algebra/scripts/watchdog_recovery_test.hip`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAIDINFER_WATCHDOG_NO_PROGRESS_MS` | 2000 | ms with no progress_counter change before declaring stuck. Set to 0 to disable watchdog entirely. |
| `BRAIDINFER_WATCHDOG_GRACE_MS` | 1000 | ms after force_exit is written before escalating to abort. |
| `BRAIDINFER_WATCHDOG_POLL_MS` | 100 | Host thread poll interval. |

### CI gate

```bash
python scripts/check_watchdog_coverage.py
```

Detects HIP files containing `cg::this_grid()` or `while (true)` that do not `#include "watchdog.h"`. Exit 1 on missing coverage.

### Adding watchdog to a new kernel

```cpp
#include "watchdog.h"

__global__ void my_kernel(WatchdogState* watchdog, ...) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    uint32_t beat_ctr = 0;

    while (true) {
        // ... do work ...
        watchdog_beat(watchdog, &beat_ctr);
        if (watchdog_poll_and_check(watchdog, grid, op_id)) {
            watchdog_signal_exit(done_flag);
            return;
        }
        grid.sync();
    }
}
```

Pass `nullptr` for `watchdog` to disable (null-safe: all primitives check for null).

### Cooperative exit granularity

`watchdog_poll_and_check()` is called only between top-level instructions (at opcode dispatch boundaries), **NOT** inside compute-heavy ops (`op_moe_ffn`, `op_linear_proj_*`, `op_attn_paged`, etc.). A wedge inside a compute op escalates directly to abort (via the host watchdog grace period expiry) rather than cooperative exit.

This is acceptable for current ops since each is bounded in time. If any future op could exceed the no-progress timeout (default 2s), add intra-op `watchdog_beat()` calls. A full `watchdog_poll_and_check()` inside a compute op requires all blocks to arrive simultaneously (grid.sync() precondition), which may require significant restructuring.

### Recovery test reference

The cooperative watchdog recovery test lives in `../exterior_algebra/scripts/watchdog_recovery_test.hip`:
- Cooperative variant: 100/100 PASS, mean recovery 4.7 ± 0.7 ms
- Stubborn variant: documented as platform-limited — `hipDeviceReset` blocks indefinitely on RDNA3/gfx1100 (ROCm has no GPU TDR preemption for compute). Process abort is the correct escalation path.

## Never Trust grep/rg — Read Files In Full

`rg`/`grep` find string matches, not structural dependencies. Using grep as the audit mechanism for "what does this function reach" is a systematic Claude failure mode. When planning ANY migration, refactor, or deletion:

1. **Read the affected file(s) in full** — entire file, not just the matching lines.
2. **Read every upstream caller in full** — every function that calls into the affected code, recursively until you hit the public API boundary.
3. **Read every downstream callee in full** — every function the affected code calls, recursively until you hit `std`/HIP/external. Pay specific attention to indirect calls (kernel forwards, trait methods, helper modules).
4. Only AFTER full Read can you grep — and only to confirm the picture Read gave you, not to construct it.

The bd 9gmh Phase 2 regression (bd pywl) is the canonical failure: a `rg "mk.execute"` audit found 6 sites to migrate; a full Read of `prefill_mixed_chunk` would have revealed `moe_ffn_forward_prefill_batched` calling `self.kernels.*.forward` + `memcpy_d2h` — completely outside `mk.execute` but reachable from `Model::prefill`. Single-GPU smoke + test suite passed because they don't take that branch. The grep-based plan was self-consistent but structurally blind. Migration shipped; multi-GPU MoE prefill deadlocked on the first user run.

**Trigger this rule whenever**: planning a beads epic, writing a PLAN doc, dispatching an implementation agent, or making any structural call (delete a method, retire an env knob, migrate a code path). The cost of full Reads is real; the cost of a shipped regression is much higher.

## Pre-existing Issues — File First, Then Move On

When you encounter a failure, bug, or stale state that pre-dates your current task (test was already failing on HEAD, doc is out of date, dead code unrelated to your change, etc.): **first run `bd list --status=open` (grep for related keywords) to see if it's tracked. If not, `bd create` immediately with a precise repro/description.** Then continue your task. Do not silently fix or silently skip — the bead is the audit trail. Applies to: pre-existing test failures, latent bugs surfaced by your build, doc drift, orphan files, dead code outside your task's scope.

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

### Retired: PERSISTENT env var (bd 9gmh Phase 3, 2026-05-20)

`PERSISTENT=0` is a no-op. The persistent cooperative megakernel path is always
used. All configurations are validated:
- Single-GPU dense: 2.1x speedup confirmed (original validation).
- Single-GPU MoE: validated 2026-05-20 (qwen35_35b_a3b, -g 1, N=5, 5/5 PASS).
- KV_QUANT=1: validated 2026-05-20 via quantize_sealed_chunk_via_worker.
- WEIGHT_QUANT=rnf4/mixed: confirmed 2026-05-20.
- Multi-GPU: always required persistent, unchanged.

`Model::persistent` field deleted. `decode_step_paged` / `decode_step_paged_quantized`
retained as public methods for test coverage (kv_quant_e2e_test, persistent_paged_test).

### Review History
- 2026-03-25 round 1: 21 findings → epic braidinfer-ji9, 13 fixed, 8 deferred
- 2026-03-25 round 2: 3 P1 + 2 P2, APPROVED 6/6
- 2026-03-26 full codebase review: 15 findings → epic braidinfer-kvn, APPROVED 5/5
