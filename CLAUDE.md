# BraidInfer

GPU-native LLM inference engine in Rust + HIP, targeting AMD RDNA3 (gfx1100, 7900XTX).

**Why this project exists**: agentic coding fails when the agent's view of the codebase is incomplete or inconsistent — text-grep misses sibling helpers, IDE indexes go stale, partial reads ship regressions (bd pywl is the canonical example: a grep-planned migration shipped a multi-GPU MoE prefill deadlock because the relevant code used a different API than the one grepped for). BraidInfer is the inference substrate for an agent architecture that puts the entire codebase (or its API skeleton, expanded on demand via tool calls) in the KV cache.

The KV API contract is the one implemented in llama.cpp at `../llama.cpp/docs/kv-cache-api.md` (mask+append updates, always-consistent state, no re-index on file change). BraidInfer reimplements that API but goes farther:
1. **Lower-latency dispatch via megakernel** — single persistent cooperative kernel polls a host-mapped mailbox, eliminating per-step `hipModuleLaunchKernel` overhead and CU spin-up latency. llama.cpp dispatches per-op; we batch a whole step.
2. **Paged / tree / radix KV cache with vLLM-style memory management** — fixed-size chunk pool, per-sequence page tables, copy-on-write for branching. llama.cpp uses a flat per-sequence cache, which fragments and limits batching.
3. **Near-linear prefill scaling with GPU count** — head-parallel attention + expert-parallel MoE + GART-host-mapped P2P handoffs. llama.cpp's multi-GPU prefill is serial across devices and bottlenecks on H2H copies; the `bd 9gmh` epic + the pywl regression are both about getting this right.

The grep-blindness failure mode that motivates the "never trust grep" discipline in `~/.claude/CLAUDE.md` is exactly the class of failure this engine aims to make architecturally impossible.

## System Software Deviates From Stock — Patch Manifest

This host runs **custom-patched** kernel and ROCm. Behaviors may differ from upstream documentation. **Check this list before debugging kernel/GPU coherence anomalies.**

> **Maintenance**: kernel/ROCm patch work happens in `../exterior_algebra/` (authoritative source for this section). When a patch is permanently added/removed/falsified there, this section MUST be re-mirrored from `../exterior_algebra/AGENTS.md` (or `CLAUDE.md` — same file via symlink) to keep agents aligned. Do not edit the manifest here directly without also updating the other two project CLAUDE.md files (`../exterior_algebra/` and `../llama.cpp/`).

### GPU / PCIe Operations — `~/ash-pcie` Is the Primary Tool

For ANY of the following on this host, use `~/ash-pcie` (Python; run `~/ash-pcie --help`). If the operation you need isn't covered, **EDIT `~/ash-pcie` to add it** — do not write ad-hoc shell or one-off Python. Re-mirror this section to the other two project CLAUDE.md files after editing.

**Inspection** — `aer [report|clear|--raw]` · `watch [interval]` (live AER monitor with auto-baseline) · `info <c>` (full per-card: link, AER, hwmon, UID, PM) · `decode <c>` (LnkSta / SltSta / LnkCtl bit breakdown).

**PCIe link control** — `retrain <c>` · `gen1`…`gen5 <c>` (force Target Link Speed + retrain) · `gen-auto <c>` (restore LnkCap2 max).

**Recovery (escalating)** — `reset <c>` (driver-mediated remove + rescan) · `kick <c> [--sbr] [--really-i-know]` (sticky-bit clear + link disable cycle + retrain; `--sbr` adds Secondary Bus Reset). **REFUSED on bd-a3l-wedged cards** — use `amdgpu-bind <c>` instead (kick corrupts PSP on-die SRAM on bd-a3l class, validated 2026-05-28 on c0). `--really-i-know` bypasses the bd-a3l guard for non-bd-a3l link-down classes only.

**Driver binding** —
- `amdgpu-unbind <c>` / `amdgpu-bind <c>` — for am-rs and userspace amdgpu work. `amdgpu-unbind` refuses if a KFD/DRI holder is present and auto-kills `btop` / `nvtop` (always-safe user observers). NEVER use raw `tee /sys/bus/pci/drivers/amdgpu/unbind` — host hard-locks if a holder exists (2026-05-25 incident: dmesg `VM memory stats ... is non-zero when fini`).
- `release <c> [--force]` / `reclaim <c>` — for vfio-pci binding (hands the GPU to userspace via VFIO).
- `reload` — `rmmod amdgpu && modprobe amdgpu` then re-apply Gen3. Refuses on any wedged card or open handle.

**Power / hwmon** — `hwmon` (8-card summary) · `hwmon fan <c> <pct|auto>` · `hwmon fanall <pct|auto>` · `hwmon powercap <c> <watts>` · `wake [<c>|all]` (disable PCIe runtime PM so hwmon reads work) · `sleep [<c>|all]` (restore).

**Diagnostics** — `airflow-sweep` (intake-fan sweep with RPM + ΔT_jc per card) · `psu-diag` (voltages idle vs load + IPMI SEL deltas) · `identify [--once]` (DRM-connector watch to locate cards physically by DP plug).

`<c>` is GPU index 0-7 (matches `rocm-smi GPU[N]` and `btop`) or full BDF; `all` is accepted where it makes sense. Most ops auto-elevate via sudo. `--force-wedged` is required on `release` / `reclaim` / `reset` / `sleep` / `reload` for cards classified wedged.

**Related discipline**: CPU-only systemd services (`llama-embed-qwen3-8b`, `llama-qwen3:4b`) carry `InaccessiblePaths=/dev/kfd /dev/dri` (or `PrivateDevices=true`) so they never register as KFD holders. Never set `*_VISIBLE_DEVICES=-1` alone — it hides devices from selection but ROCm still opens KFD at startup.

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

## gfx11 Cross-GPU / L2 Coherence — Systematized Playbook

We have re-hit RDNA3/gfx1100 L2 coherence repeatedly (bd el1f, §11.20, 9gmh, yef5.2). The root
constraint: **gfx1100 has NO L2 invalidate** — `buffer_gl2_inv`/`buffer_wbl2` don't exist (rejected by
`llvm-mc --mcpu=gfx1100`; only `buffer_gl0_inv`=L0 and `buffer_gl1_inv`=L1 exist; gfx12 adds
`global_inv scope:SYS`). **A consumer GPU cannot flush its own L2.** Everything below follows.
Full detail: `GFX1100_ARCH.md §5.x` (cache) + `§11.x` (multi-GPU).

### Decision table — pick the medium/primitive by access pattern

| Pattern | USE | NOT |
|---|---|---|
| CPU↔GPU mailbox/signal (one GPU's view) | host-UC `MappedHostBuffer` (`alloc_portable_coherent`) | — |
| Cross-GPU SENTINEL (small, hot spin-wait) | host-UC + `sentinel_spin_load_u32` (vector `glc+dlc`, VGPR addr — `rdna3/rdna3_mailbox.h`) | plain or scalar load (latches stale K$, §11.14) |
| **Cross-GPU BULK DATA (a peer GPU reads what a producer GPU wrote)** | **peer-UC VRAM**: `DeviceBuffer` on the producer GPU, MTYPE_UC-mapped to peers via kernel patch 0001 (in-tree example: `MoeP2pContext::activation_staging_vram`) | **host-UC `MappedHostBuffer`** — ASYMMETRIC-stale (see trap below) |
| Producer ordering before releasing a sentinel | payload write → `__threadfence()` (agent) → PCIe drain-probe readback (`volatile x = dst[0]`) → RELEASE-store the sentinel | SYSTEM-scope store/fence (wedges, §11.4 vscnt-drain) |
| Consumer staleness WITHIN one GPU (L0/L1, cross-CU) | `buffer_gl0_inv`+`buffer_gl1_inv` BEFORE the load, then `s_waitcnt vmcnt(0)` | invalidate-AFTER-load (re-reads stale every iteration) |

### The asymmetric-host-UC trap (§11.19(x)) — the one we keep re-hitting

`alloc_portable_coherent` / host-mapped UC resolves to **write-back + CPU-snoop (MTYPE 0x40000001)**,
NOT true device-UC — and **there is no GPU→GPU snoop path over PCIe**. So a NON-allocator GPU reading
the allocator GPU's host-UC buffer sees **stale** data *even after the sentinel passes and after
CPU/allocator reads succeed*, and with no gl2_inv there is no consumer-side cure. It's silent: a
bounded ~1/5 wrong-data race, and **MoE/routing amplifies it** (a slightly-stale activation → a
different top-k expert → large output divergence; attention averages it out and looks fine).
**RULE: any buffer a PEER GPU bulk-reads must live in peer-UC VRAM (patch-0001), never host-UC.**
(Documented at `moe_p2p.rs:166-172`; this was the yef5.2 Step-A divergence — fixed by pointing the
decode handoff at `activation_staging_vram` instead of the host-UC `moe_act_uc_handoff`.)

### Falsified / forbidden — do NOT propose these as L2 fixes

- `buffer_gl2_inv` / `buffer_wbl2` — don't exist on gfx1100.
- `s_buffer_gl0_inv`/`s_buffer_gl1_inv` (scalar K$) — silently no-op for host-UC scalar loads (§11.14). Escape: force a vector `glc+dlc` load with a VGPR address.
- `glc+dlc` to "bypass L2" — they only bypass L0/L1, NOT L2; can still read a stale L2 line (am-rs-dev, bridge #4483). Fine for a small hot sentinel re-fetch; insufficient for bulk data.
- L2-eviction by working-set pressure (large strided dummy read) — FALSIFIED, fragile, set-mapping-dependent (`l2_evict_bench_v2`: 64 MiB scratch = 11× L2, 0% fresh). Never rely on it for correctness.
- `__threadfence_system()` / SYSTEM-scope atomics on a live spin-wait — HANG (no L2 sys-invalidate to back them; `rdna3/rdna3_barrier.h:102-113`).
- Host-mediated SDMA invalidation (`hipMemcpyAsync` 4-byte poke) — real flush, but a ~1-2µs CPU round-trip: launch-boundary only, useless inside a persistent kernel.

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
  both prefill and decode. Holds CUs for the model's lifetime.

`persistent_worker` calls `dispatch_opcode(...)` in
`kernels/megakernel_dispatch.hip` — the unified routing table for every
opcode the megakernel handles. Adding a new opcode requires editing
exactly one file. Trace dump infrastructure is wired through
`WorkerQueue::dump_*` fields populated by the Rust runtime when
`Model::trace` is active.

`kernels/megakernel.hsaco` exposes `persistent_worker` (loaded by
`PersistentDispatch` via `Module::get_function`).

History:
- The one-shot `megakernel_f32` entry was deleted in bd 9gmh Phase 4
  (commit 5953598) once persistent dispatch covered both prefill and
  decode.
- Prior to the braidinfer-zqw merge, `persistent_worker` lived in
  its own `kernels/persistent_worker.hip` with a parallel (drifting)
  opcode-dispatch chain. That drift caused the OP_LM_HEAD regression
  (commit 9c21cf6: 250× slowdown on persistent decode). The merge
  (commits b1cb2de…9deeb50) collapsed both chains into one helper.

## Watchdog for Persistent Kernels

Persistent cooperative kernels (moe_worker, moe_gemv_worker, megakernel.hip's `persistent_worker`) include a host-side watchdog that detects wedged kernels.

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
