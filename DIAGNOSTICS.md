# Diagnostic Environment Variables

This file lists the environment variables kept in braidinfer **for diagnostic
purposes**. These are tools for future agents and developers to investigate
behavior — they are **not** part of the user-facing configuration surface
(that lives in `ENV_CONFIG.md`, and will eventually move to proper CLI flags
and config files).

Every variable on this list has a clear reason to exist beyond "we tried it
once". Anything else — one-shot probes from concluded investigations — should
be deleted. The companion task `braidinfer-wuf.3` removes the retired ones.

When you add a new diagnostic env var: document it here, gate it with a clear
default-off behavior, and add a `bd remember` entry referencing this file so
the next agent can find it.

---

## Operational (watchdog timing)

The cooperative-kernel watchdog runs a host thread that polls each registered
kernel's `progress_counter` and force-exits if no progress is detected. These
control its sensitivity.

| Variable | Default | Effect |
|---|---|---|
| `BRAIDINFER_WATCHDOG_NO_PROGRESS_MS` | `2000` | ms with no `progress_counter` change before declaring the kernel stuck. Set to `0` to **disable the watchdog entirely** (only use for kernel-bringup debugging — production should keep it on). |
| `BRAIDINFER_WATCHDOG_GRACE_MS` | `1000` | ms after the host writes `force_exit=1` before escalating to `std::process::abort()` (which triggers amdgpu context teardown and releases the GPU on RDNA3). |
| `BRAIDINFER_WATCHDOG_POLL_MS` | `100` | Host thread poll interval. Smaller values increase host CPU and reduce latency-to-detect; larger values reduce false positives under heavy host load. |

---

## Active hypothesis flags (snl multi-GPU MoE wedge investigation)

Multi-GPU MoE wedges MES on 8 GPUs (and produces NaN/garbage at 4 GPUs).
`braidinfer-snl` tracks the investigation; udi #236 identifies the root cause
as a §11.4 cross-GPU visibility race exposed by the wedge-fix `eb5b3d3`
removing the implicit timing gap of Phase 2' deferred-ack.

| Variable | Effect |
|---|---|
| `BRAIDINFER_MOE_WORKER_READBACK_FENCE` | Worker performs a volatile non-posted PCIe read from `out_p2p[0]` (its own UC write target on GPU 0) before returning from `op_moe_ffn_remote`. The PCIe §2.4 producer-consumer rule forces same-requester same-target writes to drain before the read returns, so the subsequent `ack=seq` is only visible to the host after `output_slots` have landed in HBM. **This is the current candidate fix; do not retire until `snl` is closed.** |

---

## Kernel build-time tuning (perf features)

These are real feature flags, not probes. Set at build time via `cargo build`
with the env var set; the kernel is compiled with the corresponding `#define`.

| Variable | Effect |
|---|---|
| `BRAIDINFER_OP_PROFILE` | Compile `op_profile` counter accumulators into the megakernel. ~5% perf cost from atomic adds; intended for per-op-cycle diagnostics, not production. Used by `cargo run --bin op_profile_dump`. |
| `BRAIDINFER_USE_DOT2` | Wire `__builtin_amdgcn_fdot2_f32_bf16` into `op_linear_proj`. ~2.5× cyc/FMA at K≥1024 on gfx1100. Coherence validated 2026-03 across 12/12 models in `braidinfer-77r.2.14`. |
| `BRAIDINFER_KV_LOAD_AUX={1,2,4}` | Cache hint on KV-load buffer instructions in `op_attn_paged` (0x1=glc, 0x2=slc, 0x4=dlc). Unset = production default (plain `global_load`). Per-shape tuning aid. |
| `BRAIDINFER_BPSM` | Override the auto-detected `blocks_per_sm` for cooperative kernel launches. Useful for testing the impact of occupancy on a wedge or perf issue. |
| `BRAIDINFER_PERSISTENT_BLOCKS` | Override `num_blocks` for `persistent_worker` launches. Defaults to `blocks_per_sm × NUM_CUS`. |

---

## Dispatcher thread tuning

Affects the host-side dispatcher thread that fires kernel launches.

| Variable | Effect |
|---|---|
| `BRAIDINFER_DISPATCH_CPU` | Pin the dispatcher thread to a specific CPU (e.g. `5`). Used to keep it off the kernel-execution cores and reduce launch latency variance. |
| `BRAIDINFER_DISPATCH_PRIO` | Standard `nice` priority for the dispatcher thread (`-20..19`). Lower = higher priority. |
| `BRAIDINFER_DISPATCH_RT` | SCHED_FIFO real-time priority for the dispatcher thread (`1..99`). Requires `CAP_SYS_NICE`. Use with care; can starve other system threads. |

---

## Activation tracing

Used for layer-by-layer reference comparison (e.g. vs an HF transformers run).

| Variable | Effect |
|---|---|
| `TRACE=<path>` | Dump per-layer activations to `<path>`. Compare with `scripts/compare_traces.py`. Activation trace requirement is documented in `CLAUDE.md` for kernel changes. |
| `BRAIDINFER_LOGIT_TRACE` | Per-decode-step logits dump (separate from full activation trace; lighter weight). |

---

## Pointer attribute auditing

| Variable | Effect |
|---|---|
| `BRAIDINFER_MTYPE_AUDIT` | At allocation time, run `hipPointerGetAttributes` on every device buffer and log the memory type + flags. Catches MTYPE_UC vs MTYPE_NC vs cached mismatches that would otherwise surface as cross-GPU coherence bugs. Used heavily during the 2026-05 P2P investigations. **Preferred invocation is the `--audit-mtypes` CLI flag** added by braidinfer-wuf.16 (chat + generate); the env var is retained as a back-compat path and is automatically set when the flag is present. |

---

## Retired (do not re-add without an open investigation)

The following are or will be removed by `braidinfer-wuf.3`. Listed here so
agents reading old logs / kb entries know they're gone.

- `BRAIDINFER_WRITE_MFENCE`, `BRAIDINFER_WRITE_CLFLUSH`, `BRAIDINFER_WRITE_FOLLOWER` — host-side cache-flush probes; investigation ruled out host-cache class.
- `BRAIDINFER_POLL_ATOMIC_LOAD`, `BRAIDINFER_POLL_INJECT_COMPUTE`, `BRAIDINFER_POLL_LOAD_L2_BYPASS`, `BRAIDINFER_POLL_LOAD_WIDE` — mailbox-poll variants; wedge fix landed.
- `BRAIDINFER_BARRIER_V2`, `BRAIDINFER_BARRIER_V4`, `BRAIDINFER_BARRIER_ASM` — `atomic_block_barrier` A/B variants; canonical is chosen.
- `BRAIDINFER_QUEUE_LINE_ISOLATE` (+ `queue_line_isolate` rustc-cfg) — `seq_num` line-isolate probe; did not help.
- `BRAIDINFER_BF16_INPUT_PROBE` — precision-loss simulator from `77r.2.14`; that gate is now part of `BRAIDINFER_USE_DOT2`.
- `BRAIDINFER_DUMP_MOE_INPUT`, `BRAIDINFER_DUMP_MOE_POST`, `BRAIDINFER_DUMP_POLL_VALUE`, `BRAIDINFER_DUMP_PROGRAM` — one-shot kernel-side prints from concluded snl probes.
- `BRAIDINFER_MOE_POST_PREDELAY`, `BRAIDINFER_ACK_DRAIN_VSCNT`, `BRAIDINFER_MOE_WORKER_FENCE_SYSTEM`, `BRAIDINFER_MOE_WORKER_DRAIN_VSCNT` — failed snl coherence probes superseded by `BRAIDINFER_MOE_WORKER_READBACK_FENCE`.
- `BRAIDINFER_P0B_DIAG`, `BRAIDINFER_P0B_VERIFY_HOST_WRITE` — `per_batch_coop` diagnostics; that code path is deleted (`braidinfer-wuf.2`).
- `BRAIDINFER_K_TRACE_5AX` — K-tensor trace for the qwen35_35b_a3b MRoPE investigation (resolved per kb `5ax-k-trace-2026-05-06-result-commit`).
- `BRAIDINFER_FORCE_TIE` — qwen3.6 lm_head investigation probe added 2026-05-14 PM; ruled out and reverted by `braidinfer-wuf.1`.
- `BRAIDINFER_MOE_ZERO_SCRATCH` — class-(b) uninit-scratch discriminator probe; class ruled out same session.

---

## Conventions for adding a new diagnostic env var

1. **Name**: `BRAIDINFER_<AREA>_<PROBE>`. Use ALL_CAPS, underscore-separated.
2. **Default behavior**: must be safe / no-op when the env var is unset.
3. **Documentation**: add a row to this file with what it does and why it exists.
4. **Build-time gating**: if the probe requires kernel recompilation, add `println!("cargo:rerun-if-env-changed=...")` in `crates/braidinfer-hip/build.rs`.
5. **Retirement**: when the investigation concludes, **move the row to the Retired section** in the same commit that removes the code. Don't leave stub knobs behind.
6. **bd link**: reference the relevant `bd` issue ID in the description column so future agents can trace the rationale.
