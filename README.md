# braidinfer

A GPU-native LLM inference engine in Rust + HIP, designed for AMD RDNA3
(gfx1100, RX 7900 XTX) and optimized for batch-1 chat with long contexts.

Distinguishing characteristics:

- **Persistent cooperative megakernel**: a single HIP kernel holds all
  CUs on each GPU for the duration of inference. Host-mapped mailbox
  dispatch drives compute at ~3 µs round-trip per batch, eliminating
  per-op kernel-launch overhead.
- **Paged KV cache** with chunked allocation, designed for the
  "codebase as KV cache" workload — files moved to the end of KV on
  edit (tail-append) rather than mid-context replacement.
- **Multi-GPU MoE expert sharding** via head-parallel attention and
  per-worker expert distribution. Cross-GPU coherence handled through
  uncached host-mapped UC staging buffers
  (`composable_kernel/GFX1100_ARCH.md` §11.4 mitigation suite).
- **Custom HIP kernels** with WMMA tile shapes tuned for gfx1100
  wave32 + GDS. Build artifacts in `kernels/`.
- **Quantization**: bf16, PcG32Q4 (5 bpv near-lossless), Rnf4G128
  (8.25 bpv lossless-class).

Project status: fast-moving development. Multi-GPU MoE persistent
decode works end-to-end for the Qwen3.5-A3B family on the configurations
listed in `MODELS.md`. The §11.4 cross-GPU coherence class has documented
fixes in tree; architectural follow-ups (epics `t8fl`, `lr6t`, `wt1`)
make this class structurally preventable. See `.beads/` for live
issue tracking.

## Target hardware

- 8× AMD Radeon RX 7900 XTX (gfx1100, 24 GiB VRAM each)
- AMD EPYC 7532 or equivalent
- PCIe Gen3 (Gen4 desirable; not required)
- ROCm 7.2+

Single-GPU dev boxes work for small-model development; production
target is the 8-GPU configuration above. See `DEPLOYMENT.md` for
production host tuning (SCHED_FIFO, isolcpus, IRQ pinning, watchdog).

## Quickstart

    git clone <repo> && cd braidinfer
    cargo build --release

    # Single-GPU smoke test
    launch-gpu -g 1 -- \
        env MODEL=models/qwen35_2b.q4.bqnt RAW=1 MAX_TOKENS=20 \
            target/release/generate "Hello"

    # Multi-GPU MoE (auto-detected when model > single-GPU VRAM)
    launch-gpu -g 4 -- \
        env MODEL=models/qwen35_35b_a3b.q4.bqnt \
            target/release/generate "write a poem"

`launch-gpu` is the unified GPU reservation wrapper (race-free flock,
sets `HIP_VISIBLE_DEVICES` per session). Source: `scripts/launch-gpu.py`
or the `launch-gpu` repo.

## Supported models

See `MODELS.md` for the live model support matrix with per-model
tok/s evidence and known-issue classifications. Representative
single-GPU baselines on gfx1100 + RX 7900 XTX:

| Model | Quant | tok/s |
|---|---|---|
| Qwen3.5-0.8B | q4 | 60–65 |
| Qwen3.5-2B | q4 | 33–37 |
| Qwen3.5-27B (dense) | q4 | 4.1–4.2 |
| Qwen3.5-35B-A3B (MoE) | q4 | 11–13 |
| Mistral-7B-v0.3 | q4 | 12.0–12.3 |
| Nemotron-Cascade-2-30B-A3B (hybrid) | q4 | 19.9–20.4 |

Multi-GPU performance and large-model results: see the perf archive
linked from `MODELS.md`.

## Performance reference

Microbenchmark envelope (canonical constants in
`kernels/rdna3/rdna3_perf_envelope.h`, regenerated from
`exterior_algebra/results/*.json`):

| Metric | Value |
|---|---|
| HIP launch round-trip (cold) | 142 µs |
| HIP launch round-trip (warm) | 26 µs |
| Persistent megakernel dispatch | 2.79 µs (median) |
| Cross-GPU peer write floor | 1.2 µs |
| 1→8 GPU fan-out latency | 2.87 µs (median) |
| GPU → host signal latency | 310 ns (median, 670 ns p99) |
| SDMA peak throughput | 6.3 GB/s |
| Blit-kernel vs SDMA threshold | 1 MB |

Sustained under load: ≤1.01× p50 / ≤1.00× p99 degradation across
same-GPU SGEMM, peer-GPU SGEMM, peer-PCIe-heavy, and CPU-stress
scenarios.

## Architecture

Key entry points for code-reading:

- **Persistent cooperative megakernel**: `kernels/megakernel.hip`
  (HIP entry) + `crates/braidinfer-runtime/src/persistent_dispatch.rs`
  (Rust dispatch + mailbox)
- **Bytecode IR** (144-byte instructions, ~45 opcodes):
  `crates/braidinfer-runtime/src/megakernel/instructions.rs`
- **KV cache** (paged primary; legacy flat-KV and per-worker variants
  being unified by epic `pc3h`): `crates/braidinfer-runtime/src/paged_kv.rs`
- **Cross-GPU staging primitives**: `kernels/rdna3/rdna3_peer.h`
  (peer-UC store + deferred-write macro),
  `kernels/rdna3/rdna3_signal_host_mapped.h` (host-mapped UC mailbox),
  `kernels/rdna3/rdna3_signal_uc_device.h` (device UC for peer signal)
- **HIP kernel patterns + ruleset**:
  `composable_kernel/GFX1100_ARCH.md` — load-bearing reference for
  any kernel work on this hardware, especially the §5.5 ruleset
  (allocation methods for cross-agent buffers) and the §11.x
  empirical archive (cooperative-grid relaunch wedge, persistent
  worker ack protocol, UC-dst cached-read class).

## Planned features

In progress:

- **§11.4 mitigation suite** (cross-GPU UC writer slabs, deferred peer
  writes, host-mapped staging) — most surfaces landed; remaining
  asymmetric cases tracked via `r7dv` family bds.
- **wt1** — Write-through VRAM cache via SDMA stream. All GPU state
  mirrored to host RAM continuously. Becomes the canonical "read GPU
  state under persistent kernel" infrastructure. Strategic spine of
  the cleanup arc; unblocks every subsequent debug cycle.

Near-term (next sessions):

- **pns** — Unify all cooperative-grid launches into a single
  `persistent_worker`. Delete `megakernel_f32` entry; eliminate the
  "second-coop-launch wedge" hazard architecturally. Exactly one
  cooperative megakernel per process.
- **pc3h** — kv-unify: collapse legacy / paged / per-worker KV cache
  variants to a single abstraction. Paged becomes primary. ~400 LOC
  reduction + ~320 MiB GPU 0 VRAM savings on multi-GPU MoE decode.
- **t8fl** — `CrossGpuStaging<T>` type-encoded UC-buffer contract.
  Makes the "every cross-GPU dst needs UC writer" class impossible
  to write incorrectly (currently maintained by code-review).
- **lr6t** — In-megakernel signal-then-fire dispatch ordering,
  replacing the current sync-then-launch handoff. Closes the
  multi-GPU NaN class for Nemotron-H and qwen3.6 architectural
  completeness.

Mid-term:

- **mla1** — Multi-head Latent Attention for DeepSeek-V4-Flash.
- **ssm1 / ssm2 / ssm3** — Chunked SSD-form SSM prefill,
  head-parallel SSM across GPUs, region-edit staleness measurement.
- **moe1 / moe2 / moe-aff** — Dynamic expert placement control
  plane, host-resident cold experts + VRAM LRU cache, routing-locality
  expert affinity.
- **dp1** — Data-parallel chunk prefill (replaces `tp1` / `tp2`
  which were infeasible at PCIe Gen3 bandwidth).
- **wt2** — Host-RAM canonical KV tier (`ChunkTier`, promote / evict).
- **reg1 / reg2** — Region-based KV API (codebase-as-KV core,
  save / restore via host RAM + disk).
- **hw-cal** — Empirical bandwidth calibration documentation
  (consolidates `exterior_algebra/results/*.json` into a canonical
  `docs/hw-bandwidth-envelope.md`).

See `.beads/` for live status (`bd ready` shows current actionable
work; `bd show <id>` for design docs and acceptance criteria).

## Documentation

- `CLAUDE.md` — build instructions, architecture overview,
  agent-development conventions
- `AGENTS.md` — agent collaboration protocol (bridge)
- `MODELS.md` — supported model status matrix
- `DEPLOYMENT.md` — production host tuning (SCHED_FIFO, isolcpus,
  IRQ pinning, watchdog)
- `COHERENCE.md` — coherence-verification harness reference
- `DIAGNOSTICS.md` — debug-instrumentation reference
- `ENV_CONFIG.md` — environment variable reference
- `composable_kernel/GFX1100_ARCH.md` — gfx1100 ISA + memory hierarchy
  + cross-GPU coherence ruleset
- `.beads/` — issue tracking, epics, kb memories
