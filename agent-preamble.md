# Agent Preamble — BraidInfer (braidinfer)

Read this BEFORE starting your task. Subagents do NOT see CLAUDE.md.

## The Project

BraidInfer is a Rust LLM inference engine targeting AMD RDNA3 GPUs (4x RX 7900 XTX, gfx1100)
via ROCm/HIP. It uses a megakernel architecture: a single persistent cooperative GPU kernel
fusing the entire LLM forward pass (384 blocks x 256 threads, 128-byte instruction set,
grid.sync() barriers). Supports hybrid models with attention + recurrent layers (Mamba-2/SSD,
DeltaNet) and MoE expert parallelism across 4 GPUs.

## Non-Negotiable Constraints

- Target hardware: AMD RX 7900 XTX (gfx1100, RDNA3). NOT CDNA, NOT NVIDIA.
- Wavefront size: 32 (NOT 64 like GCN/CDNA).
- Language: Rust + HIP C++ kernels. No Python inference path. No Triton.
- ALWAYS use `scripts/launch-gpu.py` for ANY GPU command. No exceptions. No `HIP_VISIBLE_DEVICES` bypass.
- GPU commands: always `run_in_background=True` with `timeout=600000`.
- No mocks, stubs, or fake data. Real hardware only.
- P2P between GPUs: PCIe only (~22 GB/s), no xGMI/Infinity Fabric on consumer cards.
- D2H copies: ALWAYS use pinned memory. Pageable D2H causes GPU reset on ROCm SDMA.
- Persistent worker holds ALL CUs on GPU 0. No hipMemcpy/hipLaunchKernel on GPU 0 during inference.
- Safe during inference: host-mapped buffers, write_volatile/read_volatile, GPU-side printf only.
- `--no-gpg-sign` for all git commits. Never `git add .` or `git add -A`.
- Use `bd` for task tracking. No markdown TODOs.
- rocprofv3 multi-pass PMC: broken on GFX11 — single-pass only.

## Key Proven Results (Do NOT Re-Derive)

| Result | Evidence |
|--------|----------|
| Megakernel 2.1x vs naive dispatch | 6.4ms vs 13.4ms per token (measured) |
| Head-parallel GQA across 4 GPUs | 3.1 to 12.1 tok/s (commit c6c0ca8) |
| GPU-native P2P MoE dispatch | 4.75 to 23.1 tok/s on Qwen3.5-35B-A3B (commit e796ab4) |
| hipModuleGetFunction caching | 3.9 to 9.5 tok/s on Nemotron-H 120B (commit f5f3b26) |
| Tiled-LDS GEMV for PCG32/RNF4 | 2.1x speedup on 27B (commit f9c382c) |
| Grid.sync cross-GPU deadlock on RDNA3 | Confirmed (commit e273ce1) — use per-GPU cooperative kernels |

## Terminology

| Term | Definition |
|------|------------|
| Megakernel | Single persistent cooperative GPU kernel fusing entire forward pass |
| SSM | Structured State Space Model (Mamba, Mamba-2) |
| SSD | Structured State Space Dual — Mamba-2 algorithmic formulation |
| DeltaNet | Linear attention with delta rule state update |
| MTP | Multi-Token Prediction — speculative draft heads built into model |
| gfx1100 | AMD GPU target ID for RX 7900 XTX (RDNA3) |
| bqnt | BraidInfer quantized model format |
| kbk | Kernel-by-kernel dispatch (vs megakernel); used on GPUs 1-3 for MoE |
| lean worker | Lightweight cooperative worker for MoE on secondary GPUs |
| P2P | Peer-to-peer GPU memory access over PCIe |

## Key Modules

| Module | Purpose |
|--------|---------|
| `kernels/*.hip` | HIP kernel source (36 files: megakernel, attention, SSM, MoE, quantization) |
| `crates/braidinfer-hip/` | Low-level HIP bindings, module loading, kernel launch, memory |
| `crates/braidinfer-hip/build.rs` | hipcc compilation of .hip kernels |
| `crates/braidinfer-core/` | Shared types (DeviceId, etc.) |
| `crates/braidinfer-runtime/` | Model loading, dispatch, megakernel orchestration, chat, generate |
| `scripts/launch-gpu.py` | GPU reservation and launch (flock-based, mandatory) |
| `scripts/compare_traces.py` | Activation trace comparison for kernel correctness |
| `scripts/bisect_quant.py` | Per-layer quantization bisection tool |

## Anti-Patterns

| Pattern | Why Wrong |
|---------|-----------|
| HipKittens for RDNA3 | Targets CDNA3/CDNA4 only |
| rocprofv3 multi-pass PMC on GFX11 | Broken — GPU reset on second pass |
| wavefront=64 on RDNA3 | RDNA3 wavefront = 32 |
| xGMI between consumer GPUs | Consumer 7900 XTX: PCIe only |
| D2H to pageable memory | Causes GPU reset — use pinned memory |
| hipMemcpy during persistent worker | Deadlocks — all CUs held by cooperative kernel |
| Grid.sync across multiple GPUs | Deadlocks on RDNA3 (confirmed) |
| Direct GPU access (no launch-gpu.py) | VRAM conflicts with other sessions |
| Short launch-gpu.py timeouts | GPUs may be busy for hours — use --timeout 43200 |

## Epistemological Rules

1. "Not Found" != "Doesn't Exist". Say "I found no evidence for X."
2. Code > Comments > KB > Your assumptions.
3. 5 rounds of kb-research, not 2.
4. Verify, don't infer. Grep for RESULTS, not TODO comments.
5. State your evidence. Every claim cites file:line, kb-ID, or command output.
6. kb_add before returning. Checkpoint every 10 tool uses.
7. project="braidinfer" for all kb_add/kb_search calls.
8. Cross-project search: first kb_search query MUST use project=None.

## Stopping Conditions

Stop and return partial results if:
- Same error 3 times consecutively
- 10+ tool calls with no new findings
- 5+ search phrasings with no results
- 8+ files read without concrete output
