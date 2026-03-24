# Agent Preamble — BraidInfer (braidinfer)

Read this BEFORE starting your task. Subagents do NOT see CLAUDE.md.

## The Project

BraidInfer is a green-field Rust LLM inference engine targeting AMD RDNA3 GPUs (4x RX 7900 XTX,
gfx1100) via ROCm/HIP. The architecture centers on a **megakernel** design that fuses the entire
LLM forward pass into a single persistent GPU kernel, targeting hybrid models that combine
attention layers with recurrent layers (Mamba-2/SSD, DeltaNet) and MoE expert parallelism.

## Non-Negotiable Constraints

- Target hardware: AMD RX 7900 XTX (gfx1100, RDNA3). NOT CDNA, NOT NVIDIA, NOT generic GPGPU.
- Language: Rust with HIP FFI (cubecl-hip-sys or direct bindgen). No Python inference path.
- GPU kernels written in HIP C++ (`.hip` / `.cu` compiled with hipcc). Not Triton, not CUDA.
- No mocks, stubs, or fake data. Use real hardware/files. If demo data needed, user will say so.
- Use `bd` for ALL task tracking. No markdown TODOs.
- ALWAYS use `--no-gpg-sign` for git commits.
- Non-interactive shell: use `cp -f`, `mv -f`, `rm -f` (shell aliases add `-i` and will hang).
- PUSH after every session: `git pull --rebase && bd dolt push && git push`.

## Hardware Facts (Do NOT Re-Derive)

- 4x AMD RX 7900 XTX: 96 CUs each, gfx1100, 24 GB VRAM each, 96 GB total
- P2P bandwidth: ~22 GB/s per direction PCIe (NO xGMI/Infinity Fabric between consumer GPUs)
- P2P limitation: peer-to-peer requires `hipDeviceEnablePeerAccess`; consumer cards limited to PCIe speeds
- RDNA3 wavefront: 32 threads (not 64 like GCN/CDNA)
- HipKittens tile DSL targets CDNA3/CDNA4, NOT RDNA3 — do not use for this project
- `hipLaunchCooperativeKernel` requires `cooperative_groups` support; verify device capability first
- rocprofv3 multi-pass PMC collection is broken on GFX11 — use single-pass profiling only

## Key Architecture Decisions (KB-Documented)

- **Megakernel strategy**: Fuse entire forward pass into single persistent kernel (on-GPU interpreter
  pattern, similar to "No Bubbles" from Hazy Research). Reduces kernel launch overhead and enables
  on-GPU scheduling.
- **Pipeline parallelism**: 4 GPUs × layer sharding. Each GPU owns N/4 consecutive transformer layers.
  Activation transfer cost ~22 GB/s × layer_activation_size between GPUs.
- **Recurrent state**: DeltaNet state = (H_t: [B, H, d_head, d_head]), Mamba-2 state = ([B, H, d_state, d_head]).
  States must be checkpointed for speculative decoding rollback.
- **MTP speculative decoding**: Multi-Token Prediction draft heads. Draft tokens verified in parallel.
  Recurrent states must be rolled back on rejection — checkpoint before draft, restore on mismatch.
- **Scheduling**: CPU orchestrator dispatches work to GPU via shared ring buffer; GPU persistent
  kernel polls and executes. Avoids repeated kernel launch overhead.

## Terminology

| Term | Definition |
|------|------------|
| Megakernel | Single persistent GPU kernel fusing entire LLM forward pass |
| SSM | Structured State Space Model (Mamba, Mamba-2) |
| SSD | Structured State Space Dual (Mamba-2 algorithmic formulation) |
| DeltaNet | Linear attention with delta rule update: H_t = (I - β_t k_t k_t^T) H_{t-1} + β_t v_t k_t^T |
| MTP | Multi-Token Prediction — speculative draft heads built into model |
| gfx1100 | AMD GPU target identifier for RX 7900 XTX (RDNA3) |
| CK | AMD Composable Kernel — tile-based GPU programming library |
| HipKittens | Tile DSL for CDNA3/CDNA4 — NOT for RDNA3 |
| P2P | Peer-to-peer GPU memory access over PCIe |
| chunked parallel scan | Prefill algorithm for recurrent layers: process in chunks for parallelism |

## Key Modules (Green-Field — to be created)

| Module | Purpose |
|--------|---------|
| `src/kernel/` | HIP kernel source (.hip files) |
| `src/ffi/` | Rust FFI bindings to HIP runtime |
| `src/scheduler/` | CPU-side orchestrator, ring buffer, dispatch |
| `src/model/` | Layer definitions, weight loading, quantization |
| `src/kvcache/` | KV cache management (paged or contiguous) |
| `src/sampler/` | Token sampling, speculative decoding verification |

## Anti-Patterns (Known Failure Modes)

| Pattern | Why Wrong |
|---------|-----------|
| Using HipKittens for RDNA3 | HipKittens targets CDNA3/CDNA4, not RDNA3 gfx1100 |
| Using rocprofv3 multi-pass PMC on GFX11 | Broken — second invocation gets GPU reset |
| Assuming xGMI/Infinity Fabric between consumer GPUs | Consumer 7900 XTX uses PCIe only (~22 GB/s) |
| Wavefront size = 64 on RDNA3 | RDNA3 wavefront = 32 (not 64 like GCN/CDNA) |
| D2H copy to pageable host memory with ROCm SDMA | Causes GPU reset — use pinned memory |
| Calling hipLaunchCooperativeKernel without capability check | Requires explicit device feature query |
| Storing recurrent state without checkpoint before speculative draft | Cannot roll back on token rejection |
| `git add .` or `git add -A` | Stages other sessions' files — use explicit paths |
| Polling `build-manager status` in a loop | Busy-loop anti-pattern — use `--sync` or monitor agent |

## ROCm/HIP Platform Notes

- Build with `hipcc --offload-arch=gfx1100`
- Rust HIP FFI: cubecl-hip-sys provides raw bindgen bindings (preferred over manual bindgen)
- ROCm SDMA H2D hangs: pageable memory D2H causes GPU reset; ALWAYS use pinned memory for async transfers
- MMQ vs rocBLAS dispatch: benchmark-dependent threshold, not a fixed rule

## Epistemological Rules

1. "Not Found" != "Doesn't Exist". Say "I found no evidence for X."
2. Code > Comments > KB > Your assumptions.
3. 5 rounds of kb-research, not 2.
4. Verify, don't infer. Grep for RESULTS, not TODO comments.
5. State your evidence. Every claim cites file:line, kb-ID, or command output.
6. kb_add before returning. Checkpoint every 10 tool uses.
7. project="braidinfer" for all kb_add/kb_search calls.
8. Cross-project search: first kb_search query MUST use project=None (150+ findings under variant names).

## Stopping Conditions

Stop and return partial results if:
- Same error 3 times consecutively
- 10+ tool calls with no new findings
- 5+ search phrasings with no results
- 8+ files read without concrete output
