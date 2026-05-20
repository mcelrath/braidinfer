# GFX1100 Architecture & Authoring Guide

Practical reference for `gfx1100` (RDNA3 / Navi 31) in Composable Kernel and braidinfer. Two halves: a performance-tuning guide for kernel authors, and a load-bearing **correctness ruleset** (§5.3–§5.5) for multi-agent / persistent-kernel code on this arch.

**Recent reorgs:** 2026-05-19 — §1.2 falsification methodology promoted from §11.19(n);
§11.16+§11.18 merged into §11.19 with sequential re-lettering; §5.0 extracted to
`GFX1100_CACHE_HINTS.md`; §11.13 extracted to `mes/MES_PROBE_ARCHIVE.md`; Appendix A
"Falsified hypotheses (do not retry)" added; TOC marked with status flags. See
`git log GFX1100_ARCH.md` for full history.

## Status as of 2026-05-19

The following are the currently load-bearing operational facts for any
new code on gfx1100 multi-GPU. Read these BEFORE doing anything else:

1. **§5.5 Rule 1 — Canonical cross-agent allocation method (4 sub-rules a/b/c/d).** Every
   cross-GPU buffer MUST be allocated per one of the four sub-rules.
   [CURRENT]
2. **§5.5 Rule 9** — One cooperative kernel per process lifetime. No
   relaunch. Mechanism unknown but reproducible; recovery = process exit.
   [CURRENT]
3. **§11.15** — Immediate-ack protocol mandatory for persistent_worker
   mailbox. Deferred-ack pattern (Phase 2') causes protocol deadlock.
   [CURRENT]
4. **§11.19** — Cold-start mailbox warmup-dispatch is the production
   cure for the gfx1100 multi-GPU first-mailbox-transaction NaN race
   (bd 4e2m). Production default since commit 8e3bad9. [CURRENT]
5. **Rule 10 (producer-fence)** — FALSIFIED. Do not re-propose. The
   §11.18 producer-fence-via-fence_device hypothesis was empirically
   refuted by Exp 3 (CPU producer read-back, 10/30 PASS = baseline).
   See §11.19 for the falsification cascade. [FALSIFIED]

Beyond these five, the rest of the document is reference material
(perf bench, history, falsified hypotheses preserved for record).
TOC entries below are tagged with status flags:
  [CURRENT]            — load-bearing today
  [SUPERSEDED-by-§N.M] — replaced by a newer section
  [FALSIFIED-by-§N.M]  — hypothesis tested and refuted
  [HISTORICAL-RECORD]  — preserved for context, not load-bearing

---

## Table of contents

| § | Topic | Status | When to read |
|---|---|---|---|
| 1 | Why this document exists | [CURRENT] | Once, at onboarding |
| 1.2 | Falsification methodology | [CURRENT] | Before any hypothesis-testing campaign |
| 2 | Architectural quick facts | [CURRENT] | Reference |
| 3 | Execution model & wave mechanics | [CURRENT] | Designing wave/block shapes |
| 4 | WMMA programming model | [CURRENT] | Writing any matmul / attention kernel |
| 5.0-5.2 | Memory hierarchy & cache hints (perf) | [CURRENT] | Tuning a kernel for bandwidth/latency |
| 5.3 | Cross-GPU peer coherence (correctness) | [CURRENT] | **Any** code that crosses GPU boundaries |
| 5.4 | L2 staleness — mechanism (correctness) | [CURRENT] | Understanding why §5.5 rules exist |
| **5.5** | **Authoring deterministic multi-agent code — ruleset** | [CURRENT, MANDATORY] | **Mandatory before writing or reviewing any multi-agent / persistent-kernel code.** Contains decision tree (Rule 1), antipatterns lookup, MTYPE audit code template, PR checklist, and 5 more rules including Rule 9. |
| 6 | Kernel design patterns in CK | [CURRENT] | Implementing a CK instance |
| 7 | Testing, profiling, validation | [CURRENT] | Sprint setup |
| 8 | Tooling tips | [CURRENT] | Day-to-day |
| 9 | Implementation checklist | [CURRENT] | Pre-merge gate |
| 10 | CK vs rocWMMA comparison — moved to `CK_vs_rocWMMA.md` | [EXTRACTED-to-CK_vs_rocWMMA.md] | Choosing between libraries (sibling doc) |
| 11 | Empirical measurements preamble (braidinfer 2026) | [CURRENT] | Bench archive — cite when tuning |
| 11.1 | Wave/reduction primitives | [CURRENT] | Implementing warp-level reductions |
| 11.2 | Wave/reduction primitives (cont.) | [CURRENT] | Implementing warp-level reductions |
| 11.3 | Wave/reduction primitives (cont.) | [CURRENT] | Implementing warp-level reductions |
| 11.4 | Grid-wide barrier benchmark + PCIe-write-before-barrier HAZARD | [CURRENT] (bench); [HISTORICAL-RECORD] (HAZARD subsection) | When in doubt about a multi-GPU barrier wedge |
| 11.5 | LDS bank conflicts | [CURRENT] | Tuning shared memory access patterns |
| 11.6 | cp.async equivalent | [CURRENT] | Overlapping compute and memory |
| 11.7 | ds_swizzle ctrl-bit | [CURRENT] | Using ds_swizzle for lane shuffles |
| 11.8 | Cache invalidation issue costs | [CURRENT] | Measuring cache-control overhead |
| 11.9 | WMMA fragment lane-mapping | [CURRENT] | Writing any WMMA kernel |
| 11.10 | Header consolidation (rdna3/ primitive library file layout) | [CURRENT] | Map of `rdna3_*.h` headers — actively maintained primitive library |
| 11.11 | Multi-GPU latency envelope | [CURRENT] | Cross-GPU communication budgeting |
| 11.12 | Measurement hygiene | [CURRENT] | Any benchmarking campaign |
| 11.13 | Cooperative-grid relaunch wedge — empirical archive (confirmed; mechanism unknown) | [HISTORICAL-RECORD] | Real phenomenon independent of §11.15; six MES-side patches refuted; process exit is only known recovery. Read §11.15 first if debugging a fresh wedge. |
| 11.14 | `s_buffer_gl0_inv` / `s_buffer_gl1_inv` silently no-op on gfx11+ | [CURRENT] | When relying on scalar-cache invalidation to refresh a host-mapped UC poll |
| 11.15 | Persistent worker ack protocol | [CURRENT] | Any change to persistent_worker dispatch loop |
| 11.16 | UC dst alone insufficient | [SUPERSEDED-by-§11.19] (MERGED-to-§11.19) | Historical; content merged into §11.19 (r)-(v); §11.16 header is a stub |
| 11.17 | Warmup-to-active transition stream/event state | [CURRENT] | Any change to Model::load or worker startup sequence |
| 11.18 | host-mapped portable-coherent + multi-GPU GART — producer L2 staleness — Rule 10 producer-fence hypothesis | [FALSIFIED-by-§11.19] (MERGED-to-§11.19) | Rule 10 producer-fence hypothesis tested and refuted by Exp 3 (10/30 PASS = baseline). Do not re-propose. Content merged into §11.19 (w)-(aa); §11.18 header is a stub |
| 11.19 | Cold-start mailbox visibility race + production cure | [CURRENT] | The canonical account of bd 4e2m; read before any multi-GPU warmup work |

**For agents**: if you are debugging a `"deterministic for N steps, then divergent"` bug or writing any new cross-agent buffer, jump directly to **§5.5 Rule 4** (diagnostic) and **§5.5 Rule 1** (allocation method decision tree). Those two are the highest-yield entry points.

---

## 1. Why This Document Exists
- **One place for the “gotchas”** – gfx1100 WMMA has wave-level constraints (lane replication, opsel behavior) that are easy to miss and expensive to debug.
- **Sprint linkage** – SPRINT1 expands WMMA coverage + memory access polish; SPRINT2 adds scheduling/shape specialization; SPRINT3 hardens build/test. This doc maps architectural facts to those tasks.
- **Direct mapping to CK work** – Each section ends with concrete “Do/Verify” bullets so the sprint workstream can land changes predictably.

---

## 1.2 Falsification Methodology

Cross-reference: §11.19(n) for the original context where this recipe was developed.

Falsification cascade discipline for cross-GPU coherence bugs on gfx1100 (or any hardware where reaching the actual cure layer is uncertain):

1. **Capture symptom signature** with per-trial / per-worker / per-position resolution. Don't aggregate; the diagnostic shape lives in the dimensions.
2. **Bisect by intervention class** — CPU-side / kernel-side / shader-side. Each class has different reach into the cache hierarchy.
3. **Falsify each class with measurable delta vs baseline.** N=30 minimum (binomial 3σ ≈ ±5 for N=30; smaller samples are noise traps).
4. **Don't trust partial-improvement results.** Overlapping cures from different classes signal the noise floor, not a stack of independent bugs. If glc+dlc gives 16/30 and baseline is 16/30, the 16/30 is the noise floor — not a bug-overlap.
5. **Document falsified hypotheses explicitly.** Save future investigators from re-running the same dead-ends. Failure record IS load-bearing artifact.

This is the "tractor squashing grape" failsafe — if you go through this cascade and reach an unreachable layer, you know it's an unreachable layer, not a "we didn't try hard enough" answer.

---

## 2. Architectural Quick Facts
| Subsystem | Key Facts | Why it matters |
|-----------|-----------|----------------|
| **Execution Units** | RDNA3 Workgroup Processors (WGPs) bundle **two compute units**; each CU contains **two SIMD32s that share one memory path** (RDNA3 ISA §1.1). Wave32 is native, Wave64 is optionally supported (RDNA3 ISA §2.1). | Favors Wave32-optimized WMMA pipelines (`DeviceGemm_Wmma_CShuffleV3` already defaults to Wave32). Choose `BlockSize` ≤ 256 threads to keep ≥2 resident waves per SIMD. |
| **Matrix Cores (WMMA)** | gfx1100 exposes native 16×16×16 WMMA instructions for FP16, BF16, INT8→INT32, and WaveMMA also gains INT4 support (RDNA3 architecture announcement; RDNA3 ISA Chapter 19). | Sprint 1 Task 1 (instance expansion) should emit kernels that align tile dimensions to 16-multiples and gate INT4/INT8 instances where accuracy allows. |
| **Register/LDS resources** | Every wave receives **up to 256 VGPRs** (gfx1100 allocates in 24-register chunks for Wave32 or 12 for Wave64) and **106 SGPRs + 16 TTMPs** (RDNA3 ISA §3.3). gfx1100 SIMDs expose **1536 VGPRs per SIMD32**, so waves per SIMD depend on `floor(1536 / round_up_to_allocation(vgpr_count))` where the rounding matches the 24/12-register granularity. Each WGP exposes **128 KB LDS (64 banks × 4 B)**, while the **4 KB Global Data Share is chip-wide** (§1.2.2). | Track `vgpr_count`/`sgpr_count` from compiler output, maintain LDS usage under 64 KB per work-group, and design swizzles that avoid 64-bank conflicts; budget GDS as a shared resource across all WGPs. |
| **Caches** | On RX 7900 XTX (gfx1100), ROCm reports **L1=32 KB**, **L2=6 MB**, **L3=96 MB** and **128 B cache lines** via `rocminfo` (L3 corresponds to the 96 MB Infinity Cache capacity). RDNA3 also has additional per-WGP caches (instruction/scalar/texture) that are not surfaced as HSA cache levels. | Prefer 128-bit vector loads/stores, keep base pointers aligned, and design tiles to reuse within L2/L3. Use Radeon Memory Visualizer/RGP for cache behavior; rocprofiler counter availability varies by tool/version. |
| **Memory** | gfx1100 (Navi 31) uses up to a 384-bit GDDR6 interface delivering ~960 GB/s; Infinity Cache supplies up to 96 MB of on-package bandwidth amplification (RDNA3 specs). Peak compute is 61.4 TFLOP/s FP32 / 122.8 TFLOP/s FP16. | Base roofline targets on these published numbers and capture real bandwidth with `roofline_analysis_gfx1100`/Radeon Memory Visualizer, not datacenter/HBM peak numbers. |
| **Scheduler** | RDNA3 introduces dual-issue shader ALUs capable of issuing two instructions per cycle (RDNA3 architecture overview; RDNA3 ISA §4.4). WMMA/VALU ops still monopolize vector lanes if latency hiding is insufficient. | Adaptive scheduler heuristics (Sprint 2 Task 1) should consider dual-issue pairing opportunities and rely on AMD counters (SQ_INSTS_VALU, SQ_WAVES, etc.) collected via GPU Perf API. |

### 2.1 Validated Target (This System)
The following values are **directly observed on this machine’s RX 7900 XTX cards** (gfx1100) using `rocminfo`, `rocm-smi`, and a small HIP `hipGetDeviceProperties` query:

- **GPU**: AMD Radeon RX 7900 XTX (`gfx1100`) ×4
- **Compute**: 96 CUs, 6 shader engines; HIP reports `multiProcessorCount=48` (WGPs)
- **Wavefront**: Wave32 (`warpSize=32`)
- **Caches**: L1 32 KB, L2 6 MB, L3 96 MB (Infinity Cache), cache line 128 B
- **Memory (this machine)**: 24 GiB VRAM, 384-bit bus (Pro W7900 is also `gfx1100` but typically has larger VRAM capacity; caches/CU topology should match)

**Do/Verify**
1. Confirm target GPU via `rocminfo | rg gfx1100`.
2. Compile with `--offload-arch=gfx1100` (or build CK with `-D GPU_TARGETS=gfx1100`).
3. Use `ck::is_gfx11_supported()` (`include/ck/host_utility/device_prop.hpp`) to guard gfx1100-only code paths.

---

## 3. Execution Model & Wave Mechanics

### 3.1 Wavefront Sizes
- gfx1100 defaults to **Wave32** (RDNA3 ISA §2.1). Each SIMD executes 32 lanes per wave; two Wave32s can co-reside per SIMD with low context-switch cost.
- **Wave64** is still available but reduces the number of concurrently resident wavefronts (one per SIMD). WMMA instructions have both `_w32` and `_w64` flavors (RDNA3 ISA Chapter 19; `include/ck/utility/amd_wmma.hpp`).

**Guideline:** Prefer Wave32 for WMMA on gfx1100 (latency hiding + aligns with `CK_TILE_MAX_THREAD_PER_BLOCK 256` in `ck_tile/core/config.hpp`). Consider Wave64 only when you can justify the tradeoff (lower occupancy) and you are intentionally managing VGPR pressure.

### 3.2 Register Pressure vs Occupancy
- Track `vgpr_count`/`sgpr_count` via `hipcc --save-temps` or Radeon GPU Profiler.
- RDNA3 caps each wave at **256 VGPRs** and, on gfx1100, allocates VGPRs in chunks of **24** (Wave32) or **12** (Wave64) (RDNA3 ISA §3.3.2).
- gfx1100 exposes **1536 VGPRs per SIMD32**, so resident waves per SIMD32 is `floor(1536 / round_up_to_allocation(vgpr_count))`.
- Occupancy typically stays ≥4 waves/SIMD until the rounded-up VGPR count exceeds **384**.
- CK’s launch bounds (`CK_TILE_MAX_THREAD_PER_BLOCK 256`, `CK_TILE_MIN_BLOCK_PER_CU 2`) already match gfx1100’s sweet spot. Maintain them unless profiling shows a block-size pathologically underutilizing LDS or exhausting the 64 KB per work-group cap.

**Checklist:**
1. Build kernel with `hipcc --save-temps` and inspect `.s` for `vgpr_count`/`sgpr_count`. Confirm the compiler-reported `vgpr_count ≤ 256` and `sgpr_count ≤ 106` (excluding TTMPs); e.g., a kernel with `vgpr_count=128` rounds to 144 registers on gfx1100, yielding `floor(1536/144)=10` waves/SIMD.
2. Use Radeon GPU Profiler (RGP) or ROCm profiling tools to ensure ≥4 waves per SIMD. Cross-check measured wave residency against `floor(1536 / round_up_to_allocation(vgpr_count))`.
3. When Split-K or block fusion increases LDS usage, verify mode selection (CU vs WGP mode) because CU mode slices LDS into two 32-bank halves (§2.3) and can reduce usable space per work-group.

---

## 4. WMMA Programming Model on gfx1100

`include/ck/utility/amd_wmma.hpp` exposes the gfx11 intrinsic wrappers:

```cpp
// Wave32 FP16 accumulation path used by DeviceGemm_Wmma_CShuffleV3
template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a,
                               const half16_t& reg_b,
                               FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#endif
    }
};
```

**Key takeaways**
1. **Only 16×16×16 fragments** – Hardware WMMA is fixed to 16×16×16. Larger GEMMs are built by tiling those fragments.
2. **Opsel matters for 16-bit C/D** – For WMMA ops with 16-bit C/D, `opsel` selects whether C/D elements live in the low or high 16 bits of each 32-bit VGPR (RDNA3 ISA §7.9 / GPUOpen “WMMA on RDNA3”). Use it deliberately (e.g., accumulator double-buffering).
3. **INT8 WMMA can fuse simple transforms** – gfx1100 supports `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (RDNA3 ISA Chapter 19). Use `neg_a/neg_b/clamp` to fuse common requantization steps.
4. **Fill whole fragments** – Each lane supplies a “fragment” worth of A/B/C data (held in VGPRs). For FP16/BF16, CK’s `half16_t` fragments represent 16 elements; make sure your global/LDS reads populate the full fragment expected by the intrinsic (not partial/scalar leftovers).
5. **WMMA fragment layout is asymmetric** – The RDNA3 ISA’s canonical WMMA VGPR layout has **A in column-major order**, while **B/C/D are row-major**; A/B fragments must also satisfy the lane replication rule (§4.2). Loader logic and thread mapping must respect both.

### 4.1 Fragment VGPR Footprint (Register Budgeting)
WMMA fragments live in VGPRs. Their footprint is a first-order driver of occupancy and spills.

- **A/B fragment footprint per lane** (independent of wave size; each fragment represents 16 elements distributed across a lane’s registers):
  - FP16/BF16: **8 VGPRs** each
  - `iu8`: **4 VGPRs** each
  - `iu4`: **2 VGPRs** each
- **C/D fragment footprint per lane** (regardless of C/D datatype):
  - Wave32: **8 VGPRs**
  - Wave64: **4 VGPRs**

**Optimization opportunity:** If your kernel is register-bound, the above numbers help you reason about where the pressure comes from (A/B datatype choice, accumulator type, repeat factors, and whether Wave64 is even worth exploring).

### 4.2 Required Lane Replication (RDNA3 WMMA “Gotcha”)
Per the RDNA3 ISA (§7.9) and AMD’s GPUOpen “WMMA on RDNA3” guidance, WMMA requires A/B fragment data to be replicated across lane groups:

- **Wave32:** lanes **0–15** must match lanes **16–31** for A/B fragments.
- **Wave64:** lanes **0–15** must also be replicated into lanes **32–47** and **48–63**.

**Optimization opportunity:** If you are writing custom WMMA microkernels (or custom fragment loaders), treat this as a reuse constraint:
- The simplest correct pattern is to compute fragment indices using `lane_id % 16` so both half-waves naturally address the same elements.
- If memory traffic becomes the bottleneck, you can sometimes load unique fragment values in only one lane group and use cross-lane moves (permutes/shuffles) to populate the replicated lanes. This is highly kernel- and compiler-dependent; measure carefully because extra permute instructions can erase (or exceed) the saved memory traffic.

### 4.3 Prefer Intrinsics Over Inline Assembly (Hazards)
If you use WMMA at the ISA level, prefer compiler intrinsics over hand-written inline assembly:

- Intrinsics let the compiler model scheduling constraints and data hazards around matrix instructions.
- Inline assembly can hide hazards (e.g., using results too early), leading to correctness or performance issues that are hard to diagnose.
- The RDNA3 ISA notes a specific hazard: **back-to-back dependent WMMA** ops require at least one `v_nop` (or other independent VALU op) between them when the first WMMA’s D overlaps the second WMMA’s A/B (RDNA3 ISA §7.9). Intrinsics make it much more likely the compiler will schedule a safe gap.

**Measured (this system):**
- `test/gfx1100_microbench/wmma_spacing_bench.cpp` compares independent WMMA, “dependent chain”, and an intentionally in-place (A==C==D) pattern. On this RX 7900 XTX / ROCm 7.1.1 stack, independent and dependent variants were ~17 ns/WMMA, in-place was slightly slower (~19 ns/WMMA), and manually inserting `v_nop` did **not** improve performance. Treat this as evidence to **trust intrinsics + compiler scheduling** rather than adding fixed NOPs by hand.

### 4.4 Matrix Instruction Calculator (Debugging + Tuning Aid)
When debugging register mappings (which lane holds which fragment element) or validating a planned swizzle/loader, the AMD Matrix Instruction Calculator tool can provide architecture-specific mapping details for WMMA instructions.

**Do/Verify**
1. Prefer compiler intrinsics (`__builtin_amdgcn_wmma_*`) over inline asm (§4.3).
2. If writing custom fragment loaders, enforce the lane replication rule (§4.2) and validate correctness on a small 16×16 case before scaling up.
3. Use the AMD Matrix Instruction Calculator tool when validating lane/register mappings (§4.4).
4. Unit tests should call the WMMA path using `DeviceGemm_Wmma_CShuffleV3` and compare against `ck::utils::check_err` as in Sprint 1 Task 4.3.

---

## 5. Memory Hierarchy & Access Guidelines

**Section scope at a glance**: §5.0–§5.2 are **performance characterization** (cache hints, hierarchy capacities, LDS mode tradeoffs) — read when tuning a kernel. §5.3–§5.5 are **correctness rules** for multi-agent / persistent-kernel code — read **before writing or reviewing any kernel** that crosses agent boundaries. The correctness sections are load-bearing: skipping them has cost ~30 person-days of debugging in this codebase. §5.5 is the canonical operational ruleset; §5.3 and §5.4 give the underlying mechanism.

| Level | Capacity | Latency | Notes | Optimization Hooks |
|-------|----------|---------|-------|--------------------|
| **VGPR** | 1,536 regs/SIMD32 (≤256 per wave, 24-reg allocation blocks – RDNA3 ISA §3.3.2) | 1 cycle | Holds WMMA fragments and accumulators. Occupancy = `floor(1536 / round_up_to_allocation(vgpr_count))`. | Keep fragments resident, limit repeats to keep `vgpr_count ≤ 192` and avoid spills. |
| **SGPR** | 106 SGPR + 16 TTMP per wave (ISA §3.3.1) | 1 cycle | Loop counters, descriptors, scalar pointers. Scalar cache sits in front of these loads. | Keep SRD handling simple: buffer resource descriptors (V#) are 128-bit and must be 4-SGPR aligned when resident in SGPRs (ISA §9.6). Avoid full-cache invalidates except for special low-level experiments (§5.1). |
| **LDS** | 128 KB/WGP (64 banks × 4 B; CU mode splits into two 64 KB halves) | tens of cycles (varies) | Shared memory for work-group cooperation; LDS_DIRECT/PARAM loads unavailable in WGP mode (§2.3). | Align/pad LDS tiles to avoid 256 B strides, use CK swizzles to distribute across 64 banks, keep <64 KB per work-group. |
| **L0 Instruction/Data** | Per-WGP caches (implementation-defined) | varies | Not exposed as an HSA cache level; treat as a small, fast cache that benefits from tight loops and reuse. | Favor structured, repeated access patterns; avoid instruction bloat in hot loops. |
| **L1 (GL1/Texture)** | Shader-array cache (implementation-defined) | varies | Not exposed as an HSA cache level; cache line size is 128 B on RX 7900 XTX (`rocminfo`). | Use 128-bit vector loads/stores and align base pointers; coalesce wave accesses into 128 B lines. |
| **L2** | 6 MB (RX 7900 XTX; `rocminfo`/HIP) | varies | Shared; feeds L3/Infinity Cache. On this system, `cache_chase_bench` shows a clear latency jump between ~2 MiB and ~8 MiB working sets, consistent with “exceeds L2” behavior. | Use tiles that create reuse inside L2; if you cannot, treat the kernel as memory-bound and focus on coalescing and hinting. |
| **L3 / Infinity Cache** | 96 MB (gfx1100; `rocminfo` reports this as L3) | slower than L2, faster than VRAM (varies) | Exposed as HSA “L3” but physically Infinity Cache on the MCDs. The ISA exposes `DLC` as a **temporal hint for Infinity Cache**. On this system, `cache_chase_bench` shows `dlc=1` can strongly increase latency once the working set exceeds L2 (e.g., 8 MiB: ~107 ns/load vs ~254 ns/load). | Keep working sets within L2+Infinity Cache when possible; consider non-temporal hints only for true streaming. Validate with RMV/RGP (counter availability varies). |
| **GDDR6 Memory** | 384-bit × 20 Gbps ≈ 960 GB/s | highest (varies) | Off-chip bandwidth limit for RX 7900 XTX-class parts. | Use double-buffered pipelines, pipeline distance ≥2, and base roofline targets on ~960 GB/s rather than datacenter HBM metrics. |

- **Bank-conflict avoidance** – LDS is 64 banks × 4 B in WGP mode (RDNA3 ISA §2.3.1). On this system, `lds_bank_conflict_bench` shows large slowdowns for certain regular per-thread strides (notably 128–512 B in the tested pattern). Use CK swizzles (`S<1,32,1,8>` etc.) and padding so `(thread_id % 64)` spreads accesses across banks, and validate your exact pattern with a microbench. (CU mode splits into two 32-bank halves; see §5.2.)

### 5.0 Cache Controls (GLC/SLC/DLC) — extracted to sibling

The measured microbench content for GLC/SLC/DLC behavior on gfx1100 —
raw buffer loads/stores/copies bandwidth, pointer-chase latency vs
working-set size, hot-vs-streaming pollution, paged-KV gather
(block-table indirection), stride/page sensitivity, attention-shaped
microbenches — has been extracted to **GFX1100_CACHE_HINTS.md** for
focus. The load-bearing operational rules remain in §5.5.

See: [GFX1100_CACHE_HINTS.md](./GFX1100_CACHE_HINTS.md)

Topics in the extracted sibling: GLC/SLC/DLC raw bandwidth, pointer-chase
latency vs working-set, hot-vs-streaming pollution, paged-KV gather
(block-table indirection), stride/page sensitivity, attention-shaped
microbenches across Kimi/MiniMax/GLM/Focus-A/Focus-B regimes.


### 5.1 Scalar Cache & Global Data Share (GDS)
- **Scalar cache / descriptors** – Buffer resource descriptors (V#) are 128-bit values resident in 4 SGPRs (4-SGPR aligned) and are sent to the texture/cache path with buffer instructions (RDNA3 ISA §9.6). If you store SRDs in memory (e.g., descriptor tables), keep them naturally aligned (16 B) and load with scalar buffer loads (`S_BUFFER_LOAD_B128`) before hot loops. On this system, `srd_table_load_bench` (uniform scalar 16B loads; verified to compile to `s_load_b128`) showed little sensitivity to base misalignment in the 0–15B range or to table stride (16B–1024B) — suggesting alignment is more about **correctness and keeping a single 16B scalar load** than about a large steady-state bandwidth gain.
- **Cache invalidation is heavy** – `S_GL1_INV` and `S_DCACHE_INV` invalidate entire caches (RDNA3 ISA §8.1.3) and are only known complete after `s_waitcnt lgkmcnt(0)`. The ISA also requires INV be in a group by itself (not in a clause). On this system (RX 7900 XTX), `cache_invalidate_bench --iters 4096` measured: baseline ~0.279 ms, `S_DCACHE_INV` ~0.388 ms, `S_GL1_INV` ~0.519 ms. Treat these as last-resort, low-level tools.
- **Global Data Share (GDS)** – RDNA3 devices expose a 4 KB GDS shared across all WGPs (§1.2.2.2). The ISA states GDS provides **2 integer atomic units** (shared), while LDS contains **64 integer atomic units** per WGP (§1.2.2.1). These counts refer to **unordered integer atomics on GDS/LDS** (not global-memory atomics, which are serviced in the L2/atomic path). In HIP/CK today, you should expect to use **global memory atomics** or **LDS atomics**, not GDS atomics (GDS is not a common/high-level programming surface).

### 5.2 LDS Mode Selection (CU vs WGP)
- RDNA3 supports CU mode (each CU manages its own 64 KB LDS half) and WGP mode (full 128 KB accessible to all four SIMD32s) per work-group (§2.3). Compute kernels launched through HIP/CK default to **WGP mode** on gfx1100; there is currently no CK flag to force CU mode.
- Interpret older CU-mode tuning advice carefully. Unless you explicitly modify driver launch state or future CK knobs expose CU mode, assume all kernels use the WGP configuration described above.

**Prefetching & Swizzling**
- `BlockwiseTensorSliceTransfer` supports `SrcAccessOrder` and `ThreadClusterArrangeOrder`. For gfx1100, set `SrcAccessOrder` to channel the fastest-changing dimension across the low bits of the thread id, matching bank order.
- Add software prefetch by unrolling `For` loops and reading `NextLoad` two iterations ahead when `Gemm_KPerBlock ≥ 64`.

### 5.3 Cross-GPU Peer Coherence (Multi-GPU / EP Workloads)

**Hardware limitation (gfx1100)**: `__threadfence_system()` (system-scope fence) **hangs on gfx1100**. The ISA has no L2 invalidation instruction — `buffer_gl2_inv`/`buffer_invl2` are rejected by `llvm-mc --mcpu=gfx1100`; only `buffer_gl0_inv` and `buffer_gl1_inv` exist. Without L2 invalidation, a peer write from GPU A lands in GPU B's physical VRAM but GPU B's L2 stays stale. Any spin-wait reader on GPU B sees cached zeros forever.

**Memory allocation flags determine cross-GPU visibility**:

Tested on 4× RX 7900 XTX (gfx1100): GPU 0 peer-writes to GPU 1's allocation; GPU 1 spins on a signal (1M iteration limit).

| Flag | MTYPE | Spin-wait result | After `hipStreamSynchronize` |
|------|-------|-----------------|------------------------------|
| `hipDeviceMallocFinegrained` (0x1) | CC (Cache Coherent) | TIMEOUT — signal never arrives | Data correct |
| `hipDeviceMallocUncached` (0x3)    | UC (UnCached)       | Signal at ~55K iters (~22 μs) | Data correct |

**Why fine-grained fails for spin-wait**: MTYPE=CC bypasses GPU B's L2 on *reads*, but GPU A's peer *writes* are still cached in GPU A's L2. Without a system-scope flush (unavailable on gfx1100), the data sits in GPU A's L2 and never reaches GPU B's VRAM. Fine-grained memory is correct for CPU-GPU coherence but insufficient for live GPU-GPU spin-wait.

**Why uncached works**: MTYPE=UC bypasses all caches on **all** GPUs — GPU A's writes go directly over PCIe to GPU B's physical VRAM; GPU B's reads go directly from VRAM. After `__threadfence()` (agent scope), the write is in VRAM and immediately visible to GPU B.

**Workaround pattern — no `__threadfence_system` needed**:

```cpp
// Allocate signal, counter, and data as uncached on the receiving GPU
hipExtMallocWithFlags(&signal,  sizeof(int),          hipDeviceMallocUncached);
hipExtMallocWithFlags(&blk_ctr, sizeof(unsigned int), hipDeviceMallocUncached);
hipExtMallocWithFlags(&data,    N*sizeof(float),      hipDeviceMallocUncached);

// Writer (GPU A): write data, global barrier via atomicAdd, then signal
__global__ void writer(volatile float* data, volatile int* signal,
                       volatile unsigned int* blk_ctr, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val + idx;
    __syncthreads();
    __threadfence();   // agent scope: drain this block's stores to VRAM
    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd((unsigned int*)blk_ctr, 1u);
        if (old == gridDim.x - 1) {   // last block: all data now in VRAM
            *signal = 1;
            __threadfence();
        }
    }
}

// Reader (GPU B): spin on uncached signal — bypasses L2, reads live VRAM
__global__ void reader(volatile float* data, volatile int* signal, float* out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int iters = 0;
        while (*signal == 0 && iters < 10000000) iters++;  // always cap spin loop
    }
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[idx];
}
```

**Practical guidance**:
- **Signal (4 bytes)**: always `hipDeviceMallocUncached` — zero bandwidth cost.
- **Block counter (4 bytes)**: also `hipDeviceMallocUncached` — atomicAdd increments must be immediately visible across all of GPU A's blocks before the last block writes the signal.
- **Data buffers**: use `hipDeviceMallocUncached` if GPU B reads data *during a spin-wait* (streaming, first pass). If GPU B reads data only *after* a `hipEvent` or kernel-boundary sync, regular `hipMalloc` is fine and preserves L2 reuse.
- **Always cap spin loops**: an uncapped spin-wait on a signal that never arrives hangs the GPU, triggers TDR, and can reboot the system.
- **Safe default**: `hipEventRecord` on the writer's stream + `hipStreamWaitEvent` on the reader's stream. The uncached pattern avoids kernel-launch overhead only when a persistent/fused kernel is explicitly desired.
- **Persistent-kernel hazard**: If the writer kernel does **not** exit immediately after `*signal = 1` (i.e. it continues running and reaches another grid-wide barrier), use AGENT scope on the signal store *or* defer the store to after the next barrier. A SYSTEM-scope UC store followed by `atomic_block_barrier` can wedge on `s_waitcnt_vscnt` under multi-GPU PCIe pressure; see §11.4 HAZARD subsection for the mechanism and verified mitigations.

**RDNA4 (gfx1200) context**: gfx12 adds `global_wb scope:SCOPE_SYS` and `global_inv scope:SCOPE_SYS` — full L2 writeback + invalidation. `__threadfence_system()` is expected to work on gfx1200, making this workaround gfx1100-specific.

### 5.4 L2 Staleness — When, Why, and How to Detect (gfx1100-specific)

gfx1100 has **no L2 invalidation instruction**. Only `buffer_gl0_inv` and `buffer_gl1_inv` exist; `buffer_gl2_inv`, `buffer_invl2`, and `buffer_wbl2` are rejected by `llvm-mc --mcpu=gfx1100`. This is the most consequential gfx1100 limitation that catches inference workloads — and the bug it produces is silent, non-deterministic, and often manifests as "deterministic for N steps, then divergent."

#### When L2 staleness DOES NOT occur (most of the time)

In a single kernel launch on a single agent, with normal "writer kernel → kernel boundary → reader kernel" flow:
- AGENT-scope fences flush the writer's L2 to HBM at kernel exit.
- The reader kernel starts with an empty L2 (cold) for any addresses it hadn't already cached.
- L2 is **mostly self-coherent within a single agent** because all writes go through the L2 controller, which serves subsequent reads from the same controller's view.

You can write thousands of LLM-inference kernels on a single GPU and never encounter L2 staleness as long as: writers complete a kernel before readers start, and no buffer is mutated across kernel boundaries with cached intermediate reads.

#### The cache hierarchy that makes "L2 staleness" a misnomer

A useful clarification before the failure cases: this section uses "L2 staleness" as shorthand for "stale-cache staleness," but on gfx1100 the actual offender is often **L0/L1 above L2**, not L2 itself. The hierarchy from compute-unit-local to device-global:

| Level | Scope | Capacity | Invalidation primitive on gfx1100 |
|---|---|---|---|
| L0 (vector cache) | per-SIMD | ~16 KB | implicit at kernel boundary; otherwise `buffer_gl0_inv` |
| L1 (per-WGP) | per-WGP (workgroup processor) | ~128 KB | `buffer_gl1_inv` (cheap, ~ns) |
| L2 (GL2) | device-wide | 6 MB on RX 7900 XTX | **none in ISA** (no `buffer_gl2_inv` on gfx11) |

Critical empirical observation (`l2_evict_bench_v2`, commit `96211a`): **a single-thread probe that repeatedly reads an address keeps that line in its WGP's per-CU L0/L1 caches.** L2 pressure (scratch reads through L2) does **not** displace those upper caches — they're per-CU and above L2 in the hierarchy. So even a fully-flushed L2 won't help if the WGP's own L0/L1 holds the stale value. This is why "memory pressure" cannot be a sanctioned flush primitive (Rule 3) and why the canonical mitigations (Rule 1) work by bypassing the cache hierarchy entirely (UC), forcing coherence on each read (atomic-load), or letting kernel-exit fences clear the per-CU caches as part of normal kernel teardown.

Mental model for the rest of this section: when reading "L2 staleness," substitute "any-cache-level staleness." The mechanisms are the same; the level varies with the read pattern.

#### When L2 staleness DOES occur (the failure cases)

L2 staleness manifests when **any of these conditions are simultaneously true**:

1. **Reuse**: a buffer slot is written multiple times across decode steps, kernel iterations, or batches — and the reader has previously cached the prior write.
2. **No invalidation point** between the old and new writes (no kernel boundary that drains the reader's L2 line, no `buffer_gl1_inv` issued by the reader for that address).
3. **Pure-load reader**: a kernel reads the buffer without writing back, with no `atomicAdd` (which forces a fresh fetch through the coherence machinery).
4. **Multi-agent or multi-engine writer**: another GPU (P2P), the SDMA engine, or the host (PCIe-mapped) wrote the buffer while the reader's L2 stayed warm.

Common production patterns that meet all four conditions on gfx1100:

| Pattern | Why it stales | Symptom |
|---|---|---|
| Pool-cycled scratch buffer (size N) reused every N decode steps | Step N+1 wraps to slot 0; L2 still has slot 0's stale data from step 0 | "Deterministic for N steps, then divergent" — *the most common decode-time non-determinism signature on gfx1100* |
| Persistent kernel reading a host-mapped pinned page that the host updates between iterations | Host write lands in physical RAM via GART; GPU's L2 has the stale prior contents | Stuck flag-poll, or stale instruction reads |
| Cross-GPU MoE output accumulator (cached VRAM, not UC) | Peer GPU's P2P write lands in physical VRAM; local L2 keeps the prior step's data | Garbled MoE outputs, "step N first divergence" where N = expert-rotation period |
| GDN/Mamba recurrent state buffer reused across decode steps | Same as pool-cycled scratch but with the recurrent dependency | Recurrent state drift after enough steps |
| KV cache "active chunk" that's overwritten at chunk boundaries | At chunk seal/wrap, reader L2 may serve the just-replaced slot | Chunk-boundary-aligned divergence |
| Atomic accumulator pattern but with **non-atomic reads** after the accumulator finalizes | Atomics force coherence; plain loads after may still hit stale L2 | "Last reduction wins" non-determinism |

#### How to detect L2 staleness (diagnostic patterns)

Symptoms that should immediately raise the L2 hypothesis:

- **"Deterministic for N steps, then divergent"** where N matches some buffer-pool size, chunk size, or expert-rotation period. **This is the signature pattern**.
- **Prefill bit-exact, decode divergent.** Prefill writes fresh buffers; decode reuses them.
- **Same divergence position across runs** (e.g. always at decode step 6). L2 line lifetime is workload-deterministic; pure cache thermals would produce variable divergence positions.
- **Disappears with `--ngl 0`** or with single-GPU runs (L2 staleness needs the multi-agent / persistent-kernel context).
- **Disappears under any of**: adding a `hipDeviceSynchronize` before the read, switching the buffer to `hipDeviceMallocUncached`, switching the read to `__hip_atomic_load` (forces coherence), adding `buffer_gl1_inv` on the reader path. **If any of these "fixes" make the bug disappear, the cause is L2 (or L1) staleness.**

Quick triage checklist when investigating a non-determinism bug:

1. **Trace divergence step boundary**: at exactly which decode step / iteration / batch does the first bit-divergent output appear? Note the number. Look for buffer pools / cycle counts that match.
2. **Identify the first divergent buffer**: trace upstream from the divergent output. Find the most upstream buffer where two runs first differ. *This buffer is the L2-stale candidate.*
3. **Check the buffer's allocation**: `hipDeviceMallocUncached`? Then L2 is bypassed and this isn't your problem — keep looking upstream. `hipMalloc`? Then it's L2-cached and a candidate.
4. **Check the buffer's reuse pattern**: how many slots? How often does the index wrap? Does the wrap align with the divergence step boundary?
5. **Check the writer**: who writes this buffer? Same agent (intra-device, mostly safe), or a peer GPU / host / SDMA (cross-agent, high risk)?
6. **Try `hipDeviceMallocUncached` for that buffer**. If the bug disappears, you've confirmed L2 staleness.

#### How to fix L2 staleness on gfx1100

In rough order of preference:

1. **Don't reuse buffers across iterations** — allocate per-step from a fresh region. Avoids the problem entirely.
2. **`hipDeviceMallocUncached` for the contended buffer** — peer/host writes are immediately visible; local reads bypass L2. The standard cross-agent pattern from §5.3. Bandwidth cost is real (UC has no L2 caching); only use for the contended hot path.
3. **`__hip_atomic_load` instead of plain load** — forces the L2 controller to fetch through the coherence machinery. Works for small reads (single-word flags, counters) where you don't need to invalidate a whole line.
4. **Reader-side `buffer_gl1_inv` before reading** — only invalidates L1, not L2. Useful for **L1 staleness** in single-agent multi-WGP scenarios (different CUs caching different views of the same line). **Does NOT fix L2 staleness.** Easy to confuse the two; if `buffer_gl1_inv` "fixes" the bug, the staleness was at L1, not L2.
5. **Kernel-boundary sync** (`hipStreamSynchronize`, `hipEventRecord`+`hipStreamWaitEvent`) — at the cost of kernel-launch overhead. The AGENT-scope release at kernel exit drains L2 to HBM; subsequent launches start with cold L2 for those lines.
6. **Force eviction by working-set pressure** — if your hot buffer is much smaller than L2 (6 MB on gfx1100) and you're cycling it, you can interpose a synthetic read of a large unrelated buffer between writer and reader to force the L2 line out. **Fragile, not recommended** — use UC mapping instead.

What does NOT fix L2 staleness on gfx1100:

- `__threadfence()` (agent scope) — drains *this* block's stores to L2; doesn't invalidate other readers' L2.
- `__threadfence_system()` — hangs on gfx1100 (see §5.3).
- `__syncthreads()` — block-local barrier; no L2 effect.
- `cg::grid_group::sync()` — grid-wide synchronization, agent-scope release/acquire fences; doesn't invalidate L2 lines that aren't part of this kernel's coherence sweep.
- `s_waitcnt vmcnt(0) vscnt(0)` — drains this wave's outstanding ops; doesn't invalidate anyone else's cached lines.
- `volatile` keyword — affects compiler register caching, not L2 hardware caching.

#### Authoring guidance, diagnostic harness, and "why this matters"

Operational rules for authoring kernels that touch the buffer classes above live in **§5.5** (Authoring Deterministic Multi-Agent Code — Ruleset). §5.5 is the single source of truth for: the allocation method decision tree (Rule 1), the antipatterns-at-a-glance lookup, the PR checklist + MTYPE audit code template (Rule 5), the "deterministic for N steps, then divergent" diagnostic shortcut (Rule 4), and the rationale ("Why these rules exist"). Do not duplicate that guidance here.

### 5.5 Authoring Deterministic Multi-Agent Code on gfx1100 — Ruleset

This section is a flat ruleset for agents writing or reviewing kernels that touch buffers shared between agents (multiple GPUs, host CPU, SDMA engine) on gfx1100. The mechanism is documented in §5.4; this section is the operational guide.

**Always read these rules in full before adding any cross-agent buffer to a persistent kernel. Skipping them has cost ~30 person-days of debugging over the past month.**

#### Antipatterns at-a-glance — fast PR-review lookup

When reviewing a "fix" for L2 staleness, scan against this table first. Any entry below is a non-fix; reject and route back to Rule 1.

| Proposed "fix" | Why it doesn't work | Right answer |
|---|---|---|
| Add `volatile` to the pointer/variable | Affects compiler register caching, not hardware caching | Rule 1 (UC, atomic-load, or kernel-boundary) |
| Add `__threadfence()` (agent scope) | Drains *writer*'s L2; doesn't invalidate any *reader*'s cache | Rule 1 |
| Add `__threadfence_system()` | **HANGS** on gfx1100 (see §5.3) | Rule 1 |
| Add `__syncthreads()` | Block-local barrier; no L2 effect | Rule 1 |
| Add `cg::grid_group::sync()` mid-kernel | Grid-wide fence inside one kernel; doesn't invalidate cached lines | Rule 1d (separate kernel launches) |
| Add `buffer_gl1_inv` on reader side | Invalidates L1 only, not L2; useful for L1 staleness but not the cross-agent class | Rule 1a (UC) or Rule 1c (atomic-load) |
| Add `s_waitcnt vmcnt(0) vscnt(0)` | Drains this wave's outstanding ops; doesn't touch peer cached state | Rule 1 |
| Read 64 MiB of scratch to "flush" L2 | Empirically unviable on gfx1100 (per-CU L0/L1 above L2 isn't displaced by L2 pressure; commit `96211a`) | Rule 1 |
| Convert buffer to UC, keep atomic ops on it | Atomics on UC are **undefined** (Rule 2). Silently miscompiled. | Remove atomics OR use Fine-grained allocation OR keep cached + Rule 1c/d |
| Restart the kernel | Works (kernel-boundary sync is Rule 1d) but throws out persistent-kernel perf | Rule 1d if you can; otherwise Rule 1a/b/c |
| "It works in tests" | Tests don't exercise multi-agent load; the bug is probabilistic | Rule 5 determinism test (10 byte-identical runs required) |
| `__hip_atomic_load` one sentinel word per 128B line then plain-read neighbors | Confirmed STALE in cross-kernel / persistent-kernel scenarios. See Q5 below: write ordering also required (payload must precede sentinel). In INTRA-KERNEL intra-launch scenarios only, `__threadfence()` from the writer provides enough coherence that plain-reads also work — but that scenario doesn't require atomic-loads at all. For cross-agent staleness (the class Rule 1c addresses), each word requiring freshness needs its own `__hip_atomic_load`. | Rule 1c per word, or Rule 1a/b (UC allocation) |

If the proposed fix isn't in this list, check that it maps to one of Rule 1's four sanctioned methods (a/b/c/d). If not, the fix is novel — and on gfx1100, novel L2-staleness fixes have ~0% historical success rate.

#### Allocation method ground truth (measured 2026-05-12, gfx1100 × 2)

`uc_verification_bench` (`test/gfx1100_microbench/uc_verification_bench.cpp`, results in `results/uc_verification.json`) exhaustively tested 5 allocation methods. The verdicts are:

| Method | MTYPE (local + peer) | Cross-agent observe (p50) | Cross-agent spin-wait | Verdict |
|---|---|---|---|---|
| `hipMalloc` | DEVICE 0x0 (cached) | TIMEOUT | **fails** | Default-cached: only safe with kernel-boundary sync between writer and reader. Never reuse across persistent-kernel iterations without invalidation. |
| `hipExtMallocWithFlags(hipDeviceMallocUncached)` | DEVICE 0x3 (UC) | 0.5 ms | works | **Canonical UC for device-resident buffers.** What `braidinfer::DeviceBuffer::alloc_uncached` produces. Preserves UC on peer side after `hipDeviceEnablePeerAccess`. |
| `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` | DEVICE 0x1 (FG) | 65 ms (130× slower) | works | Functionally correct but 130× slower than UC. Avoid unless atomics on the buffer are required (FG supports atomics, UC does not). |
| `hipHostMalloc` (default flags) → `hipHostGetDevicePointer` | HOST 0x2 | 0.77 ms | works | **Canonical host-mapped UC for CPU↔GPU mailboxes.** What `MappedHostBuffer` produces. |
| `hipHostMalloc(Coherent \| Portable)` → `hipHostGetDevicePointer` | HOST 0x40000001 | 0.77 ms | works | Same performance as default; use when explicit portability across processes is needed. |

#### Rule 1 — Cross-agent buffer? Pick exactly one of these allocation methods.

For any buffer that is written by one agent (peer GPU, host CPU, SDMA engine) and read by another, *during* a persistent kernel or across short-lived launches without an intervening kernel-boundary sync, use **exactly one** of:

(a) `hipExtMallocWithFlags(hipDeviceMallocUncached)` — device-resident UC. Bypasses L2. Use for device-side cross-GPU buffers (P2P writes between GPUs). The braidinfer `DeviceBuffer::alloc_uncached` wraps this.

(b) `hipHostMalloc` + `hipHostGetDevicePointer` (i.e. `MappedHostBuffer`) — host-mapped UC. Use for CPU↔GPU mailboxes, host-driven dispatch queues, or any host-side write target.

(c) Default `hipMalloc` + **`__hip_atomic_load(ptr, ACQUIRE, AGENT)`** at every read site — for single-word reads where the ~1-cycle atomic overhead is acceptable. The atomic load forces the L2 coherence machinery to re-fetch. **Each word requiring freshness needs its own atomic-load** — "atomic-load one sentinel then plain-read neighbors" is an antipattern (see Rule 3 antipattern table). Write ordering addendum (measured 2026-05-18, `atomic_load_cache_refresh_bench` Q5, gfx1100 N=10): if the writer writes sentinel FIRST then payload, the reader's atomic-load on sentinel observes stale payload in 9/10 trials — the cache-line refill only captures words written to L2 before the atomic-load fires. Writers must therefore write payload words before the trigger/sentinel word (`__threadfence()` between each write).

(d) Default `hipMalloc` + **kernel-boundary sync** (`hipDeviceSynchronize`, `hipStreamSynchronize`, `hipEventRecord`+`hipStreamWaitEvent`) between every writer kernel and every reader kernel — the AGENT-scope release at kernel exit drains L2 to HBM; the next kernel launch reads from HBM with cold L2. **Not viable inside a persistent cooperative kernel; only for separate-launch designs.**

**Empirical envelope (2026-05-13, udi+braidinfer joint).** Rule 1a (peer-VRAM UC) and Rule 1b (host-mapped UC) both pass the original `cg` bench at 2 GPUs, but the wider hazard surface from §11.4 means the two paths diverge at 4 GPUs:

- **Rule 1b** (host-mapped UC, `MappedHostBuffer`): empirically holds at 4 GPU under multi-GPU PCIe pressure when the call site does not include a SYSTEM-scope UC write preceding a barrier (§11.4). V0 minimal reproducer (4 GPU + persistent + post-poll barrier + host-mapped polling) passed 10/10 trials.
- **Rule 1a** (peer-VRAM UC, `DeviceBuffer::alloc_uncached` / `hipExtMallocWithFlags(hipDeviceMallocUncached)`): exposed at 4 GPU even with AGENT-scope writes. V7 reproducer (4 GPU + persistent + cross-GPU peer-UC store + post-poll barrier) wedges at 30% over n=10. The hazard surface of Rule 1a is therefore **wider than SYSTEM-scope alone**: AGENT-scope peer-VRAM UC writes under cross-GPU PCIe pressure are also exposed. Mitigations: deferred-write primitive (`rdna3_peer_write_deferred` in `braidinfer/kernels/rdna3/rdna3_peer.h`), or kernel-boundary sync (Rule 1d). See §11.4 for mechanism details.

Other combinations are **NOT SAFE on gfx1100** for cross-agent buffers. Specifically: `hipMalloc` + plain (non-atomic) reads + persistent kernel + multi-agent writer = the canonical L2-staleness bug.

#### Rule 2 — Hardware atomics on UC memory are UNDEFINED.

This is the most-recently-discovered foot-gun (cost: ~2 days on the braidinfer `normed_stage` investigation).

`hipExtMallocWithFlags(hipDeviceMallocUncached)` produces MTYPE=UC memory. **Hardware atomics (`atomicAdd`, `atomicCAS`, `atomicExch`, `__hip_atomic_*`) through pointers into UC memory have undefined behavior**, per AMD's own documentation and confirmed by braidinfer's `DeviceBuffer::alloc_uncached` docstring (`memory.rs:92`).

If a buffer is currently allocated `hipMalloc` and is read/written via atomics anywhere, you cannot convert it to UC without removing those atomic operations. Three options when this constraint binds:

- **Plain-load + UC**: change the atomic reads to plain (volatile) reads. Lose atomic-RMW semantics, gain L2-bypass.
- **Fine-grained allocation**: use `hipExtMallocWithFlags(hipDeviceMallocFinegrained)`. Supports atomics. 130× slower cross-agent observe than UC; only viable if not on the hot path.
- **Keep cached, change architecture**: switch to kernel-boundary sync (Rule 1d) or accept the bug.

When promoting an existing buffer to UC, grep the entire codebase for atomic operations on it. *All* must be removed or moved to a separate cached-and-atomic-only buffer.

#### Rule 3 — Do not use these mechanisms as L2-staleness fixes.

The following **DO NOT** invalidate L2 on gfx1100. Using them as a "fix" produces a kernel that works in casual testing and breaks under multi-agent load:

- `volatile` keyword — affects compiler register caching, not hardware L2.
- `__threadfence()` (agent scope) — drains this block's outstanding stores to L2. Does not invalidate any reader's L2.
- `__threadfence_system()` — **hangs** on gfx1100 (see §5.3).
- `__syncthreads()` — block-local barrier only.
- `cg::grid_group::sync()` — grid-wide release/acquire fences inside one cooperative kernel. Does not invalidate L2 lines that are mid-cycle.
- `buffer_gl1_inv` — invalidates L1 only. Useful for L1 staleness (different CUs caching different views of same line during a single kernel). Does NOT fix L2.
- `s_waitcnt vmcnt(0) vscnt(0)` — drains this wave's outstanding ops only.
- Eviction by working-set pressure — **EMPIRICALLY UNVIABLE** on gfx1100 (commits `4c5bd18` → control `0cfdb93` → redesign `96211a` proved this definitively). The redesigned `l2_evict_bench_v2` (persistent-kernel UC mailbox, no `hipSetDevice` during measurement) swept 14 scratch sizes × 4 access patterns × 1000 trials, going up to 64 MiB scratch (11× L2 capacity). Result: 0% fresh at every cell. Even 64 MiB of L2 pressure cannot invalidate a single 128B cache line that the probe is actively reading. **Mechanism**: the single-thread probe's repeated reads keep the target line in **per-CU L0/L1 caches above L2**, which scratch traffic through L2 does not displace. Eviction-by-pressure can never be a sanctioned primitive on this hardware. Use Rule 1 methods (UC or atomic-load) for cross-agent staleness.

If you find a code review where the proposed fix uses any item in this list as the L2-staleness mitigation, reject and route back to Rule 1.

#### Rule 4 — Diagnostic shortcut: "deterministic for N steps, then divergent."

This pattern is the L2-staleness signature, observed three times in two months (braidinfer 5ax prefill, 5ax decode, 4fg shutdown wedge). When you see it:

1. Find the buffer with a reuse-period that matches N (pool size, chunk size, expert rotation, ring buffer slot count, decode-step parity).
2. Identify the cross-agent write/read direction.
3. Verify the buffer's MTYPE via `hipPointerGetAttributes`. If 0x0 (default cached), it's the suspect.
4. Apply Rule 1 mitigation matching the access pattern. **Verify with `hipPointerGetAttributes` again post-conversion** that BOTH local and peer sides report the expected MTYPE.
5. Determinism test: 10 runs same input, byte-identical output. Required for sign-off.

If step 5 fails after MTYPE conversion: check Rule 2 (atomic-on-UC undefined behavior).

#### Rule 5 — Required authoring checklist (paste into PR description)

For any new kernel that touches a buffer shared with another agent OR reused across iterations:

- [ ] **List every cross-agent buffer this kernel reads or writes.**
- [ ] **For each, allocation method picked per Rule 1.** (a/b/c/d explicit.)
- [ ] **`hipPointerGetAttributes` query** confirms the expected MTYPE on **both local AND peer side** after `hipDeviceEnablePeerAccess`.
- [ ] **For UC buffers**: grep'd the codebase for atomic operations on them — none remain (Rule 2).
- [ ] **No Rule 3 anti-patterns present.** No `volatile`-as-fix, no `__threadfence_system`, no `s_waitcnt` magic.
- [ ] **Determinism test**: 10 runs, same input, byte-identical output. PR is **blocked** until this passes.

**MTYPE audit template** — paste this into a startup-time diagnostic (or run as a one-shot bench). Walks every allocation, queries local + per-peer MTYPE, flags any cross-agent-eligible buffer that is plain `hipMalloc` cached. Reference impl: `braidinfer` commit `c0cb6bb`.

```cpp
struct AllocRecord { const char* name; void* ptr; size_t bytes; };
static std::vector<AllocRecord> g_allocs;  // populate at every alloc site

void audit_mtype_for_cross_agent_buffers(int n_devices) {
    const char* mtype_name[] = {"UNREG","HOST","DEVICE","MANAGED","ARRAY","UNIFIED"};
    int saved; hipGetDevice(&saved);
    for (const auto& a : g_allocs) {
        hipPointerAttribute_t at{};
        hipError_t e = hipPointerGetAttributes(&at, a.ptr);
        if (e != hipSuccess) continue;
        printf("%-40s dev=%d mtype=%s flags=0x%x  ",
               a.name, at.device, mtype_name[at.memoryType], at.allocationFlags);
        // Query from each peer device (only meaningful if peer access enabled)
        for (int p = 0; p < n_devices; ++p) {
            if (p == at.device) continue;
            hipSetDevice(p);
            hipPointerAttribute_t pa{};
            if (hipPointerGetAttributes(&pa, a.ptr) == hipSuccess) {
                printf(" peer%d=%s/0x%x", p, mtype_name[pa.memoryType], pa.allocationFlags);
            }
        }
        printf("\n");
    }
    hipSetDevice(saved);
}
```

What to flag in the output:
- `mtype=DEVICE flags=0x0` (cached `hipMalloc`) on a cross-agent buffer → bug. Pick Rule 1 (a-d).
- `mtype=DEVICE flags=0x3` (UC) — correct for cross-GPU; verify no atomics (Rule 2).
- `mtype=HOST flags=0x2` or `0x40000001` — correct for host-mapped UC; CPU↔GPU mailboxes.
- Peer side reports different MTYPE than local — bug in `hipDeviceEnablePeerAccess` or in allocation flags; investigate.

Audit must cover all four buffer classes per Rule 8 (not just activation flow).

#### Rule 6 — Performance comparison

When choosing between Rule 1 options, the measured costs are:

| Mechanism | Per-event cost | Bandwidth penalty | Atomics supported |
|---|---|---|---|
| Rule 1a (UC device) | 0 (per-event) | ~2-4× permanent on reads through UC | NO (undefined per Rule 2) |
| Rule 1b (host-mapped UC) | 0 (per-event) | PCIe-bandwidth-limited (same as UC) | NO |
| Rule 1c (cached + atomic-load) | ~1 cycle per read | 0 | YES |
| Rule 1d (kernel-boundary sync) | 44–224 µs per launch (HIP); ~5 µs (direct doorbell) | 0 | YES |
| Eviction-by-pressure | **UNVIABLE** (proven 96211a) | n/a | n/a |
| Host-mediated SDMA invalidation | ~1-2 µs round-trip via `hipMemcpyAsync(HostToDevice)` | 0 | YES (sort of — see footnote) |

**On eviction-by-pressure**: closed empirically. See Rule 3 entry above for the proof. Don't propose it as a fix in code review.

**On host-mediated SDMA invalidation** (newly characterized by `l2_evict_bench_v2`): a small `hipMemcpyAsync(target_addr, &dummy, sizeof(uint32_t), hipMemcpyHostToDevice, stream)` from host triggers an invalidation of `target_addr`'s line in the target GPU's cache hierarchy. The probe in `l2_evict_bench_v2` measured B (the host-SDMA-written control) as 100% fresh across every cell, while A (the peer-P2P-written target) was 0% fresh — proving that the PCIe coherent path used by `hipMemcpyAsync(HostToDevice)` triggers invalidation that peer-P2P-store does not. This is a real flush primitive but with caveats:
- Host involvement per event — not in-kernel, requires returning to CPU.
- ~1-2 µs per call (PCIe round-trip + SDMA setup).
- Only useful at kernel-launch boundaries; useless inside a persistent kernel.
- Generally inferior to Rule 1a/b (UC allocation) which has zero per-event cost. Worth knowing exists, rarely worth using.

Until a future arch (gfx12+) restores `buffer_gl2_inv` to the ISA, Rule 1a–d are the available primitives. There is no in-kernel L2-flush on gfx1100.

#### Rule 7 — When converting a cached buffer to UC and it "broke things differently"

This is the most common symptom of Rule 2 (atomic-on-UC undefined). Diagnostic:

1. `git grep '__hip_atomic_\|atomicAdd\|atomicCAS\|atomicExch' -- <files touching the buffer>`
2. If hits: those are the broken atomics. Remove or move to a parallel cached-atomic buffer.
3. If no hits: check if the buffer was actually `MappedHostBuffer` (host-mapped) already — converting that to device UC may have introduced a different memory type than the original.
4. Verify post-conversion MTYPE via `hipPointerGetAttributes` matches what you intended.

Cost of skipping this audit before declaring "UC doesn't work": ~1 day per occurrence (observed twice).

#### Rule 8 — Audit ALL persistent buffers, not just activation flow

When auditing cross-agent buffers for L2 staleness, the audit MUST include all four classes below. Restricting the audit to "the obvious cross-agent activation flow" (the most common pattern) is the cause of multiple multi-day debug sessions where the bug was in a buffer no one thought to check.

Classes to audit:

(a) **Activation flow buffers** — per-token forward-pass intermediates that workers and GPU 0 exchange. The obvious case; usually caught by manual review.

(b) **KV-cache / paged-attention buffers** — written during prefill, read during decode. The prefill→decode kernel boundary is the staleness window. If any worker reads peer-mapped KV via P2P, the cached-buffer-with-cross-agent-read pattern applies.

(c) **Weight buffers with peer access enabled** — model weights are written-once at load time. They look "read-only" so the audit usually skips them. But if `hipDeviceEnablePeerAccess` enables them for peer reads, AND the peer's L2 caches them at first read, AND any model-load path runs after a peer's L2 already has stale-or-zero contents for those addresses, the peer reads garbage. "Read-only after init" does NOT exempt a buffer from coherence audit; staleness is about cache state, not content mutability.

(d) **State buffers reused across kernel boundaries** — GDN/Mamba recurrent state, expert routing tables, scratch buffers that persist across decode steps. Any cached buffer whose lifetime spans a kernel-launch-boundary OR a persistent-kernel iteration is a candidate.

Mechanical audit primitive: `braidinfer` commit `c0cb6bb` ships a `hipPointerGetAttributes` programmatic audit that enumerates every `DeviceBuffer` allocation and prints local + peer MTYPE. Extend this primitive to cover (b), (c), (d) — not just the activation flow.

Required at PR time (added to Rule 5 checklist): the MTYPE audit run must dump every cross-kernel-boundary buffer's MTYPE, not just the obvious ones.

#### Rule 9 — One cooperative-grid kernel per process lifetime

**Statement.** Second-or-later `hipLaunchCooperativeKernel` call within a Linux process wedges on its first dispatch on gfx11+ (gfx1100 confirmed; framework community reports across gfx1103/1150/1152 consistent). The wedge clears only on process exit.

**Status as of 2026-05-14 (final): CONFIRMED with mechanism unknown.** Rule 9's empirical basis was the V0 skeleton's "trial 1 PASS, trial 2+ WEDGE" pattern across in-process relaunches. The PROVISIONAL caveat that briefly occupied this slot — concerned that V0 might have shipped the same deferred-ack pattern as production — was resolved: V0's `persistent_worker_skeleton.hip:205-213` uses **immediate-ack** (writes `ack=seq` in the same iteration that processed `seq`), so V0's trial-2-wedge cannot be the §11.15 protocol deadlock. The relaunch wedge is a real, independent phenomenon. Mechanism remains unidentified; six MES-side kernel patches in `0011-drm-amdgpu-kfd-add-reset-cooperative-state-ioctl.patch` (Designs D, F, G, H, I) all failed to clear it mid-process. Process exit is the only known recovery path. Framework community reports across gfx1103/1150/1152 are consistent with the symptom.

**Operational rule.** Design persistent-kernel architectures so that exactly one cooperative-grid kernel is launched per process lifetime. Route all subsequent dispatches through that one kernel via doorbell + opcode-mux. Do not tear down and re-launch the persistent kernel within the same process.

**Production-pattern lookup.**

| Pattern | Verdict |
|---|---|
| Single `hipLaunchCooperativeKernel` at init; all dispatches handled via doorbell-poll inside that one kernel | OK |
| Per-batch cooperative-kernel relaunch (e.g. prefill launches `megakernel_f32` cooperative, then decode launches `persistent_worker` cooperative) | WEDGES on second launch's first dispatch |
| Prefill via non-cooperative `hipLaunchKernelGGL` + decode via single cooperative `persistent_worker` | OK |
| Fork+exec child for cooperative work; parent holds model weights | OK (each child gets fresh PASID) but `exec` overhead ≈ 337 ms; reserve for recovery |

**Diagnosis shortcut.** Symptom: GPU never observes a host-mapped UC poll's volatile write while CPU readback confirms the write landed; pattern is `trial 1 PASS, trial 2+ WEDGE`. Cause is almost always Rule 9 — search the call graph for a second `hipLaunchCooperativeKernel` in the same process.

**Empirical envelope and exhaustive rule-outs** documented in §11.13 below (under re-investigation after §11.15). The negative-probe archive is still useful — do not redo those probes. The conclusion attributing those probes' falsifications to Rule 9 specifically is what is under re-test.

**Mechanism (not fully understood, irrelevant to the rule).** Investigation through six kernel-side patches found no MES-side per-process clear that fires from userspace ioctls (NOTIFY_TO_UNMAP_PROCESSES, SET_SHADER_DEBUGGER, ADD_QUEUE-with-skip_proc_clear=0 single & multi-queue, REMOVE-all+ADD-all reaching spec p26 "last gang in process", per-PASID TLB flush). Process exit clears via amdgpu's hung-queue detection path; no safe mid-process equivalent exists. The only fix is architectural — no kernel patch (locally drafted or upstream) recovers from a second cooperative-grid launch.

**Why this rule exists (revised 2026-05-14).** The production wedge braidinfer hit on 2026-05-14 was originally attributed to Rule 9 (per-token cooperative `mk.execute()` in `prefill_paged` making `persistent_worker` the (P+1)-th launch). That attribution was wrong; the production wedge was a Phase 2' deferred-ack protocol deadlock inside `persistent_worker` itself, reproducible in a fresh-process standalone fixture with zero prior cooperative launches. See §11.15 for the actual fix. Rule 9 itself is supported by independent evidence: V0 skeleton (`persistent_worker_skeleton.hip:205-213`, immediate-ack) reliably wedges on trial 2+ of a cross-launch baseline in same process. Mechanism unknown after six MES-side kernel patch attempts (§11.13 archive). The framework-community symptom across gfx11+ matches.

#### Why these rules exist

Three bug investigations totaling ~30 person-days (braidinfer 5ax prefill, 5ax decode, 4fg shutdown wedge) all reduced to "cached buffer reused without proper invalidation; the diagnostic localized to the wrong layer because the bug was in a buffer that the failing op reads, not in the op itself." The rules above are the operational checklist that would have caught all three at code review time.

The cost of *not* having this section ≈ 1 person-week per new cross-agent kernel that gets it wrong. The cost of *having* it is ~5 minutes of review per PR. Asymmetric.

---

## 6. Kernel Design Patterns for gfx1100 in Composable Kernel

### 6.1 Canonical WMMA Instance Template

Use Sprint 1 Task 1 target shapes as a starting point. Example new instance:

```cpp
// library/src/tensor_operation_instance/gpu/gemm_universal/...
using DeviceGemm_Wmma_256x128x32_F16_Gfx1100 = DeviceGemm_Wmma_CShuffleV3<
    Row, Col, Row,          // Layouts
    F16, F16, F16, float, F16,      // Types
    PassThrough, PassThrough, PassThrough,
    GemmSpecialization::MNKPadding,
    256,                    // BlockSize (threads)
    256, 128, 32,           // Block tile
    8, 8,                   // AK1, BK1 (enforce 128-bit loads)
    16, 16,                 // WMMA tile
    2, 2,                   // MRepeat/NRepeat
    S<8, 32, 1>, S<0, 1, 2>, S<0, 1, 2>, 1, 8, 8, false,
    S<8, 32, 1>, S<0, 1, 2>, S<0, 1, 2>, 1, 8, 8, false,
    1, 1, S<1, 32, 1, 8>, 8,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3>;
```

**Guidelines**
1. Keep `BlockSize=256` to line up with `CK_TILE_MAX_THREAD_PER_BLOCK`.
2. Choose `MPerBlock`/`NPerBlock` multiples of 128; `KPerBlock` multiples of 32 for WMMA tiles.
3. Increase `MRepeat/NRepeat` only if VGPR budget allows; each repeat consumes 16 additional accumulator registers.

### 6.2 Recommended Block Shapes

| Workload | Block `(M,N,K)` | Notes |
|----------|-----------------|-------|
| Large square GEMM (M,N ≥ 2048) | 256×128×32 | Balances L2 reuse vs occupancy. |
| Attention (seq ≥ 512, hidden ≥ 1024) | 128×256×32 | Aligns with `QKᵀ` tall matrices. |
| Batched small (batch ≥ 32, dims ≤ 1024) | 512×64×64 | Improves batch-level reuse if `SplitK` engaged. |
| Transposed attention | 64×512×64 | Prioritize column reuse on B operand. |
| Balanced fusion | 128×128×128 | Good default for unknown shapes. |

Populate both FP16 and BF16 variants, with INT8 where accumulator accuracy permits.

### 6.3 Adaptive Pipeline Scheduler
Sprint 2 introduces a gfx1100-specific scheduler, e.g. (`include/.../gfx1100_adaptive_scheduler.hpp` in the sprint spec):

```cpp
template <index_t M, index_t N, index_t K>
struct Gfx1100AdaptiveScheduler {
    static constexpr auto scheduler =
        (M <= 128 && N >= 1024) ? BlockGemmPipelineScheduler::Intrawave :
        ((K >= 256 && M >= 512 && N >= 512) ? BlockGemmPipelineScheduler::Interwave :
                                              BlockGemmPipelineScheduler::Intrawave);
    static constexpr auto version =
        (K >= max(M, N)) ? BlockGemmPipelineVersion::v3 : BlockGemmPipelineVersion::v1;
};
```

**Usage**
- Pass the `scheduler` and `version` values into the instance type alias so each problem size picks the right pipeline at compile time.
- Interwave scheduling is beneficial when `K` is large and `SplitK` is disabled; it lets two waves share LDS double buffers efficiently on gfx1100.

### 6.4 Memory Access Upgrades
Sprint 1 Task 2 proposes 128-bit vector loads:

```cpp
#if defined(__gfx11__)
constexpr index_t VectorLoadBits = 128;
using ABlockTransfer = BlockwiseTensorSliceTransfer<
    ADataType,
    AccDataType,
    ADesc,
    ABlockDesc,
    ck::Sequence<VectorLoadBits / 16, 1, 1>,
    ...>;
#endif
```

**Apply for both A and B paths**. The cache line is 128 bytes, so aggregate 8 FP16 values (16 bytes) per thread per instruction to reduce the number of transactions by 4× compared to scalar loads.

### 6.5 LDS Swizzling
- Use `S<1,32,1,8>`-style shuffles (`CShuffleBlockTransferClusterLengths`) to keep LDS writes contiguous in memory order but strided in thread order, minimizing bank conflicts.
- When writing result tiles, set `CShuffleBlockTransferScalarPerVector_NPerBlock = 8` for FP16 to maintain 128-bit stores.

### 6.6 Handling Register Pressure
- Break `MRepeat`/`NRepeat` loops into software-pipelined segments and opportunistically dual-issue WMMA + scalar ops. Radeon GPU Profiler’s VGPR/SGPR counters show when a kernel exceeds the VGPR budget per SIMD.

```cpp
auto frag0 = blockwise_gemm_pipeline.GetCThreadBuffer();
auto frag1 = blockwise_gemm_pipeline.GetCThreadBuffer();
bool toggle = false;
pipeline.template Run<HasMainKBlockLoop>([&](auto)
{
    auto& frag = toggle ? frag1 : frag0;
    blockwise_gemm_pipeline.ComputeStep(frag);
    toggle = !toggle;
});
```

This keeps fragments live in two buffers, enabling compiler scheduling freedom without exceeding allowed VGPRs per wave.

### 6.7 Chiplet, Infinity Cache, and Memory Guidance
- gfx1100 (Navi 31) separates compute (Graphics Compute Die) and memory/cache (Memory Cache Dies). Infinity Cache (up to 96 MB) sits on the MCDs and feeds a GDDR6 bus up to 384-bit/20 Gbps. Kernels that stream large tiles should balance LDS tiling with Infinity Cache reuse; CK’s block sizes above 128×128 often keep data within L2 + Infinity Cache for a single WGP.
- For memory coalescing, prefer `buffer_load_dwordx4`/`x8` (CK vector loads) and align *data* base pointers for 128 B cache lines (reported by `rocminfo` on RX 7900 XTX). Buffer resource descriptors (SRDs) themselves are 128-bit values (16 B) in SGPRs; the ISA-supported knob for Infinity Cache residency is the `DLC` temporal hint (non-temporal vs regular), not SRD byte alignment. Use Radeon Memory Visualizer to confirm cache residency when evaluating Sprint 2 memory optimizations.

### 6.8 WMMA Backend Selection on gfx1100
- On `gfx1100`, prefer CK’s WMMA backend (`DeviceGemm_Wmma_CShuffleV3`) for FP16/BF16/INT8/INT4. Ensure instance registration (`ck::is_gfx11_supported`) routes `gfx1100` workloads into WMMA paths.

### 6.9 INT4 (pk_i4_t) Implementation Details
- CK represents packed 4-bit signed integers via `ck::pk_i4_t` (see `include/ck/utility/data_type.hpp`) and vector aliases (`pk_i4x2_t`, `pk_i4x4_t`, `pk_i4x8_t`). Conversion helpers (`ck::type_convert`) and buffer-access utilities already handle nibble packing/unpacking.
- WMMA instance examples live under `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_i4_f16_*` and `...bf16_i4_bf16_*`. Use these as templates when expanding coverage: keep A/C in FP16/BF16, B in `pk_i4_t`, accumulators in FP32, and include permute logic from `tile_engine/ops/gemm/gemm_profiler.hpp` if running profiler paths.
- Validation: leverage `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`, which already handles pk_i4_t inputs, and add INT4-specific cases to the regression harness (see §7.4).
- Performance tips: INT4 kernels typically spend extra VALU cycles unpacking. Track `SQ_INSTS_VALU`, `SQ_INSTS_VMEM`, and `SQ_WAVES` plus VGPR counts to ensure the WMMA inner loops still dominate runtime, and compare against the FP16/INT8 baselines in `gfx1100_baseline.json`.
- Build & test: keep `CK_ENABLE_INT8=ON` (default) so integer GEMM instances build, enable `USE_BITINT_EXTENSION_INT4=ON` (default OFF) when compiling with AMD clang ≥ ROCm 6.1 to unlock the bit-int converters (root `CMakeLists.txt:317`), and run `ctest -R test_pk_i4`, `ctest -R test_gemm_universal_wmma_fp16`, and `ctest -R test_gemm_universal_wmma_bf16` to exercise pk_i4_t kernels and host conversions.

---

## 7. Testing, Profiling & Validation

### 7.1 Build Configurations
```bash
# Baseline build with all gfx1100 targets (Sprint 1/2)
cmake -D GPU_TARGETS="gfx1100" \
      -D CMAKE_BUILD_TYPE=Release \
      -D CK_ENABLE_FP16=ON \
      -D CK_ENABLE_BF16=ON \
      -D CK_ENABLE_INT8=ON \
      ..
make -j$(nproc) test_gemm_wmma_f16
```

For advanced sprints:
```bash
cmake -D GPU_TARGETS="gfx1100" \
      -D CMAKE_BUILD_TYPE=Release \
      -D CK_ENABLE_GFX1100_OPTIMIZATIONS=ON \
      -D CK_ENABLE_ADVANCED_OPTIMIZATIONS=ON \
      -D CK_ENABLE_ATTENTION_OPTIMIZATIONS=ON \
      ..
```

### 7.2 Baseline Test Harness
Create `test/gemm_performance/gemm_wmma_gfx1100_baseline.cpp` per Sprint 1 Task 4.1 with canonical problem sizes. Couple with a shell runner:

```bash
# tools/test_gfx1100_performance.sh
#!/usr/bin/env bash
set -euo pipefail
sizes=("512,512,512" "1024,1024,1024" "4096,4096,4096" "4096,512,1024")
for s in "${sizes[@]}"; do
  ./bin/test_gemm_wmma_gfx1100 --size="$s" --benchmark
done
```

### 7.3 Profiling Workflow
1. **Pipeline profiling** (`tools/profile_gfx1100_pipeline.sh`) – sweeps scheduler/version combos under `HIP_LAUNCH_BLOCKING=1` for deterministic timing.
2. **Hardware counters** – on this ROCm 7.1.1 install, `rocprof` (RPL) and `rocprofv2` work for counter collection. `rocprofv3` requires a workaround for a YAML symbol lookup error:
   ```bash
   # Fix for rocprofv3 YAML symbol error on Arch Linux
   export ROCPROF_PRELOAD=/usr/lib/libyaml-cpp.so
   rocprofv3 --kernel-trace -o /tmp/trace -- ./your_binary
   ```
   Discover counter names with `rocprof --list-basic` / `rocprof --list-derived` or `rocprofv2 --list-counters`.
   - `rocprof` example (counters via `pmc.txt` groups):
     ```bash
     cat > pmc.txt <<'EOF'
     pmc : SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM
     pmc : GL2C_HIT GL2C_MISS
     pmc : TA_TA_BUSY
     EOF
     rocprof -i pmc.txt --stats -o profile.csv ./bin/profile_gemm_wmma --scheduler=Intrawave --version=v3 --size=2048,2048,2048
     ```
   - Trace alternative: use **RGP** (Radeon Developer Panel) for wave/timing traces on RDNA3, or try `rocprofv2` tracing options if available on your setup.
3. **Radeon Memory Visualizer** – inspect Infinity Cache vs GDDR6 usage for the new block sizes; target ≥60% of theoretical bandwidth.
4. **Roofline analysis** – use `./bin/roofline_analysis_gfx1100 --detailed` (updated to ingest the corrected peak numbers) to ensure kernels approach realistic FP16/FP32 limits.

### 7.4 Regression & Correctness Detection
- Maintain `gfx1100_baseline.json` with runtime+throughput targets for FP16/BF16/INT8/INT4 WMMA kernels, attention shapes, and convolution workloads.
- Add correctness hooks that compare WMMA output against CK reference kernels for FP16/BF16/INT8 and validate Split-K accumulation matches tolerances. Use CK’s `check_err` utilities with per-dtype tolerances.
- Integrate regression guard + correctness checks into CI, failing any run where throughput falls below (target × 0.9) or accuracy exceeds tolerance for any dtype.

### 7.5 Tooling Beyond ROCm CLI
- **Radeon GPU Profiler (RGP)** – visualizes wave occupancy, cache behavior, dual-issue stats, and helps correlate scheduling decisions with hardware counters. Enable RGP captures via Radeon Developer Panel or `RADV_PERFTEST=rgp`, collect a trace while running `./bin/profile_gemm_wmma`, and inspect the “Wave Occupancy” and “Instruction Timing” panes to validate §3.2’s occupancy math.
- **Radeon GPU Analyzer (RGA)** – compile kernels with gfx1100 target to inspect ISA, VGPR usage, and confirm WMMA intrinsics lower to expected instructions.
- **Radeon Memory Visualizer (RMV)** – analyze Infinity Cache hits/misses and GDDR6 bandwidth when testing memory-optimization tasks.
- **GPUPerfAPI** – scripted counter collection for automated regression dashboards.

### 7.6 On-Hardware Microbench Suite (This Repo)
`test/gfx1100_microbench/` contains standalone HIP microbenches used to validate low-level assumptions on actual gfx1100 hardware.

Run:
```bash
test/gfx1100_microbench/run_all.sh
```

Focus runners:
```bash
test/gfx1100_microbench/run_focusA.sh
test/gfx1100_microbench/run_focusB.sh
```

**Highlights observed on this system (RX 7900 XTX / gfx1100):**
- **Cache hints materially change outcomes**:
  - `raw_load_hint_bench` shows `glc=1`/`slc=1` can improve streaming-read bandwidth, while `dlc=1` can sharply reduce it for reuse-friendly reads.
  - `dlc_pollution_bench` shows `dlc=1` can reduce hot-set pollution at the cost of slower streaming.
- **Alignment matters much more for stores than for loads**:
  - `raw_store_hint_bench` shows a ~2× bandwidth regression for 16B stores with a 1–16B misalignment offset vs offset=0.
  - `raw_copy_hint_bench` shows misaligning either src or dst reduces copy bandwidth, and `dlc-store=1` can partially mitigate misaligned copies.
- **Paged KV behavior depends on working-set size**:
  - `paged_kv_gather_bench` shows Focus A-like (≈84 MiB/iter) strongly prefers `dlc=0`.
  - In this *pure gather* microbench, Focus B-like (≈268 MiB/iter) can show a slight `dlc=1` advantage, but `attention_tile_bench` (a more attention-like fused tile) still preferred `dlc=0` for Focus B-like on this system.
  - Concurrency pattern matters: rotating the page index per-CTA can improve throughput for Focus B-like settings with `dlc=0` in some kernels/microbenches.
  - `paged_kv_batch_gather_bench` shows two additional real-world stressors:
    - breaking within-page contiguity reduces throughput materially (≈10–25% in tested cases)
    - increasing effective streaming footprint (e.g., high batch) can make `dlc=1` more attractive for *pure gather* stages (but re-check fused tiles)
  - `paged_kv_scatter_bench` shows KV writes are a different regime (lower bandwidth and different `dlc` behavior); treat writes and reads as separate knobs.
- **Descriptor-table-like loads are cheap and stable**:
  - `srd_table_load_bench` (uniform scalar 16B loads; compiled to `s_load_b128`) showed little sensitivity to 0–15B base misalignment or 16B–1024B table stride in a steady-state loop.
- **Attention-shaped primitives show the same themes**:
  - `softmax_like_bench`: warp vs LDS-tree reductions were close (often within a few percent) for the tested shapes.
  - `qk_dot_gqa_bench`: Focus A-like prefers `dlc=0`; Focus B-like behavior depends on both `dlc` and page-order concurrency (e.g., `rotate_step=1` can improve `dlc=0` while making `dlc=1` worse).
  - `attention_tile_bench`: for both Focus A-like and Focus B-like settings, `dlc=0` beat `dlc=1` in a fused tile-ish pattern (paged K+V + softmax + PV-like accumulate); `glc=1` was sometimes a small extra win on top of `dlc=0`.
- **WMMA hazards are best handled by intrinsics**:
  - `wmma_spacing_bench` did not show a benefit from manually inserting `v_nop`; fixed NOPs just add overhead on this system/toolchain.
- **Page-stride patterns are extremely costly**:
  - `stride_load_bench` shows 4 KiB-stride access over a large buffer yields very low effective bandwidth compared to contiguous reads.
- **LDS layout and barriers are measurable costs**:
  - `lds_bank_conflict_bench` shows large slowdowns for certain regular per-thread strides (notably 128–512 B in the tested pattern).
  - `barrier_cost_bench` shows `__syncthreads()` cost increases with block size (more waves).
- **Atomics are extremely contention-sensitive**:
  - `atomic_contention_bench`: global atomics with one hot counter were ~3 Gops/s, while LDS atomics were ~74 Gops/s; spreading across more counters improved global atomics significantly (tens to hundreds of Gops/s) but LDS remained far higher.
- **Lane-replication shuffles aren’t free**:
  - `lane_replication_shuffle_bench` saw no win from “half-load + `ds_bpermute`” vs “full-load” in a synthetic scenario; only do this if profiling shows global memory transactions actually drop.

---

## 8. Tooling Tips

| Tool | Purpose | Command |
|------|---------|---------|
| `rocminfo` | Verify GPU SKU exposes gfx1100. | `rocminfo | rg gfx1100` |
| `hipcc` | Compile WMMA kernels with proper arch flags. | `hipcc --offload-arch=gfx1100 -O3 kernel.cpp` |
| `amdclang++` | Build CK with aggressive LLVM passes (see `src/build/build.ninja`). | `amdclang++ -mllvm -amdgpu-early-inline-all=1 ...` |
| `rocprof --list-basic` | List basic HW counters available on this ROCm install. | `rocprof --list-basic | rg SQ_WAVES` |
| `rocprof -i pmc.txt --stats` | Collect HW counters (RPL format). | `rocprof -i pmc.txt --stats -o out.csv ./bin/profile_gemm_wmma ...` |
| `rocprofv2 --list-counters` | List counters for `rocprofv2` (availability varies by ROCm install). | `rocprofv2 --list-counters | rg SQ_WAVES` |
| `ck_info` | Confirm install exposes gfx1100 kernels. | `./bin/ck_info --gpu-arch=gfx1100` |

Note: `rocprofv3` requires `ROCPROF_PRELOAD=/usr/lib/libyaml-cpp.so` on Arch Linux to fix a YAML symbol error (see §7.3). Alternatively, use `rocprof`/`rocprofv2` plus RGP/RMV.

---

## 9. Implementation Checklist

1. **Instance coverage** – Implement at least the eight block shapes from Sprint 1 for FP16/BF16/INT8. Register them via `library/src/tensor_operation_instance/gpu/CMakeLists.txt`.
2. **Memory optimizations** – Enable 128-bit vector loads/stores and LDS swizzles. Validate with RGP/RMV and rocprofiler counters (tool support varies by ROCm version).
3. **Wave optimizations** – Default to Wave32 WMMA; understand the WMMA lane-replication requirement (§4.2) and use Wave64 only when you can justify the occupancy and toolchain tradeoffs.
4. **Scheduler tuning** – Hook `Gfx1100AdaptiveScheduler` into the instance factory (`DeviceGemm_Wmma_CShuffleV3` template parameters).
5. **Testing** – Maintain baseline/performance harnesses and integrate roofline + regression scripts into CI.
6. **Documentation** – Update sprint deliverables referencing this file whenever new optimizations land.

By following these guidelines, implementors should have the architectural context and concrete CK integration steps needed to iterate safely and reach high performance on gfx1100.

---

## 10. Composable Kernel vs rocWMMA — see `CK_vs_rocWMMA.md`

The architectural comparison and benchmarking framework for CK WMMA against AMD's rocWMMA was extracted to a sibling document on 2026-05-12. It is self-contained and covers: design philosophy (10.1), technical capabilities matrix including WMMA intrinsic coverage / tile flexibility / memory hierarchy / register management (10.2), validation and correctness testing (10.3), the performance benchmarking framework with problem-size matrix, metrics collection, harness structure, and profiling comparison (10.4), the cross-validation test harness template (10.5), performance-notes hypotheses (10.6), integration recommendations (10.7), and future work (10.8). Subsection numbering preserved.

Read `CK_vs_rocWMMA.md` only when actively comparing the two libraries for optimization, validation, or benchmarking. Performance tuning of a single library does not require it.

---

## 11. Empirical Measurements (braidinfer 2026-05-06)

This section records primitive-level cycle counts measured on RX 7900 XTX
(gfx1100, wave32, ROCm 7.1.x, hipcc 22.0.0git). Source benchmarks live in
`/home/mcelrath/Projects/ai/braidinfer/kernels/diagnostic/{name}_bench/`.
Each rule below cites the bench directory that produced it. Numbers are
median in-kernel cycles per op via `s_memrealtime` / `wall_clock64`.

**Sub-blocks** (organizational grouping; section numbers UNCHANGED — these are
just readers' grouping for navigation):

- **11A: Single-GPU primitives and perf reference** — §11.1, §11.2, §11.3, §11.5,
  §11.6, §11.7, §11.8, §11.9, §11.10. Stable wave/reduction/cache/swizzle/WMMA
  fundamentals. Rarely-edited content.
- **11B: Multi-GPU latency and measurement hygiene** — §11.11, §11.12. Includes
  CPU SCHED_FIFO + amdgpu IRQ pinning requirements.
- **11C: Historical / Falsified coherence-wedge investigation** — §11.4,
  §11.13, §11.16 (stub->§11.19), §11.18 (stub->§11.19). Preserved for
  methodology record.
- **11D: Current coherence model** — §11.14, §11.15, §11.17, §11.19 (merged).
  Currently load-bearing operational guidance for cross-GPU code.

New readers should start with 11D. 11A and 11B are reference. 11C is preserved
record (do NOT re-propose falsified hypotheses).

§11.4 placement: 11C (HAZARD content). §11.9 placement: 11A (WMMA gotcha, perf-validated).

### 11.1 Wave32 / sub-wave reductions (DPP + permlanex16 vs `__shfl_down`)

Source: `kernels/diagnostic/reduce_bench/`.

| Reduction shape | DPP+permlanex16 | `__shfl_down` (LDS bpermute) | LDS-tree | Speedup |
|---|---|---|---|---|
| wave32 sum (32 lanes)  | **3.46 cyc** | 13.33 cyc | n/a    | 3.85× |
| sub-wave 16            | **2.69 cyc** | 10.68 cyc | n/a    | 3.97× |
| sub-wave 8             | **2.48 cyc** |  8.54 cyc | n/a    | 3.44× |
| sub-wave 4             | **2.27 cyc** |  6.22 cyc | n/a    | 2.74× |
| sub-wave 2             | **2.15 cyc** |  4.04 cyc | n/a    | 1.88× |
| block256 sum           | **13.72 cyc**| 23.26 cyc | 43.35 cyc | 1.7× / 3.2× |

**Rule**: All wave/sub-wave/block sum reductions on gfx1100 should use the
DPP `row_xmask:N` + `v_permlanex16` chain, not `__shfl_down`. The bare LDS
tree is the slowest path of the three. `ds_bpermute_b32` (which `__shfl_*`
lowers to) also has a same-VGPR non-determinism hazard on gfx1100 (see
braidinfer kb `bz0-root-cause-solved-2026-05-03-shfl`); DPP and permlane
have no analogous hazard.

### 11.2 Wave32 / block max reductions

Source: `kernels/diagnostic/lane_bench/`.

| Reduction shape | DPP+permlanex16 | LDS-tree max | Speedup |
|---|---|---|---|
| wave32 max (32 lanes) | **5.00 cyc**  | 46.18 cyc | 9.2× |
| block256 max          | **15.31 cyc** | 46.18 cyc | 3.0× |

**Rule**: same as 11.1 — use DPP + permlanex16. Online-softmax `m = max(m,
score)` broadcast in attention kernels saves ~30 cyc per timestep × seq_len.

### 11.3 atomicAdd float — CAS vs HW (`unsafeAtomicAdd`)

Source: `kernels/diagnostic/rdna3_compute_bench/` and
`kernels/diagnostic/rdna3_memory_bench/`.

`atomicAdd(float*, float)` in HIP on gfx1100 lowers to a
`global_atomic_cmpswap_b32` CAS loop **by default**. `unsafeAtomicAdd`
(declared in `<hip/amd_detail/amd_hip_unsafe_atomics.h>`) emits the
hardware `global_atomic_add_f32` instruction directly.

| Contention regime          | CAS-loop default | HW (unsafeAtomicAdd) | Speedup |
|---|---|---|---|
| 1 slot (all-hot)           | 68802 cyc        | 72 cyc               | **~955×** |
| 4 slots                    | 4861 cyc         | 18 cyc               | ~270× |
| 16 slots (typical split-K) | 1206 / 234.7 cyc | 10 / 5.0 cyc         | ~47× |
| 256 slots                  |   333 cyc        |  9 cyc               | ~37× |
| Per-thread unique          |    24 cyc        |  2 cyc               | **13×** |

**Rule**: Every f32 atomic accumulator on a hot path must use
`unsafeAtomicAdd`. The default `atomicAdd(float*, float)` is the 13–955×
slower CAS path. Safe for braidinfer because no kernel relies on strict
NaN/denorm propagation through atomicAdd. NEGATIVE result also recorded:
split-K GEMV with HW atomic-add **does not** beat block-cooperative
reduction at typical LLM shapes (K=1024, N=512..248320) because the GPU
is row-parallel-rich already; split-K only wins when N << CU count.

### 11.4 Grid-wide barrier — `cg::grid_group::sync()` vs hand-rolled atomic

Source: `kernels/diagnostic/rdna3_sync_bench/`.

**Bench result (synthetic, no concurrent SYSTEM-scope writes):**

| Block count | `cg::grid_group::sync()` | `atomic_block_barrier` (in `rdna3_sync.h`) | Speedup |
|---|---|---|---|
| Typical megakernel range (~32-384 blocks) | (cg path) | (atomic path) | **115–155×** |

The cg path performs a system-coverage release/acquire pair. The
atomic-counter path uses agent-scope fences + a single `global_atomic_add`
+ generation-flag spin, which on gfx1100 is the cheapest barrier that
satisfies the megakernel's actual data-visibility requirement (consumer
reads in same persistent kernel, L2-coherent VRAM).

#### HAZARD — PCIe-write-before-barrier wedge (CLASS rule, confirmed at multiple sites 2026-05-12)

**This is a CLASS rule, not a site-specific incident.** Any `atomic_block_barrier` (or any barrier whose preamble lowers to `s_waitcnt_vscnt`) that follows a SYSTEM-scope UC store will wedge intermittently under cross-GPU PCIe pressure — regardless of which opcode site the barrier appears in. Confirmed independently at: (a) the post-poll shutdown barrier (original 4fg.5 site); (b) 2026-05-12 Phase 0b, `op_moe_ffn_remote` / `dispatch_opcode` trailing barrier, wedging at MoE boundary 11's `try_wait_acks_many → stream.synchronize` after ~30 successful cooperative-kernel launches across 4 GPUs. Same `s_waitcnt_vscnt` mechanism, different call site. (c) 2026-05-13 udi+braidinfer joint reproducer at `braidinfer/kernels/diagnostic/persistent_skeleton_repro/` (skeleton at braidinfer commits `10e2e82` / `eedc254`+, runner+aggregator at `dbdfd76`). Variant V7 (4 GPU + persistent cooperative kernel + cross-GPU peer-UC store + post-poll `atomic_block_barrier`) reproduces the wedge at **30% rate over n=10 trials**, consistent with the 20-50% envelope cited above. Wedge fingerprint: `progress_pc=0x10000005, completed_dispatches=0, seq_completed=false, ack=1 visible`. Negative reproductions (V0 minimal pattern, V5 V0 + outer-loop `watchdog_alive` UC store) rule out bare host-mapped polling + barrier as sufficient — the peer-VRAM UC write is the load-bearing differential, not host-mapped polling per se.

**This hazard is a specific manifestation of the broader coherence-author-rules in §5.5.** Rule 1d (kernel-boundary sync) is the canonical primitive for cross-agent ordering; this section documents why a *grid-wide cooperative-kernel barrier* is NOT a substitute for a kernel-boundary on this hardware. Use §5.5 Rule 1 (a-d) to choose the correct primitive; use this section's HAZARD details only when working inside a persistent cooperative kernel that can't get a kernel-launch boundary between writer and reader.

`atomic_block_barrier` opens with `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")`,
which lowers on gfx1100 to `s_waitcnt_vscnt null, 0x0`. This drains **all**
prior vector-store traffic to its destination domain — including
SYSTEM-scope UC stores to host-mapped memory that are still in the GPU's
PCIe write buffer.

Under concurrent multi-GPU PCIe pressure on this system (4× RX 7900 XTX),
a single in-flight SYSTEM-scope UC write occasionally fails to drain.
`s_waitcnt_vscnt` then stalls **indefinitely**. The barrier never returns,
all blocks block, the cooperative kernel never exits, `hipStreamDestroy`
hangs forever. Empirical wedge rate on q8 4-GPU MoE persistent-worker
shutdown: **20–50%** before the fix below; 10M+ event synthetic stress
that *had no SYSTEM-scope UC writes between barriers* showed **0%** wedge.
The differential variable is the SYSTEM-scope UC write immediately
preceding the barrier — not the barrier primitive itself.

**Wrinkle on the stalling site (2026-05-13 V7 evidence).** The V7 minimal
reproducer's wedge fingerprint shows the worker DID cross the post-poll
barrier (`progress_pc` advances to `PC_PRE_ACK=0x10000005`), DID write
`ack=1` (visible to host before timeout), then wedged BEFORE writing
the next field (`completed_dispatches`). The `s_waitcnt_vscnt` that
stalls in V7 is the one emitted by the POST-barrier UC store's release
fence — the barrier's own preamble fence already drained. Whether the
preamble fence can ALSO be the stall site in other patterns (e.g.,
where the post-barrier store isn't present) is the V2-variant
hypothesis (`rdna3_barrier.h` `BRAIDINFER_BARRIER_V2`) and remains
open. Mechanism is the same in either case — release-fenced UC store
sequence under cross-GPU PCIe pressure can stall on vscnt drain — and
the mitigation (defer past barrier, AGENT-scope, kernel-boundary) is
unchanged.

Earlier `kb` entries in this codebase
(`rdna3-atomic-block-barrier-multi-gpu-fundamental-issue`,
`rdna3-atomic-block-barrier-cg-grid-group-sync`) framed this as
"atomic_block_barrier has a multi-GPU bug" or "mixing with `cg::sync` is
fundamentally broken." Both framings were wrong. The barrier primitive is
correct; the wedge is upstream of the barrier in the form of an
undrainable PCIe write that the barrier's leading `s_waitcnt_vscnt` waits
on indefinitely.

**Operational guidance** for avoiding this hazard (allocation method choice, SYSTEM→AGENT scope conversion, deferring host writes past barriers) lives in **§5.5 Rule 1** and **§5.5 Rule 3** (antipatterns table). This section's role is to document the empirical bench result and the discovered mechanism; the rules for *how to author code that doesn't hit it* are in §5.5.

**Historical note on prior misdiagnosis** (retained as cautionary tale): two earlier `bd` memories (`rdna3-atomic-block-barrier-multi-gpu-fundamental-issue`, `rdna3-atomic-block-barrier-cg-grid-group-sync`) framed this as "the barrier primitive is fundamentally broken on multi-GPU" or "mixing `cg::grid_group::sync` with `atomic_block_barrier` is structurally unsafe." Both framings were wrong. The barrier primitive is correct; the bug was in the SYSTEM-scope UC writes that preceded it at multiple call sites. The mixing experiment falsified "this specific barrier site is broken" without isolating the per-site SYSTEM-scope-write variable. Both kb entries are formally retracted via `rdna3-atomic-block-barrier-retraction-2026-05-12`. The cost of the misdiagnosis was several days of debugging — keep the diagnostic discipline of isolating per-site variables when a barrier appears to "fail" on multi-GPU.

### 11.5 LDS bank-conflict measurements (RDNA3 WGP mode)

Source: `kernels/diagnostic/rdna3_memory_bench/`.

RDNA3 WGP-mode LDS = 64 banks × 4 bytes (RDNA3 ISA §2.3.1).

| Layout | Stride | cyc/access | Notes |
|---|---|---|---|
| packed 1D `shared[256]`    | 1   | 4.0  | OK |
| packed 2D `[32][32]`        | 32  | 17.9 | **32-way bank conflict** |
| packed 2D `[32][64]`        | 64  | 17.8 | **conflict** |
| packed 2D `[32][128]`       | 128 | 18.0 | **conflict** |
| padded 2D `[32][32+1]`      | 33  | 2.7  | **fixed** (6.7× speedup vs 32) |
| padded 2D `[32][64+1]`      | 65  | 2.7  | fixed |
| padded 2D `[32][128+1]`     | 129 | 2.7  | fixed |

**Rule**: Any 2D LDS tile with inner stride a multiple of 32 floats
must add +1-float row padding. Use
`braidinfer::rdna3::lds_pad_for_tpg<Tpg>()` to express this at compile
time. Existing 1D `shared[groups_per_block]` layouts have no conflict.

### 11.6 cp.async equivalent (`global_load_lds_b{32,64,128}`)

**NOT available on gfx1100.** The `__builtin_amdgcn_global_load_lds`
intrinsic and the `global_load_b{32,64,128} ... lds` ISA encoding are
gated behind LLVM target feature `vmem-to-lds-load-insts`, which is
read-only and enabled ONLY on CDNA (gfx9xx/gfx94x). Verified by:

1. `__builtin_amdgcn_global_load_lds(...)` with `--offload-arch=gfx1100`
   produces: `error: '__builtin_amdgcn_global_load_lds' needs target
   feature vmem-to-lds-load-insts`.
2. `llvm-mc --mcpu=gfx1100` rejects `global_load_lds_b32 ...`; same
   input through `--mcpu=gfx942` is accepted.

**Rule**: Do not write stub wrappers that hide a compile error. The
gfx1100 substitute is regular `global_load_b{32,64,128}` → VGPR →
`ds_write_b{32,64,128}`, which the compiler scheduler already pipelines
optimally. Manual unrolling did NOT win in microbenchmarks. Source:
`kernels/diagnostic/rdna3_memory_bench/`. RDNA4 (gfx1200) **may** gain
the LDS-bit forms in a future LLVM — verify on hardware before adopting.

### 11.7 `ds_swizzle_b32` ctrl-bit layout (BitMaskPerm)

Source: `kernels/diagnostic/lane_bench/` (probe via swizzle_probe).

The LLVM 17 / ROCm 7.1.x intrinsic
`__builtin_amdgcn_ds_swizzle(value, ctrl)` uses the bit field layout

```
ctrl = (xor_mask << 10) | (or_mask << 5) | and_mask     // top bit clear (BitMaskPerm)
ctrl = 0x8000 | (l3<<6 | l2<<4 | l1<<2 | l0)            // top bit set   (QDMode)
```

with each lane `i` fetching from lane `((i AND and_mask) OR or_mask) XOR xor_mask`.

**Note**: this is the OPPOSITE of what some published AMDGPU SI ISA
references state (which put AND in the high bits). The braidinfer
header `kernels/rdna3_lane.h` uses the empirically-verified layout. If
ROCm changes the encoding in a future LLVM, rebuild `swizzle_probe`
and confirm before trusting other docs.

### 11.8 Cache invalidation issue costs

Source: `kernels/diagnostic/rdna3_memory_bench/`, lane-0 issue, 1024 invs/loop.

| Wrapper | Issue cost (cyc, lane-0 only) | Use |
|---|---|---|
| `gl0_invalidate()`               | +1.21 cyc (≈0.5 ns) | per-CU L0 flush |
| `gl1_invalidate()`               | +3.30 cyc (≈1.4 ns) | per-shader-array L1 flush |
| `gl0_invalidate + gl1_invalidate`| +3.36 cyc (≈1.5 ns) | combined; gl0 is essentially free on top of gl1 |

**Note**: these are issue-only costs (no `s_waitcnt vmcnt(0)` after).
The actual stall when the next VMEM op needs the invalidation to drain
is workload-dependent. There is **no `buffer_gl2_inv`** instruction on
RDNA3 (`llvm-mc --mcpu=gfx1100` rejects it); for cross-GPU peer reads
the only working coherence pattern is UC-mapped target memory + agent-
scope `__threadfence` (see §5.3).

**Placement rule**:
- Producer/consumer on different shader arrays → `gl01_invalidate()` (both)
- Producer/consumer on same SA, possibly different CUs → `gl1_invalidate()`
- Producer/consumer on same CU → none
- Cross-GPU peer reads → §5.3 UC pattern (NEITHER invalidate works)

### 11.9 WMMA fragment lane-mapping rule (3–5× perf gotcha)

Source: `kernels/diagnostic/rdna3_compute_bench/` (table 1b).

RDNA3 WMMA `f32_16x16x16_bf16_w32` requires:

- A/B fragments: lanes 16-31 hold the **same** per-lane data as lanes 0-15
  (RDNA3 lane-replication rule, §4.2 above).
- C/D output: lanes 0-15 contribute even rows (0,2,…,14); lanes 16-31
  contribute odd rows (1,3,…,15). The output is NOT replicated.
- Each lane fills a `u16x16` (A/B) or `f32x8` (C/D) fragment.

| Pre-pack strategy | Load hoisted | Load+MMA per iter |
|---|---|---|
| RDNA3-native (each lane direct-loads its K-stripe with `tid & 15`) | **1.84 cyc/MMA** | **13.06 cyc/MMA (1.00×)** |
| CUDA-style (load uniquely in lanes 0-15, permute into 16-31)        | 1.76 cyc/MMA | 27.12 cyc/MMA (2.08× SLOWER) |
| LDS-staged (store 16×16 tile to LDS, fragment-load from LDS)         | 1.83 cyc/MMA | 96.19 cyc/MMA (7.36× SLOWER) |

**Rule**: Use the native pre-pack — each lane loads its OWN K-stripe
directly from global memory using `tid & 15` for the row index. Lanes
16-31 read the SAME global addresses as lanes 0-15; the L1/L2 caches
coalesce. Never permute low-half → high-half manually. Never stage tiny
16×16 tiles through LDS. (LDS staging IS a win when one tile feeds many
WMMAs; that's a different pattern.) Use `braidinfer::rdna3::load_a_bf16`
/ `load_b_bf16` / `mma_sync_bf16` / `store_c_f32` from `rdna3_compute.h`.

### 11.10 Header consolidation

The braidinfer primitive library lives at:

```
braidinfer/kernels/rdna3_ops.h          # umbrella, single include for callers
braidinfer/kernels/rdna3_memory.h       # atomic + cache-inval + LDS pad
braidinfer/kernels/rdna3_lane.h         # ballot/broadcast/swizzle/wave32-max
braidinfer/kernels/rdna3_reduce.h       # wave32-sum/sub-wave-sum/block-sum
braidinfer/kernels/rdna3_compute.h      # WMMA wrappers + split-K GEMV
braidinfer/kernels/rdna3_sync.h         # fences + atomic_block_barrier
```

All under `namespace braidinfer::rdna3`. Wave32 + gfx1100 only;
mixing with `-mwavefrontsize64` will silently miscompile.

### 11.11 RDNA3 multi-GPU coordination — latency envelope

Source: udi epic 2026-05-12/13, joint with braidinfer-pky.3. Raw data in
`exterior_algebra/results/{launch_overhead,signal_latency_floor,
dual_megakernel,megakernel_skeleton,megakernel_fanout,
megakernel_under_load_rt,sdma_latency_curve,cross_gpu_write_latency,
wedge_repro_matrix}.json`.

All numbers under CPU 55 SCHED_FIFO. Latencies on default-scheduled
CPUs drift 1.5–2× higher; §11.12 documents the measurement-hygiene
prerequisites these constants depend on.

**CPU↔GPU primitives (single GPU):**

| Mechanism                                       | median  | p99    |
|-------------------------------------------------|---------|--------|
| GPU→pinned-host signal (`hipHostMallocMapped`)  | 310 ns  | 670 ns |
| `hipLaunchKernel` host-side API return          | 1.65 µs | 1.97 µs |
| `hipLaunchKernel` + `hipDeviceSynchronize` (warm) | 26 µs   | 27 µs  |
| `hipLaunchKernel` + sync (cold first launch)    | 142 µs  | —      |
| Persistent megakernel CPU↔GPU doorbell-ack      | 2.79 µs | 2.88 µs |
| `hipGraphLaunch` single-node (prewarmed)        | 36 µs   | 45 µs  |

`hipGraphLaunch` is **slower** than direct launch on ROCm 7.2.x —
counter to the CUDA-derived assumption. Single-node graphs add ~10 µs
host-side overhead.

**Cross-GPU primitives (two persistent megakernels, no CPU in loop):**

| Mechanism                              | median  | min    | p99    |
|----------------------------------------|---------|--------|--------|
| Peer-VRAM write (hardware floor)       | —       | 1.2 µs | —      |
| Dual-megakernel RTT (pinned-host path) | 3.73 µs | 2.58 µs | 3.92 µs |
| Dual-megakernel RTT (peer-VRAM path)   | 4.66 µs | 4.62 µs | 4.71 µs |

**1→N GPU fan-out** (1 CPU dispatch, N persistent megakernels ack):

| N | median  | p99    | per-GPU amortized |
|---|---------|--------|-------------------|
| 1 | 1.62 µs | 1.65 µs | 1.62 µs           |
| 2 | 2.16 µs | 2.84 µs | 1.08 µs           |
| 4 | 2.53 µs | 3.00 µs | 0.63 µs           |
| 8 | 2.87 µs | 3.20 µs | 0.36 µs           |

Sub-linear: 8× GPUs → 1.77× total latency.

**SDMA / blit-kernel for `hipMemcpyPeerAsync`:** `ROC_P2P_SDMA_SIZE =
1 MiB` (rocm-clr `rocclr/utils/flags.hpp:220`). Below threshold: blit
kernel, ~14.4 µs floor regardless of size. At/above: SDMA, concurrent
with compute (overlap_ratio=1.007 at 512 MB). Peak 6.3 GB/s blit,
3.34 GB/s SDMA at 512 MB.

**HSA signal note.** `hsa_amd_signal_create(HSA_AMD_SIGNAL_AMD_GPU_ONLY)`
places the signal value in cached GPU VRAM (flags=0x0, NOT UC=0x3) —
same L2-staleness exposure as polling `hipMalloc` memory. True MMIO
BAR2 doorbell (`AMD_SIGNAL_KIND_DOORBELL`) is only available via
`hsa_queue_create`'s `doorbell_signal` and is not user-pollable. There
is no user-level "sub-µs MMIO bypass" of the host-mapped polling path
on this stack.

**amdkfd doorbell mmap.** Technically feasible (mmap of `/dev/kfd`
with offset in `KFD_MMAP_TYPE_DOORBELL` range; see linux
`drivers/gpu/drm/amd/amdkfd/kfd_doorbell.c:106-146`), but the doorbell
write is interpreted as a queue wptr advance, not a free-form
notification value. Doesn't bypass AQL packet dispatch.

**`hipDeviceSynchronize` wait mechanism.** Dominated by `mwaitx` CPU
spin in `InterruptSignal::WaitRelaxed` (rocr-runtime
`interrupt_signal.cpp:138-199`); spins up to 200 µs before falling
through to `hsaKmtWaitOnEvent_Ext`. GPU completion signals arrive in
~30–38 µs, so the kernel-sleep path is never reached.
`HSA_ENABLE_INTERRUPT=0` switches to `BusyWaitSignal` which is also
pure `mwaitx` spin — no measurable difference (38 µs vs 36 µs).

### 11.12 Measurement hygiene prerequisites (load-bearing on §11.11 numbers)

The latency constants in §11.11 hold **only** under the following CPU
isolation. On default-scheduled CPUs the numbers drift 1.5–2× higher
with high variance; failure to apply these is a frequent cause of "I
can't reproduce the documented latency" reports.

This system's kernel command line (`/proc/cmdline`):

    isolcpus=55-63 nohz_full=55-63 rcu_nocbs=55-63

amdgpu IRQs (`/proc/irq/254..261/smp_affinity_list`) pinned per
`/usr/local/sbin/system-stability.sh:457-465`:

| IRQ | CPU |
|-----|-----|
| 254 | 56 |
| 255 | 57 |
| 256 | 58 |
| 257 | 59 |
| 258 | 60 |
| 259 | 61 |
| 260 | 62 |
| 261 | 63 |

Host-thread dispatch (the side that writes doorbells / observes acks)
runs on CPU 55 with `SCHED_FIFO` priority 50:

    chrt -f 50 taskset -c 55 ./your_binary

The `rdna3_timing.h::rdna3_timing_check_affinity()` primitive
(braidinfer `kernels/rdna3/`) verifies this at start-of-measurement
and emits a `WARN` to stderr if the caller is not on CPU 55 or not on
`SCHED_FIFO`.

**Side gotcha (clock64 SCLK throttle).** GPU `clock64()` reads SCLK,
which throttles when the kernel is idle. A calibration kernel using
`__builtin_amdgcn_s_sleep(N)` reports SCLK as ~18 MHz instead of
~2.5 GHz. Use `clock64()` only inside tight-spin/compute sections,
never in a calibration loop that includes `s_sleep`. CPU
`CLOCK_MONOTONIC_RAW` is canonical for sub-microsecond RTT
measurements that bracket GPU work.

**Side gotcha (clock64 spin-delta sign).** `clock64()` spin-deltas
can wrap or go negative after ~2 rounds in some configurations (SCLK
reset mid-spin). For persistent-kernel delay loops, prefer
`__builtin_amdgcn_s_sleep(N)` over `clock64()`-based busy-wait.

### 11.13 Cooperative-grid relaunch wedge — empirical archive (confirmed; mechanism unknown)

**Status: HISTORICAL-RECORD.** The empirical probe table for the
cooperative-grid relaunch wedge — including the six MES-side patches
that failed to clear it, the V0/skeleton reproducer history, and the
"process exit is only known recovery" finding — has been extracted to
**mes/MES_PROBE_ARCHIVE.md** for focus. Mechanism remains unidentified;
§11.15 confirms this is a real separate phenomenon from the immediate-ack
protocol deadlock.

See: [mes/MES_PROBE_ARCHIVE.md](./mes/MES_PROBE_ARCHIVE.md)

Topics in the extracted archive: 15+ user-space probes, 6 MES-side kernel
patches (Designs D–I), V0 skeleton reproducer history, cross-chip
corroboration (gfx1103/1150/1152 framework reports), and the
process-exit teardown sequence (what actually clears the wedge).

### 11.14 `s_buffer_gl0_inv` / `s_buffer_gl1_inv` silently no-op on gfx11+ (2026-05-14)

**Empirical finding.** During the 2026-05-14 wedge chase, a
host-mapped UC poll inside the wedged persistent kernel did not
observe a freshly-written value even after 95 000+ poll iterations
each issuing `s_buffer_gl0_inv` / `s_buffer_gl1_inv` between loads.
Adjacent-cache-line stores from the GPU were independently visible to
the host, ruling out a cache-pressure / GL2-eviction explanation. The
inference: on gfx11+ the reader-side L0/L1 scalar-cache invalidation
opcodes are either silently dropped or insufficient for host-mapped
UC lines that have been latched into the scalar cache.

This is the reader-side counterpart to the documented writer-side gap
(`buffer_wbl2` absent on gfx11+; AMD's own
`hsa-rocr-p2p-mtype-uc-gfx11.patch` works around the writer side).
We have not seen AMD acknowledge a reader-side workaround.

**Implication.** Do not rely on `__builtin_amdgcn_s_buffer_gl0_inv()`
/ `_gl1_inv()` to refresh a stale value cached in a scalar load on
gfx11+. If a value MUST be re-read fresh, force a vector load through
a non-uniform address pattern, or accept that host-mapped UC reads
have one-way (write-only-from-GPU) reliability inside long-running
persistent kernels.

**Note.** This was NOT the cooperative-grid wedge mechanism (that is
Rule 9, §11.13). It was discovered while chasing the wedge and is
preserved here as a standalone ISA-gap reference.

### 11.15 Persistent worker ack protocol — required immediate-write (2026-05-14)

**Empirical finding.** A standalone reproducer (`kernels/diagnostic/
persistent_skeleton_repro/prod_kernel_test`) that loads
`megakernel.hsaco`'s `persistent_worker` symbol with a fresh
`WorkerQueue` and dispatches 3 sequential batches wedged on the FIRST
dispatch — zero process state, zero prior cooperative launches, zero
weight load. Wedge fingerprint identical to braidinfer production:
`stuck_pc=PC_IN_POLL`, `ack=0`, host timeout. Same hsaco with a
minimal `simple_persistent_worker` symbol (identical poll loop but
immediate ack=seq write after dispatch) PASSED instantly.

**Root cause.** Phase 2' deferred-ack pattern (commit 8cf8084) wrote
`ack=last_seq` at the TOP of each outer iter's body but AFTER iter
N+1's inner-poll completion. Inner poll waited for `seq=N+1`. Host
blocked on `ack=N` before sending `seq=N+1`. **Deadlock by design.**

**The host-blocking contract.** Host's `try_wait_ack` polls
`ack == seq`, times out at 30s, panics with the wedge signature.
Worker MUST publish `ack=seq` in the same iter that processed `seq`,
before the next inner-poll spins on `seq=N+1`.

**Canonical fix** (`kernels/megakernel.hip` post-dispatch loop):

```cpp
// After dispatch_opcode loop completes:
__threadfence();
if (threadIdx.x == 0 && blockIdx.x == 0) {
    braidinfer::rdna3::host_uc_store_agent(&queue->ack, seq);
}
last_seq = seq;
// loop back to next outer iter
```

AGENT scope on the ack store is mandatory (Rule 8 §5.5); SYSTEM-scope
emits `s_waitcnt_vscnt` that wedges across the next iter's
`atomic_block_barrier` under multi-GPU PCIe pressure (§11.4).

**Canonical helper.** `kernels/rdna3/rdna3_persistent_protocol.h`
exposes `persistent_iter_poll_barrier` + `persistent_iter_ack`. The
helper encapsulates the canonical protocol so callers cannot
accidentally reintroduce a deferred-ack pattern.

**Standing regression.** `scripts/check_persistent_protocol.sh` builds
and runs the standalone `prod_kernel_test`, asserts ack matches seq
for 3 sequential dispatches. Runs in ~1 second, no model load
required. Gate any change to `kernels/megakernel.hip` against this
script.

**Why 10+ hours to root-cause.** The wedge signature matched several
distinct hypothesis classes: cache coherence (gl0/gl1_inv no-op),
launch-ordinal-N (Rule 9 §11.13), VCC/EXEC compiler hazards, IOMMU
TLB staleness, prior-DMA pressure. ~25 negative-result probes. The
protocol-deadlock hypothesis was reached only after a standalone
reproducer with ZERO process state ALSO wedged — at which point the
wedge could not be cache, hazard, or environmental, and the suspect
surface narrowed to the kernel's own outer-body logic. A bisection
(`simple_persistent_worker` vs `persistent_worker`, both in the same
hsaco) localized the bug to the deferred-ack write.

**Lesson.** Standing standalone reproducers (no model load, no
braidinfer state) catch protocol bugs in seconds. The full-system
test path's slow turnaround over-attributes cause to deep
hardware/firmware mechanisms. Add the regression script to CI; do
not let kernel-protocol changes ship without it.

### 11.16 UC dst alone is insufficient — intra-kernel cached-read staleness (2026-05-17) — MERGED INTO §11.19

**Status: SUPERSEDED-by-§11.19. Content preserved at §11.19 subsections (r)-(v).**

See §11.19 for the merged falsification record. The UC dst staleness investigation
(PROVISIONAL, r7dv 2026-05-17, intra-kernel cross-WGP cached-read on
`hipDeviceMallocUncached` buffers) is recorded as §11.19 (r)-(v) with its original
hypotheses, working mitigation, surface census, and cross-engine confirmation intact.

### 11.17 Warmup→active transition leaves stale stream/event state (2026-05-17)

**Empirical finding (CONFIRMED).** During llama.cpp graph-lifecycle work
(commit 1b20a9718, branch `graph-lifecycle-A`) on 2026-05-17, the
following pattern was characterized:

  - llama.cpp's `common_init_from_params` invokes a model warmup pass
    before any real decode. On gfx1100 with multi-GPU `-sm expert`,
    this warmup pass historically caused either (a) a libamdhip64
    segfault at offset `0x464351` (rocclr `submitKernelInternal`
    stale-descriptor class — see §11.13 lineage) or (b) downstream
    §11.4-class MES wedges that fire hundreds of decodes later.
  - Pattern (a) was previously worked around by `c3c96b5d7` (auto-skip
    warmup for recurrent/hybrid models). That workaround left
    non-hybrid models still hitting (a) on multi-turn LCP-reuse.
  - The structural fix (commit `1b20a9718`): suppress CUDA graph
    capture during the warmup pass via a per-backend `bool warmup`
    flag in `ggml_backend_cuda_context`, then DRAIN the streams on
    the warmup→active transition (`cudaDeviceSynchronize()` +
    `stream_context().reset()`).

The drain is what closes the wedge sub-class. Without the drain, the
graph-capture-suppression during warmup still left the warmup decode's
ops on concurrent streams 1-3, and the post-warmup first-real-decode
captured graphs reference stale stream/event state. That stale state
propagates through subsequent dispatches and produces a §11.4-class
MES wedge after ~410 decodes (`/tmp/A-sustain.log` initial run).
With the drain, `sustain.py 1000/1000` completes clean (~1.83s/req),
zero HW exceptions, zero GPU hangs, zero `0x464351` segfaults.

**Wedge sub-class.** This is structurally distinct from the two §11.4
sub-classes already documented:

  - **Cross-GPU peer-UC store under PCIe pressure** (V7 reproducer,
    `kernels/rdna3/rdna3_peer.h`): SYSTEM-scope `vscnt`-drain failure under
    multi-GPU concurrent UC stores. Mitigation: deferred-write
    pattern (`rdna3_peer_write_deferred`).
  - **Cached-store-on-UC-dst writer hazard** (DeinterleaveInst
    pattern, deleted comment at `braidinfer/crates/braidinfer-runtime/src/multi_gpu.rs:80-86`):
    cached vector stores into a host-mapped UC page leave dirty L2
    lines that don't drain on agent-scope fence. Mitigation: writer
    must use UC-aware stores or explicit `__threadfence_system`.
  - **Warmup→active stream state transition (this section)**: warmup
    decode emits ops to concurrent streams; post-warmup graph capture
    references stale stream/event records, producing wedges after a
    few hundred decodes. Mitigation: drain on transition
    (`cudaDeviceSynchronize` + reset concurrent-event tracking).

All three sub-classes share the §11.4 MES-wedge fingerprint at
ROCm-runtime level (HW Exception by GPU node-N: GPU Hang), but their
mitigations are different. The drain primitive (Rule 1d analog)
addresses sub-class 3; UC allocation (Rule 1a/1b) addresses sub-class
2; deferred-write (`kernels/rdna3/rdna3_peer.h`) addresses sub-class 1.

**Production fix lineage 2026-05-17.** Three commits in the llama.cpp
HIP backend stack as the canonical §11.4 mitigation suite:

| Commit | Sub-class | Surface |
|---|---|---|
| `44855024b` | fattn-sp peer FA writeback | UC staging slab (Rule 1a, device-UC) |
| `878bc9df0` | MOE_FUSED SDMA combine | UC writer slab (Rule 1a, device-UC) |
| `1b20a9718` | Warmup→active transition | stream drain on warmup→active (this section) |

Three independent failure modes, three independent fixes, all under
the §11.4 wedge umbrella. The `kb_add` entry tagged
`§11.4-mitigation-suite-2026-05-17` (documents the 3 commits above
as the canonical §11.4 recovery point on the llama.cpp side) records
this as a recovery point — the multi-turn-segfault and §11.4-class
wedge surfaces are substantially closed after these three commits.

**Acceptance evidence.**
  - Gate 1 (warmup runs): `common_init_from_params: warming up the
    model with an empty run` confirmed in log; no `skipping warmup`
    line; no `CUDA graph warmup complete` during warmup (capture
    suppressed as designed).
  - Gate 2 (35-turn LCP-reuse): 0/35 failures, no `libamdhip64
    0x464351` segfault.
  - Gate 3 (first-decode latency healthy): 567 tok/s prefill on
    1232-token first prompt; no burst of graph captures observed.
  - Gate 4 (sustained 1000-iter): 1000/1000 OK, 0 HW Exceptions, 0
    GPU Hangs, total 1823.1s (~1.83s/req). Past the 410/1000 wedge
    point that fired without the drain.

**Implementation summary.** Five files, +63/-13 lines in
`graph-lifecycle-A`:
  - `common/common.cpp`: drop hybrid-warmup-skip block (was the
    workaround for sub-class 3 before the structural fix).
  - `ggml/include/ggml-cuda.h`: declare `ggml_backend_cuda_set_warmup`.
  - `ggml/src/ggml-cuda/common.cuh`: add `bool warmup` to
    `ggml_backend_cuda_context`.
  - `ggml/src/ggml-cuda/ggml-cuda.cu`: gate `graph_compatible` behind
    `!cuda_ctx->warmup`; the setter drains on warmup→active
    transition.
  - `src/llama-context.cpp`: propagate `llama_context::set_warmup` to
    all CUDA backends via `ggml_backend_is_cuda` check.

**Lesson.** When introducing a per-mode flag to a backend, the
mode-transition is itself a structural change that may need its own
cleanup primitive — not just the steady-state behavior of each mode.
Sub-class 3 is the canonical example: warmup mode is fine, active
mode is fine, but the transition leaves caller-invisible state that
breaks after hundreds of dispatches. Treat any new mode-transition in
a long-lived backend as a §11.17 review item.

### 11.18 Host-mapped portable-coherent buffer + multi-GPU GART read → producer L2 staleness → UTCL2 TCP PERMISSION_FAULT cascade (2026-05-18) — MERGED INTO §11.19

**Status: FALSIFIED-by-§11.19. Content preserved at §11.19 subsections (w)-(aa).**

The G9 producer-fence hypothesis (Rule 10 `fence_device()` at RMSNorm exit) was
empirically tested as braidinfer sm16 p4 trial. Result: 10/30 PASS = within noise
of 16/30 baseline. FALSIFIED. See §11.19 (d) for the falsification record and §11.19 (f)-(g)
for the identified mechanism (MES μC private cache / memory-hub state, below
userspace/kernel-mode reach) and production cure (warmup-discard, commit a048318).

See §11.19 for the full merged record including: trigger/observed-data (w), mechanism
and coherence analysis (x), required-fence and TLB-hazard details (y), code patterns
(z), library integration and call-site migration (aa-i), §5.5 Rule 10 proposal (aa-ii).
Do not re-propose the G9 producer-fence hypothesis — it is empirically falsified.

### 11.19 Cold-start mailbox visibility race — falsification cascade & production cure (2026-05-19)

**Status: §11.18 Rule 10 producer-fence hypothesis FALSIFIED. Cure layer is below userspace shader and kernel-mode driver reach.**

**(a) Symptom and localization.** With multi-GPU MoE models (qwen35_35b_a3b.q4) on gfx1100 `-g 4`, the FIRST mailbox transaction per worker GPU after `Model::load` has ~40% probability of producing NaN logits (Sig A: argmax-of-NaN collapses to repeated low-id token, typically `!`). Sticky per PASID: once a worker queue triggers the race it stays broken until process reload. Producer is the CPU (Rust `write_volatile` to host-mapped UC mailbox in `crates/braidinfer-runtime/src/persistent_dispatch.rs::dispatch_batch_fire`); consumer is the GPU `persistent_worker` cooperative kernel reading from `queue->inst[]` via global_load_b64.

**(b) Cast-strip bug (megakernel.hip:203) — fixed in commit 6bd6635.** The descriptor read used `const u64* src = (const u64*)(queue->inst + pc * INST_SIZE_WORDS);`. The `(const u64*)` cast SILENTLY STRIPS `volatile` from `volatile WorkerQueue* queue`. Disasm before fix: `global_load_b64 v[0:1], v[0:1], off offset:16` — NO glc/slc/dlc bits, plain L2-cached load. Disasm after fix (now via `braidinfer::rdna3::mailbox_load_descriptor` in `kernels/rdna3/rdna3_mailbox.h`): `flat_load_b64 v[0:1], v[0:1] offset:16 glc dlc` — L0/L1 + L2 invalidate-on-load. Source-level correctness fix; does NOT cure cold-start (see (c)). The disasm also revealed ~150 other non-flagged `global_load_b64` instructions in `persistent_worker` — audit task bd 20fp.

**(c) Falsification of shader L1/L2 invalidate (Exp 1a, N=30, BRAIDINFER_WARMUP_SKIP=1).** 16/30 first-attempt clean. Within statistical noise of the ~16/30 baseline (binomial 3σ ≈ ±5 for N=30). glc+dlc on the consumer descriptor load alone is NOT sufficient — the stale layer sits below L1/L2.

**(d) Falsification of CPU producer-side propagation (Exp 3, N=30).** Inserted `read_volatile` of the last descriptor word between `write_volatile(num_instructions)` and `write_volatile(seq_num)` in `dispatch_batch_fire`. Forces CPU writeback / memory-hub propagation to drain before consumer signal. Result: 10/30 — within noise of baseline. The producer-side analog of §11.18's Rule 10 fence is therefore FALSIFIED at the CPU. By symmetry the shader-side `fence_device()` after-store proposed in §11.18 is expected to give an equivalent result; not directly tested in this work because the producer is the CPU on this code path. Future GPU-as-producer pattern should empirically re-test before relying on Rule 10.

**(e) Falsified kernel-side interventions (linux-p2p, 2026-05-15 thru 18).** Patches 0014/0014v2 (MQD HDP flush) and 0015/0015v2 (KIQ ACQUIRE_MEM) all REGRESSED PASS rate to 0-3/10. 0014 series broke HIQ MQD privilege bits; 0015 series disrupted MES on concurrent compute via GC L2 invalidate touching in-flight worker queues. All reverted. Kept: patches 0012/0013 (proc_ctx_bo + gang_ctx_bo HDP flush) — close a real CPU→DRAM ordering gap, baseline ~60% PASS standalone.

**(f) Identified mechanism layer.** MES μC private cache or memory-hub-level state. Neither exposes an invalidate primitive to userspace shader or kernel-mode driver. Indirect evidence: kernel-side KIQ ACQUIRE_MEM (deepest reachable cache control) regressed without curing; shader-side glc+dlc (deepest in-shader cache control) did not cure; CPU producer-side drain did not cure. The remaining layer that none of these touch is the layer responsible.

**(g) Production cure: warmup-discard (commit a048318).** Run a sacrificial 4-token decode of `"Hello"` immediately after `Model::load`. If output is NaN-tail (3+ consecutive `!`), drop+reload the model and retry up to `BRAIDINFER_WARMUP_RETRIES` (default 5). 10/10 PASS at ~680ms cost. The CURE-STEP is the prefill MoE FFN dispatch — bd tm5t established this by ablating the 4-step decode tail: prefill alone + per-worker mailbox round-trip gives 30/30 in ~490ms. Whatever the MES μC / memory-hub state is, the first cross-GPU MoE FFN dispatch through each worker's mailbox drains it. Subsequent dispatches are clean for the lifetime of the process.

**(h) Temporal autocorrelation.** N=30 cold-starts with no warmup show clusters longer than chance: T16-T24 burst of 9 consecutive PASS in Exp 3, surrounded by failure clusters. Each trial is a fresh process (launch-gpu.py wraps each invocation), so PASID is recycled. Implies GPU-side state survives PASID teardown for some interval — kernel-side queue pool, KFD first-queue-init cache, or amdgpu scheduler internal state. Worth a wider-N characterization (Exp 5 planned).

**(i) Forward defense: rdna3_mailbox.h primitives.** `mailbox_load_descriptor<T>(const volatile T*) -> T` and `mailbox_store_descriptor<T>(volatile T*, T)`. Forces volatile-preserving load/store. Documented at point-of-use; prevents future cast-strip recurrence. Zero runtime cost. Migrated megakernel.hip:203 to use the helper in commit 6bd6635.

**(j) Audit task — bd braidinfer-20fp (CLOSED 2026-05-19).** ~150 non-flagged `global_load_b64` instructions remained in `persistent_worker` disasm at the time (j) was written. Each was potentially a cast-strip risk if it crosses a producer-consumer cache boundary. Audit method: trace each disasm offset back to source line, classify as (a) safe-purely-local, (b) cast-strip-risk-needs-migration, (c) deliberately-cacheable. **Audit completed in subsection (p) below** (2026-05-19): ~125 distinct source sites classified, zero new mailbox_load_descriptor migrations needed beyond the megakernel.hip:203 fix already landed in commit 6bd6635, one follow-up filed as bd braidinfer-1uak (coop_copy P2P L1-invalidate question, P3). bd 20fp closed.

**(k) Cross-references.**
- bd braidinfer-4e2m (top issue, this race)
- bd braidinfer-tm5t (mailbox-warmup A2 experiment, 30/30 with prefill)
- bd braidinfer-20fp (~150 non-flagged loads audit, P3)
- bd braidinfer-upxd (rc=134 deliberate fast-exit, CLOSED — see §11.x companion shutdown note)
- linux-p2p patches 0012/0013 (HDP flush, KEPT)
- linux-p2p patches 0014/0014v2/0015/0015v2 (FALSIFIED, reverted)
- braidinfer commits e83a059 (BDF dump) → a048318 (warmup-discard production cure) → 6bd6635 (cast-strip fix + rdna3_mailbox.h + this doc section)
- bridge thread 2026-05-19 messages #3203–#3232
- `kernels/rdna3/rdna3_persistent.h:40-41` (SYSTEM-scope atomic_load hangs)
- `kernels/rdna3/rdna3_peer.h:80` (host_uc_store_agent canonical signal-write)

**(l) Upstream filing TODO.** Material is publishable; even if AMD doesn't act, the public record helps anyone hitting the same wall. File on amd-gfx mailing list when bandwidth allows.

**(m) What an upstream fix would need to do.**

*Kernel.* An MES API to invalidate / pre-fault per-PASID private cache before first `ADD_QUEUE` on a new mailbox descriptor BO. Not currently exposed. Would live next to existing `set_resources` and `add_queue` MES packets in `drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c`. The kernel would call it as part of HWQ initialization, before the queue is made schedulable.

*Firmware.* MES μC could WB+inv its private cache (and any peer-GPU state it caches per `proc_ctx_addr`) on receipt of a `NEW_QUEUE` packet that includes a fresh `proc_ctx_addr`. Would need AMD firmware change. *Hardware.* A memory-hub-level fence that all consumers can issue — probably not present on gfx11 (would have been documented in the GFX11 ACE/ISA reference). RDNA4 (gfx12) adds `global_wb scope:SCOPE_SYS` and `global_inv scope:SCOPE_SYS` which together approximate this fence; on gfx1200 the bug should be self-resolved without warmup-discard.

**(n) Methodology note (original context — PROMOTED to §1.2).**

The 5-step falsification-cascade recipe was developed in this
investigation (bd 4e2m, 2026-05-19) and PROMOTED to top-level §1.2
"Falsification Methodology". See §1.2 for the full recipe; this entry
marks the original derivation context.

**(o) Cure is at the dispatch layer, not the prefill layer (Exp 1, 2026-05-19, commit 5a89a64+1).**

CORRECTION to (g)/(j) above. Prior analysis attributed the cold-start cure to "prefill MoE FFN dispatch" because bd tm5t showed 30/30 PASS with prefill + mailbox round-trip. That framing was wrong-direction: the load-bearing step was the MAILBOX DISPATCH itself, not prefill specifically. Prefill was incidentally exercising one mailbox transaction per worker per layer.

**Exp 1 (Lane C, skip-prefill warmup, N=30):**
  - Refactor: `Model::ensure_moe_workers_started` and `init_multi_gpu_persistent` exposed as `pub(crate)`. New `Model::minimal_mailbox_warmup_no_prefill` brings up all persistent workers (no prefill, no model forward pass) and dispatches ONE OP_D2D_COPY of 4 floats per worker mailbox + verifies the round-trip via host-mapped UC.
  - Gate: `BRAIDINFER_WARMUP_MODE=mailbox-only` (opt-in in Commit 1; default flip planned for Commit 2 after user spot-checks other model configs).
  - Result: 30/30 first-attempt-clean. WARN-NaN: 0/30. rc=0 (clean exit) 30/30.
  - Cost: spawn ~318ms (irreducible — MegakernelProgram::compile_multi_gpu_p2p + cooperative kernel launch) + dispatch ~1.5ms total for 4 workers. ~320ms total vs ~680ms for full-decode warmup. **54% faster cold-start.**

**Implication.** The 1.5ms dispatch is the actual cure. The ~318ms spawn time is unavoidable startup. Prefill (which previously fronted the warmup) was decoration — it exercised the same mailbox transactions but did 50+ ops worth of forward-pass overhead doing so. The cure-step is ANY mailbox dispatch through a fresh worker; the FIRST transaction per worker is the load-bearing event.

**Updated mechanism narrative.** Whatever the MES μC / memory-hub state is, it gets primed by the FIRST mailbox transaction the worker consumes after `ADD_QUEUE`. The transaction's payload contents don't matter (an OP_D2D_COPY with sentinel data suffices); only the mailbox-poll → descriptor-fetch → ack-write round-trip matters. Per-worker primer dispatches in 250-700µs.

**No change to (a)-(n)** other than this correction; the falsification cascade and methodology are unchanged. The §11.18 Rule 10 producer-fence hypothesis remains FALSIFIED (Exp 3, 10/30). The unreachable-layer identification (MES μC / memory hub) is unchanged. The shader-side cast-strip fix (b) is unchanged.

**Cross-refs.** bd 4e2m, bd tm5t (corrects framing — "prefill is the cure-step" → "mailbox dispatch is the cure-step; prefill incidentally exercises it"), Commit 1 of Lane C ladder.

**(p) Audit completeness (bd 20fp, 2026-05-19).**

Source-level audit of the ~150 non-flagged `global_load_b64` instructions observed in persistent_worker disasm (per (b) above). Classified by access pattern at the C++ source level rather than per-instruction (the loads are predominantly duplicates of the same source patterns inlined across virtual block iterations and the ~30 op_* helpers).

Categories:
  1. **LDS reads via `(const FooInst*)inst` casts (~30 sites, dominant).** Every op_* function (op_rmsnorm, op_linear_proj, op_silu, op_moe_gate, op_moe_dispatch, op_moe_ffn_remote, ...) starts with `const FooInst* i = (const FooInst*)inst;`. The `inst` pointer here is the per-pc LDS buffer populated by `mailbox_load_descriptor` (megakernel.hip:203, post-6bd6635). LDS reads are per-CU local; not cross-cache. **PURELY-LOCAL — safe.**
  2. **Same-GPU VRAM reads (~80 sites).** Weights (`DeviceBuffer<u16>`, `DeviceBuffer<u8>`), scratch buffers, GDN/Mamba2 recurrent state, KV-cache pages — all sit in the worker GPU's own VRAM. **PURELY-LOCAL — safe; cacheable by intent.**
  3. **Cross-GPU peer-VRAM reads with explicit invalidate (~5 sites).** `op_d2d_copy` at megakernel_ops.hip:1725 reads `src[idx]` from a peer GART/VRAM page; the `buffer_gl0_inv` + `buffer_gl1_inv` pair at lines 1717-1722 explicitly invalidates L0+L1 before the load. **DELIBERATELY-CACHEABLE WITH EXPLICIT INVALIDATE — safe; comment added at the load site in this audit to document why.**
  4. **Cross-GPU host-mapped UC reads (~10 sites).** `op_moe_dispatch` at megakernel_moe_dispatch.hip:96 (`src_expert_ids[j]`) and `op_moe_ffn_remote` at megakernel_moe.hip:360 (`eids[j]`) read from the gate-output buffer, which is allocated via `MappedHostBuffer::alloc_portable_coherent` and mapped with **MTYPE_UC** per the hsa-rocr-p2p-mtype-uc-gfx11 patch. Hardware bypasses L2 unconditionally for UC pages regardless of the glc bit — the non-volatile load is functionally equivalent to a glc load on these pages. **DELIBERATELY-CACHEABLE-AT-COMPILER-LEVEL-BUT-UNCACHED-AT-HARDWARE — safe.**
  5. **One flagged-not-migrated cross-cache touch (1 site).** `op_moe_ffn_remote` line 353: `coop_copy(la, act_p2p, gupd, ...)` reads worker activations from GPU 0 peer VRAM. Unlike (3), `coop_copy` does NOT issue an explicit L1 invalidate before the load. Flagged for future investigation. **DO NOT migrate to inline-asm slc without separate experiment** — Exp 1a (§11.19 (c)) established that glc+dlc on the descriptor load did not move the cold-start needle, so the in-stream P2P read may behave similarly. Filed as bd `<followup>` (see cross-refs).

**Counts.** ~30 LDS-cast sites + ~80 own-VRAM + ~5 peer-VRAM-with-invalidate + ~10 UC-hardware-bypass + 1 flagged = ~125 source-level distinct sites; the ~150 disasm instructions reflect that several of these are duplicated across virtual-block iterations and inlined op functions.

**Migrations done in this audit.** Zero new `mailbox_load_descriptor` migrations. The descriptor copy at megakernel.hip:203 was the only host-mapped-UC payload read with stripped volatile, and was migrated in commit 6bd6635 (§11.19 (b)). The remaining sites are either purely-local, explicit-invalidate, or hardware-UC-bypass.

**Comment additions.** One comment added at megakernel_ops.hip:1725 documenting why the non-volatile read in `op_d2d_copy` is safe (relies on explicit buffer_gl0_inv+gl1_inv preceding).

**Audit completeness assertion.** All persistent_worker access patterns from C++ source enumerated. No further cast-strip-class bugs identified. The audit is complete for the purpose of bd 20fp; the one flagged item (coop_copy in op_moe_ffn_remote) is a separate research question (not a known bug) tracked as a follow-up.

**New API category that did NOT emerge.** I expected the audit might reveal a need for a dedicated `peer_vram_load_with_inv<T>` primitive distinct from `mailbox_load_descriptor`. It did not — the explicit-invalidate-before-load pattern is sufficiently localized (only `op_d2d_copy`) that a one-line C++ comment was the right intervention. If more sites adopt the pattern, a typed helper becomes worthwhile; not yet.

**Cross-refs.** bd 20fp (this audit), bd 4e2m, bd 1uak (coop_copy P2P L1-invalidate follow-up), GFX1100_ARCH.md §5.4 / §11.18 / §11.19 (b), commit 6bd6635 (cast-strip fix), this commit (audit + op_d2d_copy comment).

**(q) Production state + D1 v1 lesson + INV_GART falsification (composite; three labeled sub-blocks below).**

**(q.1) Production state — mailbox-only default with single-GPU fallback (2026-05-19).**

**Default warmup behavior.** `BRAIDINFER_WARMUP_MODE` defaults to `mailbox-only`:
  - True multi-GPU MoE: spawn persistent workers (via `ensure_moe_workers_started` + `init_multi_gpu_persistent`, both `pub(crate)`), dispatch one `OP_D2D_COPY` of 4 floats per worker mailbox + verify via host-mapped UC round-trip. ~320ms cold-start cost (mostly worker spawn ~318ms; dispatch ~1.5ms). 30/30 first-attempt-clean on qwen35_35b_a3b.q4 -g 4 + qwen36_35b_a3b.q4 -g 4.
  - Single-GPU (incl. apply_auto_modes-selected single-GPU even with `-g N>1` requested): `minimal_mailbox_warmup_no_prefill` returns `Err("single-gpu-fallback")` early; `generate.rs` falls back inline to `greedy_generate("Hello", 4)` (the prior full-decode warmup). Behavior identical to pre-flip default for these configs. 10/10 first-attempt-clean across qwen35_27b -g 2 dense, qwen36_27b -g 4 dense, nemotron_cascade_30b -g 4 (all auto-selected single-GPU). The race itself does not manifest in single-GPU mode (no cross-GPU mailbox reads), so the fallback is a safety net for the single-process invariant.

Env opt-outs (preserved):
  - `BRAIDINFER_WARMUP_MODE=full-decode` — force full-decode warmup on all configs (prior default).
  - `BRAIDINFER_WARMUP_SKIP=1` — skip warmup entirely (diagnostic / testing only; cold-start race remains unprotected).
  - `BRAIDINFER_WARMUP_RETRIES=N` — max retries per cold-start before exit-100 (default 5).

**(q.2) D1 v1 deadlock lesson (preserved for future contributors).**

First attempt at single-GPU mailbox-only warmup (Lane 1 D1 v1) extended `init_multi_gpu_persistent` to handle the `multi_gpu == None` case by allocating a single-slot `PersistentDispatch` for GPU 0 and spawning its persistent_worker. This APPEARED clean: 10/10 trials decode-coherent, mailbox round-trip ~600µs, no NaN. BUT every trial exited rc=1 with `panic at decode/mod.rs:59: 'Option::unwrap() on a None value'`. Root cause: the persistent worker holds all CUs on GPU 0 once launched. The subsequent user prompt's `prefill` saw `persistent_workers.is_some() && !has_moe` and routed AROUND `prefill_paged` — which is the ONLY init path that lazily populates `paged_seq` / `page_allocator` / `megakernel_paged`. `decode_step_persistent` then panicked unwrapping `paged_seq`. The fundamental conflict: `prefill_paged` uses `hipMemcpy` to populate paged chunks, and `hipMemcpy` deadlocks once a cooperative kernel is holding all CUs. So spawning the worker before prefill is mutually exclusive with the existing init flow.

**D1 v2 resolution (current commit).** Sentinel-based fallback: `minimal_mailbox_warmup_no_prefill` returns `Err("single-gpu-fallback")` when `multi_gpu.is_none()`. `generate.rs` detects the sentinel and runs `greedy_generate("Hello", 4)` inline. No worker spawn before prefill; no `hipMemcpy` deadlock; behavior identical to pre-flip for single-GPU configs.

**Forward lesson:** spawning the persistent worker BEFORE prefill is architecturally incompatible with the current paged-KV lazy-init pattern on this codebase. If a future contributor wants to unify warmup further, the path is to refactor `prefill_paged` to use cooperative-kernel dispatch instead of `hipMemcpy` (significant work), not to spawn the worker earlier.

**(q.3) INV_GART kernel-patch experiment — FALSIFIED at MES firmware level (2026-05-19, linux-p2p tree).** Sub-agent investigation surfaced `MESAPI_MISC__INV_GART` opcode declared in `mes_v11_api_def.h:574` with no caller anywhere in amdgpu. `kfd_chardev.c:2814-2820` has AMD's own comment naming "stale process context data saved in MES" as a known phenomenon with `SET_SHADER_DEBUGGER` as workaround — matches our cold-start race 1:1.

Kernel patch (in linux-p2p tree): new `amdgpu_mes_inv_gart()` wrapper + `MES_MISC_OP_INV_GART` enum + `mes_v11_0_misc_op` case + call from `add_queue_mes` gated on `qpd->queue_count == 0` (first queue per PASID per device). Built clean. Symbol `amdgpu_mes_inv_gart` confirmed in `/proc/kallsyms` after install.

N=30 cold-start qwen35_35b_a3b.q4 -g 4 BRAIDINFER_WARMUP_SKIP=1:
  - decode-clean:   **10/30** (WORSE than baseline ~16/30)
  - dmesg lines matching "MES failed to respond" or "MES might be in unrecoverable state": **351**
  - Process crashes (rc=250): **3/30**
  - Zero `amdgpu_mes_inv_gart failed` advisories (the call ITSELF didn't return error; the failure cascades downstream in MES firmware)

Pattern matches the previously falsified linux-p2p 0014/0015 series: invalidate-MES-state interventions disrupt the firmware when called concurrently with in-flight queue management. The fact that the call is gated on first-queue-per-PASID doesn't help — PASIDs are recycled across processes and MES is concurrently servicing other compute queues across all 4 GPUs.

**Falsification verdict:** the `MESAPI_MISC__INV_GART` opcode as currently exposed on gfx11/MES1 firmware causes an MES unrecoverable-state cascade. AMD's reason for shipping-the-opcode-but-not-wiring-it is now empirical fact. Kernel patch reverted; production kernel runs 0012+0013 only.

**Upstream amd-gfx filing TODO (highest-value forward action).** Subject: "MESAPI_MISC__INV_GART opcode causes MES unrecoverable-state cascade on gfx11/MES1 when called from add_queue_mes". Include: motivation from `kfd_chardev.c:2814` + observable cold-start NaN race; approach (wire INV_GART); result (351 MES failures + 3 crashes on 30-trial run); reproducer (braidinfer cold-start with `BRAIDINFER_WARMUP_SKIP=1` + the patch); question to AMD (is INV_GART intended safe-to-call from `add_queue_mes`, or firmware-side broken; if latter, what IS the correct path to clear MES per-PASID cached state on first-queue setup).

**Cross-refs.** bd 4e2m (top), bd 20fp / 1uak / 1i1c / iwn0 (follow-ups), this commit, linux-p2p tree (INV_GART patch reverted but preserved in branch history for upstream filing reference), bridge thread #3239-#3257.

---

**(r) [Merged from §11.16] UC dst alone is insufficient — intra-kernel cached-read staleness (2026-05-17) — status PROVISIONAL (mechanism undetermined; working mitigation active).**

During the braidinfer `r7dv` chase on 2026-05-17, the multi-GPU NaN
class was localized to the first segment after the first
head-parallel attention gather. Reference: bridge log #377, #381.

The receiving buffers (`act.attn_out`, `act.gate_attn`) are allocated
via `hipExtMallocWithFlags(hipDeviceMallocUncached)` — i.e. they are
**Rule 1a (UC device)** per §5.5. They should bypass L2. Yet:

  - 5-decode-step output: token 0 correct (prefill writes), tokens 1+
    100% NaN. NaN first appears at `moe-pre L3` on qwen3.6_35b_a3b.q8
    2-GPU and at `moe-pre L8` on nemotron_super_120b.q4 4-GPU.
    Position-invariant: ALWAYS the first segment after the first
    attention-pre, regardless of model architecture or worker count.
  - Inserting a CPU mailbox round-trip between gather batch and
    consumer batch (split into two megakernel dispatches with
    `dispatch_batch_slice` boundary) — i.e. an implicit Rule 1d
    kernel-boundary sync — eliminates the NaN. Output becomes
    `" Paris."`, all 40 layers × 2 tokens clean.
  - Per `dump_mtype_audit`, `act.attn_out` reportedly shows
    `mem_type=2 alloc_flags=0x0` (cached) at runtime, contradicting
    the alloc-site call. This audit may be stale or probing the wrong
    buffer; a runtime `hipPointerGetAttributes` probe at the actual
    pointer (mandatory for r7dv close) is pending.

**(s) [Merged from §11.16] Three hypotheses (to be disambiguated by the runtime MTYPE probe).**

  (i) **RDNA3 per-WGP L1 caches MTYPE=UC lines.** If the writer (gather
  D2D) executes on WGP-A and the consumer (`op_output_gate` /
  residual+rmsnorm) executes on WGP-B, B's L1 may hold a stale line
  from the prior decode step. The `hipDeviceMallocUncached` MTYPE
  bypasses L2 but per-WGP L1 may cache regardless. Rule 1a's promise
  "bypass L2" would not cover this case. If true: §5.5 Rule 1a needs
  a qualifier — UC bypasses L2 but not L1 on gfx1100; intra-kernel
  cross-WGP writer→reader on UC buffers needs either a kernel-boundary
  sync (Rule 1d analog via batch-mailbox round-trip) or a `buffer_gl1_inv`
  on the reader (Rule 3 caveat: this is the case where `buffer_gl1_inv`
  DOES help, because the staleness is at L1).

  (ii) **Writer-side cached store to UC dst.** Even if dst MTYPE=UC, if
  the writer kernel emits `buffer_global_store_dwordx4` (cached) rather
  than a UC-aware store intrinsic, the writer's L1 may hold the
  prior-iteration's value of that line, and the consumer reading from
  the same WGP's L1 sees stale. The deleted comment at
  `braidinfer/crates/braidinfer-runtime/src/multi_gpu.rs:80-86` documented
  this exact pattern for `DeinterleaveInst` writes to host-mapped UC
  pages. If true: a Rule 10 is needed — "writer's store-path must be
  UC-aware when dst is MTYPE=UC."

  (iii) **`hipExtMallocWithFlags(hipDeviceMallocUncached)` does not
  produce MTYPE=UC on ROCm 7.2.x.** Verifiable by runtime
  `hipPointerGetAttributes`. If true: this is a ROCm-version-specific
  regression worth filing upstream; mitigation is to use
  `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` (130× slower
  cross-agent observe per the §5.5 Rule 1 allocation method table) or
  `MappedHostBuffer` (host-mapped UC) for any persistent-kernel-shared
  cached-write-target buffer.

**(t) [Merged from §11.16] Working mitigation.**

Split the gather batch and the consumer batch
into two `dispatch_batch_slice` dispatches with an intervening CPU
mailbox round-trip. Cost: one mailbox RTT per attention layer
(per `RDNA3_PERF_MEGAKERNEL_DISPATCH_MEDIAN_US ≈ 2.79 µs`). On a 40-layer
model with attention every 4th layer, that's ~10 attention boundaries ×
2.79 µs ≈ 28 µs per token of decode overhead. Acceptable cost given
the alternative is 100% NaN.

**(u) [Merged from §11.16] Surface census (pattern lineage).**

Today's r7dv brings the count of
distinct buffers requiring §5.5 Rule 1 promotion in the braidinfer
persistent-kernel architecture to seven, distributed across the snl /
vo0 / r7dv investigations:

| Buffer | Class | First fix | Rule |
|---|---|---|---|
| `normed_stage` | per-GPU RMSNorm output staging | snl `5f1d745` | Rule 1b -> portable-coherent |
| `worker.attn_out` | head-parallel attn output | snl `3da5618` | Rule 1b -> portable-coherent |
| `worker.output_slots` | MoE worker output ring | snl                   | Rule 1b |
| `worker.moe_act_uc_handoff` | MoE expert input handoff | vo0 `8f8c0e4` (today) | Rule 1b |
| `attn_gate` | gate side of GqaAttn | (still TODO post-r7dv) | TBD |
| `act.attn_out` (GPU 0) | gather destination | r7dv `(C) split` (today) | TBD per (i)/(ii)/(iii) |
| `act.gate_attn` (GPU 0) | gate destination | r7dv `(C) split` (today) | TBD per (i)/(ii)/(iii) |

This census is consistent with Rule 8 ("audit ALL persistent buffers,
not just activation flow") — every persistent buffer crossing a
write-then-read boundary inside the persistent kernel is a candidate.
The remaining intra-kernel writer→reader pattern (last three rows) is
distinct from the cross-agent pattern (first four rows) and is the
class this investigation documents.

**(v) [Merged from §11.16] Cross-engine confirmation and lesson.**

llama.cpp encountered the same §11.4
mitigation pattern on its HIP backend during the 2026-05-17 SSM /
prefix-cache work:

  - `44855024b` — fattn-sp peer FA output writeback migrated to SDMA
    via a UC staging slab. Rule 1a (UC device) applied to a previously
    cached peer-buffer.
  - `878bc9df0` — MoE_FUSED SDMA combine path routed through a
    per-secondary UC writer slab, with `ep_sync_signal_kernel` fence
    deleted. Rule 1a applied; fence dropped because UC bypasses the
    coherence requirement it was guarding.
  - `6161adee5` — server warn-once when prompt cache silently no-ops in
    KV-shard mode. (Separate concern, surfaced same day.)
  - `29c424825` — auto-default `GGML_KV_SHARD_READ=0` for
    hybrid/recurrent models pending bd `llamacpp-cb5` root cause. (SSM
    attractor lock-in under shard-mode reductions; separate concern
    from this finding but documented same day.)

Pattern: Rule 1a / 1b promotion is the structural fix for any
cross-agent buffer the persistent / cooperative kernel touches. The
audit framework should be type-level (a `CrossGpuStaging<T>` Rust
abstraction is planned in braidinfer's `t8fl` epic; the analogous
encoding in llama.cpp's HIP backend would be a per-buffer-class
allocator wrapper).

Lesson: MTYPE=UC is necessary but provisionally not sufficient on
gfx1100 within persistent-kernel execution. The §5.5 Rule 1 allocation
table guarantees cross-AGENT coherence (Rule 1's framing) but the
intra-kernel cross-WGP behavior on UC is not yet characterized.
Treat any new persistent-buffer addition that crosses
`dispatch_batch_slice` boundaries as a review item for §11.19 (r)-(v)
until the runtime MTYPE audit and mechanism determination are complete.

Pending work to close: (1) braidinfer runtime MTYPE probe at model init
for `act.attn_out`, `act.gate_attn`, `worker.attn_out`, `worker.attn_gate`
— disambiguates (i)/(ii)/(iii), r7dv close-gate. (2) Once mechanism known,
§5.5 updated with appropriate Rule 1a qualifier, Rule 10, or ROCm-version
note. (3) Until then, working mitigation is canonical fix.

---

**(w) [Merged from §11.18] Trigger and observed data (2026-05-18) — status: G9 producer-fence hypothesis FALSIFIED (see (d) above).**

GPU 0 RMSNorm writes per-token activations into `normed_stage`, allocated via
`MappedHostBuffer::alloc_portable_coherent` (`hipHostMallocMapped | hipHostMallocPortable
| hipHostMallocCoherent`, flags `0x40000003`). Worker GPUs (GPU 1+) then read `normed_stage`
through the GART (host-mapped page) during `D2dCopyInst` broadcast. Under a subset of
cold-start trials (~10-20%), the worker observes partially-stale data: a 128-fp32
head-aligned cluster of NaN / zero values at the start of the buffer, with correct data
in the tail. In the failure-path extreme, the UTCL2 TLB controller on the worker GPU
issues a TCP PERMISSION_FAULT (UTCL2 error code `PERMISSION_FAULT=0x3, MAPPING_ERROR=0x0`)
against a GART page address.

Observed data (braidinfer sm16 investigation, boot of 2026-05-18):

- 28/28 gfxhub page faults this boot originated from braidinfer `generate` process.
- Fault type: `UTCL2 TCP PERMISSION_FAULT=0x3 MAPPING_ERROR=0x0` — kernel attempted
  read/write of a GPU VA that was either unmapped or had stale TLB state.
- Page-fault storm -> MES `REMOVE_QUEUE` timeout -> `MODE1 GPU reset` -> VRAM loss ->
  llama-server processes on the same GPU receive `HSA HW Exception GPU_HANG` ->
  `HwExceptionHandler` -> abort.
- 94 GPU reset events total this boot. 5 llama-server coredumps, all with identical
  victim-path stack.
- Failure rate at consumer: ~10-20% per cold-start trial (`H-B` alone showed 8/20 PASS).
  Per-trial PASS/FAIL is binary; no gradual degradation.
- NaN cluster geometry: exactly 128 FP32 values (512 bytes) aligned to a GPU 0 L2
  cache-line cluster boundary (RDNA3 cache line = 128 B; 512 B = 4 lines). Fresh data
  appears in the tail. This is the L2-dirty-line-cluster fingerprint documented in §5.4.

**(x) [Merged from §11.18] Mechanism (G9 producer-fence hypothesis) and coherence analysis.**

`alloc_portable_coherent` flags force `MTYPE=UC` (uncached) on the allocating GPU at
allocation time. However, on gfx1100 with `hipHostMallocPortable`, the MTYPE seen from
the allocating GPU's perspective is `0x40000001` (fine-grained coherent host memory) —
write-back + hardware snoops, NOT MTYPE=UC. This is distinct from
`hipExtMallocWithFlags(hipDeviceMallocUncached)` (MTYPE=0x3, pure UC on device side)
documented in §5.5 Rule 1a.

Critical difference: a `MappedHostBuffer::alloc_portable_coherent` buffer is GART-mapped
host memory. When GPU 0 writes to it via normal vector stores (`buffer_store_dwordx4`),
those stores enter GPU 0's L2 as write-back dirty lines. The hardware snoop mechanism is
designed to keep the host-side view coherent, but the snoop is triggered on CPU-side
access or on the next hardware-managed eviction — NOT on GPU-side peer reads through GART.

  1. GPU 0 RMSNorm writes to `normed_stage` (GART-backed, write-back + snoops).
  2. GPU 0's L2 holds the fresh bytes as dirty write-back lines.
  3. Worker GPU reads `normed_stage` through GART (P2P path via PCIe bridge).
  4. If GPU 0 has NOT flushed those L2 lines to host DRAM yet, the GART read from the
     worker hits the physical host-DRAM address before GPU 0's dirty lines have landed.

Two sub-outcomes: (a) stale-data path — worker reads stale bytes, observable as the
128-fp32 head-NaN pattern; (b) fault path — if GART page TLB entry on the worker side
hasn't been committed yet, the worker's UTCL2 issues a TCP PERMISSION_FAULT.

Why intuition is wrong: "host-mapped coherent" does NOT equal "write-through" on RDNA3.
`hipHostMallocCoherent` refers to CPU-GPU coherence via hardware snoops — guarantees the
CPU can observe GPU writes, but NOT that GPU-to-GPU peer reads through the GART observe
the latest GPU-0 dirty bytes. There is no GPU-to-GPU snoop path through PCIe on gfx1100.

Contrast with `hipExtMallocWithFlags(hipDeviceMallocUncached)` (device-side MTYPE=UC):
writes bypass L2 entirely, going directly to VRAM/HBM. Peer reads via P2P then read from
VRAM (already up-to-date). This is why UC device buffers (Rule 1a) work for cross-GPU
spin-wait but host-mapped portable-coherent buffers do not without producer-side fencing.

**(y) [Merged from §11.18] Required producer fence and consumer cache state (G9 hypothesis — FALSIFIED by §11.19 (d), preserved for record).**

GPU 0 must issue `__threadfence()` (agent-scope fence, equivalent to
`__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent")`, see `rdna3_barrier.h:96-99`) at
RMSNorm exit, BEFORE the ack/signal that triggers worker reads. This drains GPU 0's L2
dirty lines to DRAM so that the GART-mapped host address holds the fresh values when
workers read via PCIe.

NOTE: `__threadfence_system()` (system-scope fence) HANGS on gfx1100 (see §5.3 and
`rdna3_barrier.h:102-113`). The agent-scope fence `__threadfence()` is the correct
primitive here because the goal is to push GPU 0's L2 dirty lines to host DRAM.

Consumer cache state: if a worker GPU has previously read `normed_stage` (prior iteration,
same GART page), its L1 may hold a stale line. Before re-reading `normed_stage`:

```asm
buffer_gl0_inv      // invalidate per-SIMD vector cache (L0)
buffer_gl1_inv      // invalidate per-WGP shader-array cache (L1)
s_waitcnt vmcnt(0)  // wait for the invalidation to complete
```

Note: `buffer_gl2_inv` does not exist on gfx1100 (see §5.3). L2 staleness for GART reads
is handled by the producer-side fence (above), not by consumer-side L2 invalidation.

TLB/mapping commit timing hazard: `alloc_portable_coherent` pages are demand-mapped. On
gfx1100, a GART page may not have its GPU VA TLB entry committed in a peer GPU's UTCL2
until first touch. The `PERMISSION_FAULT=0x3 MAPPING_ERROR=0x0` event indicates the page
table entry exists but the read is attempted before the entry is fully propagated.
Mitigation: ensure worker GPU contexts access `normed_stage` at least once (dummy read)
during model-load / init time to force UTCL2 TLB entry establishment.

**(z) [Merged from §11.18] SAFE vs UNSAFE code patterns.**

UNSAFE (triggers NaN / fault at ~10-20% cold-start rate):

```cpp
// GPU 0 RMSNorm kernel exit — NO fence
__global__ void rmsnorm_kernel(..., float* normed_stage, ...) {
    normed_stage[idx] = result;      // write-back into GPU 0 L2
    // kernel exits: L2 dirty lines NOT guaranteed flushed to host DRAM
}
// host: signal workers; workers read normed_stage via GART
// => workers may read stale host DRAM (dirty bytes still in GPU 0 L2)
```

SAFE (G9 fix — `__threadfence()` at RMSNorm exit; note G9 hypothesis FALSIFIED for
cold-start cure but pattern may still be correct engineering for other GPU-as-producer paths):

```cpp
__global__ void rmsnorm_kernel(..., float* normed_stage, ...) {
    normed_stage[idx] = result;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence();             // == __builtin_amdgcn_fence(ACQ_REL, "agent")
        status_word[0] = RMSNORM_DONE;
    }
}
```

Consumer-side worker (with L1 invalidation before re-read):

```cpp
asm volatile(
    "buffer_gl0_inv\n\t"
    "buffer_gl1_inv\n\t"
    "s_waitcnt vmcnt(0)\n\t"
    ::: "memory");
float val = normed_stage_dev_ptr[idx];
```

**(aa) [Merged from §11.18] Library integration, call sites, §5.5 Rule 10 proposal, and RDNA4 context.**

Library: the proposed `rdna3_coherent_host_publish.h` extension (D1 design) wraps these
patterns into `RDNA3_COHERENT_WRITE_PUBLISH(buf, status, done_val)` (producer side) and
`RDNA3_COHERENT_WAIT_READ(buf, status, expected_val)` (consumer side). See kb entry tagged
`rdna3-coherent-publish-wait-design-2026-05-18` for full API and macro signatures.

Call sites to migrate in braidinfer:
- `normed_stage` write exit in `megakernel_f32` / `persistent_worker` RMSNorm dispatch:
  add `fence_device()` (from `rdna3_barrier.h:96`) after final store, before writing ack
  status. See `rdna3_persistent_protocol.h:141-149` for canonical ack pattern.
- `D2dCopyInst` consumer path for `normed_stage`: add `buffer_gl0_inv` + `buffer_gl1_inv`
  before the first GART load on each new dispatch (worker GPU's copy kernel, not host).
- `moe_p2p.rs` allocations using `alloc_portable_coherent` for `output_slots` and
  `moe_act_uc_handoff` (see `moe_p2p.rs:220,239`): audit whether writer GPU needs
  `fence_device()` before posting the ack.
- Add a UTCL2 TLB warm-up pass during `MoeP2pState::new` init (after
  `hipDeviceEnablePeerAccess`, before first inference) to establish TLB entries.

§5.5 Rule 10 proposal — **FALSIFIED**. The Rule 10 producer-fence hypothesis
was tested at the CPU producer analog in Exp 3 (10/30 PASS = baseline noise
floor; see subsection (d) above). Per the identified mechanism layer in
subsection (f) — MES μC private cache or memory-hub state, below
userspace-shader/kernel-mode reach — the GPU-as-producer analog is expected
to fail equivalently and is therefore **not separately tested** (the
producer is the CPU on this code path; testing the GPU-as-producer analog
would require synthesizing a code path we don't ship). DO NOT re-propose
Rule 10 as a cold-start cure. The original proposal text follows for the
historical record, including the consumer-side `buffer_gl0_inv` +
`buffer_gl1_inv` pair which remains structurally correct hygiene for cached
peer-VRAM reads (see §11.18 (y) merged into here) even though it is NOT
sufficient for cold-start cure:

> "For any `alloc_portable_coherent` (or equivalent GART-backed
> host-mapped) buffer written by a GPU kernel and read by a PEER GPU
> kernel, the GPU writer MUST issue `fence_device()` (agent-scope fence)
> after the final store and before signaling the consumer.
> `__threadfence_system()` is FORBIDDEN on gfx1100 (§5.3). The
> `buffer_gl0_inv` + `buffer_gl1_inv` sequence on the consumer side MAY
> additionally be required if the consumer runs in a persistent kernel
> that has previously cached the same GART page."

Cross-process blast radius: a UTCL2 fault from any process triggers MES `REMOVE_QUEUE`
timeout, then `MODE1 GPU reset`, which drops ALL queues on that device including unrelated
processes (e.g., llama-server). See §11.13 for the cooperative-grid relaunch wedge
(different mechanism, same MES-storm fingerprint).

RDNA4 context: gfx12 adds `global_wb scope:SCOPE_SYS` and `global_inv scope:SCOPE_SYS`.
On gfx1200, `__threadfence_system()` is expected to function. This workaround is
gfx1100-specific.

Evidence chain reference: braidinfer sm16 investigation, G9 hypothesis, p4 test (2026-05-18).
kb entry `gpu-hang-cascade` (if present).

---

**Subsection cross-reference (post-merge backward-compat):**

- old §11.16 (2026-05-17) symptom and localization -> §11.19 (r)
- old §11.16 three hypotheses -> §11.19 (s)
- old §11.16 working mitigation -> §11.19 (t)
- old §11.16 surface census -> §11.19 (u)
- old §11.16 cross-engine confirmation and lesson -> §11.19 (v)
- old §11.18 trigger and observed data -> §11.19 (w)
- old §11.18 mechanism and coherence analysis -> §11.19 (x)
- old §11.18 required producer fence and consumer cache state -> §11.19 (y)
- old §11.18 SAFE vs UNSAFE patterns -> §11.19 (z)
- old §11.18 library integration, call sites, Rule 10, RDNA4 context -> §11.19 (aa)
- old §11.19 (a)-(q) -> §11.19 (a)-(q) [unchanged]

**(ab) Cure mechanism narrowed: poll-barrier first-iteration, not opcode-specific (2026-05-20, bridge thread #3322–#3331).**

The (g)/(o) framing established "mailbox dispatch is the cure-step, not prefill". This subsection narrows further: **the cure is `persistent_iter_poll_barrier` first-iteration success on the worker, not any specific opcode dispatched within that batch.**

**Falsification of host-side weight-coherence cure (Angle B, N=10).** Added `hipDeviceSynchronize()` per worker GPU at end of `init_split_attn_weights` (after the `copy_weight_slice` loop, before `init_multi_gpu_persistent`). Hypothesis: synchronous fence on the dst device flushes any pending peer SDMA writes to L2 before persistent_worker reads them. Result: 0/10 PASS with `BRAIDINFER_WARMUP_SKIP=1` on qwen35_35b_a3b.q4 -g 4. Reverted. **Falsifies "weight L2 staleness is the cure target".** Mechanism: `hipDeviceSynchronize` only orders host-visible ops on the calling device; the worker's own L2 is touched lazily on first VRAM read, AFTER the sync.

**Probe (a) variant — empty packet (N=10, 10/10 PASS).** New `Model::minimal_mailbox_warmup_empty_packet` dispatches `dispatch_batch_fire(gpu_idx, &[])` per worker. The worker exits `persistent_iter_poll_barrier` on `seq > last_seq`, reads `num_inst=0`, skips the inner instruction loop, ack=seq. No opcode executes. 10/10 PASS on qwen35_35b_a3b.q4 -g 4 cold-start `BRAIDINFER_WARMUP_MODE=empty-packet`. Per-trial: `warmup-empty-packet attempt 1: OK — spawn=304-379ms empty-dispatch=1.46-1.49ms`. Latency essentially identical to D2D_COPY variant (~1.25ms).

**MES log corroboration (udi bridge #3331).** Post-run MES_INTR_ME_1 count across 8 cards × 10 trials averaged **2.3/trial — firmly in PASS territory** (preserved-data baselines: PASS=3.3, FAIL_nan=6.8, FAIL_rc250=31.4). Zero FAIL-class dmesg signatures. The empty-packet cure produces a PASS-class MES signature, not a post-hoc-corrected FAIL signature.

**Updated mechanism narrative.** Whatever the MES μC / memory-hub state is, the FIRST poll-barrier exit per worker primes it. The worker's first `seq > last_seq` transition includes an `atomic_block_barrier` with `__threadfence_system`-class ordering that either (i) primes the worker's L2 walkers for subsequent VRAM reads, or (ii) flushes MES μC per-PASID state via the side effect of MES processing the ack write. Either way, the cure is dispatch-layer-first-iteration, not opcode-payload-specific.

**Production change (2026-05-20 commit follow-this-section).** `BRAIDINFER_WARMUP_MODE` defaults to `empty-packet`. Opt-outs:
  - `BRAIDINFER_WARMUP_MODE=mailbox` — D2D_COPY variant (preserved for diagnostic comparison; equivalent latency)
  - `BRAIDINFER_WARMUP_MODE=full-decode` — full 4-token decode warmup (pre-flip default)
  - `BRAIDINFER_WARMUP_SKIP=1` — skip warmup entirely (testing only)

**Optional principled fix (deferred).** Adding an explicit `buffer_inv` / `__threadfence_system` at `persistent_worker` entry (one-shot at outer-loop entry, before the per-batch loop) might eliminate the need for any warmup packet — the spawn itself would prime the worker. Not implemented in this commit; tracked as a follow-up. The empty-packet cure is sufficient for production and ~1.5ms is acceptable cold-start overhead on top of the unavoidable ~300ms spawn time.

**Cross-refs.** bd 4e2m, bridge thread 2026-05-20 #3321–#3331 (udi Angle B falsification, Probe A withdrawal, Probe (a) variant design), commit (this) — empty-packet default + this subsection.

---

## Appendix A — Falsified hypotheses (do not retry)

Cross-investigation falsified-hypothesis index covering bd 4e2m
(cold-start mailbox race) AND §11.13 (cooperative-grid relaunch wedge)
AND adjacent investigations. Preserved so future investigators don't
re-run dead-ends. Grouped by source investigation to make ownership
clear. Each entry: hypothesis, verdict, where to read the falsification
record.

### bd 4e2m falsifications (cold-start mailbox race)

- `global_load_b32 ... glc dlc` on consumer descriptor (Exp 1a). 16/30
  PASS = baseline noise floor. Volatile-cast → L0/L1/L2
  invalidate-on-load is necessary correctness hygiene but NOT sufficient
  for cold-start cure. See §11.19 (c).
- CPU `read_volatile` of last descriptor word between
  `num_instructions` and `seq_num` writes (Exp 3). 10/30 PASS = baseline
  noise floor. Refutes §11.18 Rule 10 producer-fence hypothesis at the
  CPU analog. See §11.19 (d).
- `MESAPI_MISC__INV_GART` opcode wired from `add_queue_mes` on
  first-queue-per-PASID-per-device (2026-05-19). 10/30 PASS PLUS 351
  `MES failed to respond` / `MES might be in unrecoverable state`
  advisories + 3 process crashes on N=30 cold-starts. REGRESSION.
  Opcode declared in upstream `mes_v11_api_def.h:574` but firmware-side
  broken on gfx11/MES1 in this call context. AMD shipped it but did NOT
  wire it for a real reason. See §11.19 (q.3); reverted in linux-p2p
  tree; also documented in `kernels/rdna3/rdna3_compat.h` as FORBIDDEN.
- linux-p2p patches 0014 / 0014v2 (MQD HDP flush during init_mqd_hiq).
  REGRESSED to 0-3/10. Broke HIQ MQD privilege bits. See §11.19 (e),
  `~/builds/linux-p2p/0014*.patch.reverted`.
- linux-p2p patches 0015 / 0015v2 (KIQ ACQUIRE_MEM with GC L2
  invalidate). REGRESSED to 0/10 (and 5/5 only with warmup-discard
  already active, indicating near-zero kernel-side contribution).
  Disrupted MES on concurrent compute. See §11.19 (e),
  `~/builds/linux-p2p/0015*.patch.reverted`.
- "§11.18 Rule 10 producer-fence (fence_device after store) is the
  systematic fix for cross-cache GART staleness". FALSIFIED by Exp 3 at
  the CPU producer analog. The cure layer is below userspace; MES μC
  private cache or memory-hub state. See §11.19 (d), (f), (y), (aa).
- "MTYPE_UC on the destination buffer alone is sufficient for cross-
  cache reads" (early §11.16 framing). SUPERSEDED — UC on the dst is
  necessary but the FIRST cross-cache read from a fresh worker still
  reads stale state on cold-start. See §11.19 (r)-(v) (merged §11.16).
- "Prefill is the cold-start cure-step" (bd tm5t framing, 2026-05-19
  morning take). REFINED: Lane C Exp 1 (commit 38b05b1) showed
  worker-spawn + 1 mailbox round-trip (NO prefill) = 30/30 at ~320ms.
  Prefill was decoration; the actual cure-step is ANY mailbox dispatch
  through a fresh worker. See §11.19 (o).
- Inline-asm `slc` bypass on consumer load: NOT RUN end-to-end (per
  udi directive: not worth experimenting given Exp 1a result).
  Recorded as DEFERRED, not falsified.

### §11.13 falsifications (cooperative-grid relaunch wedge)

Userspace probes (all WEDGE persists):
- `buffer_gl0_inv` + `buffer_gl1_inv` before poll-address load.
- `global_load_b32 ... glc dlc` on poll address.
- `global_atomic_or_b32` with mask 0 on poll address.
- Wide 16-byte loads, inject 128 v_mad cycles between polls, cache-line
  isolation (60-byte pad).
- CPU `_mm_mfence`, `_mm_clflush + mfence` after host-mapped write.
- `hipHostMallocCoherent`, `hipHostRegister` mapping-mode variants.
- `mmap(MAP_HUGETLB)`, fresh-page mmap.
- Stream rotation (`hipStreamDestroy` + `hipStreamCreate` between trials).
- Non-cooperative-kernel interleave between trials.
- 10-100ms zero-queue-delay between trials.

Kernel-side patches (all WEDGE persists):
- MES MISC `NOTIFY_TO_UNMAP_PROCESSES` (Design D).
- MES MISC `SET_SHADER_DEBUGGER` (Design F); HIP already uses the
  first SHADER_DEBUGGER call at `hipInit`.
- Raw MES `REMOVE_QUEUE` + `ADD_QUEUE` with `skip_process_ctx_clear=0`
  override (Design G).
- Raw REMOVE-all + ADD-all reaching "last gang in process" condition
  (Design H).
- Per-PASID heavy `amdgpu_gmc_flush_gpu_tlb_pasid` + Design H.

Configurations:
- `amdgpu.cwsr_enable=0` (helps SOME framework users for adjacent wedge
  classes). REJECTED for our case via empirical reboot test.

All §11.13 entries cross-reference `mes/MES_PROBE_ARCHIVE.md` for the
full probe table and dmesg evidence.

Framing:
- "Rule 9 violation fully explains the §11.13 wedge" (2026-05-14
  morning take). WRONG. Standalone reproducer with ZERO prior
  cooperative launches wedged identically. Actual root cause was the
  Phase 2' deferred-ack protocol deadlock (§11.15). Rule 9 itself is
  still valid as a separate guideline; it just isn't the §11.13
  mechanism. See §11.15 + §11.13 stub.
- `hsa-rocr-coop-force-destroy-gfx11.patch` (obsolete approach).
  REJECTED due to double-free risk. See bridge thread / linux-p2p
  history.

### Common reasoning pitfalls (apply across both investigations)

- "Partial improvement at one intervention class is additive evidence
  that combining classes will cure" — NO. If glc+dlc gives 16/30 and
  CPU read-back gives 10/30 and baseline is ~16/30, the 16/30 is the
  noise floor, not a stack of partial cures. See §11.19 (n) and §1.2
  methodology recipe.
- "AMD-declared opcode in `mes_v11_api_def.h` implies firmware support"
  — NO. INV_GART is exhibit A. Treat declared-but-unwired opcodes as
  N/A unless empirically validated on hardware. See §11.19 (q) +
  `kernels/rdna3/rdna3_compat.h` FORBIDDEN section.
- "First-coop-launch wedge on a fresh process can be explained by Rule
  9" — NO. See §11.13 historical postscript.

**What is NOT in this appendix**

Falsified hypotheses INTERNAL to a development sprint that did not
result in a published mitigation or test artifact are documented within
the §11.x subsections themselves rather than promoted here. This
appendix covers the cross-investigation thread that produced testable
patches or published probe results.

**Cross-references.** bd braidinfer-4e2m (top investigation issue);
bd 20fp / 1uak / 1i1c / iwn0 / ew00 / upxd (follow-ups);
linux-p2p tree (kernel patch history);
`kernels/rdna3/rdna3_compat.h` (FORBIDDEN block);
`mes/MES_PROBE_ARCHIVE.md` (extracted §11.13 probe data).
