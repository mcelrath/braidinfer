# GFX1100 Cache Hints — Measured Microbenches (extracted from GFX1100_ARCH.md §5.0)

This file contains the measured microbench content extracted from
`GFX1100_ARCH.md` §5.0 to keep the main arch doc focused on operational
guidance. The data here is GLC/SLC/DLC empirical behavior on RX 7900 XTX
(gfx1100), captured by the suite in `test/gfx1100_microbench/`.

See `GFX1100_ARCH.md` §5.5 for the load-bearing correctness ruleset.

---

### 5.0 Cache Controls (GLC/SLC/DLC) — What They Mean and What We Measured
RDNA3 exposes three relevant cache-control bits on vector memory ops (RDNA3 ISA §4.1.1):

- **`GLC`** – affects the graphics client first-level cache behavior and is also a scope bit (CU vs device scope for loads).
- **`SLC`** – a **temporal hint for graphics client caches** (`0` regular, `1` stream/non-temporal).
- **`DLC`** – a **temporal hint for Infinity Cache** (`0` regular, `1` non-temporal).

On this system (RX 7900 XTX / gfx1100), the microbench suite shows these can be meaningful levers, but **their effect depends strongly on access pattern and working-set size**.

#### 5.0.1 Measured Bandwidth: Raw Buffer Loads/Stores/Copies
All results below are from `test/gfx1100_microbench/` and are intended as A/B guidance (not “architecture constants”):

- **Streaming read bandwidth can improve with `glc=1` and/or `slc=1`** (64 MiB repeated reads, 16B loads):
  - `raw_load_hint_bench` aux=0: ~1237 GB/s
  - `raw_load_hint_bench` aux=1 (`glc=1`): ~1365 GB/s
  - `raw_load_hint_bench` aux=2 (`slc=1`): ~1369 GB/s
- **`dlc=1` (Infinity Cache non-temporal) strongly reduced bandwidth for reuse-friendly reads** in the same test:
  - `raw_load_hint_bench` aux=4 (`dlc=1`): ~697 GB/s
- **Aligned stores are far more sensitive to *alignment* than to `glc/slc/dlc`**:
  - `raw_store_hint_bench` (16B stores, offset=0): ~836 GB/s across aux settings (differences were small)
  - `raw_store_hint_bench` (16B stores, offset=1): ~423 GB/s (a ~2× regression from alignment alone)
- **Non-temporal store hint can partially mitigate misaligned stores/copies**:
  - `raw_store_hint_bench` (16B stores, offset=1): `dlc=0` ~425 GB/s vs `dlc=1` ~560 GB/s
  - `raw_copy_hint_bench` (16B load+store, src_off=1 dst_off=1): `dlc-store=0` ~555 GB/s vs `dlc-store=1` ~648 GB/s (read+write)
- **Copy (read+write aggregate) is modestly sensitive to `dlc`** (256 MiB copy, 16B loads/stores):
  - `raw_copy_hint_bench` `dlc-load=0 dlc-store=0`: ~751 GB/s (read+write)
  - `raw_copy_hint_bench` `dlc-load=1 dlc-store=1`: ~695 GB/s (read+write)

#### 5.0.2 Measured Latency: Pointer-Chase vs Working-Set Size
`cache_chase_bench` (dependent raw-buffer loads; random permutation) shows the expected “more bytes ⇒ more latency” behavior, and it makes the `dlc` effect easy to see once the working set exceeds L2:

- 32 KiB: ~29 ns/load (`dlc` had no visible effect)
- 2 MiB: ~74 ns/load (`dlc` had no visible effect)
- 8 MiB: ~107 ns/load with `dlc=0`, ~254 ns/load with `dlc=1`
- 32 MiB: ~170 ns/load with `dlc=0`, ~289 ns/load with `dlc=1`
- 256 MiB: ~240 ns/load with `dlc=0`, ~256 ns/load with `dlc=1`

Interpretation (practical, not microarchitectural): **when the working set is larger than L2 but still cacheable by Infinity Cache, `dlc=1` can force more traffic past Infinity Cache and increase latency substantially**.

#### 5.0.3 Measured “Hot vs Streaming” Pollution
`dlc_pollution_bench` (32 MiB hot set + 256 MiB stream) shows:

- `dlc=0`: streaming phase slowed a subsequent hot-set pass (pollution)
- `dlc=1`: hot-set performance stayed close to “hot_before” (less pollution), but the streaming phase itself ran slower

#### 5.0.4 Paged KV Gather (Block-Table Indirection) — Focus-Model Guidance
Attention decode/prefill with `use_cache=true` often reads KV in a “paged” pattern. `paged_kv_gather_bench` simulates KV reads through a page table and measures aggregate K+V read bandwidth.

Two important regimes show up clearly on RX 7900 XTX:

- **Focus A-like (head_dim=128, tokens≈40k, kv_heads=4)**: `dlc=0` was consistently best (~1100–1230 GB/s). `dlc=1` was consistently much worse (~420–525 GB/s).
- **Focus B-like (head_dim=64, tokens≈131k, kv_heads=8)**: in this *pure gather* microbench, `dlc=1` was often slightly better than `dlc=0` (~916 GB/s vs ~860 GB/s).

Important: this “pure gather” result does **not** necessarily carry over to a fused attention tile. In `attention_tile_bench` (paged K+V load + softmax-like compute + PV-like accumulate), `dlc=0` was consistently better than `dlc=1` for **both** Focus A-like and Focus B-like settings on this system.

**Other long-context LLM regimes we tested on real hardware**
These are the additional “focus models” we pulled shapes from `config.json` and measured directly on RX 7900 XTX (gfx1100) using the microbench suite in `test/gfx1100_microbench/`:

- **Kimi-K2-Thinking-like (head_dim=112, tokens=262144, kv_heads=64, gqa_group=1)**:
  - `paged_kv_batch_gather_bench` (K+V reads through a page table): `dlc=1` was modestly better than `dlc=0` when within-page order was contiguous (~901 GB/s vs ~858 GB/s), and slightly better even with within-page scrambling (~804 GB/s vs ~782 GB/s).
  - `paged_kv_scatter_gather_bench` (prefill→decode staged, moving tail): `aux_load=4` (`dlc=1`) improved the paired measurement (~887 GB/s vs ~844 GB/s).
  - `attention_tile_bench` (fused-ish): `dlc` was essentially a tie (~433–435 GB/s).
- **MiniMax-M2.1-like (head_dim=128, tokens=196608, kv_heads=8, gqa_group=6)**:
  - `paged_kv_batch_gather_bench`: `dlc=1` improved read bandwidth in this test (~699 GB/s vs ~625 GB/s).
  - `paged_kv_scatter_gather_bench` (moving tail): best paired measurement used `aux_load=0` (`dlc=0`) (~613 GB/s); `aux_load=4` was worse (~553 GB/s).
  - `attention_tile_bench`: `dlc=0` was better (~182 GB/s best-case) than `dlc=1` (down to ~113 GB/s in the same sweep).
- **GLM-4.7-like (head_dim=128, tokens=202752, kv_heads=8, gqa_group=12)**:
  - `paged_kv_batch_gather_bench`: `dlc=0` and `dlc=1` were both viable; best observed read bandwidth was with `dlc=0` (~935 GB/s).
  - `paged_kv_scatter_gather_bench` (moving tail): best paired measurement used `aux_load=0` (`dlc=0`) (~914 GB/s).
  - `attention_tile_bench`: `dlc=0` was better (~95 GB/s) than `dlc=1` (~82–89 GB/s).

**Paged QK dot (K-only) is a different `dlc` regime than KV gathers**
`paged_qk_dot_bench` uses raw buffer loads for K (paged indirection) and computes a QK dot-like inner loop. Across all three “new” regimes above, it strongly preferred `dlc=0` on loads:

- Kimi-like (gqa_group=1): `dlc=0` best (~760 GB/s K-read), `dlc=1` slightly worse (~730 GB/s).
- MiniMax-like (gqa_group=6): `dlc=1` was much worse (~131 GB/s) than `dlc=0` (~201 GB/s).
- GLM-like (gqa_group=12): `dlc=1` was catastrophic (~55 GB/s) vs `dlc=0` (~166 GB/s).

Practical takeaway: **treat `dlc` as per-kernel (and sometimes per-stage) tuning**. “Pure KV gather” and “QK dot” can want opposite `dlc` settings on gfx1100.

**Measured: “rotate page order per block” can matter**
- `paged_kv_gather_bench` has a `--rotate-step` option that applies a per-CTA rotation of the logical page index to change the concurrency pattern.
- For Focus B-like settings with `dlc=0`, `rotate_step=1` consistently improved bandwidth (~845–850 GB/s → ~920 GB/s in repeated runs). For `dlc=1`, `rotate_step` was typically closer to neutral in this microbench.

Interpretation: for very long-context KV reads, having “all CTAs walk the same page order” can be suboptimal; distributing page indices across CTAs can improve effective throughput (likely by reducing cache/TLB/memory-partition contention).

**Measured: batch + within-page order also matter (and can change which `dlc` wins)**
- `paged_kv_batch_gather_bench` adds a batch dimension (per-sequence page tables and per-batch KV regions) and an adversarial “scramble within page” option that breaks within-page contiguity.
- **Within-page contiguity matters**: scrambling token order within each page reduced aggregate read bandwidth:
  - Focus B-like (`head_dim=64`, batch=1): ~828 GB/s → ~715 GB/s (`dlc=0`) in one sweep. (For Focus A-like in this microbench, the effect was much smaller/noisier.)
- **Very large effective working sets can flip `dlc` behavior for pure gathers**:
  - Focus A-like: batch=1 strongly prefers `dlc=0` (~1375 GB/s vs ~410 GB/s), but batch=32 can prefer `dlc=1` (~899 GB/s vs ~852 GB/s) because the access becomes “more purely streaming”.
  - Focus B-like: `dlc=1` is often slightly better than `dlc=0` even at batch=1 (~912 GB/s vs ~848 GB/s), and at batch=32 we saw ~928 GB/s vs ~875 GB/s.

**Measured: KV writes are a different regime than KV reads**
- `paged_kv_scatter_bench` measures paged K+V writes (prefill-like). KV writes were lower bandwidth than KV reads, and the `dlc` behavior can be batch-dependent:
  - Focus A-like, batch=1: `dlc=0` was better (~750 GB/s) than `dlc=1` (~270 GB/s).
  - Focus A-like, batch=32: `dlc=1` can slightly beat `dlc=0` (~596 GB/s vs ~563 GB/s).
  - Focus B-like, batch=1: `dlc=0` and `dlc=1` were close (~508 GB/s vs ~503 GB/s).
  - Focus B-like, batch=32: `dlc=1` can be better (~598 GB/s vs ~554 GB/s).
- `paged_kv_scatter_gather_bench` measures a prefill→decode transition (write then read) and reports both baselines and a paired timing. It supports **windowed** ranges (e.g., write the last 1–32 tokens then read the full context) via `--scatter-start-token/--scatter-tokens` and `--gather-start-token/--gather-tokens`, and also supports a decode-like **moving tail** via `--scatter-advance-step` (advances the write window between iterations).
  - **Focus A-like windowed decode (tokens=40960, scatter last 32, gather full)**:
    - Fixed tail: `dlc=0` remained best for the read stage at batch=1 (paired ~865–976 GB/s for `dlc=0` vs ~697–700 GB/s for `dlc=1` in one sweep), but at batch=32 the paired measurement preferred `dlc=1` (~826 GB/s) over `dlc=0` (~786 GB/s).
    - Moving tail (`--scatter-advance-step 32`, `scramble_within_page=1`): same conclusion in a longer run (batch=1 paired ~1215 GB/s for `dlc=0` vs ~767 GB/s for `dlc=1`; batch=32 paired ~826 GB/s for `dlc=1` vs ~787 GB/s for `dlc=0`).
  - **Focus B-like windowed decode (tokens=131072, scatter last 32, gather full)**:
    - Fixed tail: `dlc=1` consistently improved the paired measurement in the more adversarial “scramble within page” setting (batch=1: ~760 GB/s vs ~700 GB/s; batch=32: ~815 GB/s vs ~743 GB/s).
    - Moving tail (`--scatter-advance-step 32`, `scramble_within_page=1`): same pattern (batch=1 paired ~795 GB/s for `dlc=1` vs ~713 GB/s for `dlc=0`; batch=32 paired ~802 GB/s for `dlc=1` vs ~737 GB/s for `dlc=0`).
  - Interpretation: **even a small “write tail” can change the best choice for subsequent reads**. If your decode step is staged (scatter then gather), A/B `dlc` on the *load* path with a paired benchmark; do not assume the fused attention tile will follow pure-gather behavior.

**Practical guidance (general across models):**
- Default to **`dlc=0`** when you expect reuse (weights, K/V reused across multiple query heads, activations reused across tiles).
- Consider **`dlc=1`** only for **true streaming** where the working set is well beyond Infinity Cache and the goal is to reduce cache pollution — and only if an on-hardware A/B shows it helps the *full* kernel (isolated gathers can be misleading).
- For streaming reads, consider **`slc=1`** (stream hint) and/or **`glc=1`** (scope/cache behavior) as knobs; validate per kernel because these can change caching and consistency behavior.

#### 5.0.5 Stride/Page Sensitivity (Proxy for TLB/Page-Walk Cost)
`stride_load_bench` performs strided 16B loads across a large buffer. It’s not a pure “TLB benchmark”, but it does show the practical effect of page-scale striding:

- With 512 MiB working set and 4 KiB stride: ~8.7 GB/s (`dlc=0`) to ~9.2 GB/s (`dlc=1`)

This is orders of magnitude below contiguous streaming loads and is a good reminder that **making accesses contiguous within a page (and avoiding page-stride patterns) matters** for long-context attention kernels.

#### 5.0.6 Attention-Shaped Microbenches (Focus A vs Focus B)
These benches are useful because they look more like attention than pure bandwidth tests:

- `softmax_like_bench` (BF16 logits, FP32 max/sum + exp): for both Focus A-like (`cols=40960`) and Focus B-like (`cols=131072`) settings, the warp-based reduction and LDS-tree reduction were **close** on this system (typically within a few percent); treat this as a kernel-level tuning choice (register pressure/occupancy can flip the winner).
- `qk_dot_gqa_bench` (load K once per token, reuse across GQA query-head warps):
  - Focus A-like (`head_dim=128`, `kv_heads=4`, `gqa_group=16`): `dlc=0` beat `dlc=1` (~49 GB/s vs ~42 GB/s K-read; ~0.79 vs ~0.67 TFLOP/s in this microbench).
  - Focus B-like (`head_dim=64`, `kv_heads=8`, `gqa_group=8`): `dlc=1` slightly beat `dlc=0` for the “same page order everywhere” setup, but changing the concurrency pattern changed which setting won (e.g., in one run, `rotate_step=1` improved `dlc=0` from ~60 GB/s to ~71 GB/s while making `dlc=1` worse).
- `attention_tile_bench` (paged K+V load + softmax + PV-like scalar accumulate):
  - Focus A-like: `dlc=0` beat `dlc=1` (~62–64 GB/s vs ~56–57 GB/s K+V read). `glc=1` sometimes provided a small additional win on top of `dlc=0` (~+1–2% in one sweep).
  - Focus B-like: `dlc=0` beat `dlc=1` (~90–92 GB/s vs ~81–82 GB/s K+V read); `glc=1`/`slc=1` were small wins in some runs, and `rotate_step=1` was roughly neutral in this microbench.
  - Batch stress: with `batch=32` and per-batch page tables, `dlc=0` still did not lose (Focus A-like: `dlc=0` remained slightly better; Focus B-like: `dlc=0` and `dlc=1` were essentially tied). This is another example where *pure gather* behavior does not reliably predict fused attention-tile behavior.

This supports a practical policy: **default to `dlc=0` for real attention tiles on this system**, and for Focus B-like long-context cases prioritize **concurrency/layout tuning** (page-table order, CTA staggering/rotation) before reaching for `dlc=1`.
