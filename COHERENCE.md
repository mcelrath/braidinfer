# RDNA3 PCIe + Cache Coherence Catalog                                                   
                                                        
Hardware: 8× AMD Radeon RX 7900 XTX (gfx1100), AMD host, ROCm 7.x, mixed bifurcation
(some cards on direct PCIe gen4 x16, some on x8 via MCIO cables).

Source: exterior_algebra-vhs PCIe systematic eval + prior P2P phase + 2026-05-02 debug.

## TL;DR

- **PCIe atomic-op support is fully wired end-to-end on all 8 GPUs.** Not a coherence bottleneck.
- **CPU↔GPU host-mapped signaling works** at ~5 µs round-trip (single-pair, fresh state)
  but is the ONLY consistently-working cross-GPU coordination primitive on this setup.
- **Cross-GPU `atomicAdd` HANGS even on adjacent GPU0↔1.** Despite PCIe atomics being
  enabled in hardware, the runtime/kernel implementation does not complete reliably.
- **Cross-GPU `uncached_vram` polling is topology-dependent.** Works GPU0↔1, hangs GPU0↔2.
- **Aggressive cross-GPU coherence tests can wedge the kernel TTM subsystem** without any
  dmesg signal. Only reboot recovers. Document and avoid; do not run unattended.
- **`iommu=pt` is a host-crash multiplier**: with it active, bad GPU DMA from misbehaving
  cooperative kernels can corrupt host RAM and cause CPU MCEs. Run with `iommu=pt` REMOVED;
  bad DMA then surfaces as IO_PAGE_FAULT in dmesg instead.

## Measured cross-GPU latency (partial matrix, 2026-05-02 post-reboot)

| Pair | Primitive | Median µs | Status |
|---|---|---|---|
| GPU0↔1 | host_mapped (single-pair, fresh init) | 5.06 | OK (session start) |
| GPU0↔1 | host_mapped (-g 4 launcher, peer mesh setup) | 49.66 | OK |
| GPU0↔1 | host_mapped (direct HIP_VISIBLE_DEVICES, 2nd run) | 211.79 | OK |
| GPU0↔1 | host_mapped (direct, 3rd run) | 412.25 | OK |
| GPU0↔1 | uncached_vram | 94.14 / 415.28 | OK (high variance) |
| GPU0↔1 | atomic_add | — | HANG |
| GPU0↔2 | host_mapped | 124.33 | OK |
| GPU0↔2 | uncached_vram | — | HANG |
| GPU0↔2..7 | atomic_add (untested past 0↔1) | — | (not tested; cumulative wedge risk) |
| Other 18 pairs | (not tested) | — | (cumulative wedge risk + 3 sibling sessions) |

**Variance pattern**: same pair, same primitive, 4-8× variance across runs. Likely caused
by sibling Claude session contention (3 active sessions). For real per-topology numbers
the tests would need to run on a quiesced system (no other GPU users beyond llama-server
on GPU 3).

## What works (use these)

| Primitive | Cost | Pattern |
|---|---|---|
| `volatile` reads | n/a | Standard CPU semantics; required for cross-GPU spin-poll |
| `hipDeviceMallocUncached` for cross-GPU signaling | 5.47 µs | MTYPE_UC, peer writes immediately visible |
| `hipHostMallocMapped` (no `Portable` flag) | 4.77 µs | CPU↔GPU round-trip via host-mapped device pointer |
| `__threadfence()` (intra-kernel only) | ns | Memory ordering within a single kernel invocation |
| Cooperative `grid.sync()` | ~1 µs at 96 blocks | Single-GPU intra-kernel synchronization |
| Hierarchical barrier across CUs (single GPU) | 114 µs flat / 244 µs hier | Single GPU only |
| `atomicAdd`/`atomicCAS`/`atomicExch` on uncached peer VRAM | varies by topology | Confirmed working in prior P2P phase |
| `buffer_gl1_inv` (`asm volatile("buffer_gl1_inv" ::: "memory");`) | ns | Reader-side per-WGP L1 invalidation |
| Per-pair `hipDeviceEnablePeerAccess` (NOT N×N mesh) | one-shot | Setup time scales with topology |
| `cpu_to_gpu_mailbox` (round-trip / 2 approximation) | 0.79–0.93 µs typical, max 26 µs | Pinned-host page; CPU writes flag, GPU polls; round-trip /2 = upper bound on one-way. Most reliable cross-process coordination primitive on this hardware. |
| `gpu_to_gpu_peer_write` (rt/2) | 2.16–2.71 µs typical | GPU A writes peer VRAM word, GPU B observes via host-mapped ack pong. Works on 26/28 pairs. |
| Cooperative-exit watchdog protocol (kernel polls `force_exit`, host writes it) | 4.7 ± 0.7 ms recovery time, 100/100 PASS in unit test | Only viable cooperative recovery path on RDNA3 (no GPU TDR for compute). See `braidinfer/kernels/watchdog.h`. |
| `hipMemcpyPeerAsync` 64 B–1 MB payloads | 12–325 µs p50 (size + topology dependent) | Reliable median; p99 spikes 4–50 ms on specific cross-root pairs (filter or avoid those pairs in production) |
| `segmented_graph_launch` (HIP graph cross-pair) | 10–22 µs typical | Outliers 33–267 µs on certain pairs involving topo GPU 0 |
| **SPSC ring queue per directed pair** (recipe below) | **130 Mmsg/s same-root, 25–72 Mmsg/s cross-root @ 16 B** | High-throughput cross-GPU primitive. Use for streaming workloads. **Recipe**: `hipHostMallocCoherent + Portable` flags, ring buffer in pinned host with head/tail in separate cachelines, `__threadfence_system()` on producer after writing payload + bumping head, `buffer_gl1_inv` on consumer before re-reading head. Single-producer single-consumer per directed pair (56 queues for 8 GPUs). Zero errors validated all pairs. |
| `__threadfence_system()` against **pinned host memory** | nanoseconds | Works for ordering writes that target host memory. Used in 2PC and SPSC patterns. **Distinct from peer VRAM** (which hangs — see "BROKEN"). |

PCIe topology data (definitive, all 8 GPUs):
- GPU function: `AtomicOpsCap: 32bit+ 64bit+ 128bitCAS-`, `AtomicOpsCtl: ReqEn+`, x16 16GT/s
- Switches (between root port and GPU): `AtomicOpsCap: Routing+`, `AtomicOpsCtl: EgressBlck-`, x16 16GT/s
- **Root ports (c0:01.1, 80:01.1, etc.): `AtomicOpsCap: Routing-`** (the silicon-level reason cross-GPU atomics hang — see new section below)

→ 32-bit and 64-bit PCIe AtomicOps requested by every GPU; switches route them; **root ports refuse to route them between sub-trees or across IODs** — silicon limitation on EPYC 7532 / Rome IODs.

### Mandatory env vars for cross-GPU coordination through host memory

| Env var | Why |
|---------|-----|
| `HIP_HOST_COHERENT=1` | **REQUIRED** for GPU→GPU observation via host memory on gfx1100. Without it, the GPU's L2 caches host-memory reads and never invalidates (no `buffer_gl2_inv` on RDNA3). Symptom: GPU A writes pinned host page, GPU B's reader is stuck reading old value forever. The Lamport-clock and SPSC-ring patterns set this. |
| `HSA_ENABLE_SDMA=0` | Optional — disables SDMA path entirely. SDMA `hipMemcpyPeer` for small transfers is broken on RDNA3 in some configurations (see BROKEN table); disabling it forces the compute-engine peer-copy path which is fine for ≤64 KB. |

## What is BROKEN — do not use

| Primitive | Failure mode | Source |
|---|---|---|
| `__threadfence_system()` against **peer VRAM** | Hangs gfx1100 indefinitely (no routing path for the system-scope ordering operation through Routing- root ports) | P2P phase commit `663731c` + 2026-05-03 root cause |
| `__threadfence_system()` paired with **pinned-host-memory** flag store | **Works** (used in 2PC, SPSC, llama.cpp `ggml-cuda/moe-ep.cu` `ep_sync_signal_kernel`). Distinct from peer VRAM. | 2026-05-03 patterns-bench |
| `__hip_atomic_load(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)` | Hangs | P2P phase commit `663731c` |
| `s_waitcnt vmcnt(0)` per-grid-sync inside megakernel | Hard hang (D state on `drm_sched_entity_flush`) | bz0 investigation |
| Naive cross-GPU spin-poll on cached peer VRAM | L2 on polling GPU never invalidates | P2P phase commits `4f56691`, `574b729` |
| GPU-initiated kernel dispatch via doorbell | Kernel patch required; not viable | P2P phase |
| SDMA `hipMemcpyPeer` for small transfers | Broken on RDNA3 | P2P phase |
| `buffer_wbl2` | CDNA-only opcode; not present on RDNA3 | LLVM AMDGPU defs |
| `buffer_gl2_inv` / `buffer_invl2` | Rejected by `llvm-mc --mcpu=gfx1100` | composable_kernel doc §5.3 |
| Any L2 invalidation on gfx1100 | Doesn't exist in the ISA at all | composable_kernel doc §5.3 |
| Full N×N `hipDeviceEnablePeerAccess` mesh | Stalls on cross-root-complex bridges | 2026-05-02 |
| Cross-GPU `atomicAdd` even on adjacent GPU0↔1 | Process never returns; outer timeout fires | 2026-05-02 matrix run |
| Cross-GPU writer-kernel launch in `p2p_coherence_force` test 4a | Kernel<<<>>> never returns to host | 2026-05-02 lower mode |
| `p2p_coherence_force` test 4a + tests 4c/4d/4e | Various hangs / can wedge TTM | 2026-05-02 |
| `hipDeviceReset` on a GPU running a stuck compute kernel | Blocks indefinitely (15+ min observed); RDNA3 has no GPU TDR for compute | 2026-05-03 watchdog Phase 4 |
| ~~Single-writer + cross-GPU peer-VRAM polling (any pair)~~ | ~~TIMEOUT on all 28 pairs~~ | RETRACTED — was a peer-mapping bug in the test, not a hardware limit. See "2026-05-03 corrected: uncached-pool peer mapping" below. |

Pattern: any primitive that **waits on global completion ordering** has cliff-edge failure
modes. Primitives that **just emit memory operations** work. **Cross-GPU operations beyond
host_mapped polling are unreliable on this hardware/runtime combination.** PCIe atomics
support is wired in hardware (per lspci) but the runtime/driver path does not deliver
working cross-GPU atomic_add.

**`hipDeviceReset` corollary**: there is no soft single-GPU recovery on consumer RDNA3.
For a wedged compute kernel, only host-process death (SIGKILL/`std::process::abort()`) frees
the GPU, by triggering amdgpu-driver-level context cleanup. Watchdog escalation paths must
go directly from cooperative grace-expiry to telemetry-dump-then-abort; do not attempt
`hipDeviceReset` as an intermediate step. (See watchdog plan / kernels/watchdog.h primitive.)

## RDNA3 cache hierarchy — what's invalidated when

| Level | Scope | Invalidated by kernel boundary? |
|---|---|---|
| L0 (vector cache) | per-SIMD, 16 KB | Yes, always |
| L1 (CU cache) | per-WGP, 128 KB | **Depends on AQL packet scope bits** |
| L2 (GL2) | device-wide, 6 MB | On `hipDeviceSynchronize`, not always between same-stream cooperative kernels |

`hipExtModuleLaunchKernel` sets AGENT-scope release/acquire fences. **AGENT scope flushes
L2 but NOT L1.** Cross-cooperative-launch reads of multi-WGP-written HBM can return stale
L1 data if the consumer WGP's L1 cached a prior value. This is the textbook cause of
"step 0 deterministic, step N+1 divergent" patterns.

**Reader-side fix** (cheap, always safe):
  if (threadIdx.x == 0) {
      asm volatile("buffer_gl1_inv" ::: "memory");
  }
  __syncthreads();

**Writer-side fix** (more invasive, RDNA3-specific):
- `buffer_wbl2` exists on CDNA, NOT on RDNA3.
- Equivalent for RDNA3: `__builtin_amdgcn_buffer_wbinvl1_vol()` per-block at writer kernel exit.

**Builtins**:
- `__builtin_amdgcn_buffer_wbinvl1_vol()` — older naming
- `__builtin_amdgcn_s_dcache_inv_vol()` — alternative

**Caveat (from bz0)**: L1 invalidation alone may not solve all cross-launch determinism
issues. Suspect remaining causes: L2-level staleness, async-copy/stream-ordering bugs,
HBM-level race on shared buffers, DMA staleness on host-mapped writes.

## NEW failure mode discovered 2026-05-02 — TTM wedge

**Symptom**: `hipGetDeviceCount` hangs indefinitely on every new HIP-using process.
`rocm-smi` works (talks via sysfs). `dmesg` is clean.

**Diagnosis**:
  /proc/modules:  amdgpu 16883712 -1 - Unloading
  ps -e -o pid,stat,wchan,cmd:
    kworker/u285:0+ttm    D<   01:37:49
    kworker/u285:1+ttm    D<   01:25:57
    rmmod amdgpu          D+   02:09

TTM kworkers stuck in D state for over an hour, blocking `rmmod amdgpu`. They wait on GPU
operations from earlier hung tests that got SIGKILL'd. Each hung-test-then-SIGKILL
accumulates one stuck TTM cleanup; after enough accumulate, HIP runtime can no longer
enumerate devices.

**Recovery**: REBOOT only. D-state procs cannot be killed even with `kill -9`.
`rocm-smi --gpureset` does NOT clear TTM state. `modprobe -r amdgpu` waits forever.

**Mitigation**:
1. Never run cross-GPU coherence tests unattended — each hung test damages the system.
2. Test runners MUST hard-abort on first timeout (rc=124).
3. After any hung test, snapshot `cat /proc/modules | grep amdgpu` (refcount stable?)
 and `ps -e -o stat,wchan,cmd | grep ttm`. D-state procs accumulating → reboot.
4. Always run with `iommu=pt` REMOVED.

## bz0 single-GPU compute non-determinism (RESOLVED 2026-05-03)

**Distinct from cross-GPU bz0** (resolved earlier by hsa-rocr-p2p-mtype-uc-gfx11.patch).
This is a *single-GPU, in-process* non-determinism where the same kernel with the same
inputs produces different output across consecutive runs in the same process.

### Symptoms

`bench_coherence` on qwen35_2b paged decode: ~50% of consecutive seq-vs-seq pairs
disagreed. `top-10 match: 1/10`, `n_diff_logits=248320/248320`, `max_abs_diff=4.22`.
Not flakey hardware — fully deterministic in the worst-case manifestation: ALL logits
diverge whenever it triggers.

### Localization

Per-instruction dump comparison (`BZ0_DUMP=1` in `braid_bench`) showed the first
divergent op was `op_attn_paged` at slot 427, inst_idx=65. All inputs to that op
were bit-exact across runs. 5 of 8 q_heads diverged.

A/B test split the two `block_reduce_sum` call sites inside `op_attn_paged`:
- K-rms (`sum(my_k * my_k)`, line 1707): replacing with sequential reduce → bit-exact.
- Q·K dot product (`sum(q[i]*k[i])`, line 1752): keeping shfl → still bit-exact.

Conclusion: K-rms reduction was the *sole* source. Q·K dot reduction with the same
`block_reduce_sum` helper was deterministic.

### Root cause: `ds_bpermute_b32` with same-VGPR dst/src

Compile with `-save-temps`, inspect the `.s`. Across the entire 27,882-line
megakernel asm, only **2 of 221** `ds_bpermute_b32` instructions have the destination
register equal to the data-source register:

```
asm 13215 (op_attn_paged):       ds_bpermute_b32 v81, v55, v81
asm 15523 (op_attn_paged_quant): ds_bpermute_b32 v101, v87, v101
```

These are exactly the two K-rms sites. Every other `ds_bpermute_b32` (including Q·K dot
in the same kernel function) has distinct dst/src.

The same-VGPR pattern arises because:
1. Input is a self-multiply `my_k * my_k`.
2. `-ffp-contract=fast` lets LLVM fuse the algebraic expression `bpermute(x²) + x²`
   into a single FMA: `v_fmac_f32_e32 v81, v80, v80`.
3. With FMA recomputing `x²` from the original operand `v80`, the producer `v_mul_f32`
   that wrote `x²` to `v81` is no longer needed live past the bpermute.
4. The register allocator therefore picks `v81` as both the bpermute destination AND the
   data source, saving one VGPR.
5. **Empirically, `ds_bpermute_b32 vX, vY, vX` produces non-deterministic output on
   gfx1100 wave64 mode.** Mechanism unconfirmed; consistent with a missed wait-state /
   scoreboarding interaction between VALU writeback and the LDS-fabric cross-lane
   path under wave-on-WGP contention.

The buggy producer + bpermute sequence:

```asm
v_mul_f32_e32 v81, v80, v80       ; v81 = my_k * my_k  (warp_reduce input)
ds_bpermute_b32 v81, v55, v81     ; v81 = bpermute(v81)  ← SAME VGPR as src
s_waitcnt lgkmcnt(0)
v_fmac_f32_e32 v81, v80, v80      ; FMA: v81 += my_k*my_k (recomputed)
```

The non-buggy producer + bpermute sequence (Q·K dot, same kernel):

```asm
v_mul_f32_e32 v5, v5, v34         ; v5 = q * sh_k  (NOT a self-multiply)
s_or_b64 exec, exec, s[22:23]
ds_bpermute_b32 v34, v55, v5      ; v34 = bpermute(v5)  ← DISTINCT VGPRs
s_waitcnt lgkmcnt(0)
v_add_f32_e32 v5, v5, v34         ; standard add
```

### Workaround (production fix)

Replace `__shfl_down`-based reductions with a shared-memory tree reduction at any
sum-of-squares site that feeds an online softmax. New helper in
`kernels/megakernel_ops.hip`:

```c
__device__ __forceinline__ float tree_reduce_sum_256(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    return shared[0];
}
```

Applied at `op_attn_paged` K-rms and `op_attn_paged_quant` K-rms (commit a5c2f66
+ uncommitted fix). Verification: `seq->seq same-process determinism: 10/10 top,
bit-exact=true, n_diff_logits=0/248320`. Decode 26 tok/s (no regression).

### Detection rule (for future kernels)

After compiling, grep emitted asm for self-overlapping bpermute:

```bash
hipcc --offload-arch=gfx1100 --genco -O3 -ffp-contract=fast -mwavefrontsize64 \
    -save-temps -o out.hsaco kernel.hip
python3 -c '
import re, sys
for line in open(sys.argv[1]):
    m = re.search(r"ds_bpermute_b32\s+v(\d+),\s*v\d+,\s*v(\d+)", line)
    if m and m.group(1) == m.group(2):
        print("HAZARD:", line.strip())
' *.s
```

Zero hits = safe. Any hit = at risk; use shared-memory tree reduction at that site
(or rewrite to break the self-multiply pattern).

### Triggers / risk surface

Combination required:
- gfx1100 (RDNA3) — wave64 confirmed; wave32 untested but suspected similar
- `-ffp-contract=fast` — without this, FMA fusion doesn't happen and register
  allocator keeps producer + bpermute on distinct VGPRs
- `__shfl_down` over a sum-of-squares (or any self-multiply) reduction
- Online-softmax-style consumer (`m_new = max(m, score)`) that amplifies bit-level
  noise into observable logit divergence

Without the softmax-max amplifier, latent reductions in FFN/RMSNorm/GDN/etc. might
still produce slightly non-deterministic outputs, but they would be dampened by
downstream smooth functions and not observable in current tests. **The other 30+
`block_reduce_sum` sites in megakernel_ops.hip / megakernel_moe.hip are not
currently observable as buggy, but should be considered at-risk for any new kernel
that feeds a max-comparison.**

### Recommendations

- For RDNA3 single-block reductions of `sum(x²)` (or any self-multiply input)
  followed by softmax/max: use shared-memory tree reduction, not warp-shfl.
- For other reductions (FFN gate/up, RMSNorm, GEMV accumulation): keep shfl for
  performance, but run a per-instruction dump comparison test (BZ0_DUMP-style) on
  the production model before shipping any new reduction.
- When adding any new reduction, grep the emitted asm for the same-VGPR bpermute
  pattern (above). Treat any hit as a defect.
- Do not patch the local LLVM toolchain. File the hazard upstream (LLVM AMDGPU
  backend) with a minimal reproducer. Keep the application-level tree-reduction
  workaround as our shipped fix.

## Test infrastructure ground rules

(Apply to ANY future PCIe coherence harness.)

1. `iommu=pt` MUST be off during stress testing.
2. `HIP_LAUNCH_BLOCKING=1` and `AMD_LOG_LEVEL=2` add ~10× sync overhead. Lethal for orchestrators.
3. Per-test isolation, not orchestrator batching. Each test in its own process, with `sync; sleep 30` between.
4. Pre-flight check: enumerate `/dev/kfd` and `/dev/dri/render*` openers. Any unexpected → abort.
5. Pair-only `hipDeviceEnablePeerAccess`, never the full N×N mesh.
6. Hard-abort on rc=124 (timeout). Each timeout potentially leaves a stuck TTM kworker.
7. Snapshot `dmesg` and `ras-mc-ctl --summary` before/after each test.
8. Single-GPU first for every Risky-class test.
9. Tear down persistent megakernel workers before any p2p test.
10. Risky-mode launches at `-g 2`, not `-g 4` (PSU stress).

## Recommendations for braidinfer multi-GPU code

- For cross-GPU coordination: prefer `hipHostMallocMapped` (CPU as coherence point, ~5 µs)
over peer VRAM with custom synchronization.
- For intra-GPU multi-WGP HBM-shared state: insert reader-side `buffer_gl1_inv` at every
kernel entry that reads HBM written by a prior kernel.
- Avoid: `__threadfence_system`, `__hip_atomic_load(SYSTEM)`, `s_waitcnt vmcnt(0)` inside
megakernels, doorbell-based GPU-initiated dispatch.
- Persistent cooperative megakernel: do not have it resident while running unrelated p2p tests.
- For multi-GPU MoE dispatch: match the weight-format dispatch to the actual format
(PcG32Q4 vs RNF4G128) — see commit `071a9ff`.

## Topology map (definitive)

4 root complexes on this host:
- 00c0: GPUs 0, 1, 2 (cross-switch within same root)
- 0080: GPUs 3, 4 (same root)
- 0040: GPUs 5, 6 (same root)
- 0000: GPU 7 (alone)

Pair classes: 5 same_root_cross_switch + 16 cross_root_complex (skipping GPU 3 = llama-server gives 21 testable pairs).

**Key finding**: topology class is NOT a complete predictor of cross-GPU primitive availability. `uncached_vram` works on GPU0↔1 (same_root) but HANGS on GPU0↔2 (same_root). Per-pair PCIe routing path matters beyond root-complex grouping.

## Empirical test of the CK spin-wait pattern (2026-05-02)

I implemented the composable_kernel §5.3 spin-wait pattern (`p2p_ck_spinwait.hip`) and tested
on 5 pairs spanning topology classes. **All 5 pairs FAILED.**

Two failure modes observed:
1. **READER_HANG + 224/256 data errors** (4 pairs, both same-root and cross-root): caused
   by a sync bug in the documented pattern itself — `__syncthreads()` after the block 0
   thread 0 spin only syncs within block 0. Blocks 1-7 proceed to read data without waiting
   for signal, racing the writer. 224 = 7 blocks × 32 threads. The CK doc's reader example
   has this same bug; their pattern is not actually a true cross-GPU spin-wait, only a
   stream-sync'd dispatch primitive.
2. **SPIN_TIMEOUT** (1 pair, cross_root_complex 0↔7): writer kernel finished, reader spun
   10M iterations without ever seeing signal=1. **Cross-root-complex peer write of a single
   int doesn't propagate to the polling GPU on this hardware/topology.**

**Implications**:
- The CK pattern is not a drop-in solution for cross-GPU signaling.
- Even with sync bug fixed (single-block reader), cross-root-complex pairs may not see
  peer writes at all without explicit cache-bypass tricks beyond MTYPE_UC.
- llama.cpp's pattern (fine-grained staging on writer GPU + system-scope stores from
  secondary + flags in pinned host memory) appears more robust because it routes flags
  through host memory rather than relying on cross-GPU VRAM coherence.

## Cross-references to composable_kernel's GFX1100_ARCH.md (highly relevant prior art)

`/home/mcelrath/Projects/ai/composable_kernel/GFX1100_ARCH.md` §5.3 has substantial
prior gfx1100 cross-GPU coherence work. Key findings that update our catalog:

- **gfx1100 has NO L2 invalidation instruction at all.** `buffer_gl2_inv` and
  `buffer_invl2` are REJECTED by `llvm-mc --mcpu=gfx1100`. Only `buffer_gl0_inv` and
  `buffer_gl1_inv` exist. (Our prior note that `buffer_wbl2` is CDNA-only is correct
  but understates this — gfx1100 has nothing equivalent.)
- **Cache invalidation cost measured**: `S_DCACHE_INV` adds ~0.11ms over baseline,
  `S_GL1_INV` adds ~0.24ms. So `buffer_gl1_inv` per-kernel-entry is NOT free —
  treat as a tuning tradeoff, not a default.
- **MTYPE distinction**: `hipDeviceMallocFinegrained` (MTYPE_CC) FAILS for
  spin-wait (GPU A's writes stay in GPU A's L2; without system flush they never
  reach GPU B's VRAM). `hipDeviceMallocUncached` (MTYPE_UC) bypasses all caches and
  works at ~22µs / 55K iters round-trip.
- **Working cross-GPU spin-wait pattern** (their §5.3, no `__threadfence_system` needed):
  ```cpp
  hipExtMallocWithFlags(&signal,  sizeof(int), hipDeviceMallocUncached);
  hipExtMallocWithFlags(&blk_ctr, sizeof(unsigned int), hipDeviceMallocUncached);
  hipExtMallocWithFlags(&data,    N*sizeof(float), hipDeviceMallocUncached);

  __global__ void writer(volatile float* data, volatile int* signal,
                         volatile unsigned int* blk_ctr, float val, int n) {
      // ... write data ...
      __threadfence();   // agent scope: drain this block's stores to VRAM
      if (threadIdx.x == 0) {
          unsigned int old = atomicAdd((unsigned int*)blk_ctr, 1u);
          if (old == gridDim.x - 1) {  // last block: all data now in VRAM
              *signal = 1; __threadfence();
          }
      }
  }
  ```
  Note: `atomicAdd` here is INTRA-GPU on a block counter (not cross-GPU peer). That works.
  Cross-GPU atomicAdd remains broken.
- **Hard practical**: uncapped spin can trigger TDR and reboot the system.
  Always cap spin loops with iteration limit.
- **RDNA4 (gfx1200) adds `global_wb scope:SCOPE_SYS` and `global_inv scope:SCOPE_SYS`**
  — full L2 writeback + invalidation. `__threadfence_system()` is expected to work on
  gfx1200. Cross-GPU coherence problem solves itself with hardware refresh.

## Findings on intra-GPU L1 invalidation (gfx1100)

Standalone test (`p2p_gl1_inv_test.hip`): two cooperative kernel launches on the same stream, each kernel writes its blockIdx to `counter[blockIdx]`, grid.sync, then reads `counter[(blockIdx+1) % N]`. Compares with vs without `asm volatile("buffer_gl1_inv" ::: "memory")` at start of kernel B.

5 runs each variant on GPU 0:
- variant 0 (no inv): 0 stale errors / 96 blocks
- variant 1 (with gl1_inv): 0 stale errors / 96 blocks

**Conclusion**: HIP runtime DOES invalidate L1 between cooperative kernel launches in this simple cross-kernel pattern. `buffer_gl1_inv` is functionally a no-op here. The bz0 staleness pattern (step 0 deterministic, step 1+ divergent) must require more specific conditions:
- The persistent megakernel context (worker stays resident across launches)
- Vector loads vs scalar
- Specific HBM regions written by multiple WGPs without grid.sync between writes
- Or the bug is not actually L1 invalidation at all — could be HBM-level race, async-copy ordering, etc.

This narrows the bz0 hypothesis space: simple per-WGP L1 invalidation alone won't fix it.

## ROCm runtime: we have a local P2P patch that upstream rejected/reverted

`/home/mcelrath/Projects/ai/arch-linux/hsa-rocr/hsa-rocr-p2p-mtype-uc-gfx11.patch`
patches `amd_memory_region.cpp:MapMemoryToNodes` to set `HSA_CACHING_NONCACHED` for
gfx11+ P2P mappings (when `gpu_agent->isMES() && whitelist_nodes.size() > 1`). This
makes peer-mapped VRAM bypass the writer's L2, working around the missing
`buffer_wbl2` instruction.

**Upstream history**: ROCR-Runtime commit `ed0a1be` (Jan 2023) implemented similar
behavior at the allocation path (`AllocatePCIeRW` flag) but was reverted on Feb 23, 2023
(`37b5b42`) with no reason given. Our patch's mapping-path approach is different and may
avoid the reverted regression. The patch should be submitted upstream targeting
`ROCm/rocm-systems` against `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_memory_region.cpp`.

**Kernel driver is NOT the gate**: `drivers/gpu/drm/amd/amdkfd/kfd_topology.c:kfd_set_iolink_no_atomics`
correctly clears `NO_ATOMICS_32/64_BIT` for our hardware (`p2p_links` flags = 3 =
`ENABLED | NON_COHERENT`). PCIe atomics are reported as supported.

**The actual hang root cause** (per LLVM AMDGPU backend analysis): SYSTEM-scope atomics
emit `buffer_gl1_inv`. RDNA3's PCIe implementation does NOT support the GL1 cache
coherency protocol that CDNA does. The instruction waits for a coherency ACK that never
comes → hang. **No upstream LLVM/ROCm fix exists.** gfx1250 (next arch) got memory
model improvements in Jan 2026; gfx1100 received none.

## Untested experiment that might unlock cross-GPU atomics

**Hypothesis**: SYSTEM-scope `atomicAdd_system` may succeed against memory that's
P2P-mapped with `HSA_CACHING_NONCACHED` (via our local patch), because `buffer_gl1_inv`
on already-uncached memory is a no-op (no coherency wait).

Our prior `[[clang::atomic]]` test used `hipDeviceMallocUncached` which sets the LOCAL
allocation uncached. That's different from `HSA_CACHING_NONCACHED` on the P2P MAPPING.
The patch ensures peer-mapped VRAM is noncached on the writer's side.

**Next experiment**: Allocate counter on dst with default `hipMalloc` (so writer's local
view is normal), let our local patch make it noncached when peer-mapped on src, then run
`atomicAdd_system` from src kernel. Compare against `hipDeviceMallocUncached` allocation.

## CORRECTED diagnosis: cross-GPU `atomicAdd` is silicon/firmware, not LLVM (2026-05-03)

Earlier hypothesis "LLVM emits `buffer_gl1_inv` for SYSTEM-scope atomics, which hangs on
RDNA3 PCIe" turned out to be **wrong for `atomicAdd`**. Assembly inspection of all 4
variants in `p2p_atomic_scope_test.hip` shows IDENTICAL codegen:
- variant 0 `atomicAdd` (AGENT scope): `global_atomic_add_u32` × N
- variants 1/2/3 SYSTEM scope (`atomicAdd_system`, `__hip_atomic_fetch_add(SYSTEM)`,
  `[[clang::atomic]]`-annotated): `s_waitcnt lgkmcnt(0); global_atomic_add_u32` × N

**No `buffer_gl1_inv` is emitted by any variant.** The `insertAcquire` codepath in
`SIMemoryLegalizer.cpp` only fires for ACQUIRE-ordered loads, not RELAXED atomicAdd.

**Where the hang actually lives**: the kernel issues `global_atomic_add_u32` to peer
VRAM. The kernel returns (no implicit `vmcnt(0)` wait at exit). Host calls
`hipStreamQuery` which the runtime resolves by waiting for outstanding vector memory
ops to complete. **The atomic ops never return their PCIe completion ACK** because
consumer RDNA3 doesn't reliably drain peer atomic completions over PCIe. Stream stays
busy forever.

This is **silicon/firmware-level**: AMD did not implement PCIe atomic completion routing
on consumer RDNA3 (only on Instinct/CDNA via Infinity Fabric). PCIe AtomicOpsCap bits
in the config space are advertised; the completer just never responds. **No software
patch can fix this** — not LLVM, not HIP runtime, not amdkfd.

The previously-suggested LLVM patch (skip `buffer_gl1_inv` for SYSTEM-scope acquire on
gfx11) MAY help `__hip_atomic_load(SYSTEM, ACQUIRE)` cross-GPU loads — a different code
path. It does NOT fix `atomicAdd`. Not worth the rebuild cost unless we have specific
code that uses acquire-scope loads for cross-GPU coordination.

## What atomic operations CAN we use, and how to simulate the rest

**Works on this hardware:**
- All atomics on **local VRAM** (intra-GPU): `atomicAdd`, `atomicCAS`, `atomicExch`,
  `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor` — all fine. Default
  AGENT scope.
- All atomics on **uncached local VRAM** (`hipDeviceMallocUncached`): same as above,
  works fine. Use when you need cross-WGP visibility WITHIN one GPU.
- `atomicAdd` on **host-mapped memory** (`hipHostMallocMapped`): works. CPU and GPU
  see each other's atomic updates. Useful for CPU↔GPU coordination.

**Does NOT work on this hardware (consumer RDNA3, no fix coming):**
- Any atomic op with a **peer-VRAM target**, regardless of scope or ordering. Issues
  the op but never gets a completion ACK, hangs the stream.
- All `*_system` variants when target is peer VRAM.

### Cross-GPU patterns that DO work

**Pattern A: Per-GPU slot + intra-GPU atomic + sum-on-receiver**
For "fan-in" use cases (e.g., MoE expert outputs from N GPUs combined on GPU 0):
```c
// Receiver allocates one slot per source GPU, uncached:
hipExtMallocWithFlags(&slots, N_SRC * sizeof(float), hipDeviceMallocUncached);

// Each source GPU's kernel writes its contribution to its OWN slot
// (peer write to receiver's uncached VRAM — works after our hsa-rocr+kernel patches):
__global__ void source(float* my_slot, float val) { *my_slot = val; }

// Receiver does intra-GPU summation (no atomic needed if each src writes to its own slot):
__global__ void combine(float* slots, int n_src, float* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < n_src; i++) sum += slots[i];
        *result = sum;
    }
}
```
Replaces cross-GPU atomicAdd with peer-write + intra-GPU sum. No atomic needed.

**Pattern B: Block-counter + signal (composable_kernel §5.3)**
For "all blocks done writing data, then signal reader":
```c
// All buffers on receiver as uncached
__global__ void writer(volatile float* data, volatile int* signal,
                       volatile unsigned int* blk_ctr, ...) {
    // ... write data ...
    __threadfence();   // agent-scope: drain this block's stores to VRAM
    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd((unsigned int*)blk_ctr, 1u);  // INTRA-GPU atomic on uncached — works
        if (old == gridDim.x - 1) {
            *signal = 1;    // last block writes signal
            __threadfence();
        }
    }
}
```
The atomicAdd here is intra-GPU on uncached memory. Works fine. The signal write
is to peer (receiver) memory and is a regular volatile write, not an atomic.

**Pattern C: CPU as arbiter**
For complex coordination (locks, CAS, fetch-and-add with returned value):
- src GPU writes intent to host-mapped memory
- CPU reads, applies, writes result back
- src GPU reads result via host-mapped memory
- ~5 µs round-trip. Only viable if not in tight inner loop.

**Pattern D: Single-writer protocol (no atomicity needed)**
For most multi-GPU producer/consumer patterns, you can avoid atomics by giving each
GPU its own write-only region and having the consumer read all regions sequentially.
This is what braidinfer's MoE dispatch does (each expert writes to its own slot;
GPU 0 reads all slots).

### What you can't easily simulate

- **True fetch-and-add** with returned old value across GPUs: hard. Workaround is to
  pre-allocate slots so each contributor knows its index without needing a counter.
- **CAS for distributed lock acquisition**: hard. Workaround is to use the CPU as the
  lock holder (5 µs to acquire/release) or design lock-free.
- **Atomic min/max across GPUs**: hard. Workaround is per-GPU local min/max + reduce
  via Pattern A.

### Bottom line

**Don't write code that depends on cross-GPU atomic operations.** Decompose into
(peer write to dedicated slot + intra-GPU atomic + reader-side aggregation). All
braidinfer cross-GPU code already follows this pattern via `dispatch_batch` host-mapped
mailbox. Keep it that way.

## `[[clang::atomic]]` annotation experiment — DOES NOT FIX (verified 2026-05-02)

Tested 4 variants on GPU pair 1↔2 (`p2p_atomic_scope_test.hip`):

| Variant | Code | Result |
|---|---|---|
| 0 | `atomicAdd(ptr, 1)` (AGENT scope) | Kernel runs, atomic never completes, 8s deadline → HANG |
| 1 | `atomicAdd_system(ptr, 1)` (SYSTEM scope + `[[clang::atomic(fine_grained_memory, remote_memory)]]` per HIP headers) | Hangs BEFORE first DBG print — runtime/JIT can't handle the annotation |
| 2 | `__hip_atomic_fetch_add(ptr, 1, RELAXED, SCOPE_SYSTEM)` (no annotation) | (untested due to v1 hang in same process) |
| 3 | Same as 2 but with manual `[[clang::atomic(fine_grained_memory, remote_memory)]]` block | (untested) |

**Verdict**: The `[[clang::atomic]]` annotation is documented in `/opt/rocm/include/hip/amd_detail/amd_hip_atomic.h` and used by the `*_system` family, but it does NOT fix the cross-GPU atomic hang on consumer RDNA3. The annotation's job is to tell the AMDGPU compiler backend what memory model to assume; it can't paper over missing runtime/driver plumbing for PCIe atomic completer routing.

## Findings on cross-GPU atomicAdd hang — RESOLVED as known-broken on consumer RDNA3

PCIe AtomicOpsCap (32+/64+) is set on every GPU + every upstream bridge (verified via lspci). Yet `atomicAdd` from kernel A on GPU 0 to a peer-VRAM counter on GPU 1 HANGS — kernel never returns.

**Root cause** (per ROCm changelog research, 2026-05-02):
- RDNA3 (gfx1100) has **no xGMI / Infinity Fabric** between GPUs. Cross-GPU atomics must travel over PCIe.
- AMD's amdkfd kernel driver does not reliably complete system-scope atomic RMW operations to peer VRAM on consumer Radeon. The GPU stalls all waves on the chip waiting for PCIe round-trip to complete; if the remote completer doesn't ACK in the expected window, hang.
- This is **known-broken across ROCm 5.6 → 7.2.2** with no fix in sight. ROCm 8.x **does not exist** (current ceiling is 7.2.2).
- AMD officially gates the working multi-GPU P2P path to Instinct/CDNA (MI series). Consumer Radeon shows the PCIe AtomicOpsCap bit but lacks the runtime plumbing.

**Reference issues**:
- [ROCm/ROCm #2429](https://github.com/ROCm/ROCm/issues/2429): "PCIe atomics not enabled, hostcall not supported on gfx1100 RX7900" — closed without fix.
- [ROCm/rccl #1055](https://github.com/ROCm/rccl/issues/1055): Multi-GPU invalid pointer on 7900 XTX, failed across ROCm 5.6 → 6.0 — closed without fix.

**Workaround / mitigation paths**:
1. **Abandon cross-GPU `atomicAdd`** — use device-scope atomics within each GPU + SDMA/memcpy + local atomics for cross-GPU coordination. This is the only reliable pattern on consumer RDNA3.
2. Try `[[clang::atomic]]` attribute (ROCm 7.0+) with `remote_memory` / `fine_grained_memory` hints. Tells AMDGPU backend what scope to assume; may help but unverified for our workload.
3. **`iommu=pt` tension**: AMD's official recommendation for multi-GPU is `iommu=pt`. But our prior testing showed `iommu=pt` is also a HOST CRASH multiplier (bad GPU DMA from misbehaving kernels corrupts host memory → CPU MCE). For development/test: `iommu=pt` OFF (bad DMA → IO_PAGE_FAULT, no crash). For production multi-GPU MoE: may need `iommu=pt` ON.
4. Consider `amdgpu.tmz=0` if seeing TLB fence timeouts alongside hangs.

**RDNA4 status** (gfx1200/1201, RX 9000 series): supported in ROCm 7.1.0+ for compute. Whether RDNA4 fixes cross-GPU PCIe atomics is unconfirmed — no public documentation. RDNA4 still has no xGMI on consumer cards.

## Open probes still pending

These need a quiesced system (no sibling Claude sessions, only llama-server on GPU 3) +
`iommu=pt` off + the safety harness from `exterior_algebra/scripts/p2p_isolated_run.sh`:

- **Full 21-pair latency matrix** (host_mapped + uncached_vram). 4/42 datapoints collected
  before TTM-wedge risk forced abort. The script is ready (`p2p_matrix_run.sh`); needs
  quiescent run.
- **Why does cross-GPU `atomicAdd` hang despite PCIe atomics enabled in hardware?**
  Test 4d (CAS pingpong on uncached VRAM) probes this differently; risky; needs single-pair
  isolated run with abort-on-hang.
- **Why does `uncached_vram` work on GPU0↔1 but hang on GPU0↔2?** Topology-specific —
  may be the cross-bridge transition. Test on each topology class once available.
- **Bisect when host_mapped variance starts** — single-pair fresh-state was 5.06 µs;
  same pair under contention runs 211-412 µs. Quantify the contention model.
- **Phase 4c (`buffer_gl1_inv` reader-side invalidate) confirmation** — bz0's d139f9f
  probe is the field test for this; PCIe-side standalone test still pending.
- **Phase 4b (write-combining host memory)** for cross-GPU CPU-staged signaling.
- **Phase 5 (lock-free SPSC ring buffer between GPUs)** — likely doesn't work given
  cross-GPU atomic_add hangs, but worth confirming.

## Recovery checklist after a TTM wedge

When `hipGetDeviceCount` hangs / `rmmod amdgpu` blocks in D state:

1. Check `cat /proc/modules | grep amdgpu` — refcount `-1 - Unloading` confirms wedge.
2. Check `ps -e -o pid,stat,wchan,cmd | grep ttm` — D-state TTM kworkers confirm.
3. Try `rocm-smi --gpureset -d N` for affected GPUs — usually does NOT clear TTM state.
4. Try `modprobe -r amdgpu && modprobe amdgpu` — only after killing all GPU openers.
5. If rmmod hangs (D-state), only **REBOOT** recovers.

## What braidinfer should do based on this catalog

- **Cross-GPU coordination: use host-mapped CPU mailbox** (~5 µs single-pair fresh,
  may degrade under contention). This is what `dispatch_batch` already uses; keep it.
- **NEVER add cross-GPU atomic operations** to MoE dispatch or any other path. They are
  not supported on consumer RDNA3 — known-broken since ROCm 5.6 with no fix planned. Use
  device-scope atomics within each GPU + SDMA/memcpy + local atomics for cross-GPU
  coordination. This is the only reliable pattern.
- **For multi-WGP HBM-shared state within a single GPU**: insert reader-side
  `buffer_gl1_inv` (or `__builtin_amdgcn_buffer_wbinvl1_vol()`) at kernel entry that
  reads HBM written by a prior kernel. This is what bz0 is testing in d139f9f.
  Note: standalone test (`p2p_gl1_inv_test.hip`) shows HIP runtime DOES invalidate L1
  between simple cooperative launches without the explicit op — bz0's pattern must
  require more specific conditions.
- **Do not introduce primitives that wait on global completion ordering**:
  `__threadfence_system`, `__hip_atomic_load(SYSTEM)`, `s_waitcnt vmcnt(0)` in megakernels.
- **GPU 3 is permanently held by llama-server** — multi-GPU code that needs all 8 will
  collide. Either skip GPU 3 or coordinate with llama-server lifetime.
- **iommu=pt is a tradeoff**: AMD's official multi-GPU recommendation is `iommu=pt` ON,
  but this turns misbehaving kernels into host crashes (CPU MCE from DMA-corrupted RAM).
  For development: `iommu=pt` OFF (bad DMA → IO_PAGE_FAULT, no crash). For production:
  test with both modes; keep ON only if all cross-GPU code is proven safe.
- **Recommended kernel params for RDNA3 stability** (per
  [RDNA3 stability guide](https://gist.github.com/danielrosehill/6a531b079906f160911a87dea50e1507)
  and ROCm changelog research):
  - `amdgpu.tmz=0` — disables Trusted Memory Zone, reduces TLB Fence Timeouts (RDNA3-specific kernel bug)
  - `amdgpu.gfx_off=0` — disables power gating; helps avoid wake-from-idle wedges during long stress runs
  - `amdgpu.lockup_timeout=10000,10000,10000,10000` — extends GFX/COMP/SDMA/VIDEO timeout to 10s (already set on this system)
  - `amdgpu.mcbp=0` — disables mid-command-buffer preemption (already set; helps stability for cooperative kernels)
  - `iommu=soft` — alternative to `iommu=pt`/default if both have problems
- **No upgrade path available**: ROCm 8 does not exist (current ceiling is 7.2.2). Don't
  wait for a runtime fix to the cross-GPU atomicAdd hang — design around it.

## 2026-05-03 — Findings from systematic 28-pair × 7-primitive sweep + watchdog Phase 4

Three load-bearing findings landed on 2026-05-03 from the exterior_algebra `aat`
topology epic (results: `../exterior_algebra/results/peer_topology_full.json`,
302 measurements) and the `5pc` watchdog epic (commits in
`braidinfer/.worktrees/watchdog`, branch `feature/watchdog-primitive`):

### Finding 1: `hipDeviceReset` cannot recover a wedged compute kernel on RDNA3

Watchdog Phase 4 stubborn-buggy variant: launched `while(1){}` cooperative kernel,
called `hipDeviceReset` from host. **Blocked indefinitely (15+ min observed)**. ROCm
on gfx1100 has no GPU TDR (Timeout Detection and Recovery) for compute kernels;
`hipDeviceReset` waits for the kernel to exit naturally, which never happens for a
deliberately-spinning kernel.

**Implication for production**: any persistent or long-running cooperative kernel
must be designed for cooperative recovery only (kernel polls a `force_exit` flag and
exits voluntarily). When that fails, the only recovery path is host-process death
(`std::process::abort()` or SIGKILL), which triggers driver-level GPU context cleanup.
There is no soft single-GPU recovery mechanism on consumer RDNA3.

The braidinfer watchdog primitive (`kernels/watchdog.h` + `crates/braidinfer-runtime/src/watchdog.rs`)
implements this: cooperative-exit when the kernel honors `force_exit`, telemetry-dump-then-abort
otherwise. Cooperative-path validated 100/100 PASS in unit testing (mean recovery 4727 ± 667 µs).

### Finding 2: Latent block-count bug in braidinfer `moe_p2p.rs` (FIXED in watchdog branch)

Probed `moe_worker_kernel` via `hipOccupancyMaxActiveBlocksPerMultiprocessor` on
gfx1100 (256 threads, register/shared-mem usage): result is **9 blocks/CU**. The
correct cooperative launch is therefore `48 CUs × 9 blocks = 432 blocks`. Existing
braidinfer code launched 48 blocks (1/CU). With 432 expected slots vs 48 launched,
all `grid.sync()` calls would have been waiting forever for the 384 absent blocks —
this is a latent intermittent-hang risk that may have caused unattributed wedge
incidents. Fixed in `feature/watchdog-primitive` branch.

Going forward: any new cooperative kernel launch in braidinfer must call
`hipOccupancyMaxActiveBlocksPerMultiprocessor` and assert the launch matches.

### Finding 3 [RETRACTED 2026-05-03]: `peer_uncached_signal` was test-side bug, not hardware

Initial reading: 28/28 pair TIMEOUT for "writer writes MTYPE_UC peer VRAM, reader spin-polls."

**Corrected diagnosis (after the c3↔83 system crash on 2026-05-03)**:
The TIMEOUT was caused by a peer-mapping bug in the test, not a hardware limit. Sequence
of failure:

1. Test allocated `flag_dev` on gpu_b via `hipExtMallocWithFlags(hipDeviceMallocUncached)`.
   This API uses a **special "fine-grained noncached" memory pool** distinct from the
   standard device allocation pool.
2. `hipDeviceEnablePeerAccess(gpu_b, 0)` from gpu_a auto-maps the **standard pool's**
   allocations into gpu_a's address space, but does NOT auto-map the noncached pool.
3. When the writer kernel on gpu_a executes `*flag_dev = r`, the GPU MMU has no peer
   translation for that address. The hardware reports `GCVM_L2_PROTECTION_FAULT` with
   `PERMISSION_FAULTS: 0x3` — appears as "permission denied" but functionally is "no
   peer mapping for this address from this GPU."
4. The kernel got stuck in fault recovery; HIP timed out the stream.
5. On the c3↔83 cross-IO-die pair specifically, the fault recovery race escalated to
   a full system crash. (See journalctl --boot=-1 for the page fault trail showing
   adjacent virtual addresses, same PID, faults on both GPUs simultaneously.)

The crash was a kernel/firmware fault-recovery escalation, not a PCIe hardware fault.
Both cards link as healthy at full Gen 4 ×16 with `EqualizationComplete+`.

**Fix paths**:
- (test-side) Call `hipDeviceEnablePeerAccess` BEFORE the `hipExtMallocWithFlags`. Some
  paths auto-map allocations into all currently-peer-enabled devices.
- (correct fix) After the uncached allocation, explicitly call
  `hsa_amd_agents_allow_access(num_agents, agents, NULL, ptr)` to add the peer agent to
  the allocation's access list.
- (architectural fix) Update the `hsa-rocr` MTYPE_UC patch to also trigger on the uncached
  pool's allocation hook, so peer mappings are auto-created for that pool.

**Why `gpu_to_gpu_peer_write` (Primitive 6) WORKS at 2.3 µs across 26/28 pairs**: it uses
regular `hipMalloc` (standard pool) for the peer-VRAM target, which IS auto-peer-mapped.
The MTYPE on the peer side reflects whatever the standard mapping path uses — which
prior to our hsa-rocr patch was MTYPE_CC (and saw L2-staleness bugs), and post-patch is
MTYPE_UC for `whitelist_nodes.size() > 1` peer mappings.

So peer VRAM writes DO land at sub-3-µs latency cross-GPU. The "MTYPE_UC peer signaling
broken on RDNA3" claim was wrong; it was a test-code bug.

**Result of re-test after Fix A (peer enable before alloc) landed**: the page-fault crash
is gone (no more dmesg faults, no wedge), but the spin-poll still TIMEOUT on all 3 verified
pairs (ROCm1+2, ROCm6+7, ROCm1+6). Three additional reader-side experiments tried and all
failed to make the primitive observe writes:

1. Insert dummy host-pinned write inside reader spin loop (analogous to k_pong_agg's
   `*ack = r` write) — TIMEOUT
2. Insert `__threadfence()` in reader spin loop — TIMEOUT
3. Replace `volatile *src` read with `__hip_atomic_load(ACQUIRE, AGENT)` — TIMEOUT

So the reader-side cache is NOT the issue. But primitive 6 (`gpu_to_gpu_peer_write`) reads
the same MTYPE_UC peer-VRAM memory at 2.3 µs in `k_pong_agg`. The asymmetry is not
explained by reader-side mechanisms.

**Untested hypothesis for the asymmetry**: k_one_writer has `__builtin_amdgcn_s_sleep(1)`
between stores; k_ping_agg has `while (*ack != r)` (a peer-host read) right after each
store. If RDNA3 has a coalescing write buffer in the LSU, the read in k_ping_agg forces
the prior store to drain via memory ordering; k_one_writer's sleep does not. This would
make the bug "writer store buffer not draining," not "reader cache stale." Worth testing
in a future round if the persistent megakernel architecture needs this primitive working.

**Practical conclusion for now**:
- The single-writer + cross-GPU peer-VRAM polling spin pattern as currently written does
  NOT work on RDNA3 in this test, regardless of cache invalidation discipline on the
  reader side
- The crash on c3↔83 was a real bug (Fix A applied); other pairs were always cleanly
  timing out (no faults)
- **Use `gpu_to_gpu_peer_write` (k_ping_agg + k_pong_agg ping-pong) as the working
  cross-GPU GPU-VRAM coordination primitive at ~2.3 µs**
- Do NOT use the single-writer-then-poll pattern; replace with ping-pong if cross-GPU
  peer-VRAM signaling is needed

Tracked in `results/peer_topology_phase1_5_uncached_fix.json`. Source comments in
`scripts/p2p_latency_matrix.hip` `k_one_reader` document the exhausted experiments for
the next investigator.

### Finding 4: NUMA/per-pair anomalies dominate over root-complex grouping

For `host_mapped_roundtrip` (the most common cross-GPU coordination primitive), root-complex
grouping is NOT predictive of latency. Several cross-root pairs are faster than several
same-root pairs:

- Best pair: topo (5,6) [same-root] at p50=3.81 µs
- Best cross-root: topo (6,7) at p50=3.89 µs
- Worst same-root: topo (3,4) at p50=**795 µs** (200× anomaly)
- Worst cross-root: topo (0,3) at p50=26.78 µs (still 6× normal)

**Topology GPU 0 (PCI c3:00.0 / HIP index 5) showed apparent latency anomalies across
multiple primitives** (host_mapped, mailbox, peer_write, segmented_graph). Investigated
2026-05-03: `rocm-smi --showtemp` shows normal temps (53/57/52 °C); `lspci -vvv` shows
"AMD GPU device(s) is/are in a low-power state" warning when system is idle.

**Root cause: amdgpu runtime power management.** When GPUs are idle, amdgpu transitions
them to low-power states (D3-cold or similar). First cross-GPU access on a sleeping GPU
must wake it up, producing 2-7× higher latency for the first pair tested. The (0,1) pair
was the FIRST pair in the sweep order, so it caught the deepest sleep state. Subsequent
(0,*) pairs were progressively faster as GPU 0 warmed up. NOT a hardware issue.

The `Status: ... MAbort+` (sticky master-abort) bit on c3:00.0 is set as a side effect of
prior cross-GPU `atomicAdd` hang attempts (PCIe transactions that got no response register
as master aborts on the target). Sticky but not actively affecting current operation.

**For benchmarking**: pre-warm all GPUs with a no-op kernel before measurement, OR set
`amdgpu.runpm=0` kernel parameter to disable runtime PM (revert for production deployment).

**For multi-GPU placement**: avoid the (3,4) pair for host-mediated handshake (795 µs
host_mapped_roundtrip is real and not power-state related — both GPU 3 and GPU 4 were
tested adjacent in the sweep order, so warm-up is not the cause; likely NUMA mismatch
on the pinned-host page for that specific pair). Otherwise no GPU has a confirmed
hardware-level handicap.

**Large-BAR confirmed**: `Region 0: Memory at ... [size=32G]` on c3:00.0 — full VRAM
peer-addressable. No BAR-resize work needed for the udi (amdkfd doorbells) Phase 3 BAR
audit; large-BAR is already in effect.

### Root cause for cross-GPU atomic hangs: EPYC root ports lack AtomicOps Routing (2026-05-03)

`sudo lspci -vvv` of the c3↔83 PCIe path shows the root ports c0:01.1 and 80:01.1 have:
- `AtomicOpsCap: Routing- 32bit+ 64bit+ 128bitCAS-`
- `AtomicOpsCtl: ReqEn- EgressBlck-`

**`Routing-`** = the root port has 32+64-bit AtomicOps Completer support (it can BE the target
of an atomic, e.g., for atomics to host memory) but **WILL NOT FORWARD atomic requests to other
PCIe sub-trees**. Per PCIe spec, every link in a peer-to-peer atomic path needs `Routing+`.
With root ports refusing to route, peer-to-peer atomics across the EPYC root complex are
**not defined to work** by the hardware contract.

This is a Rome (Zen 2 / EPYC 7xx2) silicon limitation. Milan (Zen 3) and Genoa (Zen 4) have
better PCIe atomic-routing support. BIOS settings cannot enable Routing+ on hardware that
doesn't implement it. **No firmware fix is possible on this hardware; the only paths forward
are software-emulated atomics or upgrading the host CPU.**

Implications:
- **All cross-GPU `atomicAdd`/`atomicCAS`/`atomicExch`/etc. with `__HIP_MEMORY_SCOPE_SYSTEM`
  are guaranteed to never complete** on this host. They wait forever for a completion ACK
  that the silicon refuses to generate.
- Same applies to **same-root** peer atomics — root ports also refuse to route between two
  of their own downstream sub-trees, not just across IODs.
- HIP runtime should check the PCIe routing capability before issuing peer atomics and either
  reject the call or transparently fall back to a software path. **It does not.** This is a
  HIP/ROCr layer bug worth filing upstream.
- **Posted writes** (e.g., `*peer_addr = value`) work fine cross-GPU because they're
  fire-and-forget at the PCIe level — no atomic completion required.
- The recovery path's IF transient (which can escalate to whole-system MCE on certain pairs
  like c3↔83) is a **separate** Linux-kernel/amdgpu bug worth filing independently.

**Runtime detection**: `scripts/p2p_atomic_capability.py` (added 2026-05-03) reads
`/sys/bus/pci/devices/.../config` and verifies AtomicOps Routing on every link in the path
between two GPUs. Use this in any code that contemplates issuing cross-GPU atomics — it will
tell you "no path supports this, fall back to software" before the hang happens.

### Additional anomalies surfaced by the systematic sweep (2026-05-03)

**GPU 4 (PCI 86:00.0, root 0080) is 3.5× slower for 1 MB memcpy_peer_async.**
All pair measurements involving GPU 4 at 1 MB payload show ~322-329 µs vs normal ~91 µs.
Affects both same-root (3,4) and all cross-root pairs (0-2,4) and (4,5-7). Possible
PCIe link degradation, slower SDMA engine on that specific card, or persistent thermal
throttle (memory at 62 °C in idle state, highest of the 8 cards). Worth investigating
with `lspci -vvv -s 0000:86:00.0` for link state, retrains, or AER errors.

**Pair (3,7) [root 0080 ↔ root 0000] times out on host_mapped_roundtrip and
gpu_to_gpu_peer_write.** Other primitives (memcpy 13-55 µs, segmented_graph 213 µs)
work fine on the same pair. This is a specific cross-root pair issue, NOT a generic
cross-root problem (most cross-root pairs work normally). Suggests TTM or driver-level
state for this specific root-complex pairing. Avoid as a critical path in production.

**HARDWARE-LEVEL CONFIRMED HEALTHY (2026-05-03 post-crash investigation)**: Both PCI 83
and PCI c3 lspci as healthy. `LnkSta: Speed 16GT/s, Width x16`. `EqualizationComplete+`.
All `UESta` flags negative. The crash on the c3↔83 pair was a userspace GPU page-fault
escalation triggered by a peer-mapping bug in our test (see Finding 3 retraction above),
not a degraded link. Reseating cards is NOT necessary. The kernel-level escalation of a
userspace page fault to a full-system crash IS a real amdgpu/firmware issue worth filing
upstream — a userspace fault should kill the process, not the system.

**Slot map for the affected cards** (ASRockRack ROMED8-2T/BCM motherboard):
- PCI c3:00.0 = SMBIOS slot ID 19 (upstream switch at c1:00.0). Currently has display cable.
- PCI 83:00.0 = SMBIOS slot with bus 81:00.0 upstream.
- Both cards confirmed at full Gen 4 ×16 link, no AER errors.

**`gpu_to_gpu_peer_write` is the fastest reliable bidirectional primitive at ~2.3 µs
across all paths** (same-root and cross-root, no significant difference). Recommended
as the primary GPU↔GPU signaling primitive for the persistent megakernel architecture
(uses peer-VRAM writes + host-mapped ack pong; works on 26/28 pairs).

**Power-state caveat for benchmarking**: rocm-smi reports "AMD GPU device(s) is/are
in a low-power state" warning when system is idle. `amdgpu.gfx_off=0` only disables
the GFXOFF feature, not D3 runtime PM. Use `amdgpu.runpm=0` kernel argument to disable
runtime PM entirely (or `echo on > /sys/bus/pci/devices/.../power/control` per device
at runtime). Without this, first cross-GPU access on a sleeping GPU has 2-7× higher
latency as the device wakes up — explains apparent "first-pair anomalies" in cold-start
sweeps. Tradeoff: ~30 W extra idle power per GPU.

## References

- braidinfer prior P2P commits: `4f56691`, `574b729`, `663731c`
- braidinfer Bug 1 fix (RNF4 multi-GPU MoE dispatch): `071a9ff`
- braidinfer bz0 candidate L1-invalidation probe (UNVERIFIED): `d139f9f`
- braidinfer watchdog primitive (cooperative recovery + abort escalation):
  `feature/watchdog-primitive` branch in `.worktrees/watchdog`
- exterior_algebra PCIe research: `../exterior_algebra/scripts/p2p_*.hip`,
results in `../exterior_algebra/results/peer_topology_full.json` (302-measurement sweep,
2026-05-03), `../exterior_algebra/results/pcie_topology.json` (definitive 8-GPU PCIe
topology from `lspci -tv`).
