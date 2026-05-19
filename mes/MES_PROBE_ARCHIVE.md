# MES Probe Archive — §11.13 Cooperative-Grid Relaunch Wedge (extracted from GFX1100_ARCH.md)

Preserved empirical record for the gfx1100 cooperative-grid relaunch wedge.
Mechanism remains unknown; six MES-side patches refuted; process exit is
the only known recovery. Cross-reference: `GFX1100_ARCH.md` §11.13 stub
(section number preserved for external refs) and §11.15 (the
immediate-ack protocol fix that established §11.13 is a separate
phenomenon from the deferred-ack deadlock).

---

### 11.13 Cooperative-grid relaunch wedge — empirical archive (confirmed; mechanism unknown)

**History (2026-05-14, settled).** The "Re-confirmation" text that
previously occupied this slot — claiming the production wedge was a
Rule 9 violation via per-token cooperative `mk.execute()` in
`prefill_paged` — was wrong. A standalone reproducer
(`braidinfer kernels/diagnostic/persistent_skeleton_repro/prod_kernel_test`)
loading `megakernel.hsaco`'s `persistent_worker` symbol with a fresh
`WorkerQueue` and ZERO prior cooperative launches also wedged with the
identical fingerprint. The actual root cause was a Phase 2'
deferred-ack protocol deadlock inside the worker (§11.15). The
6-hour 2026-05-14 chase through bulk-write, VCC/readfirstlane
hazards, gl0/gl1_inv (silently no-op on gfx11+, §11.14),
`__threadfence_system`, and 10 GB prior-DMA pressure was all real
evidence ruling things out, but the conclusion that "Rule 9 fully
explains it" was incorrect.

**Status post-investigation: Rule 9 stands as a separate, real
phenomenon independent of §11.15.** Verified 2026-05-14 by inspection
of the V0 skeleton's ack-write code
(`persistent_worker_skeleton.hip:205-213`): V0 uses **immediate-ack**
(writes `ack=seq` in the same iteration that processed `seq`). V0's
trial-2-wedge in cross-launch baseline therefore cannot be the
deferred-ack deadlock and represents a true relaunch-state
phenomenon. The negative-probe archive below is the canonical record
of attempts to clear the relaunch state mid-process; all failed.
Process exit remains the only known recovery path.

**Wedge fingerprint.** Persistent cooperative kernel polls a host-mapped
MTYPE_UC `seq_num` field. Host writes `seq=1`; CPU readback confirms
the write landed in host RAM. Worker's volatile read of the same
address returns 0 forever. `dmesg` SILENT during wedge. debugfs
`amdgpu_fence_info` shows fences ADVANCING (gfx + compute) over the
30s wedge interval — MES scheduling continues normally. The wedge is
specific to the SECOND-or-later `hipLaunchCooperativeKernel` call's
FIRST dispatch within a process.

**Verified pattern (rock-solid, 2 complementary tests).**

| Test | Mode | Result |
|---|---|---|
| `skeleton_V0` PERSISTENT_KERNEL=1 n=100 | One cooperative kernel, 100 seq increments without exit/relaunch | 100/100 PASS, monotonic `completed_dispatches`=1..100, each dispatch <1ms |
| `skeleton_V0` default mode n=3 | 3 separate cooperative-kernel launches (launch / dispatch / shutdown / repeat) | trial 1 PASS, trials 2-3 WEDGE with 31s host timeout |

The pattern matches every prior multi-trial wedge run (Designs D-I
test logs). Wedge is per-process, per-cooperative-kernel-LAUNCH;
within-kernel dispatch is fine.

**Rule-outs (do not retry).**

User-space probes (15+ from the broader investigation, summary):

| Probe | Result |
|---|---|
| `buffer_gl0_inv`, `buffer_gl1_inv` before load | WEDGED |
| `global_load_b32 ... glc dlc` | WEDGED |
| `global_atomic_or_b32` with mask 0 | WEDGED |
| Wide 16-byte load | WEDGED |
| Inject 128 v_mad cycles between polls | WEDGED |
| Cache-line isolation (60-byte pad) | WEDGED |
| CPU `_mm_mfence`, `_mm_clflush + mfence` after write | WEDGED |
| `hipHostMallocCoherent`, `hipHostRegister` | WEDGED |
| `mmap(MAP_HUGETLB)` | WEDGED |
| Fresh mmap (different physical page) | WEDGED |
| `BENCH_WARMUP=0`, prefill-priming variations | WEDGED |
| Stream rotation (hipStreamDestroy+hipStreamCreate) | WEDGED |
| Non-cooperative kernel interleave between trials | WEDGED |
| 10-100ms zero-queue-delay between trials | WEDGED |

Kernel-side patches (6 attempts; all built and tested against running
`amdgpu.ko` for gfx1100):

| Design | Mechanism | Result |
|---|---|---|
| D | MES MISC NOTIFY_TO_UNMAP_PROCESSES (spec p46) | WEDGE persists; MES processed packets cleanly per dev_info trace |
| F | MES MISC SET_SHADER_DEBUGGER (per kfd_chardev.c comment "first call clears stale process context") | WEDGE persists; HIP already used the first SHADER_DEBUGGER call at hipInit |
| G | Raw MES REMOVE_QUEUE + ADD_QUEUE for one HIP queue with `skip_process_ctx_clear=0` override (spec p25) | WEDGE persists; override applied per trace, MES accepted both packets |
| H | Raw REMOVE-all queues then ADD-all reaching "last gang in process" condition (spec p26) | WEDGE persists; 3 of 3 HIP queues cycled, no MES errors |
| (TLB flush + H) | Per-PASID heavy `amdgpu_gmc_flush_gpu_tlb_pasid(adev, pasid, 2, true, 0)` matching what fires at process-exit teardown, then Design H | WEDGE persists |
| (GPU reset via debugfs) | `cat /sys/kernel/debug/dri/0000:XX:00.0/amdgpu_gpu_recover` | NOT tested empirically — would clear wedge but destroys all in-flight work, not viable for production multi-tenant |

**What actually clears at process exit** (per WEDGE_TRACE
instrumentation, retained as `~/builds/linux-p2p/0011-*.patch` for
future revival): HIP atexit destructor sends `REMOVE_QUEUE` for each
of its queues (typically 3 doorbells: 0x1000 / 0x1002 / 0x1004) in
host_runner process context. Then amdkfd's
`kfd_process_destroy_pdds` fires in a kworker async context, per-pdd:
`destroy_cwsr_dgpu` → `destroy_ib_mem` → `fput(drm_file)` (queues
`amdgpu_vm_fini`) → `kfd_free_process_doorbells` → 2 MES MISC
`WAIT_REG_MEM` packets (per-PASID TLB flush via
`gmc_v11_0_flush_gpu_tlb_pasid`) → `proc_ctx_bo` free. Finally
`amdgpu_vm_fini` runs ×8 in kworker context, batched at the end.
Replicating each of these mid-process individually does NOT clear the
wedge. The wedge-clear is likely an emergent property of the full
teardown sequence, possibly including an amdgpu hung-queue-detection
path that triggers GPU reset late in teardown.

**Cross-chip-family corroboration.** Framework community thread
`community.frame.work/t/71364` has 96+ posts of similar MES wedges on
gfx1103, gfx1150, gfx1152. Mario_Limonciello (AMD) participates;
`amdgpu.mes=0` is a no-op on gfx11+ (hardcoded
`adev->enable_mes=true` at `amdgpu_discovery.c:2588`);
`amdgpu.cwsr_enable=0` helps SOME framework users for ADJACENT wedge
classes (rejected for our case via empirical reboot test).
`buffer_wbl2` and `buffer_gl2_inv` are documented absent from gfx11+
ISA (AMD's own `hsa-rocr-p2p-mtype-uc-gfx11.patch` works around the
writer-side gap; reader-side has no workaround). Our wedge likely
shares the same underlying ISA gap class but the per-process latching
mechanism is firmware-internal.

**Reference artifacts** at
`~/Projects/ai/exterior_algebra/mes/`:
- `AMD_MES_specification_April2024.pdf` (54 pages, authoritative)
- `mes_sch_decompiled.c` / `mes_sch_disasm.s` (942 functions, RV32IMCV)
- `mes_sch_coop_sites.txt` (7 sites with `andi rd,rs,0x200`
  cooperative-dispatch-bit check; container function f000aa2c et al)
- `mes_sch_prime_clear.txt` (43 accesses to offset -0x3dc in MES SRAM
  per-PASID slot; bit-0 set on cooperative dispatch; clear path at
  f000ad50 — verified not our wedge mechanism via Design G/H)

**Production pattern that works** (per Rule 9 §5.5): single
cooperative `persistent_worker` kernel launched at model-init time
BEFORE any other cooperative-grid activity; route all subsequent
dispatch types (prefill, decode, etc.) through doorbell + opcode-mux
in the persistent kernel.

> **2026-05-14 POSTSCRIPT — the "Rule 9 fully explains" conclusion
> above was wrong.** A standalone reproducer (`kernels/diagnostic/
> persistent_skeleton_repro/prod_kernel_test`) that loads
> `megakernel.hsaco`'s `persistent_worker` symbol and dispatches with
> ZERO prior cooperative launches and ZERO process state also wedged
> with the identical fingerprint. Rule 9 cannot be the mechanism if a
> first-coop-launch in a fresh process wedges. The actual root cause
> was a Phase 2' deferred-ack protocol deadlock in the worker's
> outer-body code — `ack=last_seq` was written in iter N+1 AFTER iter
> N+1's inner-poll completion, but the inner-poll required `seq=N+1`
> which the host wouldn't send before observing `ack=N`. Documented
> in §11.15 with the canonical fix and the standing regression test.
> Rule 9 itself (avoid re-launching cooperative kernels) is still
> valid as a separate architectural guideline; the 100/100 within-
> lifetime PASS and the 1+N relaunch WEDGE observations support it.
> But the production wedge braidinfer hit on 2026-05-14 was NOT a
> Rule 9 violation — it was protocol mis-design that the standalone
> reproducer also triggers.
