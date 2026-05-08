# braidinfer

GPU-native LLM inference engine in Rust + HIP, targeting AMD RDNA3 (gfx1100,
RX 7900 XTX). See `CLAUDE.md` for build instructions and architecture.

## Host system tuning for low-latency dispatch

braidinfer's persistent megakernel is a cooperative HIP kernel that holds all
CUs on its GPU for the duration of inference. Work is dispatched via a
host-mapped mailbox (`WorkerQueue::seq_num` / `ack`): the host writes a
sequence number and spin-polls the ack flag, since cooperative kernels conflict
with HIP's interrupt/event paths (any DMA under a cooperative kernel deadlocks
on RDNA3, kb `77r-2-1`).

The spin-poll runs at normal scheduler priority by default. Under CPU
contention the host can be descheduled mid-poll, leaving the GPU idle until
the dispatch thread is rescheduled. Per-batch round-trip rises from
microseconds to tens of milliseconds, producing 5-23x decode throughput
regressions. The fix is `SCHED_FIFO` + CPU affinity on the dispatch thread.

### Required configuration (one-time)

#### `/etc/security/limits.conf`

    <user> - rtprio 99

`-` means both soft and hard. Default `rtprio` for unprivileged users is 0,
which makes `sched_setscheduler(SCHED_FIFO, ...)` return `EPERM`. Setting to
99 allows the highest SCHED_FIFO priority. Takes effect on next PAM login.
To apply to an already-running shell without relogin:
`sudo prlimit --rtprio=99 --pid=$$`.

If `hipHostMallocMapped` host buffers should be mlocked without limit:

    <user> - memlock unlimited

#### `/etc/sysctl.d/99-sched-fifo.conf`

    kernel.sched_rt_runtime_us = -1

Default is 950000 (95% RT throttle per 1-second window). For a CPU-spin
dispatch loop running at 100%, throttling injects 50 ms stalls every second
— exactly the latency RT scheduling is meant to eliminate. `-1` disables
throttling entirely. Apply without reboot:

    sudo sysctl --system

Verify: `cat /proc/sys/kernel/sched_rt_runtime_us` shows `-1`.

#### Kernel cmdline core isolation (8-GPU box, recommended layout)

For an 8-GPU host, reserve **9 cores** at the top end of the CPU range
(64-core box uses cores 55-63):

    isolcpus=55-63 nohz_full=55-63 rcu_nocbs=55-63

After editing `/etc/default/grub`:

    sudo grub-mkconfig -o /boot/grub/grub.cfg
    sudo reboot

Layout once active:

| Core | Pinned to | How |
|---|---|---|
| 55 | braidinfer dispatch thread | server-side `sched_setaffinity` (single thread polls all 8 GPU mailboxes) |
| 56 | amdgpu card 0 IRQ | `/proc/irq/<N>/smp_affinity` via `amdgpu-irq-pin.sh` |
| 57-63 | amdgpu cards 1-7 IRQs | same |

What each cmdline param does:

- `isolcpus=` removes listed cores from the general scheduler's
  load-balancing domain. Only tasks explicitly affined via
  `sched_setaffinity()` (or `taskset -c`) will land there.
- `nohz_full=` runs those cores tickless when only one task is runnable;
  eliminates microsecond-scale scheduler jitter from the periodic timer.
- `rcu_nocbs=` moves RCU callback processing off the isolated cores so
  unrelated subsystems don't briefly run there.

amdgpu IRQ pinning notes:

- On stock kernels (no `threadirqs` cmdline, no `force_threading=1`)
  amdgpu IRQs are **hard IRQs only**: `/proc/irq/<N>/smp_affinity` is
  the complete pinning. There are no `irq/<N>-amdgpu` kthreads to also
  pin.
- If `threadirqs` is in the cmdline OR amdgpu is built with
  `force_threading=1`, threaded IRQ handlers (`irq/<N>-amdgpu`
  kthreads) DO exist and must be pinned separately via `taskset -p` and
  optionally promoted to `SCHED_FIFO` at higher priority than braidinfer
  dispatch.
- Verify which path applies: `ps -eLo tid,comm | grep '^\s*[0-9]\+\s\+irq/.*amdgpu'`
  — if empty, hard IRQs only.

Why core 55 for braidinfer (not co-located with IRQs):

- With hard amdgpu IRQs, co-locating works fine — IRQs preempt
  `SCHED_FIFO` automatically.
- A separate dispatch core is preferred for clean perf measurements
  (no IRQ cycles attributed to dispatch) and for future-proofing
  against a future kernel that flips amdgpu to threaded IRQs (which
  would otherwise starve dispatch via SCHED_FIFO competition).

Not required for single-process workloads or single-user dev boxes —
`SCHED_FIFO` + dispatch-thread affinity to any one CPU is sufficient.
The 9-core layout is the production-server recipe.

### Required cmdline tuning (independent of RT)

The following kernel cmdline parameters are necessary baseline; most distros
do not set them:

| Parameter | Effect |
|---|---|
| `processor.max_cstate=1` | Disables deep CPU C-states (no wakeup latency) |
| `transparent_hugepage=always` | Hugepage backing for working sets |
| `numa_balancing=disable` | No automatic NUMA page migration |
| `amd_pstate=passive` | Fixed P-state, no governor lag |
| `pci=pcie_bus_perf` | High-performance PCIe MPS sizing |
| `amdgpu.runpm=0` | Disables GPU runtime power management |
| `amdgpu.gfx_off=0` | Keeps GPU graphics block always on |
| `amdgpu.mcbp=0` | Disables mid-command-buffer preemption |

### Cores for the dispatch thread

The server architecture (epic `braidinfer-wks`, in progress) consolidates
dispatch onto a **single thread** that polls all 8 GPU mailboxes
(`try_wait_acks_many` in `crates/braidinfer-runtime/src/persistent_dispatch.rs`,
landed in commit `977f002`). One core serves one server — and that
server can have an arbitrary number of clients, since each in-flight
generation is just a few hundred nanoseconds of mailbox-polling work
amortized over ~10 ms of GPU compute per dispatch.

The pre-server in-process binary (`bin/generate`) follows the older model
where each invocation owns its own dispatch loop. Running multiple
concurrent `generate` processes is not the supported multi-tenant
configuration — for that, run `braidinfer-server` once and connect
multiple clients (Phase 4+ of the epic).

### Verification

Per-batch dispatch round-trip is exposed via `DISPATCH_RTT=1`:

    DISPATCH_RTT=1 MODEL=models/qwen35_2b.q4.bqnt RAW=1 MAX_TOKENS=20 \
      target/release/generate "Hello"

Each `dispatch_batch` call prints `rtt=<microseconds>`. Single-GPU decode on
an unloaded system: median <1 ms. Under CPU contention without RT scheduling:
10-50 ms+ (this is the regression RT scheduling fixes).

For per-op cycle breakdown of one decode:

    BRAIDINFER_OP_PROFILE=1 cargo build --release -p braidinfer-runtime --bin op_profile_dump
    BRAIDINFER_OP_PROFILE=1 MODEL=models/qwen35_2b.q4.bqnt MAX_TOKENS=50 \
      target/release/op_profile_dump

Reports per-opcode ticks/call (10 ns/tick on gfx1100) and achieved memory
bandwidth for known-shape ops (currently lm_head).

### Watchdog

Persistent cooperative kernels include a host-side watchdog that detects
wedged kernels and escalates to `std::process::abort()` (`hipDeviceReset`
blocks indefinitely on RDNA3 — there is no GPU TDR preemption for compute on
gfx1100). Tunables:

| Variable | Default | Description |
|---|---|---|
| `BRAIDINFER_WATCHDOG_NO_PROGRESS_MS` | 2000 | ms with no progress before declaring stuck. Set 0 to disable. |
| `BRAIDINFER_WATCHDOG_GRACE_MS` | 1000 | ms after force_exit before escalating to abort |
| `BRAIDINFER_WATCHDOG_POLL_MS` | 100 | host poll interval |
