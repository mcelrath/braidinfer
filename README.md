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

#### Optional: kernel cmdline core isolation

For multi-tenant deployments (multiple concurrent braidinfer processes),
add to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

    isolcpus=56-63 nohz_full=56-63 rcu_nocbs=56-63

Adjust the CPU range to dedicate one core per concurrent process. After editing:

    sudo grub-mkconfig -o /boot/grub/grub.cfg
    sudo reboot

- `isolcpus=` removes listed cores from the general scheduler's load-balancing
  domain. Only tasks explicitly affined via `sched_setaffinity()` will land
  there.
- `nohz_full=` runs those cores tickless when only one task is runnable;
  eliminates microsecond-scale scheduler jitter.
- `rcu_nocbs=` moves RCU callback processing off the isolated cores.

Not required for single-process workloads — `SCHED_FIFO` plus sane affinity
is sufficient.

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

### Cores to dedicate

| Workload | Cores needed | Note |
|---|---|---|
| Single braidinfer process | 1 | One pinned dispatch thread |
| N concurrent processes | N | SCHED_FIFO threads of equal priority do not preempt each other; two spinners on one core = one starves |
| Single process, multi-GPU MoE | N (one per GPU) | Each GPU has its own dispatch thread in the same process |

A single thread can poll all 8 GPU mailboxes in one tight inner loop (each
mailbox check is sub-µs); this would let one core serve eight GPUs but is
an architectural change to the dispatch path, not currently implemented.

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
