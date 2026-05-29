#!/usr/bin/env python3
"""GPU-managed launcher for braidinfer tests and binaries.

Reserves GPUs via flock before running any GPU binary (cargo test, benchmarks, etc.).
Prevents VRAM conflicts with other sessions on shared machines.

Usage:
    # Run a specific test:
    python3 scripts/launch-gpu.py -- cargo test -p braidinfer-runtime --test megakernel_test -- --nocapture

    # Run with 1 GPU (default), min 4GB VRAM:
    python3 scripts/launch-gpu.py -g 1 -- cargo test ...

    # Status/cleanup:
    python3 scripts/launch-gpu.py --status
    python3 scripts/launch-gpu.py --cleanup
"""

import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Shared with llama.cpp/scripts/launch-llama.py — same lock dir for cross-project GPU reservation
LOCK_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "launch-llama"
VRAM_IN_USE_THRESHOLD_MB = 100  # Idle GPU has ~27MB used; 512 was too high

# Per-launch BDF audit log: one JSONL record per reserved card per run, so a
# later `ash-pcie events --bdf <BDF>` can be correlated to the exact PID /
# session / command that touched a card. Requested on the agent bridge after a
# GPU page-fault death made card attribution impossible (the prior launches.log
# recorded HIP indices, not BDFs, and had gone stale). Append-only.
BDF_LOG = Path.home() / ".cache" / "launch-gpu" / "bdf-launches.jsonl"

# Process names that open the GPU device for monitoring but do not allocate work.
# A held device with only these holders is treated as available.
MONITORING_COMMS = {"btop", "nvtop", "amdgpu_top", "radeontop", "rocm-smi", "rocprof", "rocminfo"}

# Threshold for "this PID holds the GPU" via /sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>.
KFD_HOLD_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MiB — anything non-trivial


def compute_rocr_visible_devices():
    """Compute ROCR_VISIBLE_DEVICES to make HIP device IDs match PCI bus order.

    HIP enumerates via KFD topology nodes, which may not follow PCI bus order.
    This reads KFD topology to find native HIP->PCI mapping, then returns the
    reordering string so HIP dev 0 = lowest PCI bus, matching rocm-smi.
    Returns None if already in order or on non-ROCm systems.
    Ported from ../llama.cpp/scripts/launch-llama.py.
    """
    topo = Path("/sys/class/kfd/kfd/topology/nodes")
    if not topo.exists():
        return None
    hip_to_pci = {}
    gpu_index = 0
    for node_dir in sorted(topo.iterdir(), key=lambda p: int(p.name)):
        props = node_dir / "properties"
        if not props.exists():
            continue
        simd_count = 0
        location = None
        for line in props.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "simd_count":
                    simd_count = int(parts[1])
                elif parts[0] == "location_id":
                    location = int(parts[1])
        if simd_count > 0 and location is not None:
            bus = (location >> 8) & 0xff
            dev = (location >> 3) & 0x1f
            func = location & 0x7
            pci_bus = f"{bus:02x}:{dev:02x}.{func}"
            hip_to_pci[gpu_index] = pci_bus
            gpu_index += 1
    if not hip_to_pci:
        return None
    sorted_by_pci = sorted(hip_to_pci.items(), key=lambda x: x[1])
    reorder = [str(hip_dev) for hip_dev, _ in sorted_by_pci]
    if reorder == [str(i) for i in range(len(reorder))]:
        return None
    return ",".join(reorder)


# Ensure HIP device IDs match PCI bus order (and thus rocm-smi).
# Must be set before any HIP initialization.
if "ROCR_VISIBLE_DEVICES" not in os.environ:
    _rocr = compute_rocr_visible_devices()
    if _rocr:
        os.environ["ROCR_VISIBLE_DEVICES"] = _rocr
GPU_WAIT_POLL_S = 5
DEFAULT_GPU_WAIT_TIMEOUT_S = 3600
DEFAULT_MIN_VRAM_MB = 4096
KILL_GRACE_S = 5


def is_pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def kill_process(proc, grace_s=KILL_GRACE_S):
    """SIGTERM the process group, wait, SIGKILL, log if D-state.

    Ported from llama.cpp/scripts/launch-llama.py kill_process(). Uses
    killpg so any forked threads/subprocesses are reaped together. Bounded
    wait after SIGKILL so a D-state (uninterruptible-sleep on GPU) process
    doesn't hang the launcher forever — the next invocation's stale-lock
    sweep cleans up leftovers.
    """
    pid = proc.pid

    def _signal(sig):
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, sig)
            except (OSError, ProcessLookupError):
                pass

    _signal(signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.2)
    _signal(signal.SIGKILL)
    # Bounded wait — D-state may hold for a while after SIGKILL until the
    # outstanding GPU kernel call returns to user-space. Don't block forever.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.2)
    print(
        f"WARNING: process {pid} did not exit after SIGKILL "
        f"(likely D-state on GPU; will be reaped on next invocation)",
        file=sys.stderr,
    )


def lock_path(gpu_idx):
    # Use ROCmN naming to match llama.cpp/scripts/launch-llama.py lock files
    return LOCK_DIR / f"gpu-ROCm{gpu_idx}.lock"


def read_lock_info(gpu_idx):
    p = lock_path(gpu_idx)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def try_reserve_gpu(gpu_idx, pid):
    p = lock_path(gpu_idx)
    try:
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = json.dumps({
            "pid": pid,
            "session": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": gpu_idx,
        })
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, info.encode())
        return True, fd
    except (OSError, BlockingIOError):
        existing = read_lock_info(gpu_idx)
        if existing and not is_pid_alive(existing.get("pid", -1)):
            try:
                p.unlink(missing_ok=True)
                return try_reserve_gpu(gpu_idx, pid)
            except OSError:
                pass
        return False, None


def release_gpu(gpu_idx):
    try:
        lock_path(gpu_idx).unlink(missing_ok=True)
    except OSError:
        pass


def get_gpu_vram():
    """Query AMD GPU VRAM via rocm-smi. Returns list of (hip_index, total_mb, used_mb).

    Uses rocm-smi which reports HIP device indices (0-based), matching what
    HIP_VISIBLE_DEVICES expects. DRM card indices may differ from HIP indices.
    """
    gpus = []
    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_data = {}
        for line in result.stdout.splitlines():
            m = re.match(r"GPU\[(\d+)\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)", line)
            if m:
                idx = int(m.group(1))
                gpu_data.setdefault(idx, {})["total"] = int(m.group(2)) // (1024 * 1024)
            m = re.match(r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)", line)
            if m:
                idx = int(m.group(1))
                gpu_data.setdefault(idx, {})["used"] = int(m.group(2)) // (1024 * 1024)
        for idx in sorted(gpu_data.keys()):
            d = gpu_data[idx]
            if "total" in d and "used" in d:
                gpus.append((idx, d["total"], d["used"]))
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        print(f"WARNING: rocm-smi failed ({e}), falling back to sysfs", file=sys.stderr)
        gpus = _get_gpu_vram_sysfs()
    return gpus


def _get_gpu_vram_sysfs():
    """Fallback: query via sysfs. WARNING: indices are DRM card numbers, not HIP indices."""
    gpus = []
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return gpus
    gpu_idx = 0
    for card_dir in sorted(drm.iterdir(), key=lambda p: int(re.search(r"\d+", p.name).group()) if re.search(r"\d+", p.name) else 0):
        if not re.match(r"card\d+$", card_dir.name):
            continue
        vram_total = card_dir / "device" / "mem_info_vram_total"
        vram_used = card_dir / "device" / "mem_info_vram_used"
        if vram_total.exists() and vram_used.exists():
            try:
                total = int(vram_total.read_text().strip()) // (1024 * 1024)
                used = int(vram_used.read_text().strip()) // (1024 * 1024)
                gpus.append((gpu_idx, total, used))
                gpu_idx += 1
            except (ValueError, OSError):
                pass
    return gpus


def _kfd_gpu_id_to_hip_index():
    """Build {kfd_gpu_id: hip_index}. HIP index = PCI-BDF-sorted position among
    nodes with simd_count>0, matching compute_rocr_visible_devices() ordering."""
    topo = Path("/sys/class/kfd/kfd/topology/nodes")
    if not topo.exists():
        return {}
    entries = []  # (location_id, gpu_id)
    for node_dir in sorted(topo.iterdir(), key=lambda p: int(p.name)):
        props = node_dir / "properties"
        gpu_id_file = node_dir / "gpu_id"
        if not props.exists() or not gpu_id_file.exists():
            continue
        simd_count = 0
        location = None
        for line in props.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "simd_count":
                    simd_count = int(parts[1])
                elif parts[0] == "location_id":
                    location = int(parts[1])
        if simd_count > 0 and location is not None:
            try:
                gpu_id = int(gpu_id_file.read_text().strip())
            except (ValueError, OSError):
                continue
            entries.append((location, gpu_id))
    entries.sort(key=lambda x: x[0])
    return {gpu_id: hip_idx for hip_idx, (_, gpu_id) in enumerate(entries)}


def _hip_index_to_bdf():
    """Build {hip_index: "0000:bb:dd.f"} from KFD topology.

    HIP index = PCI-BDF-sorted position among nodes with simd_count>0 (same
    ordering as compute_rocr_visible_devices / _kfd_gpu_id_to_hip_index). The
    topology location_id encodes bus/dev/func; PCI domain is 0000 on this host
    (all cards: see braidinfer CLAUDE.md PCIe card list). Returns {} on
    non-ROCm systems. BDF format matches `ash-pcie events --bdf`.
    """
    topo = Path("/sys/class/kfd/kfd/topology/nodes")
    if not topo.exists():
        return {}
    entries = []  # (location_id, bdf)
    for node_dir in sorted(topo.iterdir(), key=lambda p: int(p.name)):
        props = node_dir / "properties"
        if not props.exists():
            continue
        simd_count = 0
        location = None
        for line in props.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "simd_count":
                    simd_count = int(parts[1])
                elif parts[0] == "location_id":
                    location = int(parts[1])
        if simd_count > 0 and location is not None:
            bus = (location >> 8) & 0xff
            dev = (location >> 3) & 0x1f
            func = location & 0x7
            entries.append((location, f"0000:{bus:02x}:{dev:02x}.{func}"))
    entries.sort(key=lambda x: x[0])
    return {hip_idx: bdf for hip_idx, (_, bdf) in enumerate(entries)}


def log_launch_bdfs(reserved_indices, pid, cmd_args):
    """Append one JSONL record per reserved card to BDF_LOG for wedge auditing.

    Best-effort: never raises into the launch path. Each record carries the
    BDF so `ash-pcie events --bdf <BDF>` can be tied back to this PID/session/
    command after a fault.
    """
    try:
        hip_to_bdf = _hip_index_to_bdf()
        BDF_LOG.parent.mkdir(parents=True, exist_ok=True)
        session = os.environ.get("CLAUDE_SESSION_ID", "unknown")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        cmd = " ".join(cmd_args)
        with open(BDF_LOG, "a") as f:
            for idx in reserved_indices:
                rec = {
                    "timestamp": ts,
                    "hip_index": idx,
                    "bdf": hip_to_bdf.get(idx, "unknown"),
                    "pid": pid,
                    "session": session,
                    "cmd": cmd,
                }
                f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def get_held_gpus():
    """Return set of HIP indices held by any non-monitoring process.

    Reads /sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>. A PID is considered
    a holder of a GPU if its vram_<gpu_id> exceeds KFD_HOLD_THRESHOLD_BYTES.
    PIDs whose /proc/<pid>/comm is in MONITORING_COMMS are skipped.
    Self (this script's pid) is also skipped — we are not yet a holder.
    """
    kfd_proc = Path("/sys/class/kfd/kfd/proc")
    if not kfd_proc.exists():
        return set()
    gpu_id_to_hip = _kfd_gpu_id_to_hip_index()
    if not gpu_id_to_hip:
        return set()
    self_pid = os.getpid()
    held = set()
    for pid_dir in kfd_proc.iterdir():
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        try:
            comm = (Path("/proc") / str(pid) / "comm").read_text().strip()
        except OSError:
            continue
        if comm in MONITORING_COMMS:
            continue
        for vram_file in pid_dir.glob("vram_*"):
            try:
                gpu_id = int(vram_file.name.split("_", 1)[1])
            except ValueError:
                continue
            hip_idx = gpu_id_to_hip.get(gpu_id)
            if hip_idx is None:
                continue
            try:
                used = int(vram_file.read_text().strip())
            except (ValueError, OSError):
                continue
            if used >= KFD_HOLD_THRESHOLD_BYTES:
                held.add(hip_idx)
    return held


def _parse_avoid_gpus():
    """HIP indices to exclude from selection, from BRAIDINFER_AVOID_GPUS
    (comma-separated). Each token is a HIP index (e.g. "1") or a BDF substring
    (e.g. "47:00.0" or "47"). For marking known-degraded cards (e.g. card 47
    pending reboot) so cold-launches don't keep landing on them — see the
    launch-gpu card-selection-bias note (mes-researcher 2026-05-29)."""
    raw = os.environ.get("BRAIDINFER_AVOID_GPUS", "").strip()
    if not raw:
        return set()
    hip_to_bdf = _hip_index_to_bdf()
    avoid = set()
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if tok.isdigit():
            avoid.add(int(tok))
        else:
            for idx, bdf in hip_to_bdf.items():
                if tok in bdf:
                    avoid.add(idx)
    return avoid


def find_free_gpus(count, min_vram_mb):
    gpus = get_gpu_vram()
    held = get_held_gpus()
    avoid = _parse_avoid_gpus()
    candidates = []
    for idx, total, used in gpus:
        if idx in avoid:
            continue
        free = total - used
        if free < min_vram_mb:
            continue
        lock_info = read_lock_info(idx)
        if lock_info and is_pid_alive(lock_info.get("pid", -1)):
            continue
        if idx in held:
            continue
        if used > VRAM_IN_USE_THRESHOLD_MB:
            continue
        candidates.append((idx, total, free))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:count]


def wait_for_gpus(count, min_vram_mb, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        gpus = find_free_gpus(count, min_vram_mb)
        if len(gpus) >= count:
            return gpus
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"ERROR: Timeout waiting for {count} GPUs after {timeout_s}s", file=sys.stderr)
            sys.exit(1)
        print(f"Waiting for GPUs: need {count}, found {len(gpus)}. {remaining:.0f}s remaining...", file=sys.stderr)
        time.sleep(GPU_WAIT_POLL_S)


def _kill_pid(pid, grace_s=KILL_GRACE_S):
    """SIGTERM-then-SIGKILL by PID (and process group if available). For
    do_kill which only has a PID from the lock file, not a Popen handle.
    """
    def _signal(sig):
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, sig)
            except (OSError, ProcessLookupError):
                pass

    _signal(signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.2)
    _signal(signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return
        time.sleep(0.2)
    print(
        f"WARNING: process {pid} did not exit after SIGKILL "
        f"(likely D-state on GPU; will be reaped on next invocation)",
        file=sys.stderr,
    )


def _find_orphans_by_proc_env(current_session):
    """Walk /proc/*/environ for processes whose CLAUDE_SESSION_ID matches
    `current_session`. Returns a list of PIDs. Catches orphans whose
    parent launcher died (so lockfile is already released) but children
    keep running outside any process group we can killpg.
    """
    orphans = []
    my_uid = os.getuid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == os.getpid():
            continue
        try:
            st = p.stat()
            if st.st_uid != my_uid:
                continue
            env_bytes = (p / "environ").read_bytes()
        except (OSError, PermissionError):
            continue
        # /proc/<pid>/environ is NUL-separated KEY=VAL entries.
        for entry in env_bytes.split(b"\0"):
            if entry.startswith(b"CLAUDE_SESSION_ID="):
                val = entry.split(b"=", 1)[1].decode(errors="replace")
                if val == current_session:
                    orphans.append(pid)
                break
    return orphans


def do_kill(session_id=None):
    """Kill processes launched by this session (or a specified session).

    Two-phase:
      1. Lockfile sweep: kill anything tracked by a session-matching lock.
      2. Orphan sweep (/proc): kill any process owned by us with
         CLAUDE_SESSION_ID matching — catches children whose parent
         launcher died and released the flock but child is still alive.
    """
    if not LOCK_DIR.exists():
        print("No lock directory.")
        return
    current_session = session_id or os.environ.get("CLAUDE_SESSION_ID", "unknown")
    killed = 0
    skipped = 0
    killed_pids = set()
    for lp in LOCK_DIR.glob("gpu-*.lock"):
        try:
            info = json.loads(lp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        lock_session = info.get("session", "")
        pid = info.get("pid", -1)
        if lock_session != current_session:
            skipped += 1
            continue
        if is_pid_alive(pid):
            print(f"Killing PID {pid} (GPU {info.get('gpu', '?')}, session {lock_session})")
            _kill_pid(pid)
            killed_pids.add(pid)
            killed += 1
        lp.unlink(missing_ok=True)

    # Phase 2: orphan sweep via /proc/<pid>/environ.
    orphan_killed = 0
    if current_session != "unknown":
        for opid in _find_orphans_by_proc_env(current_session):
            if opid in killed_pids:
                continue  # already targeted
            if is_pid_alive(opid):
                print(f"Killing orphan PID {opid} (session {current_session}, no lockfile)")
                _kill_pid(opid)
                orphan_killed += 1

    summary = f"Killed {killed} process(es)"
    if orphan_killed:
        summary += f" + {orphan_killed} orphan(s)"
    if skipped:
        summary += f". Skipped {skipped} from other sessions"
    print(summary + ".")


def do_cleanup(silent=False):
    if not LOCK_DIR.exists():
        if not silent:
            print("No lock directory.")
        return
    cleaned = 0
    for lp in LOCK_DIR.glob("gpu-*.lock"):
        try:
            info = json.loads(lp.read_text())
            if not is_pid_alive(info.get("pid", -1)):
                lp.unlink()
                cleaned += 1
        except (json.JSONDecodeError, OSError):
            lp.unlink(missing_ok=True)
            cleaned += 1
    if not silent:
        print(f"Removed {cleaned} stale lock(s)." if cleaned else "No stale locks.")


def do_status():
    if not LOCK_DIR.exists():
        print("No lock directory.")
        return
    locks = sorted(LOCK_DIR.glob("gpu-*.lock"))
    if not locks:
        print("No GPU reservations.")
        return
    print(f"{'GPU':<6} {'PID':<8} {'Session':<20} {'Since':<20} {'Alive'}")
    print(f"{'---':<6} {'---':<8} {'-------':<20} {'-----':<20} {'-----'}")
    for lp in locks:
        try:
            info = json.loads(lp.read_text())
            alive = is_pid_alive(info.get("pid", -1))
            print(f"{info.get('gpu', '?'):<6} {info.get('pid', '?'):<8} "
                  f"{info.get('session', '?'):<20} {info.get('timestamp', '?'):<20} "
                  f"{'yes' if alive else 'DEAD'}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"{lp.stem:<6} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU-managed launcher for braidinfer")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--min-vram", type=int, default=DEFAULT_MIN_VRAM_MB, help="Min free VRAM in MiB")
    parser.add_argument("--gpu-timeout", type=int, default=DEFAULT_GPU_WAIT_TIMEOUT_S, help="GPU wait timeout (s)")
    parser.add_argument("--timeout", type=int, default=600, help="Process timeout (s, default: 600)")
    parser.add_argument("--status", "-s", action="store_true")
    parser.add_argument("--cleanup", "-c", action="store_true")
    parser.add_argument("--kill", "-k", action="store_true",
                        help="Kill processes launched by this session (CLAUDE_SESSION_ID)")

    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        our_argv = argv[:split_idx]
        cmd_args = argv[split_idx + 1:]
    else:
        our_argv = argv
        cmd_args = []

    args = parser.parse_args(our_argv)

    if args.status:
        do_status()
        return
    if args.cleanup:
        do_cleanup()
        return
    if args.kill:
        do_kill()
        return

    if not cmd_args:
        parser.error("No command specified. Use: launch-gpu.py [options] -- <command>")

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    do_cleanup(silent=True)

    gpus = wait_for_gpus(args.gpus, args.min_vram, args.gpu_timeout)
    indices = [str(g[0]) for g in gpus]
    vis = ",".join(indices)
    print(f"Reserved GPUs: {vis} (free VRAM: {', '.join(f'{g[2]}MiB' for g in gpus)})", file=sys.stderr)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = vis
    env["CUDA_VISIBLE_DEVICES"] = vis

    # start_new_session: own process group → killpg can reap forked workers
    # cleanly when this script's signal handler / cleanup fires.  Matches
    # llama.cpp/scripts/launch-llama.py convention so all four launchers
    # sharing $XDG_RUNTIME_DIR/launch-llama/ behave consistently.
    proc = subprocess.Popen(cmd_args, env=env, start_new_session=True)

    reserved = []
    lock_fds = {}
    for idx, _, _ in gpus:
        ok, fd = try_reserve_gpu(idx, proc.pid)
        if ok:
            reserved.append(idx)
            lock_fds[idx] = fd

    # Per-launch BDF audit trail (best-effort, never blocks the run).
    log_launch_bdfs(reserved, proc.pid, cmd_args)

    def _release():
        for gpu in reserved:
            release_gpu(gpu)
        for fd in lock_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        lock_fds.clear()

    def _sig(signum, frame):
        kill_process(proc)
        _release()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(_release)

    try:
        proc.wait(timeout=args.timeout)
        sys.exit(proc.returncode)
    except subprocess.TimeoutExpired:
        print(f"\n*** TIMEOUT: process killed after {args.timeout}s (exit 124) ***", file=sys.stderr)
        kill_process(proc)
        _release()
        sys.exit(124)


if __name__ == "__main__":
    main()
