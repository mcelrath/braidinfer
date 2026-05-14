#!/usr/bin/env python3
"""
CI gate: any persistent_worker entry point must use the canonical protocol
helpers from kernels/rdna3/rdna3_persistent_protocol.h.

Specifically, a function declared as a __global__ persistent worker
(takes volatile WorkerQueue*, contains its own outer poll loop) must:
  - call queue->ack = seq OR host_uc_store_agent(&queue->ack, seq) in
    the same iter as the dispatch that processed seq (not deferred to
    a future iter's body top), OR
  - call persistent_iter_ack from the canonical helper.

This is the static counterpart to scripts/check_persistent_protocol.sh
(which does the runtime regression test).

The 2026-05-14 Phase 2' deferred-ack deadlock is the bug this gate prevents.
"""
import re
import sys
from pathlib import Path

KERNELS = Path(__file__).parent.parent / "kernels"

# Files declaring a persistent-worker-style entry: __global__ taking
# any volatile *Queue* pointer (WorkerQueue, MoeWorkerQueue, future variants).
SYMBOL_PAT = re.compile(
    r'__global__\s+[^{]*\b\w+\s*\(\s*volatile\s+\w*Queue\s*\*',
    re.MULTILINE,
)

# Acceptable ack-write patterns. Matches any local queue pointer name.
ACCEPT_PATTERNS = [
    re.compile(r'persistent_iter_ack\s*\('),
    re.compile(r'host_uc_store_agent\s*\(\s*[^)]*\b\w+\s*->\s*ack\s*,\s*seq\b'),
    re.compile(r'\b\w+\s*->\s*ack\s*=\s*seq\b'),
]

# Forbidden patterns (the deferred-ack deadlock).
FORBID_PATTERNS = [
    re.compile(r'host_uc_store_agent\s*\(\s*[^)]*\b\w+\s*->\s*ack\s*,\s*last_seq\b'),
    re.compile(r'\b\w+\s*->\s*ack\s*=\s*last_seq\b'),
]

# Exclude diagnostic-only worker fixtures (not production hot path).
EXCLUDE_DIRS = {"diagnostic"}

failures = []
hip_files = [p for p in KERNELS.rglob("*.hip")
             if not any(part in EXCLUDE_DIRS for part in p.relative_to(KERNELS).parts)]
for hip in sorted(hip_files):
    text = hip.read_text(errors="replace")
    if not SYMBOL_PAT.search(text):
        continue
    accepted = any(p.search(text) for p in ACCEPT_PATTERNS)
    forbidden = [p.pattern for p in FORBID_PATTERNS if p.search(text)]
    if not accepted or forbidden:
        failures.append((hip.name, accepted, forbidden))

if failures:
    print("PERSISTENT-PROTOCOL STATIC CHECK FAILED:", file=sys.stderr)
    for fname, accepted, forbidden in failures:
        if not accepted:
            print(f"  kernels/{fname}: no immediate ack=seq write detected",
                  file=sys.stderr)
        if forbidden:
            print(f"  kernels/{fname}: forbidden ack=last_seq pattern found: "
                  f"{forbidden}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Reference: kb persistent-wedge-fix-2026-05-14; "
          "GFX1100_ARCH.md §11.15; kernels/rdna3/rdna3_persistent_protocol.h",
          file=sys.stderr)
    sys.exit(1)

print("Persistent protocol static check OK.")
sys.exit(0)
