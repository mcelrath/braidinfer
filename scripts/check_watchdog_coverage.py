#!/usr/bin/env python3
"""
CI gate: verify all persistent HIP kernels include watchdog.h AND call
watchdog_poll_and_check(). Both are required for a kernel to be considered covered.

A kernel file requires watchdog coverage if it contains:
  - cg::this_grid() (uses cooperative groups — persistent cooperative kernel)
  - while (true) or while(true) (persistent spin loop)

Each such file must:
  1. #include "watchdog.h" (or path variant)
  2. Call watchdog_poll_and_check (at least one call site)
"""
import sys
import re
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"

COOPERATIVE_PAT  = re.compile(r'\bcg::this_grid\s*\(\s*\)')
PERSISTENT_PAT   = re.compile(r'\bwhile\s*\(\s*true\s*\)')
INCLUDE_PAT      = re.compile(r'#\s*include\s+[<"](?:[^"<>]*/)?watchdog\.h[">]')
POLL_CHECK_PAT   = re.compile(r'\bwatchdog_poll_and_check\b')

failures = []

for hip_file in sorted(KERNELS_DIR.glob("*.hip")):
    text = hip_file.read_text(errors="replace")
    needs_watchdog = COOPERATIVE_PAT.search(text) or PERSISTENT_PAT.search(text)
    if not needs_watchdog:
        continue
    has_include    = bool(INCLUDE_PAT.search(text))
    has_poll_check = bool(POLL_CHECK_PAT.search(text))
    if not has_include or not has_poll_check:
        missing_parts = []
        if not has_include:
            missing_parts.append('#include "watchdog.h"')
        if not has_poll_check:
            missing_parts.append('watchdog_poll_and_check() call')
        failures.append((hip_file.name, missing_parts))

if failures:
    print("WATCHDOG COVERAGE MISSING:", file=sys.stderr)
    for fname, parts in failures:
        print(f"  kernels/{fname}: missing {', '.join(parts)}", file=sys.stderr)
    print(
        "\nEach persistent/cooperative kernel must #include \"watchdog.h\" AND call\n"
        "watchdog_poll_and_check() + watchdog_beat() in its main loop.",
        file=sys.stderr,
    )
    sys.exit(1)

covered = []
for hip_file in sorted(KERNELS_DIR.glob("*.hip")):
    text = hip_file.read_text(errors="replace")
    if INCLUDE_PAT.search(text) and POLL_CHECK_PAT.search(text):
        covered.append(hip_file.name)

print(f"Watchdog coverage OK: {len(covered)} kernel(s) patched, 0 missing.")
for f in covered:
    print(f"  [OK] {f}")
sys.exit(0)
