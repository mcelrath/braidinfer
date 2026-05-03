#!/usr/bin/env python3
"""
CI gate: verify all persistent HIP kernels include watchdog.h.

A kernel file requires watchdog coverage if it contains:
  - cg::this_grid() (uses cooperative groups — persistent cooperative kernel)
  - while (true) or while(true) (persistent spin loop)

Each such file must also #include "watchdog.h".
"""
import sys
import re
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"

COOPERATIVE_PAT = re.compile(r'\bcg::this_grid\s*\(\s*\)')
PERSISTENT_PAT  = re.compile(r'\bwhile\s*\(\s*true\s*\)')
INCLUDE_PAT     = re.compile(r'#\s*include\s+"watchdog\.h"')

missing = []

for hip_file in sorted(KERNELS_DIR.glob("*.hip")):
    text = hip_file.read_text(errors="replace")
    needs_watchdog = COOPERATIVE_PAT.search(text) or PERSISTENT_PAT.search(text)
    if not needs_watchdog:
        continue
    has_include = INCLUDE_PAT.search(text)
    if not has_include:
        missing.append(hip_file.name)

if missing:
    print("WATCHDOG COVERAGE MISSING in the following kernel files:", file=sys.stderr)
    for f in missing:
        print(f"  kernels/{f}", file=sys.stderr)
    print(
        "\nEach persistent/cooperative kernel must #include \"watchdog.h\" and call\n"
        "watchdog_poll_and_check() + watchdog_beat() in its main loop.",
        file=sys.stderr,
    )
    sys.exit(1)

covered = []
for hip_file in sorted(KERNELS_DIR.glob("*.hip")):
    text = hip_file.read_text(errors="replace")
    if INCLUDE_PAT.search(text):
        covered.append(hip_file.name)

print(f"Watchdog coverage OK: {len(covered)} kernel(s) patched, 0 missing.")
for f in covered:
    print(f"  [OK] {f}")
sys.exit(0)
