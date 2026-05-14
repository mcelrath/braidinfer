#!/bin/bash
# Run all in-tree correctness gates. Intended for pre-merge / manual
# verification. Static gates run without GPU; runtime gates require
# a free GPU and the cargo build to have produced megakernel.hsaco.
#
# Usage:
#   scripts/check_all.sh              # all gates, runtime included
#   scripts/check_all.sh --static     # static gates only (no GPU)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STATIC_ONLY=0
[[ "${1:-}" == "--static" ]] && STATIC_ONLY=1

FAIL=0
run() {
    local name="$1"; shift
    echo "==> $name"
    if ! "$@"; then
        echo "FAIL: $name" >&2
        FAIL=1
    fi
}

run "watchdog coverage"        python3 scripts/check_watchdog_coverage.py
run "persistent protocol (static)"  python3 scripts/check_persistent_protocol_static.py

if [[ $STATIC_ONLY -eq 0 ]]; then
    run "persistent protocol (runtime)"  scripts/check_persistent_protocol.sh
fi

if [[ $FAIL -eq 0 ]]; then
    echo
    echo "All checks PASS."
    exit 0
fi
exit 1
