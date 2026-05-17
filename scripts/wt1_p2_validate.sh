#!/usr/bin/env bash
# wt1_p2_validate.sh — Validation harness for wt1 P2-{b,c,d,e} on top of fixed P2-a
#
# Usage:
#   wt1_p2_validate.sh <WT1_FIX_HEAD> <p2-variant>
#
#   WT1_FIX_HEAD  — git commit hash at tip of wt1-p2a-fix after sub-agent 1 lands its fix
#   p2-variant    — one of: b, c, d, e
#
# Environment variables (optional overrides):
#   MODEL         — path to model file (default: qwen35_35b_a3b.q4.bqnt)
#   MODEL_MIRROR  — path to mirror-mode model (default: same as MODEL but with MIRROR=1)
#   REPO_ROOT     — project root (default: auto-detected from script location)
#
# Invocation order for full chain: b → d → c → e
# (b first: API foundation; d additive on b; c chunk-seal; e probe migration)
#
# Each invocation:
#   1. Creates a fresh worktree at .worktrees/wt1-validate-p2-<variant>
#   2. Cherry-picks the single P2-<variant> commit onto WT1_FIX_HEAD
#   3. cargo clean + cargo build
#   4. 5-token Paris regression WITHOUT mirror
#   5. 50-token sustained run WITH mirror
#   6. Reports PASS/FAIL, cleans up worktree unconditionally

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <WT1_FIX_HEAD> <b|c|d|e>" >&2
    exit 1
fi

WT1_FIX_HEAD="$1"
VARIANT="$2"

case "$VARIANT" in
    b|c|d|e) ;;
    *) echo "ERROR: p2-variant must be one of: b c d e (got: $VARIANT)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKTREE_PATH="$REPO_ROOT/.worktrees/wt1-validate-p2-$VARIANT"
BRANCH_NAME="wt1-p2-$VARIANT"
VALIDATE_BRANCH="wt1-validate-p2-$VARIANT"

MODEL="${MODEL:-qwen35_35b_a3b.q4.bqnt}"
LAUNCH_GPU="$SCRIPT_DIR/launch-gpu.py"
LOG_DIR="/tmp/wt1-validate-p2-$VARIANT-$$"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Cleanup trap — runs on ANY exit (success, failure, or signal)
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "[cleanup] keeping worktree $WORKTREE_PATH and logs $LOG_DIR for diagnosis (exit $exit_code)"
        exit $exit_code
    fi
    echo "[cleanup] removing worktree $WORKTREE_PATH"
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "$VALIDATE_BRANCH" 2>/dev/null || true
    rm -rf "$LOG_DIR"
    exit $exit_code
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

run_gpu() {
    local label="$1"; shift
    local log="$LOG_DIR/$label.log"
    echo "[run_gpu] $label — logging to $log"
    python3 "$LAUNCH_GPU" --timeout 300 -- "$@" >"$log" 2>&1
    return $?
}

# ---------------------------------------------------------------------------
# Step 1: Verify WT1_FIX_HEAD exists
# ---------------------------------------------------------------------------
echo "=== wt1_p2_validate.sh P2-$VARIANT ==="
echo "WT1_FIX_HEAD: $WT1_FIX_HEAD"

git -C "$REPO_ROOT" cat-file -t "$WT1_FIX_HEAD" >/dev/null 2>&1 \
    || fail "WT1_FIX_HEAD $WT1_FIX_HEAD not found in repo"

# Confirm it is a DESCENDANT of unfixed P2-a base (e5b3c45):
UNFIXED_P2A="e5b3c45eb5969ec90b0742261664a27a4ea2c730"
if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$UNFIXED_P2A" "$WT1_FIX_HEAD"; then
    fail "WT1_FIX_HEAD $WT1_FIX_HEAD is not a descendant of unfixed P2-a base $UNFIXED_P2A — wrong commit?"
fi

# Confirm WT1_FIX_HEAD is strictly NEWER than unfixed base (i.e. fix was actually committed):
if [[ "$WT1_FIX_HEAD" == "$UNFIXED_P2A"* || "$(git -C "$REPO_ROOT" rev-parse $WT1_FIX_HEAD)" == "$UNFIXED_P2A" ]]; then
    fail "WT1_FIX_HEAD equals unfixed P2-a base — sub-agent 1 has not yet committed its fix. Abort."
fi

# ---------------------------------------------------------------------------
# Step 2: Resolve the single P2-X commit to cherry-pick
# ---------------------------------------------------------------------------
P2X_COMMIT="$(git -C "$REPO_ROOT" rev-parse "refs/heads/$BRANCH_NAME")"
echo "P2-$VARIANT commit to cherry-pick: $P2X_COMMIT (from branch $BRANCH_NAME)"

# Verify it is still exactly 1 commit ahead of unfixed base (safety check):
COUNT="$(git -C "$REPO_ROOT" rev-list "${UNFIXED_P2A}..${P2X_COMMIT}" --count)"
if [[ "$COUNT" -ne 1 ]]; then
    fail "$BRANCH_NAME has $COUNT commits ahead of unfixed base (expected 1). Branch may have drifted."
fi

# ---------------------------------------------------------------------------
# Step 3: Create worktree + cherry-pick
# ---------------------------------------------------------------------------
if [[ -d "$WORKTREE_PATH" ]]; then
    echo "[warn] worktree $WORKTREE_PATH already exists — removing first"
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "$VALIDATE_BRANCH" 2>/dev/null || true
fi

echo "[step 3] creating worktree at $WORKTREE_PATH on new branch $VALIDATE_BRANCH"
git -C "$REPO_ROOT" worktree add -b "$VALIDATE_BRANCH" "$WORKTREE_PATH" "$WT1_FIX_HEAD"

echo "[step 3] cherry-picking $P2X_COMMIT"
git -C "$WORKTREE_PATH" cherry-pick --no-gpg-sign "$P2X_COMMIT" \
    || fail "cherry-pick of $P2X_COMMIT onto $WT1_FIX_HEAD failed (likely merge conflict)"

echo "[step 3] worktree HEAD after cherry-pick: $(git -C "$WORKTREE_PATH" rev-parse HEAD)"

# ---------------------------------------------------------------------------
# Step 4: cargo clean + build
# ---------------------------------------------------------------------------
echo "[step 4] cargo clean"
(cd "$WORKTREE_PATH" && cargo clean -p braidinfer-runtime 2>&1) \
    || fail "cargo clean failed"

echo "[step 4] cargo build --release"
(cd "$WORKTREE_PATH" && cargo build --release -p braidinfer-runtime 2>&1) \
    || fail "cargo build failed"

GENERATE="$WORKTREE_PATH/target/release/generate"
[[ -x "$GENERATE" ]] || fail "binary not found at $GENERATE after build"

# ---------------------------------------------------------------------------
# Step 5: 5-token Paris regression WITHOUT mirror
# ---------------------------------------------------------------------------
echo "[step 5] 5-token Paris regression (no mirror)"
SMOKE_RC=0
MODEL="$MODEL" RAW=1 MAX_TOKENS=5 \
    run_gpu "smoke_5tok" "$GENERATE" "The Eiffel Tower is located in Paris" \
    || SMOKE_RC=$?

if [[ $SMOKE_RC -ne 0 ]]; then
    fail "5-token smoke FAILED (exit $SMOKE_RC). Log: $LOG_DIR/smoke_5tok.log"
fi
echo "[step 5] 5-token output:"
cat "$LOG_DIR/smoke_5tok.log" | tail -10

pass "5-token Paris smoke"

# ---------------------------------------------------------------------------
# Step 6: 50-token sustained run WITH mirror
# ---------------------------------------------------------------------------
echo "[step 6] 50-token sustained mirror-on run"
MIRROR_RC=0
MODEL="$MODEL" RAW=1 MAX_TOKENS=50 BRAIDINFER_DECODE_MIRROR=1 \
    run_gpu "mirror_50tok" "$GENERATE" "The Eiffel Tower is located in Paris" \
    || MIRROR_RC=$?

if [[ $MIRROR_RC -ne 0 ]]; then
    fail "50-token mirror run FAILED (exit $MIRROR_RC). Log: $LOG_DIR/mirror_50tok.log"
fi
echo "[step 6] 50-token mirror output (tail):"
cat "$LOG_DIR/mirror_50tok.log" | tail -20

pass "50-token mirror-on sustained run"

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PASS  P2-$VARIANT validated on top of WT1_FIX_HEAD=$WT1_FIX_HEAD"
echo "================================================================"
