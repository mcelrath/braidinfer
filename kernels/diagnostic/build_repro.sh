#!/usr/bin/env bash
# Build the bpermute reproducer for gfx1100 with --save-temps so we can
# inspect the emitted ds_bpermute_b32 instructions in the .s file.
set -euo pipefail

cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1100}
OUT=${OUT:-bpermute_repro}

# Clean prior intermediates so save-temps is unambiguous.
rm -f *.s *.bc *.ll *.o *.hsaco "$OUT"

# Build a host executable (so we can run end-to-end on a real machine)
# AND keep the device .s emitted by --save-temps.
$HIPCC \
    --offload-arch=$ARCH \
    -O3 -std=c++17 \
    -ffp-contract=fast \
    -mwavefrontsize64 \
    --save-temps \
    -o "$OUT" \
    bpermute_repro.hip

ASM=$(ls *amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [[ -z "$ASM" ]]; then
    echo "No device .s found. --save-temps may have failed." >&2
    exit 1
fi
echo
echo "Device asm: $ASM"
echo
# Use the system grep explicitly: some distros alias grep -> ugrep,
# which does not support backreferences (\1) needed for the dst==src match.
GREP=$(command -v /usr/bin/grep || command -v /bin/grep || echo grep)

echo "All ds_bpermute_b32 instructions emitted:"
$GREP -nE 'ds_bpermute_b32' "$ASM" || echo "(none found)"
echo
echo "Same-VGPR (dst == src) ds_bpermute_b32 instructions:"
# Pattern: ds_bpermute_b32 vN, vM, vN  (first and third operand identical)
$GREP -nE 'ds_bpermute_b32[[:space:]]+v([0-9]+),[[:space:]]*v[0-9]+,[[:space:]]*v\1\b' "$ASM" \
    || echo "(none found -- bug pattern NOT triggered)"
