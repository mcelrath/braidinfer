#!/usr/bin/env python3
import struct
import sys
import argparse


def read_trace(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"BTRC":
            raise ValueError(f"{path}: bad magic {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        checkpoints = []
        while True:
            header = f.read(4)
            if len(header) < 4:
                break
            name_len = struct.unpack("<I", header)[0]
            # Last 4 bytes in file is checkpoint count, not a name_len.
            # We detect end-of-data by trying to read name_len bytes.
            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                break
            num_elements_bytes = f.read(4)
            if len(num_elements_bytes) < 4:
                break
            num_elements = struct.unpack("<I", num_elements_bytes)[0]
            data_bytes = f.read(num_elements * 4)
            if len(data_bytes) < num_elements * 4:
                break
            name = name_bytes.decode("utf-8")
            data = struct.unpack(f"<{num_elements}f", data_bytes)
            checkpoints.append((name, data))
        return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Compare two BTRC activation trace files")
    parser.add_argument("ref", help="Reference trace file")
    parser.add_argument("test", help="Test trace file")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Relative tolerance threshold (default 0.01 = 1%%)")
    args = parser.parse_args()

    ref_checkpoints = read_trace(args.ref)
    test_checkpoints = read_trace(args.test)

    ref_map = {name: data for name, data in ref_checkpoints}
    test_map = {name: data for name, data in test_checkpoints}

    all_names = [name for name, _ in ref_checkpoints]
    diverged = False

    header = f"{'Checkpoint':<30}  {'MaxAbsDiff':>12}  {'MaxRelDiff':>12}  {'Status':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name in all_names:
        if name not in test_map:
            print(f"{name:<30}  {'MISSING':>12}  {'MISSING':>12}  {'MISSING':>8}")
            diverged = True
            continue

        ref_data = ref_map[name]
        test_data = test_map[name]

        if len(ref_data) != len(test_data):
            print(f"{name:<30}  {'SIZE_MISMATCH':>12}  {'SIZE_MISMATCH':>12}  {'FAIL':>8}")
            diverged = True
            continue

        max_abs = 0.0
        max_rel = 0.0
        for r, t in zip(ref_data, test_data):
            abs_diff = abs(r - t)
            if abs_diff > max_abs:
                max_abs = abs_diff
            denom = max(abs(r), abs(t), 1e-8)
            rel_diff = abs_diff / denom
            if rel_diff > max_rel:
                max_rel = rel_diff

        status = "OK" if max_rel <= args.tolerance else "FAIL"
        if status == "FAIL":
            diverged = True
        print(f"{name:<30}  {max_abs:>12.6f}  {max_rel:>12.6f}  {status:>8}")

    for name in test_map:
        if name not in ref_map:
            print(f"{name:<30}  {'EXTRA':>12}  {'EXTRA':>12}  {'WARN':>8}")

    print(sep)
    if diverged:
        print("RESULT: DIVERGED")
        sys.exit(1)
    else:
        print("RESULT: MATCH")
        sys.exit(0)


if __name__ == "__main__":
    main()
