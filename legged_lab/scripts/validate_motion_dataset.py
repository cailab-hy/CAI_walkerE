#!/usr/bin/env python3
# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import json
import math
import os
from typing import Iterable

from rsl_rl.utils.motion_loader import AMPLoader


def iter_txt_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".txt"):
                yield os.path.join(dirpath, name)


def validate_file(path: str, expected_dim: int) -> list[str]:
    errors: list[str] = []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return [f"Failed to load JSON: {exc}"]

    for key in ("Frames", "FrameDuration", "MotionWeight"):
        if key not in data:
            errors.append(f"Missing key: {key}")

    frames = data.get("Frames", [])
    if not isinstance(frames, list) or not frames:
        errors.append("Frames must be a non-empty list")
        return errors

    frame_len = len(frames[0])
    if frame_len != expected_dim:
        errors.append(f"Unexpected frame length: {frame_len} (expected {expected_dim})")

    for idx, frame in enumerate(frames):
        if len(frame) != frame_len:
            errors.append(f"Frame length mismatch at index {idx}: {len(frame)} != {frame_len}")
            break
        for val in frame:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                errors.append(f"Non-finite value at frame {idx}")
                return errors

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AMP motion datasets.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="legged_lab/envs/tienkung/datasets",
        help="Root directory containing motion .txt files.",
    )
    parser.add_argument(
        "--expected_dim",
        type=int,
        default=AMPLoader.END_POS_END_IDX,
        help="Expected frame length. Defaults to AMPLoader.END_POS_END_IDX.",
    )
    args = parser.parse_args()

    files = sorted(iter_txt_files(args.dataset_dir))
    if not files:
        print(f"[ERROR] No .txt files found under: {args.dataset_dir}")
        return 1

    total_errors = 0
    for path in files:
        errors = validate_file(path, args.expected_dim)
        if errors:
            total_errors += 1
            print(f"[FAIL] {path}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"[OK] {path}")

    if total_errors:
        print(f"[SUMMARY] {total_errors} file(s) failed validation.")
        return 1

    print(f"[SUMMARY] All {len(files)} file(s) passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
