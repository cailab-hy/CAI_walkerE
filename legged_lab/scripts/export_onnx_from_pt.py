#!/usr/bin/env python3
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# Convert a TorchScript policy (.pt) to ONNX.

import argparse
import os
import sys

import torch


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("input shape must be comma-separated integers, e.g. 1,750")
    shape = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid dimension '{part}'") from exc
        if dim <= 0:
            raise argparse.ArgumentTypeError("all dimensions must be positive integers")
        shape.append(dim)
    return tuple(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TorchScript policy (.pt) to ONNX.")
    parser.add_argument("--pt", required=True, help="Path to TorchScript policy (.pt)")
    parser.add_argument(
        "--onnx",
        default=None,
        help="Output ONNX path (default: same dir/name as .pt)",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        default=(1, 1050),
        help="Comma-separated input shape (default: 1,1050). Use 1050 for 1D input.",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for export (default: cpu)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Mark batch dimension as dynamic when input is N x D",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pt):
        print(f"[ERROR] Policy file not found: {args.pt}")
        sys.exit(1)

    onnx_path = args.onnx
    if onnx_path is None:
        root, _ = os.path.splitext(args.pt)
        onnx_path = f"{root}.onnx"

    try:
        policy = torch.jit.load(args.pt, map_location=args.device)
    except Exception as exc:
        print(f"[ERROR] Failed to load TorchScript policy: {exc}")
        sys.exit(1)

    policy.eval()
    dummy_input = torch.zeros(args.input_shape, dtype=torch.float32, device=args.device)

    dynamic_axes = None
    if args.dynamic_batch and len(args.input_shape) >= 2:
        dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}}

    torch.onnx.export(
        policy,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[INFO] Saved ONNX to: {onnx_path}")


if __name__ == "__main__":
    main()
