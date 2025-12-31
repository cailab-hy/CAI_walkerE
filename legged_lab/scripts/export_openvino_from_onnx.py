#!/usr/bin/env python3
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# Convert an ONNX policy to OpenVINO IR (.xml/.bin).

import argparse
import os
import sys


def _parse_shape(value: str) -> tuple[int, ...]:
    if value is None:
        return ()
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("input shape must be comma-separated integers, e.g. 1,1050")
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


def _resolve_output_xml(onnx_path: str, output_xml: str | None, output_dir: str | None) -> str:
    if output_xml:
        return output_xml
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(onnx_path))
    return os.path.join(out_dir, f"{base}.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenVINO IR from ONNX.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output-xml", default=None, help="Output XML path (default: same dir/name as ONNX)")
    parser.add_argument("--output-dir", default=None, help="Output directory (ignored if --output-xml set)")
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        default=None,
        help="Comma-separated input shape override (e.g. 1,1050).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Try to compress weights to FP16 when saving.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX file not found: {args.onnx}")
        sys.exit(1)

    output_xml = _resolve_output_xml(args.onnx, args.output_xml, args.output_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output_xml)), exist_ok=True)

    try:
        import openvino as ov
    except ImportError:
        print("[ERROR] openvino is not installed. Try: pip install openvino")
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(args.onnx)

    if args.input_shape:
        try:
            model.reshape({model.input(0): args.input_shape})
        except Exception:
            model.reshape(args.input_shape)

    try:
        ov.save_model(model, output_xml, compress_to_fp16=args.fp16)
    except TypeError:
        if args.fp16:
            print("[WARN] compress_to_fp16 not supported in this OpenVINO version; saving FP32.")
        ov.save_model(model, output_xml)

    output_bin = os.path.splitext(output_xml)[0] + ".bin"
    print(f"[INFO] Saved OpenVINO IR to: {output_xml}")
    print(f"[INFO] Saved OpenVINO weights to: {output_bin}")


if __name__ == "__main__":
    main()
