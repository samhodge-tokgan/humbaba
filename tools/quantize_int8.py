#!/usr/bin/env python3
"""Experimental INT8 dynamic quantization of the DA3 ONNX model.

FP16 is the shipped default (see export_onnx.py). INT8 is an experiment gated on
quality: there is no published INT8 quality data for DA3 metric, so this must be
validated (validate_onnx.py against the FP32/FP16 reference) before use. Dynamic
quantization is weight-only and typically gives limited speedup for transformer
attention on GPU/ANE, so treat any INT8 artifact as opt-in.

Usage:
    python tools/quantize_int8.py --onnx build/model/DA3METRIC-LARGE.fp32.onnx \
        --output build/model/DA3METRIC-LARGE.int8.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, help="FP32 ONNX input (not FP16)")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[quantize] dynamic INT8: {args.onnx} -> {out}", flush=True)
    quantize_dynamic(
        model_input=args.onnx,
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    print("[quantize] done. Validate quality with tools/validate_onnx.py before shipping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
