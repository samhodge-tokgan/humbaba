#!/usr/bin/env python3
"""Export Depth Anything 3 metric model to ONNX (depth + predicted intrinsics).

Loads `depth-anything/DA3METRIC-LARGE`, wraps it so the graph returns both the
canonical metric depth map and the model's predicted camera intrinsics, exports
an FP32 ONNX graph on CPU (avoiding the model's internal bfloat16 autocast, which
ONNX cannot represent), then optionally converts weights to FP16.

Preprocessing note: DA3 does NOT normalize internally. Callers must feed an
ImageNet-normalized RGB tensor. By default we bake the ImageNet normalization
into the exported graph (``--bake-normalization``) so the OFX plugin can hand the
model a plain [0,1] RGB tensor; pass ``--no-bake-normalization`` to require the
caller to normalize.

Usage:
    python tools/export_onnx.py \
        --model-id depth-anything/DA3METRIC-LARGE \
        --height 504 --width 504 \
        --output build/model/DA3METRIC-LARGE.onnx --fp16

See docs/DEVELOPMENT.md and tools/README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ImageNet normalization constants (RGB), verified from DA3 input_processor.py.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14


def _force_no_bf16() -> None:
    """Make DA3's autocast dtype selection pick fp16 instead of bf16.

    api.py chooses ``bfloat16 if torch.cuda.is_bf16_supported() else float16``.
    We export on CPU with autocast disabled anyway, but this guards against any
    code path that inspects bf16 support during tracing.
    """
    try:
        torch.cuda.is_bf16_supported = lambda *a, **k: False  # type: ignore[assignment]
    except Exception:
        pass


class DA3ExportWrapper(nn.Module):
    """Adapt DA3's [B,N,3,H,W] multi-view forward to a single-image ONNX graph.

    Input:  image [B, 3, H, W]
    Output: depth [B, H, W]        (canonical metric depth; scale by focal px)
            intrinsics [B, 3, 3]   (predicted; fx = intrinsics[:,0,0])
    """

    def __init__(self, api_model: nn.Module, bake_normalization: bool) -> None:
        super().__init__()
        self.model = api_model
        self.bake_normalization = bake_normalization
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # Registered as buffers so they export as graph constants and follow dtype casts.
        self.register_buffer("norm_mean", mean, persistent=False)
        self.register_buffer("norm_std", std, persistent=False)

    def forward(self, image: torch.Tensor):
        if self.bake_normalization:
            image = (image - self.norm_mean) / self.norm_std
        x = image.unsqueeze(1)  # [B, 1, 3, H, W]
        out = self.model(
            x,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )
        depth = out["depth"].squeeze(1)  # [B, H, W]
        intrinsics = out["intrinsics"].squeeze(1)  # [B, 3, 3]
        return depth, intrinsics


def load_model(model_id: str) -> nn.Module:
    from depth_anything_3.api import DepthAnything3  # type: ignore

    model = DepthAnything3.from_pretrained(model_id)
    model = model.eval().float().to("cpu")
    return model


def export(args: argparse.Namespace) -> Path:
    if args.height % PATCH_SIZE or args.width % PATCH_SIZE:
        raise SystemExit(
            f"height/width must be multiples of {PATCH_SIZE} "
            f"(got {args.height}x{args.width})"
        )

    _force_no_bf16()
    print(f"[export] loading {args.model_id} (CPU, fp32) ...", flush=True)
    model = load_model(args.model_id)
    wrapper = DA3ExportWrapper(model, bake_normalization=args.bake_normalization).eval()

    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out_path.with_suffix(".fp32.onnx") if args.fp16 else out_path

    dynamic_axes = {
        "image": {0: "batch", 2: "height", 3: "width"},
        "depth": {0: "batch", 1: "height", 2: "width"},
        "intrinsics": {0: "batch"},
    }

    print(f"[export] tracing → {fp32_path} (opset {args.opset}) ...", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        torch.onnx.export(
            wrapper,
            dummy,
            str(fp32_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth", "intrinsics"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    import onnx

    onnx.checker.check_model(str(fp32_path))
    print(f"[export] fp32 graph OK: {fp32_path}", flush=True)

    if not args.fp16:
        _write_metadata(fp32_path, args, precision="fp32")
        return fp32_path

    print("[export] converting fp32 → fp16 ...", flush=True)
    from onnxconverter_common import float16  # type: ignore

    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(out_path))
    onnx.checker.check_model(str(out_path))
    print(f"[export] fp16 model written: {out_path}", flush=True)
    if not args.keep_fp32:
        fp32_path.unlink(missing_ok=True)
    _write_metadata(out_path, args, precision="fp16")
    return out_path


def _write_metadata(path: Path, args: argparse.Namespace, precision: str) -> None:
    """Record how the model expects to be fed, next to the .onnx file."""
    import json

    meta = {
        "model_id": args.model_id,
        "precision": precision,
        "opset": args.opset,
        "export_height": args.height,
        "export_width": args.width,
        "patch_size": PATCH_SIZE,
        "base_resolution": 504,
        "inputs": {
            "image": {
                "layout": "NCHW",
                "channel_order": "RGB",
                "range": "[0,1]" if args.bake_normalization else "imagenet-normalized",
                "normalization_baked_in": bool(args.bake_normalization),
                "imagenet_mean": IMAGENET_MEAN,
                "imagenet_std": IMAGENET_STD,
            }
        },
        "outputs": {
            "depth": "canonical metric depth [B,H,W]; meters = focal_px * depth / SCALE",
            "intrinsics": "[B,3,3]; fx=intrinsics[0,0], fy=intrinsics[1,1]",
        },
        "metric_scale_constant": "UNVERIFIED — determined by validate_onnx.py",
    }
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote metadata: {meta_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--height", type=int, default=504)
    p.add_argument("--width", type=int, default=504)
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--output", default="build/model/DA3METRIC-LARGE.onnx")
    p.add_argument("--fp16", action="store_true", help="also convert to fp16 (default output)")
    p.add_argument("--keep-fp32", action="store_true", help="keep the intermediate fp32 graph")
    p.add_argument(
        "--bake-normalization",
        dest="bake_normalization",
        action="store_true",
        default=True,
        help="bake ImageNet normalization into the graph (default; plugin feeds [0,1])",
    )
    p.add_argument(
        "--no-bake-normalization",
        dest="bake_normalization",
        action="store_false",
        help="require the caller to feed an ImageNet-normalized tensor",
    )
    args = p.parse_args(argv)
    path = export(args)
    print(f"[export] done: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
