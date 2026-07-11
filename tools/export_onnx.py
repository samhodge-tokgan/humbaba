#!/usr/bin/env python3
"""Export Depth Anything 3 metric model to ONNX (depth + sky).

Loads `depth-anything/DA3METRIC-LARGE` and exports its inner network to ONNX. We
call the inner net (`DepthAnything3Net`) directly rather than the high-level
`forward`, because the latter wraps everything in a hard-coded bfloat16 autocast
that ONNX cannot represent. Exporting on CPU in float32 avoids bf16 entirely; the
weights are then optionally converted to float16.

Outputs:
  * ``depth``  [B, H, W]  — the model's (metric) depth, matching ``inference()``
  * ``sky``    [B, H, W]  — sky logits/mask (useful for masking infinite depth)

Note: DA3METRIC-LARGE does NOT predict camera intrinsics in monocular mode
(``inference().intrinsics`` is None), so intrinsics are not exported. The
canonical->metric relationship is characterised by tools/validate_onnx.py.

Preprocessing: DA3 does NOT normalize internally. By default we bake ImageNet
normalization into the graph (``--bake-normalization``) so the OFX plugin can feed
a plain [0,1] RGB tensor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must run before importing depth_anything_3.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _da3_compat import ensure_da3_importable  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
REF_VIEW_STRATEGY = "saddle_balanced"


def _force_no_bf16() -> None:
    try:
        torch.cuda.is_bf16_supported = lambda *a, **k: False  # type: ignore[assignment]
    except Exception:
        pass


class DA3ExportWrapper(nn.Module):
    """Single-image [B,3,H,W] -> (depth [B,H,W], sky [B,H,W]) over the inner net."""

    def __init__(self, inner_net: nn.Module, bake_normalization: bool) -> None:
        super().__init__()
        self.net = inner_net
        self.bake_normalization = bake_normalization
        self.register_buffer("norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor):
        if self.bake_normalization:
            image = (image - self.norm_mean) / self.norm_std
        x = image.unsqueeze(1)  # [B, 1, 3, H, W]
        out = self.net(x, None, None, [], False, False, REF_VIEW_STRATEGY)
        depth = out["depth"].squeeze(1)  # [B, H, W]
        sky = out["sky"].squeeze(1)      # [B, H, W]
        return depth, sky


def load_inner_net(model_id: str) -> nn.Module:
    ensure_da3_importable()
    from depth_anything_3.api import DepthAnything3  # type: ignore

    api = DepthAnything3.from_pretrained(model_id).eval().float().to("cpu")
    net = api.model  # the DepthAnything3Net

    # Neutralise the mono sky post-process: it clamps sky-region depth to the 99th
    # percentile of non-sky depth using torch.quantile + boolean-mask indexing +
    # randint, none of which export to ONNX. We export the raw depth and the `sky`
    # output; the OFX plugin performs any sky handling itself.
    net._process_mono_sky_estimation = lambda output: output  # type: ignore[assignment]
    return net


def export(args: argparse.Namespace) -> Path:
    if args.height % PATCH_SIZE or args.width % PATCH_SIZE:
        raise SystemExit(f"height/width must be multiples of {PATCH_SIZE}")

    _force_no_bf16()
    print(f"[export] loading {args.model_id} inner net (CPU, fp32) ...", flush=True)
    net = load_inner_net(args.model_id)
    wrapper = DA3ExportWrapper(net, bake_normalization=args.bake_normalization).eval()

    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out_path.with_suffix(".fp32.onnx") if args.fp16 else out_path

    dynamic_axes = {
        "image": {0: "batch", 2: "height", 3: "width"},
        "depth": {0: "batch", 1: "height", 2: "width"},
        "sky": {0: "batch", 1: "height", 2: "width"},
    }

    print(f"[export] tracing -> {fp32_path} (opset {args.opset}) ...", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        torch.onnx.export(
            wrapper, (dummy,), str(fp32_path),
            export_params=True, opset_version=args.opset, do_constant_folding=True,
            input_names=["image"], output_names=["depth", "sky"],
            dynamic_axes=dynamic_axes, dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    import onnx

    onnx.checker.check_model(str(fp32_path))
    print(f"[export] fp32 graph OK: {fp32_path}", flush=True)

    if not args.fp16:
        _write_metadata(fp32_path, args, "fp32")
        return fp32_path

    print("[export] converting fp32 -> fp16 ...", flush=True)
    from onnxconverter_common import float16  # type: ignore

    m = onnx.load(str(fp32_path))
    # Block Cast (and shape/resize ops) from fp16 conversion: converting an explicit
    # aten Cast-to-float leaves an output/type mismatch that ONNX Runtime rejects.
    m16 = float16.convert_float_to_float16(
        m, keep_io_types=True, op_block_list=["Cast", "Range", "Resize", "Upsample"]
    )
    onnx.save(m16, str(out_path))
    onnx.checker.check_model(str(out_path))

    if _onnx_loads(out_path):
        print(f"[export] fp16 model written and loads in ORT: {out_path}", flush=True)
        if not args.keep_fp32:
            fp32_path.unlink(missing_ok=True)
        _write_metadata(out_path, args, "fp16")
        return out_path

    # Fallback: fp16 graph is not runnable — ship fp32 (CoreML EP still runs it in
    # fp16 on-device). Keep the working model at the requested output path.
    print("[export] WARNING: fp16 graph failed to load; falling back to fp32", flush=True)
    import shutil
    shutil.copyfile(fp32_path, out_path)
    if not args.keep_fp32:
        fp32_path.unlink(missing_ok=True)
    _write_metadata(out_path, args, "fp32-fallback")
    return out_path


def _onnx_loads(path: Path) -> bool:
    try:
        import onnxruntime as ort
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[export] fp16 load check failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return False


def _write_metadata(path: Path, args: argparse.Namespace, precision: str) -> None:
    import json

    meta = {
        "model_id": args.model_id, "precision": precision, "opset": args.opset,
        "export_height": args.height, "export_width": args.width,
        "patch_size": PATCH_SIZE, "base_resolution": 504,
        "inputs": {
            "image": {
                "layout": "NCHW", "channel_order": "RGB",
                "range": "[0,1]" if args.bake_normalization else "imagenet-normalized",
                "normalization_baked_in": bool(args.bake_normalization),
                "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
            }
        },
        "outputs": {
            "depth": "metric depth [B,H,W] (matches inference().depth); decimeters = 10 * depth * gain",
            "sky": "sky logits/mask [B,H,W]",
        },
        "intrinsics_exported": False,
        "notes": "DA3METRIC-LARGE predicts no intrinsics in monocular mode; "
                 "metric relationship verified by validate_onnx.py.",
    }
    (path.with_suffix(path.suffix + ".json")).write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote metadata: {path}.json", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--height", type=int, default=504)
    p.add_argument("--width", type=int, default=504)
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--output", default="build/model/DA3METRIC-LARGE.onnx")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep-fp32", action="store_true")
    p.add_argument("--bake-normalization", dest="bake_normalization", action="store_true", default=True)
    p.add_argument("--no-bake-normalization", dest="bake_normalization", action="store_false")
    args = p.parse_args(argv)
    print(f"[export] done: {export(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
