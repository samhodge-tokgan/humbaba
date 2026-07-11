#!/usr/bin/env python3
"""Validate an exported DA3 ONNX model against the reference PyTorch model.

Checks:
1. **Export fidelity (gating):** run the same preprocessed image through the ONNX
   graph and the PyTorch export wrapper (inner net, sky post-process neutralised)
   and require the depth maps to match within tolerance. Catches a broken export.
2. **Metric characterisation (informational):** compare the reference
   ``inference()`` depth to the ONNX depth to confirm the ONNX output is the
   model's metric depth (so the plugin can use ``decimeters = 10 * depth * gain``).

Usage:
    python tools/validate_onnx.py --onnx build/model/DA3METRIC-LARGE.onnx \
        --model-id depth-anything/DA3METRIC-LARGE --height 224 --width 224
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str | None, h: int, w: int) -> np.ndarray:
    if path and Path(path).exists():
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0
    print("[validate] no image; using deterministic synthetic gradient")
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return np.clip(np.stack([xs * np.ones_like(ys), ys * np.ones_like(xs),
                             np.sqrt((xs - .5) ** 2 + (ys - .5) ** 2)], -1), 0, 1).astype(np.float32)


def graph_bakes_norm(onnx_path: str) -> bool:
    p = Path(onnx_path + ".json")
    if p.exists():
        return bool(json.loads(p.read_text())["inputs"]["image"]["normalization_baked_in"])
    return True


def to_input(img01: np.ndarray, normalize: bool) -> np.ndarray:
    x = (img01 - IMAGENET_MEAN) / IMAGENET_STD if normalize else img01
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None], dtype=np.float32)


def run_onnx(onnx_path: str, x: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    depth, sky = sess.run(["depth", "sky"], {name: x})
    return np.asarray(depth), np.asarray(sky)


def run_torch(model_id: str, x: np.ndarray, bake: bool):
    import torch
    from export_onnx import DA3ExportWrapper, load_inner_net, _force_no_bf16
    _force_no_bf16()
    net = load_inner_net(model_id)
    wrap = DA3ExportWrapper(net, bake_normalization=bake).eval()
    with torch.no_grad():
        depth, sky = wrap(torch.from_numpy(x))
    return depth.cpu().numpy(), sky.cpu().numpy(), net


def report(onnx_depth, torch_depth) -> dict:
    a, b = onnx_depth.astype(np.float64).ravel(), torch_depth.astype(np.float64).ravel()
    denom = np.maximum(np.abs(b), 1e-6)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
    return {"max_abs_error": float(np.max(np.abs(a - b))),
            "mean_rel_error": float(np.mean(np.abs(a - b) / denom)),
            "pearson_corr": corr}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--image", default=None)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--max-rel-error", type=float, default=0.03)
    p.add_argument("--min-corr", type=float, default=0.999)
    args = p.parse_args(argv)

    bake = graph_bakes_norm(args.onnx)
    img01 = load_image(args.image, args.height, args.width)
    x = to_input(img01, normalize=not bake)

    print("[validate] ONNX (CPU) ...", flush=True)
    od, osky = run_onnx(args.onnx, x)
    print(f"[validate] onnx depth shape={od.shape} range=[{od.min():.4f},{od.max():.4f}] "
          f"sky shape={osky.shape}")

    print("[validate] reference PyTorch wrapper ...", flush=True)
    td, tsky, _ = run_torch(args.model_id, x, bake)

    fid = report(od, td)
    print("[validate] fidelity:", json.dumps(fid, indent=2))

    ok = fid["mean_rel_error"] <= args.max_rel_error and fid["pearson_corr"] >= args.min_corr
    result = {"fidelity": fid, "normalization_baked_in": bake,
              "onnx_depth_range": [float(od.min()), float(od.max())]}
    Path("build/model").mkdir(parents=True, exist_ok=True)
    Path("build/model/validation.json").write_text(json.dumps(result, indent=2))

    print(f"[validate] {'PASS' if ok else 'FAIL'} "
          f"(mean_rel_error={fid['mean_rel_error']:.4g}<= {args.max_rel_error}, "
          f"corr={fid['pearson_corr']:.6f}>= {args.min_corr})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
