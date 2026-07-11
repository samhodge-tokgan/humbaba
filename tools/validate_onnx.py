#!/usr/bin/env python3
"""Validate an exported DA3 ONNX model against the reference PyTorch model.

Two independent checks:

1. **Export fidelity (gating):** run the same preprocessed image through the ONNX
   graph and through the PyTorch export wrapper, and require the canonical depth
   maps to match within tolerance (this catches a broken export regardless of
   metric scaling). Reported as max-abs error, mean-relative error and Pearson
   correlation; exits non-zero if it regresses past ``--max-rel-error``.

2. **Metric-scale discovery (informational):** compare the reference model's
   ``inference()`` depth to ``focal_px * canonical_depth`` to empirically recover
   the canonical→metric constant (the ``/300`` figure that is unverified in the
   docs). Printed for use by the OFX plugin in M3.

Usage:
    python tools/validate_onnx.py \
        --onnx build/model/DA3METRIC-LARGE.onnx \
        --model-id depth-anything/DA3METRIC-LARGE \
        --image test-assets/marcie.png \
        --height 504 --width 504 --max-rel-error 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str | None, height: int, width: int) -> np.ndarray:
    """Return an [H,W,3] float32 RGB image in [0,1], resized to (height,width)."""
    if path and Path(path).exists():
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0
    # Deterministic synthetic fallback (no external asset needed for a parity check).
    print("[validate] no image provided; using deterministic synthetic gradient")
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    r = np.sqrt(((xs - 0.5) ** 2 + (ys - 0.5) ** 2))
    rgb = np.stack([ys * np.ones_like(xs), xs * np.ones_like(ys), r], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def to_input_tensor(img01: np.ndarray, normalize: bool) -> np.ndarray:
    """[H,W,3] [0,1] → [1,3,H,W]; normalize only if the graph does NOT bake it in."""
    x = img01.copy()
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x, dtype=np.float32)


def graph_bakes_normalization(onnx_path: str) -> bool:
    meta_path = Path(onnx_path + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return bool(meta.get("inputs", {}).get("image", {}).get("normalization_baked_in", True))
    print("[validate] no metadata sidecar; assuming normalization IS baked in")
    return True


def run_onnx(onnx_path: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    depth, intrinsics = sess.run(["depth", "intrinsics"], {in_name: x})
    return np.asarray(depth), np.asarray(intrinsics)


def run_torch_wrapper(model_id: str, x: np.ndarray, bake_norm: bool):
    import torch
    from export_onnx import DA3ExportWrapper, load_model, _force_no_bf16

    _force_no_bf16()
    model = load_model(model_id)
    wrapper = DA3ExportWrapper(model, bake_normalization=bake_norm).eval()
    with torch.no_grad():
        depth, intrinsics = wrapper(torch.from_numpy(x))
    return depth.cpu().numpy(), intrinsics.cpu().numpy(), model


def fidelity_report(onnx_depth, torch_depth) -> dict:
    a = onnx_depth.astype(np.float64).ravel()
    b = torch_depth.astype(np.float64).ravel()
    denom = np.maximum(np.abs(b), 1e-6)
    max_abs = float(np.max(np.abs(a - b)))
    mean_rel = float(np.mean(np.abs(a - b) / denom))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return {"max_abs_error": max_abs, "mean_rel_error": mean_rel, "pearson_corr": corr}


def metric_scale_discovery(model, img01, onnx_depth, onnx_intrinsics) -> dict | None:
    """Compare reference inference() depth to focal*canonical to recover the scale."""
    try:
        import torch  # noqa: F401
        from PIL import Image

        pil = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
        pred = model.inference([pil]) if _accepts_list(model) else model.inference(pil)
        ref_depth = _extract_depth(pred)
        if ref_depth is None:
            return None
        fx = float(onnx_intrinsics[0, 0, 0])
        canonical = onnx_depth[0]
        ref = np.asarray(ref_depth, dtype=np.float64)
        if ref.shape != canonical.shape:
            # reference may be at a different resolution; skip precise fit
            return {"note": "resolution mismatch; skipped", "fx_pixels": fx}
        mask = (ref > 1e-3) & (canonical > 1e-3)
        if mask.sum() < 100:
            return {"note": "insufficient valid pixels", "fx_pixels": fx}
        scale = float(np.median(fx * canonical[mask] / ref[mask]))
        return {"fx_pixels": fx, "recovered_scale_constant": scale,
                "hint": "meters = fx * canonical / scale ; compare to the documented 300"}
    except Exception as e:  # informational only
        return {"error": f"{type(e).__name__}: {e}"}


def _accepts_list(model) -> bool:
    return True


def _extract_depth(pred):
    for attr in ("depth", "depths", "metric_depth"):
        v = getattr(pred, attr, None)
        if v is not None:
            arr = np.asarray(v)
            return arr[0] if arr.ndim == 3 else arr
    if isinstance(pred, dict):
        for k in ("depth", "depths", "metric_depth"):
            if k in pred:
                arr = np.asarray(pred[k])
                return arr[0] if arr.ndim == 3 else arr
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--image", default=None)
    p.add_argument("--height", type=int, default=504)
    p.add_argument("--width", type=int, default=504)
    p.add_argument("--max-rel-error", type=float, default=0.02,
                   help="fail if ONNX-vs-torch mean relative depth error exceeds this")
    p.add_argument("--min-corr", type=float, default=0.999,
                   help="fail if ONNX-vs-torch depth correlation falls below this")
    p.add_argument("--skip-metric-discovery", action="store_true")
    args = p.parse_args(argv)

    # tools/ on path so we can import export_onnx helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    bake_norm = graph_bakes_normalization(args.onnx)
    img01 = load_image(args.image, args.height, args.width)
    x = to_input_tensor(img01, normalize=not bake_norm)

    print("[validate] running ONNX (CPU) ...", flush=True)
    onnx_depth, onnx_intr = run_onnx(args.onnx, x)
    print(f"[validate] onnx depth shape={onnx_depth.shape} intrinsics fx={onnx_intr[0,0,0]:.3f}")

    print("[validate] running reference PyTorch wrapper ...", flush=True)
    torch_depth, torch_intr, model = run_torch_wrapper(args.model_id, x, bake_norm)

    fid = fidelity_report(onnx_depth, torch_depth)
    print("[validate] fidelity:", json.dumps(fid, indent=2))

    result = {"fidelity": fid, "input_normalization_baked_in": bake_norm}

    if not args.skip_metric_discovery:
        scale = metric_scale_discovery(model, img01, onnx_depth, onnx_intr)
        result["metric_scale"] = scale
        print("[validate] metric-scale discovery:", json.dumps(scale, indent=2))

    Path("build/model").mkdir(parents=True, exist_ok=True)
    Path("build/model/validation.json").write_text(json.dumps(result, indent=2))

    ok = fid["mean_rel_error"] <= args.max_rel_error and fid["pearson_corr"] >= args.min_corr
    print(f"[validate] {'PASS' if ok else 'FAIL'} "
          f"(mean_rel_error={fid['mean_rel_error']:.4g} <= {args.max_rel_error}, "
          f"corr={fid['pearson_corr']:.6f} >= {args.min_corr})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
