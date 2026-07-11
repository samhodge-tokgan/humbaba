#!/usr/bin/env python3
"""Export Depth Anything 3 metric model to ONNX (dynamic resolution).

Produces ONE ONNX graph that runs at ARBITRARY input resolutions (H, W each a
multiple of 14). Input `image` is [1,3,H,W] RGB in [0,1] (ImageNet normalization
baked into the graph). Outputs `depth` [1,H,W] and `sky` [1,H,W].

Why dynamo: the legacy `torch.onnx.export(dynamo=False)` bakes H/W to constants
(the DINOv2 positional embedding and the DPT head), so its graph only runs at the
trace resolution. The TorchDynamo / `torch.export` exporter (`dynamo=True`) tracks
symbolic shapes, so H/W stay dynamic. Two monkeypatches keep shapes symbolic:

  1. DinoVisionTransformer.interpolate_pos_encoding -- drop the `npatch==N and w==h`
     short-circuit and interpolate the pos-embed to a size derived from the dynamic
     input (drops the interpolate_offset float kludge that bakes constants).
  2. DPTHead._forward_impl -- stock `int(ph * patch_size / down_ratio)` specializes
     the symbolic dim to the trace value; use integer floor-division and a direct
     F.interpolate (custom_interpolate has a data-dependent INT_MAX branch).

The metric model predicts no intrinsics in monocular mode; its depth is metric
(the OFX plugin uses decimeters = 10 * depth * gain). The sky post-process
(`_process_mono_sky_estimation`, torch.quantile) is neutralised for export; the
`sky` output lets the plugin handle sky regions.

Requires: torch>=2, onnx, onnxruntime, onnxscript (for the dynamo exporter).
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _da3_compat import ensure_da3_importable  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
REF_VIEW_STRATEGY = "saddle_balanced"


def _force_no_bf16() -> None:
    try:
        torch.cuda.is_bf16_supported = lambda *a, **k: False  # type: ignore[assignment]
    except Exception:
        pass


# ------------------------- monkeypatches (symbolic shapes) --------------------
def _make_dynamic_interpolate_pos_encoding(vit):
    ps = vit.patch_size
    N = vit.pos_embed.shape[1] - 1
    M = int(math.sqrt(N))
    assert M * M == N, f"pos_embed grid not square: N={N}"
    antialias = vit.interpolate_antialias

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        w0 = w // ps
        h0 = h // ps
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(w0, h0), mode="bicubic", antialias=antialias
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1
        ).to(previous_dtype)

    return interpolate_pos_encoding


def _patched_dpt_forward_impl(self, feats, H, W, patch_start_idx):
    B, _, C = feats[0].shape
    ph, pw = H // self.patch_size, W // self.patch_size
    resized_feats = []
    for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
        x = feats[take_idx][:, patch_start_idx:]
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
        x = self.projects[stage_idx](x)
        if self.pos_embed:
            x = self._add_pos_embed(x, W, H)
        x = self.resize_layers[stage_idx](x)
        resized_feats.append(x)
    fused = self._fuse(resized_feats)
    h_out = ph * self.patch_size // self.down_ratio
    w_out = pw * self.patch_size // self.down_ratio
    fused = self.scratch.output_conv1(fused)
    fused = F.interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
    if self.pos_embed:
        fused = self._add_pos_embed(fused, W, H)
    feat = fused
    main_logits = self.scratch.output_conv2(feat)
    outs = {}
    if self.has_conf:
        fmap = main_logits.permute(0, 2, 3, 1)
        pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
        outs[self.head_main] = pred.squeeze(1)
        outs[f"{self.head_main}_conf"] = conf.squeeze(1)
    else:
        outs[self.head_main] = self._apply_activation_single(
            main_logits, self.activation
        ).squeeze(1)
    if self.use_sky_head:
        sky_logits = self.scratch.sky_output_conv2(feat)
        outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)
    return outs


# --------------------------------- wrapper ------------------------------------
class DA3ExportWrapper(nn.Module):
    """[1,3,H,W] RGB in [0,1] -> (depth [1,H,W], sky [1,H,W]); normalization baked in."""

    def __init__(self, inner_net: nn.Module, bake_normalization: bool = True) -> None:
        super().__init__()
        self.net = inner_net
        self.bake_normalization = bake_normalization
        self.register_buffer("norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor):
        if self.bake_normalization:
            image = (image - self.norm_mean) / self.norm_std
        x = image.unsqueeze(1)  # [1, 1, 3, H, W]
        out = self.net(x, None, None, [], False, False, REF_VIEW_STRATEGY)
        return out["depth"].squeeze(1), out["sky"].squeeze(1)


def load_and_patch(model_id: str, dynamic: bool = True) -> nn.Module:
    ensure_da3_importable()
    from depth_anything_3.api import DepthAnything3  # type: ignore

    api = DepthAnything3.from_pretrained(model_id).eval().float().to("cpu")
    net = api.model
    # torch.quantile-based sky post-process does not export; neutralize it.
    net._process_mono_sky_estimation = lambda output: output  # type: ignore[assignment]
    if dynamic:
        vit = net.backbone.pretrained  # DinoVisionTransformer
        vit.interpolate_pos_encoding = types.MethodType(
            _make_dynamic_interpolate_pos_encoding(vit), vit
        )
        net.head._forward_impl = types.MethodType(_patched_dpt_forward_impl, net.head)
    return net


# Kept for validate_onnx.py backward-compat (it imports load_inner_net).
def load_inner_net(model_id: str, dynamic: bool = True) -> nn.Module:
    return load_and_patch(model_id, dynamic=dynamic)


def export(args: argparse.Namespace) -> Path:
    if args.height % PATCH_SIZE or args.width % PATCH_SIZE:
        raise SystemExit(f"height/width must be multiples of {PATCH_SIZE}")

    _force_no_bf16()
    print(f"[export] loading {args.model_id} (CPU, fp32, dynamic={not args.static}) ...", flush=True)
    net = load_and_patch(args.model_id, dynamic=not args.static)
    wrapper = DA3ExportWrapper(net, bake_normalization=args.bake_normalization).eval()

    dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import onnx

    if args.static:
        dynamic_axes = {"image": {0: "batch"}, "depth": {0: "batch"}, "sky": {0: "batch"}}
        print(f"[export] tracing (legacy, fixed {args.height}x{args.width}) ...", flush=True)
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            torch.onnx.export(
                wrapper, (dummy,), str(out_path), export_params=True,
                opset_version=args.opset, do_constant_folding=True,
                input_names=["image"], output_names=["depth", "sky"],
                dynamic_axes=dynamic_axes, dynamo=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
    else:
        from torch.export import Dim

        dyn = {"image": {2: Dim("height", min=PATCH_SIZE, max=4116),
                         3: Dim("width", min=PATCH_SIZE, max=4116)}}
        print(f"[export] exporting (dynamo, dynamic H/W, trace {args.height}x{args.width}) ...", flush=True)
        with torch.no_grad():
            prog = torch.onnx.export(
                wrapper, (dummy,), dynamic_shapes=dyn, dynamo=True,
                input_names=["image"], output_names=["depth", "sky"],
                opset_version=args.opset,
            )
        prog.save(str(out_path))

    onnx.checker.check_model(str(out_path))
    print(f"[export] ONNX graph OK: {out_path}", flush=True)
    _write_metadata(out_path, args)
    return out_path


def _write_metadata(path: Path, args: argparse.Namespace) -> None:
    import json

    meta = {
        "model_id": args.model_id, "precision": "fp32", "opset": args.opset,
        "dynamic_resolution": not args.static,
        "trace_height": args.height, "trace_width": args.width,
        "patch_size": PATCH_SIZE, "base_resolution": 504,
        "inputs": {"image": {"layout": "NCHW", "channel_order": "RGB",
                             "range": "[0,1]" if args.bake_normalization else "imagenet-normalized",
                             "normalization_baked_in": bool(args.bake_normalization),
                             "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
                             "resolution": "any multiple of 14" if not args.static
                                           else f"{args.height}x{args.width} only"}},
        "outputs": {"depth": "metric depth [1,H,W]; decimeters = 10 * depth * gain",
                    "sky": "sky logits/mask [1,H,W]"},
        "intrinsics_exported": False,
    }
    (path.with_suffix(path.suffix + ".json")).write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote metadata: {path}.json", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--height", type=int, default=490, help="trace height (multiple of 14)")
    p.add_argument("--width", type=int, default=490, help="trace width (multiple of 14)")
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--output", default="build/model/DA3METRIC-LARGE.onnx")
    p.add_argument("--static", action="store_true",
                   help="legacy fixed-resolution export (default is dynamic via dynamo)")
    p.add_argument("--bake-normalization", dest="bake_normalization", action="store_true", default=True)
    p.add_argument("--no-bake-normalization", dest="bake_normalization", action="store_false")
    args = p.parse_args(argv)
    print(f"[export] done: {export(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
