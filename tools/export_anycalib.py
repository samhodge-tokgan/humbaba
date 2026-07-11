#!/usr/bin/env python3
"""Export the AnyCalib network (DINOv2 field predictor) to a dynamic-resolution ONNX.

AnyCalib (github.com/javrtg/AnyCalib, Apache-2.0) predicts a dense per-pixel ray
field; the camera fit (DLT + Gauss-Newton) is done host-side in the OFX plugin
(src/AnyCalibEngine.cpp). This exports ONLY the network (backbone -> decoder ->
head -> rays), before the calibrator.

Like DA3, the DINOv2 positional embedding must be made symbolic for a dynamic
export (drop the `npatch==N` short-circuit and the `interpolate_offset` kludge),
otherwise the graph only runs at the trace resolution.

Requires: torch>=2, onnx, onnxruntime, onnxscript, and AnyCalib installed.
"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")  # use plain (exportable) attention

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _dynamic_interp(vit):
    ps = vit.patch_size
    N = vit.pos_embed.shape[1] - 1
    M = int(math.sqrt(N))
    aa = vit.interpolate_antialias

    def interpolate_pos_encoding(self, x, w, h):
        prev = x.dtype
        pe = self.pos_embed.float()
        cls = pe[:, 0]
        patch = pe[:, 1:]
        dim = pe.shape[-1]
        w0, h0 = w // ps, h // ps
        patch = patch.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(w0, h0), mode="bicubic", antialias=aa)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls.unsqueeze(0), patch), dim=1).to(prev)

    return interpolate_pos_encoding


class Net(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        out = self.m.head(self.m.decoder(self.m.backbone(image)))
        b, _, h, w = image.shape
        return out["rays"].permute(0, 2, 3, 1).reshape(b, h * w, 3)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="anycalib_dist")
    p.add_argument("--output", default="build/model/anycalib_dist.onnx")
    p.add_argument("--trace-h", type=int, default=238)
    p.add_argument("--trace-w", type=int, default=420)
    p.add_argument("--opset", type=int, default=18)
    a = p.parse_args(argv)

    from anycalib import AnyCalib

    print(f"[export] loading {a.model_id} ...", flush=True)
    m = AnyCalib(model_id=a.model_id).eval()
    vit = m.backbone.model
    vit.interpolate_pos_encoding = types.MethodType(_dynamic_interp(vit), vit)

    net = Net(m).eval()
    from torch.export import Dim

    dummy = torch.rand(1, 3, a.trace_h, a.trace_w)
    dyn = {"image": {2: Dim("h", min=14, max=4116), 3: Dim("w", min=14, max=4116)}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    print("[export] exporting (dynamo, dynamic H/W) ...", flush=True)
    prog = torch.onnx.export(
        net, (dummy,), dynamic_shapes=dyn, dynamo=True,
        input_names=["image"], output_names=["rays"], opset_version=a.opset,
    )
    prog.save(a.output)

    import numpy as np
    import onnxruntime as ort

    s = ort.InferenceSession(a.output, providers=["CPUExecutionProvider"])
    for (H, W) in [(a.trace_h, a.trace_w), (322, 322)]:
        r = s.run(None, {"image": np.random.rand(1, 3, H, W).astype(np.float32)})[0]
        ok = r.shape == (1, H * W, 3) and abs(np.linalg.norm(r, axis=-1).mean() - 1.0) < 0.05
        print(f"[verify] {H}x{W}: rays {r.shape} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[export] done: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
