# Model → ONNX tooling

Scripts that turn the HuggingFace `depth-anything/DA3METRIC-LARGE` checkpoint into
the ONNX model the OFX plugin runs. Run on **CPU/Linux** (CI) or locally.

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | Load DA3METRIC-LARGE, export its inner net to ONNX emitting **`depth` + `sky`**, FP32 (CPU, no bf16) then FP16 (with an FP32 fallback if the FP16 graph won't load). |
| `validate_onnx.py` | Gate: ONNX-vs-PyTorch depth parity. |
| `quantize_int8.py` | Experimental INT8 (opt-in, must pass validation). |
| `_da3_compat.py` | Stubs DA3's heavy optional deps so it imports for a CPU export. |

## Local setup

DA3 must be installed **without its own deps** (a full install builds `xformers`,
which has no macOS-arm64 wheel):

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /tmp/da3
pip install "torch>=2" torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r tools/requirements.txt
pip install -e /tmp/da3 --no-deps      # deps come from requirements.txt; stubs from _da3_compat
```

## Run

```bash
python tools/export_onnx.py --height 504 --width 504 \
    --output build/model/DA3METRIC-LARGE.onnx --fp16 --keep-fp32
python tools/validate_onnx.py --onnx build/model/DA3METRIC-LARGE.onnx --height 224 --width 224
```

## Key facts (verified against the real model)

- **Outputs:** the inner net returns `depth [B,1,H,W]` and `sky [B,1,H,W]` (we squeeze to
  `[B,H,W]`). It does **not** return camera intrinsics — `inference().intrinsics` is `None` for
  the metric model in monocular mode, so predicted-intrinsics scaling is not available; the OFX
  plugin uses a scale/gain (and optional manual focal) instead. `decimeters = 10 * depth * gain`.
- **Sky post-process removed for export:** `DepthAnything3Net._process_mono_sky_estimation`
  (which clamps sky depth via `torch.quantile` + boolean-mask indexing + `randint`) does not
  export to ONNX. It is neutralised; the plugin handles sky via the exported `sky` output.
- **bf16:** the model's high-level `forward` forces a bf16/fp16 autocast, so we call the inner net
  directly on CPU in fp32 and convert to fp16 afterwards.
- **Preprocessing:** DA3 does not normalize internally; the exporter bakes ImageNet normalization
  into the graph by default so the plugin feeds a plain `[0,1]` RGB tensor. The `.json` sidecar
  records the convention.
- Model weights are never committed; CI produces the `.onnx` as a build artifact.
