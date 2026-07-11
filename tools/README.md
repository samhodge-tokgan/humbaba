# Model → ONNX tooling

Scripts that turn the HuggingFace `depth-anything/DA3METRIC-LARGE` checkpoint into
the ONNX model the OFX plugin runs. These run on **CPU/Linux** (in CI) or locally.

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | Load DA3METRIC-LARGE, wrap to emit **depth + intrinsics**, export FP32 (CPU, no bf16) then convert to FP16. Writes a `.json` sidecar describing expected input/output. |
| `validate_onnx.py` | Gate: ONNX-vs-PyTorch canonical-depth parity. Also empirically recovers the canonical→metric scale constant. |
| `quantize_int8.py` | Experimental INT8 (opt-in, must pass validation). |

## Local setup

DA3 is not on PyPI as a ready wheel with the model code wired up; install from source:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /tmp/da3
pip install "torch>=2" torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e /tmp/da3          # pulls xformers, einops, safetensors, huggingface_hub, ...
pip install -r tools/requirements.txt
```

`numpy<2` is required by DA3. On a CPU-only host, xformers' fused attention is
unavailable and the DINOv2 backbone should fall back to PyTorch SDPA — if import
or tracing fails there, that's the first thing to check.

## Run

```bash
# Export FP16 (default). Height/width must be multiples of 14; 504 is the base res.
python tools/export_onnx.py --height 504 --width 504 \
    --output build/model/DA3METRIC-LARGE.onnx --fp16 --keep-fp32

# Validate export fidelity + discover the metric scale (uses a real image if given)
python tools/validate_onnx.py --onnx build/model/DA3METRIC-LARGE.onnx \
    --image test-assets/marcie.png --height 504 --width 504

# (optional) INT8 experiment from the FP32 graph, then re-validate
python tools/quantize_int8.py --onnx build/model/DA3METRIC-LARGE.fp32.onnx \
    --output build/model/DA3METRIC-LARGE.int8.onnx
```

## Notes

- **Metric scaling:** the model outputs *canonical* depth. `meters = focal_px * canonical / SCALE`.
  The docs' `SCALE = 300` is unverified; `validate_onnx.py` recovers it empirically and writes it to
  `build/model/validation.json`. The OFX plugin (M3) uses `decimeters = 10 * meters`.
- **Preprocessing:** DA3 does not normalize internally. By default the exporter **bakes ImageNet
  normalization into the graph**, so the plugin can feed a plain `[0,1]` RGB tensor. The `.json`
  sidecar records which convention a given `.onnx` expects.
- Model weights are never committed; CI produces the `.onnx` as a build artifact.
