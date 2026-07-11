# openfx-onnx-depthanything3

A hardware-accelerated **OpenFX** plugin that predicts **metric depth** from an image using
**[Depth Anything 3](https://depth-anything-3.github.io/)** (the `DA3METRIC-LARGE` checkpoint),
run through **ONNX Runtime** with the **CoreML execution provider** on Apple Silicon.

- **Input:** RGB(A) frame buffer, assumed to be **ACEScg** (linear, AP1 primaries).
- **Output:** a same-size **float32 grayscale depth (Z) map in decimeters**.
- **Acceleration:** ONNX Runtime CoreML EP (Apple Neural Engine / GPU / CPU).
- **Host tested:** [Natron](https://www.natrongithub.com/) on an Apple M1.

> Status: **early development.** See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) and the
> milestone roadmap below. Built as a series of incremental PRs.

## Why

Depth passes are useful for compositing (defocus, atmospheric fog, relighting, 3D reprojection).
Depth Anything 3's metric variant recovers *scaled* depth from a single frame. This plugin brings
that into an OFX host with a native, GPU-accelerated inference path and a color pipeline that
assumes an ACEScg working space.

## How it works

```
ACEScg RGBA float in
   └─▶ ACEScg → sRGB (AP1→Rec.709 primaries + sRGB transfer)
        └─▶ resize to a multiple of 14 (DINOv2 patch size); feed [0,1] RGB
             (ImageNet normalization is baked into the ONNX graph)
             └─▶ ONNX Runtime (CoreML EP) → metric depth + sky
                  └─▶ decimeters = 10 × depth × gain   (optional manual scale)
                       └─▶ bilinear upsample to source resolution
                            └─▶ float32 Z written to output (grayscale)
```

### The metric scaling

`DA3METRIC-LARGE` is fine-tuned for **metric** monocular depth: its `depth` output (as produced
by the reference `inference()`) is the metric depth we consume directly, so
**`decimeters = 10 × depth`** (with an optional user `gain` for correction). The exact scale is
characterised in CI by `tools/validate_onnx.py`.

Note: this checkpoint does **not** predict camera intrinsics in monocular mode
(`inference().intrinsics` is `None`), so there is no focal-length/intrinsics path — a manual
scale/gain parameter covers correction instead. The model also emits a **`sky`** output, which the
plugin uses to handle sky regions (the model's internal sky-depth clamp is removed for export
because it is not ONNX-representable).

### Resource / "VRAM" control

Apple's unified memory and the CoreML EP expose **no VRAM byte-limit API** (that is a CUDA-only
ONNX Runtime option). The plugin's "maximum resources" control therefore governs cost through:

- **Compute units** — `ALL` / `CPU+GPU` / `CPU+ANE` / `CPU only` (`MLComputeUnits`).
- **Thread caps** — intra/inter-op thread counts.
- **Processing resolution & tiling** — smaller inference resolution, or overlapping tiles with
  scale/shift alignment and feathered blending, to bound peak memory.

### Operational notes (important)

- **Model file:** the ONNX model is selected by the *Model file* parameter, or the
  `DA3_MODEL_PATH` environment variable (the release build bundles it in `Contents/Resources`).
  Build it with `tools/export_onnx.py` or download the CI artifact.
- **Fixed processing resolution:** the exported model is traced at a fixed resolution (default
  **504×504**) because the DINOv2 positional embedding is baked at trace time. The plugin's
  *Processing resolution* must match the model's export resolution. (Multi-resolution support is a
  M4 concern.)
- **Depth is data, not color:** the output is float metric depth, not an image. In your host, read
  the source with its correct colorspace (the plugin does ACEScg→sRGB internally) and write the
  depth output through a **raw/data** colorspace so the values are not tone-mapped or clamped.
- Verified locally: on a sample frame the plugin's output (26–104 decimeters ≈ 2.6–10.4 m) matches
  the reference `inference()` depth (2.4–12.9 m), running on the CoreML EP.

## Model & license

- Model: [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
  — ViT-Large, patch size 14, **Apache-2.0** (commercial use permitted).
- This plugin is **Apache-2.0** (see [`LICENSE`](LICENSE)).
- ONNX Runtime **v1.27.1** (arm64, CoreML EP) is fetched by CMake and bundled at release time.
- ONNX Runtime is MIT-licensed (bundled at release time).

## Roadmap (milestones = PRs)

| Milestone | Content |
|-----------|---------|
| **M0** | Repo bootstrap: license, docs, structure, CI skeleton |
| **M1** | Model → ONNX pipeline (FP16; INT8 experiment), quality validation in CI |
| **M2** | OFX plugin skeleton: universal `.ofx.bundle`, float-RGBA passthrough, loads in Natron |
| **M3** | ONNX Runtime + CoreML inference, ACEScg color pipeline, decimeter depth output |
| **M4** | Resource controls (compute units, threads, resolution/tiling) + image-sequence support |
| **M5** | Packaging: bundled runtime + model, `.pkg` installer, release workflow |

## Pinned versions

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the pinned OpenFX / ONNX Runtime / model /
Natron versions and full local setup instructions.

## Building

Requires CMake ≥ 3.20 and a C++17 compiler (Apple clang). The OpenFX SDK is fetched
automatically by CMake (`FetchContent`, pinned to OFX 1.5.1).

```sh
# Universal arm64+x86_64 bundle (default). Use -DDA3_UNIVERSAL=OFF for host-arch only.
cmake -S . -B build -DDA3_UNIVERSAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Result: build/DepthAnything3.ofx.bundle
# Install into the user OFX dir (override with -DOFX_INSTALL_DIR=...):
cmake --build build --target install-local
```

### Testing in Natron (headless)

```sh
# NOTE: this Natron build does not scan ~/Library/OFX/Plugins by default — point it there.
OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins" \
  /Applications/Natron-2.6-arm64.app/Contents/MacOS/NatronRenderer \
  --clear-openfx-cache -t tests/natron/check_plugin.py < /dev/null   # discovery -> RESULT: PASS

# Full Read -> DepthAnything3 -> Write render:
python3 tests/natron/make_test_image.py test-assets/synthetic_test.png
DA3_INPUT=$PWD/test-assets/synthetic_test.png DA3_OUTPUT=$PWD/build/test/out.png \
  OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins" \
  /Applications/Natron-2.6-arm64.app/Contents/MacOS/NatronRenderer \
  --clear-openfx-cache -t tests/natron/render_passthrough.py < /dev/null
```
