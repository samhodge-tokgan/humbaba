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
        └─▶ ImageNet normalize, resize to a multiple of 14 (DINOv2 patch size)
             └─▶ ONNX Runtime (CoreML EP) → canonical depth + camera intrinsics
                  └─▶ decimeters = focal_px × depth / 30   (focal: predicted or manual override)
                       └─▶ bilinear upsample to source resolution
                            └─▶ float32 Z written to output (grayscale)
```

### The metric scaling

`DA3METRIC-LARGE` emits **focal-normalized (canonical) depth**, not meters. The documented
conversion is `meters = focal_px × output / 300`, hence **`decimeters = focal_px × output / 30`**.
The focal length in pixels comes from the model's **predicted intrinsics** by default, with a
**manual override** parameter for known camera geometry.

### Resource / "VRAM" control

Apple's unified memory and the CoreML EP expose **no VRAM byte-limit API** (that is a CUDA-only
ONNX Runtime option). The plugin's "maximum resources" control therefore governs cost through:

- **Compute units** — `ALL` / `CPU+GPU` / `CPU+ANE` / `CPU only` (`MLComputeUnits`).
- **Thread caps** — intra/inter-op thread counts.
- **Processing resolution & tiling** — smaller inference resolution, or overlapping tiles with
  scale/shift alignment and feathered blending, to bound peak memory.

## Model & license

- Model: [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
  — ViT-Large, patch size 14, **Apache-2.0** (commercial use permitted).
- This plugin is **Apache-2.0** (see [`LICENSE`](LICENSE)).
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
