# openfx-onnx-depthanything3

A hardware-accelerated **OpenFX** plugin that predicts **metric depth** from an image using
**[Depth Anything 3](https://depth-anything-3.github.io/)** (the `DA3METRIC-LARGE` checkpoint),
run through **ONNX Runtime** — the **CoreML execution provider** on Apple Silicon and the
**CUDA execution provider** on Linux (NVIDIA), with automatic CPU fallback.

- **Input:** RGB(A) frame buffer, assumed to be **ACEScg** (linear, AP1 primaries).
- **Output:** a same-size **float32 grayscale depth (Z) map in decimeters**.
- **Acceleration:** ONNX Runtime — CoreML EP (Apple Neural Engine / GPU / CPU) on macOS,
  CUDA EP on Linux x86-64; CPU fallback everywhere.
- **Host tested:** [Natron](https://www.natrongithub.com/) on an Apple M1 (CoreML) and on
  Rocky Linux 8 with RTX 3090s (CUDA).

> Status: **working end-to-end** on **Apple Silicon** (ACEScg → CoreML depth → decimeter Z)
> and on **Linux x86-64 / NVIDIA CUDA** (verified in headless Natron). See
> [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md), Linux build/deploy in
> [`docs/LINUX.md`](docs/LINUX.md), packaging in [`packaging/README.md`](packaging/README.md),
> and the milestone roadmap below.

## Plugins in this bundle

The single `.ofx.bundle` provides two plugins (sharing the ONNX Runtime/CoreML stack):

- **Depth Anything 3** — metric depth (decimeters, float32) from ACEScg, on CoreML.
- **MoGe Focal** — estimates camera **focal length / FOV / intrinsics** from one frame using
  MoGe-2 (MIT). It's an analysis node: press **Analyze current frame** and it fills output
  parameters (focal px, horizontal/vertical FOV, principal point) you can link to a camera.
  DA3 doesn't predict intrinsics, so MoGe covers that gap. (MoGe runs on CPU — its dynamic graph
  isn't CoreML-executable; the depth plugin uses CoreML.)
- **Lens Distortion** — estimates lens distortion from a natural image via **AnyCalib** (MIT/
  Apache-2.0) and outputs the downstream lens data: **OpenCV coefficients**, **3DEqualizer**
  coefficients (Radial Std Deg 4), and the **overscan / padding** needed to render CG that will be
  re-distorted to the plate. It does *not* apply distortion — the image passes through. Press
  **Estimate (AnyCalib)** to fill k1,k2,focal (or enter them manually / from a sidecar — there is
  no standard OFX lens-metadata channel); overscan and 3DE outputs update live. Overscan is
  validated against OpenCV (`getOptimalNewCameraMatrix`).

> **Lens Distortion — AnyCalib estimator:** the ML estimator is **AnyCalib** — its DINOv2 field
> network runs in ONNX (CoreML/CPU) and its camera fit (DLT init + Gauss-Newton) is reimplemented
> host-side in C++. Its `radial` model maps directly to OpenCV (`fx,fy,cx,cy,k1,k2`). Verified: the
> plugin's **focal recovery matches AnyCalib within ~0.2%**. Distortion coefficients track AnyCalib
> but are small/noisy on near-undistorted images, and the dynamic-resolution ONNX export shifts
> them slightly vs the original net (see `docs/BACKLOG.md`). The deterministic overscan + OpenCV/3DE
> outputs are exact and tested.

## Install

Grab the `.pkg` from the [Releases](../../releases) page (or build one — see
[`packaging/README.md`](packaging/README.md)) and install it; the bundle lands in
`/Library/OFX/Plugins` with the model included, and appears in your OFX host under
**TokGan › Depth Anything 3**. The build is ad-hoc signed (not yet notarized) — see the
packaging doc for the Gatekeeper note.

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

- **Compute units** — `All` / `CPU+GPU` / `CPU+ANE` / `CPU only` (`MLComputeUnits`).
- **Max threads** — the ONNX Runtime intra-op thread cap.
- **Processing (long side)** — the inference resolution (aspect preserved, rounded to a multiple of
  14). This is the primary memory/speed lever, made possible by the **dynamic-resolution ONNX
  model** (one graph runs at any resolution — see `tools/export_onnx.py`).
- **Auto resolution from memory budget** — when enabled, a *Memory budget (MB)* maps to a
  processing resolution (an approximate calibration, since CoreML has no hard cap).

Note: the dynamic model trades some CoreML node coverage (more ops fall back to CPU) for
resolution flexibility, so it can be slower than a fixed-resolution export at the same size.
Image **sequences** are supported; the inference session is cached across frames.

### Operational notes (important)

- **Model file:** the ONNX model is selected by the *Model file* parameter, or the
  `DA3_MODEL_PATH` environment variable (the release build bundles it in `Contents/Resources`).
  Build it with `tools/export_onnx.py` or download the CI artifact.
- **Dynamic resolution:** the model is exported with the **dynamo** exporter so one ONNX graph runs
  at any resolution (H, W multiples of 14). The plugin feeds an aspect-preserving resolution derived
  from the *Processing (long side)* / memory-budget controls. (A legacy fixed-resolution export is
  available via `export_onnx.py --static`.)
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
