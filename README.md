# Depth Anything 3 — OpenFX plugin

A hardware-accelerated **OpenFX** plugin that predicts **metric depth** from an image using
**[Depth Anything 3](https://depth-anything-3.github.io/)** (the `DA3METRIC-LARGE` checkpoint),
run through **ONNX Runtime** — the **CoreML execution provider** on Apple Silicon and the
**CUDA execution provider** on Linux and Windows (NVIDIA), with automatic CPU fallback.

- **Input:** RGB(A) frame buffer, assumed to be **ACEScg** (linear, AP1 primaries).
- **Output:** a same-size **float32 grayscale depth (Z) map in decimeters**.
- **Acceleration:** ONNX Runtime — CoreML EP (Apple Neural Engine / GPU / CPU) on macOS,
  CUDA EP on Linux x86-64 and Windows x64; CPU fallback everywhere.
- **Host tested:** [Natron](https://www.natrongithub.com/) on an Apple M1 (CoreML), on
  Rocky Linux 8 with RTX 3090s (CUDA), and on Windows 11 with RTX 3090s (CUDA).

> Status: **working end-to-end** on **Apple Silicon** (ACEScg → CoreML depth → decimeter Z),
> **Linux x86-64 / NVIDIA CUDA**, and **Windows x64 / NVIDIA CUDA** — all verified in headless
> Natron, producing numerically matching depth. See
> [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md), platform build/deploy notes in
> [`docs/LINUX.md`](docs/LINUX.md) and [`docs/WINDOWS.md`](docs/WINDOWS.md), packaging in
> [`packaging/README.md`](packaging/README.md), and the milestone roadmap below.

## Plugins in this bundle

The single `.ofx.bundle` provides three plugins (sharing one ONNX Runtime stack):

- **Depth Anything 3** — metric depth (decimeters, float32) from ACEScg, on the platform
  accelerator (**CoreML** on macOS, **CUDA** on Linux/Windows) with automatic **CPU fallback**.
- **MoGe Focal** — estimates camera **focal length / FOV / intrinsics** from one frame using
  MoGe-2 (MIT). It's an analysis node: press **Analyze current frame** and it fills output
  parameters (focal px, horizontal/vertical FOV, principal point) you can link to a camera.
  DA3 doesn't predict intrinsics, so MoGe covers that gap. (MoGe's dynamic graph is **not**
  CoreML-executable, so it runs on **CPU on macOS**; on Linux/Windows it attempts **CUDA** first
  and falls back to CPU.)
- **Lens Distortion** — estimates lens distortion from a natural image via **AnyCalib** (MIT/
  Apache-2.0) and outputs the downstream lens data: **OpenCV coefficients**, **3DEqualizer**
  coefficients (Radial Std Deg 4), and the **overscan / padding** needed to render CG that will be
  re-distorted to the plate. It does *not* apply distortion — the image passes through. Press
  **Estimate (AnyCalib)** to fill k1,k2,focal (or enter them manually / from a sidecar — there is
  no standard OFX lens-metadata channel); overscan and 3DE outputs update live. Overscan is
  validated against OpenCV (`getOptimalNewCameraMatrix`).

> **Lens Distortion — AnyCalib estimator:** the ML estimator is **AnyCalib** — its DINOv2 field
> network runs in ONNX (on the platform accelerator / CPU) and its camera fit (DLT init + Gauss-Newton) is reimplemented
> host-side in C++. Its `radial` model maps directly to OpenCV (`fx,fy,cx,cy,k1,k2`). Verified: the
> plugin's **focal recovery matches AnyCalib within ~0.2%**. Distortion coefficients track AnyCalib
> but are small/noisy on near-undistorted images, and the dynamic-resolution ONNX export shifts
> them slightly vs the original net (see `docs/BACKLOG.md`). The deterministic overscan + OpenCV/3DE
> outputs are exact and tested.

## Install

Grab the installer for your platform from the [Releases](../../releases) page (or build one —
see [`packaging/README.md`](packaging/README.md)):

| Platform | Installer | Installs to |
|----------|-----------|-------------|
| **macOS** (Apple Silicon) | `DepthAnything3-<ver>-macos-arm64.pkg` | `/Library/OFX/Plugins` |
| **Linux** (x86-64) | `DepthAnything3-<ver>-linux-x86_64.tar.xz` | extract to `/usr/OFX/Plugins` or `~/OFX/Plugins` |
| **Windows** (x64) | `DepthAnything3-<ver>-windows-x64.zip` | extract to `%CommonProgramFiles%\OFX\Plugins` |

The plugins then appear in your OFX host under **TokGan › Depth Anything 3** (and MoGe Focal,
Lens Distortion).

> **Models are downloaded, not bundled.** To stay under GitHub's 2 GB per-asset limit, the
> installers are **model-less**; each ships a `fetch_models` script (in the bundle's `Contents/`)
> that pulls the ONNX models (~1.3 GB, published under the `models-v1` release tag) into
> `Contents/Resources`. Run it once after installing:
> `bash Contents/fetch_models.sh` (macOS/Linux) or `powershell -File Contents\fetch_models.ps1`
> (Windows). The models are pinned by SHA-256. You can also point the *Model file* parameter or
> `DA3_MODEL_PATH` at a model you already have.

**GPU prerequisite (Linux/Windows):** the CUDA execution provider needs an NVIDIA driver, the
**CUDA 12.x runtime**, and **cuDNN 9** on the loader path — see [`docs/LINUX.md`](docs/LINUX.md) /
[`docs/WINDOWS.md`](docs/WINDOWS.md). Without them the plugin still loads and runs on **CPU**.
The macOS build is ad-hoc signed (not yet notarized) — see the packaging doc for the Gatekeeper
note.

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
             └─▶ ONNX Runtime (CoreML on macOS / CUDA on Linux+Windows / CPU) → metric depth + sky
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

On macOS, Apple's unified memory and the CoreML EP expose **no VRAM byte-limit API** (a hard
byte cap is a CUDA-only ONNX Runtime option). The plugin's "maximum resources" control therefore
governs cost primarily through **processing resolution**, which works identically on every
platform/accelerator:

- **Compute units** — `All` / `CPU+GPU` / `CPU+ANE` / `CPU only` (`MLComputeUnits`); applies to the
  CoreML EP on macOS. On Linux/Windows the accelerator is the CUDA EP (device 0) with CPU fallback.
- **Max threads** — the ONNX Runtime intra-op thread cap.
- **Processing (long side)** — the inference resolution (aspect preserved, rounded to a multiple of
  14). This is the primary memory/speed lever on **all** platforms, made possible by the
  **dynamic-resolution ONNX model** (one graph runs at any resolution — see `tools/export_onnx.py`).
- **Auto resolution from memory budget** — when enabled, a *Memory budget (MB)* maps to a
  processing resolution (an approximate calibration; exact on neither unified memory nor VRAM).

Note: the dynamic model trades some accelerator node coverage (more ops fall back to CPU — most
visible on CoreML) for resolution flexibility, so it can be slower than a fixed-resolution export
at the same size. Image **sequences** are supported; the inference session is cached across frames.

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
- Verified: on a sample frame the plugin's output (26–104 decimeters ≈ 2.6–10.4 m) matches the
  reference `inference()` depth (2.4–12.9 m). The result is **numerically consistent across
  accelerators** — the same model at 504×504 gives a depth mean of 0.702644 (macOS CoreML),
  0.702642 (Linux CUDA), and 0.702647 (Windows CUDA).

## Model & license

- Model: [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
  — ViT-Large, patch size 14, **Apache-2.0** (commercial use permitted).
- This plugin is **Apache-2.0** (see [`LICENSE`](LICENSE)).
- ONNX Runtime **v1.27.1** is fetched by CMake and bundled at release time, per platform: the
  CoreML-enabled arm64 package on macOS, and the CUDA 12 GPU packages on Linux x86-64 / Windows x64.
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
| **M6** | **MoGe Focal** — 2nd plugin in the bundle: camera focal / FOV / intrinsics estimation |
| **M7** | **Lens Distortion** — 3rd plugin: overscan + OpenCV / 3DEqualizer coefficient outputs |
| **M8** | **AnyCalib** ML lens-distortion estimator wired into the Lens Distortion plugin |
| **M9** | **Linux (CUDA) + Windows (CUDA) ports** — cross-platform GPU acceleration, library isolation, model-less installers, self-hosted GPU CI |

## Pinned versions

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the pinned OpenFX / ONNX Runtime / model /
Natron versions and full local setup instructions.

## Building

Requires CMake ≥ 3.20 and a C++17 compiler. The OpenFX SDK **and** the correct per-platform ONNX
Runtime package are fetched automatically by CMake (`FetchContent`, OFX pinned to 1.5.1). The same
`cmake` invocation works on every platform; the generator and toolchain differ:

```sh
# macOS (Apple clang) — universal arm64+x86_64 by default; -DDA3_UNIVERSAL=OFF for host-arch only.
# (ORT's CoreML build is arm64-only, so the inference-enabled build is arm64.)
cmake -S . -B build -DDA3_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Linux (gcc/g++, CUDA 12 ORT) — Rocky 8's default gcc 8.5 is fine (CMake links stdc++fs for it).
cmake -S . -B build -DDA3_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# Windows (MSVC / Visual Studio 2022, CUDA 12 ORT)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DDA3_WITH_ONNX=ON
cmake --build build --config Release --parallel
```

The result is `build/DepthAnything3.ofx.bundle`, laid out per the OFX spec:
`Contents/{MacOS,Linux-x86-64,Win64}` for the binary and its bundled ONNX Runtime. Full
platform build/deploy notes: [`docs/LINUX.md`](docs/LINUX.md), [`docs/WINDOWS.md`](docs/WINDOWS.md).
Host-coexistence / library-isolation design: [`docs/HOST_COMPATIBILITY.md`](docs/HOST_COMPATIBILITY.md).

### Testing in Natron (headless)

All three platforms are validated end-to-end in headless Natron (`NatronRenderer`); the macOS
invocation is shown here (Linux/Windows equivalents are in `docs/LINUX.md` / `docs/WINDOWS.md`):

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

A cross-platform GPU smoke test (`tests/ort_check.cpp`, built as `ort_check`) loads a model on the
platform accelerator and runs one inference; CI runs it on real NVIDIA GPUs via a self-hosted
runner (see [`docs/CI_RUNNERS.md`](docs/CI_RUNNERS.md)).
