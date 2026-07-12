# Development guide

## Target platforms

The plugin is fully supported on **three OS / accelerator combinations**, all built from the same
CMake project and verified end-to-end in headless Natron with numerically matching depth:

| OS | Arch | Accelerator (ONNX Runtime EP) | Compiler | Bundle arch dir |
|----|------|-------------------------------|----------|-----------------|
| **macOS** (primary dev target) | arm64 (universal arm64+x86_64) | **CoreML** (ANE/GPU/CPU) | Apple clang | `Contents/MacOS` |
| **Linux** (Rocky 8 verified) | x86-64 | **CUDA 12** (NVIDIA) | gcc/g++ (8.5+) | `Contents/Linux-x86-64` |
| **Windows** (11 verified) | x64 | **CUDA 12** (NVIDIA) | MSVC / VS 2022 | `Contents/Win64` |

Accelerator selection is automatic (`src/OrtAccel.h`: CoreML on macOS, CUDA device 0 on
Linux/Windows) with **CPU fallback** if the accelerator session can't be created. On macOS the
binary is built **universal (arm64 + x86_64)** so it loads in a native-arm64 or Rosetta host; the
CoreML-enabled ORT is arm64, so the inference-enabled build is arm64. Platform specifics live in
[`LINUX.md`](LINUX.md), [`WINDOWS.md`](WINDOWS.md), and [`HOST_COMPATIBILITY.md`](HOST_COMPATIBILITY.md);
GPU-CI runner setup in [`CI_RUNNERS.md`](CI_RUNNERS.md).

## Pinned versions

| Component | Version / source | Notes |
|-----------|------------------|-------|
| OpenFX API + Support lib | `AcademySoftwareFoundation/openfx` **v1.5.1** | fetched via CMake FetchContent |
| ONNX Runtime | **1.27.1** | per-OS package: CoreML arm64 (macOS), `gpu_cuda12` x64 (Linux), `win-x64-gpu_cuda12` (Windows) |
| CUDA / cuDNN (Linux, Windows) | **CUDA 12.x + cuDNN 9** | host prerequisite on the loader path for the CUDA EP; not bundled (too large) |
| Model | `depth-anything/DA3METRIC-LARGE` | Apache-2.0, ViT-L, patch 14; distributed under the `models-v1` release tag |
| Natron (test hosts) | macOS: **arm64** RB-2.6; Linux/Windows: portable **2.5.0** | headless `NatronRenderer` drives CI/local checks on each OS |
| CMake | ≥ 3.20 | |
| Compilers | Apple clang (macOS), gcc 8.5+ (Linux), MSVC 19.4x / VS 2022 (Windows) | GCC < 9 links `stdc++fs` automatically |
| Python | ≥ 3.10 | model export/validation only |

## Local prerequisites

Already present on the dev machine: `git`, Xcode + Apple clang, miniconda Python, `gh`
(authenticated), `cmake`, `git-lfs`, Natron 2.5.0 (x86_64).

Still required before the model/build milestones:

1. **Free disk space** — keep ~15–20 GB clear (model weights + ONNX Runtime + build tree).
2. **arm64 Natron** — download a native/universal build from the NatronGitHub releases page into
   `/Applications` for testing the plugin without Rosetta.

For **Linux** and **Windows** dev/build machines the prerequisites differ (NVIDIA driver + CUDA
12.x runtime + cuDNN 9, gcc/patchelf on Linux, VS 2022 on Windows) — see [`LINUX.md`](LINUX.md)
and [`WINDOWS.md`](WINDOWS.md) for the full per-OS setup, and [`CI_RUNNERS.md`](CI_RUNNERS.md) for
standing up a GPU CI runner.

### Git remote note (two-account machine)

This machine has a global `insteadOf` rule rewriting `https://github.com/` → `ssh://`, and the
SSH identity differs from the `gh` (repo-owning) identity. This repo is pinned to https + the
`gh` token via a repo-local config override:

```sh
git config --local url."https://github.com/samhodge-tokgan/".insteadOf "https://github.com/samhodge-tokgan/"
```

`git ls-remote origin` should succeed. Pushing `.github/workflows/*` additionally requires the
`workflow` token scope: `gh auth refresh -s workflow`.

## Repository layout (as it grows)

```
.
├── LICENSE                     # Apache-2.0
├── README.md
├── docs/                       # this guide, design notes
├── tools/                      # Python: ONNX export, validation, quantization  (M1)
├── src/                        # C++ OFX plugin sources                          (M2+)
├── third_party/openfx/         # vendored OpenFX SDK (submodule/fetch)           (M2)
├── cmake/                      # CMake helpers, toolchain                        (M2)
├── packaging/                  # bundle assembly, .pkg installer                 (M5)
├── test-assets/                # fetched test media (gitignored, has manifests)
└── .github/workflows/          # CI: build, model export/validate, release
```

## Milestone workflow

Each milestone (M0–M9) is one branch → PR → self-review → merge to `main`. Every PR that changes
plugin runtime behavior is exercised end-to-end before merge (build, load in Natron, inspect
output). M6–M8 added the MoGe Focal, Lens Distortion, and AnyCalib estimator plugins; **M9** added
the Linux + Windows CUDA ports, library isolation, model-less installers, and real-GPU CI. See the
roadmap table in the README.

## Headless testing (Natron)

The native arm64 build ships `NatronRenderer` (and `natron-python`) for GUI-less
verification, which we drive from CI and local checks:

- Point Natron at the built plugin without a system install via **`OFX_PLUGIN_PATH`**
  (see the [Natron environment reference](https://natron.readthedocs.io/en/rb-2.5/_environment.html)).
- Build a graph procedurally (`Read → DepthAnything3 → Write`) as a **`.ntp`** project or a Python
  script and render it headlessly (see
  [Natron execution docs](https://natron.readthedocs.io/en/rb-2.5/devel/natronexecution.html)),
  e.g. `NatronRenderer -w Write1 1-1 project.ntp` or `NatronRenderer -t script.py`.
- Diff the rendered depth output against a reference to gate plugin changes (wired up in M2/M3).

Note: the arm64 Natron is unsigned; on first launch it may need a Gatekeeper allow
(`xattr -dr com.apple.quarantine /Applications/Natron-2.6-arm64.app` if it was quarantined).

## Color pipeline (design intent)

Input is treated as **ACEScg**. Before inference: ACEScg (AP1, linear) → sRGB (Rec.709 primaries +
sRGB transfer function), then ImageNet normalize (`mean=(0.485,0.456,0.406)`,
`std=(0.229,0.224,0.225)`) — unless the exported ONNX graph already bakes normalization in (to be
verified in M1). Output depth is linear float32 decimeters, written to the output clip's channels.
