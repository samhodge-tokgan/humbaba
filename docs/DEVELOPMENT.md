# Development guide

## Target platform

Primary development + test target: **Apple Silicon (M1), macOS 26.x**, inference via
**ONNX Runtime CoreML execution provider**. The plugin binary is built **universal
(arm64 + x86_64)** so it loads in either a native-arm64 or a Rosetta (x86_64) OFX host.

## Pinned versions

| Component | Version / source | Notes |
|-----------|------------------|-------|
| OpenFX API + Support lib | `AcademySoftwareFoundation/openfx` **v1.5.x** | vendored in M2 |
| ONNX Runtime | **1.20.x** (arm64/universal, CoreML EP) | verify `GetAvailableProviders()` lists CoreML |
| Model | `depth-anything/DA3METRIC-LARGE` | Apache-2.0, ViT-L, patch 14 |
| Natron (test host) | **arm64/universal** build from `NatronGitHub/Natron` releases | installed Natron 2.5.0 is x86_64-only |
| CMake | ≥ 3.20 (installed 4.4.0) | |
| Python | ≥ 3.10 (miniconda 3.13 present) | model export/validation only |

## Local prerequisites

Already present on the dev machine: `git`, Xcode + Apple clang, miniconda Python, `gh`
(authenticated), `cmake`, `git-lfs`, Natron 2.5.0 (x86_64).

Still required before the model/build milestones:

1. **Free disk space** — keep ~15–20 GB clear (model weights + ONNX Runtime + build tree).
2. **arm64 Natron** — download a native/universal build from the NatronGitHub releases page into
   `/Applications` for testing the plugin without Rosetta.

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

Each milestone (M0–M5) is one branch → PR → self-review → merge to `main`. Every PR that changes
plugin runtime behavior is exercised end-to-end before merge (build, load in Natron, inspect
output). See the roadmap table in the README.

## Color pipeline (design intent)

Input is treated as **ACEScg**. Before inference: ACEScg (AP1, linear) → sRGB (Rec.709 primaries +
sRGB transfer function), then ImageNet normalize (`mean=(0.485,0.456,0.406)`,
`std=(0.229,0.224,0.225)`) — unless the exported ONNX graph already bakes normalization in (to be
verified in M1). Output depth is linear float32 decimeters, written to the output clip's channels.
