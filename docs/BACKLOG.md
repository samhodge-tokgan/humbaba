# Backlog

Post-MVP work, not yet scheduled.

## Cross-platform ports (Windows 11, Rocky Linux 8) — GPU + CPU

The current build targets **macOS arm64** with the ONNX Runtime **CoreML** execution
provider. Once the arm64 macOS version is complete, port to:

- **Windows 11** — GPU (CUDA and/or DirectML EP) and CPU builds.
- **Rocky Linux 8** — GPU (CUDA EP) and CPU builds.

Notes / scope:
- The plugin code is already EP-agnostic through `da3::DepthEngine` / the engines —
  the main work is: per-platform ONNX Runtime binaries (CUDA/DirectML/CPU), the
  `.ofx.bundle` layout for Windows (`Win64/`) and Linux (`Linux-x86_64/`,
  `Linux-aarch64/`), rpath/loader handling per OS, and CI build matrices.
- The universal-vs-arm64 constraint (CoreML → arm64-only) is macOS-specific; other
  platforms can be built for their native arch without that restriction.
- MoGe and AnyCalib currently run on **CPU** even on macOS (dynamic graphs not
  CoreML-executable); on CUDA/DirectML they should run on GPU — revisit EP selection.
- The user has offered access to another machine for the non-macOS work.

## AnyCalib distortion accuracy

The dynamic-resolution ONNX export uses a monkeypatched DINOv2 positional-embedding
(drops the `interpolate_offset=0.1` kludge to keep shapes symbolic). This slightly
shifts the distortion (k1,k2) estimate vs the original AnyCalib net (focal is
unaffected, ~0.2%). Options: (a) a fixed-resolution export that preserves the
original interpolation, (b) transcribe the analytic Jacobians (currently
finite-difference), (c) match AnyCalib's bicubic+antialias resize exactly.

## Lens metadata passthrough

Revisit if OpenFX issue #142 (clip/image metadata, incl. lens info) is implemented,
to allow reading lens data from the host instead of manual/sidecar entry.
