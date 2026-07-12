# Backlog

Post-MVP work, not yet scheduled.

## Cross-platform ports (Windows 11, Rocky Linux 8) — GPU + CPU

The current build targets **macOS arm64** with the ONNX Runtime **CoreML** execution
provider. Port status:

- **Rocky Linux 8 — GPU (CUDA EP): DONE + HARDENED + END-TO-END VERIFIED.** Built on
  a 2× RTX 3090 VM (CUDA 12.6, cuDNN 9, gcc-toolset-12). Cross-platform CMake emits
  the `Contents/Linux-x86-64/` bundle (note: **hyphens** — the OFX-spec arch dir; an
  underscore makes hosts silently skip the binary) with `$ORIGIN` RUNPATH; ships
  `libonnxruntime.so.1` + the CUDA/TensorRT provider libs. `src/OrtAccel.h` selects
  CoreML (macOS) vs CUDA (Linux/Windows); DepthEngine/MoGeEngine/AnyCalibEngine all
  route through `da3::AppendAccelerator`, with a **CPU fallback** if CUDA/runtime libs
  are missing. Verified: DA3 metric depth runs on CUDA — 504×504 in ~72 ms, GPU 55%,
  ~2.4 GB VRAM (`tests/ort_check.cpp <model.onnx>` is the GPU smoke test). **End-to-end
  in Natron (headless NatronRenderer): all 3 plugins load, DA3 renders a full-res
  (1920×1080) float32 metric-depth EXR (~9 dm on the synthetic test), same size as
  input, CUDA session active.** Standard install paths work (`~/OFX/Plugins`,
  `/usr/OFX/Plugins`, `OFX_PLUGIN_PATH`). Linux CI job added (`ci.yml` `build-linux`,
  CPU-only runner — validates build/layout/CUDA-EP-present; GPU inference is the
  self-hosted step). CUDA runtime deps documented as a host prerequisite in
  [`docs/LINUX.md`](LINUX.md) (bundling them is impractical: ~1 GB + cuDNN's multi-GB
  sublibs, on top of the 1.3 GB model).
- **Windows 11** — GPU (CUDA and/or DirectML EP) and CPU builds. NOT STARTED —
  needs a Windows machine. `.ofx.bundle` uses `Contents/Win64/`; ORT DLL + provider
  DLLs alongside the `.ofx`; MSVC/CMake build; the `dladdr`-based bundle-path lookup
  needs a `GetModuleFileName`/`GetModuleHandleEx` equivalent for Win64.

Notes / scope:
- The universal-vs-arm64 constraint (CoreML → arm64-only) is macOS-specific; other
  platforms build for their native arch without that restriction.
- MoGe and AnyCalib run on **CPU** on macOS (dynamic graphs not CoreML-executable);
  on Linux they now *attempt* CUDA first (with CPU fallback) via `OrtAccel.h`.

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
