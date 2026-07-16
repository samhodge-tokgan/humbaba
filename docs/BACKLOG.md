# Backlog

Post-MVP work. The cross-platform ports below are **complete and merged** (M9) and kept here as a
record; the remaining sections (AnyCalib accuracy, lens metadata, and the deferred items at the
bottom) are the genuinely open work.

## Cross-platform ports (Windows 11, Rocky Linux 8) — GPU + CPU — ✅ DONE (M9)

The plugin ships on **macOS/arm64 (CoreML)**, **Linux/x86-64 (CUDA)**, and **Windows/x64 (CUDA)**,
all with automatic CPU fallback, from one cross-platform CMake project. Port status:

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
- **Windows 11 — GPU (CUDA EP): DONE + END-TO-END VERIFIED.** Built with MSVC / Visual
  Studio 2022 (CUDA 12.8, cuDNN 9.8) on a Windows 11 box with 2× RTX 3090. CMake emits
  the `Contents/Win64/` bundle with `onnxruntime.dll` + CUDA/TensorRT provider DLLs.
  The `.ofx` delay-loads `onnxruntime.dll` and a DllMain hook ([`src/WinLoader.cpp`])
  loads it by full path from the bundle dir, so the plugin is self-contained without
  perturbing the host's DLL search. Bundle-path lookup uses `GetModuleHandleExW`/
  `GetModuleFileNameA`; ORT session paths go through `da3::OrtPath` (Windows ORT wants
  `const wchar_t*`). Verified: `ort_check` runs DA3 on CUDA (504×504 ~92 ms); all 3
  plugins load in headless Natron and render a full-res metric-depth EXR numerically
  matching Linux/macOS. cuDNN 9 is a host prerequisite (install to CUDA `bin`; see
  [`docs/WINDOWS.md`](WINDOWS.md)). Open item: Natron 2.5.0 raises an access violation
  during **shutdown** (after the output is written) — teardown/ordering with the ORT/
  CUDA runtime; does not affect output. DirectML EP (for non-NVIDIA GPUs) not pursued
  since the target has NVIDIA hardware.

Notes / scope:
- The universal-vs-arm64 constraint (CoreML → arm64-only) is macOS-specific; other
  platforms build for their native arch without that restriction.
- MoGe and AnyCalib run on **CPU** on macOS (dynamic graphs not CoreML-executable);
  on **Linux and Windows** they *attempt* CUDA first (with CPU fallback) via `OrtAccel.h`.
  DA3 uses the platform accelerator (CoreML/CUDA) on every OS.

## Host compatibility — verified in Foundry Nuke ✅ (macOS, 2026-07-13)

The **v0.5.1** release `.pkg` was installed and loaded in **Nuke 16.0v8** (Apple
Silicon, arm64 / CoreML): all three plugins (Depth Anything 3, MoGe Focal, Lens
Distortion) are discovered and run, and DA3 depth renders correctly — tested up to a
**2016** patch-aligned processing long side on a 1920×1080 plate (supersampling above
native). This validates the **M9 library isolation in a real host**: Nuke ships its own
**libtorch** ML stack, and our privately-renamed (`libonnxruntime_da3.*`), symbol-isolated
ONNX Runtime coexists with **no clash** — the whole point of exporting only the OFX entry
points + renaming the bundled ORT. (Install detail: Nuke doesn't scan `~/Library/OFX/Plugins`
by default; point it there with `OFX_PLUGIN_PATH`, or install to `/Library/OFX/Plugins`.)

### Nuke on Windows — two DLL-isolation axes (found 2026-07-13, Nuke 16.1v3)

Testing DA3 in **Nuke 16.1 on Windows** surfaced two distinct problems, diagnosed in
that host's own process:

1. **`onnxruntime.dll` base-name collision with Windows ML — ✅ FIXED in 0.7.0.**
   Windows 11 ships `C:\Windows\System32\onnxruntime.dll` (Windows ML, ORT `1.17.x`).
   Nuke makes it resident; the Windows loader dedups DLLs by base name, so our
   full-path load of a same-named `onnxruntime.dll` returned the **OS** module — binding
   us to 1.17 and rejecting our API-27 calls (`only [1,17] supported`) / crashing. Fixed
   by privately renaming the bundled runtime to **`onnxruntime_da3.dll`** (the delay-load
   hook loads that name). Providers don't import `onnxruntime.dll` (shared-provider
   bridge), so no PE patching was needed. **Proven:** in a process with System32's 1.17
   resident, `onnxruntime_da3.dll` (1.27) loads as a *distinct* module and both coexist.

2. **MSVC runtime version — ✅ FIXED in 0.8.0.** Prebuilt ORT 1.27 needs the dynamic VC++
   runtime **≥ 14.40**. Nuke 16.1 bundles **14.36** in its app dir, which wins the search
   order and becomes the process's only `MSVCP140.dll` — so `onnxruntime_da3.dll` failed to
   initialize (`LoadLibrary` error 1114). Not fixable from inside the plugin by bundling a
   newer redist (can't have two `MSVCP140.dll` in one process). **Fixed by building the
   bundled ONNX Runtime from source with the *static* MSVC runtime**
   (`--enable_msvc_static_runtime`): `dumpbin /dependents` confirms the runtime + provider
   DLLs carry **zero** `MSVCP140`/`VCRUNTIME140` imports, so the host's CRT version is
   irrelevant and the 1114 failure is structurally impossible — no host hacks needed.
   **Verified in Nuke 16.1v3**: the ctypes load that previously crashed now returns
   `LOADED OK / ORT VERSION 1.27.1` inside Nuke's process, and a full DA3 depth render runs
   on the RTX 3090 (CUDA EP, GPU-confirmed). The static ORT is a pinned repo release asset
   (`ort-static-win-1.27.1`); Windows CMake FetchContent points at it (mac/Linux unchanged).

Still open on the host-compat front:
- **Static-CRT ONNX Runtime build — ✅ DONE for Windows (0.8.0), Linux still potential.**
  See axis 2 above; the general fix for host-bundled-runtime skew. The Windows static ORT
  is currently built from source out-of-CI and hosted as a pinned release asset
  (`ort-static-win-1.27.1`); a from-source CI job would let it rebuild automatically on ORT
  bumps. Linux (glibc/libstdc++ skew vs a host) could want the same treatment if a host
  conflict surfaces.
- **Nuke on Linux (CUDA)** — Nuke ships its own CUDA/cuDNN and pins a CUDA major per
  release, so our CUDA-EP ORT could contend with Nuke's runtime (cuDNN 9.x SONAME
  sharing, VRAM). See `HOST_COMPATIBILITY.md`.
- **Co-load clash test** — a CI harness that loads our bundle alongside another ORT-using
  OFX plugin to prove the isolation under a genuine duplicate-library scenario.

### DaVinci Resolve on Linux — crashes in Resolve's `vfork` render worker (found 2026-07-15)

Applying the DepthAnything3 node in **DaVinci Resolve 21 on Rocky Linux 8** crashes Resolve
with **SIGSEGV in ORT session creation** (`da3::DepthEngine::DepthEngine` → `Ort::Session`,
parsing the 1.3 GB model). Root cause is **not** a library clash — it was mis-diagnosed as a
CUDA/cuDNN version clash at first, but ruled out systematically:

- Forcing **CPU** (`DA3_FORCE_CPU=1`, a new `OrtAccel.h` override) crashes **identically** —
  so it is provider-independent, not the CUDA EP.
- **NOT** cuDNN (`LD_PRELOAD` of system cuDNN 9.24 didn't help), **NOT** libstdc++ (Resolve
  doesn't bundle it; system `GLIBCXX_3.4.25` ≥ needed `3.4.21`), **NOT** protobuf/abseil
  (0 exported from ORT), **NOT** stack size (`ulimit -s 131072` didn't help).
- **Actual cause (confirmed via gdb):** Resolve renders OFX in a **`vfork()`'d worker
  subprocess** (gdb: *"Can not resume the parent process over vfork"*, inferior image =
  `resolve`). ORT's session-create does large heap allocations + thread/global init that are
  **unsafe in a post-`vfork` child** (inherited allocator-lock / thread state). Lightweight
  OFX plugins tolerate it; a heavyweight ML runtime does not → deterministic segfault.

**Proof it is environmental, not our code:** the same ORT build + same DA3 model run fine in
`ort_check` (clean standalone process) **and in Nuke** on the same box (Nuke renders in
in-process threads, never `fork`s). So there is **no in-process workaround** — CPU mode,
`ulimit`, and library preloading were all tried and fail.

**Fix — out-of-process inference helper.** The `.ofx` spawns a small helper *executable*
(`fork`+**`exec`** → a fresh clean process with none of the `vfork`-inherited state) that runs
ORT inference and returns frames via shared memory / pipe. High confidence it works because
`ort_check`/Nuke already prove ORT+model run in a clean process here; it also future-proofs the
same class of host-integration risk (Nuke-on-Linux CUDA coexistence, etc.). This is a dedicated
design effort (helper exe + IPC + refactor across DepthEngine/MoGeEngine/AnyCalibEngine), not a
patch. Everything else is validated: macOS Nuke+Resolve (CoreML), Windows Resolve (CUDA),
Linux Nuke (CUDA); Resolve-on-**Linux** is the one host that needs this.

gdb recipe for next time: `set detach-on-fork off` + `set schedule-multiple on` +
`handle SIGSEGV stop nopass` + `run` (do **not** `catch exec` — Resolve execs `gsettings`
et al. at startup and stops too early).

### Autodesk Flame — host to validate (macOS arm64 + Linux CUDA) (requested 2026-07-15)

Flame supports OFX, so it's a wanted target on both host matrices. The Flame install itself is
vendor/licensed software and is **not tracked in this repo** — this is only a note on the
testing possibility.

Key unknown common to both platforms: **does Flame render OFX in-process (like Nuke → should
just work) or in a `fork`/`vfork`'d worker (like Resolve → would need the out-of-process helper
above)?** Determine that first (gdb / process tree while a render fires), since it dictates
whether the plugin runs as-is or needs the helper. Also check Flame's own bundled runtime/CUDA
for coexistence issues (same analysis as `HOST_COMPATIBILITY.md`).

- **macOS arm64 (CoreML):** Flame 2024+ is Apple-Silicon-native, so an arm64 Flame should load
  the arm64-only `.ofx` (CoreML EP) directly — no rebuild. (Flame 2023 was x86_64 and could not
  load the arm64 plugin; it has been uninstalled.) Just drop the bundle in `/Library/OFX/Plugins`
  and watch Flame's render path.
- **Linux CUDA:** feasible on the vengabus QEMU/libvirt host via IOMMU GPU passthrough. Passthrough
  gives the guest the real GPU, so GL+CUDA interop is native — this avoids the VirtualGL interop
  failure that broke Resolve's viewer. Rocky 8.10 is certified for Flame 2027, so a fresh Rocky
  8.10 VM (leave the working rocky8 alone) can host it; reuse the native-Xorg + x11vnc/VNC display
  pattern from Resolve. Caveats: RTX 3090 is consumer GeForce (not on Autodesk's certified list —
  runs in practice, DKU may warn); pin the kernel to the DKU + install Autodesk's certified NVIDIA
  driver; prefer subscription sign-in (node-locked/ULF licensing is MAC-sensitive); the VM shares
  the single passed-through 3090 with rocky8/windows-11. The v0.9.0 Linux `.ofx` (glibc-2.27 floor)
  is already proven in Nuke on this box, so if Flame loads it in-process it likely just works.

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
