# Windows (x64, CUDA) build & deployment

The plugin builds as a native `x64` `.ofx` DLL that runs Depth Anything 3 on
**NVIDIA CUDA** via ONNX Runtime. Verified on **Windows 11** with two RTX 3090s
(CUDA 12.8, cuDNN 9.8, MSVC 19.44 / Visual Studio 2022). Accelerator selection is
automatic — CoreML on macOS, CUDA on Windows/Linux — via [`src/OrtAccel.h`](../src/OrtAccel.h);
if CUDA (or its runtime libs) is unavailable the engines fall back to CPU.

## Build prerequisites

- **Visual Studio 2022** (Community or Build Tools) with the **Desktop C++** workload
  (MSVC x64 toolset). The bundled CMake (3.31) and Ninja are used automatically.
- **CUDA Toolkit 12.x** (for the ONNX Runtime CUDA package the build fetches:
  `onnxruntime-win-x64-gpu_cuda12`). Building needs only the toolkit headers/import
  libs; cuDNN is a *runtime* dependency (see below).

```powershell
# From any shell (the VS generator locates MSVC itself — no vcvars needed):
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DDA3_WITH_ONNX=ON
& $cmake --build build --config Release --target ort_check DepthAnything3 --parallel
```

The build assembles `build\DepthAnything3.ofx.bundle` with this layout:

```
DepthAnything3.ofx.bundle\
  Contents\
    Win64\                              # OFX-spec arch dir for 64-bit Windows
      DepthAnything3.ofx                # the plugin DLL (exports OfxGetNumberOfPlugins/OfxGetPlugin)
      onnxruntime_da3.dll               # our ONNX Runtime, PRIVATELY renamed (see below)
      onnxruntime_providers_cuda.dll
      onnxruntime_providers_tensorrt.dll
      onnxruntime_providers_shared.dll
    Resources\                          # optional: bundled *.onnx models (release build)
```

### How the plugin finds — and isolates — its bundled ONNX Runtime

Windows does **not** search a `LoadLibrary`'d module's own directory for that
module's dependencies, so a normally-linked `onnxruntime.dll` next to the `.ofx`
would fail to resolve when a host loads the plugin. We solve this without touching
the host's DLL search order:

- The `.ofx` **delay-loads** `onnxruntime.dll` (`/DELAYLOAD:onnxruntime.dll`).
- [`src/WinLoader.cpp`](../src/WinLoader.cpp) installs a delay-load hook that loads
  our runtime by **explicit full path** from the plugin's own directory
  (`Contents\Win64`) with `LOAD_WITH_ALTERED_SEARCH_PATH`. ORT then loads its
  provider DLLs (which are located relative to its own module dir) itself.

**Private name — critical (fixed in 0.7.0).** The bundled runtime is renamed to
`onnxruntime_da3.dll` and the hook loads *that*, never the bare `onnxruntime.dll`.
The reason: **Windows 11 ships its own `onnxruntime.dll` in `System32`** (part of
Windows ML — version `1.17.x`). Windows keys loaded modules by **base name**, so if
any host component makes the OS copy resident first (Nuke triggers Windows ML), a
`LoadLibraryExW` of our full path to a file *also* named `onnxruntime.dll` just
returns the already-resident **OS** module — binding the plugin to ORT 1.17 instead
of ours. Our API-27 calls then fail (`requested API version [27] … only [1,17]
supported`) or crash. A unique base name can never be deduplicated against the OS
copy; the two runtimes coexist as distinct modules. (macOS/Linux use the same
private-naming trick via install-name / soname — see `OFX_LIB_BUNDLED` in CMake.)

The bundle-relative model lookup in the plugins uses
`GetModuleHandleExW`/`GetModuleFileNameA` (the Windows analogue of `dladdr`).

### Runtime requirement: Visual C++ 2015–2022 redistributable ≥ 14.40

The prebuilt ONNX Runtime 1.27 is linked against the **dynamic** MSVC C++ runtime
(`MSVCP140.dll` / `VCRUNTIME140*.dll`) and requires version **≥ 14.40** (VS 2022
17.10). A too-old runtime makes `onnxruntime_da3.dll` fail to initialize with
`LoadLibrary` error **1114** (`ERROR_DLL_INIT_FAILED`). Most machines already have a
current redist in `System32`; if not, install the latest
[VC++ redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

> **Nuke 16.1 caveat (known limitation).** Nuke bundles its own **14.36** VC runtime
> in its application directory (`C:\Program Files\Nuke16.1v3\MSVCP140.dll`, …). Because
> the app directory wins the DLL search order, that 14.36 copy becomes resident
> process-wide, and there can only be one `MSVCP140.dll` per process — so even a
> current `System32` redist is shadowed, and ORT 1.27 fails to init (error 1114).
> The plugin works in Nuke on macOS (CoreML, no MSVC runtime) and in DaVinci Resolve
> on Windows (ships a current runtime); only **Nuke-on-Windows** hits this.
> Workarounds until a static-CRT ORT build lands (tracked in [BACKLOG](BACKLOG.md)):
> rename Nuke's bundled `msvcp140*.dll` / `vcruntime140*.dll` / `concrt140.dll` /
> `vccorlib140.dll` aside so Nuke uses the `System32` 14.44 copies (reversible; only
> for names `System32` also provides), or use a Nuke build that ships a newer runtime.

## Runtime requirement: cuDNN 9 (host-provided)

The CUDA provider needs **cuDNN 9** (for CUDA 12) at runtime. As on Linux, this is
**not bundled** (cuDNN is large). Install it where the loader finds it — the simplest
is the CUDA `bin` directory, which is on `PATH`:

```powershell
# Public NVIDIA redist (no login required):
$u = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip"
curl.exe -L -o $env:TEMP\cudnn.zip $u
tar.exe -xf $env:TEMP\cudnn.zip -C $env:TEMP
Copy-Item "$env:TEMP\cudnn-windows-x86_64-9.8.0.87_cuda12-archive\bin\*.dll" `
  -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin" -Force
```

> Putting the cuDNN DLLs only *inside the bundle* is **not** enough: cuDNN 9 is split
> across several DLLs, and once the host loads the first one, its siblings are resolved
> via the default search order (the host's exe dir + `System32` + `PATH`), not the
> bundle. The CUDA `bin` on `PATH` covers all of them. If cuDNN is missing, the depth
> engine falls back to CPU.

## Install location

Copy the bundle to any OFX plugin path (the standard machine-wide location is
`%CommonProgramFiles%\OFX\Plugins`), or point `OFX_PLUGIN_PATH` at a directory that
contains it:

```powershell
Copy-Item -Recurse build\DepthAnything3.ofx.bundle "$env:CommonProgramFiles\OFX\Plugins\"
# or:
$env:OFX_PLUGIN_PATH = "C:\path\to\dir-containing-the-bundle"
```

## Models

Release installers are model-less (to stay under GitHub's 2 GB asset limit). Pull the
models into the installed bundle's `Contents\Resources\` with the bundled fetch script
(SHA-256-verified, idempotent) — use an **elevated** PowerShell for the system OFX dir:

```powershell
.\fetch_models.ps1
```

Or point the plugin's `modelFile` param at an existing `.onnx`.

## Verification

Standalone GPU smoke test (no host):

```powershell
& build\Release\ort_check.exe C:\path\to\DA3METRIC-LARGE-dyn.onnx
# lists providers (CUDAExecutionProvider present), creates a CUDA session, runs one
# inference. Reference: 504x504 DA3 ~92 ms on an RTX 3090. (ort_check needs the ORT
# DLLs — copied next to it by the build — and cuDNN on PATH.)
```

End-to-end in a host (headless Natron):

```powershell
$env:OFX_PLUGIN_PATH = "C:\dev\da3\build"
$env:DA3_INPUT = "C:\dev\da3\test-assets\synthetic_test.png"
$env:DA3_OUTPUT = "C:\dev\da3\out\depth.exr"
$env:DA3_MODEL_PATH = "C:\path\to\DA3METRIC-LARGE-dyn.onnx"
$env:DA3_ACESCG = "0"; $env:DA3_PROCRES = "504"
& "C:\...\Natron\bin\NatronRenderer.exe" --clear-openfx-cache -t tests\natron\render_depth.py
```

Expect `RESULT: PASS` and a full-size float32 depth EXR (decimeters). Verified: all
three plugins load; DA3 renders a 1920x1080 metric-depth EXR matching the Linux/CUDA
and macOS/CoreML output (depth mean ~9.4 dm on the synthetic test).

### Known issue

With Natron 2.5.0, the process can raise an access violation **during shutdown**
(after the render has finished and the output file is written) — a teardown/ordering
crash between the host and the CUDA/ORT runtime. It does not affect the rendered
output. Under investigation.
