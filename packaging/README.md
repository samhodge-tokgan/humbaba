# Packaging & installation

The plugin ships a per-platform installer, all built by the release workflow
(`.github/workflows/release.yml`, triggered on a `v*` tag or manual dispatch):

| Platform | Installer | Installs / extracts to |
|----------|-----------|------------------------|
| **macOS** (Apple Silicon) | `DepthAnything3-<ver>-macos-arm64.pkg` | `/Library/OFX/Plugins` |
| **Linux** (x86-64) | `DepthAnything3-<ver>-linux-x86_64.tar.xz` | `/usr/OFX/Plugins` or `~/OFX/Plugins` |
| **Windows** (x64) | `DepthAnything3-<ver>-windows-x64.zip` | `%CommonProgramFiles%\OFX\Plugins` |

## Models are fetched, not bundled

The ONNX models (~1.3 GB for DA3, plus MoGe and AnyCalib) would push each installer past GitHub's
**2 GB per-asset** release limit, so the installers are **model-less**. Each carries a
`fetch_models` script in the bundle's `Contents/`, and the models are published as a separate,
independently-versioned set of release assets under the **`models-v1`** tag, pinned by SHA-256
inside the script. After installing, run the script once to populate `Contents/Resources`:

```sh
# macOS / Linux
bash /Library/OFX/Plugins/DepthAnything3.ofx.bundle/Contents/fetch_models.sh
# Windows (PowerShell)
powershell -File "%CommonProgramFiles%\OFX\Plugins\DepthAnything3.ofx.bundle\Contents\fetch_models.ps1"
```

The repo is **public**, so the scripts download the models with a plain anonymous `curl` of the
release URL — no auth needed. (They still use `gh release download` when the CLI is present, and a
`GITHUB_TOKEN` is optional, only to lift API rate limits.) Override the model set with
`DA3_MODELS_TAG` / `DA3_MODELS_BASE_URL`. You can also skip the fetch entirely and point the
plugin's *Model file* parameter (or `DA3_MODEL_PATH`) at a model you already have.

## Building an installer locally

The same CMake project builds a **model-less** bundle on every platform (do **not** pass
`-DDA3_*_MODEL_FILE`); the fetch script supplies the model at install time.

```sh
# macOS — build, then ad-hoc sign + package into a .pkg
cmake -S . -B build -DDA3_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target DepthAnything3 -j
packaging/make_pkg.sh --bundle build/DepthAnything3.ofx.bundle \
      --out dist/DepthAnything3-0.5.0-macos-arm64.pkg --version 0.5.0 \
      --extra-contents tools/fetch_models.sh          # injected AFTER code-signing (see below)

# Linux — build, then tar the bundle (carry the fetch script in Contents/)
cmake -S . -B build -DDA3_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target DepthAnything3 -j"$(nproc)"
cp tools/fetch_models.sh build/DepthAnything3.ofx.bundle/Contents/
tar -C build -caf dist/DepthAnything3-0.5.0-linux-x86_64.tar.xz DepthAnything3.ofx.bundle

# Windows — build, then zip the bundle
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DDA3_WITH_ONNX=ON
cmake --build build --config Release --target DepthAnything3 --parallel
Copy-Item tools/fetch_models.ps1 build/DepthAnything3.ofx.bundle/Contents/
Compress-Archive build/DepthAnything3.ofx.bundle dist/DepthAnything3-0.5.0-windows-x64.zip
```

### macOS code-signing note (`make_pkg.sh`)

`make_pkg.sh` ad-hoc signs the bundled ONNX Runtime dylib(s) and the plugin binary, then packages
the bundle. Signing the bundle's main executable puts `codesign` into **bundle-seal mode**, so at
sign time `Contents/` must hold only normal bundle structure — a loose file there (e.g.
`fetch_models.sh`) is unsignable "nested code" and fails the sign. Pass such files via
**`--extra-contents`** so they are injected into the staged copy *after* signing (they ride along
in the payload, intentionally outside the signature seal — fine for an ad-hoc distribution). The
bundled ORT is privately renamed by the isolation step (`libonnxruntime_da3.*`), so `make_pkg.sh`
globs `Contents/Frameworks/*.dylib` rather than hardcoding a name.

## Installing

- **macOS:** double-click the `.pkg`, or `sudo installer -pkg <file>.pkg -target /`. Lands at
  `/Library/OFX/Plugins/DepthAnything3.ofx.bundle`.
- **Linux:** `tar -xaf <file>.tar.xz -C ~/OFX/Plugins` (create the dir if needed), or extract to
  `/usr/OFX/Plugins` (system-wide).
- **Windows:** extract the `.zip` into `%CommonProgramFiles%\OFX\Plugins`.

Then run the `fetch_models` script (above), launch your OFX host, and the plugins appear under
**TokGan › Depth Anything 3** (plus MoGe Focal, Lens Distortion).

**GPU prerequisite (Linux/Windows):** the CUDA execution provider needs an NVIDIA driver, the
**CUDA 12.x runtime**, and **cuDNN 9** on the loader path (see [`../docs/LINUX.md`](../docs/LINUX.md) /
[`../docs/WINDOWS.md`](../docs/WINDOWS.md)). Without them the plugin still loads and runs on **CPU**.

## Gatekeeper (unsigned macOS build)

Until the project has an Apple Developer ID, the macOS plugin and installer are **ad-hoc signed**
(not notarized). macOS may warn on first use. To allow it:

- Approve it in **System Settings › Privacy & Security** after the first block, or
- Remove the quarantine attribute: `xattr -dr com.apple.quarantine <file>.pkg`.
- A hardened host that refuses the bundled dylib needs a consistent Team ID — resolved once a
  Developer ID is wired into `make_pkg.sh --identity` / `--pkg-identity` and notarization is added.

## Notes

- macOS targets `/Library/OFX/Plugins` (requires admin) because some host builds do not scan
  `~/Library/OFX/Plugins` by default.
- To point a host at a dev build without installing, set
  `OFX_PLUGIN_PATH=/path/to/dir/containing/the/bundle`.
- Host coexistence / library isolation (why the ORT is privately renamed and only the OFX entry
  points are exported) is documented in [`../docs/HOST_COMPATIBILITY.md`](../docs/HOST_COMPATIBILITY.md).
