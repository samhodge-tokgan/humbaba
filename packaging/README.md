# Packaging & installation

## Building the installer

The release `.pkg` bundles everything needed (plugin binary + ONNX Runtime dylib +
the ~1.3 GB ONNX model) into `DepthAnything3.ofx.bundle` and installs it into
**`/Library/OFX/Plugins`** (the system OFX path scanned by hosts).

```sh
# 1. Build the plugin with the model bundled into Contents/Resources
cmake -S . -B build -DDA3_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release \
      -DDA3_MODEL_FILE=/path/to/DA3METRIC-LARGE.onnx
cmake --build build -j

# 2. Ad-hoc sign + package
packaging/make_pkg.sh --bundle build/DepthAnything3.ofx.bundle \
      --out dist/DepthAnything3-0.1.0.pkg
```

CI does this automatically — see `.github/workflows/release.yml` (triggered on a
`v*` tag or manual dispatch): it exports the model on Linux, builds + packages on
macOS, and attaches the `.pkg` to the GitHub Release.

## Installing

Double-click the `.pkg`, or:

```sh
sudo installer -pkg DepthAnything3-0.1.0.pkg -target /
```

The bundle lands at `/Library/OFX/Plugins/DepthAnything3.ofx.bundle`. Launch your
OFX host; the plugin appears under **TokGan › Depth Anything 3**. The model is found
automatically inside the bundle (no path configuration needed).

## Gatekeeper (unsigned build)

Until the project has an Apple Developer ID, the plugin and installer are **ad-hoc
signed** (not notarized). macOS may warn on first use. To allow it:

- Approve it in **System Settings › Privacy & Security** after the first block, or
- Remove the quarantine attribute before installing:
  ```sh
  xattr -dr com.apple.quarantine DepthAnything3-0.1.0.pkg
  ```
- If a hardened host refuses to load the bundled dylib, it must be signed by a
  consistent Team ID — this is resolved once a Developer ID is wired into
  `make_pkg.sh --identity` / `--pkg-identity` and notarization is added.

## Notes

- The installer targets `/Library/OFX/Plugins` (requires admin) because some host
  builds do not scan `~/Library/OFX/Plugins` by default.
- To point a host at a dev build without installing, set
  `OFX_PLUGIN_PATH=/path/to/dir/containing/the/bundle`.
