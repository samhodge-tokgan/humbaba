#!/usr/bin/env bash
# Build a macOS .pkg installer for the DepthAnything3 OFX plugin.
#
# Ad-hoc signs the bundled dylib + plugin binary (no Apple Developer ID yet), then
# packages the .ofx.bundle to install into /Library/OFX/Plugins (the system OFX
# path scanned by hosts). Notarization is deferred until a Developer ID is available.
#
# Usage:
#   packaging/make_pkg.sh --bundle build/DepthAnything3.ofx.bundle \
#       --out dist/DepthAnything3-0.1.0.pkg [--identity "Developer ID Installer: ..."]
set -euo pipefail

IDENTIFIER="com.tokgan.openfx.DepthAnything3"
VERSION="0.1.0"
INSTALL_LOCATION="/Library/OFX/Plugins"
BUNDLE=""
OUT=""
CODESIGN_ID="-"        # "-" = ad-hoc
PKG_SIGN_ID=""         # empty = unsigned pkg

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --identity) CODESIGN_ID="$2"; shift 2;;
    --pkg-identity) PKG_SIGN_ID="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$BUNDLE" && -d "$BUNDLE" ]] || { echo "need --bundle <dir>"; exit 2; }
[[ -n "$OUT" ]] || { echo "need --out <pkg>"; exit 2; }

BIN="$BUNDLE/Contents/MacOS/DepthAnything3.ofx"
DYLIB="$BUNDLE/Contents/Frameworks/libonnxruntime.1.dylib"

echo "== codesign (identity: $CODESIGN_ID) =="
# Sign inner-to-outer: dylib, then the plugin binary. install_name_tool was already
# run at build; signing must come last so signatures stay valid.
[[ -f "$DYLIB" ]] && codesign --force --sign "$CODESIGN_ID" --timestamp=none "$DYLIB"
codesign --force --sign "$CODESIGN_ID" --timestamp=none "$BIN"
codesign --verify --verbose=2 "$BIN" || true

echo "== stage =="
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$BUNDLE" "$STAGE/"

mkdir -p "$(dirname "$OUT")"

echo "== pkgbuild =="
PKG_ARGS=(--root "$STAGE" --identifier "$IDENTIFIER" --version "$VERSION"
          --install-location "$INSTALL_LOCATION")
if [[ -n "$PKG_SIGN_ID" ]]; then
  PKG_ARGS+=(--sign "$PKG_SIGN_ID")
fi
pkgbuild "${PKG_ARGS[@]}" "$OUT"

echo "== done =="
ls -lh "$OUT"
echo "Installs $(basename "$BUNDLE") into $INSTALL_LOCATION"
if [[ "$CODESIGN_ID" == "-" ]]; then
  echo "NOTE: ad-hoc signed + unsigned pkg. Users may need to allow it in System"
  echo "Settings > Privacy & Security, or run: xattr -dr com.apple.quarantine \"$OUT\""
fi
