#!/usr/bin/env bash
# Copyright the humbaba authors.
# SPDX-License-Identifier: Apache-2.0
#
# fetch_models.sh — download the plugin's ONNX models into the installed
# DepthAnything3.ofx.bundle's Contents/Resources.
#
# The models (~1.3 GB each) are shipped as SEPARATE GitHub release assets, not
# inside the installer, so the installer stays small and under GitHub's 2 GB
# per-asset limit. Run this once after installing (the installer may also invoke
# it). It is idempotent and SHA-256-verified: already-correct models are skipped.
#
# Usage:
#   fetch_models.sh [RESOURCES_DIR]     # explicit Contents/Resources target
#   DA3_MODELS_TAG=models-v1 fetch_models.sh
#   DA3_MODELS_BASE_URL=https://host/path fetch_models.sh   # mirror override
#
# With no argument it locates the installed bundle in the standard OFX dirs.
set -euo pipefail

TAG="${DA3_MODELS_TAG:-models-v1}"
REPO="samhodge-tokgan/humbaba"
BASE="${DA3_MODELS_BASE_URL:-https://github.com/${REPO}/releases/download/${TAG}}"

# Manifest: "<release-asset-name> <installed-filename> <sha256> <bytes>"
MODELS=(
  "DA3METRIC-LARGE-dyn.onnx DA3METRIC-LARGE.onnx 60e4e27167b1f4b225f433afe187a6fc685d52eb88316334fb10ca6629499b6a 1339094191"
  "moge-2-vitb.onnx moge-2-vitb.onnx bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21 419411850"
  "anycalib_dist.onnx anycalib_dist.onnx 291cbcde52f26feac19445dbbbcad17ce418033c3198aa2adceab005624e1d8b 1282939650"
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

find_resources() {
  # $1 may be an explicit Resources dir OR a bundle OR empty (auto-detect).
  if [ -n "${1:-}" ]; then
    case "$1" in
      */Contents/Resources) echo "$1"; return 0 ;;
      *.ofx.bundle) echo "$1/Contents/Resources"; return 0 ;;
      *) echo "$1"; return 0 ;;
    esac
  fi
  local dirs=(
    "$HOME/Library/OFX/Plugins" "/Library/OFX/Plugins"          # macOS
    "$HOME/OFX/Plugins" "/usr/OFX/Plugins" "/usr/local/OFX/Plugins"  # Linux
  )
  local d
  for d in "${dirs[@]}"; do
    if [ -d "$d/DepthAnything3.ofx.bundle" ]; then
      echo "$d/DepthAnything3.ofx.bundle/Contents/Resources"; return 0
    fi
  done
  return 1
}

RES="$(find_resources "${1:-}")" || {
  echo "error: could not find DepthAnything3.ofx.bundle in the standard OFX dirs." >&2
  echo "       pass the bundle's Contents/Resources path as an argument." >&2
  exit 1
}

if ! mkdir -p "$RES" 2>/dev/null || [ ! -w "$RES" ]; then
  echo "error: '$RES' is not writable. Re-run with sudo (system install), e.g.:" >&2
  echo "       sudo $0 ${1:-}" >&2
  exit 1
fi

echo "Installing models into: $RES"
echo "Source: $BASE"
for entry in "${MODELS[@]}"; do
  read -r asset name want_sha want_bytes <<<"$entry"
  dest="$RES/$name"
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want_sha" ]; then
    echo "  [skip] $name (already present, checksum OK)"
    continue
  fi
  echo "  [get]  $asset -> $name (${want_bytes} bytes)"
  tmp="$dest.part"
  # The repo is public, so a plain anonymous curl of the release download URL works.
  # Prefer the GitHub CLI when present (nicer progress / handles any future privacy),
  # otherwise curl the public URL (a GITHUB_TOKEN is optional, only for rate limits).
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --output "$tmp" --clobber
  else
    hdr=()
    [ -n "${GITHUB_TOKEN:-}" ] && hdr=(-H "Authorization: token ${GITHUB_TOKEN}")
    curl -fL --retry 3 --progress-bar "${hdr[@]}" -o "$tmp" "$BASE/$asset"
  fi
  got_sha="$(sha256 "$tmp")"
  if [ "$got_sha" != "$want_sha" ]; then
    rm -f "$tmp"
    echo "  error: checksum mismatch for $asset" >&2
    echo "         expected $want_sha" >&2
    echo "         got      $got_sha" >&2
    exit 1
  fi
  mv -f "$tmp" "$dest"
  echo "  [ok]   $name verified"
done
echo "Done. All models present in $RES"
