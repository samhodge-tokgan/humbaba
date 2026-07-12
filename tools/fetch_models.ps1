<#
Copyright the openfx-onnx-depthanything3 authors.
SPDX-License-Identifier: Apache-2.0

fetch_models.ps1 — download the plugin's ONNX models into the installed
DepthAnything3.ofx.bundle's Contents\Resources.

The models (~1.3 GB each) ship as SEPARATE GitHub release assets, not inside the
installer, so the installer stays small and under GitHub's 2 GB per-asset limit.
Run once after installing (the installer may also invoke it). Idempotent and
SHA-256-verified: already-correct models are skipped.

Usage:
  .\fetch_models.ps1 [-ResourcesDir <path>] [-Tag models-v1] [-BaseUrl <url>]

With no -ResourcesDir it locates the installed bundle in the standard OFX dirs.
Writing to %CommonProgramFiles%\OFX\Plugins needs an elevated PowerShell.
#>
[CmdletBinding()]
param(
  [string]$ResourcesDir,
  [string]$Tag = $(if ($env:DA3_MODELS_TAG) { $env:DA3_MODELS_TAG } else { "models-v1" }),
  [string]$BaseUrl
)

$ErrorActionPreference = "Stop"
$repo = "samhodge-tokgan/openfx-onnx-depthanything3"
if (-not $BaseUrl) {
  $BaseUrl = if ($env:DA3_MODELS_BASE_URL) { $env:DA3_MODELS_BASE_URL }
             else { "https://github.com/$repo/releases/download/$Tag" }
}

# asset name, installed filename, sha256, bytes
$models = @(
  @{ asset="DA3METRIC-LARGE-dyn.onnx"; name="DA3METRIC-LARGE.onnx"; sha="60e4e27167b1f4b225f433afe187a6fc685d52eb88316334fb10ca6629499b6a"; bytes=1339094191 },
  @{ asset="moge-2-vitb.onnx";          name="moge-2-vitb.onnx";     sha="bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21"; bytes=419411850 },
  @{ asset="anycalib_dist.onnx";        name="anycalib_dist.onnx";   sha="291cbcde52f26feac19445dbbbcad17ce418033c3198aa2adceab005624e1d8b"; bytes=1282939650 }
)

function Find-Resources {
  param([string]$Dir)
  if ($Dir) {
    if ($Dir -like "*\Contents\Resources") { return $Dir }
    if ($Dir -like "*.ofx.bundle")         { return (Join-Path $Dir "Contents\Resources") }
    return $Dir
  }
  $candidates = @(
    (Join-Path $env:CommonProgramFiles "OFX\Plugins"),
    "C:\Program Files\Common Files\OFX\Plugins"
  ) | Select-Object -Unique
  foreach ($d in $candidates) {
    $b = Join-Path $d "DepthAnything3.ofx.bundle"
    if (Test-Path $b) { return (Join-Path $b "Contents\Resources") }
  }
  return $null
}

$res = Find-Resources -Dir $ResourcesDir
if (-not $res) {
  throw "Could not find DepthAnything3.ofx.bundle in the standard OFX dirs. Pass -ResourcesDir <bundle>\Contents\Resources."
}
New-Item -ItemType Directory -Force $res | Out-Null

Write-Host "Installing models into: $res"
Write-Host "Source: $BaseUrl"
foreach ($m in $models) {
  $dest = Join-Path $res $m.name
  if (Test-Path $dest) {
    $h = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
    if ($h -eq $m.sha) { Write-Host "  [skip] $($m.name) (already present, checksum OK)"; continue }
  }
  Write-Host "  [get]  $($m.asset) -> $($m.name) ($($m.bytes) bytes)"
  $tmp = "$dest.part"
  # The repo may be private (release assets need auth). Prefer the GitHub CLI, which
  # handles auth; fall back to a plain/token'd curl of the public download URL.
  $gh = Get-Command gh -ErrorAction SilentlyContinue
  if ($gh -and (gh auth status 2>$null; $LASTEXITCODE -eq 0)) {
    & gh release download $Tag --repo $repo --pattern $m.asset --output $tmp --clobber
    if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $($m.asset)" }
  } else {
    $hdr = @()
    if ($env:GITHUB_TOKEN) { $hdr = @("-H", "Authorization: token $($env:GITHUB_TOKEN)") }
    & curl.exe -fL --retry 3 --progress-bar @hdr -o $tmp "$BaseUrl/$($m.asset)"
    if ($LASTEXITCODE -ne 0) { throw "download failed for $($m.asset)" }
  }
  $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
  if ($got -ne $m.sha) {
    Remove-Item $tmp -Force
    throw "checksum mismatch for $($m.asset): expected $($m.sha), got $got"
  }
  Move-Item -Force $tmp $dest
  Write-Host "  [ok]   $($m.name) verified"
}
Write-Host "Done. All models present in $res"
