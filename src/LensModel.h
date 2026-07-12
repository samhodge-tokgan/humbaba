// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic lens-distortion math: OpenCV Brown-Conrady distort/undistort,
// overscan computation, and OpenCV -> 3DEqualizer parameter conversion. No ML;
// this drives the plugin's overscan + downstream-parameter outputs.
#pragma once

namespace da3 {

// OpenCV Brown-Conrady, focal-normalized (x=(u-cx)/fx). Principal point at centre.
struct BrownConrady {
  double k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
};

// Forward: undistorted normalized (x,y) -> distorted normalized (xd,yd).
void DistortNorm(const BrownConrady& d, double x, double y, double& xd, double& yd);

// Inverse: distorted normalized (xd,yd) -> undistorted normalized (x,y), iterative
// (OpenCV undistortPoints method).
void UndistortNorm(const BrownConrady& d, double xd, double yd, double& x, double& y);

// Overscan needed to render CG that will be re-distorted to cover the plate: the
// plate (distorted) boundary is undistorted; the bounding box beyond [0,W]x[0,H]
// is the padding. Samples the whole boundary (not just corners), since the radial
// extremum can lie at edge midpoints.
struct Overscan {
  double left = 0, right = 0, top = 0, bottom = 0;  // pixels (>=0 = expansion)
  double pct_x = 0, pct_y = 0;                       // fractional width/height growth
  double und_w = 0, und_h = 0;                       // undistorted bbox size (px)
};
Overscan ComputeOverscan(const BrownConrady& d, int W, int H, double fx, double fy,
                         double cx, double cy, int samples_per_edge = 33);

// OpenCV -> 3DE4 "Radial - Standard, Degree 4" (diagonally-normalized). c2/c4 are
// the UI "Distortion"/"Quartic Distortion"; decentering u1/v1 map from p2/p1.
// Radius rescale s = (half-diagonal px)/focal px: c2 = k1*s^2, c4 = k2*s^4.
struct TDE4Radial {
  double distortion_c2 = 0;
  double quartic_c4 = 0;
  double u1 = 0;  // decentering (from p2)
  double v1 = 0;  // decentering (from p1)
};
TDE4Radial OpenCVToTDE4(const BrownConrady& d, int W, int H, double fx, double fy);

}  // namespace da3
