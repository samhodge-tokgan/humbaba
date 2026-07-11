// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
#include "LensModel.h"

#include <algorithm>
#include <cmath>

namespace da3 {

void DistortNorm(const BrownConrady& d, double x, double y, double& xd, double& yd) {
  double r2 = x * x + y * y;
  double radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2;
  double dx = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x);
  double dy = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y;
  xd = x * radial + dx;
  yd = y * radial + dy;
}

void UndistortNorm(const BrownConrady& d, double xd, double yd, double& x, double& y) {
  x = xd;
  y = yd;
  for (int it = 0; it < 30; ++it) {
    double r2 = x * x + y * y;
    double radial = 1.0 / (1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2);
    double dx = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x);
    double dy = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y;
    x = (xd - dx) * radial;
    y = (yd - dy) * radial;
  }
}

Overscan ComputeOverscan(const BrownConrady& d, int W, int H, double fx, double fy,
                         double cx, double cy, int samples_per_edge) {
  double xmin = 0, ymin = 0, xmax = static_cast<double>(W), ymax = static_cast<double>(H);
  const int n = std::max(2, samples_per_edge);
  auto sample = [&](double u, double v) {
    double xd = (u - cx) / fx, yd = (v - cy) / fy;
    double x, y;
    UndistortNorm(d, xd, yd, x, y);
    double uu = x * fx + cx, vv = y * fy + cy;
    xmin = std::min(xmin, uu); xmax = std::max(xmax, uu);
    ymin = std::min(ymin, vv); ymax = std::max(ymax, vv);
  };
  for (int i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / (n - 1);
    sample(t * W, 0.0);          // top edge
    sample(t * W, H);            // bottom edge
    sample(0.0, t * H);          // left edge
    sample(W, t * H);            // right edge
  }
  Overscan o;
  o.left = std::max(0.0, -xmin);
  o.right = std::max(0.0, xmax - W);
  o.top = std::max(0.0, -ymin);
  o.bottom = std::max(0.0, ymax - H);
  o.und_w = xmax - xmin;
  o.und_h = ymax - ymin;
  o.pct_x = W > 0 ? o.und_w / W - 1.0 : 0.0;
  o.pct_y = H > 0 ? o.und_h / H - 1.0 : 0.0;
  return o;
}

TDE4Radial OpenCVToTDE4(const BrownConrady& d, int W, int H, double fx, double fy) {
  // 3DE uses radius normalized by the half filmback diagonal; OpenCV by focal.
  // s = (half-diagonal px) / focal px ; a coeff on r^n scales by s^n.
  double f = 0.5 * (fx + fy);
  double half_diag = 0.5 * std::sqrt(static_cast<double>(W) * W + static_cast<double>(H) * H);
  double s = f > 0 ? half_diag / f : 1.0;
  TDE4Radial t;
  t.distortion_c2 = d.k1 * s * s;
  t.quartic_c4 = d.k2 * s * s * s * s;
  t.u1 = d.p2 * s * s;  // decentering; approximate mapping
  t.v1 = d.p1 * s * s;
  return t;
}

}  // namespace da3
