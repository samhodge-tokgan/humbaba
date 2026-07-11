// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the MoGe-2 ONNX model (ONNX Runtime + CoreML) to estimate camera focal
// length / field of view from a single image. MoGe's ONNX graph outputs an
// affine-invariant point map + mask; the focal (and z-shift) are recovered here
// on the host via a small reprojection least-squares (ported from microsoft/MoGe
// recover_focal_shift), then converted to pixels/FOV. Model + code are MIT.
#pragma once

#include <memory>
#include <string>

#include "DepthEngine.h"  // for da3::ComputeUnits

namespace da3 {

struct FocalResult {
  bool ok = false;
  double focal_norm = 1.0;   // MoGe normalized (diagonal) focal
  double shift = 0.0;        // recovered z-shift
  double fx_px = 0.0, fy_px = 0.0;   // focal in pixels for (out_w x out_h)
  double cx_px = 0.0, cy_px = 0.0;   // principal point (image centre)
  double fov_x_deg = 0.0, fov_y_deg = 0.0;
  int out_w = 0, out_h = 0;  // resolution the pixel values refer to
};

class MoGeEngine {
 public:
  // num_tokens controls MoGe's internal working resolution (default 1800, range
  // ~[1200,3600]). cap_long caps the ONNX input long side for speed (focal only
  // needs a 64x64 solve). intra_threads <= 0 leaves the ORT default.
  MoGeEngine(const std::string& model_path, ComputeUnits units, int intra_threads,
             int num_tokens, int cap_long);
  ~MoGeEngine();
  MoGeEngine(const MoGeEngine&) = delete;
  MoGeEngine& operator=(const MoGeEngine&) = delete;

  // rgb: interleaved RGB in [0,1], size in_w x in_h. focal_px is reported for
  // (report_w, report_h) — usually the original source resolution.
  FocalResult EstimateFocal(const float* rgb, int in_w, int in_h,
                            int report_w, int report_h, double fov_x_deg_known = 0.0);

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace da3
