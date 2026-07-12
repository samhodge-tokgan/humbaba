// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the AnyCalib network (DINOv2 ViT-L) via ONNX Runtime to predict a dense
// per-pixel ray field, then fits an OpenCV radial (Brown-Conrady) camera model to
// that field on the host: DLT linear init + Gauss-Newton refinement (logmap
// angular residual), reproducing AnyCalib's `radial` calibrator. Outputs OpenCV
// intrinsics + k1,k2 (k3=p1=p2=0). AnyCalib is Apache-2.0.
#pragma once

#include <memory>
#include <string>

#include "DepthEngine.h"  // da3::ComputeUnits

namespace da3 {

struct CameraFit {
  bool ok = false;
  double fx = 0, fy = 0, cx = 0, cy = 0;  // pixels, original resolution
  double k1 = 0, k2 = 0;                  // OpenCV radial (focal-normalized)
};

class AnyCalibEngine {
 public:
  AnyCalibEngine(const std::string& model_path, ComputeUnits units, int intra_threads);
  ~AnyCalibEngine();
  AnyCalibEngine(const AnyCalibEngine&) = delete;
  AnyCalibEngine& operator=(const AnyCalibEngine&) = delete;

  // rgb: interleaved RGB in [0,1], size in_w x in_h. Returns OpenCV intrinsics for
  // the original (in_w x in_h) resolution.
  CameraFit Estimate(const float* rgb, int in_w, int in_h);

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace da3
