// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
//
// Thin wrapper around an ONNX Runtime session running the DA3 metric depth model
// with the CoreML execution provider. Handles preprocessing (resize to the model
// resolution) and postprocessing (resize depth back to the source resolution).
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace da3 {

enum class ComputeUnits { All, CPUAndGPU, CPUAndNeuralEngine, CPUOnly };

struct DepthResult {
  std::vector<float> depth;  // width*height, row-major, metric depth (model units)
  int width = 0;
  int height = 0;
};

class DepthEngine {
 public:
  // Loads `model_path` and creates a session. `procW`/`procH` are the fixed model
  // processing resolution (multiples of 14); keeping them fixed avoids CoreML
  // recompiles between frames. `intra_threads` <= 0 leaves the ORT default.
  DepthEngine(const std::string& model_path, ComputeUnits units, int intra_threads,
              int procW, int procH);
  ~DepthEngine();

  DepthEngine(const DepthEngine&) = delete;
  DepthEngine& operator=(const DepthEngine&) = delete;

  // Runs inference on an RGB image (interleaved, row-major, values in [0,1]) of
  // size in_w x in_h, returning depth resampled back to in_w x in_h.
  DepthResult Run(const float* rgb, int in_w, int in_h);

  // True if the session actually placed nodes on the CoreML EP.
  bool coreml_active() const;

  // Whether "CoreMLExecutionProvider" is compiled into this ORT build.
  static bool CoreMLAvailable();

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

// Round `v` down to the nearest multiple of `m` (>= m).
int RoundToMultiple(int v, int m);

}  // namespace da3
