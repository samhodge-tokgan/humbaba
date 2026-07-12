// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
#include "DepthEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

#include "OrtAccel.h"

namespace da3 {

int RoundToMultiple(int v, int m) {
  if (m <= 0) return v;
  int r = (v / m) * m;
  return r < m ? m : r;
}

namespace {

const char* ComputeUnitsString(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CPUAndGPU: return "CPUAndGPU";
    case ComputeUnits::CPUAndNeuralEngine: return "CPUAndNeuralEngine";
    case ComputeUnits::CPUOnly: return "CPUOnly";
  }
  return "ALL";
}

// Bilinear resample of an interleaved image with `c` channels from
// (sw x sh) to (dw x dh). src/dst are row-major, channel-interleaved.
void ResampleBilinear(const float* src, int sw, int sh, float* dst, int dw, int dh, int c) {
  const float sx = sw > 1 && dw > 1 ? static_cast<float>(sw - 1) / (dw - 1) : 0.f;
  const float sy = sh > 1 && dh > 1 ? static_cast<float>(sh - 1) / (dh - 1) : 0.f;
  for (int y = 0; y < dh; ++y) {
    float fy = y * sy;
    int y0 = static_cast<int>(fy);
    int y1 = std::min(y0 + 1, sh - 1);
    float wy = fy - y0;
    for (int x = 0; x < dw; ++x) {
      float fx = x * sx;
      int x0 = static_cast<int>(fx);
      int x1 = std::min(x0 + 1, sw - 1);
      float wx = fx - x0;
      const float* p00 = src + (static_cast<size_t>(y0) * sw + x0) * c;
      const float* p01 = src + (static_cast<size_t>(y0) * sw + x1) * c;
      const float* p10 = src + (static_cast<size_t>(y1) * sw + x0) * c;
      const float* p11 = src + (static_cast<size_t>(y1) * sw + x1) * c;
      float* d = dst + (static_cast<size_t>(y) * dw + x) * c;
      for (int k = 0; k < c; ++k) {
        float top = p00[k] * (1 - wx) + p01[k] * wx;
        float bot = p10[k] * (1 - wx) + p11[k] * wx;
        d[k] = top * (1 - wy) + bot * wy;
      }
    }
  }
}

}  // namespace

struct DepthEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DepthAnything3"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::string input_name;
  std::vector<std::string> output_names;  // want "depth" (and "sky")
  int procW = 504;
  int procH = 504;
  bool coreml_active = false;
};

DepthEngine::DepthEngine(const std::string& model_path, ComputeUnits units,
                         int intra_threads, int procW, int procH)
    : impl_(std::make_unique<Impl>()) {
  impl_->procW = RoundToMultiple(procW, 14);
  impl_->procH = RoundToMultiple(procH, 14);

  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);

  bool used_coreml = false;
  if (da3::AcceleratorAvailable()) {
    try {
      // CoreML on macOS (MLProgram, static shapes for no per-frame recompiles);
      // CUDA on Linux/Windows. See src/OrtAccel.h.
      da3::AppendAccelerator(so, ComputeUnitsString(units),
                             /*coreml_static=*/true, /*coreml_mlprogram=*/true);
      used_coreml = true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string(da3::AcceleratorSubstr()) + " EP append failed: " + e.what();
      used_coreml = false;
    }
  }

  try {
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), so);
  } catch (const Ort::Exception& e) {
    // The accelerator EP can fail at session-creation time (not at append) if its
    // runtime libraries are missing/incompatible — e.g. on Linux when the CUDA
    // toolkit / cuDNN is not installed. Retry once on a plain CPU session so the
    // plugin still works (slower) instead of failing outright.
    if (used_coreml) {
      last_error_ = std::string(da3::AcceleratorSubstr()) +
                    " session create failed (falling back to CPU): " + e.what();
      Ort::SessionOptions cpu_so;
      cpu_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      if (intra_threads > 0) cpu_so.SetIntraOpNumThreads(intra_threads);
      used_coreml = false;
      try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), cpu_so);
      } catch (const Ort::Exception& e2) {
        last_error_ = std::string("session create failed: ") + e2.what();
        impl_->session.reset();
        return;
      }
    } else {
      last_error_ = std::string("session create failed: ") + e.what();
      impl_->session.reset();
      return;
    }
  }
  impl_->coreml_active = used_coreml;

  Ort::AllocatedStringPtr in = impl_->session->GetInputNameAllocated(0, impl_->alloc);
  impl_->input_name = in.get();
  size_t nout = impl_->session->GetOutputCount();
  for (size_t i = 0; i < nout; ++i) {
    Ort::AllocatedStringPtr on = impl_->session->GetOutputNameAllocated(i, impl_->alloc);
    impl_->output_names.emplace_back(on.get());
  }
}

DepthEngine::~DepthEngine() = default;

bool DepthEngine::coreml_active() const { return impl_ && impl_->coreml_active; }

bool DepthEngine::CoreMLAvailable() { return da3::AcceleratorAvailable(); }

DepthResult DepthEngine::Run(const float* rgb, int in_w, int in_h) {
  DepthResult out;
  if (!impl_ || !impl_->session) return out;

  const int pw = impl_->procW, ph = impl_->procH;

  // 1. Downsample RGB to the model resolution, then pack into NCHW.
  std::vector<float> resized(static_cast<size_t>(pw) * ph * 3);
  ResampleBilinear(rgb, in_w, in_h, resized.data(), pw, ph, 3);

  std::vector<float> nchw(static_cast<size_t>(pw) * ph * 3);
  const size_t plane = static_cast<size_t>(pw) * ph;
  for (int y = 0; y < ph; ++y) {
    for (int x = 0; x < pw; ++x) {
      const float* src = resized.data() + (static_cast<size_t>(y) * pw + x) * 3;
      size_t idx = static_cast<size_t>(y) * pw + x;
      nchw[0 * plane + idx] = src[0];
      nchw[1 * plane + idx] = src[1];
      nchw[2 * plane + idx] = src[2];
    }
  }

  // 2. Run.
  std::array<int64_t, 4> shape{1, 3, ph, pw};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(mem, nchw.data(), nchw.size(),
                                                     shape.data(), shape.size());
  const char* in_names[] = {impl_->input_name.c_str()};
  const char* out_names[] = {"depth"};
  std::vector<Ort::Value> results;
  try {
    results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("inference failed: ") + e.what();
    return out;
  }
  if (results.empty()) return out;

  // 3. Read depth [1,ph,pw] (or [1,1,ph,pw]) and upsample to source size.
  float* depth = results[0].GetTensorMutableData<float>();
  std::vector<float> depth_up(static_cast<size_t>(in_w) * in_h);
  ResampleBilinear(depth, pw, ph, depth_up.data(), in_w, in_h, 1);

  out.depth = std::move(depth_up);
  out.width = in_w;
  out.height = in_h;
  return out;
}

}  // namespace da3
