// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
#include "MoGeEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "OrtAccel.h"

namespace da3 {

namespace {

const char* ComputeUnitsStr(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CPUAndGPU: return "CPUAndGPU";
    case ComputeUnits::CPUAndNeuralEngine: return "CPUAndNeuralEngine";
    case ComputeUnits::CPUOnly: return "CPUOnly";
  }
  return "ALL";
}

void ResampleBilinearRGB(const float* src, int sw, int sh, float* dst, int dw, int dh) {
  const float sx = sw > 1 && dw > 1 ? static_cast<float>(sw - 1) / (dw - 1) : 0.f;
  const float sy = sh > 1 && dh > 1 ? static_cast<float>(sh - 1) / (dh - 1) : 0.f;
  for (int y = 0; y < dh; ++y) {
    float fy = y * sy;
    int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, sh - 1);
    float wy = fy - y0;
    for (int x = 0; x < dw; ++x) {
      float fx = x * sx;
      int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, sw - 1);
      float wx = fx - x0;
      const float* p00 = src + (static_cast<size_t>(y0) * sw + x0) * 3;
      const float* p01 = src + (static_cast<size_t>(y0) * sw + x1) * 3;
      const float* p10 = src + (static_cast<size_t>(y1) * sw + x0) * 3;
      const float* p11 = src + (static_cast<size_t>(y1) * sw + x1) * 3;
      float* d = dst + (static_cast<size_t>(y) * dw + x) * 3;
      for (int c = 0; c < 3; ++c) {
        float top = p00[c] * (1 - wx) + p01[c] * wx;
        float bot = p10[c] * (1 - wx) + p11[c] * wx;
        d[c] = top * (1 - wy) + bot * wy;
      }
    }
  }
}

// MoGe normalized_view_plane_uv: diagonal-normalized plane, principal point at 0.
void NormalizedUV(int W, int H, std::vector<float>& u, std::vector<float>& v) {
  const double ar = static_cast<double>(W) / H;
  const double span_x = ar / std::sqrt(1.0 + ar * ar);
  const double span_y = 1.0 / std::sqrt(1.0 + ar * ar);
  u.resize(static_cast<size_t>(W) * H);
  v.resize(static_cast<size_t>(W) * H);
  for (int x = 0; x < W; ++x) {
    double uu = W > 1 ? (-span_x * (W - 1) / W + (2.0 * span_x * (W - 1) / W) * x / (W - 1)) : 0.0;
    for (int y = 0; y < H; ++y) {
      double vv = H > 1 ? (-span_y * (H - 1) / H + (2.0 * span_y * (H - 1) / H) * y / (H - 1)) : 0.0;
      u[static_cast<size_t>(y) * W + x] = static_cast<float>(uu);
      v[static_cast<size_t>(y) * W + x] = static_cast<float>(vv);
    }
  }
}

// Objective E(shift): folds out the closed-form focal for a given shift.
double CostAndFocal(const std::vector<double>& X, const std::vector<double>& Y,
                    const std::vector<double>& Z, const std::vector<double>& U,
                    const std::vector<double>& V, double shift, double* focal_out) {
  double num = 0, den = 0;
  const size_t n = X.size();
  for (size_t i = 0; i < n; ++i) {
    double zp = Z[i] + shift;
    if (std::fabs(zp) < 1e-6) { if (focal_out) *focal_out = 0; return 1e30; }
    double px = X[i] / zp, py = Y[i] / zp;
    num += px * U[i] + py * V[i];
    den += px * px + py * py;
  }
  double f = den > 0 ? num / den : 0.0;
  if (focal_out) *focal_out = f;
  double e = 0;
  for (size_t i = 0; i < n; ++i) {
    double zp = Z[i] + shift;
    double px = X[i] / zp, py = Y[i] / zp;
    double rx = f * px - U[i], ry = f * py - V[i];
    e += rx * rx + ry * ry;
  }
  return e;
}

}  // namespace

struct MoGeEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "MoGe"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::string image_input, tokens_input;
  int num_tokens = 1800;
  int cap_long = 512;
};

MoGeEngine::MoGeEngine(const std::string& model_path, ComputeUnits units,
                       int intra_threads, int num_tokens, int cap_long)
    : impl_(std::make_unique<Impl>()) {
  impl_->num_tokens = num_tokens > 0 ? num_tokens : 1800;
  impl_->cap_long = cap_long > 63 ? cap_long : 512;

  // Try CoreML, then fall back to CPU. MoGe's graph (num_tokens-driven dynamic
  // reshapes) can fail CoreML's MLModel build entirely, so the fallback is required.
  auto tryCreate = [&](bool use_accel) -> bool {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);
    if (use_accel && da3::AcceleratorAvailable()) {
      try {
        // No static-shape/MLProgram constraint: MoGe's graph is dynamic (num_tokens).
        da3::AppendAccelerator(so, ComputeUnitsStr(units),
                               /*coreml_static=*/false, /*coreml_mlprogram=*/false);
      } catch (const Ort::Exception&) {
        return false;
      }
    }
    try {
      impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), so);
      return true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string("MoGe session create failed: ") + e.what();
      impl_->session.reset();
      return false;
    }
  };
#ifdef __APPLE__
  // MoGe's dynamic (num_tokens-driven) graph is not CoreML-executable (the MLProgram
  // build fails and NeuralNetwork partitions error at runtime), so run it on CPU.
  // It's an on-demand analysis, not a per-frame effect, so CPU latency is acceptable.
  (void)units;
  if (!tryCreate(false)) {
    return;
  }
#else
  // On Linux/Windows the CUDA EP handles the dynamic graph; fall back to CPU if not.
  if (!tryCreate(true) && !tryCreate(false)) {
    return;
  }
#endif
  last_error_.clear();
  // Identify the image vs num_tokens inputs by name.
  size_t nin = impl_->session->GetInputCount();
  for (size_t i = 0; i < nin; ++i) {
    Ort::AllocatedStringPtr nm = impl_->session->GetInputNameAllocated(i, impl_->alloc);
    std::string name = nm.get();
    if (name.find("token") != std::string::npos) impl_->tokens_input = name;
    else impl_->image_input = name;
  }
}

MoGeEngine::~MoGeEngine() = default;

FocalResult MoGeEngine::EstimateFocal(const float* rgb, int in_w, int in_h,
                                      int report_w, int report_h, double fov_x_deg_known) {
  FocalResult out;
  if (!impl_ || !impl_->session) return out;

  // Cap the working resolution (aspect preserved) for speed.
  int W = in_w, H = in_h;
  const int mx = std::max(in_w, in_h);
  std::vector<float> work;
  const float* img = rgb;
  if (mx > impl_->cap_long) {
    const double s = static_cast<double>(impl_->cap_long) / mx;
    W = std::max(1, static_cast<int>(std::lround(in_w * s)));
    H = std::max(1, static_cast<int>(std::lround(in_h * s)));
    work.resize(static_cast<size_t>(W) * H * 3);
    ResampleBilinearRGB(rgb, in_w, in_h, work.data(), W, H);
    img = work.data();
  }

  // Build NCHW input.
  std::vector<float> nchw(static_cast<size_t>(W) * H * 3);
  const size_t plane = static_cast<size_t>(W) * H;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float* s = img + (static_cast<size_t>(y) * W + x) * 3;
      size_t idx = static_cast<size_t>(y) * W + x;
      nchw[0 * plane + idx] = s[0];
      nchw[1 * plane + idx] = s[1];
      nchw[2 * plane + idx] = s[2];
    }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 4> ishape{1, 3, H, W};
  Ort::Value image = Ort::Value::CreateTensor<float>(mem, nchw.data(), nchw.size(),
                                                     ishape.data(), ishape.size());
  int64_t tok = impl_->num_tokens;
  Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(mem, &tok, 1, nullptr, 0);

  std::vector<const char*> in_names = {impl_->image_input.c_str(), impl_->tokens_input.c_str()};
  std::vector<Ort::Value> ins;
  ins.push_back(std::move(image));
  ins.push_back(std::move(tokens));
  const char* out_names[] = {"points", "mask"};
  std::vector<Ort::Value> res;
  try {
    res = impl_->session->Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(),
                              ins.size(), out_names, 2);
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("MoGe inference failed: ") + e.what();
    return out;
  }
  if (res.size() < 2) return out;

  // points: [1,H,W,3] channels-last; mask: [1,H,W] sigmoid.
  const float* points = res[0].GetTensorData<float>();
  const float* mask = res[1].GetTensorData<float>();

  std::vector<float> U, V;
  NormalizedUV(W, H, U, V);

  // Mask-aware nearest downsample to 64x64.
  const int DS = 64;
  std::vector<double> Xs, Ys, Zs, Us, Vs;
  Xs.reserve(DS * DS);
  for (int gy = 0; gy < DS; ++gy) {
    int sy = std::min(H - 1, static_cast<int>((gy + 0.5) * H / DS));
    for (int gx = 0; gx < DS; ++gx) {
      int sx = std::min(W - 1, static_cast<int>((gx + 0.5) * W / DS));
      size_t idx = static_cast<size_t>(sy) * W + sx;
      if (mask[idx] <= 0.5f) continue;
      const float* p = points + idx * 3;
      Xs.push_back(p[0]); Ys.push_back(p[1]); Zs.push_back(p[2]);
      Us.push_back(U[idx]); Vs.push_back(V[idx]);
    }
  }
  if (Xs.size() < 2) {
    out.ok = false;
    last_error_ = "MoGe: too few valid points for focal recovery";
    return out;
  }

  double focal_norm = 1.0, shift = 0.0;
  if (fov_x_deg_known > 0.0) {
    const double ar = static_cast<double>(W) / H;
    const double span_x = ar / std::sqrt(1.0 + ar * ar);
    focal_norm = span_x / std::tan(fov_x_deg_known * M_PI / 180.0 / 2.0);
    // shift-only: 1-D min of the fixed-focal residual, reuse grid+golden on E'(shift)
  }

  // 1-D minimisation of E(shift): coarse grid bracket + golden-section refine.
  double zmin = *std::min_element(Zs.begin(), Zs.end());
  double lo = -zmin + 0.02, hi = zmin + std::fabs(zmin) + 2.0;
  if (hi <= lo) hi = lo + 1.0;
  auto eval = [&](double sh, double* f) {
    if (fov_x_deg_known > 0.0) {
      // fixed focal: residual with focal_norm held
      double e = 0;
      for (size_t i = 0; i < Xs.size(); ++i) {
        double zp = Zs[i] + sh;
        if (std::fabs(zp) < 1e-6) return 1e30;
        double px = Xs[i] / zp, py = Ys[i] / zp;
        double rx = focal_norm * px - Us[i], ry = focal_norm * py - Vs[i];
        e += rx * rx + ry * ry;
      }
      if (f) *f = focal_norm;
      return e;
    }
    return CostAndFocal(Xs, Ys, Zs, Us, Vs, sh, f);
  };

  const int GRID = 40;
  double best = 1e300, bshift = 0;
  for (int i = 0; i <= GRID; ++i) {
    double sh = lo + (hi - lo) * i / GRID;
    double e = eval(sh, nullptr);
    if (e < best) { best = e; bshift = sh; }
  }
  // golden-section around the best grid point
  double a = bshift - (hi - lo) / GRID, b = bshift + (hi - lo) / GRID;
  a = std::max(a, lo); b = std::min(b, hi);
  const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
  double c = b - gr * (b - a), d = a + gr * (b - a);
  double fc = eval(c, nullptr), fd = eval(d, nullptr);
  for (int it = 0; it < 60 && (b - a) > 1e-6; ++it) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = eval(c, nullptr); }
    else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = eval(d, nullptr); }
  }
  shift = 0.5 * (a + b);
  eval(shift, &focal_norm);

  // Convert normalized focal -> pixels/FOV for the reported resolution.
  const int RW = report_w > 0 ? report_w : in_w;
  const int RH = report_h > 0 ? report_h : in_h;
  const double diag = std::sqrt(static_cast<double>(RW) * RW + static_cast<double>(RH) * RH);
  out.ok = true;
  out.focal_norm = focal_norm;
  out.shift = shift;
  out.fx_px = out.fy_px = focal_norm * diag / 2.0;
  out.cx_px = 0.5 * RW;
  out.cy_px = 0.5 * RH;
  out.fov_x_deg = 2.0 * std::atan((RW / 2.0) / out.fx_px) * 180.0 / M_PI;
  out.fov_y_deg = 2.0 * std::atan((RH / 2.0) / out.fy_px) * 180.0 / M_PI;
  out.out_w = RW;
  out.out_h = RH;
  return out;
}

}  // namespace da3
