// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
#include "AnyCalibEngine.h"

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

const char* CU(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CPUAndGPU: return "CPUAndGPU";
    case ComputeUnits::CPUAndNeuralEngine: return "CPUAndNeuralEngine";
    case ComputeUnits::CPUOnly: return "CPUOnly";
  }
  return "ALL";
}

// Area-averaging (box) resample. AnyCalib downsamples ~5x, so proper antialiasing
// matters — plain bilinear aliases and corrupts the distortion estimate.
void ResizeRGB(const float* src, int sw, int sh, float* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y) {
    int sy0 = int((int64_t(y) * sh) / dh);
    int sy1 = std::max(sy0 + 1, int((int64_t(y + 1) * sh) / dh));
    sy1 = std::min(sy1, sh);
    for (int x = 0; x < dw; ++x) {
      int sx0 = int((int64_t(x) * sw) / dw);
      int sx1 = std::max(sx0 + 1, int((int64_t(x + 1) * sw) / dw));
      sx1 = std::min(sx1, sw);
      double acc[3] = {0, 0, 0};
      for (int yy = sy0; yy < sy1; ++yy)
        for (int xx = sx0; xx < sx1; ++xx) {
          const float* p = src + (size_t(yy) * sw + xx) * 3;
          acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2];
        }
      double n = double(sy1 - sy0) * (sx1 - sx0);
      float* o = dst + (size_t(y) * dw + x) * 3;
      o[0] = float(acc[0] / n); o[1] = float(acc[1] / n); o[2] = float(acc[2] / n);
    }
  }
}

// Solve A x = b for a small dense n x n system (Gaussian elimination, partial pivot).
bool SolveDense(std::vector<double>& A, std::vector<double>& b, int n) {
  for (int col = 0; col < n; ++col) {
    int piv = col; double best = std::fabs(A[col * n + col]);
    for (int r = col + 1; r < n; ++r) {
      double v = std::fabs(A[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-15) return false;
    if (piv != col) {
      for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
      std::swap(b[col], b[piv]);
    }
    double d = A[col * n + col];
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      double f = A[r * n + col] / d;
      if (f == 0) continue;
      for (int c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
      b[r] -= f * b[col];
    }
  }
  for (int i = 0; i < n; ++i) b[i] /= A[i * n + i];
  return true;
}

int RoundTo14(double v) {
  int r = int(std::lround(v / 14.0)) * 14;
  return r < 14 ? 14 : r;
}

// AnyCalib target size: area ~102400, aspect (H/W) clamped to [0.5,2], multiples of 14.
void TargetSize(int Wo, int Ho, int& Wp, int& Hp) {
  const double AREA = 102400.0;
  double ar = std::max(0.5, std::min(double(Ho) / Wo, 2.0));  // H/W
  double w = std::sqrt(AREA / ar);
  double h = ar * w;
  Wp = RoundTo14(w);
  Hp = RoundTo14(h);
}

// unproject a pixel (u,v) to a unit bearing under params [fx,fy,cx,cy,k1,k2].
void Unproject(const double* p, double u, double v, double* ray) {
  double fx = p[0], fy = p[1], cx = p[2], cy = p[3], k1 = p[4], k2 = p[5];
  double xu = (u - cx) / fx, yu = (v - cy) / fy;
  double rd = std::sqrt(xu * xu + yu * yu);
  double ru = rd;
  for (int it = 0; it < 20; ++it) {
    double ru2 = ru * ru;
    double f = ru * (1 + k1 * ru2 + k2 * ru2 * ru2) - rd;
    double df = 1 + 3 * k1 * ru2 + 5 * k2 * ru2 * ru2;
    ru -= f / std::max(df, 1e-9);
  }
  double scale = rd > 1e-9 ? ru / rd : 1.0;
  double xn = xu * scale, yn = yu * scale;
  double nn = std::sqrt(xn * xn + yn * yn + 1.0);
  ray[0] = xn / nn; ray[1] = yn / nn; ray[2] = 1.0 / nn;
}

}  // namespace

struct AnyCalibEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "AnyCalib"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::string input_name;
};

AnyCalibEngine::AnyCalibEngine(const std::string& model_path, ComputeUnits units, int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  auto tryCreate = [&](bool accel) -> bool {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);
    if (accel && da3::AcceleratorAvailable()) {
      try {
        da3::AppendAccelerator(so, CU(units),
                               /*coreml_static=*/false, /*coreml_mlprogram=*/false);
      } catch (const Ort::Exception&) { return false; }
    }
    try {
      impl_->session = std::make_unique<Ort::Session>(impl_->env, da3::OrtPath(model_path).c_str(), so);
      return true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string("AnyCalib session create failed: ") + e.what();
      impl_->session.reset();
      return false;
    }
  };
  if (tryCreate(true) || tryCreate(false)) {
    last_error_.clear();
    Ort::AllocatedStringPtr in = impl_->session->GetInputNameAllocated(0, impl_->alloc);
    impl_->input_name = in.get();
  }
}

AnyCalibEngine::~AnyCalibEngine() = default;

CameraFit AnyCalibEngine::Estimate(const float* rgb, int in_w, int in_h) {
  CameraFit out;
  if (!impl_ || !impl_->session) return out;

  int Wp, Hp;
  TargetSize(in_w, in_h, Wp, Hp);
  std::vector<float> resized(size_t(Wp) * Hp * 3);
  ResizeRGB(rgb, in_w, in_h, resized.data(), Wp, Hp);
  std::vector<float> nchw(size_t(Wp) * Hp * 3);
  const size_t plane = size_t(Wp) * Hp;
  for (int y = 0; y < Hp; ++y)
    for (int x = 0; x < Wp; ++x) {
      const float* s = resized.data() + (size_t(y) * Wp + x) * 3;
      size_t i = size_t(y) * Wp + x;
      nchw[i] = s[0]; nchw[plane + i] = s[1]; nchw[2 * plane + i] = s[2];
    }

  std::array<int64_t, 4> shape{1, 3, Hp, Wp};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(mem, nchw.data(), nchw.size(),
                                                     shape.data(), shape.size());
  const char* in_names[] = {impl_->input_name.c_str()};
  const char* out_names[] = {"rays"};
  std::vector<Ort::Value> res;
  try {
    res = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("AnyCalib inference failed: ") + e.what();
    return out;
  }
  if (res.empty()) return out;
  const float* rays = res[0].GetTensorData<float>();  // [1, N, 3], N = Hp*Wp, row-major

  // Subsample ~4096 pixels for the fit.
  const int N = Wp * Hp;
  const int TARGET = 4096;
  const int stride = std::max(1, N / TARGET);
  std::vector<double> U, V, PX, PY, RX, RY, RZ;  // im coords, proj, ray
  for (int i = 0; i < N; i += stride) {
    double rz = rays[i * 3 + 2];
    if (std::fabs(rz) < 1e-6) continue;
    double rx = rays[i * 3 + 0], ry = rays[i * 3 + 1];
    int px = i % Wp, py = i / Wp;
    U.push_back(px + 0.5); V.push_back(py + 0.5);
    PX.push_back(rx / rz); PY.push_back(ry / rz);
    RX.push_back(rx); RY.push_back(ry); RZ.push_back(rz);
  }
  const int M = int(U.size());
  if (M < 20) { last_error_ = "AnyCalib: too few valid rays"; return out; }

  // ---- DLT linear init: unknowns [ax,ay,cxn,cyn,k1,k2] ----
  // Normalize pixel coords by their max for conditioning (the distortion coeffs are
  // otherwise ill-conditioned in the normal equations); undo the scale afterwards.
  double sc_dlt = std::max(1.0, double(std::max(Wp, Hp)));
  std::vector<double> AtA(36, 0.0), Atb(6, 0.0);
  for (int i = 0; i < M; ++i) {
    double u = U[i] / sc_dlt, v = V[i] / sc_dlt, px = PX[i], py = PY[i];
    double r2 = px * px + py * py, r4 = r2 * r2;
    double rowx[6] = {u, 0, -1, 0, -px * r2, -px * r4};
    double rowy[6] = {0, v, 0, -1, -py * r2, -py * r4};
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) AtA[a * 6 + b] += rowx[a] * rowx[b] + rowy[a] * rowy[b];
      Atb[a] += rowx[a] * px + rowy[a] * py;
    }
  }
  std::vector<double> s = Atb, A = AtA;
  if (!SolveDense(A, s, 6)) { last_error_ = "AnyCalib: DLT solve failed"; return out; }
  // fx,cx were solved in the normalized-pixel frame -> multiply by sc_dlt.
  double p[6] = {sc_dlt / s[0], sc_dlt / s[1], s[2] * sc_dlt / s[0], s[3] * sc_dlt / s[1], s[4], s[5]};

  // ---- Gauss-Newton refine (finite-diff Jacobian, logmap angular residual) ----
  auto residuals = [&](const double* pp, std::vector<double>& r) {
    r.resize(size_t(M) * 3);
    for (int i = 0; i < M; ++i) {
      double ray[3];
      Unproject(pp, U[i], V[i], ray);
      double ox = RX[i], oy = RY[i], oz = RZ[i];
      double onrm = std::sqrt(ox * ox + oy * oy + oz * oz);
      ox /= onrm; oy /= onrm; oz /= onrm;
      double cosang = std::max(-1.0, std::min(1.0, ray[0] * ox + ray[1] * oy + ray[2] * oz));
      double perp[3] = {ox - cosang * ray[0], oy - cosang * ray[1], oz - cosang * ray[2]};
      double pn = std::sqrt(perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]);
      double th = std::acos(cosang);
      double sc = pn > 1e-9 ? th / pn : 0.0;
      r[i * 3 + 0] = sc * perp[0]; r[i * 3 + 1] = sc * perp[1]; r[i * 3 + 2] = sc * perp[2];
    }
  };
  std::vector<double> r0, rp;
  for (int it = 0; it < 12; ++it) {
    residuals(p, r0);
    std::vector<double> JtJ(36, 0.0), Jtr(6, 0.0);
    double col[6][3];  // reused per-pixel Jacobian columns
    // finite-diff Jacobian per parameter
    std::array<std::vector<double>, 6> J;
    for (int j = 0; j < 6; ++j) {
      double pp[6]; std::copy(p, p + 6, pp);
      double h = std::max(std::fabs(p[j]) * 1e-4, 1e-6);
      pp[j] += h;
      residuals(pp, rp);
      J[j].resize(r0.size());
      for (size_t t = 0; t < r0.size(); ++t) J[j][t] = (rp[t] - r0[t]) / h;
    }
    (void)col;
    for (size_t t = 0; t < r0.size(); ++t) {
      for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) JtJ[a * 6 + b] += J[a][t] * J[b][t];
        Jtr[a] += J[a][t] * r0[t];
      }
    }
    for (int a = 0; a < 6; ++a) JtJ[a * 6 + a] += 1e-9;
    std::vector<double> delta(6);
    for (int a = 0; a < 6; ++a) delta[a] = -Jtr[a];
    std::vector<double> JJ = JtJ;
    if (!SolveDense(JJ, delta, 6)) break;
    for (int a = 0; a < 6; ++a) p[a] += delta[a];
  }

  // ---- reverse resize (per axis; no crop for AR in range) ----
  double sx = double(Wp) / in_w, sy = double(Hp) / in_h;
  out.ok = true;
  out.fx = p[0] / sx; out.fy = p[1] / sy;
  out.cx = p[2] / sx; out.cy = p[3] / sy;
  out.k1 = p[4]; out.k2 = p[5];
  return out;
}

}  // namespace da3
