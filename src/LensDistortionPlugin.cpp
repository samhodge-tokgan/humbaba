// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
//
// Lens Distortion: holds Brown-Conrady distortion parameters (entered manually,
// from a sidecar, or estimated by an optional ML model), and outputs the
// downstream lens data — OpenCV coefficients, 3DEqualizer coefficients, and the
// overscan / padding needed to render CG that will be re-distorted to the plate.
// It does NOT apply distortion; the image is passed through.
//
// Note: there is no standard OFX mechanism to read lens metadata from a clip, so
// "metadata passthrough" is realized as manual/sidecar parameter override here.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "AnyCalibEngine.h"
#include "LensModel.h"
#include "Register.h"

static inline float ld_srgb(float c) {
  c = std::min(std::max(c, 0.0f), 1.0f);
  return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}
static inline void ld_acescgToSrgb(float r, float g, float b, float* o) {
  float lr = 1.70505f * r - 0.62179f * g - 0.08326f * b;
  float lg = -0.13026f * r + 1.14080f * g - 0.01055f * b;
  float lb = -0.02400f * r - 0.12897f * g + 1.15297f * b;
  o[0] = ld_srgb(lr); o[1] = ld_srgb(lg); o[2] = ld_srgb(lb);
}
static std::string ld_bundleModel() {
#if defined(_WIN32)
  HMODULE hm = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(&ld_bundleModel), &hm)) {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(hm, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      std::string p(buf, n);
      auto pos = p.rfind("\\Contents\\Win64\\");
      if (pos != std::string::npos) {
        std::string cand = p.substr(0, pos) + "\\Contents\\Resources\\anycalib_dist.onnx";
        if (GetFileAttributesA(cand.c_str()) != INVALID_FILE_ATTRIBUTES) return cand;
      }
    }
  }
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__APPLE__)
  const std::string marker = "/Contents/MacOS/";
#else
  const std::string marker = "/Contents/Linux-x86-64/";
#endif
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&ld_bundleModel), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    auto pos = p.rfind(marker);
    if (pos != std::string::npos) {
      std::string cand = p.substr(0, pos) + "/Contents/Resources/anycalib_dist.onnx";
      struct stat st;
      if (stat(cand.c_str(), &st) == 0) return cand;
    }
  }
#endif
  return std::string();
}

#define kLdName "Lens Distortion"
#define kLdGrouping "TokGan"
#define kLdDescription \
  "Estimate / hold lens distortion (Brown-Conrady) and output OpenCV + 3DE " \
  "parameters and the overscan padding for CG re-distortion. Image is passed through."
#define kLdIdentifier "com.tokgan.openfx.LensDistortion"

#define kLdK1 "k1"
#define kLdK2 "k2"
#define kLdK3 "k3"
#define kLdP1 "p1"
#define kLdP2 "p2"
#define kLdFocal "focalPx"
#define kLdSamples "boundarySamples"
#define kLdModelFile "ldModelFile"
#define kLdACEScg "ldInputIsACEScg"
#define kLdEstimate "ldEstimate"
#define kLdOverL "overscanLeft"
#define kLdOverR "overscanRight"
#define kLdOverT "overscanTop"
#define kLdOverB "overscanBottom"
#define kLdOverPctX "overscanPctX"
#define kLdOverPctY "overscanPctY"
#define kLdC2 "tde4_c2"
#define kLdC4 "tde4_c4"
#define kLdU1 "tde4_u1"
#define kLdV1 "tde4_v1"

////////////////////////////////////////////////////////////////////////////////

class LensDistortionPlugin : public OFX::ImageEffect {
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;
  OFX::DoubleParam *_k1, *_k2, *_k3, *_p1, *_p2, *_focal;
  OFX::IntParam* _samples;
  OFX::StringParam* _modelFile;
  OFX::BooleanParam* _acescg;
  OFX::DoubleParam *_oL, *_oR, *_oT, *_oB, *_oPx, *_oPy;
  OFX::DoubleParam *_c2, *_c4, *_u1, *_v1;

 public:
  explicit LensDistortionPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _k1 = fetchDoubleParam(kLdK1); _k2 = fetchDoubleParam(kLdK2); _k3 = fetchDoubleParam(kLdK3);
    _p1 = fetchDoubleParam(kLdP1); _p2 = fetchDoubleParam(kLdP2); _focal = fetchDoubleParam(kLdFocal);
    _samples = fetchIntParam(kLdSamples);
    _modelFile = fetchStringParam(kLdModelFile);
    _acescg = fetchBooleanParam(kLdACEScg);
    _oL = fetchDoubleParam(kLdOverL); _oR = fetchDoubleParam(kLdOverR);
    _oT = fetchDoubleParam(kLdOverT); _oB = fetchDoubleParam(kLdOverB);
    _oPx = fetchDoubleParam(kLdOverPctX); _oPy = fetchDoubleParam(kLdOverPctY);
    _c2 = fetchDoubleParam(kLdC2); _c4 = fetchDoubleParam(kLdC4);
    _u1 = fetchDoubleParam(kLdU1); _v1 = fetchDoubleParam(kLdV1);
  }

  void render(const OFX::RenderArguments& args) override;
  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override;

 private:
  void recompute(double time);
  void estimate(double time);
  bool srcSize(double time, int& W, int& H);
};

bool LensDistortionPlugin::srcSize(double time, int& W, int& H) {
  if (!_srcClip || !_srcClip->isConnected()) return false;
  OfxRectD rod = _srcClip->getRegionOfDefinition(time);
  W = static_cast<int>(std::lround(rod.x2 - rod.x1));
  H = static_cast<int>(std::lround(rod.y2 - rod.y1));
  return W > 1 && H > 1;
}

void LensDistortionPlugin::recompute(double time) {
  int W = 1920, H = 1080;
  srcSize(time, W, H);
  da3::BrownConrady d;
  _k1->getValue(d.k1); _k2->getValue(d.k2); _k3->getValue(d.k3);
  _p1->getValue(d.p1); _p2->getValue(d.p2);
  double f = 2000.0;
  _focal->getValue(f);
  if (f <= 0) f = 1.5 * std::max(W, H);
  int ns = 33; _samples->getValue(ns);

  da3::Overscan o = da3::ComputeOverscan(d, W, H, f, f, 0.5 * W, 0.5 * H, ns);
  _oL->setValue(o.left); _oR->setValue(o.right); _oT->setValue(o.top); _oB->setValue(o.bottom);
  _oPx->setValue(o.pct_x); _oPy->setValue(o.pct_y);

  da3::TDE4Radial t = da3::OpenCVToTDE4(d, W, H, f, f);
  _c2->setValue(t.distortion_c2); _c4->setValue(t.quartic_c4);
  _u1->setValue(t.u1); _v1->setValue(t.v1);
}

void LensDistortionPlugin::estimate(double time) {
  // ML estimation via AnyCalib (DINOv2 field net in ONNX + host-side camera fit).
  std::string model;
  _modelFile->getValue(model);
  if (model.empty()) {
    if (const char* e = std::getenv("ANYCALIB_MODEL_PATH")) model = e;
  }
  if (model.empty()) model = ld_bundleModel();
  if (model.empty()) {
    setPersistentMessage(OFX::Message::eMessageMessage, "",
                         "No AnyCalib model available. Set parameters manually (or a sidecar); "
                         "overscan and 3DE outputs update live.");
    return;
  }

  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(time));
  if (!src.get() || src->getPixelDepth() != OFX::eBitDepthFloat) {
    setPersistentMessage(OFX::Message::eMessageError, "", "AnyCalib needs a float source image.");
    return;
  }
  const OfxRectI rod = src->getRegionOfDefinition();
  const int W = rod.x2 - rod.x1, H = rod.y2 - rod.y1;
  const int nc = src->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  if (W <= 0 || H <= 0 || nc < 3) {
    setPersistentMessage(OFX::Message::eMessageError, "", "AnyCalib needs an RGB(A) float image.");
    return;
  }
  bool acescg = true;
  _acescg->getValue(acescg);
  std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float* s = static_cast<const float*>(src->getPixelAddress(rod.x1 + x, rod.y1 + y));
      float* d = rgb.data() + (static_cast<size_t>(y) * W + x) * 3;
      if (!s) { d[0] = d[1] = d[2] = 0; continue; }
      if (acescg) ld_acescgToSrgb(s[0], s[1], s[2], d);
      else { d[0] = std::min(std::max(s[0], 0.f), 1.f);
             d[1] = std::min(std::max(s[1], 0.f), 1.f);
             d[2] = std::min(std::max(s[2], 0.f), 1.f); }
    }

  da3::AnyCalibEngine eng(model, da3::ComputeUnits::All, 0);
  if (!eng.last_error().empty()) {
    setPersistentMessage(OFX::Message::eMessageError, "", eng.last_error());
    return;
  }
  da3::CameraFit fit = eng.Estimate(rgb.data(), W, H);
  if (!fit.ok) {
    setPersistentMessage(OFX::Message::eMessageError, "",
                         eng.last_error().empty() ? "AnyCalib fit failed" : eng.last_error());
    return;
  }
  clearPersistentMessage();
  // AnyCalib's radial maps directly to OpenCV: fx,fy,cx,cy,k1,k2 (k3=p1=p2=0).
  _focal->setValue(0.5 * (fit.fx + fit.fy));
  _k1->setValue(fit.k1); _k2->setValue(fit.k2); _k3->setValue(0.0);
  _p1->setValue(0.0); _p2->setValue(0.0);
  sendMessage(OFX::Message::eMessageMessage, "",
              "AnyCalib: fx=" + std::to_string(fit.fx) + " k1=" + std::to_string(fit.k1) +
              " k2=" + std::to_string(fit.k2));
}

void LensDistortionPlugin::render(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const int nc = dst->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const size_t bytes = bd == OFX::eBitDepthFloat ? 4 : (bd == OFX::eBitDepthUShort ? 2 : 1);
  const OfxRectI w = args.renderWindow;
  for (int y = w.y1; y < w.y2; ++y) {
    if (abort()) break;
    char* d = static_cast<char*>(dst->getPixelAddress(w.x1, y));
    const char* s = src.get() ? static_cast<const char*>(src->getPixelAddress(w.x1, y)) : nullptr;
    size_t rowBytes = static_cast<size_t>(w.x2 - w.x1) * nc * bytes;
    if (s) std::memcpy(d, s, rowBytes);
    else std::memset(d, 0, rowBytes);
  }
}

void LensDistortionPlugin::changedParam(const OFX::InstanceChangedArgs& args,
                                        const std::string& name) {
  if (name == kLdEstimate) {
    estimate(timeLineGetTime());
    recompute(timeLineGetTime());
    return;
  }
  if (name == kLdK1 || name == kLdK2 || name == kLdK3 || name == kLdP1 || name == kLdP2 ||
      name == kLdFocal || name == kLdSamples || args.reason == OFX::eChangeUserEdit) {
    recompute(timeLineGetTime());
  }
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(LensDistortionFactory, {}, {});

using namespace OFX;

void LensDistortionFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kLdName, kLdName, kLdName);
  desc.setPluginGrouping(kLdGrouping);
  desc.setPluginDescription(kLdDescription);
  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);
  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void LensDistortionFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->setSupportsTiles(false);
  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  auto coeff = [&](const char* n, const char* label, const char* hint) {
    DoubleParamDescriptor* p = desc.defineDoubleParam(n);
    p->setLabel(label); p->setHint(hint);
    p->setRange(-10, 10); p->setDisplayRange(-1, 1); p->setDefault(0.0);
    page->addChild(*p);
  };
  coeff(kLdK1, "k1 (OpenCV)", "Radial r^2 coefficient (focal-normalized).");
  coeff(kLdK2, "k2 (OpenCV)", "Radial r^4 coefficient.");
  coeff(kLdK3, "k3 (OpenCV)", "Radial r^6 coefficient.");
  coeff(kLdP1, "p1 (OpenCV)", "Tangential coefficient 1.");
  coeff(kLdP2, "p2 (OpenCV)", "Tangential coefficient 2.");
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kLdFocal);
    p->setLabel("Focal length (px)");
    p->setHint("Focal length in pixels (e.g. from the MoGe Focal plugin). 0 = auto (1.5*long side).");
    p->setRange(0, 1e6); p->setDisplayRange(0, 8000); p->setDefault(0.0);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kLdSamples);
    p->setLabel("Boundary samples");
    p->setHint("Samples per edge for overscan (>=8; the radial extremum can be at edge midpoints).");
    p->setRange(8, 129); p->setDisplayRange(9, 65); p->setDefault(33);
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kLdModelFile);
    p->setLabel("AnyCalib model (optional)");
    p->setHint("AnyCalib ONNX estimator. Empty = use the bundled model / $ANYCALIB_MODEL_PATH, "
               "else set distortion parameters manually.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kLdACEScg);
    p->setLabel("Input is ACEScg");
    p->setHint("Convert ACEScg to sRGB before the ML estimate.");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    PushButtonParamDescriptor* p = desc.definePushButtonParam(kLdEstimate);
    p->setLabel("Estimate (AnyCalib) / Recompute");
    p->setHint("Run AnyCalib on the current frame (fills k1,k2,focal), then recompute overscan + 3DE.");
    page->addChild(*p);
  }
  auto out = [&](const char* n, const char* label, const char* hint) {
    DoubleParamDescriptor* p = desc.defineDoubleParam(n);
    p->setLabel(label); p->setHint(hint);
    p->setEvaluateOnChange(false); p->setAnimates(false);
    p->setRange(-1e7, 1e7); p->setDisplayRange(0, 4000); p->setDefault(0.0);
    page->addChild(*p);
  };
  out(kLdOverL, "→ Overscan left (px)", "Padding needed on the left.");
  out(kLdOverR, "→ Overscan right (px)", "Padding needed on the right.");
  out(kLdOverT, "→ Overscan top (px)", "Padding needed on the top.");
  out(kLdOverB, "→ Overscan bottom (px)", "Padding needed on the bottom.");
  out(kLdOverPctX, "→ Overscan width %", "Fractional width growth (0.08 = 8%).");
  out(kLdOverPctY, "→ Overscan height %", "Fractional height growth.");
  out(kLdC2, "→ 3DE Distortion (c2)", "3DE4 Radial Std Deg4 'Distortion'.");
  out(kLdC4, "→ 3DE Quartic (c4)", "3DE4 Radial Std Deg4 'Quartic Distortion'.");
  out(kLdU1, "→ 3DE decentering u1", "Approx from p2.");
  out(kLdV1, "→ 3DE decentering v1", "Approx from p1.");
}

OFX::ImageEffect* LensDistortionFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) {
  return new LensDistortionPlugin(handle);
}

namespace da3reg {
void appendLensDistortion(OFX::PluginFactoryArray& ids) {
  static LensDistortionFactory p(kLdIdentifier, 0, 1);
  ids.push_back(&p);
}
}  // namespace da3reg
