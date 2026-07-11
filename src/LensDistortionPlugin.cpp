// Copyright the openfx-onnx-depthanything3 authors.
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "LensModel.h"
#include "Register.h"

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
  // Optional ML estimation hook. No distortion model is bundled by default (the
  // strong single-image models need a host-side solver or training), so this looks
  // for an ONNX and reports if absent. The expected model is feed-forward, output
  // a vector [k1,k2,p1,p2,k3] (Deep-BrownConrady style).
  std::string p;
  _modelFile->getValue(p);
  if (p.empty()) {
    if (const char* e = std::getenv("LENS_MODEL_PATH")) p = e;
  }
  struct stat st;
  if (p.empty() || stat(p.c_str(), &st) != 0) {
    setPersistentMessage(OFX::Message::eMessageMessage, "",
                         "No lens-distortion ML model available. Set the distortion parameters "
                         "manually (or from a sidecar); overscan and 3DE outputs update live.");
    return;
  }
  // A model was provided; running it is delegated to a future estimator engine.
  setPersistentMessage(OFX::Message::eMessageMessage, "",
                       "A model path is set, but the ML estimator engine is not enabled in this "
                       "build. Parameters remain manual.");
  (void)time;
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
    p->setLabel("ML model (optional)");
    p->setHint("Optional ONNX lens-distortion estimator. If empty, set parameters manually.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    PushButtonParamDescriptor* p = desc.definePushButtonParam(kLdEstimate);
    p->setLabel("Estimate (ML) / Recompute");
    p->setHint("Run the ML estimator if a model is set, then recompute overscan + 3DE outputs.");
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
