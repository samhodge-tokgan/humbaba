// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
//
// MoGe Focal: estimates camera focal length / FOV / intrinsics from a single
// frame using the MoGe-2 ONNX model (ONNX Runtime + CoreML). It's an analysis
// node: the image is passed through, and an "Analyze current frame" button runs
// inference and writes the recovered focal length (px), FOV and principal point
// into output parameters the artist can link to a camera.

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

#include "MoGeEngine.h"
#include "Register.h"

#define kMoGeName "MoGe Focal"
#define kMoGeGrouping "TokGan"
#define kMoGeDescription \
  "Estimate camera focal length / FOV / intrinsics from a single frame using MoGe-2 " \
  "(ONNX Runtime + CoreML). Press Analyze; the image is passed through."
#define kMoGeIdentifier "com.tokgan.openfx.MoGeFocal"

#define kMoGeModelFile "mogeModelFile"
#define kMoGeCompute "mogeComputeUnits"
#define kMoGeTokens "mogeNumTokens"
#define kMoGeCap "mogeCapLong"
#define kMoGeACEScg "mogeInputIsACEScg"
#define kMoGeKnownFov "mogeKnownFovX"
#define kMoGeAnalyze "mogeAnalyze"
#define kMoGeFocalPx "focalPx"
#define kMoGeFovX "fovX"
#define kMoGeFovY "fovY"
#define kMoGeCx "principalX"
#define kMoGeCy "principalY"

static inline float srgbEnc(float c) {
  c = std::min(std::max(c, 0.0f), 1.0f);
  return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}
static inline void acescgToSrgb(float r, float g, float b, float* o) {
  float lr = 1.70505f * r - 0.62179f * g - 0.08326f * b;
  float lg = -0.13026f * r + 1.14080f * g - 0.01055f * b;
  float lb = -0.02400f * r - 0.12897f * g + 1.15297f * b;
  o[0] = srgbEnc(lr); o[1] = srgbEnc(lg); o[2] = srgbEnc(lb);
}

static std::string mogeBundleModelPath() {
#if defined(_WIN32)
  HMODULE hm = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(&mogeBundleModelPath), &hm)) {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(hm, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      std::string p(buf, n);
      auto pos = p.rfind("\\Contents\\Win64\\");
      if (pos != std::string::npos) {
        std::string cand = p.substr(0, pos) + "\\Contents\\Resources\\moge-2-vitb.onnx";
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
  if (dladdr(reinterpret_cast<const void*>(&mogeBundleModelPath), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    auto pos = p.rfind(marker);
    if (pos != std::string::npos) {
      std::string cand = p.substr(0, pos) + "/Contents/Resources/moge-2-vitb.onnx";
      struct stat st;
      if (stat(cand.c_str(), &st) == 0) return cand;
    }
  }
#endif
  return std::string();
}

////////////////////////////////////////////////////////////////////////////////

class MoGePlugin : public OFX::ImageEffect {
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;
  OFX::StringParam* _modelFile;
  OFX::ChoiceParam* _compute;
  OFX::IntParam* _tokens;
  OFX::IntParam* _cap;
  OFX::BooleanParam* _acescg;
  OFX::DoubleParam* _knownFov;
  OFX::DoubleParam* _focalPx;
  OFX::DoubleParam* _fovX;
  OFX::DoubleParam* _fovY;
  OFX::DoubleParam* _cx;
  OFX::DoubleParam* _cy;

 public:
  explicit MoGePlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _modelFile = fetchStringParam(kMoGeModelFile);
    _compute = fetchChoiceParam(kMoGeCompute);
    _tokens = fetchIntParam(kMoGeTokens);
    _cap = fetchIntParam(kMoGeCap);
    _acescg = fetchBooleanParam(kMoGeACEScg);
    _knownFov = fetchDoubleParam(kMoGeKnownFov);
    _focalPx = fetchDoubleParam(kMoGeFocalPx);
    _fovX = fetchDoubleParam(kMoGeFovX);
    _fovY = fetchDoubleParam(kMoGeFovY);
    _cx = fetchDoubleParam(kMoGeCx);
    _cy = fetchDoubleParam(kMoGeCy);
  }

  void render(const OFX::RenderArguments& args) override;
  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override;

 private:
  std::string resolveModel() {
    std::string p;
    _modelFile->getValue(p);
    if (!p.empty()) return p;
    if (const char* e = std::getenv("MOGE_MODEL_PATH")) return std::string(e);
    return mogeBundleModelPath();
  }
  void analyze(double time);
};

// Passthrough: copy source to output over the render window (float RGBA and others).
void MoGePlugin::render(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const int nc = dst->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  const OfxRectI w = args.renderWindow;
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const size_t bytes = bd == OFX::eBitDepthFloat ? 4 : (bd == OFX::eBitDepthUShort ? 2 : 1);
  for (int y = w.y1; y < w.y2; ++y) {
    if (abort()) break;
    char* d = static_cast<char*>(dst->getPixelAddress(w.x1, y));
    const char* s = src.get() ? static_cast<const char*>(src->getPixelAddress(w.x1, y)) : nullptr;
    size_t rowBytes = static_cast<size_t>(w.x2 - w.x1) * nc * bytes;
    if (s) std::memcpy(d, s, rowBytes);
    else std::memset(d, 0, rowBytes);
  }
}

void MoGePlugin::changedParam(const OFX::InstanceChangedArgs& /*args*/, const std::string& name) {
  // A push-button only "changes" when pressed (GUI or programmatic), so run on any reason.
  if (name == kMoGeAnalyze) {
    analyze(timeLineGetTime());
  }
}

void MoGePlugin::analyze(double time) {
  const std::string model = resolveModel();
  if (model.empty()) {
    setPersistentMessage(OFX::Message::eMessageError, "",
                         "No MoGe model set (parameter 'MoGe model', $MOGE_MODEL_PATH or bundle).");
    return;
  }
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(time));
  if (!src.get() || src->getPixelDepth() != OFX::eBitDepthFloat) {
    setPersistentMessage(OFX::Message::eMessageError, "", "MoGe needs a float source image.");
    return;
  }
  const OfxRectI rod = src->getRegionOfDefinition();
  const int W = rod.x2 - rod.x1, H = rod.y2 - rod.y1;
  const int nc = src->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  if (W <= 0 || H <= 0 || nc < 3) {
    setPersistentMessage(OFX::Message::eMessageError, "", "MoGe needs an RGB(A) float image.");
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
      if (acescg) acescgToSrgb(s[0], s[1], s[2], d);
      else { d[0] = std::min(std::max(s[0], 0.f), 1.f);
             d[1] = std::min(std::max(s[1], 0.f), 1.f);
             d[2] = std::min(std::max(s[2], 0.f), 1.f); }
    }

  int tok = 1800, cap = 512, cu = 0;
  _tokens->getValue(tok); _cap->getValue(cap); _compute->getValue(cu);
  double knownFov = 0.0;
  _knownFov->getValue(knownFov);

  da3::MoGeEngine eng(model, static_cast<da3::ComputeUnits>(cu), 0, tok, cap);
  if (!eng.last_error().empty()) {
    setPersistentMessage(OFX::Message::eMessageError, "", eng.last_error());
    return;
  }
  da3::FocalResult r = eng.EstimateFocal(rgb.data(), W, H, W, H, knownFov);
  if (!r.ok) {
    setPersistentMessage(OFX::Message::eMessageError, "",
                         eng.last_error().empty() ? "MoGe focal estimation failed" : eng.last_error());
    return;
  }
  clearPersistentMessage();
  _focalPx->setValue(r.fx_px);
  _fovX->setValue(r.fov_x_deg);
  _fovY->setValue(r.fov_y_deg);
  _cx->setValue(r.cx_px);
  _cy->setValue(r.cy_px);
  sendMessage(OFX::Message::eMessageMessage, "",
              "MoGe: focal = " + std::to_string(r.fx_px) + " px, FOV_x = " +
              std::to_string(r.fov_x_deg) + " deg");
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(MoGeFactory, {}, {});

using namespace OFX;

void MoGeFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kMoGeName, kMoGeName, kMoGeName);
  desc.setPluginGrouping(kMoGeGrouping);
  desc.setPluginDescription(kMoGeDescription);
  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void MoGeFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->setSupportsTiles(false);
  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");
  {
    StringParamDescriptor* p = desc.defineStringParam(kMoGeModelFile);
    p->setLabel("MoGe model");
    p->setHint("Path to the MoGe ONNX model. Falls back to $MOGE_MODEL_PATH, then the bundle.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kMoGeACEScg);
    p->setLabel("Input is ACEScg");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kMoGeCompute);
    p->setLabel("Compute units");
    p->setHint("MoGe currently runs on CPU (its dynamic graph is not CoreML-executable); "
               "this is reserved for a future CoreML-compatible export.");
    p->appendOption("All (ANE/GPU/CPU)");
    p->appendOption("CPU + GPU");
    p->appendOption("CPU + Neural Engine");
    p->appendOption("CPU only");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kMoGeTokens);
    p->setLabel("Num tokens (quality)");
    p->setHint("MoGe internal working resolution; higher = sharper/slower.");
    p->setRange(1200, 3600);
    p->setDisplayRange(1200, 3600);
    p->setDefault(1800);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kMoGeCap);
    p->setLabel("Max long side");
    p->setHint("Cap the ONNX input long side for speed (focal only needs a small solve).");
    p->setRange(128, 2048);
    p->setDisplayRange(256, 1024);
    p->setDefault(512);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kMoGeKnownFov);
    p->setLabel("Known horizontal FOV (deg, 0=estimate)");
    p->setHint("If > 0, focal is fixed from this FOV and only the depth shift is solved.");
    p->setRange(0.0, 179.0);
    p->setDisplayRange(0.0, 120.0);
    p->setDefault(0.0);
    page->addChild(*p);
  }
  {
    PushButtonParamDescriptor* p = desc.definePushButtonParam(kMoGeAnalyze);
    p->setLabel("Analyze current frame");
    p->setHint("Run MoGe on the current frame and fill the focal/FOV outputs below.");
    page->addChild(*p);
  }
  auto outDouble = [&](const char* name, const char* label, double def, double lo, double hi) {
    DoubleParamDescriptor* p = desc.defineDoubleParam(name);
    p->setLabel(label);
    p->setEvaluateOnChange(false);   // output only, doesn't affect the render
    p->setAnimates(false);
    p->setRange(lo, hi);
    p->setDisplayRange(lo, hi);
    p->setDefault(def);
    page->addChild(*p);
  };
  outDouble(kMoGeFocalPx, "→ Focal length (px)", 0.0, 0.0, 1e6);
  outDouble(kMoGeFovX, "→ Horizontal FOV (deg)", 0.0, 0.0, 180.0);
  outDouble(kMoGeFovY, "→ Vertical FOV (deg)", 0.0, 0.0, 180.0);
  outDouble(kMoGeCx, "→ Principal point X (px)", 0.0, 0.0, 1e6);
  outDouble(kMoGeCy, "→ Principal point Y (px)", 0.0, 0.0, 1e6);
}

OFX::ImageEffect* MoGeFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) {
  return new MoGePlugin(handle);
}

namespace da3reg {
void appendMoGe(OFX::PluginFactoryArray& ids) {
  static MoGeFactory p(kMoGeIdentifier, 0, 1);
  ids.push_back(&p);
}
}  // namespace da3reg
