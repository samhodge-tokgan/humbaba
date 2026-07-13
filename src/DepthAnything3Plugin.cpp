// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
//
// Depth Anything 3 OpenFX plugin.
//
// Predicts metric depth (decimeters, float32) from an ACEScg RGB image using the
// DA3 metric model via ONNX Runtime with the CoreML execution provider. Output is
// a same-size grayscale depth (Z) written to R=G=B, alpha passed through.
//
// If built without ONNX Runtime (DA3_WITH_ONNX undefined), render() is a passthrough.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#if DA3_WITH_ONNX && (defined(__APPLE__) || defined(__linux__))
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.H"

#if DA3_WITH_ONNX
#include "DepthEngine.h"
#include "Register.h"
#endif

#define kPluginName "Depth Anything 3"
#define kPluginGrouping "Tokgan"
#define kPluginDescription \
  "Predict metric depth (decimeters, float32) from an ACEScg image using " \
  "Depth Anything 3 via ONNX Runtime with hardware acceleration."
#define kPluginIdentifier "com.tokgan.openfx.DepthAnything3"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

#define kParamModelFile "modelFile"
#define kParamComputeUnits "computeUnits"
#define kParamLongSide "procLongSide"
#define kParamAutoRes "autoResolution"
#define kParamMemBudget "memBudgetMB"
#define kParamThreads "intraThreads"
#define kParamGain "depthGain"
#define kParamInputACEScg "inputIsACEScg"

////////////////////////////////////////////////////////////////////////////////
// Color: ACEScg (AP1, linear) -> linear Rec.709 -> sRGB, per channel in [0,1].

static inline float srgbEncode(float c) {
  c = std::min(std::max(c, 0.0f), 1.0f);
  return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

static inline void acescgToSrgb(float r, float g, float b, float* out) {
  // AP1 -> Rec.709 (linear) matrix.
  float lr = 1.70505f * r - 0.62179f * g - 0.08326f * b;
  float lg = -0.13026f * r + 1.14080f * g - 0.01055f * b;
  float lb = -0.02400f * r - 0.12897f * g + 1.15297f * b;
  out[0] = srgbEncode(lr);
  out[1] = srgbEncode(lg);
  out[2] = srgbEncode(lb);
}

// Map an approximate memory budget (MB) to a processing long-side resolution.
// CoreML exposes no VRAM cap, so on Apple Silicon the effective memory lever is
// the inference resolution. This is a calibrated approximation (documented as such).
static inline int budgetToLongSide(int mb) {
  double L = 7.9 * std::sqrt(static_cast<double>(mb < 64 ? 64 : mb));
  int v = static_cast<int>(std::lround(L));
  if (v < 140) v = 140;
  if (v > 1540) v = 1540;
  return v;
}

static inline float clamp01(float v) { return std::min(std::max(v, 0.0f), 1.0f); }

// Bit-depth-aware pixel IO. Hosts hand us RGBA in float, 8-bit or 16-bit (e.g. an
// 8-bit DaVinci Resolve timeline). The depth model works in float, so we normalize
// on read and quantize on write. Integer inputs are treated as display-referred
// sRGB (Rec.709) — an 8-bit 0-255 value maps straight to the [0,1] sRGB the model
// expects, bypassing the ACEScg->sRGB step (which only applies to linear float).
static inline void readRGBA01(const void* p, OFX::BitDepthEnum bd, float* rgb, float* a) {
  switch (bd) {
    case OFX::eBitDepthUByte: {
      const unsigned char* s = static_cast<const unsigned char*>(p);
      rgb[0] = s[0] * (1.0f / 255.0f); rgb[1] = s[1] * (1.0f / 255.0f);
      rgb[2] = s[2] * (1.0f / 255.0f); *a = s[3] * (1.0f / 255.0f); break;
    }
    case OFX::eBitDepthUShort: {
      const unsigned short* s = static_cast<const unsigned short*>(p);
      rgb[0] = s[0] * (1.0f / 65535.0f); rgb[1] = s[1] * (1.0f / 65535.0f);
      rgb[2] = s[2] * (1.0f / 65535.0f); *a = s[3] * (1.0f / 65535.0f); break;
    }
    default: {  // eBitDepthFloat
      const float* s = static_cast<const float*>(p);
      rgb[0] = s[0]; rgb[1] = s[1]; rgb[2] = s[2]; *a = s[3]; break;
    }
  }
}

// Write depth z (decimeters) to R=G=B and a normalized alpha, in the dst's depth.
// Float keeps full metric precision; integer clips quantize/clamp the decimeter value
// (use a float/32-bit timeline for lossless depth).
static inline void writeDepthPixel(void* p, OFX::BitDepthEnum bd, float z, float a01) {
  switch (bd) {
    case OFX::eBitDepthUByte: {
      unsigned char* d = static_cast<unsigned char*>(p);
      unsigned char zv =
          static_cast<unsigned char>(std::lround(std::min(std::max(z, 0.0f), 255.0f)));
      d[0] = d[1] = d[2] = zv;
      d[3] = static_cast<unsigned char>(std::lround(clamp01(a01) * 255.0f)); break;
    }
    case OFX::eBitDepthUShort: {
      unsigned short* d = static_cast<unsigned short*>(p);
      unsigned short zv =
          static_cast<unsigned short>(std::lround(std::min(std::max(z, 0.0f), 65535.0f)));
      d[0] = d[1] = d[2] = zv;
      d[3] = static_cast<unsigned short>(std::lround(clamp01(a01) * 65535.0f)); break;
    }
    default: {  // eBitDepthFloat
      float* d = static_cast<float*>(p);
      d[0] = d[1] = d[2] = z; d[3] = a01; break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Passthrough copy processor (used when ONNX is unavailable or depth unsupported).

class CopyBase : public OFX::ImageProcessor {
 protected:
  OFX::Image* _srcImg;
 public:
  explicit CopyBase(OFX::ImageEffect& e) : OFX::ImageProcessor(e), _srcImg(nullptr) {}
  void setSrcImg(OFX::Image* v) { _srcImg = v; }
};

template <class PIX, int nComponents>
class CopyProcessor : public CopyBase {
 public:
  explicit CopyProcessor(OFX::ImageEffect& e) : CopyBase(e) {}
  void multiThreadProcessImages(OfxRectI w) override {
    for (int y = w.y1; y < w.y2; ++y) {
      if (_effect.abort()) break;
      PIX* dst = static_cast<PIX*>(_dstImg->getPixelAddress(w.x1, y));
      for (int x = w.x1; x < w.x2; ++x) {
        const PIX* src = static_cast<const PIX*>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);
        if (src)
          for (int c = 0; c < nComponents; ++c) dst[c] = src[c];
        else
          for (int c = 0; c < nComponents; ++c) dst[c] = PIX(0);
        dst += nComponents;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

class DepthAnything3Plugin : public OFX::ImageEffect {
 protected:
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;
  OFX::StringParam* _modelFile;
  OFX::ChoiceParam* _computeUnits;
  OFX::IntParam* _longSide;
  OFX::BooleanParam* _autoRes;
  OFX::IntParam* _memBudget;
  OFX::IntParam* _threads;
  OFX::DoubleParam* _gain;
  OFX::BooleanParam* _inputACEScg;

 public:
  explicit DepthAnything3Plugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _modelFile = fetchStringParam(kParamModelFile);
    _computeUnits = fetchChoiceParam(kParamComputeUnits);
    _longSide = fetchIntParam(kParamLongSide);
    _autoRes = fetchBooleanParam(kParamAutoRes);
    _memBudget = fetchIntParam(kParamMemBudget);
    _threads = fetchIntParam(kParamThreads);
    _gain = fetchDoubleParam(kParamGain);
    _inputACEScg = fetchBooleanParam(kParamInputACEScg);
  }

  void render(const OFX::RenderArguments& args) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
#if DA3_WITH_ONNX
  bool renderDepth(const OFX::RenderArguments& args);
  std::string resolveModelPath();
  std::unique_ptr<da3::DepthEngine> _engine;
  std::string _engineKey;
#endif
};

////////////////////////////////////////////////////////////////////////////////

#if DA3_WITH_ONNX

// Locate DA3METRIC-LARGE.onnx inside the plugin bundle's Contents/Resources, by
// finding this binary's own path via dladdr.
static std::string bundleModelPath() {
#if defined(_WIN32)
  HMODULE hm = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(&bundleModelPath), &hm)) {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(hm, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      std::string p(buf, n);  // ...\Contents\Win64\DepthAnything3.ofx
      auto pos = p.rfind("\\Contents\\Win64\\");
      if (pos != std::string::npos) {
        std::string cand = p.substr(0, pos) + "\\Contents\\Resources\\DA3METRIC-LARGE.onnx";
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
  if (dladdr(reinterpret_cast<const void*>(&bundleModelPath), &info) && info.dli_fname) {
    std::string p(info.dli_fname);  // .../Contents/<arch>/DepthAnything3.ofx
    auto pos = p.rfind(marker);
    if (pos != std::string::npos) {
      std::string cand = p.substr(0, pos) + "/Contents/Resources/DA3METRIC-LARGE.onnx";
      struct stat st;
      if (stat(cand.c_str(), &st) == 0) return cand;
    }
  }
#endif
  return std::string();
}

std::string DepthAnything3Plugin::resolveModelPath() {
  std::string p;
  _modelFile->getValue(p);
  if (!p.empty()) return p;
  if (const char* env = std::getenv("DA3_MODEL_PATH")) return std::string(env);
  return bundleModelPath();  // bundled model (release build)
}

bool DepthAnything3Plugin::renderDepth(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));
  if (!dst.get() || !src.get()) return false;
  // Components must be RGBA (all we advertise); depth may be float / 8-bit / 16-bit
  // — normalized on read and quantized on write by the pixel helpers below.
  if (dst->getPixelComponents() != OFX::ePixelComponentRGBA ||
      src->getPixelComponents() != OFX::ePixelComponentRGBA)
    return false;
  const OFX::BitDepthEnum sbd = src->getPixelDepth();
  const OFX::BitDepthEnum dbd = dst->getPixelDepth();
  const bool srcInteger =
      (sbd == OFX::eBitDepthUByte || sbd == OFX::eBitDepthUShort);

  const OfxRectI rod = src->getRegionOfDefinition();
  const int W = rod.x2 - rod.x1;
  const int H = rod.y2 - rod.y1;
  if (W <= 0 || H <= 0) return false;

  const std::string modelPath = resolveModelPath();
  if (modelPath.empty()) {
    setPersistentMessage(OFX::Message::eMessageError, "",
                         "No ONNX model set (parameter 'Model file' or $DA3_MODEL_PATH).");
    return false;
  }

  int threads = 0;
  _threads->getValue(threads);
  int cu = 0;
  _computeUnits->getValue(cu);
  const da3::ComputeUnits units = static_cast<da3::ComputeUnits>(cu);

  // Resolution: either an explicit long side, or derived from a memory budget.
  bool autoRes = false;
  _autoRes->getValue(autoRes);
  int longSide = 504;
  if (autoRes) {
    int mb = 4096;
    _memBudget->getValue(mb);
    longSide = budgetToLongSide(mb);
  } else {
    _longSide->getValue(longSide);
  }
  // Aspect-preserving processing resolution (multiples of 14). The dynamic ONNX
  // model runs at any such resolution.
  const int mx = std::max(W, H);
  const double s = static_cast<double>(longSide) / static_cast<double>(mx);
  const int pw = da3::RoundToMultiple(std::max(14, static_cast<int>(std::lround(W * s))), 14);
  const int ph = da3::RoundToMultiple(std::max(14, static_cast<int>(std::lround(H * s))), 14);

  const std::string key = modelPath + "|" + std::to_string(cu) + "|" +
                          std::to_string(threads) + "|" + std::to_string(pw) + "x" +
                          std::to_string(ph);
  if (!_engine || key != _engineKey) {
    _engine = std::make_unique<da3::DepthEngine>(modelPath, units, threads, pw, ph);
    _engineKey = key;
    if (!_engine->last_error().empty()) {
      setPersistentMessage(OFX::Message::eMessageError, "", _engine->last_error());
      _engine.reset();
      _engineKey.clear();
      return false;
    }
  }

  // Build an RGB [0,1] buffer over the source RoD (row-major, y increasing).
  bool acescg = true;
  _inputACEScg->getValue(acescg);
  std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const void* s = src->getPixelAddress(rod.x1 + x, rod.y1 + y);
      float* d = rgb.data() + (static_cast<size_t>(y) * W + x) * 3;
      if (!s) {
        d[0] = d[1] = d[2] = 0.f;
        continue;
      }
      float t[3], a;
      readRGBA01(s, sbd, t, &a);  // normalized [0,1]
      // ACEScg->sRGB only applies to linear float input; integer input is already
      // display-referred sRGB (per the 8-bit = 0-255 sRGB Rec.709 assumption).
      if (acescg && !srcInteger) {
        acescgToSrgb(t[0], t[1], t[2], d);
      } else {
        d[0] = clamp01(t[0]);
        d[1] = clamp01(t[1]);
        d[2] = clamp01(t[2]);
      }
    }
  }

  da3::DepthResult res = _engine->Run(rgb.data(), W, H);
  if (res.depth.empty()) {
    setPersistentMessage(OFX::Message::eMessageError, "",
                         _engine->last_error().empty() ? "Inference failed" : _engine->last_error());
    return false;
  }
  clearPersistentMessage();

  double gain = 1.0;
  _gain->getValue(gain);
  const float scale = static_cast<float>(10.0 * gain);  // meters -> decimeters * gain

  // Write depth to R=G=B, alpha passed through, over the render window. Pixel address
  // is fetched per pixel because the element size depends on the dst bit depth.
  const OfxRectI w = args.renderWindow;
  for (int y = w.y1; y < w.y2; ++y) {
    if (abort()) break;
    for (int x = w.x1; x < w.x2; ++x) {
      int lx = x - rod.x1, ly = y - rod.y1;
      float z = 0.f;
      if (lx >= 0 && lx < W && ly >= 0 && ly < H)
        z = res.depth[static_cast<size_t>(ly) * W + lx] * scale;
      void* dpix = dst->getPixelAddress(x, y);
      if (!dpix) continue;
      // Alpha passthrough (normalized), read in the source's depth.
      float a = 1.0f;
      const void* s = src->getPixelAddress(x, y);
      if (s) {
        float t[3];
        readRGBA01(s, sbd, t, &a);
      }
      writeDepthPixel(dpix, dbd, z, a);
    }
  }
  return true;
}

#endif  // DA3_WITH_ONNX

void DepthAnything3Plugin::renderPassthrough(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const OFX::PixelComponentEnum comp = dst->getPixelComponents();
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));

  auto run = [&](CopyBase& p) {
    p.setSrcImg(src.get());
    p.setDstImg(dst.get());
    p.setRenderWindow(args.renderWindow);
    p.process();
  };
  if (comp == OFX::ePixelComponentRGBA) {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 4> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 4> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 4> p(*this); run(p); }
  } else {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 1> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 1> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 1> p(*this); run(p); }
  }
}

void DepthAnything3Plugin::render(const OFX::RenderArguments& args) {
#if DA3_WITH_ONNX
  const OFX::BitDepthEnum bd = _dstClip->getPixelDepth();
  // Depth runs for RGBA in float / 8-bit / 16-bit; renderDepth normalizes the input
  // to float and quantizes the output back to the host's depth.
  if (_dstClip->getPixelComponents() == OFX::ePixelComponentRGBA &&
      (bd == OFX::eBitDepthFloat || bd == OFX::eBitDepthUByte ||
       bd == OFX::eBitDepthUShort)) {
    if (renderDepth(args)) return;
    // fall through to passthrough on failure so the graph still renders
  }
#endif
  renderPassthrough(args);
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(DepthAnything3Factory, {}, {});

using namespace OFX;

void DepthAnything3Factory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  // Depth inference needs the whole image: no tiling.
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void DepthAnything3Factory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                              OFX::ContextEnum /*context*/) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->setTemporalClipAccess(false);
  src->setSupportsTiles(false);
  src->setIsMask(false);

  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    StringParamDescriptor* p = desc.defineStringParam(kParamModelFile);
    p->setLabel("Model file");
    p->setHint("Path to the DA3 ONNX model. Falls back to $DA3_MODEL_PATH, then the bundle.");
    p->setStringType(eStringTypeFilePath);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamInputACEScg);
    p->setLabel("Input is ACEScg");
    p->setHint("Convert ACEScg (AP1 linear) to sRGB before inference.");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamComputeUnits);
    p->setLabel("Compute units");
    p->setHint("CoreML hardware selection.");
    p->appendOption("All (ANE/GPU/CPU)");
    p->appendOption("CPU + GPU");
    p->appendOption("CPU + Neural Engine");
    p->appendOption("CPU only");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamAutoRes);
    p->setLabel("Auto resolution from memory budget");
    p->setHint("Derive the processing resolution from the memory budget below.");
    p->setDefault(false);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamMemBudget);
    p->setLabel("Memory budget (MB)");
    p->setHint("Approximate resource budget; maps to a processing resolution. "
               "CoreML has no VRAM cap, so on Apple Silicon resolution is the memory lever.");
    p->setRange(256, 65536);
    p->setDisplayRange(1024, 16384);
    p->setDefault(4096);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamLongSide);
    p->setLabel("Processing (long side)");
    p->setHint("Longest side of the inference resolution, aspect preserved and rounded to a "
               "multiple of 14. Lower = less memory/faster. Used unless Auto resolution is on.");
    p->setRange(140, 1540);
    p->setDisplayRange(224, 1036);
    p->setDefault(504);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamThreads);
    p->setLabel("Max threads");
    p->setHint("Intra-op thread cap (0 = ORT default).");
    p->setRange(0, 32);
    p->setDisplayRange(0, 16);
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamGain);
    p->setLabel("Depth gain");
    p->setHint("Multiplier on the decimeter output for correction.");
    p->setRange(0.0, 1000.0);
    p->setDisplayRange(0.1, 10.0);
    p->setDefault(1.0);
    page->addChild(*p);
  }
}

OFX::ImageEffect* DepthAnything3Factory::createInstance(OfxImageEffectHandle handle,
                                                        OFX::ContextEnum /*context*/) {
  return new DepthAnything3Plugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static DepthAnything3Factory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
  ids.push_back(&p);
#if DA3_WITH_ONNX
  da3reg::appendMoGe(ids);             // second plugin in the same bundle
  da3reg::appendLensDistortion(ids);   // third plugin
#endif
}
}  // namespace Plugin
}  // namespace OFX
