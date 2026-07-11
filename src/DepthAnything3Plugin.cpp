// Copyright the openfx-onnx-depthanything3 authors.
// SPDX-License-Identifier: Apache-2.0
//
// Depth Anything 3 OpenFX plugin — M2 skeleton.
//
// At this milestone the effect is a straight passthrough (copies the source clip
// to the output) that builds as a universal .ofx bundle and loads in an OFX host.
// M3 replaces the copy processor with the ACEScg -> ONNX Runtime (CoreML) depth
// inference and writes float32 decimeter depth to the output.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <cstdio>
#include <memory>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.H"

#define kPluginName "Depth Anything 3"
#define kPluginGrouping "TokGan"
#define kPluginDescription \
  "Predict metric depth (decimeters, float32) from an ACEScg image using " \
  "Depth Anything 3 via ONNX Runtime with hardware acceleration."
#define kPluginIdentifier "com.tokgan.openfx.DepthAnything3"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

////////////////////////////////////////////////////////////////////////////////
// Non-templated processor base so setupAndProcess() can wire the source image
// without knowing the pixel type.

class CopyBase : public OFX::ImageProcessor {
protected:
  OFX::Image *_srcImg;

public:
  explicit CopyBase(OFX::ImageEffect &instance)
      : OFX::ImageProcessor(instance), _srcImg(nullptr) {}

  void setSrcImg(OFX::Image *v) { _srcImg = v; }
};

// Templated copy: byte / short / float, RGBA or single-channel Alpha.
template <class PIX, int nComponents>
class CopyProcessor : public CopyBase {
public:
  explicit CopyProcessor(OFX::ImageEffect &instance) : CopyBase(instance) {}

  void multiThreadProcessImages(OfxRectI procWindow) override {
    for (int y = procWindow.y1; y < procWindow.y2; ++y) {
      if (_effect.abort()) break;

      PIX *dstPix = static_cast<PIX *>(_dstImg->getPixelAddress(procWindow.x1, y));

      for (int x = procWindow.x1; x < procWindow.x2; ++x) {
        const PIX *srcPix =
            static_cast<const PIX *>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);

        if (srcPix) {
          for (int c = 0; c < nComponents; ++c) dstPix[c] = srcPix[c];
        } else {
          for (int c = 0; c < nComponents; ++c) dstPix[c] = PIX(0);
        }
        dstPix += nComponents;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin instance. */

class DepthAnything3Plugin : public OFX::ImageEffect {
protected:
  // Managed by the host; do not delete.
  OFX::Clip *_dstClip;
  OFX::Clip *_srcClip;

public:
  explicit DepthAnything3Plugin(OfxImageEffectHandle handle)
      : OFX::ImageEffect(handle), _dstClip(nullptr), _srcClip(nullptr) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
  }

  void render(const OFX::RenderArguments &args) override;

private:
  void setupAndProcess(CopyBase &processor, const OFX::RenderArguments &args);
};

void DepthAnything3Plugin::setupAndProcess(CopyBase &processor,
                                           const OFX::RenderArguments &args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);

  const OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
  const OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));
  if (src.get()) {
    if (src->getPixelDepth() != dstBitDepth ||
        src->getPixelComponents() != dstComponents) {
      OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    }
  }

  processor.setSrcImg(src.get());
  processor.setDstImg(dst.get());
  processor.setRenderWindow(args.renderWindow);
  processor.process();
}

void DepthAnything3Plugin::render(const OFX::RenderArguments &args) {
  const OFX::BitDepthEnum dstBitDepth = _dstClip->getPixelDepth();
  const OFX::PixelComponentEnum dstComponents = _dstClip->getPixelComponents();

  if (dstComponents == OFX::ePixelComponentRGBA) {
    switch (dstBitDepth) {
      case OFX::eBitDepthUByte: {
        CopyProcessor<unsigned char, 4> p(*this);
        setupAndProcess(p, args);
      } break;
      case OFX::eBitDepthUShort: {
        CopyProcessor<unsigned short, 4> p(*this);
        setupAndProcess(p, args);
      } break;
      case OFX::eBitDepthFloat: {
        CopyProcessor<float, 4> p(*this);
        setupAndProcess(p, args);
      } break;
      default:
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
  } else {
    switch (dstBitDepth) {
      case OFX::eBitDepthUByte: {
        CopyProcessor<unsigned char, 1> p(*this);
        setupAndProcess(p, args);
      } break;
      case OFX::eBitDepthUShort: {
        CopyProcessor<unsigned short, 1> p(*this);
        setupAndProcess(p, args);
      } break;
      case OFX::eBitDepthFloat: {
        CopyProcessor<float, 1> p(*this);
        setupAndProcess(p, args);
      } break;
      default:
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The factory. */

mDeclarePluginFactory(DepthAnything3Factory, {}, {});

using namespace OFX;

void DepthAnything3Factory::describe(OFX::ImageEffectDescriptor &desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);

  // Depth inference works on float; byte/short supported for the passthrough skeleton.
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  desc.setSupportsTiles(true);  // revisited in M3/M4 (whole-image inference / tiling)
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void DepthAnything3Factory::describeInContext(OFX::ImageEffectDescriptor &desc,
                                              OFX::ContextEnum /*context*/) {
  ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  srcClip->addSupportedComponent(ePixelComponentRGBA);
  srcClip->addSupportedComponent(ePixelComponentAlpha);
  srcClip->setTemporalClipAccess(false);
  srcClip->setSupportsTiles(true);
  srcClip->setIsMask(false);

  ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
  dstClip->addSupportedComponent(ePixelComponentRGBA);
  dstClip->addSupportedComponent(ePixelComponentAlpha);
  dstClip->setSupportsTiles(true);
}

OFX::ImageEffect *DepthAnything3Factory::createInstance(OfxImageEffectHandle handle,
                                                        OFX::ContextEnum /*context*/) {
  return new DepthAnything3Plugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray &ids) {
  static DepthAnything3Factory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
  ids.push_back(&p);
}
}  // namespace Plugin
}  // namespace OFX
