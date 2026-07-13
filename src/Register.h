// Copyright the humbaba authors.
// SPDX-License-Identifier: Apache-2.0
//
// Lets the single .ofx binary register more than one plugin (the depth plugin
// plus the MoGe focal plugin) from OFX::Plugin::getPluginIDs.
#pragma once

#include "ofxsImageEffect.h"

namespace da3reg {
// Appends the MoGe focal-length plugin factory to the host's plugin list.
void appendMoGe(OFX::PluginFactoryArray& ids);
// Appends the Lens Distortion plugin factory to the host's plugin list.
void appendLensDistortion(OFX::PluginFactoryArray& ids);

// Best-effort user messages. DaVinci Resolve's render context does not support the
// OFX persistent-message suite (MessageSuiteV2); calling set/clearPersistentMessage
// there THROWS, which unwinds out of the render action and the host reports the whole
// render as kOfxStatErrUnsupported. Wrap the calls so a missing/unsupported message
// suite can never fail a render (the message is simply dropped on such hosts).
inline void SafeSetMessage(OFX::ImageEffect& e, OFX::Message::MessageTypeEnum type,
                           const std::string& id, const std::string& msg) {
  try { e.setPersistentMessage(type, id, msg); } catch (...) {}
}
inline void SafeClearMessage(OFX::ImageEffect& e) {
  try { e.clearPersistentMessage(); } catch (...) {}
}
}  // namespace da3reg
