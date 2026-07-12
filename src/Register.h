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
}  // namespace da3reg
