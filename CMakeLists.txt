cmake_minimum_required(VERSION 3.20)
project(DepthAnything3OFX VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(DA3_UNIVERSAL "Build a universal arm64+x86_64 binary (macOS)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(APPLE AND DA3_UNIVERSAL)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "OFX host arch must match" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vendor the OpenFX SDK (populate only; we compile the Support library ourselves
# rather than adding OpenFX's own CMake targets). Pointing SOURCE_SUBDIR at a
# directory with no CMakeLists.txt makes MakeAvailable download without building.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(openfx
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
  GIT_TAG OFX_Release_1.5.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR include            # no CMakeLists here -> populate only
)
FetchContent_MakeAvailable(openfx)

set(OFX_ROOT "${openfx_SOURCE_DIR}")
set(OFX_SUPPORT "${OFX_ROOT}/Support")

# ---------------------------------------------------------------------------
# OpenFX C++ Support library
# ---------------------------------------------------------------------------
add_library(ofxsupport STATIC
  ${OFX_SUPPORT}/Library/ofxsCore.cpp
  ${OFX_SUPPORT}/Library/ofxsImageEffect.cpp
  ${OFX_SUPPORT}/Library/ofxsInteract.cpp
  ${OFX_SUPPORT}/Library/ofxsLog.cpp
  ${OFX_SUPPORT}/Library/ofxsMultiThread.cpp
  ${OFX_SUPPORT}/Library/ofxsParams.cpp
  ${OFX_SUPPORT}/Library/ofxsProperty.cpp
  ${OFX_SUPPORT}/Library/ofxsPropertyValidation.cpp
)
target_include_directories(ofxsupport PUBLIC
  ${OFX_ROOT}/include
  ${OFX_SUPPORT}/include
  ${OFX_SUPPORT}/Plugins/include
)
# The Support library expects the plugin to define getPluginIDs; kOfxThreadsuite etc.
target_compile_definitions(ofxsupport PUBLIC OFX_SUPPORTS_MULTITHREAD)
set_target_properties(ofxsupport PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# The plugin: a loadable module named DepthAnything3.ofx
# ---------------------------------------------------------------------------
add_library(DepthAnything3 MODULE src/DepthAnything3Plugin.cpp)
target_link_libraries(DepthAnything3 PRIVATE ofxsupport)
set_target_properties(DepthAnything3 PROPERTIES
  PREFIX ""
  OUTPUT_NAME "DepthAnything3"
  SUFFIX ".ofx"
)
if(APPLE)
  # OFX plugins are Mach-O bundles.
  target_link_options(DepthAnything3 PRIVATE -bundle)
endif()

# ---------------------------------------------------------------------------
# Assemble the .ofx.bundle (macOS layout: Contents/{Info.plist,MacOS/*.ofx})
# ---------------------------------------------------------------------------
set(OFX_BUNDLE_NAME "DepthAnything3")
set(OFX_BUNDLE_BINARY "DepthAnything3.ofx")
set(OFX_BUNDLE_IDENTIFIER "com.tokgan.openfx.DepthAnything3")
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle")

configure_file(
  "${CMAKE_SOURCE_DIR}/packaging/Info.plist.in"
  "${CMAKE_BINARY_DIR}/Info.plist"
  @ONLY
)

add_custom_command(TARGET DepthAnything3 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/MacOS"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/Info.plist" "${BUNDLE_DIR}/Contents/Info.plist"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:DepthAnything3>" "${BUNDLE_DIR}/Contents/MacOS/${OFX_BUNDLE_BINARY}"
  COMMENT "Assembling ${OFX_BUNDLE_NAME}.ofx.bundle"
  VERBATIM
)

# ---------------------------------------------------------------------------
# Install: drop the bundle into the user's OFX plugin dir (override with
# -DOFX_INSTALL_DIR=...). Use `cmake --install build` or the `install-local` target.
# ---------------------------------------------------------------------------
if(NOT DEFINED OFX_INSTALL_DIR)
  set(OFX_INSTALL_DIR "$ENV{HOME}/Library/OFX/Plugins")
endif()
install(DIRECTORY "${BUNDLE_DIR}" DESTINATION "${OFX_INSTALL_DIR}" USE_SOURCE_PERMISSIONS)

add_custom_target(install-local
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_INSTALL_DIR}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR}" "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  DEPENDS DepthAnything3
  COMMENT "Installing bundle to ${OFX_INSTALL_DIR}"
)
