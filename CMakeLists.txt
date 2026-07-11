cmake_minimum_required(VERSION 3.20)
project(DepthAnything3OFX VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(DA3_UNIVERSAL "Build a universal arm64+x86_64 binary (macOS)" ON)
option(DA3_WITH_ONNX "Link ONNX Runtime (CoreML EP) and enable depth inference" ON)
set(DA3_ORT_VERSION "1.27.1" CACHE STRING "ONNX Runtime version to fetch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The prebuilt ONNX Runtime with the CoreML EP is arm64-only, so an ONNX-enabled
# build is arm64-only (which is also what the Apple Neural Engine path requires).
if(APPLE AND DA3_WITH_ONNX)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "arm64 (ONNX Runtime CoreML)" FORCE)
elseif(APPLE AND DA3_UNIVERSAL)
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
# ONNX Runtime (prebuilt, arm64, with the CoreML execution provider compiled in).
# Populate-only: it's a binary release (headers + dylib), not a CMake project.
# ---------------------------------------------------------------------------
if(DA3_WITH_ONNX)
  FetchContent_Declare(onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${DA3_ORT_VERSION}/onnxruntime-osx-arm64-${DA3_ORT_VERSION}.tgz"
    URL_HASH SHA256=e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6
  )
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_DIR "${onnxruntime_SOURCE_DIR}")
  set(ORT_DYLIB "${ORT_DIR}/lib/libonnxruntime.${DA3_ORT_VERSION}.dylib")

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DYLIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
  )
endif()

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

if(DA3_WITH_ONNX)
  target_sources(DepthAnything3 PRIVATE src/DepthEngine.cpp)
  target_link_libraries(DepthAnything3 PRIVATE onnxruntime)
  target_compile_definitions(DepthAnything3 PRIVATE DA3_WITH_ONNX=1)
  # Find the bundled dylib relative to the .ofx binary at runtime.
  set_target_properties(DepthAnything3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@loader_path/../Frameworks"
  )
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

# Bundle the ONNX Runtime dylib inside the .ofx.bundle (Contents/Frameworks) so
# the plugin is self-contained. The install name is @rpath/..., resolved by the
# plugin's @loader_path/../Frameworks rpath set above.
if(DA3_WITH_ONNX)
  # The plugin links against the install name @rpath/libonnxruntime.1.dylib, so the
  # bundled file must carry that exact name (the tarball's real file is versioned).
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Frameworks"
    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_DYLIB}"
            "${BUNDLE_DIR}/Contents/Frameworks/libonnxruntime.1.dylib"
    COMMENT "Bundling ONNX Runtime dylib"
    VERBATIM
  )
endif()

# Optionally bundle the ONNX model into Contents/Resources (release/self-contained
# build). Configure with -DDA3_MODEL_FILE=/path/to/DA3METRIC-LARGE.onnx.
if(DA3_WITH_ONNX AND DA3_MODEL_FILE)
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy "${DA3_MODEL_FILE}"
            "${BUNDLE_DIR}/Contents/Resources/DA3METRIC-LARGE.onnx"
    COMMENT "Bundling ONNX model into Resources"
    VERBATIM
  )
endif()

# ---------------------------------------------------------------------------
# Provider-check tool: prints the ONNX Runtime execution providers (verifies
# CoreML is available). Built only with ONNX enabled.
# ---------------------------------------------------------------------------
if(DA3_WITH_ONNX)
  add_executable(ort_check tests/ort_check.cpp)
  target_link_libraries(ort_check PRIVATE onnxruntime)
  set_target_properties(ort_check PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib"
  )
endif()

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
