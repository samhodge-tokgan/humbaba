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
# ONNX Runtime (prebuilt binary release: headers + shared library).
#   macOS arm64 -> CoreML EP ; Linux x64 -> CUDA EP (GPU) + CPU.
# Populate-only (not a CMake project).
# ---------------------------------------------------------------------------
set(ORT_BASE "https://github.com/microsoft/onnxruntime/releases/download/v${DA3_ORT_VERSION}")
if(DA3_WITH_ONNX)
  set(ORT_HASH_ARG "")
  if(APPLE)
    set(ORT_URL "${ORT_BASE}/onnxruntime-osx-arm64-${DA3_ORT_VERSION}.tgz")
    set(ORT_HASH_ARG URL_HASH SHA256=e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6)
    set(ORT_SHLIB "lib/libonnxruntime.${DA3_ORT_VERSION}.dylib")
    set(DA3_ACCEL "CoreML")
  elseif(UNIX)  # Linux
    set(ORT_URL "${ORT_BASE}/onnxruntime-linux-x64-gpu_cuda12-${DA3_ORT_VERSION}.tgz")
    # (hash added once pinned; FetchContent fetches without it meanwhile)
    set(ORT_SHLIB "lib/libonnxruntime.so.${DA3_ORT_VERSION}")
    set(DA3_ACCEL "CUDA")
  else()
    message(FATAL_ERROR "Unsupported platform for DA3_WITH_ONNX")
  endif()

  FetchContent_Declare(onnxruntime URL "${ORT_URL}" ${ORT_HASH_ARG})
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_DIR "${onnxruntime_SOURCE_DIR}")
  set(ORT_DYLIB "${ORT_DIR}/${ORT_SHLIB}")

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DYLIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
  )
  message(STATUS "ONNX Runtime accelerator EP: ${DA3_ACCEL}")
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
# Per-platform .ofx.bundle layout (OFX packaging spec).
# ---------------------------------------------------------------------------
if(APPLE)
  set(OFX_ARCH_DIR "MacOS")        # Contents/MacOS/*.ofx
  set(OFX_LIB_DIR "Frameworks")    # bundled shared libs
  set(OFX_RPATH "@loader_path/../Frameworks")
  set(OFX_LIB_BUNDLED "libonnxruntime.1.dylib")
else()                             # Linux
  # OFX packaging spec: the 64-bit Linux arch dir is "Linux-x86-64" (hyphens).
  # Using an underscore makes OFX hosts (Natron) silently skip the binary.
  set(OFX_ARCH_DIR "Linux-x86-64") # Contents/Linux-x86-64/*.ofx
  set(OFX_LIB_DIR "Linux-x86-64")  # .so alongside the .ofx
  set(OFX_RPATH "$ORIGIN")
  set(OFX_LIB_BUNDLED "libonnxruntime.so.1")
endif()

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
  target_sources(DepthAnything3 PRIVATE
    src/DepthEngine.cpp
    src/MoGeEngine.cpp
    src/MoGePlugin.cpp
    src/LensModel.cpp
    src/LensDistortionPlugin.cpp
    src/AnyCalibEngine.cpp
  )
  target_link_libraries(DepthAnything3 PRIVATE onnxruntime)
  target_compile_definitions(DepthAnything3 PRIVATE DA3_WITH_ONNX=1)
  # Find the bundled shared library relative to the .ofx binary at runtime.
  set_target_properties(DepthAnything3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${OFX_RPATH}"
  )
endif()

# ---------------------------------------------------------------------------
# Assemble the .ofx.bundle (macOS layout: Contents/{Info.plist,MacOS/*.ofx})
# ---------------------------------------------------------------------------
set(OFX_BUNDLE_NAME "DepthAnything3")
set(OFX_BUNDLE_BINARY "DepthAnything3.ofx")
set(OFX_BUNDLE_IDENTIFIER "com.tokgan.openfx.DepthAnything3")
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle")

add_custom_command(TARGET DepthAnything3 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:DepthAnything3>" "${BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/${OFX_BUNDLE_BINARY}"
  COMMENT "Assembling ${OFX_BUNDLE_NAME}.ofx.bundle (${OFX_ARCH_DIR})"
  VERBATIM
)

# Info.plist is a macOS bundle requirement (Linux bundles don't use it).
if(APPLE)
  configure_file("${CMAKE_SOURCE_DIR}/packaging/Info.plist.in" "${CMAKE_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/Info.plist" "${BUNDLE_DIR}/Contents/Info.plist"
    VERBATIM)
endif()

# Bundle the ONNX Runtime shared library inside the .ofx.bundle so the plugin is
# self-contained. The plugin links against the SONAME (@rpath/libonnxruntime.1.dylib
# on macOS, libonnxruntime.so.1 on Linux), so the bundled file must carry that name
# (the tarball's real file is fully versioned).
if(DA3_WITH_ONNX)
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_DYLIB}"
            "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/${OFX_LIB_BUNDLED}"
    COMMENT "Bundling ONNX Runtime shared library"
    VERBATIM
  )
  # On Linux the CUDA EP lives in a separate shared library (libonnxruntime_providers_cuda.so)
  # plus a shared provider lib; bundle them alongside so the EP loads.
  if(NOT APPLE)
    file(GLOB ORT_PROVIDER_LIBS "${ORT_DIR}/lib/libonnxruntime_providers_*.so")
    foreach(pl ${ORT_PROVIDER_LIBS})
      add_custom_command(TARGET DepthAnything3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${pl}" "${BUNDLE_DIR}/Contents/${OFX_LIB_DIR}/"
        VERBATIM)
    endforeach()
  endif()
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

# Optionally bundle the MoGe model too (for the MoGe Focal plugin).
if(DA3_WITH_ONNX AND DA3_MOGE_MODEL_FILE)
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy "${DA3_MOGE_MODEL_FILE}"
            "${BUNDLE_DIR}/Contents/Resources/moge-2-vitb.onnx"
    COMMENT "Bundling MoGe model into Resources"
    VERBATIM
  )
endif()

# Optionally bundle the AnyCalib model (for the Lens Distortion estimator).
if(DA3_WITH_ONNX AND DA3_ANYCALIB_MODEL_FILE)
  add_custom_command(TARGET DepthAnything3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy "${DA3_ANYCALIB_MODEL_FILE}"
            "${BUNDLE_DIR}/Contents/Resources/anycalib_dist.onnx"
    COMMENT "Bundling AnyCalib model into Resources"
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
  if(APPLE)
    set(OFX_INSTALL_DIR "$ENV{HOME}/Library/OFX/Plugins")
  else()  # Linux
    set(OFX_INSTALL_DIR "$ENV{HOME}/OFX/Plugins")
  endif()
endif()
install(DIRECTORY "${BUNDLE_DIR}" DESTINATION "${OFX_INSTALL_DIR}" USE_SOURCE_PERMISSIONS)

add_custom_target(install-local
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_INSTALL_DIR}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR}" "${OFX_INSTALL_DIR}/${OFX_BUNDLE_NAME}.ofx.bundle"
  DEPENDS DepthAnything3
  COMMENT "Installing bundle to ${OFX_INSTALL_DIR}"
)
