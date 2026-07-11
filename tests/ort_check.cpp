// Prints the available ONNX Runtime execution providers and confirms the CoreML
// EP is present. Exit code 0 if CoreML is available, 1 otherwise.
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>

int main() {
  bool coreml = false;
  std::cout << "ONNX Runtime available execution providers:\n";
  for (const auto& p : Ort::GetAvailableProviders()) {
    std::cout << "  - " << p << "\n";
    if (p.find("CoreML") != std::string::npos) coreml = true;
  }
  std::cout << "CoreML EP available: " << (coreml ? "YES" : "NO") << "\n";
  return coreml ? 0 : 1;
}
