#include "ServiceCompilationBackend.h"

#include <stdexcept>

namespace at::neuron {

ServiceCompilationBackend::ServiceCompilationBackend(const std::string& socket_path) {
  if (!client_.EnsureServiceRunning(socket_path)) {
    throw std::runtime_error("Failed to start compilation service at: " + socket_path);
  }
}

std::vector<uint8_t> ServiceCompilationBackend::Compile(
    const std::vector<uint8_t>& hlo_bytes, const std::string& ir_type, const std::string& op_name,
    const std::string& cache_key, const std::string& optimization_level,
    const std::string& additional_args, CompileFn /*compile_fn*/) {
  std::string error;
  auto neff_bytes = client_.CompileSync(hlo_bytes, ir_type, op_name, cache_key, optimization_level,
                                        additional_args, &error);
  if (neff_bytes.empty()) {
    throw std::runtime_error(error.empty() ? "Compilation failed: service returned empty NEFF"
                                           : error);
  }
  return neff_bytes;
}

}  // namespace at::neuron
