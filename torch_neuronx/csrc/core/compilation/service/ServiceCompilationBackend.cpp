#include "ServiceCompilationBackend.h"

#include <stdexcept>

namespace at::neuron {

ServiceCompilationBackend::ServiceCompilationBackend(const std::string& socket_path) {
  if (!client_.EnsureServiceRunning(socket_path)) {
    throw std::runtime_error("Failed to start compilation service at: " + socket_path);
  }
}

std::vector<uint8_t> ServiceCompilationBackend::Compile(const std::vector<uint8_t>& hlo_bytes,
                                                        const std::string& persistent_cache_key,
                                                        const std::string& ir_type,
                                                        const std::string& optimization_level,
                                                        const std::string& additional_args,
                                                        CompileFn /*compile_fn*/) {
  auto neff_bytes = client_.CompileSync(hlo_bytes, persistent_cache_key, ir_type,
                                        optimization_level, additional_args);
  if (neff_bytes.empty()) {
    throw std::runtime_error("Compilation failed: service returned empty NEFF");
  }
  return neff_bytes;
}

}  // namespace at::neuron
