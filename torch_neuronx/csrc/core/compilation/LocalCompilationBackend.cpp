#include "LocalCompilationBackend.h"

namespace at::neuron {

std::vector<uint8_t> LocalCompilationBackend::Compile(const std::vector<uint8_t>& /*hlo_bytes*/,
                                                      const std::string& /*ir_type*/,
                                                      const std::string& /*optimization_level*/,
                                                      const std::string& /*additional_args*/,
                                                      CompileFn compile_fn) {
  return compile_fn();
}

}  // namespace at::neuron
