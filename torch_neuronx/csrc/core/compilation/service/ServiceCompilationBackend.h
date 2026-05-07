#pragma once

#include <string>

#include "torch_neuronx/csrc/core/compilation/CompilationBackend.h"
#include "torch_neuronx/csrc/core/compilation/service/CompilationServiceClient.h"

namespace at::neuron {

// Service backend: sends HLO to shared compilation service via synchronous RPC.
// The service compiles the HLO, writes NEFF to persistent cache, returns the NEFF bytes.
// All coordination (caching, file locks, pending dedup) stays in CompilationCache.
class ServiceCompilationBackend : public CompilationBackend {
 public:
  explicit ServiceCompilationBackend(const std::string& socket_path);

  std::vector<uint8_t> Compile(const std::vector<uint8_t>& hlo_bytes, const std::string& ir_type,
                               const std::string& op_name, const std::string& cache_key,
                               const std::string& optimization_level,
                               const std::string& additional_args, CompileFn compile_fn) override;

  std::string GetName() const override { return "service"; }

 private:
  CompilationServiceClient client_;
};

}  // namespace at::neuron
