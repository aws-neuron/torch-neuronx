#pragma once

#include "torch_neuronx/csrc/core/compilation/CompilationBackend.h"

namespace at::neuron {

// Local backend: calls compile_fn() directly on the calling thread.
// This is the original behavior — neuronx-cc runs in-process.
class LocalCompilationBackend : public CompilationBackend {
 public:
  std::vector<uint8_t> Compile(const std::vector<uint8_t>& hlo_bytes, const std::string& ir_type,
                               const std::string& optimization_level,
                               const std::string& additional_args, CompileFn compile_fn) override;

  std::string GetName() const override { return "local"; }
};

}  // namespace at::neuron
