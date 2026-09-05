#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"

namespace at::neuron {

// Abstract interface for compilation backends.
//
// The backend is a simple compiler: takes HLO bytes, returns a CompileResult
// (NEFF bytes plus, for WLO-enabled compiles, the wrapped_neff.hlo protobuf).
// All coordination (caching, file locks, pending dedup) stays in CompilationCache.
//
// Two implementations:
//   LocalCompilationBackend: calls kernel_execution.CompileToNeffResult() directly
//   ServiceCompilationBackend: sends HLO to shared service via RPC, blocks until done
//     (returns NEFF only; wrapped_neff.hlo is empty — see TODO(wlo-service))
class CompilationBackend {
 public:
  // Performs the local HLO→NEFF compile (wraps kernel_execution.CompileToNeffResult()).
  using CompileFn = std::function<CompileResult()>;

  virtual ~CompilationBackend() = default;

  // Compile HLO to NEFF, blocking until done. compile_fn does the local compile (used by the
  // local backend, ignored by the service). Returns neff_bytes empty on failure;
  // wrapped_neff_hlo is populated only for WLO compiles on the local backend.
  virtual CompileResult Compile(const std::vector<uint8_t>& hlo_bytes, const std::string& ir_type,
                                const std::string& op_name, const std::string& cache_key,
                                const std::string& optimization_level,
                                const std::string& additional_args, CompileFn compile_fn) = 0;

  virtual std::string GetName() const = 0;
};

}  // namespace at::neuron
