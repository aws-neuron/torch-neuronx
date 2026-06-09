#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace at::neuron {

// Abstract interface for compilation backends.
//
// The backend is a simple compiler: takes HLO bytes, returns NEFF bytes.
// All coordination (caching, file locks, pending dedup) stays in CompilationCache.
//
// Two implementations:
//   LocalCompilationBackend: calls kernel_execution.CompileToNeff() directly
//   ServiceCompilationBackend: sends HLO to shared service via RPC, blocks until done
class CompilationBackend {
 public:
  // Function that performs local HLO→NEFF compilation (wraps kernel_execution.CompileToNeff)
  using CompileFn = std::function<std::vector<uint8_t>()>;

  virtual ~CompilationBackend() = default;

  // Compile HLO to NEFF. Blocks until compilation is complete.
  //
  // Args:
  //   hlo_bytes: Serialized HLO IR
  //   ir_type: "StableHLO"
  //   op_name: Operation name for IR dumping
  //   cache_key: Cache key for IR dumping
  //   compile_fn: Local compilation function (used by local backend, ignored by service)
  //
  // Returns: compiled NEFF bytes (empty on failure)
  virtual std::vector<uint8_t> Compile(const std::vector<uint8_t>& hlo_bytes,
                                       const std::string& ir_type, const std::string& op_name,
                                       const std::string& cache_key,
                                       const std::string& optimization_level,
                                       const std::string& additional_args,
                                       CompileFn compile_fn) = 0;

  virtual std::string GetName() const = 0;
};

}  // namespace at::neuron
