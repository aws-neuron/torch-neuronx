#pragma once

#include <memory>
#include <string>
#include <vector>

#include "OperationPrologue.h"
#include "TransformationTypes.h"

// Forward declarations

namespace at::neuron {
class OperationContext;
class XLACompilableKernelExecution;
struct OperationExecutionTask;
}  // namespace at::neuron

namespace c10_neuron {
namespace lazy {

// Main manager for lazy transformations
// Provides centralized handling of all lazy transformation operations
class LazyTransformationManager {
 public:
  // Process pending transformations for operation inputs
  // This is the main preprocessing entry point called from OperationExecutionEngine
  // Throws an exception if preprocessing fails
  static void ProcessOperationInputs(at::neuron::OperationContext* operation_context);

  // Cache management functions for merged operation cache
  static void ClearMergedOperationCache();
  static size_t GetMergedOperationCacheSize();

 private:
  // Process transformations using MLIR merging with OperationPrologue
  // Returns merged MLIR bytes, throws exception on failure
  static std::vector<uint8_t> ProcessPrologueWithMlirMerge(
      const PrologueResult& prologue_result, const std::string& op_name,
      at::neuron::XLACompilableKernelExecution* compilable_kernel);
};

}  // namespace lazy
}  // namespace c10_neuron
