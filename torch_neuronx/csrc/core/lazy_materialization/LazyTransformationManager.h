#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "OperationTransformations.h"
#include "TransformationTypes.h"

// Forward declarations

namespace at::neuron {
class OperationContext;
class XLACompilableKernelExecution;
struct OperationExecutionTask;
}  // namespace at::neuron

namespace c10_neuron {
namespace lazy {

// Key for input deduplication: inputs only share a DMA slot when they have the
// same storage base pointer AND the same shape. Different shapes cannot share a
// slot because ApplyInputDedup picks one shape for the buffer size.
struct InputDedupKey {
  uintptr_t base_ptr;
  std::vector<int64_t> shape;
  bool operator==(const InputDedupKey& other) const {
    return base_ptr == other.base_ptr && shape == other.shape;
  }
};

struct InputDedupKeyHash {
  size_t operator()(const InputDedupKey& k) const {
    size_t h = std::hash<uintptr_t>{}(k.base_ptr);
    for (auto d : k.shape) {
      h ^= std::hash<int64_t>{}(d) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

// Compute input deduplication map from pointers and contexts.
// Returns {dedup_map, deduped_count} where dedup_map[i] is the slot index for input i.
// Inputs share a slot only when they have the same storage base pointer AND shape.
inline std::pair<std::vector<size_t>, size_t> ComputeInputDedupMap(
    const std::vector<void*>& input_ptrs,
    const std::vector<at::neuron::TensorContext>& input_contexts) {
  std::unordered_map<InputDedupKey, size_t, InputDedupKeyHash> base_to_idx;
  std::vector<size_t> dedup_map(input_ptrs.size());
  size_t deduped_count = 0;
  for (size_t i = 0; i < input_ptrs.size(); ++i) {
    uintptr_t base_ptr =
        reinterpret_cast<uintptr_t>(input_ptrs[i]) -
        (input_contexts[i].get_storage_offset() * input_contexts[i].get_element_size());
    InputDedupKey key{base_ptr, input_contexts[i].shape};
    auto [it, inserted] = base_to_idx.try_emplace(std::move(key), deduped_count);
    if (inserted) {
      dedup_map[i] = deduped_count;
      deduped_count++;
    } else {
      dedup_map[i] = it->second;
    }
  }
  return {std::move(dedup_map), deduped_count};
}

// Main manager for lazy transformations
// Provides centralized handling of all lazy transformation operations
class LazyTransformationManager {
 public:
  // Process pending transformations for operation inputs (prologue)
  // This is the main preprocessing entry point called from OperationExecutionEngine
  // Throws an exception if preprocessing fails
  static void ProcessOperationInputs(at::neuron::OperationContext* operation_context);

  // Process pending transformations for operation outputs (epilogue)
  // Generates inverse transformation MLIR and merges it after the operation.
  // Called from OperationExecutionEngine after ProcessOperationInputs.
  static void ProcessOperationOutputs(at::neuron::OperationContext* operation_context);

  // Cache management functions for merged operation cache
  static void ClearMergedOperationCache();
  static size_t GetMergedOperationCacheSize();

 private:
  // Process transformations using MLIR merging with OperationPrologue
  // Returns merged MLIR bytes, throws exception on failure
  static std::vector<uint8_t> ProcessPrologueWithMlirMerge(
      const PrologueResult& prologue_result, const std::string& op_name,
      at::neuron::XLACompilableKernelExecution* compilable_kernel);

  // Process epilogue transformations using MLIR merging with OperationEpilogue
  // Returns merged MLIR bytes (operation + epilogue), throws exception on failure
  static std::vector<uint8_t> ProcessEpilogueWithMlirMerge(
      const EpilogueResult& epilogue_result, const std::string& op_name,
      at::neuron::XLACompilableKernelExecution* compilable_kernel);
};

}  // namespace lazy
}  // namespace c10_neuron
