#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "TransformationTypes.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace c10_neuron {
namespace lazy {

// ============================================================================
// Transformation MLIR Utilities - Standalone helper functions
// ============================================================================

// Utility namespace for generating StableHLO MLIR representations
// from transformation tasks. These are simple utility functions that delegate to
// the actual MLIR generators in the mlir_generators namespace.
namespace transformation_utils {

// Generate StableHLO MLIR for a transformation task
std::string GenerateMlir(const TransformationTask& task);

}  // namespace transformation_utils

// ============================================================================
// PrologueResult - Result of prologue generation
// ============================================================================

// Contains the merged prologue StableHLO and tensor mappings
// The prologue takes ALL operation inputs and produces transformed outputs
struct PrologueResult {
  bool success = true;
  std::string error_message;
  bool has_transformations = false;

  // Merged prologue StableHLO for all transformations
  std::string prologue_mlir_str;
  std::string prologue_cache_key;

  // Mapping to identify which inputs were transformed
  std::vector<bool> input_was_transformed;

  // Original transformation tasks (for metadata updates in LazyTransformationManager)
  std::vector<TransformationTask> tasks;
};

// ============================================================================
// OperationPrologue - Generates merged transformation prologues
// ============================================================================

// Main class for generating operation prologues from pending transformations
//
// The prologue generation process:
// 1. Collect transformation tasks from input tensors
// 2. Group tasks by input index
// 3. For each input, merge its transformations sequentially
// 4. Merge all per-input transformation HLOs into a single prologue
// 5. Ensure prologue takes ALL inputs and produces transformed outputs
//
// The resulting prologue can be directly merged with the original operation HLO
class OperationPrologue {
 public:
  // Generate a merged prologue StableHLO from input pointers and contexts
  //
  // This is the main entry point. It:
  // - Uses kernel-provided transformations
  // - Merges transformations per-input, then across all inputs
  // - Returns a prologue that takes ALL inputs and produces transformed outputs
  //
  // Parameters:
  //   input_ptrs: All input data pointers for the operation
  //   input_contexts: Tensor contexts for all inputs (contains shape, dtype, etc.)
  //   op_name: Name of the operation (for logging and caching)
  //   kernel_transforms: Per-input transformations from kernel metadata
  //   prologue_cache_key: Pre-computed cache key (computed by ComputePrologueCacheKey)
  //
  // Returns:
  //   PrologueResult containing the merged prologue
  static PrologueResult GeneratePrologue(
      const std::vector<void*>& input_ptrs,
      const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms,
      const std::string& prologue_cache_key);

  // Compute prologue cache key efficiently without creating transformation tasks
  //
  // This should be called once by the caller (LazyTransformationManager) to generate
  // the cache key, which is then passed to GeneratePrologue and used for caching.
  //
  // Parameters:
  //   input_ptrs: Input data pointers
  //   input_contexts: Tensor contexts (for shape/dtype of NONE transformations)
  //   op_name: Operation name
  //   kernel_transforms: Per-input transformations from kernel metadata
  //
  // Returns:
  //   Cache key string, or empty string if no transformations are needed
  static std::string ComputePrologueCacheKey(
      const std::vector<void*>& input_ptrs,
      const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms = {});

 private:
  // Collect transformation tasks from kernel transformations
  // Converts them into TransformationTask objects
  // Uses input_contexts to get dtype information for NONE transformations
  static std::vector<TransformationTask> CollectTransformationTasks(
      const std::vector<void*>& input_ptrs,
      const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms);

  // Merge all transformation tasks into a single prologue HLO
  //
  // Multi-level merging process:
  // Level 1: Merge transformations per-input (chain consecutive transformations)
  // Level 2: Merge all per-input HLOs into complete prologue
  // Level 3: Ensure prologue handles ALL inputs (pass-through for non-transformed)
  //
  // Parameters:
  //   prologue_cache_key: Pre-computed cache key to use for caching this prologue
  static PrologueResult MergeTransformations(const std::vector<TransformationTask>& tasks,
                                             const std::string& op_name,
                                             const std::vector<void*>& all_input_ptrs,
                                             const std::string& prologue_cache_key);
};

}  // namespace lazy
}  // namespace c10_neuron
