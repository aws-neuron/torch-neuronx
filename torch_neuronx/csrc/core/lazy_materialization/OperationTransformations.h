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
// PrologueResult / EpilogueResult - Results of prologue/epilogue generation
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

  // Input deduplication: maps original input index -> deduplicated input index
  // When multiple inputs share the same storage, they map to the same deduped index.
  // Empty if no deduplication occurred.
  std::vector<size_t> input_dedup_map;
  size_t deduped_input_count = 0;
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
      const std::string& prologue_cache_key, const std::vector<size_t>& input_dedup_map,
      size_t deduped_input_count);

  // Compute prologue cache key efficiently without creating transformation tasks
  //
  // This should be called once by the caller (LazyTransformationManager) to generate
  // the cache key, which is then passed to GeneratePrologue and used for caching.
  //
  // The returned key incorporates (i) the per-input transformation shapes /
  // dtypes / parameters, (ii) the op name, and (iii) an input
  // deduplication signature when shared-storage inputs collapse to
  // fewer slots. (iii) is required because MergeTransformations builds
  // its MLIR with ``input_slots(deduped_input_count)`` — the function
  // arity depends on dedup state, so two calls with identical
  // transforms + dtypes + op_name but different dedup state must not
  // share a cached prologue.
  //
  // Parameters:
  //   input_ptrs: Input data pointers
  //   input_contexts: Tensor contexts (for shape/dtype of NONE transformations,
  //                   and storage offset / element size for dedup)
  //   op_name: Operation name
  //   kernel_transforms: Per-input transformations from kernel metadata
  //
  // Returns:
  //   Cache key string, or empty string if no transformations are needed
  static std::string ComputePrologueCacheKey(
      const std::vector<void*>& input_ptrs,
      const std::vector<at::neuron::TensorContext>& input_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms = {});

  // Allow OperationEpilogue to reuse CollectTransformationTasks
  friend class OperationEpilogue;

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
  //   input_dedup_map: Maps original input index -> deduped slot index
  //   deduped_input_count: Number of unique input slots after dedup
  static PrologueResult MergeTransformations(const std::vector<TransformationTask>& tasks,
                                             const std::string& op_name, size_t num_inputs,
                                             const std::string& prologue_cache_key,
                                             const std::vector<size_t>& input_dedup_map,
                                             size_t deduped_input_count);
};

// ============================================================================
// EpilogueResult - Result of epilogue generation
// ============================================================================

// Contains the merged epilogue StableHLO and tensor mappings.
// Reuses the same MLIR generation pipeline as PrologueResult — the only
// difference is the transformations themselves (inverse permutations, etc.)
struct EpilogueResult {
  bool success = true;
  std::string error_message;
  bool has_transformations = false;

  std::string epilogue_mlir_str;
  std::string epilogue_cache_key;

  std::vector<bool> output_was_transformed;
  std::vector<TransformationTask> tasks;

  // True if any output chain contains SCATTER_IOTA.
  // When true, the epilogue MLIR has an extra input (flat storage)
  // that must be wired to the kernel's output buffer in ProcessEpilogueWithMlirMerge.
  bool has_scatter_iota = false;

  // Convenience: true if any epilogue transform has an extra base input
  bool has_extra_base_input() const { return has_scatter_iota; }
};

// ============================================================================
// OperationEpilogue - Generates merged inverse transformation epilogues
//
// Reuses OperationPrologue's internal machinery (CollectTransformationTasks,
// MergeTransformations, transformation_utils::GenerateMlir) since the MLIR
// generation pipeline is identical for both directions. The only difference
// is that the transformations are inverse (e.g., inverse transpose permutation).
// ============================================================================

class OperationEpilogue {
 public:
  // Generate a merged epilogue StableHLO from output transformations.
  // Reuses the same task collection and MLIR merge pipeline as prologue.
  static EpilogueResult GenerateEpilogue(
      const std::vector<void*>& output_ptrs,
      const std::vector<at::neuron::TensorContext>& output_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms,
      const std::string& epilogue_cache_key);

  // Compute epilogue cache key (same format as prologue, different prefix)
  static std::string ComputeEpilogueCacheKey(
      const std::vector<void*>& output_ptrs,
      const std::vector<at::neuron::TensorContext>& output_contexts, const std::string& op_name,
      const std::vector<std::vector<TensorTransformation>>& kernel_transforms = {});
};

}  // namespace lazy
}  // namespace c10_neuron
