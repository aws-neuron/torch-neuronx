#pragma once

#include <ATen/Tensor.h>

#include <optional>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"

namespace c10_neuron {
namespace lazy {

/**
 * @brief Unified transformation components
 *
 * This file consolidates all transformation logic (Creators, Handlers, Materializers)
 * into a single location per the Registry Architecture. This eliminates fragmentation
 * and ensures consistent support across all transformation types.
 */

// ============================================================================
// STRIDE DECOMPOSITION: Shared analysis for prologue and epilogue
// ============================================================================

/**
 * @brief Result of decomposing a non-contiguous tensor's strides into
 *        a contiguous base tensor + transformation parameters.
 *
 * This is the shared analysis used by TryCreateCompound (prologue, forward chain).
 * TryCreateEpilogue uses its own dedicated AnalyzeEpilogueLayout helper instead.
 * The stride decomposition determines:
 * - The contiguous base tensor (physical bounding box in storage)
 * - Per-dimension slice parameters (start indices, step factors)
 * - Physical-to-logical dimension mapping (for transpose)
 * - Broadcast dimension identification
 */
struct StrideDecompositionResult {
  at::Tensor base_tensor;  ///< Contiguous base tensor (physical bounding box)

  // Physical layout information (in stride-sorted order mapped to full dims)
  std::vector<int64_t> phys_base_shape;    ///< Base shape per physical dimension
  std::vector<int64_t> phys_starts;        ///< Start index per dim (from offset decomposition)
  std::vector<int64_t> phys_target_shape;  ///< Target (view) shape per dimension
  std::vector<int64_t> phys_steps;         ///< Step factor per dim (1=contiguous, >1=strided)

  // Dimension mapping
  std::vector<int64_t> phys_to_logical;  ///< Maps physical dim position → original logical axis
  std::vector<int64_t> logical_to_phys;  ///< Maps original logical axis → physical dim position
  std::vector<bool> is_broadcast;        ///< Per-dim flag: true if broadcast (stride=0 or size=1)

  // Metadata
  std::string element_type;  ///< Element type string (e.g., "f32", "bf16")
  int64_t total_dims;        ///< Total number of dimensions
};

/**
 * @brief Decompose a non-contiguous tensor's strides into a contiguous base tensor
 *        and transformation parameters.
 *
 * Extracts the stride analysis logic (GCD-based factorization, offset decomposition,
 * physical-to-logical mapping) into a reusable function. TryCreateCompound calls this
 * to get the analysis, then builds its forward transformation chain.
 *
 * @param tensor The non-contiguous tensor to analyze
 * @param op_name Name of the operation (for logging)
 * @param index Index of this tensor in the operation (for logging)
 * @param anchor_at_view_offset If true, start the base tensor at the view storage offset.
 *        If false, start the base tensor at storage offset 0 and slice to the view.
 * @return StrideDecompositionResult if decomposition succeeds, nullopt otherwise
 */
std::optional<StrideDecompositionResult> DecomposeStrides(const at::Tensor& tensor,
                                                          const std::string& op_name, size_t index,
                                                          bool anchor_at_view_offset = false);

// ============================================================================
// CREATORS: Detect transformation patterns from tensor properties
// ============================================================================

namespace Creators {

/**
 * @brief Try to create a TRANSPOSE transformation
 *
 * Detects transpose patterns by analyzing tensor strides and shape.
 *
 * @param input Input tensor to analyze
 * @param op_name Name of the operation for logging
 * @param input_index Index of this input in the operation
 * @return TransformationResult if pattern detected, nullopt otherwise
 */
std::optional<TransformationCreationResult> TryCreateTranspose(const at::Tensor& input,
                                                               const std::string& op_name,
                                                               size_t input_index);

/**
 * @brief Try to create a SLICE transformation
 *
 * Detects slice patterns by analyzing tensor storage offset and strides.
 *
 * @param input Input tensor to analyze
 * @param op_name Name of the operation for logging
 * @param input_index Index of this input in the operation
 * @return TransformationResult if pattern detected, nullopt otherwise
 */
std::optional<TransformationCreationResult> TryCreateSlice(const at::Tensor& input,
                                                           const std::string& op_name,
                                                           size_t input_index);

/**
 * @brief Try to create a BROADCAST transformation
 *
 * Detects broadcast patterns by analyzing tensor strides for zero-stride dimensions.
 *
 * @param input Input tensor to analyze
 * @param op_name Name of the operation for logging
 * @param input_index Index of this input in the operation
 * @return TransformationResult if pattern detected, nullopt otherwise
 */
std::optional<TransformationCreationResult> TryCreateBroadcast(const at::Tensor& input,
                                                               const std::string& op_name,
                                                               size_t input_index);

/**
 * @brief Try to create a COMPOUND transformation (general stride decomposition)
 *
 * Decomposes arbitrary non-contiguous stride patterns into a sequence of
 * slice and transpose transformations. Handles compound patterns like
 * transpose+slice (common in attention/RoPE) that individual TryCreateTranspose
 * and TryCreateSlice cannot detect.
 *
 * @param input Input tensor to analyze
 * @param op_name Name of the operation for logging
 * @param input_index Index of this input in the operation
 * @return TransformationCreationResult with multiple transformations if pattern detected
 */
std::optional<TransformationCreationResult> TryCreateCompound(const at::Tensor& input,
                                                              const std::string& op_name,
                                                              size_t input_index);

/**
 * @brief Try to create a COMPOUND_FOLDED transformation (folded decomposition)
 *
 * Handles narrow→view→permute patterns where DecomposeStrides fails due to
 * coprime stride ratios from view-split. Detects compact sub-dimension groups
 * and emits SLICE→RESHAPE→TRANSPOSE.
 *
 * @param input Input tensor to analyze
 * @param op_name Name of the operation for logging
 * @param input_index Index of this input in the operation
 * @return TransformationCreationResult if pattern detected, nullopt otherwise
 */
std::optional<TransformationCreationResult> TryCreateFoldedDecomposition(const at::Tensor& input,
                                                                         const std::string& op_name,
                                                                         size_t input_index);

/**
 * @brief Try to create EPILOGUE (inverse) transformations for a non-contiguous output
 *
 * Orchestrates a 2-tier strategy (implementation in Transformations.cpp):
 *
 *   Tier 1  AnalyzeEpilogueLayout → TryCreateDecomposedEpilogue
 *           The view DECOMPOSES into transpose ∘ slice of a contiguous base.
 *           Emits a clean INVERSE TRANSPOSE (+ SCATTER_IOTA for the slice part):
 *             - EPILOGUE_TRANSPOSE: pure transpose (INVERSE TRANSPOSE only)
 *             - EPILOGUE_SLICE:     slice without transpose (SCATTER_IOTA)
 *             - EPILOGUE_COMPOUND:  transpose + slice (INVERSE TRANSPOSE → SCATTER_IOTA)
 *
 *   Tier 2  TryCreateGenericScatterEpilogue
 *           The view does NOT decompose (diagonal / select / non-multiple gap).
 *           Falls back to a single RAW-STRIDE scatter (EPILOGUE_GENERIC_SCATTER).
 *
 *   none    Both tiers return nullopt → caller bounces through a contiguous temp
 *           (CopyBackToOriginal), which is always correct.
 *
 * @param output Output tensor to analyze (non-contiguous)
 * @param op_name Name of the operation for logging
 * @param output_index Index of this output in the operation
 * @return TransformationCreationResult with inverse transformations if supported
 */
std::optional<TransformationCreationResult> TryCreateEpilogue(const at::Tensor& output,
                                                              const std::string& op_name,
                                                              size_t output_index);

}  // namespace Creators

// ============================================================================
// HANDLERS: Group and process consecutive transformations
// ============================================================================

namespace Handlers {

/**
 * @brief Handler for TRANSPOSE transformations
 */
class TransposeHandler : public TransformationHandler {
 public:
  /**
   * @brief Find the end of a TRANSPOSE transformation group
   *
   * Consecutive TRANSPOSE transformations can be combined by composing permutations.
   *
   * @param transformations Full list of transformations
   * @param start_index Starting index
   * @param state Current transformation state
   * @return Index of last transformation in group (inclusive)
   */
  size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations, size_t start_index,
                      const TransformationState& state) const override;

  /**
   * @brief Process a group of TRANSPOSE transformations
   *
   * @param input_index Index of the input being processed
   * @param state Current transformation state
   * @param group Vector of TRANSPOSE transformations
   * @param op_name Operation name for logging
   * @return TransformationTask containing transformation metadata
   */
  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override;
};

/**
 * @brief Handler for SLICE transformations
 */
class SliceHandler : public TransformationHandler {
 public:
  /**
   * @brief Find the end of a SLICE transformation group
   *
   * Consecutive SLICE transformations can be combined into a single slice.
   *
   * @param transformations Full list of transformations
   * @param start_index Starting index
   * @param state Current transformation state
   * @return Index of last transformation in group (inclusive)
   */
  size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations, size_t start_index,
                      const TransformationState& state) const override;

  /**
   * @brief Process a group of SLICE transformations
   *
   * @param input_index Index of the input being processed
   * @param state Current transformation state
   * @param group Vector of SLICE transformations
   * @param op_name Operation name for logging
   * @return TransformationTask containing transformation metadata
   */
  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override;
};

/**
 * @brief Handler for BROADCAST transformations
 */
class BroadcastHandler : public TransformationHandler {
 public:
  /**
   * @brief Find the end of a BROADCAST transformation group
   *
   * BROADCAST transformations are processed individually for now.
   *
   * @param transformations Full list of transformations
   * @param start_index Starting index
   * @param state Current transformation state
   * @return Index of last transformation in group (inclusive)
   */
  size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations, size_t start_index,
                      const TransformationState& state) const override;

  /**
   * @brief Process a group of BROADCAST transformations
   *
   * @param input_index Index of the input being processed
   * @param state Current transformation state
   * @param group Vector of BROADCAST transformations
   * @param op_name Operation name for logging
   * @return TransformationTask containing transformation metadata
   */
  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override;
};

/**
 * @brief Handler for SCATTER_IOTA transformations (epilogue only)
 *
 * Scatter with iota-computed flat indices for slice patterns.
 * Uses flat storage as base instead of shaped tensor, avoiding base shape ambiguity.
 * Matches TorchNeuron's SCATTER_IOTA approach.
 */
class ScatterIotaHandler : public TransformationHandler {
 public:
  size_t FindGroupEnd(const std::vector<TensorTransformation>& transformations, size_t start_index,
                      const TransformationState& state) const override;

  TransformationTask ProcessGroup(size_t input_index, TransformationState& state,
                                  const std::vector<TensorTransformation>& group,
                                  const std::string& op_name) const override;
};

}  // namespace Handlers

// ============================================================================
// INTERNAL UTILITIES (exposed for unit testing)
// ============================================================================

/**
 * @brief A compact group of consecutive dimensions (in stride-sorted order)
 *        that share a contiguous stride progression.
 */
struct CompactGroup {
  std::vector<int64_t> sorted_positions;  ///< positions in the pre-sorted order
  int64_t inner_stride;                   ///< smallest stride in this group
  int64_t shape_product;                  ///< product of all shapes in this group
};

/**
 * @brief Discover compact groups from pre-sorted (stride-descending) shape and stride arrays.
 *
 * A group boundary occurs where act_stride[i] != act_stride[i+1] * act_shape[i+1].
 * Returns groups ordered from outermost (largest stride) to innermost (smallest stride).
 *
 * @param act_shape  Shapes of active dimensions, sorted by stride descending
 * @param act_stride Strides of active dimensions, sorted descending
 * @return Vector of compact groups from outermost to innermost
 */
std::vector<CompactGroup> DiscoverCompactGroups(const std::vector<int64_t>& act_shape,
                                                const std::vector<int64_t>& act_stride);

}  // namespace lazy
}  // namespace c10_neuron
