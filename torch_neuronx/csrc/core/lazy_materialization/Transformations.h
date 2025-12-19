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

}  // namespace Handlers

}  // namespace lazy
}  // namespace c10_neuron
