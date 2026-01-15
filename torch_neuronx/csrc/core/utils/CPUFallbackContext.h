#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <variant>
#include <vector>

#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace at::neuron {

/**
 * Marker for a single tensor input.
 * Actual tensor data is retrieved from cpu_inputs during reconstruction.
 */
struct TensorMarker {};

/**
 * Marker for a list of tensors.
 * Stores the count; actual tensor data is retrieved from cpu_inputs during reconstruction.
 */
struct TensorListMarker {
  size_t count = 0;

  TensorListMarker() = default;
  explicit TensorListMarker(size_t c) : count(c) {}
};

/**
 * Wrapper for scalar/non-tensor values.
 * Stores the actual c10::IValue which can be None, int, double, bool, string, lists, etc.
 */
struct ScalarValue {
  c10::IValue value;

  ScalarValue() = default;
  explicit ScalarValue(c10::IValue v) : value(std::move(v)) {}
};

/**
 * Metadata used to store the type of inputs.
 * Uses std::variant to represent one of:
 * - TensorMarker: single tensor (data from cpu_inputs)
 * - TensorListMarker: list of tensors (data from cpu_inputs)
 * - ScalarValue: any non-tensor value including None
 */
using InputMetadata = std::variant<TensorMarker, TensorListMarker, ScalarValue>;

// Helper functions for InputMetadata

inline bool IsTensor(const InputMetadata& meta) {
  return std::holds_alternative<TensorMarker>(meta);
}

inline bool IsTensorList(const InputMetadata& meta) {
  return std::holds_alternative<TensorListMarker>(meta);
}

inline bool IsScalar(const InputMetadata& meta) {
  return std::holds_alternative<ScalarValue>(meta);
}

inline size_t GetTensorCount(const InputMetadata& meta) {
  if (IsTensor(meta)) return 1;
  if (auto* tl = std::get_if<TensorListMarker>(&meta)) return tl->count;
  return 0;
}

inline size_t GetListSize(const InputMetadata& meta) {
  if (auto* tl = std::get_if<TensorListMarker>(&meta)) return tl->count;
  return 0;
}

inline const c10::IValue& GetScalarValue(const InputMetadata& meta) {
  static const c10::IValue empty;
  if (auto* sv = std::get_if<ScalarValue>(&meta)) return sv->value;
  return empty;
}

/**
 * Encapsulates all metadata needed for CPU fallback execution.
 *
 * Design:
 * - Tensor positions are tracked via InputMetadata (TensorMarker/TensorListMarker)
 * - Tensor data comes from tensor_metadata + tensor_data_ptrs
 * - Scalars are stored directly in InputMetadata as ScalarValue
 *
 * The input_metadata vector maintains the same order as original Python args,
 * allowing correct reconstruction of the argument stack during CPU fallback.
 */
struct CPUFallbackContext {
  // Metadata for each positional input argument (same order as original args)
  std::vector<InputMetadata> input_metadata;

  // Original keyword arguments - scalar values only
  // For tensor kwargs, the value is stored as None and the kwarg name is tracked
  // in tensor_kwarg_names
  c10::impl::GenericDict original_kwargs;

  // Names of kwargs that are tensors, in the order they appear in tensor_metadata
  // Used to map kwarg names to their corresponding cpu_inputs during reconstruction
  std::vector<std::string> tensor_kwarg_names;

  // Tensor metadata (shape, dtype, etc.) extracted from original tensor inputs
  // Order: positional tensors first (flattened, including TensorLists), then kwarg tensors
  std::vector<TensorContext> tensor_metadata;

  // Data pointers for tensor inputs, extracted from original tensor inputs
  // Same order as tensor_metadata
  std::vector<void*> tensor_data_ptrs;

  // Default constructor creates empty context
  CPUFallbackContext()
      : original_kwargs(c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get())) {}

  // Check if context is empty
  bool IsEmpty() const { return input_metadata.empty() && original_kwargs.empty(); }

  // Get the number of positional tensors (excludes kwarg tensors)
  size_t GetPositionalTensorCount() const {
    size_t count = 0;
    for (const auto& meta : input_metadata) {
      count += GetTensorCount(meta);
    }
    return count;
  }

  // Get total tensor count (positional + kwarg tensors)
  size_t GetTotalTensorCount() const { return tensor_metadata.size(); }

  // Check if a kwarg name corresponds to a tensor
  bool IsKwargTensor(const std::string& name) const {
    return std::find(tensor_kwarg_names.begin(), tensor_kwarg_names.end(), name) !=
           tensor_kwarg_names.end();
  }

  // Get the index into cpu_inputs for a tensor kwarg
  // Returns the index after all positional tensors
  size_t GetKwargTensorIndex(const std::string& name) const {
    size_t positional_count = GetPositionalTensorCount();
    for (size_t i = 0; i < tensor_kwarg_names.size(); ++i) {
      if (tensor_kwarg_names[i] == name) {
        return positional_count + i;
      }
    }
    TORCH_CHECK(false, "Tensor kwarg '", name, "' not found in context");
    return 0;  // Unreachable
  }
};

}  // namespace at::neuron
