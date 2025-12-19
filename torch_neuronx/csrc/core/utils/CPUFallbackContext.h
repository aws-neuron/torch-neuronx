#pragma once

#include <torch/torch.h>

#include <vector>

#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace at::neuron {

/**
 * Encapsulates all metadata needed for CPU fallback execution.
 * Original inputs are stored in base.py and passed here.
 * Metadata and data pointers are extracted in C++ from original_inputs.
 */
struct CPUFallbackContext {
  // All original positional inputs (tensors + scalars, lists preserved)
  // Populated from Python in base.py before any implementation modifies inputs
  std::vector<c10::IValue> original_inputs;

  // Original keyword arguments (e.g., alpha=, out=, dim=)
  // Populated from Python in base.py
  c10::impl::GenericDict original_kwargs;

  // Tensor metadata (shape, dtype, etc.) extracted from original_inputs
  // Populated in ConvertToCPUFallbackContext
  std::vector<TensorContext> tensor_metadata;

  // Data pointers for tensor inputs, extracted from original_inputs
  // Updated by ProcessTensorsWithSlicing for sliced tensors
  std::vector<void*> tensor_data_ptrs;

  // Default constructor creates empty context
  CPUFallbackContext()
      : original_kwargs(c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get())) {}

  // Constructor takes only original_inputs
  // Metadata and pointers are extracted later in ConvertToCPUFallbackContext
  explicit CPUFallbackContext(std::vector<c10::IValue> inputs)
      : original_inputs(std::move(inputs)),
        original_kwargs(c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get())) {}

  // Check if context is empty
  bool IsEmpty() const { return original_inputs.empty() && original_kwargs.empty(); }
};

}  // namespace at::neuron
