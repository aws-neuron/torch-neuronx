#pragma once

#include <torch/torch.h>

#include <vector>

#include "torch_neuronx/csrc/core/KernelExecution.h"

namespace at::neuron::testing {

// Helper function to create test tensors for unit tests.
// Creates a zero-initialized tensor with the specified properties.
//
// NOTE: For unit tests that don't require actual device operations:
// - Use default parameters to create CPU tensors
// - Tests can verify device-agnostic logic
//
// Parameters:
//   sizes: Shape of the tensor (e.g., {2, 3} for a 2x3 tensor)
//   dtype: Data type of the tensor (default: float32)
//   device: Device to create the tensor on (default: CPU)
//
// Returns:
//   A zero-initialized torch::Tensor with the specified properties
inline torch::Tensor create_test_tensor(const std::vector<int64_t>& sizes,
                                        torch::ScalarType dtype = torch::kFloat32,
                                        torch::Device device = torch::kCPU) {
  return torch::zeros(sizes, torch::TensorOptions().dtype(dtype).device(device));
}

// Helper function to create an empty vector of tensors.
// Useful for testing operations that don't require inputs/outputs.
//
// Returns:
//   An empty vector of torch::Tensor
inline std::vector<torch::Tensor> create_empty_tensor_vector() {
  return std::vector<torch::Tensor>();
}

// Helper function to extract data pointers from tensors.
// Converts a vector of tensors to a vector of void pointers.
//
// Parameters:
//   tensors: Vector of torch::Tensor objects
//
// Returns:
//   A vector of void* pointing to the tensor data
inline std::vector<void*> get_tensor_data_ptrs(const std::vector<torch::Tensor>& tensors) {
  std::vector<void*> ptrs;
  ptrs.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ptrs.push_back(tensor.data_ptr());
  }
  return ptrs;
}

// Helper to create fake TensorDataRef for tests (no registry needed)
inline std::vector<at::neuron::TensorDataRef> create_fake_tensor_refs(
    const std::vector<void*>& ptrs) {
  std::vector<at::neuron::TensorDataRef> refs;
  for (auto ptr : ptrs) {
    refs.emplace_back(nullptr, ptr);  // null tensor_ptr for tests
  }
  return refs;
}

}  // namespace at::neuron::testing
