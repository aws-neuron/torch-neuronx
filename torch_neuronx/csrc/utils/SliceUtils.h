#pragma once

#include <sys/mman.h>
#include <torch/torch.h>
#include <unistd.h>

#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace at::neuron::utils {

/**
 * Process tensors with storage offsets by enqueuing slice operations.
 * Updates tensor_data_ptrs in fallback_ctx for tensors that were sliced.
 *
 * @param tensors Input tensors to process
 * @param fallback_ctx CPU fallback context to update with sliced pointers
 * @param stream Stream to enqueue slice operations on
 * @param device_id Device ID for the operations
 * @return Vector of effective data pointers (slice VAs or original pointers)
 */
std::vector<void*> ProcessTensorsWithSlicing(const std::vector<torch::Tensor>& tensors,
                                             at::neuron::CPUFallbackContext& fallback_ctx,
                                             const NeuronStream& stream, int device_id);

/**
 * Helper to check if a tensor needs slicing
 */
bool TensorNeedsSlicing(const torch::Tensor& tensor);

}  // namespace at::neuron::utils
