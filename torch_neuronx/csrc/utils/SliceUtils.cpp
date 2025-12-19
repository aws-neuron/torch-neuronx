#include "SliceUtils.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron::utils {

std::vector<void*> ProcessTensorsWithSlicing(const std::vector<torch::Tensor>& tensors,
                                             at::neuron::CPUFallbackContext& fallback_ctx,
                                             const NeuronStream& stream, int device_id) {
  std::vector<void*> result_ptrs;
  result_ptrs.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Expected PrivateUse1 device type, got ", tensor.device().type());

    // Check if tensor is contiguous
    if (!tensor.is_contiguous()) {
      throw std::runtime_error("Cannot process non-contiguous tensors");
    }
    void* original_ptr = tensor.data_ptr();

    if (tensor.storage_offset() == 0) {
      // No slicing needed, use original pointer
      result_ptrs.push_back(original_ptr);
      continue;
    }

    // Tensor needs slicing - enqueue slice operation
    size_t offset_bytes = tensor.storage_offset() * tensor.element_size();
    void* base_ptr = GetBaseTensorPtr(original_ptr, offset_bytes);
    size_t slice_size_bytes = tensor.numel() * tensor.element_size();

    void* slice_ptr = c10_neuron::NeuronCachingAllocator::ReserveVirtualAddress();
    c10_neuron::NeuronCachingAllocator::addSliceEntry(base_ptr, slice_ptr, slice_size_bytes,
                                                      device_id, stream);
    auto slice_kernel = std::make_unique<SliceDirectKernelExecution>(
        "neuron::memory::slice", base_ptr, slice_ptr, offset_bytes, slice_size_bytes, device_id);
    auto context = std::make_unique<at::neuron::OperationContext>(std::move(slice_kernel));
    at::neuron::SubmitOperationContext(stream, std::move(context));

    TORCH_NEURONX_DEBUG("Enqueued slice operation", "base_ptr=", base_ptr, "slice_va=", slice_ptr,
                        "offset=", offset_bytes, "size=", slice_size_bytes);

    // Update fallback_ctx.tensor_data_ptrs for this sliced tensor
    for (size_t i = 0; i < fallback_ctx.tensor_data_ptrs.size(); ++i) {
      if (fallback_ctx.tensor_data_ptrs[i] == original_ptr) {
        fallback_ctx.tensor_data_ptrs[i] = slice_ptr;
        TORCH_NEURONX_DEBUG("Updated fallback_ctx.tensor_data_ptrs[", i, "] from ", original_ptr,
                            " to ", slice_ptr);
        break;
      }
    }

    result_ptrs.push_back(slice_ptr);
  }

  return result_ptrs;
}

}  // namespace at::neuron::utils
