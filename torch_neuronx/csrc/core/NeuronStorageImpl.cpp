#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"

#include <c10/util/Exception.h>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace c10_neuron {

NeuronStorageImpl::NeuronStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes,
                                     at::DataPtr data_ptr, at::Allocator* allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, std::move(data_ptr), allocator, resizable) {
  // The neuron tensor is stored in the DataPtr context
  // We'll retrieve it when needed through the public API
}

// Constructor without data_ptr - let allocator handle allocation
NeuronStorageImpl::NeuronStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes,
                                     at::Allocator* allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, allocator, resizable) {
  // Allocator handles memory allocation internally
}

void NeuronStorageImpl::release_resources() {
  if (has_pending_operation()) {
    try {
      auto& future = pending_operation_future_.value();
      if (future.valid()) {
        future.get();
      }
    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("Failure releasing Neuron tensor: ", std::string(e.what()));
    }
    pending_operation_future_.reset();
  }
  // The DataPtr deleter will handle freeing the neuron tensor
  // We just need to call the base class implementation
  StorageImpl::release_resources();
}

}  // namespace c10_neuron
