#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <nrt/nrt.h>

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace c10_neuron {

struct NeuronStorageImpl : public c10::StorageImpl {
  // Constructor with data_ptr (for pre-allocated data)
  NeuronStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::DataPtr data_ptr,
                    at::Allocator* allocator, bool resizable);

  // Constructor without data_ptr (let allocator handle allocation)
  NeuronStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::Allocator* allocator,
                    bool resizable);

  ~NeuronStorageImpl() override = default;

  void release_resources() override;

  // Initially we don't support resizing
  // We'll add this functionality later when we have an allocator

  // Getters
  nrt_tensor_t* neuron_tensor() const {
    // Get tensor from DataPtr context
    void* ctx = data_ptr().get_context();
    return NeuronCachingAllocator::getTensorFromContext(ctx);
  }

 private:
  uint64_t unique_id_{0};
};

}  // namespace c10_neuron
