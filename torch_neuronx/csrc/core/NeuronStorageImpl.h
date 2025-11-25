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

  // Record that this tensor was produced by an async operation.
  // Used for validation that the operation completes before tensor access.
  void record_pending_operation(std::shared_future<at::neuron::OperationContextResult> future) {
    pending_operation_future_ = std::move(future);
  }

  // Ensure that the underlying storage is acquired and valid.
  // If the request is not blocking, it merely checks if it is pending or successful.
  bool is_valid(bool non_blocking) {
    if (!has_pending_operation()) {
      return true;
    }

    auto& future = pending_operation_future_.value();
    if (non_blocking) {
      // Return valid is the operation is still pending in case of a non-blocking request.
      if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        return true;
      }
    } else {
      // Wait indefinitely for the operation to complete
      future.wait();
    }

    bool success = future.get().IsSuccess();
    pending_operation_future_.reset();
    return success;
  }

 private:
  uint64_t unique_id_{0};

  std::optional<std::shared_future<at::neuron::OperationContextResult>> pending_operation_future_;

  bool has_pending_operation() const {
    return pending_operation_future_.has_value() && pending_operation_future_.value().valid();
  }
};

}  // namespace c10_neuron
