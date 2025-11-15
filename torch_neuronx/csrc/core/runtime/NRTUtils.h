#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

// Include NRT headers
extern "C" {
#include <nrt/nrt.h>
}

#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Forward declarations
namespace c10_neuron {
class NeuronCachingAllocator {
 public:
  static nrt_tensor_t* findTensor(void* ptr);
};
}  // namespace c10_neuron

namespace at::neuron::nrt {

// RAII wrapper for slice tensors to ensure automatic cleanup
class SliceTensorGuard {
 public:
  explicit SliceTensorGuard(nrt_tensor_t* tensor) : tensor_(tensor), released_(false) {}

  ~SliceTensorGuard() {
    if (!released_ && tensor_) {
      nrt_tensor_free(&tensor_);
    }
  }

  // Non-copyable, movable
  SliceTensorGuard(const SliceTensorGuard&) = delete;
  SliceTensorGuard& operator=(const SliceTensorGuard&) = delete;

  SliceTensorGuard(SliceTensorGuard&& other) noexcept
      : tensor_(std::exchange(other.tensor_, nullptr)),
        released_(std::exchange(other.released_, true)) {}

  SliceTensorGuard& operator=(SliceTensorGuard&& other) noexcept {
    if (this != &other) {
      if (!released_ && tensor_) {
        nrt_tensor_free(&tensor_);
      }
      tensor_ = std::exchange(other.tensor_, nullptr);
      released_ = std::exchange(other.released_, true);
    }
    return *this;
  }

  void release() noexcept { released_ = true; }
  nrt_tensor_t* get() const noexcept { return tensor_; }

 private:
  nrt_tensor_t* tensor_;
  bool released_;
};

// Provides automatic resource management for NRT tensor sets with
// exception-safe operations and move semantics. Automatically allocates
// the tensor set on construction and destroys it on destruction.
class TensorSet {
 public:
  TensorSet();
  ~TensorSet() noexcept;

  // Non-copyable but movable for efficiency
  TensorSet(const TensorSet&) = delete;
  TensorSet& operator=(const TensorSet&) = delete;

  TensorSet(TensorSet&& other) noexcept;
  TensorSet& operator=(TensorSet&& other) noexcept;

  // Get the underlying NRT tensor set pointer
  nrt_tensor_set_t* get() const noexcept { return tensor_set_; }

  void AddTensor(std::string_view name, const at::Tensor& tensor);

  bool IsValid() const noexcept { return tensor_set_ != nullptr; }

  explicit operator bool() const noexcept { return IsValid(); }

  // Get number of tracked slice tensors (for debugging/monitoring)
  size_t SliceTensorCount() const noexcept;

 private:
  nrt_tensor_set_t* tensor_set_;
  std::vector<nrt_tensor_t*> slice_tensors_;
  mutable std::mutex slice_tensors_mutex_;

  void validate() const;
  void cleanup_slice_tensors() noexcept;
  void reset() noexcept;
};

// Handles NEFF loading and automatic cleanup with comprehensive
// error handling and resource tracking.
class Model {
 public:
  Model() noexcept;
  ~Model() noexcept;

  // Load NEFF data into the model
  void Load(const std::vector<uint8_t>& neff_bytes, int device_id = 0, int num_cores = 1);

  // Load NEFF data with collective operations
  void LoadCollectives(const std::vector<uint8_t>& neff_bytes, int device_id = 0,
                       int num_cores = 1);

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&& other) noexcept;
  Model& operator=(Model&& other) noexcept;

  // Access the underlying model
  nrt_model_t* get() const noexcept { return model_; }

  // Execute the model
  void Execute(nrt_tensor_set_t* input_set, nrt_tensor_set_t* output_set);

  bool IsValid() const noexcept { return model_ != nullptr; }
  explicit operator bool() const noexcept { return IsValid(); }

  // Check if model is loaded (alias for IsValid for clarity)
  bool IsLoaded() const noexcept { return IsValid(); }

 private:
  nrt_model_t* model_;

  void validate() const;
  void reset() noexcept;
};

}  // namespace at::neuron::nrt
