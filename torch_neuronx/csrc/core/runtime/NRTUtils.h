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
namespace NeuronCachingAllocator {
nrt_tensor_t* findTensor(void* ptr);
}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron

namespace at::neuron::nrt {

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

  NRT_STATUS AddTensor(std::string_view name, nrt_tensor_t* data_ptr);

  bool IsValid() const noexcept { return tensor_set_ != nullptr; }

  explicit operator bool() const noexcept { return IsValid(); }

 private:
  nrt_tensor_set_t* tensor_set_;

  void validate() const;
  void reset() noexcept;
};

// Handles NEFF loading and automatic cleanup with comprehensive
// error handling and resource tracking.
class Model {
 public:
  Model() noexcept;
  ~Model() noexcept;

  // Load NEFF data into the model
  NRT_STATUS Load(const std::vector<uint8_t>& neff_bytes, int device_id = 0, int num_cores = 1);

  // Load NEFF data with collective operations
  NRT_STATUS LoadCollectives(const std::vector<uint8_t>& neff_bytes, int device_id = 0,
                             int num_cores = 1);

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&& other) noexcept;
  Model& operator=(Model&& other) noexcept;

  // Access the underlying model
  nrt_model_t* get() const noexcept { return model_; }

  // Execute the model
  NRT_STATUS Execute(nrt_tensor_set_t* input_set, nrt_tensor_set_t* output_set);

  bool IsValid() const noexcept { return model_ != nullptr; }
  explicit operator bool() const noexcept { return IsValid(); }

  // Check if model is loaded (alias for IsValid for clarity)
  bool IsLoaded() const noexcept { return IsValid(); }

 private:
  nrt_model_t* model_;

  void validate() const;
  void reset() noexcept;
};

// Standalone NRT tensor allocation/deallocation functions
// These provide direct access to NRT tensor operations

/**
 * Allocate an NRT tensor
 * @param placement Tensor placement (device or host)
 * @param vnc_id Virtual NeuronCore ID
 * @param size_bytes Size in bytes
 * @param name Tensor name for debugging
 * @param tensor Output pointer to allocated tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateTensor(nrt_tensor_placement_t placement, int vnc_id, size_t size_bytes,
                                 const char* name, nrt_tensor_t** tensor) {
  return nrt_tensor_allocate(placement, vnc_id, size_bytes, name, tensor);
}

/**
 * Free an NRT tensor
 * @param tensor Pointer to tensor pointer (will be set to nullptr)
 */
inline void FreeTensor(nrt_tensor_t** tensor) { nrt_tensor_free(tensor); }

/**
 * Copy data between NRT tensors
 * @param src Source tensor
 * @param src_offset Offset in source tensor
 * @param dst Destination tensor
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to copy
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS CopyTensor(nrt_tensor_t* src, size_t src_offset, nrt_tensor_t* dst,
                             size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_copy(src, src_offset, dst, dst_offset, size_bytes);
}

/**
 * Write data from CPU memory to NRT tensor
 * @param dst Destination NRT tensor
 * @param src Source CPU memory pointer
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to write
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS WriteTensor(nrt_tensor_t* dst, void* src, size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_write(dst, src, dst_offset, size_bytes);
}

/**
 * Read data from NRT tensor to CPU memory
 * @param src Source NRT tensor
 * @param dst Destination CPU memory pointer
 * @param src_offset Offset in source tensor
 * @param size_bytes Number of bytes to read
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ReadTensor(nrt_tensor_t* src, void* dst, size_t src_offset, size_t size_bytes) {
  return nrt_tensor_read(src, dst, src_offset, size_bytes);
}

/**
 * Get the size of an NRT tensor
 * @param tensor Tensor to query
 * @return Size in bytes
 */
inline size_t GetTensorSize(nrt_tensor_t* tensor) { return nrt_tensor_get_size(tensor); }

/**
 * Allocate a slice of an existing NRT tensor
 * @param base_tensor Base tensor to slice from
 * @param offset Offset in base tensor
 * @param size Size of slice
 * @param name Name for the slice tensor
 * @param slice_tensor Output pointer to slice tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateSliceTensor(nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                      const char* name, nrt_tensor_t** slice_tensor) {
  return nrt_tensor_allocate_slice(base_tensor, offset, size, name, slice_tensor);
}

}  // namespace at::neuron::nrt
