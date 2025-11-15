#include "NRTUtils.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "torch_neuronx/csrc/core/NeuronDevice.h"

namespace at::neuron::nrt {

// Helper for consistent NRT error formatting
namespace {
template <typename... Args>
std::string format_nrt_error(const char* operation, NRT_STATUS status, Args&&... args) {
  std::ostringstream oss;
  oss << "NRT " << operation << " failed (status=" << status << ")";
  ((oss << ", " << std::forward<Args>(args)), ...);
  return oss.str();
}

std::string format_nrt_error(const char* operation, NRT_STATUS status) {
  std::ostringstream oss;
  oss << "NRT " << operation << " failed (status=" << status << ")";
  return oss.str();
}
}  // namespace

// TensorSet implementation
TensorSet::TensorSet() : tensor_set_(nullptr) {
  NRT_STATUS status = nrt_allocate_tensor_set(&tensor_set_);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(format_nrt_error("tensor_set_allocate", status));
  }
}

TensorSet::~TensorSet() noexcept {
  if (tensor_set_) {
    // Clean up any slice tensors we created
    cleanup_slice_tensors();
    nrt_destroy_tensor_set(&tensor_set_);
  }
}

TensorSet::TensorSet(TensorSet&& other) noexcept
    : tensor_set_(std::exchange(other.tensor_set_, nullptr)),
      slice_tensors_(std::move(other.slice_tensors_)) {}

TensorSet& TensorSet::operator=(TensorSet&& other) noexcept {
  if (this != &other) {
    reset();
    tensor_set_ = std::exchange(other.tensor_set_, nullptr);
    slice_tensors_ = std::move(other.slice_tensors_);
  }
  return *this;
}

void TensorSet::AddTensor(std::string_view name, const at::Tensor& tensor) {
  validate();

  // Check tensor device
  if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
    throw std::runtime_error("Tensor must be on neuron device, but got: " + tensor.device().str());
  }

  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Cannot add non-contiguous tensor to NRT tensor set");
  }

  // Get data pointer - adjust for storage offset to get base storage pointer
  void* data_ptr = tensor.data_ptr();
  size_t storage_offset_bytes = tensor.storage_offset() * tensor.element_size();
  void* base_ptr = static_cast<char*>(data_ptr) - storage_offset_bytes;

  // Find the corresponding nrt_tensor_t using the allocator with base pointer
  nrt_tensor_t* base_tensor = c10_neuron::NeuronCachingAllocator::findTensor(base_ptr);
  if (!base_tensor) {
    throw std::runtime_error(
        "Could not find NRT tensor for PyTorch tensor. Ensure tensor is on neuron device.");
  }

  nrt_tensor_t* nrt_tensor = base_tensor;
  std::unique_ptr<SliceTensorGuard> slice_guard;

  // If there's a storage offset, create a slice tensor
  if (storage_offset_bytes > 0) {
    // Create a slice tensor that points to the correct offset within the base tensor
    size_t view_size_bytes = tensor.numel() * tensor.element_size();
    size_t base_size_bytes = nrt_tensor_get_size(base_tensor);

    if (storage_offset_bytes > base_size_bytes ||
        base_size_bytes - storage_offset_bytes < view_size_bytes) {
      throw std::runtime_error("Tensor view exceeds base storage bounds for NRT slice allocation");
    }

    NRT_STATUS offset_status = nrt_tensor_allocate_slice(base_tensor, storage_offset_bytes,
                                                         view_size_bytes, name.data(), &nrt_tensor);
    if (offset_status != NRT_SUCCESS) {
      throw std::runtime_error(format_nrt_error(
          "tensor_allocate_slice", offset_status, "tensor_name=" + std::string(name),
          "offset_bytes=" + std::to_string(storage_offset_bytes),
          "view_size_bytes=" + std::to_string(view_size_bytes)));
    }

    // Use RAII guard for automatic cleanup on exception
    slice_guard = std::make_unique<SliceTensorGuard>(nrt_tensor);
  }

  // Add to tensor set
  NRT_STATUS status = nrt_add_tensor_to_tensor_set(tensor_set_, name.data(), nrt_tensor);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(
        format_nrt_error("add_tensor_to_set", status, "tensor_name=" + std::string(name)));
  }

  // Successfully added - track slice tensor for cleanup and release guard
  if (slice_guard) {
    std::lock_guard<std::mutex> lock(slice_tensors_mutex_);
    slice_tensors_.push_back(nrt_tensor);
    slice_guard->release();  // Successfully added, don't auto-cleanup
  }
}

size_t TensorSet::SliceTensorCount() const noexcept {
  std::lock_guard<std::mutex> lock(slice_tensors_mutex_);
  return slice_tensors_.size();
}

void TensorSet::validate() const {
  if (!IsValid()) {
    throw std::runtime_error("TensorSet is not properly initialized");
  }
}

void TensorSet::cleanup_slice_tensors() noexcept {
  std::lock_guard<std::mutex> lock(slice_tensors_mutex_);
  for (nrt_tensor_t* slice_tensor : slice_tensors_) {
    if (slice_tensor) {
      nrt_tensor_free(&slice_tensor);
    }
  }
  slice_tensors_.clear();
}

void TensorSet::reset() noexcept {
  if (tensor_set_) {
    cleanup_slice_tensors();
    nrt_destroy_tensor_set(&tensor_set_);
    tensor_set_ = nullptr;
  }
}

// Model implementation
Model::Model() noexcept : model_(nullptr) {}

Model::~Model() noexcept {
  if (model_) {
    nrt_unload(model_);
  }
}

Model::Model(Model&& other) noexcept : model_(std::exchange(other.model_, nullptr)) {}

Model& Model::operator=(Model&& other) noexcept {
  if (this != &other) {
    reset();
    model_ = std::exchange(other.model_, nullptr);
  }
  return *this;
}

void Model::Load(const std::vector<uint8_t>& neff_bytes, int device_id, int num_cores) {
  // Unload existing model if any
  reset();

  int vnc_id = c10_neuron::get_vnc_id(device_id);
  NRT_STATUS status = nrt_load(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores, &model_);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(format_nrt_error("load", status,
                                              "device_id=" + std::to_string(device_id),
                                              "num_cores=" + std::to_string(num_cores),
                                              "neff_size=" + std::to_string(neff_bytes.size())));
  }
}

void Model::LoadCollectives(const std::vector<uint8_t>& neff_bytes, int device_id, int num_cores) {
  // Unload existing model if any
  reset();

  int rank = c10_neuron::get_rank();
  int world_size = c10_neuron::get_world_size();
  int vnc_id = c10_neuron::get_vnc_id(device_id);

  NRT_STATUS status = nrt_load_collectives(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores,
                                           rank, world_size, &model_);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(
        format_nrt_error("load_collectives", status, "device_id=" + std::to_string(device_id),
                         "rank=" + std::to_string(rank), "world_size=" + std::to_string(world_size),
                         "neff_size=" + std::to_string(neff_bytes.size())));
  }
}

void Model::Execute(nrt_tensor_set_t* input_set, nrt_tensor_set_t* output_set) {
  validate();
  if (!input_set || !output_set) {
    throw std::invalid_argument("Tensor sets cannot be null");
  }

  NRT_STATUS status = nrt_execute(model_, input_set, output_set);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(format_nrt_error("execute", status));
  }
}

void Model::validate() const {
  if (!IsValid()) {
    throw std::runtime_error("Model is not properly loaded");
  }
}

void Model::reset() noexcept {
  if (model_) {
    nrt_unload(model_);
    model_ = nullptr;
  }
}

}  // namespace at::neuron::nrt
