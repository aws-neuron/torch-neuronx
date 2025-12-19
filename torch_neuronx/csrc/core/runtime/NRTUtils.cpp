#include "NRTUtils.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron::nrt {

// TensorSet implementation
TensorSet::TensorSet() : tensor_set_(nullptr) {
  NRT_STATUS status = nrt_allocate_tensor_set(&tensor_set_);
  if (status != NRT_SUCCESS || !tensor_set_) {
    throw std::runtime_error("Failed to allocate NRT tensor set. Status: " +
                             std::to_string(status));
  }
}

TensorSet::~TensorSet() noexcept {
  if (tensor_set_) {
    nrt_destroy_tensor_set(&tensor_set_);
  }
}

TensorSet::TensorSet(TensorSet&& other) noexcept
    : tensor_set_(std::exchange(other.tensor_set_, nullptr)) {}

TensorSet& TensorSet::operator=(TensorSet&& other) noexcept {
  if (this != &other) {
    reset();
    tensor_set_ = std::exchange(other.tensor_set_, nullptr);
  }
  return *this;
}

NRT_STATUS TensorSet::AddTensor(std::string_view name, nrt_tensor_t* data_ptr) {
  return nrt_add_tensor_to_tensor_set(tensor_set_, name.data(), data_ptr);
}

void TensorSet::validate() const {
  if (!IsValid()) {
    throw std::runtime_error("TensorSet is not properly initialized");
  }
}

void TensorSet::reset() noexcept {
  if (tensor_set_) {
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

NRT_STATUS Model::Load(const std::vector<uint8_t>& neff_bytes, int device_id, int num_cores) {
  // Unload existing model if any
  reset();

  int vnc_id = c10_neuron::get_vnc_id(device_id);
  return nrt_load(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores, &model_);
}

NRT_STATUS Model::LoadCollectives(const std::vector<uint8_t>& neff_bytes, int device_id,
                                  int num_cores) {
  // Unload existing model if any
  reset();

  int rank = c10_neuron::get_rank();
  int world_size = c10_neuron::get_world_size();
  int vnc_id = c10_neuron::get_vnc_id(device_id);

  return nrt_load_collectives(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores, rank,
                              world_size, &model_);
}

NRT_STATUS Model::Execute(nrt_tensor_set_t* input_set, nrt_tensor_set_t* output_set) {
  validate();
  if (!input_set || !output_set) {
    throw std::invalid_argument("Tensor sets cannot be null");
  }
  return nrt_execute(model_, input_set, output_set);
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
