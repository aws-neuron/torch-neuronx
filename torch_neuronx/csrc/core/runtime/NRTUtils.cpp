#include "NRTUtils.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron::nrt {

void TensorSetDeleter::operator()(TensorSetHandle* raw) const {
  if (raw) nrt_destroy_tensor_set(&raw);
}

TensorSetPtr CreateTensorSet() {
  nrt_tensor_set_t* set = nullptr;
  NRT_STATUS status = nrt_allocate_tensor_set(&set);
  if (status != NRT_SUCCESS || !set) {
    throw std::runtime_error("Failed to allocate NRT tensor set. Status: " +
                             std::to_string(status));
  }
  return TensorSetPtr{set};
}

NRT_STATUS AddTensorToSet(const TensorSetPtr& set, const std::string& name,
                          nrt_tensor_t* data_ptr) {
  return nrt_add_tensor_to_tensor_set(set.get(), name.c_str(), data_ptr);
}

void ModelDeleter::operator()(ModelHandle* raw) const {
  if (raw) nrt_unload(raw);
}

NRT_STATUS LoadModel(ModelPtr& model, const std::vector<uint8_t>& neff_bytes, int vnc_id,
                     int num_cores) {
  nrt_model_t* raw = nullptr;
  NRT_STATUS status = nrt_load(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores, &raw);
  if (status == NRT_SUCCESS) {
    model.reset(raw);
  }
  return status;
}

NRT_STATUS LoadModelCollectives(ModelPtr& model, const std::vector<uint8_t>& neff_bytes, int vnc_id,
                                int num_cores, int rank, int world_size) {
  nrt_model_t* raw = nullptr;
  NRT_STATUS status = nrt_load_collectives(neff_bytes.data(), neff_bytes.size(), vnc_id, num_cores,
                                           rank, world_size, &raw);
  if (status == NRT_SUCCESS) {
    model.reset(raw);
  }
  return status;
}

NRT_STATUS DispatchExecution(ModelHandle* model, TensorSetHandle* input_set,
                             TensorSetHandle* output_set, int queue, NRT_STATUS* ret,
                             SequenceId* req_sequence) {
  if (!model) {
    throw std::runtime_error("Model is not properly loaded");
  }
  if (!input_set || !output_set) {
    throw std::invalid_argument("Tensor sets cannot be null");
  }

  TORCH_CHECK(ret != nullptr && req_sequence != nullptr,
              "Schedule requires both ret and req_sequence");
  return nrta_execute_schedule(model, input_set, output_set, queue, ret, req_sequence);
}

std::vector<std::pair<std::string, size_t>> GetMemoryUsageBreakdown(uint32_t vnc_id) {
  std::vector<std::pair<std::string, size_t>> breakdown;

  nrt_memory_usage_breakdown_t* result = nullptr;
  NRT_STATUS status = nrt_get_memory_usage_breakdown(vnc_id, &result);
  if (status == NRT_SUCCESS && result != nullptr) {
    breakdown.reserve(result->count);
    for (size_t i = 0; i < result->count; i++) {
      breakdown.emplace_back(result->entries[i].category, result->entries[i].bytes);
    }
  }
  // Free unconditionally: the runtime allocates the result on success even when
  // count == 0, and free(nullptr) is a safe no-op otherwise.
  nrt_memory_usage_breakdown_free(result);
  return breakdown;
}

}  // namespace at::neuron::nrt
