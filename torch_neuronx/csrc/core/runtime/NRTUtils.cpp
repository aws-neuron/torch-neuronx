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

void ErrorTrackerDeleter::operator()(ErrorTrackerHandle* raw) const {
  if (raw) nrta_error_tracker_destroy(raw);
}

void TensorSetDeleter::operator()(TensorSetHandle* raw) const {
  if (raw) nrt_destroy_tensor_set(&raw);
}

ErrorTrackerPtr CreateErrorTracker(uint32_t lnc_idx) {
  nrta_error_tracker_t* tracker = nullptr;
  NRT_STATUS status = nrta_error_tracker_create(lnc_idx, &tracker);
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to create NRT error tracker for LNC " +
                             std::to_string(lnc_idx) + ". Status: " + std::to_string(status));
  }
  return ErrorTrackerPtr{tracker};
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

std::vector<AsyncError> GetAndClearErrors(const ErrorTrackerPtr& tracker) {
  std::vector<AsyncError> errors;
  if (!tracker) {
    return errors;
  }

  const nrta_error_t* error_list = nullptr;
  size_t error_count = 0;
  NRT_STATUS status = nrta_error_tracker_get_list(tracker.get(), &error_list, &error_count);
  if (status != NRT_SUCCESS) {
    TORCH_NEURONX_ERROR("Failed to retrieve errors from error tracker. Status: ", status);
    return errors;
  }

  if (error_count > 0 && error_list) {
    errors.reserve(error_count);
    for (size_t i = 0; i < error_count; ++i) {
      errors.push_back({error_list[i].seq_id, error_list[i].error_code});
    }
  }
  return errors;
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
                             TensorSetHandle* output_set, int queue,
                             ErrorTrackerHandle* err_tracker, SequenceId* req_sequence) {
  if (!model) {
    throw std::runtime_error("Model is not properly loaded");
  }
  if (!input_set || !output_set) {
    throw std::invalid_argument("Tensor sets cannot be null");
  }

  bool async_mode_enabled =
      NeuronResourceManager::Instance().GetOperationExecutionEngine().IsNRTAsyncModeEnabled();

  if (async_mode_enabled) {
    TORCH_CHECK(err_tracker != nullptr && req_sequence != nullptr,
                "Async mode requires both err_tracker and req_sequence");
    return nrta_execute_schedule(model, input_set, output_set, queue, err_tracker, req_sequence);
  }
  return nrt_execute(model, input_set, output_set);
}

}  // namespace at::neuron::nrt
