#include "NRTHandler.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {

#define NRT_CHECK_STATUS(call, error_msg)                                \
  do {                                                                   \
    NRT_STATUS status = (call);                                          \
    if (status != NRT_SUCCESS) {                                         \
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status); \
    }                                                                    \
  } while (0)

void NRTHandler::DispatchModelExecution(const std::shared_ptr<nrt::ModelHandle>& model,
                                        const std::vector<nrt_tensor_t*>& src_data_ptrs,
                                        const std::vector<nrt_tensor_t*>& dst_data_ptrs,
                                        const ExecutionConfig& config, NRT_STATUS* nrt_status,
                                        nrt::SequenceId* sequence_id, c10::StreamId stream_id,
                                        const std::vector<std::string>& input_names,
                                        const std::vector<std::string>& output_names) {
  TORCH_NEURONX_DEBUG("NRTHandler dispatching model execution", "device_id=", config.device_id,
                      "num_cores=", config.num_cores);
  // Prepare tensor sets for NRT execution
  auto input_set = nrt::CreateTensorSet();
  auto output_set = nrt::CreateTensorSet();

  PrepareInputTensorSet(src_data_ptrs, input_set, input_names);
  PrepareOutputTensorSet(dst_data_ptrs, output_set, output_names);

  int queue = nrt::GetHardwareQueue(NRTA_XU_COMPUTE, stream_id);
  NRT_CHECK_STATUS(nrt::DispatchExecution(model.get(), input_set.get(), output_set.get(), queue,
                                          nrt_status, sequence_id),
                   "NRT model scheduling failed");
}

nrt_tensor_t* NRTHandler::AllocateSliceTensor(nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                              const char* name) {
  TORCH_NEURONX_DEBUG("NRTHandler allocating tensor slice");
  nrt_tensor_t* slice = nullptr;
  NRT_STATUS status = nrt::AllocateSliceTensor(base_tensor, offset, size, name, &slice);
  EXECUTION_RUNTIME_CHECK_STATUS(status == NRT_SUCCESS, status, "NRT slice allocation failed",
                                 " name=", name, " offset=", offset, " size=", size);
  EXECUTION_RUNTIME_CHECK(slice != nullptr,
                          "NRT slice allocation returned null slice despite NRT_SUCCESS",
                          " name=", name, " offset=", offset, " size=", size);
  TORCH_NEURONX_DEBUG("NRTHandler allocated tensor slice", "name=", name);
  return slice;
}

nrt_tensor_t* NRTHandler::AllocateTensor(nrt_tensor_placement_t placement, int vnc_id, size_t size,
                                         const char* name) {
  TORCH_NEURONX_DEBUG("NRTHandler allocating tensor", "size=", size, "vnc_id=", vnc_id);
  nrt_tensor_t* tensor = nullptr;
  NRT_STATUS status = nrt::AllocateTensor(placement, vnc_id, size, name, &tensor);
  EXECUTION_RUNTIME_CHECK_STATUS(status == NRT_SUCCESS, status, "NRT tensor allocation failed",
                                 " size=", size, " vnc_id=", vnc_id);
  EXECUTION_RUNTIME_CHECK(tensor != nullptr,
                          "NRT tensor allocation returned null tensor despite NRT_SUCCESS",
                          " size=", size, " vnc_id=", vnc_id);
  return tensor;
}

void NRTHandler::PrepareInputTensorSet(const std::vector<nrt_tensor_t*>& inputs,
                                       nrt::TensorSetPtr& input_set,
                                       const std::vector<std::string>& names) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing input tensor set");
  const bool use_custom_names = !names.empty();
  TORCH_CHECK(!use_custom_names || names.size() == inputs.size(),
              "NRTHandler::PrepareInputTensorSet: input_names size (", names.size(),
              ") does not match inputs size (", inputs.size(), ")");
  for (size_t i = 0; i < inputs.size(); ++i) {
    TORCH_NEURONX_DEBUG("Processing input tensor", "index=", i);
    nrt_tensor_t* data_ptr = inputs[i];
    std::string name = use_custom_names ? names[i] : ("input" + std::to_string(i));
    NRT_CHECK_STATUS(nrt::AddTensorToSet(input_set, name, data_ptr),
                     "Failed to add input tensor to tensor set");
    TORCH_NEURONX_DEBUG("Added input tensor to tensor set", "name=", name);
  }
}

void NRTHandler::PrepareOutputTensorSet(const std::vector<nrt_tensor_t*>& outputs,
                                        nrt::TensorSetPtr& output_set,
                                        const std::vector<std::string>& names) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing output tensor set");
  const bool use_custom_names = !names.empty();
  TORCH_CHECK(!use_custom_names || names.size() == outputs.size(),
              "NRTHandler::PrepareOutputTensorSet: output_names size (", names.size(),
              ") does not match outputs size (", outputs.size(), ")");
  for (size_t i = 0; i < outputs.size(); ++i) {
    TORCH_NEURONX_DEBUG("Processing output tensor", "index=", i);
    nrt_tensor_t* data_ptr = outputs[i];
    std::string name = use_custom_names ? names[i] : ("output" + std::to_string(i));
    NRT_CHECK_STATUS(nrt::AddTensorToSet(output_set, name, data_ptr),
                     "Failed to add output tensor to tensor set");
    TORCH_NEURONX_DEBUG("Added output tensor to tensor set", "name=", name);
  }
}

std::shared_ptr<nrt::ModelHandle> NRTHandler::GetOrLoadModel(
    const CompilableKernelExecution& compilable_kernel, const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler getting model from cache", "device_id=", config.device_id,
                      "num_cores=", config.num_cores);

  auto& cache = NeuronResourceManager::Instance().GetModelHandleCache();
  const auto& neff_bytes = compilable_kernel.GetCachedNeff();
  std::string cache_key = cache.MakeCacheKey(compilable_kernel.GetCompileKey(), config.device_id);

  // Try cache first
  auto model = cache.GetModel(cache_key);
  if (model) return model;

  // Cache miss — resolve device params and load
  int vnc_id = c10_neuron::get_vnc_id(config.device_id);
  bool has_collectives = compilable_kernel.HasCollectives();
  nrt::ModelPtr model_ptr;

  if (has_collectives) {
    int rank = c10_neuron::get_rank();
    int world_size = c10_neuron::get_world_size();
    NRT_CHECK_STATUS(nrt::LoadModelCollectives(model_ptr, neff_bytes, vnc_id, config.num_cores,
                                               rank, world_size),
                     "Failed to load model with collectives");
  } else {
    NRT_CHECK_STATUS(nrt::LoadModel(model_ptr, neff_bytes, vnc_id, config.num_cores),
                     "Failed to load model");
  }

  // Store in cache
  return cache.StoreModel(cache_key, std::move(model_ptr), config.device_id, has_collectives);
}

}  // namespace at::neuron
