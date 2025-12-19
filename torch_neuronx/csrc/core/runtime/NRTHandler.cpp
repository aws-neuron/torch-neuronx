#include "NRTHandler.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

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

void NRTHandler::PrepareAndExecuteModel(nrt::Model* model,
                                        const std::vector<nrt_tensor_t*>& src_data_ptrs,
                                        const std::vector<nrt_tensor_t*>& dst_data_ptrs,
                                        const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler executing model", "device_id=", config.device_id,
                      "num_cores=", config.num_cores);
  // Prepare tensor sets for NRT execution
  nrt::TensorSet input_set;
  nrt::TensorSet output_set;

  PrepareInputTensorSet(src_data_ptrs, input_set);
  PrepareOutputTensorSet(dst_data_ptrs, output_set);

  NRT_CHECK_STATUS(model->Execute(input_set.get(), output_set.get()), "NRT model execution failed");
}

void NRTHandler::PrepareInputTensorSet(const std::vector<nrt_tensor_t*>& inputs,
                                       nrt::TensorSet& input_set) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing input tensor set");
  for (size_t i = 0; i < inputs.size(); ++i) {
    TORCH_NEURONX_DEBUG("Processing input tensor", "index=", i);
    nrt_tensor_t* data_ptr = inputs[i];
    std::string name = "input" + std::to_string(i);
    NRT_CHECK_STATUS(input_set.AddTensor(name, data_ptr),
                     "Failed to add input tensor to tensor set");
    TORCH_NEURONX_DEBUG("Added input tensor to tensor set", "name=", name);
  }
}

void NRTHandler::PrepareOutputTensorSet(const std::vector<nrt_tensor_t*>& outputs,
                                        nrt::TensorSet& output_set) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing output tensor set");
  for (size_t i = 0; i < outputs.size(); ++i) {
    TORCH_NEURONX_DEBUG("Processing output tensor", "index=", i);
    nrt_tensor_t* data_ptr = outputs[i];
    std::string name = "output" + std::to_string(i);
    NRT_CHECK_STATUS(output_set.AddTensor(name, data_ptr),
                     "Failed to add output tensor to tensor set");
    TORCH_NEURONX_DEBUG("Added output tensor to tensor set", "name=", name);
  }
}

nrt::Model* NRTHandler::GetOrLoadModel(const CompilableKernelExecution& compilable_kernel,
                                       const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler getting model from cache", "device_id=", config.device_id,
                      "num_cores=", config.num_cores);

  auto& model_handle_cache = NeuronResourceManager::Instance().GetModelHandleCache();

  auto cached_model =
      model_handle_cache.GetOrLoadModel(compilable_kernel, config.device_id, config.num_cores);

  if (!cached_model || !cached_model->IsValid()) {
    throw std::runtime_error("Failed to get valid model from cache");
  }
  return cached_model.get();
}

}  // namespace at::neuron
