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
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {

NRTHandler::ExecutionResult NRTHandler::ExecuteCompilableKernel(
    const CompilableKernelExecution& compilable_kernel, const std::string& op_name,
    const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler executing compilable kernel", "op=", op_name,
                      "device_id=", config.device_id, "num_cores=", config.num_cores);

  try {
    // Prepare tensor sets for NRT execution
    nrt::TensorSet input_set;
    nrt::TensorSet output_set;

    PrepareInputTensorSet(compilable_kernel, input_set);
    PrepareOutputTensorSet(compilable_kernel, output_set);

    // Get or load model from cache
    auto model = GetOrLoadModel(compilable_kernel, config);
    if (!model || !model->IsValid()) {
      return ExecutionResult::Error("Failed to load model from cache");
    }

    // Execute with proper resource management
    return ExecuteModel(op_name, model, input_set, output_set, config);

  } catch (const std::exception& e) {
    TORCH_NEURONX_ERROR("NRTHandler execution failed", "op=", op_name, "error=", e.what());
    return ExecutionResult::Error("NRT execution failed: " + std::string(e.what()));
  }
}

void NRTHandler::PrepareInputTensorSet(const CompilableKernelExecution& compilable_kernel,
                                       nrt::TensorSet& input_set) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing input tensor set");

  const auto& input_tensors = compilable_kernel.GetInputs();

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    const auto& tensor = input_tensors[i];
    TORCH_NEURONX_DEBUG("Processing input tensor", "index=", i, "device=", tensor.device(),
                        "sizes=", tensor.sizes());

    std::string name = "input" + std::to_string(i);
    input_set.AddTensor(name, tensor);

    TORCH_NEURONX_DEBUG("Added input tensor to tensor set", "name=", name);
  }
}

void NRTHandler::PrepareOutputTensorSet(const CompilableKernelExecution& compilable_kernel,
                                        nrt::TensorSet& output_set) {
  TORCH_NEURONX_DEBUG("NRTHandler preparing output tensor set");

  const auto& output_tensors = compilable_kernel.GetInternalOutputs();

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    const auto& tensor = output_tensors[i];
    TORCH_NEURONX_DEBUG("Processing output tensor", "index=", i, "device=", tensor.device(),
                        "sizes=", tensor.sizes());

    std::string name = "output" + std::to_string(i);
    output_set.AddTensor(name, tensor);

    TORCH_NEURONX_DEBUG("Added output tensor to tensor set", "name=", name);
  }
}

std::shared_ptr<nrt::Model> NRTHandler::GetOrLoadModel(
    const CompilableKernelExecution& compilable_kernel, const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler getting model from cache", "device_id=", config.device_id,
                      "num_cores=", config.num_cores);

  auto& model_handle_cache = NeuronResourceManager::Instance().GetModelHandleCache();

  auto cached_model =
      model_handle_cache.GetOrLoadModel(compilable_kernel, config.device_id, config.num_cores);

  if (cached_model && cached_model->IsValid()) {
    TORCH_NEURONX_DEBUG("Model obtained from cache",
                        "model_handle=", static_cast<void*>(cached_model->get()));
  } else {
    TORCH_NEURONX_ERROR("Failed to get valid model from cache");
  }

  return cached_model;
}

NRTHandler::ExecutionResult NRTHandler::ExecuteModel(const std::string& op_name,
                                                     std::shared_ptr<nrt::Model> model,
                                                     nrt::TensorSet& input_set,
                                                     nrt::TensorSet& output_set,
                                                     const ExecutionConfig& config) {
  TORCH_NEURONX_DEBUG("NRTHandler executing model with resource management", "op=", op_name);

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    model->Execute(input_set.get(), output_set.get());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    TORCH_NEURONX_DEBUG("NRT synchronous execution completed",
                        "execution_time_ms=", duration.count());

    return ExecutionResult::Success(duration);

  } catch (const std::exception& e) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    TORCH_NEURONX_ERROR("NRT synchronous execution failed", "error=", e.what(),
                        "execution_time_ms=", duration.count());

    return ExecutionResult::Error("Synchronous execution failed: " + std::string(e.what()));
  }
}

}  // namespace at::neuron
