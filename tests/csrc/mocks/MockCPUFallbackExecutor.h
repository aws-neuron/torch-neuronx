#pragma once

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"

namespace at::neuron::testing {

/**
 * Mock CPUFallbackExecutor that avoids device transfer issues in test environment
 *
 * This mock overrides the methods
 * to avoid PrivateUse1 (Neuron) device operations that fail in test environments
 * where the Neuron backend is not fully initialized.
 */
class MockCPUFallbackExecutor : public CPUFallbackExecutor {
 public:
  MockCPUFallbackExecutor();
  virtual ~MockCPUFallbackExecutor() = default;

 protected:
  void TransferTensorsFromCpuToNeuron(const std::vector<torch::Tensor>& cpu_results,
                                      const std::vector<nrt_tensor_t*>& data_ptrs);

  std::vector<torch::Tensor> TransferTensorsToCpu(
      const std::vector<nrt_tensor_t*>& data_ptrs,
      const std::vector<TensorContext>& tensor_contexts);
};

}  // namespace at::neuron::testing
