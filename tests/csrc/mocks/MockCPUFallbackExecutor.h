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
                                      OperationContext* op) override;

  std::vector<torch::Tensor> TransferTensorsToCpu(
      const std::vector<torch::Tensor>& neuron_tensors,
      const std::vector<TensorContext>& tensor_contexts) override;
};

}  // namespace at::neuron::testing
