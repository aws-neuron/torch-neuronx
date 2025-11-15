#pragma once

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"

namespace at::neuron::testing {

/**
 * Mock CPUFallbackExecutor that avoids device transfer issues in test environment
 *
 * This mock overrides the problematic handle_neuron_output_tensor_reuse method
 * to avoid PrivateUse1 (Neuron) device operations that fail in test environments
 * where the Neuron backend is not fully initialized.
 */
class MockCPUFallbackExecutor : public CPUFallbackExecutor {
 public:
  MockCPUFallbackExecutor();
  virtual ~MockCPUFallbackExecutor() = default;

 protected:
  /**
   * Override the problematic function to avoid PrivateUse1 device issues
   *
   * In test environment, this method avoids the .to(neuron_device) call that
   * fails when the Neuron backend is not properly initialized. Instead, it
   * performs direct CPU tensor copies.
   */
  void HandleNeuronOutputTensorReuse(const std::vector<torch::Tensor>& cpu_results,
                                     OperationContext* op) override;
};

}  // namespace at::neuron::testing
