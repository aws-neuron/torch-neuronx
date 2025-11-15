#include "MockCPUFallbackExecutor.h"

namespace at::neuron::testing {

MockCPUFallbackExecutor::MockCPUFallbackExecutor() : CPUFallbackExecutor() {
  // Initialize with default configuration
}

void MockCPUFallbackExecutor::HandleNeuronOutputTensorReuse(
    const std::vector<torch::Tensor>& cpu_results, OperationContext* op) {
  // In test environment, just copy CPU results directly to original tensors
  // This avoids the .to(neuron_device) call that fails in test environment
  const auto& original_neuron_outputs = op->kernel_execution->GetOutputs();

  TORCH_CHECK(original_neuron_outputs.size() == cpu_results.size(),
              "Mismatch output tensor sizes (", original_neuron_outputs.size(), ") and (",
              cpu_results.size(), ") for operation ", op->GetOpName());

  for (size_t i = 0; i < original_neuron_outputs.size(); ++i) {
    const auto& original_neuron_tensor = original_neuron_outputs[i];
    const auto& cpu_result = cpu_results[i];

    try {
      // Get mutable reference to original tensor
      torch::Tensor& mutable_original = const_cast<torch::Tensor&>(original_neuron_tensor);

      // Resize original tensor if needed
      if (mutable_original.sizes() != cpu_result.sizes()) {
        mutable_original.resize_(cpu_result.sizes());
      }

      // In test environment, just copy CPU result directly (no device transfer)
      mutable_original.copy_(cpu_result, /*non_blocking=*/false);

      TORCH_NEURONX_DEBUG("Mock: Reused original output tensor", "operation=", op->GetOpName(),
                          "index=", i, "final_shape=", mutable_original.sizes());

    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("Mock: Failed to reuse original output tensor",
                          "operation=", op->GetOpName(), "index=", i, "error=", e.what());
      throw std::runtime_error("Mock: Output tensor reuse failed for index " + std::to_string(i) +
                               ": " + std::string(e.what()));
    }
  }
}

}  // namespace at::neuron::testing
