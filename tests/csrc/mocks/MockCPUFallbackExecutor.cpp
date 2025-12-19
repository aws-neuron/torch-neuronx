#include "MockCPUFallbackExecutor.h"

namespace at::neuron::testing {

MockCPUFallbackExecutor::MockCPUFallbackExecutor() : CPUFallbackExecutor() {
  // Initialize with default configuration
}

void MockCPUFallbackExecutor::TransferTensorsFromCpuToNeuron(
    const std::vector<torch::Tensor>& cpu_results, const std::vector<nrt_tensor_t*>& data_ptrs) {
  // Just validate sizes match
  TORCH_CHECK(data_ptrs.size() == cpu_results.size(), "Mismatch output tensor sizes (",
              data_ptrs.size(), ") and (", cpu_results.size(), ")");
}

std::vector<torch::Tensor> MockCPUFallbackExecutor::TransferTensorsToCpu(
    const std::vector<nrt_tensor_t*>& data_ptrs,
    const std::vector<TensorContext>& tensor_contexts) {
  // In test environment, create CPU tensors without actual device transfer
  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensor_contexts.size());

  for (const auto& context : tensor_contexts) {
    auto options = context.get_options().device(torch::kCPU);
    cpu_tensors.emplace_back(torch::empty(context.get_shape(), options));
  }
  return cpu_tensors;
}

}  // namespace at::neuron::testing
