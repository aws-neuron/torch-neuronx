#include "torch_neuron/csrc/core/KernelExecution.h"
#include "torch_neuron/csrc/core/runtime/NRTHandler.h"

namespace at::neuron {

// Mock implementation of NRTHandler::ExecuteCompilableKernel for testing
// This allows tests to run without actual Neuron hardware
NRTHandler::ExecutionResult NRTHandler::ExecuteCompilableKernel(
    const CompilableKernelExecution& compilable_kernel, const std::string& op_name,
    const ExecutionConfig& config) {
  // Return a successful mock result
  ExecutionResult result;
  result.success = true;
  result.execution_time = std::chrono::milliseconds(1);
  result.error_message = "";

  return result;
}

}  // namespace at::neuron
