#pragma once

#include <memory>
#include <string>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

namespace at::neuron {

class CPUFallbackExecutor;

/**
 * ErrorRecovery provides recovery strategies for failed Neuron operations.
 * Currently supports CPU fallback
 */
class ErrorRecovery {
 public:
  ErrorRecovery();
  ~ErrorRecovery();

  OperationContextResult AttemptRecovery(OperationContext* op,
                                         const torch_neuronx::CompilationRuntimeException& e);
  OperationContextResult AttemptRecovery(OperationContext* op,
                                         const torch_neuronx::ExecutionRuntimeException& e);

  bool ShouldAttemptCpuFallback();
  OperationContextResult ExecuteCpuFallback(OperationContext* op);

 private:
  ErrorRecovery(const ErrorRecovery&) = delete;
  ErrorRecovery& operator=(const ErrorRecovery&) = delete;

  void LoadConfiguration();
  OperationContextResult AttemptCompilableRecovery(OperationContext* operation);

  bool fallback_enabled_;
  std::unique_ptr<CPUFallbackExecutor> cpu_fallback_executor_;
};

/**
 * Handle concatenation failure by invoking the callback stored in ConcatenationState
 * This allows error handling without direct reference to ConcatenationEngine
 * @param operation The operation whose concatenation failed (can be individual or concatenated op)
 */
void RecoverConcatenationFailure(OperationContext* operation);

}  // namespace at::neuron
