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

  // --- Per-op error handlers (opt-in via register_aten(error_handler=...)) ---
  // True if `e` is an out-of-bounds device fault (NRT_EXEC_OOB). On such a
  // fault the error path invokes the faulting op's registered error handler (if
  // any) to raise the exact op-specific exception; see OpErrorHandler.
  bool ShouldSurfaceOOBException(const torch_neuronx::ExecutionRuntimeException& e) const;
  // Invoke the faulting op's error handler to raise the exact op-specific
  // exception; propagates it so the caller records it as the pending error.
  // No-ops if the op has no handler. Does not run the op. See OpErrorHandler.
  void SurfaceOOBException(OperationContext* op);

 private:
  ErrorRecovery(const ErrorRecovery&) = delete;
  ErrorRecovery& operator=(const ErrorRecovery&) = delete;

  void LoadConfiguration();
  OperationContextResult AttemptCompilableRecovery(OperationContext* operation);

  bool ShouldAttemptOOMRecovery(const torch_neuronx::ExecutionRuntimeException& e);
  OperationContextResult AttemptOOMRecovery(OperationContext* op);

  bool fallback_enabled_;
  bool oom_recovery_enabled_;
  std::unique_ptr<CPUFallbackExecutor> cpu_fallback_executor_;
};

/**
 * Handle concatenation failure by invoking the callback stored in ConcatenationState
 * This allows error handling without direct reference to ConcatenationEngine
 * @param operation The operation whose concatenation failed (can be individual or concatenated op)
 */
void RecoverConcatenationFailure(OperationContext* operation);

}  // namespace at::neuron
