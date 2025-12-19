#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

namespace at::neuron {

// Forward declarations
class CPUFallbackExecutor;

/**
 * ErrorRecovery provides recovery strategies for failed Neuron operations
 * Per-stream instance that attempts various recovery strategies including:
 * - OOM recovery (clear cache and retry)
 * - CPU fallback for compilation failures
 */
class ErrorRecovery {
 public:
  /**
   * Constructor for per-stream ErrorRecovery instance
   */
  ErrorRecovery();

  /**
   * Attempt to recover from an operation compilation failure
   *
   * @param op The failed operation
   * @param e CompilationRuntimeException/ExecutionRuntimeException exception
   * @return OperationContextResult indicating success or continued failure
   */
  OperationContextResult AttemptRecovery(OperationContext* op,
                                         const torch_neuronx::CompilationRuntimeException& e);
  OperationContextResult AttemptRecovery(OperationContext* op,
                                         const torch_neuronx::ExecutionRuntimeException& e);

  /**
   * Check if an error is an OOM error that can be recovered
   * @param nrt_status The NRT status code
   * @return true if this is a recoverable OOM error
   */
  bool IsRecoverableOOM(int nrt_status) const;

  /**
   * Attempt OOM recovery by clearing cache and retrying
   * @param op The failed operation
   * @param device The device where OOM occurred
   * @return OperationContextResult indicating success or failure
   */
  OperationContextResult AttemptOOMRecovery(OperationContext* op, int device);

  /**
   * Check if we should attempt CPU fallback for a failed operation
   */
  bool ShouldAttemptCpuFallback();

  /**
   * Check if we should attempt OOM recovery for a failed operation
   */
  bool ShouldAttemptOOMRecovery(const torch_neuronx::ExecutionRuntimeException& e);

  /**
   * Execute CPU fallback for a failed compilable kernel
   */
  OperationContextResult ExecuteCpuFallback(OperationContext* op);

  /**
   * Destructor
   */
  ~ErrorRecovery();

 private:
  // Disable copy and assignment
  ErrorRecovery(const ErrorRecovery&) = delete;
  ErrorRecovery& operator=(const ErrorRecovery&) = delete;

  /**
   * Load configuration from environment variables
   */
  void LoadConfiguration();

  /**
   * Attempt to recover from a compilable operation failure
   *
   * @param operation OperationContext for a compilable kernel
   * @return OperationContextResult indicating success or continued failure
   */
  OperationContextResult AttemptCompilableRecovery(OperationContext* operation);
  /**
   * Attempt to recover from a executable allocation kernel failure
   *
   * @param kernel The allocation direct kernel execution
   * @return OperationContextResult indicating success or continued failure
   */
  OperationContextResult AttemptRecovery(const AllocDirectKernelExecution& kernel,
                                         const torch_neuronx::ExecutionRuntimeException& e);

  // Configuration flags
  bool fallback_enabled_;
  bool oom_recovery_enabled_;

  // CPU fallback executor
  std::unique_ptr<CPUFallbackExecutor> cpu_fallback_executor_;
};

}  // namespace at::neuron
