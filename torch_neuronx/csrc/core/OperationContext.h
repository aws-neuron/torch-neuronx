#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "KernelExecution.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"

namespace at::neuron {

// Result of an operation execution with status and error information.
struct OperationContextResult {
  enum Status {
    kPending = 1,    // Operation is pending/in-progress
    kCompleted = 0,  // Operation completed successfully (keeping 0 for backward compatibility)
    kFailed = -1     // Operation failed (keeping negative for backward compatibility)
  };

  Status status;              // Execution status
  std::string error_message;  // Error details if failed

  // Constructors
  OperationContextResult() : status(kPending) {}

  OperationContextResult(Status status_code, const std::string& error = "")
      : status(status_code), error_message(error) {}

  // Status check methods
  bool IsSuccess() const { return status == kCompleted; }
  bool IsPending() const { return status == kPending; }
  const std::string& GetError() const { return error_message; }

  static OperationContextResult CreateSuccess() { return OperationContextResult(kCompleted); }

  static OperationContextResult CreateError(const std::string& message) {
    return OperationContextResult(kFailed, message);
  }
};

// Represents a complete operation that transforms from input to result state.
// Combines the functionality of PipelineOperation and ExecutionResult into a single object
// that evolves through the execution pipeline stages.
struct OperationContext {
  // Core execution data (immutable after creation)
  std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_execution;

  // Pipeline state (populated during execution)
  std::string python_stack_trace;  // Python stack trace for debugging

  // Completion handling
  std::promise<OperationContextResult> promise;

  // Event dependencies - events this operation must wait for before starting
  std::vector<NeuronEvent> events_to_wait_for;

  // Timing information for performance analysis
  std::chrono::steady_clock::time_point submit_time;    // When operation was submitted
  std::chrono::steady_clock::time_point compile_start;  // When compilation started
  std::chrono::steady_clock::time_point compile_end;    // When compilation completed
  std::chrono::steady_clock::time_point execute_start;  // When execution started
  std::chrono::steady_clock::time_point execute_end;    // When execution completed

  explicit OperationContext(std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_exec,
                            const std::string& python_stack_trace = "")
      : kernel_execution(std::move(kernel_exec)),
        python_stack_trace(python_stack_trace),
        promise() {}

  // Timing accessors - return zero duration if timing points are not set
  std::chrono::microseconds GetCompilationTime() const {
    if (IsTimePointUnset(compile_start) || IsTimePointUnset(compile_end)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(compile_end - compile_start);
  }

  std::chrono::microseconds GetExecutionTime() const {
    if (IsTimePointUnset(execute_start) || IsTimePointUnset(execute_end)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(execute_end - execute_start);
  }

  std::chrono::microseconds GetQueueTime() const {
    if (IsTimePointUnset(submit_time) || IsTimePointUnset(execute_start)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(execute_start - submit_time);
  }

  std::chrono::microseconds GetTotalTime() const {
    return GetCompilationTime() + GetQueueTime() + GetExecutionTime();
  }

  std::chrono::microseconds GetPipelineTime() const {
    if (IsTimePointUnset(submit_time) || IsTimePointUnset(execute_start)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(execute_start - submit_time);
  }

  // Validation and accessors
  bool IsValid() const { return kernel_execution->IsValid(); }

  std::string GetOpName() const { return kernel_execution->GetOpName(); }

  // Lifecycle management
  void StartCompilation() { compile_start = std::chrono::steady_clock::now(); }
  void CompleteCompilation() { compile_end = std::chrono::steady_clock::now(); }

  void StartExecution() { execute_start = std::chrono::steady_clock::now(); }
  void CompleteExecution() { execute_start = std::chrono::steady_clock::now(); }

  // Set submission time (should be called when operation is first submitted).
  void MarkSubmitted() { submit_time = std::chrono::steady_clock::now(); }

  // Check if this operation requires compilation.
  bool RequiresCompilation() const { return kernel_execution->RequiresCompilation(); }

  // Get the compilable kernel execution (only valid if RequiresCompilation() returns true).
  CompilableKernelExecution& GetCompilableKernel() const {
    if (!kernel_execution->RequiresCompilation()) {
      throw std::logic_error("GetCompilableKernel() called on direct kernel execution (op: " +
                             kernel_execution->GetOpName() + ")");
    }

    return *static_cast<CompilableKernelExecution*>(kernel_execution.get());
  }

  // Get the direct kernel execution (only valid if RequiresCompilation() returns false).
  DirectKernelExecution& GetDirectKernel() const {
    if (kernel_execution->RequiresCompilation()) {
      throw std::logic_error("GetDirectKernel() called on compilable kernel execution (op: " +
                             kernel_execution->GetOpName() + ")");
    }

    return *static_cast<DirectKernelExecution*>(kernel_execution.get());
  }

  void Execute() {
    StartExecution();
    kernel_execution->Execute();
    CompleteExecution();
  }

 private:
  OperationContext() = default;

  // Helper function to check if a time point is unset
  static bool IsTimePointUnset(const std::chrono::steady_clock::time_point& time_point) {
    return time_point.time_since_epoch().count() == 0;
  }
};

}  // namespace at::neuron
