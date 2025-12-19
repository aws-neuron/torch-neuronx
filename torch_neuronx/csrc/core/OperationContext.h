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
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace at::neuron {

// Traits to map KernelTypeEnum to actual kernel types
template <KernelTypeEnum KernelType>
struct KernelTypeTraits;

#define DEFINE_KERNEL_TYPE_TRAITS(ENUM_VALUE, KERNEL_TYPE) \
  template <>                                              \
  struct KernelTypeTraits<KernelTypeEnum::ENUM_VALUE> {    \
    using type = KERNEL_TYPE;                              \
  };

DEFINE_KERNEL_TYPE_TRAITS(kHLO, XLACompilableKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kCollective, CollectiveDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kAlloc, AllocDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kDealloc, DeallocDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kCopy, CopyDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kWrite, WriteDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kRead, ReadDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kSlice, SliceDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kEvent, EventDirectKernelExecution)

#undef DEFINE_KERNEL_TYPE_TRAITS

// Alias template for cleaner syntax
template <KernelTypeEnum KernelType>
using KernelType_t = typename KernelTypeTraits<KernelType>::type;

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

// Forward declaration
class StreamImpl;

// Represents a complete operation that transforms from input to result state.
// Combines the functionality of PipelineOperation and ExecutionResult into a single object
// that evolves through the execution pipeline stages.
struct OperationContext {
  // Core execution data (immutable after creation)
  std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_execution;

  // Tensor metadata (immutable after creation)
  // Applicable only for xla_kernel and nki_kernel
  std::vector<TensorContext> input_tensors_context;
  std::vector<TensorContext> output_tensors_context;

  // CPU fallback metadata (bundles static args and list arg reconstruction data)
  CPUFallbackContext cpu_fallback_context;

  // Pipeline state (populated during execution)
  std::string python_stack_trace;  // Python stack trace for debugging

  // Stream context - pointer to the stream this operation belongs to
  StreamImpl* stream;

  // Completion handling
  std::promise<OperationContextResult> promise;

  // Execution readiness state (set after compilation completes or for direct ops)
  std::atomic<bool> execution_ready_{false};

  // Timing information for performance analysis
  std::chrono::steady_clock::time_point submit_time;    // When operation was submitted
  std::chrono::steady_clock::time_point compile_start;  // When compilation started
  std::chrono::steady_clock::time_point compile_end;    // When compilation completed
  std::chrono::steady_clock::time_point execute_start;  // When execution started
  std::chrono::steady_clock::time_point execute_end;    // When execution completed

  explicit OperationContext(std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_exec,
                            const std::vector<TensorContext>& input_contexts = {},
                            const std::vector<TensorContext>& output_contexts = {},
                            const std::string& python_stack_trace = "",
                            CPUFallbackContext fallback_context = {})
      : kernel_execution(std::move(kernel_exec)),
        input_tensors_context(input_contexts),
        output_tensors_context(output_contexts),
        cpu_fallback_context(std::move(fallback_context)),
        python_stack_trace(python_stack_trace),
        stream(nullptr),
        promise() {}

  // Accessor for CPU fallback context
  const CPUFallbackContext& GetCPUFallbackContext() const { return cpu_fallback_context; }

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

  // Mark operation as ready for execution (compilation complete or direct op)
  void MarkExecutionReady() { execution_ready_.store(true, std::memory_order_release); }
  // Check if operation is ready for execution
  bool IsExecutionReady() const { return execution_ready_.load(std::memory_order_acquire); }

  // Check if this operation is ready to execute (for event wait primitives)
  bool IsSchedulable() const {
    if (GetKernelType() == KernelTypeEnum::kEvent) {
      return GetKernel<KernelTypeEnum::kEvent>().IsReady();
    }
    return IsExecutionReady();
  }

  KernelTypeEnum GetKernelType() const { return kernel_execution->GetKernelType(); }

  // Get the compilable kernel execution (only valid if RequiresCompilation() returns true).
  template <KernelTypeEnum KernelType>
  KernelType_t<KernelType>& GetKernel() const {
    return *static_cast<KernelType_t<KernelType>*>(kernel_execution.get());
  }

  CompilableKernelExecution& GetCompilableKernel() const {
    auto* compilable = dynamic_cast<CompilableKernelExecution*>(kernel_execution.get());
    TORCH_CHECK(compilable, "Kernel execution is not a compilable kernel");
    return *compilable;
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
