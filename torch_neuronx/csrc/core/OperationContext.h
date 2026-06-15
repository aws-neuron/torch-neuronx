#pragma once

#include <c10/core/Device.h>
#include <sys/syscall.h>
#include <torch/torch.h>
#include <unistd.h>

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "KernelExecution.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace at::neuron {

// OS thread ID via gettid(2) — avoids depending on Kineto's ThreadUtil.h from core code.
inline int32_t getOsTid() { return static_cast<int32_t>(syscall(SYS_gettid)); }

// Sequence number for kHost completion ordering.
using HostCompletionSeq = uint64_t;

// Forward declaration
struct OperationContext;

// Type alias for concatenation failure callback
// The callback takes the failed concatenated operation as parameter
using ConcatenationFailureCallback = std::function<void(OperationContext*)>;

// Represents a concatenation group - shared between all operations in the group.
// This class encapsulates all state related to operation concatenation,
// providing a single source of truth for the concatenation relationship.
// Immutable after construction - all data must be provided at construction time.
//
// ## Ownership Model
//
// ConcatenationState owns the concatenated operation via unique_ptr and is kept alive
// by cascading operations via shared_ptr. The concatenated operation uses a non-owning
// raw pointer back to the state to avoid circular references.
//
// ```
// ┌─────────────────────────────────────────────┐
// │           ConcatenationState                │
// │  ┌─────────────────────────────────────┐   │
// │  │ unique_ptr<OperationContext>        │───┼──> Concatenated Op
// │  │     concatenated_operation_         │   │        │
// │  └─────────────────────────────────────┘   │        │
// │                                             │        │ concatenation_state_raw_ (raw ptr)
// │                                             │<───────┘ (non-owning, breaks cycle)
// │                                             │
// │  cascading_ops = [op1, op2, ...]           │
// └─────────────────────────────────────────────┘
//          ↑                ↑
//          │                │
//     shared_ptr       shared_ptr
//          │                │
//     Cascading Op1    Cascading Op2
// ```
//
// - Cascading ops hold shared_ptr<ConcatenationState> to keep the state (and concat op) alive
// - Concatenated op holds raw ConcatenationState* to avoid circular reference
// - When all cascading ops are done, shared_ptr ref count drops to 0, state is destroyed,
//   and unique_ptr automatically deletes the concatenated operation
//
class ConcatenationState {
 public:
  // Constructor requires all state upfront - immutable after construction
  // @param concat_op The merged/concatenated operation (must not be null, ownership transferred)
  // @param cascading_ops Individual operations that were merged (must not contain nulls)
  // @param failure_callback Callback invoked on concatenation failure (required)
  ConcatenationState(std::unique_ptr<OperationContext> concat_op,
                     std::vector<OperationContext*> cascading_ops,
                     ConcatenationFailureCallback failure_callback)
      : concatenated_operation_(std::move(concat_op)),
        cascading_operations_(std::move(cascading_ops)),
        failure_callback_(std::move(failure_callback)) {
    TORCH_CHECK(concatenated_operation_ != nullptr,
                "ConcatenationState requires a non-null concatenated operation");
    TORCH_CHECK(failure_callback_ != nullptr,
                "ConcatenationState requires a non-null failure callback");
    // Validate no null cascading operations
    for (auto* op : cascading_operations_) {
      TORCH_CHECK(op != nullptr, "ConcatenationState cannot contain null cascading operations");
    }
  }

  // Disable copy/move due to std::atomic member
  ConcatenationState(const ConcatenationState&) = delete;
  ConcatenationState& operator=(const ConcatenationState&) = delete;
  ConcatenationState(ConcatenationState&&) = delete;
  ConcatenationState& operator=(ConcatenationState&&) = delete;

  // Get the concatenated (merged) operation
  OperationContext* GetConcatenatedOperation() const { return concatenated_operation_.get(); }

  // Invoke the failure callback (always set, required by constructor)
  void InvokeFailureCallback(OperationContext* failed_op) { failure_callback_(failed_op); }

  // Get all cascading operations
  const std::vector<OperationContext*>& GetCascadingOperations() const {
    return cascading_operations_;
  }

  // Get the number of cascading operations
  size_t GetCascadingOperationsCount() const { return cascading_operations_.size(); }

  // Increment the count of compiled cascading operations and check if all are done
  // Returns true if this was the last operation to compile
  bool IncrementAndCheckCompiledCascadingOpsCount() {
    size_t new_count = compiled_cascading_ops_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
    EXECUTION_RUNTIME_CHECK(new_count <= cascading_operations_.size(),
                            "Compiled count exceeds cascading operations count");
    return new_count >= cascading_operations_.size();
  }

  // Get the current count of compiled cascading operations
  size_t GetCompiledCascadingOpsCount() const {
    return compiled_cascading_ops_count_.load(std::memory_order_acquire);
  }

 private:
  // The merged operation that will actually execute (sole owner)
  const std::unique_ptr<OperationContext> concatenated_operation_;

  // Individual operations that were merged into concatenated_operation
  const std::vector<OperationContext*> cascading_operations_;

  // Callback to handle concatenation failures (invalidate cache, cleanup, etc.)
  const ConcatenationFailureCallback failure_callback_;

  // Compilation completion tracking (only mutable member - tracks runtime progress)
  std::atomic<size_t> compiled_cascading_ops_count_{0};
};

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
DEFINE_KERNEL_TYPE_TRAITS(kCopy, CopyDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kWrite, WriteDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kRead, ReadDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kEvent, EventDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kHint, HintDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kAlloc, AllocDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kDealloc, DeallocDirectKernelExecution)
DEFINE_KERNEL_TYPE_TRAITS(kHost, HostKernelExecution)

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
  std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_execution = nullptr;

  // CPU fallback metadata (bundles static args and list arg reconstruction data)
  CPUFallbackContext cpu_fallback_context = {};

  // Pipeline state (populated during execution)
  std::string python_stack_trace = "";

  // Per-execution hierarchy string for profiler Layer/Hierarchy view.
  // Captured at submission time from the live Python call stack.
  // Forward: "ClassName[attr][count]/..." from nn.Module frames.
  // Backward: "op_label[backward][count]/" from op_name.
  std::string hierarchy_string = "";

  // Stream context - pointer to the stream this operation belongs to
  StreamImpl* stream = nullptr;

  // Logical (user-facing) Stream ID
  c10::StreamId stream_id{0};

  // Priority from PyTorch stream (decoded from stream_id at submission)
  int stream_priority = 0;

  // Completion handling
  std::promise<OperationContextResult> promise = {};
  std::shared_future<OperationContextResult> result_future = {};

  // Execution readiness state (set after compilation completes or for direct ops)
  std::atomic<bool> execution_ready_{false};

  // nrt-async execution state
  nrt::AsyncSchedulingState nrt_async_state_ = {};

  // Per-operation async completion status, written on completion.
  // Default NRT_FAILURE ensures unset status is treated as error (fail-closed).
  NRT_STATUS nrt_status_{NRT_FAILURE};

  // For kHost ops: the per-stream host-completion seq stamped at kHost dispatch.
  HostCompletionSeq host_completion_seq_{0};

  bool HasNRTError() const { return nrt_status_ != NRT_SUCCESS; }

  torch_neuronx::ExecutionRuntimeException GetNRTError() const {
    return torch_neuronx::ExecutionRuntimeException(
        "Async execution failed with NRT status: " + std::to_string(nrt_status_), nrt_status_);
  }

  // Concatenation support - state for operations in a concatenation group
  // For individual cascading ops: shared_ptr keeps the state (and concatenated op) alive
  // For concatenated op itself: raw pointer (non-owning) to avoid circular reference
  //   since ConcatenationState owns the concatenated op via unique_ptr
  std::shared_ptr<ConcatenationState> concatenation_state_ = nullptr;
  ConcatenationState* concatenation_state_raw_ = nullptr;

  // Timing information for performance analysis
  std::chrono::steady_clock::time_point submit_start =
      {};  // Top of SubmitOperationContext (PyTorch thread)
  std::chrono::steady_clock::time_point submit_time = {};
  std::chrono::steady_clock::time_point compile_start = {};
  std::chrono::steady_clock::time_point compile_end = {};
  std::chrono::steady_clock::time_point execute_start = {};
  std::chrono::steady_clock::time_point execute_end = {};
  std::chrono::steady_clock::time_point concat_start = {};
  std::chrono::steady_clock::time_point concat_end = {};
  std::chrono::steady_clock::time_point dispatch_start = {};  // RuntimeWorker picks up op

  // PyTorch tracing identifiers for linking with torch profiler and neuron-tools.
  // Upon SubmitOperationContext, pytorch_sequence_nr and pytorch_thread_id are populated with the
  // corresponding torch sequence_number and thread properties.

  // global op sequence number
  int64_t pytorch_sequence_nr = -1;
  // PyTorch's per-thread ID. Pytorch thread ids are different from os thread ids.
  uint64_t pytorch_thread_id = 0;

  // OS thread ID - named in the standardize naming format of tid
  uint64_t os_tid = 0;
  // Kineto correlation ID for linking device activities to CPU ops
  uint64_t kineto_correlation_id{0};

  // OS thread ID (gettid) of the Neuron stream execution worker that scheduled this operation
  int32_t neuron_stream_os_tid{-1};  // -1 = not yet assigned; valid tids are positive (gettid())
  int32_t xu_type{-1};               // Resolved XU type after scheduling (NRTA_XU_COMPUTE etc.)
  int32_t dispatch_slot{-1};         // Inflight slot for parallel trace visualization (0-7 per XU)

  // For kHost ops: index of the HostWorkerPool worker that ran RunHost (-1 if
  // not a host-pool op).
  int32_t host_worker_idx{-1};

  // Unique event ID for framework-to-NRT event linking
  uint64_t event_id = 0;

  explicit OperationContext(std::unique_ptr<at::neuron::NeuronKernelExecution> kernel_exec,
                            const std::string& python_stack_trace = "",
                            CPUFallbackContext fallback_context = {},
                            const std::string& hierarchy_string = "")
      : kernel_execution(std::move(kernel_exec)),
        cpu_fallback_context(std::move(fallback_context)),
        python_stack_trace(python_stack_trace),
        hierarchy_string(hierarchy_string),
        stream(nullptr),
        promise(),
        result_future(promise.get_future().share()) {}

  // OperationContext is non-copyable / non-movable: it owns a std::promise
  // member (promise) which is move-only, plus std::atomic members which are
  // neither.
  OperationContext(const OperationContext&) = delete;
  OperationContext& operator=(const OperationContext&) = delete;
  OperationContext(OperationContext&&) = delete;
  OperationContext& operator=(OperationContext&&) = delete;

  // Accessor for CPU fallback context
  const CPUFallbackContext& GetCPUFallbackContext() const { return cpu_fallback_context; }

  // Timing accessors - return zero duration if timing points are not set
  std::chrono::microseconds GetConcatenationTime() const {
    if (IsTimePointUnset(concat_start) || IsTimePointUnset(concat_end)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(concat_end - concat_start);
  }

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

  std::chrono::microseconds GetTimeBetween(std::chrono::steady_clock::time_point& t1,
                                           std::chrono::steady_clock::time_point& t2) const {
    if (IsTimePointUnset(t1) || IsTimePointUnset(t2)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  }

  std::chrono::microseconds GetExecutionGap() const {
    if (IsTimePointUnset(concat_end) || IsTimePointUnset(execute_start)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(execute_start - concat_end);
  }

  std::chrono::microseconds GetQueueTime() const {
    if (IsTimePointUnset(submit_time) || IsTimePointUnset(execute_start)) {
      return std::chrono::microseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(execute_start - submit_time);
  }

  std::chrono::microseconds GetTotalTime() const {
    return GetConcatenationTime() + GetCompilationTime() + GetQueueTime() + GetExecutionTime();
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
  void StartConcatenation() { concat_start = std::chrono::steady_clock::now(); }
  void CompleteConcatenation() { concat_end = std::chrono::steady_clock::now(); }

  void StartCompilation() { compile_start = std::chrono::steady_clock::now(); }
  void CompleteCompilation() { compile_end = std::chrono::steady_clock::now(); }

  void StartExecution() {
    execute_start = std::chrono::steady_clock::now();
    neuron_stream_os_tid = getOsTid();
  }
  void CompleteExecution() { execute_end = std::chrono::steady_clock::now(); }

  // Set submission time (should be called when operation is first submitted).
  void MarkSubmitted() { submit_time = std::chrono::steady_clock::now(); }

  // Check if this operation requires compilation.
  bool RequiresCompilation() const { return kernel_execution->RequiresCompilation(); }

  // Check if this operation requires preparation (e.g., collectives).
  bool RequiresPrepare() const { return kernel_execution->RequiresPrepare(); }

  // Mark operation as ready for execution (compilation complete or direct op)
  void MarkExecutionReady() { execution_ready_.store(true, std::memory_order_release); }
  // Check if operation is ready for execution
  bool IsExecutionReady() const { return execution_ready_.load(std::memory_order_acquire); }

  // Check if this operation is ready to execute (for event wait primitives)
  bool IsSchedulable() const {
    // Events must be both execution ready (pipeline processed) AND event ready (dependencies met)
    if (GetKernelType() == KernelTypeEnum::kEvent) {
      return IsExecutionReady() && GetKernel<KernelTypeEnum::kEvent>().IsReady();
    }

    // kAlloc: if we deferred due to NRT_RESOURCE, gate re-entry until the
    // stream fully drains and clears the pending flag from a completion hook.
    if (GetKernelType() == KernelTypeEnum::kAlloc &&
        GetKernel<KernelTypeEnum::kAlloc>().GetAllocState() ==
            at::neuron::AllocDirectKernelExecution::AllocState::Deferred) {
      return false;
    }

    // If this operation has a concatenated operation, check if the concatenated operation is ready
    // instead of checking this individual operation
    if (HasConcatenatedOperation()) {
      return GetConcatenatedOperation()->IsExecutionReady() && IsExecutionReady();
    }
    return IsExecutionReady();
  }

  // Async scheduling helper
  void MarkScheduled(nrt::SequenceId sequence_id) {
    nrt_async_state_.sequence_id = sequence_id;
    nrt_async_state_.is_scheduled = true;
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

  // ============== Concatenation State Helpers ==============

  // Get the concatenation state (works for both cascading ops and concatenated op)
  ConcatenationState* GetConcatenationState() const {
    if (concatenation_state_) return concatenation_state_.get();
    return concatenation_state_raw_;
  }

  // Check if this operation IS the concatenated (merged) operation
  // Returns true only for the merged operation itself (uses raw pointer)
  bool IsConcatenatedOperation() const { return concatenation_state_raw_ != nullptr; }

  // Check if this operation is part of a concatenation group
  // Returns true for both individual cascading ops AND the concatenated operation itself
  bool HasConcatenatedOperation() const {
    auto* state = GetConcatenationState();
    return state && state->GetConcatenatedOperation() != nullptr;
  }

  // Get the concatenated operation (returns nullptr if not part of a concatenation group)
  OperationContext* GetConcatenatedOperation() const {
    auto* state = GetConcatenationState();
    return state ? state->GetConcatenatedOperation() : nullptr;
  }

 private:
  OperationContext() = default;

  // Helper function to check if a time point is unset
  static bool IsTimePointUnset(const std::chrono::steady_clock::time_point& time_point) {
    return time_point.time_since_epoch().count() == 0;
  }
};

}  // namespace at::neuron
