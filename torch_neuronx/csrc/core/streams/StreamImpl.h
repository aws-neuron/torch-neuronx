#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/ErrorHandler.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron {

// Tracking info for the last scheduled operation on a stream
// Used for cross-XU synchronization
struct LastScheduledInfo {
  nrta_xu_t xu{NRTA_XU_COMPUTE};
  nrt::SequenceId sequence_id{0};
  bool valid{false};
};

// Result of scheduling a device operation
// Used to communicate what action was taken during scheduling
enum class ScheduleResult {
  kScheduled,  // New operation was scheduled to device
  kAdvanced,   // Cascading op advanced (concat op already scheduled)
  kNotReady    // Could not schedule (needs to wait for cross-XU sync or queue full)
};

// Forward declarations
class NeuronEvent;
class ModelHandleCache;
class OperationExecutionEngine;

// Per-XU completion tracking, sized to each XU's actual queue count.
struct XUCompletionState {
  std::array<nrt::SequenceId, nrt::kTensorOpQueues> tensor_op{};
  std::array<nrt::SequenceId, nrt::kComputeQueues> compute{};
  std::array<nrt::SequenceId, nrt::kCollectivesQueues> collectives{};

  nrt::SequenceId Get(nrta_xu_t xu, int queue) const {
    switch (xu) {
      case NRTA_XU_TENSOR_OP:
        return tensor_op[queue];
      case NRTA_XU_COMPUTE:
        return compute[queue];
      case NRTA_XU_COLLECTIVES:
        return collectives[queue];
      case NRTA_XU_TYPE_NUM:
        break;
    }
    EXECUTION_RUNTIME_CHECK(false, "Unknown XU type: ", static_cast<int>(xu));
  }

  void Set(nrta_xu_t xu, int queue, nrt::SequenceId seq) {
    switch (xu) {
      case NRTA_XU_TENSOR_OP:
        tensor_op[queue] = seq;
        return;
      case NRTA_XU_COMPUTE:
        compute[queue] = seq;
        return;
      case NRTA_XU_COLLECTIVES:
        collectives[queue] = seq;
        return;
      case NRTA_XU_TYPE_NUM:
        break;
    }
    EXECUTION_RUNTIME_CHECK(false, "Unknown XU type: ", static_cast<int>(xu));
  }
};

// StreamImpl provides the core implementation for Neuron stream execution.
// This class implements a multi-threaded pipeline for async operation execution:
// - Build IR thread: Handles MLIR lowering and NEFF compilation
// - Execution thread: Handles NRT async execution
// - Unified operation tracking: Manages operation lifecycle and cleanup
struct StreamImpl : public std::enable_shared_from_this<StreamImpl> {
  friend class OperationExecutionEngine;

  // Global device index
  c10::DeviceIndex device_index;
  // Physical stream ID
  c10::StreamId stream_id;
  // Operation tracking
  std::list<std::shared_ptr<OperationContext>> active_operations_;
  std::mutex active_operations_mutex_;

  // Iterator pointing to the next operation to be scheduled (for nrt-async mode)
  std::list<std::shared_ptr<OperationContext>>::iterator next_to_schedule_;

  // Per-stream error handling
  std::unique_ptr<ErrorHandler> error_handler_;

  // Tracking for last scheduled operation (for cross-XU synchronization)
  LastScheduledInfo last_scheduled_info_;

  // Per-stream completion tracking state for async NRT operations
  XUCompletionState xu_completed_seqs_;

  // Last seq registered for eventfd notification
  nrt::SequenceId last_registered_efd_seq_{0};

  // Cleanup timeout configuration
  std::chrono::milliseconds operation_timeout_;

  // Stashed exception from runtime thread - rethrown on Synchronize()
  std::exception_ptr stashed_exception_;
  std::mutex stashed_exception_mutex_;

  // Completion tracking for synchronization
  std::condition_variable completion_cv;

  StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id);
  ~StreamImpl();

  // Main interface methods
  std::shared_future<OperationContextResult> SubmitOperationContext(
      std::unique_ptr<OperationContext> operation_context);

  bool Query() const;
  void Synchronize();

  void CompleteOperation(OperationContext* operation);

  // Ensure all prior operations preceding the given one are completed.
  c10::StreamId GetStreamId() const { return stream_id; }

  // For async execution worker (operation scheduling)
  OperationContext* GetNextOperationToSchedule();
  void AdvanceToNextOperation();

  // Check if stream has pending async operations (scheduled or pending signal)
  bool HasPendingAsyncOperations() const;

  // Get/set last scheduled info for cross-XU synchronization
  LastScheduledInfo& GetLastScheduledInfo() { return last_scheduled_info_; }
  const LastScheduledInfo& GetLastScheduledInfo() const { return last_scheduled_info_; }

  // Update completion state by querying NRT for latest completed sequences
  void UpdateCompletionState();

  // Schedule a device operation (kHLO, kCollective, kCopy, kWrite, kRead)
  // Handles concatenated operations: cascading ops schedule their concat op first time,
  // subsequent cascading ops just advance since concat op is already scheduled.
  // Returns ScheduleResult indicating what action was taken:
  //   kScheduled: New operation was scheduled to device
  //   kAdvanced: Cascading op advanced (concat op already scheduled by previous cascading op)
  //   kNotReady: Could not schedule (needs to wait for cross-XU sync or queue full)
  ScheduleResult ScheduleDeviceOperation(OperationContext* op, KernelTypeEnum kernel_type);

  // Find operation by sequence ID (for nrt-async error handling)
  // Linear search bounded by kDefaultMaxInflightPerXU and only triggered on errors
  OperationContext* FindOperationBySeq(nrt::SequenceId sequence_id);

 private:
  // Prepare operation on the calling thread if RequiresPrepare() is true.
  void PrepareOperation(OperationContext* op);

  // Operation management
  void RemoveActiveOperation(const OperationContext* operation);

  // Handle the respective error for the current operation
  template <typename ExceptionType>
  void HandleErrorWithCleanup(OperationContext* operation, const ExceptionType& e);

  void HandleErrorWithExceptionPtr(OperationContext* operation, std::exception_ptr eptr);

  void StashException(std::exception_ptr eptr) {
    std::lock_guard<std::mutex> lock(stashed_exception_mutex_);
    if (!stashed_exception_) {
      stashed_exception_ = std::move(eptr);
    }
  }
};

}  // namespace at::neuron
