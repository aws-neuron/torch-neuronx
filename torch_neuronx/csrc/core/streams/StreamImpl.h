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

// Discriminator for the last scheduled operation's "kind".
// kDeviceXU: prior op was scheduled on an NRT device XU (kHLO/kCollective/
//            kCopy/kWrite/kRead). Sync via xu_completed_seqs_ retire check.
// kHost:     prior op was a kHost (host-compute on HostWorkerPool). Sync via
//            host_completion_state_.completed_seq_id >= the producer's
//            host_completion_seq (seq compare, same shape as the device gate).
enum class LastKind : uint8_t { kDeviceXU, kHost };

// Tracking info for the last scheduled operation on a stream.
// Used for cross-XU and cross-kind (device↔host) synchronization.
struct LastScheduledInfo {
  LastKind kind{LastKind::kDeviceXU};

  // Valid when kind == kDeviceXU
  nrta_xu_t xu{NRTA_XU_COMPUTE};
  nrt::SequenceId sequence_id{0};

  // Valid when kind == kHost. The host-completion seq stamped to the prior
  // kHost op;
  HostCompletionSeq host_completion_seq{0};

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

// Per-stream kHost completion tracking — host-side analogue of XUCompletionState.
struct HostCompletionState {
  std::atomic<HostCompletionSeq> next_seq_id{1};
  std::atomic<HostCompletionSeq> completed_seq_id{0};

  void AdvanceCompleted(HostCompletionSeq seq_id) {
    HostCompletionSeq cur = completed_seq_id.load(std::memory_order_relaxed);
    while (cur < seq_id && !completed_seq_id.compare_exchange_weak(
                               cur, seq_id, std::memory_order_release, std::memory_order_relaxed)) {
    }
  }
};

// StreamImpl owns the per-stream committed runtime queue — the FIFO that the
// RuntimeWorker schedules from. The engine owns the global dispatch_queue_
// in front of it; the concat worker transfers a partitioned prefix from
// dispatch_queue_ into this stream's runtime_queue_ at flush time.
struct StreamImpl : public std::enable_shared_from_this<StreamImpl> {
  // Engine splices ops, advances the cursor, and inspects runtime_queue_.
  friend class OperationExecutionEngine;

  // Global device index
  c10::DeviceIndex device_index;
  // Physical stream ID
  c10::StreamId stream_id;

  // Number of kAllocs currently parked in AllocState::Deferred on this stream.
  std::atomic<int> deferred_alloc_count_{0};

  // Per-stream error handling
  std::unique_ptr<ErrorHandler> error_handler_;

  // Tracking for last scheduled operation (for cross-XU synchronization)
  LastScheduledInfo last_scheduled_info_;

  // Per-stream completion tracking state for async NRT operations
  XUCompletionState xu_completed_seqs_;

  // Per-stream kHost completion tracking (host-side analogue of the above).
  // See HostCompletionState.
  HostCompletionState host_completion_state_;

  // Last seq registered for eventfd notification
  nrt::SequenceId last_registered_efd_seq_{0};

  // Cleanup timeout configuration
  std::chrono::milliseconds operation_timeout_;

  // Stashed exception from runtime thread - rethrown on Synchronize()
  std::exception_ptr stashed_exception_;
  std::mutex stashed_exception_mutex_;

  // Completion tracking for synchronization
  std::condition_variable completion_cv_;
  std::mutex completion_mutex_;

  StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id);
  ~StreamImpl();

  // Main interface methods
  std::shared_future<OperationContextResult> SubmitOperationContext(
      std::unique_ptr<OperationContext> operation_context);

  bool Query() const;
  void Synchronize();

  // Waits for the stream to drain without reporting operation errors. Safe in a
  // destructor, where a rethrow from Synchronize() would call std::terminate.
  void Drain();

  void CompleteOperation(OperationContext* operation);

  // Ensure all prior operations preceding the given one are completed.
  c10::StreamId GetStreamId() const { return stream_id; }

  // For async execution worker (operation scheduling)
  OperationContext* GetNextOperationToSchedule();

  // Single source of truth for the four cross-kind / cross-XU / cross-stream
  // sync gates that decide whether `op` may be scheduled now or must wait for a
  // prior op on this stream to retire. Returns true iff `op` is gated (must
  // park / not-ready).
  // Reads last_scheduled_info_ / xu_completed_seqs_ / host_completion_state_,
  // all written only on the runtime-worker thread (also the only caller of this
  // method), so those reads need no extra synchronization.
  bool IsScheduleGatedByPriorOp(OperationContext* op, KernelTypeEnum kernel_type,
                                bool prior_ops_in_flight, bool refresh_completion);
  // True iff `op` is a same-stream kWait eligible for run-ahead bypass; the
  // free-HBM headroom check lives in IsRunAheadMemorySafe.
  static bool IsSameStreamWaitBypass(OperationContext* op, c10::StreamId stream_id);
  // True iff free device HBM has enough headroom to safely advance past a
  // same-stream kWait. Disabled => always returns true.
  static bool IsRunAheadMemorySafe(c10::DeviceIndex device_index);
  // Advance next_to_schedule_ past the op the caller just scheduled.
  void AdvanceToNextOperation(OperationContext* just_scheduled = nullptr);

  // Check if stream has pending async operations (scheduled or pending signal)
  bool HasPendingAsyncOperations() const;

  // Park `alloc_kernel` as Deferred iff the stream has drainable work.
  // Returns false when there's nothing whose completion would unblock the
  // retry — caller must escalate. Single-mutex-acquisition closes the
  // drain-check vs state-flip race.
  bool TryDeferAlloc(AllocDirectKernelExecution& alloc_kernel);

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
  OperationContext* FindOperationBySeq(nrt::SequenceId sequence_id);

  void RemoveActiveOperation(const OperationContext* operation);

 private:
  // Prepare operation on the calling thread if RequiresPrepare() is true.
  void PrepareOperation(OperationContext* op);

  // True iff no scheduled / pending-signal ops remain (Deferred kAllocs are
  // ignored). Caller must hold runtime_queue_mutex_.
  bool IsStreamDrainedWithLockHeld() const;

  // True iff the front-of-queue op is in flight (scheduled / pending-signal /
  // host-inflight, or its concat op scheduled). Lock-held analogue of
  // HasPendingAsyncOperations() for callers that already hold
  // runtime_queue_mutex_ (e.g. GetNextOperationToSchedule's gate pre-check),
  // avoiding the self-deadlock of re-locking. Caller must hold the mutex.
  bool FrontOpInFlightWithLockHeld() const;

  // Flip parked kAllocs from Deferred to RetryReady when the stream is
  // drained. Caller must hold runtime_queue_mutex_.
  void ReleaseDeferredAllocGates();

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

  // Committed FIFO scheduled by the engine.
  std::list<std::shared_ptr<OperationContext>> runtime_queue_;
  mutable std::mutex runtime_queue_mutex_;
  // Iterator into runtime_queue_; equals end() while empty.
  std::list<std::shared_ptr<OperationContext>>::iterator next_to_schedule_;
};

// Resolve the persistent cache key (under which the NEFF lives on disk) for a
// profiler event. Kernels carrying HLO bytes (eager) recompute it directly;
// kernels without (torch.compile execution, NeffDirectKernelExecution) — whose
// CompilationCacheKey lacks the bytes/config to recompute it correctly — read
// it from the CacheEntry the compile step stored. Returns "" if unresolvable
// (e.g. the cache entry was evicted).
std::string ResolvePersistentCacheKey(const CompilableKernelExecution& kernel,
                                      CompilationCache& compilation_cache);

}  // namespace at::neuron
