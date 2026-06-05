#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
namespace at::neuron {

// Forward declarations
class CompilationCache;
class ModelHandleCache;
class ConcatenationEngine;

// Default queue depth configuration for concatenation
constexpr int64_t DEFAULT_QUEUE_LOW_THRESHOLD = 15;

// Default max inflight requests per XU (hardware queue depth limit)
constexpr int kDefaultMaxInflightPerXU = 8;

// Two-queue topology:
//   - dispatch_queue_   : global mutable buffer; ConcatenationWorker reorders
//                         and merges, then transfers ops to runtime_queue_.
//                         Concat-OFF submits skip this entirely.
//   - runtime_queue_    : per-stream committed FIFO on StreamImpl;
//                         RuntimeWorker schedules from here.
//
// Storage invariant: at every observable instant, an op is in exactly one
// of dispatch_queue_ or stream->runtime_queue_. ConcatenationEngine's
// internal buffer is raw-pointer aliases into a contiguous prefix of
// dispatch_queue_ (ownership stays in dispatch_queue_ until transfer).
class OperationExecutionEngine {
 public:
  // Shutdown coordination
  void Shutdown();

  // Constructor and destructor (now public since managed by NeuronResourceManager)
  OperationExecutionEngine(CompilationCache* compilation_cache,
                           ModelHandleCache* model_handle_cache);
  ~OperationExecutionEngine();

  // Deleted copy/move constructors and assignment operators
  OperationExecutionEngine(const OperationExecutionEngine&) = delete;
  OperationExecutionEngine& operator=(const OperationExecutionEngine&) = delete;
  OperationExecutionEngine(OperationExecutionEngine&&) = delete;
  OperationExecutionEngine& operator=(OperationExecutionEngine&&) = delete;

  // Single submit entry point. Routes internally based on whether
  // concatenation is enabled:
  //   - concat ON  → push onto dispatch_queue_ and wake the
  //                  ConcatenationWorker, which reorders the buffered
  //                  window and transfers a partitioned prefix into
  //                  op->stream->runtime_queue_ at flush.
  //   - concat OFF → park the shared_ptr directly in
  //                  op->stream->runtime_queue_, no worker hop.
  void SubmitOperation(std::shared_ptr<OperationContext> op);

  // Push a host-side ALLOC/DEALLOC hint through dispatch_queue_. Hints are
  // consumed by the concat engine and dropped at transfer time — they never
  // reach runtime_queue_.
  void RecordHint(std::unique_ptr<OperationContext> hint, const NeuronStream& stream);

  void SubmitCompilationTask(OperationContext* operation);

  // Notify the runtime worker that an operation is ready
  void NotifyExecutionReady() { WriteEventfd(); }

  // Execution queue depth management
  int64_t GetExecutionQueueDepth() const { return execution_queue_depth_.load(); }
  int64_t GetQueueLowThreshold() const { return queue_low_threshold_; }
  void DecrementExecutionQueueDepth(OperationContext* operation);

  // Check if concatenation is enabled via environment variable
  bool IsConcatenationEnabled() const;

  // Handle concatenation failure - called by StreamImpl during error handling
  void HandleConcatenationFailure(OperationContext* operation);

  // True when both dispatch_queue_ and stream->runtime_queue_ are empty for
  // `stream`. Used by Synchronize / Query.
  bool IsStreamDrained(const StreamImpl* stream) const;

 private:
  // Worker threads
  std::thread concatenation_thread_;
  std::thread runtime_thread_;

  // Concatenation infrastructure (only initialized if flag is enabled)
  std::unique_ptr<ConcatenationEngine> concatenation_engine_;
  std::condition_variable concatenation_cv_;
  bool concatenation_enabled_{false};

  std::deque<OperationContext*> compilation_queue_;
  mutable std::mutex runtime_mutex_;

  int completion_efd_{-1};

  // Shutdown coordination
  std::atomic<bool> shutdown_requested_{false};

  // Shared resources (accessed through resource manager)
  CompilationCache* compilation_cache_{nullptr};
  ModelHandleCache* model_handle_cache_{nullptr};

  // Execution queue depth tracking for dynamic buffering
  int64_t queue_low_threshold_;

  // Worker thread implementations
  void ConcatenationWorker();
  void RuntimeWorker();
  bool CompileNextPending();

  // Hand a single shared_ptr off to op->stream->runtime_queue_. Sets the
  // cursor to the inserted op when the queue was empty. Takes the stream's
  // runtime_queue_mutex_ internally.
  void EmitToRuntimeQueue(std::shared_ptr<OperationContext> operation_context);

  // Async execution helpers
  bool HasPendingAsyncOperations() const { return pending_async_ops_count_.load() > 0; }

  // Check and process completions for all devices
  void CheckCompletions() noexcept;

  // Register eventfd on front-of-queue sequences
  void RegisterCompletionEvents();
  void RegisterSeqCompletion(StreamImpl* stream, nrt::SequenceId seq);

  // Write to completion_efd_ with EINTR retry
  void WriteEventfd();

  // Process a single async execution task (schedule without waiting)
  void DispatchAsyncExecutionTask(OperationContext* operation) noexcept;

  // Preprocessing helpers prologue attached with operations if needed
  void ProcessPreprocessingTask(OperationContext* operation);

  // Shared completion logic for both sync and async compilation paths.
  void HandleCompilationComplete(OperationContext* operation);

  // Handle execution failure for concatenated operations
  // Returns true if this was a concatenated op failure that requires retry
  bool HandleConcatenatedExecutionFailure(OperationContext* operation,
                                          const std::string& error_msg);

 protected:
  void ProcessPrepareTask(OperationContext* operation) noexcept;
  virtual void ProcessCompilationTask(OperationContext* operation);
  void ProcessCompletions(StreamImpl* stream, const XUCompletionState& completed_seqs);
  bool ScheduleOperation(OperationContext* op, StreamImpl* stream);
  bool ScheduleAllocOperation(OperationContext* op, StreamImpl* stream);

  // Move ops onto each target stream->runtime_queue_ in the order of
  // `partitioned_raws`. `owners` maps each raw pointer to its shared_ptr;
  // kHint entries are dropped instead of transferred.
  void MoveToRuntimeQueueInOrder(
      const std::list<OperationContext*>& partitioned_raws,
      std::unordered_map<OperationContext*, std::shared_ptr<OperationContext>>& owners);

  // Test-only helpers. Routed through the engine's friendship with StreamImpl.
  void TestInjectIntoRuntimeQueue(StreamImpl* stream, std::shared_ptr<OperationContext> op);
  void TestClearRuntimeQueue(StreamImpl* stream);
  size_t TestRuntimeQueueSize(const StreamImpl* stream) const;
  bool TestRuntimeCursorAtEnd(const StreamImpl* stream) const;
  bool TestRuntimeCursorAt(const StreamImpl* stream, const OperationContext* op) const;
  bool TestSetRuntimeCursorTo(StreamImpl* stream, const OperationContext* op);
  OperationContext* TestRuntimeQueueFront(const StreamImpl* stream) const;
  std::vector<OperationContext*> TestRuntimeQueueSnapshot(const StreamImpl* stream) const;

  // Global, mutable. ConcatenationWorker drains in submission order.
  // Sole owner during dispatch stage — engine's operation_buffer_ holds raw
  // pointer aliases into a contiguous prefix of this list.
  std::list<std::shared_ptr<OperationContext>> dispatch_queue_;
  mutable std::mutex dispatch_mutex_;

  std::atomic<int64_t> execution_queue_depth_{0};

  // Track pending async operations count to avoid expensive iteration
  std::atomic<int> pending_async_ops_count_{0};

  // Per-device per-XU count of inflight operations.
  int xu_inflight_counts_[kMaxDevices][NRTA_XU_TYPE_NUM]{};

  // Per-device per-XU slot bitmask for profiler dispatch slot assignment.
  // Each bit represents an inflight slot (0 = free, 1 = occupied).
  uint8_t xu_slot_bitmask_[kMaxDevices][NRTA_XU_TYPE_NUM]{};

  // Configurable max inflight per XU
  size_t max_inflight_per_xu_{kDefaultMaxInflightPerXU};
};

}  // namespace at::neuron
