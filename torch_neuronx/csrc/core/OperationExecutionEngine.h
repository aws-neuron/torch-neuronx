#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <deque>
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

// Global execution engine that manages compilation and execution threads across all streams
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

  void SubmitCompilationTask(OperationContext* operation);
  // Submit operation to concatenation queue (if enabled) or directly to compilation
  void SubmitOperation(OperationContext* operation);
  // Notify the execution worker that an operation is ready
  void NotifyExecutionReady() { WriteEventfd(); }

  // Execution queue depth management
  int64_t GetExecutionQueueDepth() const { return execution_queue_depth_.load(); }
  int64_t GetQueueLowThreshold() const { return queue_low_threshold_; }
  void DecrementExecutionQueueDepth(OperationContext* operation);

  // Check if concatenation is enabled via environment variable
  bool IsConcatenationEnabled() const;

  // Handle concatenation failure - called by StreamImpl during error handling
  void HandleConcatenationFailure(OperationContext* operation);

  // Submit a hint to the concatenation queue, bypassing active_operations_ and compilation.
  void RecordHint(std::unique_ptr<OperationContext> hint, const NeuronStream& stream);

 private:
  // Worker threads
  std::thread concatenation_thread_;
  std::thread runtime_thread_;

  // Concatenation infrastructure (only initialized if flag is enabled)
  std::unique_ptr<ConcatenationEngine> concatenation_engine_;
  std::deque<OperationContext*> concatenation_queue_;
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

  std::atomic<int64_t> execution_queue_depth_{0};

  // Track pending async operations count to avoid expensive iteration
  std::atomic<int> pending_async_ops_count_{0};

  // Per-device per-XU count of inflight operations.
  int xu_inflight_counts_[kMaxDevices][NRTA_XU_TYPE_NUM]{};

  // Configurable max inflight per XU
  size_t max_inflight_per_xu_{kDefaultMaxInflightPerXU};

  // Guards concatenation_queue_ and hint_ownership_map_.
  mutable std::mutex concatenation_mutex_;

  // Owns hint unique_ptrs while the ConcatenationEngine holds their raw pointers.
  // Keyed by raw pointer so we can free the correct hint when it's emitted back
  // (hints may be emitted out of insertion order due to buffering in the engine).
  std::unordered_map<OperationContext*, std::unique_ptr<OperationContext>> hint_ownership_map_;
};

}  // namespace at::neuron
