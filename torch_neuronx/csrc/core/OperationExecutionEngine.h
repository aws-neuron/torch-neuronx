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
  void NotifyExecutionReady();

  // Execution queue depth management
  int64_t GetExecutionQueueDepth() const { return execution_queue_depth_.load(); }
  int64_t GetQueueLowThreshold() const { return queue_low_threshold_; }

  // Check if concatenation is enabled via environment variable
  bool IsConcatenationEnabled() const;

  // Check if async NRT mode is enabled
  bool IsNRTAsyncModeEnabled() const { return nrt_async_mode_enabled_; }

  // Set async NRT mode (for testing purposes)
  void SetAsyncModeEnabled(bool enabled) { nrt_async_mode_enabled_ = enabled; }

  // Handle concatenation failure - called by StreamImpl during error handling
  void HandleConcatenationFailure(OperationContext* operation);

 private:
  // Worker threads
  std::thread concatenation_thread_;
  std::thread compilation_thread_;
  std::thread execution_thread_;

  // Concatenation infrastructure (only initialized if flag is enabled)
  std::unique_ptr<ConcatenationEngine> concatenation_engine_;
  std::deque<OperationContext*> concatenation_queue_;
  mutable std::mutex concatenation_mutex_;
  std::condition_variable concatenation_cv_;
  bool concatenation_enabled_{false};

  std::deque<OperationContext*> compilation_queue_;

  mutable std::mutex compilation_mutex_;
  std::condition_variable compilation_cv_;

  mutable std::mutex execution_mutex_;
  std::condition_variable execution_cv_;
  bool execution_pending_{false};

  // Shutdown coordination
  std::atomic<bool> shutdown_requested_{false};

  bool nrt_async_mode_enabled_{false};

  // Shared resources (accessed through resource manager)
  CompilationCache* compilation_cache_{nullptr};
  ModelHandleCache* model_handle_cache_{nullptr};

  // Execution queue depth tracking for dynamic buffering
  int64_t queue_low_threshold_;

  // Worker thread implementations
  void ConcatenationWorker();
  void CompilationWorker();
  void ExecutionWorker();
  void AsyncExecutionWorker();

  // Async execution helpers
  bool HasPendingAsyncOperations() const { return pending_async_ops_count_.load() > 0; }

  // Check and process completions for all devices
  void CheckCompletions() noexcept;

  // Process a single async execution task (schedule without waiting)
  void DispatchAsyncExecutionTask(OperationContext* operation) noexcept;

  // Async scheduling helpers
  bool ScheduleDeviceOperation(OperationContext* op, StreamImpl* stream,
                               KernelTypeEnum kernel_type);
  bool ScheduleHostOperation(OperationContext* op, StreamImpl* stream);
  void ProcessAsyncErrors(StreamImpl* stream, c10::DeviceIndex device_idx);

  // Preprocessing helpers prologue attached with operations if needed
  void ProcessPreprocessingTask(OperationContext* operation);

  // Execute operation, handling concatenation logic internally
  void ExecuteOperation(OperationContext* operation);

  // Handle execution when operation has a concatenated operation
  void ExecuteConcatenatedOperation(OperationContext* operation);

  // Handle execution failure for concatenated operations
  // Returns true if this was a concatenated op failure that requires retry
  bool HandleConcatenatedExecutionFailure(OperationContext* operation,
                                          const std::string& error_msg);

 protected:
  void ProcessPrepareTask(OperationContext* operation) noexcept;
  virtual void ProcessCompilationTask(OperationContext* operation);
  virtual void ProcessExecutionTask(OperationContext* operation) noexcept;
  void ProcessCompletions(StreamImpl* stream, const nrt::CompletionState& completion_state);
  void DecrementExecutionQueueDepth();
  std::atomic<int64_t> execution_queue_depth_{0};

  // Track pending async operations count to avoid expensive iteration
  std::atomic<int> pending_async_ops_count_{0};
};

}  // namespace at::neuron
