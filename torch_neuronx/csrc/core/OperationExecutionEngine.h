#pragma once

#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationEngine.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
namespace at::neuron {

// Forward declarations
class CompilationCache;
class ModelHandleCache;
class ConcatenationEngine;

// Priority levels for task execution
enum class TaskPriority { HIGH = 0, DEFAULT = 1, LOW = 2 };

inline TaskPriority BucketizePriority(int priority) {
  if (priority < 0) {
    return TaskPriority::HIGH;
  } else if (priority > 0) {
    return TaskPriority::LOW;
  }
  return TaskPriority::DEFAULT;
}

// Priority-based queue that maintains bucketed queues of priority levels
// Tasks are processed in priority order, with FIFO ordering within each priority level
template <typename T>
class PriorityQueue {
 public:
  // Check if any tasks are queued
  bool empty() const;

  // Get total number of tasks across all priority levels
  size_t size() const;

  // Add task to the back of the appropriate priority queue
  void push_back(T task);

  // Add task to the front of the appropriate priority queue
  // Used when promoting deferred tasks to preserve submission order
  void push_front(T task);

  // Pop and return the highest priority task
  // Caller must ensure !empty() before calling
  T pop_front();

 private:
  // Select the respective queue based on priority
  std::deque<T>& GetQueueForPriority(int priority);

  std::deque<T> high_;
  std::deque<T> default_;
  std::deque<T> low_;
};

// Global execution engine that manages compilation and execution threads across all streams
class OperationExecutionEngine {
 public:
  // Shutdown coordination
  void Shutdown();

  // Constructor and destructor (now public since managed by NeuronResourceManager)
  OperationExecutionEngine(CompilationCache* compilation_cache,
                           ModelHandleCache* model_handle_cache, int64_t queue_low_threshold = 10,
                           int64_t queue_target_depth = 50);
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
  int64_t GetQueueTargetDepth() const { return queue_target_depth_; }

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

  PriorityQueue<OperationContext*> compilation_queue_;

  mutable std::mutex compilation_mutex_;
  std::condition_variable compilation_cv_;

  mutable std::mutex execution_mutex_;
  std::condition_variable execution_cv_;
  bool execution_pending_{false};

  // Shutdown coordination
  std::atomic<bool> shutdown_requested_{false};

  // Shared resources (accessed through resource manager)
  CompilationCache* compilation_cache_{nullptr};
  ModelHandleCache* model_handle_cache_{nullptr};

  // Execution queue depth tracking for dynamic buffering
  std::atomic<int64_t> execution_queue_depth_{0};
  int64_t queue_low_threshold_;
  int64_t queue_target_depth_;

  // Worker thread implementations
  void ConcatenationWorker();
  void CompilationWorker();
  void ExecutionWorker();

  // Check if concatenation is enabled via environment variable
  bool IsConcatenationEnabled() const;

  // Preprocessing helpers prologue attached with operations if needed
  bool ProcessPreprocessingTask(OperationContext* operation) noexcept;

 protected:
  // Virtual for testing - allows mocking execution without real device operations
  virtual void ProcessExecutionTask(OperationContext* operation) noexcept;
  virtual void ProcessCompilationTask(OperationContext* operation) noexcept;
};

}  // namespace at::neuron
