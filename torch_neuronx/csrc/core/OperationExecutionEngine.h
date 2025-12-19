#pragma once

#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron {

// Forward declarations
class CompilationCache;
class ModelHandleCache;

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
                           ModelHandleCache* model_handle_cache);
  ~OperationExecutionEngine();

  // Deleted copy/move constructors and assignment operators
  OperationExecutionEngine(const OperationExecutionEngine&) = delete;
  OperationExecutionEngine& operator=(const OperationExecutionEngine&) = delete;
  OperationExecutionEngine(OperationExecutionEngine&&) = delete;
  OperationExecutionEngine& operator=(OperationExecutionEngine&&) = delete;

  void SubmitCompilationTask(OperationContext* operation);
  // Notify the execution worker that an operation is ready
  void NotifyExecutionReady();

 private:
  // Worker threads
  std::thread compilation_thread_;
  std::thread execution_thread_;

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

  // Worker thread implementations
  void CompilationWorker();
  void ExecutionWorker();

  // Preprocessing helpers prologue attached with operations if needed
  bool ProcessPreprocessingTask(OperationContext* operation) noexcept;

 protected:
  // Virtual for testing - allows mocking execution without real device operations
  virtual void ProcessExecutionTask(OperationContext* operation) noexcept;
  virtual void ProcessCompilationTask(OperationContext* operation) noexcept;
};

}  // namespace at::neuron
