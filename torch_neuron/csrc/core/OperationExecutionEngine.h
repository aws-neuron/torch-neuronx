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

#include "torch_neuron/csrc/core/OperationContext.h"

namespace at::neuron {

// Forward declarations
class StreamImpl;
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
  std::deque<T> high_;
  std::deque<T> default_;
  std::deque<T> low_;
};

enum class TaskType {
  COMPILATION,  // Task requires compilation (and subsequent execution)
  EXECUTION     // Task only requires execution (no compilation needed)
};

// Unified task structure for both compilation and execution operations
struct OperationExecutionTask {
  OperationContext* operation;
  StreamImpl* stream;
  TaskPriority priority;
  TaskType type;

  OperationExecutionTask(OperationContext* op, StreamImpl* stream, int prio, TaskType task_type)
      : operation(op), stream(stream), priority(BucketizePriority(prio)), type(task_type) {}

  bool RequiresCompilation() const { return operation->RequiresCompilation(); }
};

// Global execution engine that manages compilation and execution threads across all streams
class OperationExecutionEngine {
 public:
  void SubmitTask(OperationExecutionTask task);

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

 private:
  // Worker threads
  std::thread compilation_thread_;
  std::thread execution_thread_;

  // Priority-based task FIFO queues
  PriorityQueue<OperationExecutionTask> compilation_queue_;
  PriorityQueue<OperationExecutionTask> execution_queue_;

  // Deferred execution tasks waiting on event dependencies
  std::deque<OperationExecutionTask> deferred_execution_tasks_;

  // Synchronization primitives
  std::mutex compilation_mutex_;
  std::mutex execution_mutex_;
  std::condition_variable compilation_cv_;
  std::condition_variable execution_cv_;

  // Shutdown coordination
  std::atomic<bool> shutdown_requested_{false};

  // Shared resources (accessed through resource manager)
  CompilationCache* compilation_cache_{nullptr};
  ModelHandleCache* model_handle_cache_{nullptr};

  // Performance monitoring metrics
  std::atomic<size_t> total_compilation_tasks_submitted_{0};
  std::atomic<size_t> total_compilation_tasks_completed_{0};
  std::atomic<size_t> total_execution_tasks_submitted_{0};
  std::atomic<size_t> total_execution_tasks_completed_{0};
  std::chrono::steady_clock::time_point creation_time_;

  // Worker thread implementations
  void CompilationWorker();
  void ExecutionWorker();

  // Submission tasks
  void SubmitCompilationTask(OperationExecutionTask task);
  void SubmitExecutionTask(OperationExecutionTask task);

  // Helper methods
  bool ProcessCompilationTask(OperationExecutionTask& task) noexcept;
  bool AreEventDependenciesSatisfied(const OperationContext* op) const noexcept;
  void PromoteReadyDeferredTasks() noexcept;

 protected:
  // Virtual for testing - allows mocking execution without real device operations
  virtual void ProcessExecutionTask(OperationExecutionTask& task) noexcept;

 public:
  // Performance monitoring methods
  struct GlobalPerformanceStats {
    size_t compilation_tasks_submitted;
    size_t compilation_tasks_completed;
    size_t execution_tasks_submitted;
    size_t execution_tasks_completed;
    size_t compilation_queue_current_size;
    size_t execution_queue_current_size;
    size_t compilation_queue_peak_size;
    size_t execution_queue_peak_size;
    size_t priority_inversions_detected;
    std::chrono::seconds uptime;
    bool is_active;
  };

  GlobalPerformanceStats GetPerformanceStats() const;
  // Simple statistics for NeuronResourceManager
  struct Statistics {
    size_t queue_size;
    bool worker_active;
  };
  Statistics GetStatistics() const;
};

}  // namespace at::neuron
