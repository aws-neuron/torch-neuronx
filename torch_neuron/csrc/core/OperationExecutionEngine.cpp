#include "OperationExecutionEngine.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "torch_neuron/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuron/csrc/core/NeuronDevice.h"
#include "torch_neuron/csrc/core/NeuronLogging.h"
#include "torch_neuron/csrc/core/NeuronOpTracking.h"
#include "torch_neuron/csrc/core/compilation/CompilationCache.h"
#include "torch_neuron/csrc/core/runtime/NRTHandler.h"
#include "torch_neuron/csrc/core/streams/StreamImpl.h"

namespace at::neuron {

namespace {

bool IsNeuronDevice(const c10::Device& device) {
  return device.type() == c10::DeviceType::PrivateUse1;
}

// Helper function to extract device ID from a compilable kernel's inputs and outputs.
// Iterates through all tensors to ensure they're on the same device and returns that device ID.
int GetTargetDeviceId(const NeuronKernelExecution& kernel_execution) {
  int device_id = -1;
  bool device_found = false;

  // Check all input tensors
  const auto& inputs = kernel_execution.GetInputs();
  for (const auto& tensor : inputs) {
    TORCH_CHECK(IsNeuronDevice(tensor.device()), "Input tensor is not on Neuron device");
    int current_device = tensor.device().index();
    if (!device_found) {
      device_id = current_device;
      device_found = true;
    } else if (device_id != current_device) {
      throw std::runtime_error("Inconsistent device IDs found in kernel inputs: expected device " +
                               std::to_string(device_id) + " but found device " +
                               std::to_string(current_device));
    }
  }

  // Check all output tensors
  const auto& outputs = kernel_execution.GetOutputs();
  for (const auto& tensor : outputs) {
    TORCH_CHECK(IsNeuronDevice(tensor.device()), "Output tensor is not on Neuron device");
    int current_device = tensor.device().index();
    if (!device_found) {
      device_id = current_device;
      device_found = true;
    } else if (device_id != current_device) {
      throw std::runtime_error("Inconsistent device IDs found in kernel outputs: expected device " +
                               std::to_string(device_id) + " but found device " +
                               std::to_string(current_device));
    }
  }

  TORCH_CHECK(device_found, "No device found in kernel inputs and outputs");
  return device_id;
}

}  // anonymous namespace

template <typename T>
bool PriorityQueue<T>::empty() const {
  return high_.empty() && default_.empty() && low_.empty();
}

template <typename T>
size_t PriorityQueue<T>::size() const {
  return high_.size() + default_.size() + low_.size();
}

template <typename T>
void PriorityQueue<T>::push_back(T task) {
  switch (task.priority) {
    case TaskPriority::HIGH:
      high_.push_back(std::move(task));
      break;
    case TaskPriority::DEFAULT:
      default_.push_back(std::move(task));
      break;
    case TaskPriority::LOW:
      low_.push_back(std::move(task));
      break;
  }
}

template <typename T>
void PriorityQueue<T>::push_front(T task) {
  switch (task.priority) {
    case TaskPriority::HIGH:
      high_.push_front(std::move(task));
      break;
    case TaskPriority::DEFAULT:
      default_.push_front(std::move(task));
      break;
    case TaskPriority::LOW:
      low_.push_front(std::move(task));
      break;
  }
}

template <typename T>
T PriorityQueue<T>::pop_front() {
  if (!high_.empty()) {
    auto task = std::move(high_.front());
    high_.pop_front();
    return task;
  }
  if (!default_.empty()) {
    auto task = std::move(default_.front());
    default_.pop_front();
    return task;
  }
  auto task = std::move(low_.front());
  low_.pop_front();
  return task;
}

// Explicit template instantiation for OperationExecutionTask
template class PriorityQueue<OperationExecutionTask>;

OperationExecutionEngine::OperationExecutionEngine(CompilationCache* compilation_cache,
                                                   ModelHandleCache* model_handle_cache)
    : compilation_cache_(compilation_cache),
      model_handle_cache_(model_handle_cache),
      creation_time_(std::chrono::steady_clock::now()) {
  // Start worker threads
  compilation_thread_ = std::thread(&OperationExecutionEngine::CompilationWorker, this);
  execution_thread_ = std::thread(&OperationExecutionEngine::ExecutionWorker, this);

  TORCH_NEURON_DEBUG("OperationExecutionEngine worker threads started",
                     "compilation_thread_id=", compilation_thread_.get_id(),
                     "execution_thread_id=", execution_thread_.get_id());
}

OperationExecutionEngine::~OperationExecutionEngine() { Shutdown(); }

void OperationExecutionEngine::SubmitTask(OperationExecutionTask task) {
  // Route task to the appropriate thread based on task type
  if (task.type == TaskType::COMPILATION) {
    SubmitCompilationTask(std::move(task));
  } else if (task.type == TaskType::EXECUTION) {
    SubmitExecutionTask(std::move(task));
  } else {
    TORCH_CHECK(false, "Unknown task type: ", task.type);
  }
}

void OperationExecutionEngine::Shutdown() {
  TORCH_NEURON_DEBUG("Shutting down OperationExecutionEngine");

  // Signal shutdown to worker threads
  shutdown_requested_.store(true, std::memory_order_release);

  // Wake up all worker threads
  compilation_cv_.notify_all();
  execution_cv_.notify_all();

  if (compilation_thread_.joinable()) {
    compilation_thread_.join();
  }
  if (execution_thread_.joinable()) {
    execution_thread_.join();
  }
  TORCH_NEURON_DEBUG("OperationExecutionEngine shutdown complete");
}

void OperationExecutionEngine::CompilationWorker() {
  TORCH_NEURON_DEBUG("Compilation worker thread started");

  while (!shutdown_requested_.load(std::memory_order_acquire)) {
    // Wait for a task or shutdown signal
    std::unique_lock<std::mutex> lock(compilation_mutex_);
    compilation_cv_.wait(lock, [this] {
      return !compilation_queue_.empty() || shutdown_requested_.load(std::memory_order_acquire);
    });

    if (shutdown_requested_.load(std::memory_order_acquire) && compilation_queue_.empty()) {
      break;
    }

    if (!compilation_queue_.empty()) {
      OperationExecutionTask task = compilation_queue_.pop_front();
      lock.unlock();

      if (!task.RequiresCompilation() || ProcessCompilationTask(task)) {
        SubmitExecutionTask(std::move(task));
      }
    }
  }
  TORCH_NEURON_DEBUG("Compilation worker thread terminated");
}

bool OperationExecutionEngine::ProcessCompilationTask(OperationExecutionTask& task) noexcept {
  auto* stream = task.stream;
  auto* op = task.operation;
  try {
    // Set compilation start time
    // TODO(rpsilva): Introduce a global metric collection
    op->compile_start = std::chrono::steady_clock::now();

    // Get the compilable kernel
    auto& compilable_kernel = op->GetCompilableKernel();

    // Compile NEFF through shared compilation cache
    compilation_cache_->GetOrCompileNeff(compilable_kernel, stream->stream_id);

    if (!compilable_kernel.HasCachedNeff()) {
      throw std::runtime_error("NEFF creation failed for operation: " + op->GetOpName());
    }

    // Set compilation end time
    op->compile_end = std::chrono::steady_clock::now();

    TORCH_NEURON_DEBUG("Compilation task completed successfully", "op=", op->GetOpName(),
                       "compilation_time_us=", op->GetCompilationTime().count());
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(op, e.what(), ErrorContext::Stage::COMPILATION);
    return false;
  }
  total_compilation_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void OperationExecutionEngine::ExecutionWorker() {
  TORCH_NEURON_DEBUG("Execution worker thread started");

  while (!shutdown_requested_.load(std::memory_order_acquire)) {
    // Wait for a task or shutdown signal
    std::unique_lock<std::mutex> lock(execution_mutex_);
    execution_cv_.wait(lock, [this] {
      return !execution_queue_.empty() || shutdown_requested_.load(std::memory_order_acquire);
    });

    if (shutdown_requested_.load(std::memory_order_acquire) && execution_queue_.empty()) {
      break;
    }

    // Process a task from the main execution queue if available
    if (!execution_queue_.empty()) {
      OperationExecutionTask task = execution_queue_.pop_front();

      // Check if event dependencies are satisfied before processing
      // If not, defer the task to avoid blocking other streams
      if (!AreEventDependenciesSatisfied(task.operation)) {
        TORCH_NEURON_DEBUG("Event dependencies not yet satisfied, deferring task",
                           "stream_id=", task.stream->stream_id, "op=", task.operation->GetOpName(),
                           "num_events=", task.operation->events_to_wait_for.size());
        deferred_execution_tasks_.push_back(std::move(task));
        continue;
      } else {
        lock.unlock();
        ProcessExecutionTask(task);
        PromoteReadyDeferredTasks();
      }
    }
  }

  TORCH_NEURON_DEBUG("Execution worker thread terminated");
}

void OperationExecutionEngine::ProcessExecutionTask(OperationExecutionTask& task) noexcept {
  auto* stream = task.stream;
  auto* op = task.operation;
  const std::string op_name = op->GetOpName();
  try {
    TORCH_NEURON_DEBUG("Processing execution task", "stream_id=", stream->stream_id,
                       "op=", op_name);

    // Set execution start time
    // TODO(rpsilva): Introduce a global metric collection
    op->execute_start = std::chrono::steady_clock::now();
    bool is_executable_neff = op->RequiresCompilation();
    if (is_executable_neff) {
      uint32_t stream_id = stream->stream_id;

      // Execute compilable operation inline
      TORCH_NEURON_DEBUG("Executing compilable operation", "stream_id=", stream_id,
                         "op=", op->GetOpName());

      TORCH_NEURON_DEBUG("Preparing compilable operation for NRT execution",
                         "stream_id=", stream_id, "op=", op->GetOpName());

      // Get the compilable kernel and operation details
      const auto& compilable_kernel = op->GetCompilableKernel();

      // Configure NRT execution
      int device_id = GetTargetDeviceId(compilable_kernel);
      NRTHandler::ExecutionConfig config(device_id, 1 /* num_cores */);

      // Execute using static method
      auto result = NRTHandler::ExecuteCompilableKernel(compilable_kernel, op_name, config);

      if (!result.success) {
        TORCH_NEURON_ERROR("NRT execution failed", "stream_id=", stream_id, "op=", op_name,
                           "error=", result.error_message);
        throw std::runtime_error("NRT execution failed: " + result.error_message);
      }
      TORCH_NEURON_DEBUG("NRT execution completed", "stream_id=", stream_id, "op=", op_name,
                         "execution_time_ms=", result.execution_time.count());
    } else {
      throw std::logic_error("Direct kernel execution is not yet implemented for operation: " +
                             op->GetOpName());
    }

    if (torch_neuron::shouldLogExecuted(op_name)) {
      torch_neuron::NeuronLogger::getInstance().log(
          torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::OPERATOR_EXECUTED,
          "Operator '" + op_name + "' executed on Neuron");
      torch_neuron::markExecutedLogged(op_name);
    }
    stream->CompleteOperation(op);
    TORCH_NEURON_DEBUG("Execution task completed successfully", "op=", op_name);
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(op, e.what(), ErrorContext::Stage::EXECUTION);
  }
  total_execution_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
}

bool OperationExecutionEngine::AreEventDependenciesSatisfied(
    const OperationContext* op) const noexcept {
  if (op->events_to_wait_for.empty()) {
    return true;
  }

  for (const NeuronEvent* event : op->events_to_wait_for) {
    if (event && !event->query()) {
      return false;
    }
  }

  return true;
}

void OperationExecutionEngine::PromoteReadyDeferredTasks() noexcept {
  std::lock_guard<std::mutex> lock(execution_mutex_);

  // Collect ready tasks in order
  std::vector<OperationExecutionTask> ready_tasks;
  for (auto it = deferred_execution_tasks_.begin(); it != deferred_execution_tasks_.end();) {
    if (AreEventDependenciesSatisfied(it->operation)) {
      ready_tasks.push_back(std::move(*it));
      it = deferred_execution_tasks_.erase(it);
    } else {
      ++it;
    }
  }

  // Push to front in reverse order to maintain submission order
  for (auto it = ready_tasks.rbegin(); it != ready_tasks.rend(); ++it) {
    execution_queue_.push_front(std::move(*it));
  }
}

void OperationExecutionEngine::SubmitCompilationTask(OperationExecutionTask task) {
  if (shutdown_requested_.load()) {
    TORCH_NEURON_DEBUG("Rejecting compilation task submission during shutdown",
                       "op=", task.operation->GetOpName());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(compilation_mutex_);
    compilation_queue_.push_back(std::move(task));

    // Update metrics atomically
    total_compilation_tasks_submitted_.fetch_add(1, std::memory_order_relaxed);
  }
  compilation_cv_.notify_one();
}

void OperationExecutionEngine::SubmitExecutionTask(OperationExecutionTask task) {
  if (shutdown_requested_.load()) {
    TORCH_NEURON_DEBUG("Rejecting execution task submission during shutdown",
                       "op=", task.operation->GetOpName());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    execution_queue_.push_back(std::move(task));

    // Update metrics atomically
    total_execution_tasks_submitted_.fetch_add(1, std::memory_order_relaxed);
  }
  execution_cv_.notify_one();
}

OperationExecutionEngine::GlobalPerformanceStats OperationExecutionEngine::GetPerformanceStats()
    const {
  GlobalPerformanceStats stats;

  stats.compilation_tasks_submitted = total_compilation_tasks_submitted_.load();
  stats.compilation_tasks_completed = total_compilation_tasks_completed_.load();
  stats.execution_tasks_submitted = total_execution_tasks_submitted_.load();
  stats.execution_tasks_completed = total_execution_tasks_completed_.load();

  // Get current queue sizes (requires locking)
  {
    std::lock_guard<std::mutex> comp_lock(const_cast<std::mutex&>(compilation_mutex_));
    stats.compilation_queue_current_size = compilation_queue_.size();
  }
  {
    std::lock_guard<std::mutex> exec_lock(const_cast<std::mutex&>(execution_mutex_));
    stats.execution_queue_current_size = execution_queue_.size();
  }

  // Calculate uptime
  // TODO(rpsilva): Introduce a global metric collection
  auto now = std::chrono::steady_clock::now();
  stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - creation_time_);

  stats.is_active = !shutdown_requested_.load();

  return stats;
}

OperationExecutionEngine::Statistics OperationExecutionEngine::GetStatistics() const {
  Statistics stats;

  // Get total queue size (compilation + execution)
  {
    std::lock_guard<std::mutex> comp_lock(const_cast<std::mutex&>(compilation_mutex_));
    std::lock_guard<std::mutex> exec_lock(const_cast<std::mutex&>(execution_mutex_));
    stats.queue_size = compilation_queue_.size() + execution_queue_.size();
  }

  // Worker is active if not shutdown requested
  stats.worker_active = !shutdown_requested_.load(std::memory_order_acquire);

  return stats;
}

}  // namespace at::neuron
