#include "OperationExecutionEngine.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationEngine.h"
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/streams/StreamManager.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron {

template <typename T>
bool PriorityQueue<T>::empty() const {
  return high_.empty() && default_.empty() && low_.empty();
}

template <typename T>
size_t PriorityQueue<T>::size() const {
  return high_.size() + default_.size() + low_.size();
}

template <typename T>
std::deque<T>& PriorityQueue<T>::GetQueueForPriority(int priority) {
  if (priority < 0) {
    return high_;
  } else if (priority > 0) {
    return low_;
  } else {
    return default_;
  }
}

template <typename T>
void PriorityQueue<T>::push_back(T operation) {
  GetQueueForPriority(operation->stream->priority).push_back(operation);
}

template <typename T>
void PriorityQueue<T>::push_front(T operation) {
  GetQueueForPriority(operation->stream->priority).push_front(operation);
}

template <typename T>
T PriorityQueue<T>::pop_front() {
  if (!high_.empty()) {
    auto operation = high_.front();
    high_.pop_front();
    return operation;
  }
  if (!default_.empty()) {
    auto operation = default_.front();
    default_.pop_front();
    return operation;
  }
  auto operation = low_.front();
  low_.pop_front();
  return operation;
}

template class PriorityQueue<OperationContext*>;

OperationExecutionEngine::OperationExecutionEngine(CompilationCache* compilation_cache,
                                                   ModelHandleCache* model_handle_cache,
                                                   int64_t queue_low_threshold,
                                                   int64_t queue_target_depth)
    : compilation_cache_(compilation_cache),
      model_handle_cache_(model_handle_cache),
      queue_low_threshold_(queue_low_threshold),
      queue_target_depth_(queue_target_depth) {
  // Check if concatenation is enabled
  concatenation_enabled_ = IsConcatenationEnabled();

  if (concatenation_enabled_) {
    TORCH_NEURONX_DEBUG("Concatenation is enabled, creating ConcatenationEngine",
                        "queue_low_threshold=", queue_low_threshold_,
                        "queue_target_depth=", queue_target_depth_);
    concatenation_engine_ = std::make_unique<ConcatenationEngine>(
        &execution_queue_depth_, queue_low_threshold_, queue_target_depth_);
    concatenation_thread_ = std::thread(&OperationExecutionEngine::ConcatenationWorker, this);
  }

  // Start worker threads
  compilation_thread_ = std::thread(&OperationExecutionEngine::CompilationWorker, this);
  execution_thread_ = std::thread(&OperationExecutionEngine::ExecutionWorker, this);

  TORCH_NEURONX_DEBUG("OperationExecutionEngine worker threads started",
                      "concatenation_enabled=", concatenation_enabled_,
                      "compilation_thread_id=", compilation_thread_.get_id(),
                      "execution_thread_id=", execution_thread_.get_id());
}

OperationExecutionEngine::~OperationExecutionEngine() { Shutdown(); }

void OperationExecutionEngine::Shutdown() {
  bool expected = false;
  if (!shutdown_requested_.compare_exchange_strong(expected, true, std::memory_order_release)) {
    return;
  }
  TORCH_NEURONX_DEBUG("Shutting down OperationExecutionEngine");

  // Wake up all worker threads
  if (concatenation_enabled_) {
    concatenation_cv_.notify_all();
  }
  compilation_cv_.notify_all();
  execution_cv_.notify_all();

  if (concatenation_enabled_ && concatenation_thread_.joinable()) {
    concatenation_thread_.join();
  }
  if (compilation_thread_.joinable()) {
    compilation_thread_.join();
  }
  if (execution_thread_.joinable()) {
    execution_thread_.join();
  }
  TORCH_NEURONX_DEBUG("OperationExecutionEngine shutdown complete");
}

void OperationExecutionEngine::CompilationWorker() {
  TORCH_NEURONX_DEBUG("Compilation worker thread started");

  while (true) {
    // Wait for a task or shutdown signal
    std::unique_lock<std::mutex> lock(compilation_mutex_);
    compilation_cv_.wait(lock, [this] {
      return !compilation_queue_.empty() || shutdown_requested_.load(std::memory_order_acquire);
    });

    if (shutdown_requested_.load(std::memory_order_acquire) && compilation_queue_.empty()) {
      break;
    }

    if (!compilation_queue_.empty()) {
      OperationContext* operation = compilation_queue_.pop_front();
      lock.unlock();

      if (!ProcessPreprocessingTask(operation)) {
        continue;  // Error already handled
      }

      ProcessCompilationTask(operation);
      NotifyExecutionReady();
    }
  }
  TORCH_NEURONX_DEBUG("Compilation worker thread terminated");
}

void OperationExecutionEngine::ProcessCompilationTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;
  // Set compilation start time
  // TODO(rpsilva): Introduce a global metric collection
  operation->compile_start = std::chrono::steady_clock::now();
  // Get the compilable kernel
  auto& kernel = operation->GetCompilableKernel();
  const std::string& op_name = operation->GetOpName();
  try {
    int stream_id = stream->stream_id;
    TORCH_NEURONX_DEBUG("Processing compilation task", "stream_id=", stream_id, "op=", op_name);
    // Compile NEFF through shared compilation cache
    compilation_cache_->GetOrCompileNeff(kernel, stream_id);

    if (!kernel.HasCachedNeff()) {
      throw torch_neuronx::CompilationRuntimeException("NEFF creation failed for operation: " +
                                                       op_name);
    }

    // Set compilation end time
    operation->compile_end = std::chrono::steady_clock::now();

    operation->MarkExecutionReady();
    TORCH_NEURONX_DEBUG("Compilation task completed successfully", "stream_id=", stream_id,
                        "op=", op_name);
  } catch (const torch_neuronx::CompilationRuntimeException& e) {
    stream->HandleErrorWithCleanup(operation, e);
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(operation, torch_neuronx::CompilationRuntimeException(e.what()));
  }
}

void OperationExecutionEngine::ExecutionWorker() {
  TORCH_NEURONX_DEBUG("Execution worker thread started");

  c10::DeviceIndex num_devices = c10_neuron::device_count();
  auto& stream_manager = NeuronResourceManager::Instance().GetStreamManager();
  while (true) {
    bool work_found;
    do {
      work_found = false;
      for (c10::DeviceIndex device = 0; device < num_devices; device++) {
        stream_manager.ForEachStreamImpl(device, [&](StreamImpl* stream_impl) -> bool {
          OperationContext* operation = stream_impl->GetNextReadyOperation();
          if (operation != nullptr) {
            ProcessExecutionTask(operation);
            work_found = true;
          }
          // Return false to continue iterating through all streams
          // We want to check all streams for ready operations
          return false;
        });
      }
    } while (work_found);

    if (shutdown_requested_.load(std::memory_order_acquire)) {
      break;
    }

    {
      std::unique_lock<std::mutex> lock(execution_mutex_);
      execution_cv_.wait(lock, [this] {
        return execution_pending_ || shutdown_requested_.load(std::memory_order_acquire);
      });
      execution_pending_ = false;
    }
  }
  TORCH_NEURONX_DEBUG("Execution worker thread terminated");
}

void OperationExecutionEngine::NotifyExecutionReady() {
  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    execution_pending_ = true;
  }
  execution_cv_.notify_one();
}

void OperationExecutionEngine::ProcessExecutionTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;
  const std::string operation_name = operation->GetOpName();
  try {
    TORCH_NEURONX_DEBUG("Processing execution task", "stream_id=", stream->stream_id,
                        "operation_name=", operation_name);
    // TODO(rpsilva): Introduce a global metric collection
    operation->execute_start = std::chrono::steady_clock::now();
    operation->Execute();
    if (torch_neuronx::shouldLogExecuted(operation_name)) {
      torch_neuronx::NeuronLogger::getInstance().log(
          torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::OPERATOR_EXECUTED,
          "Operator '" + operation_name + "' executed on Neuron");
      torch_neuronx::markExecutedLogged(operation_name);
    }
    stream->CompleteOperation(operation);

    // Decrement execution queue depth after execution
    if (concatenation_enabled_) {
      int64_t new_depth = execution_queue_depth_.fetch_sub(1) - 1;

      // Check if we need to trigger refill
      if (new_depth < queue_low_threshold_) {
        TORCH_NEURONX_DEBUG("Execution queue depth below threshold, triggering refill",
                            "current_depth=", new_depth, "threshold=", queue_low_threshold_);
        // Enqueue nullptr with priority to front of concatenation queue
        {
          std::lock_guard<std::mutex> lock(concatenation_mutex_);
          concatenation_queue_.push_front(nullptr);
        }
        concatenation_cv_.notify_one();
      }
    }

    TORCH_NEURONX_DEBUG("Execution task completed successfully", "operation_name=", operation_name);
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    stream->HandleErrorWithCleanup(operation, e);
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(operation, torch_neuronx::ExecutionRuntimeException(e.what()));
  }
}

bool OperationExecutionEngine::ProcessPreprocessingTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;

  try {
    TORCH_NEURONX_DEBUG("Processing preprocessing task", "op=", operation->GetOpName());

    c10_neuron::lazy::LazyTransformationManager::ProcessOperationInputs(operation);

    // Transformations were applied if needed, kernel has been updated via MLIR merge
    return true;

  } catch (const torch_neuronx::CompilationRuntimeException& e) {
    stream->HandleErrorWithCleanup(operation, e);
    return false;
  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("preprocessing failed with=", e.what());
    stream->HandleErrorWithCleanup(operation, torch_neuronx::CompilationRuntimeException(e.what()));
    return false;
  }
}

void OperationExecutionEngine::SubmitOperation(OperationContext* operation) {
  if (shutdown_requested_.load()) {
    TORCH_NEURONX_DEBUG("Rejecting operation submission during shutdown",
                        "op=", operation->GetOpName());
    return;
  }

  if (concatenation_enabled_) {
    // Route through concatenation queue
    {
      std::lock_guard<std::mutex> lock(concatenation_mutex_);
      concatenation_queue_.push_back(operation);
    }
    concatenation_cv_.notify_one();
  } else {
    // Route directly to compilation queue
    SubmitCompilationTask(operation);
  }
}

void OperationExecutionEngine::SubmitCompilationTask(OperationContext* operation) {
  if (shutdown_requested_.load()) {
    TORCH_NEURONX_DEBUG("Rejecting compilation task submission during shutdown",
                        "op=", operation->GetOpName());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(compilation_mutex_);
    compilation_queue_.push_back(operation);
  }
  compilation_cv_.notify_one();
}

void OperationExecutionEngine::ConcatenationWorker() {
  TORCH_NEURONX_DEBUG("Concatenation worker thread started");

  while (true) {
    // Wait for a task or shutdown signal
    std::unique_lock<std::mutex> lock(concatenation_mutex_);
    concatenation_cv_.wait(lock, [this] {
      return !concatenation_queue_.empty() || shutdown_requested_.load(std::memory_order_acquire);
    });

    if (shutdown_requested_.load(std::memory_order_acquire) && concatenation_queue_.empty()) {
      break;
    }

    if (!concatenation_queue_.empty()) {
      OperationContext* operation = concatenation_queue_.front();
      concatenation_queue_.pop_front();
      lock.unlock();

      TORCH_NEURONX_DEBUG("Processing concatenation task",
                          "op=", operation ? operation->GetOpName() : "Flush");

      // Process the operation through concatenation engine
      auto operations_to_compile = concatenation_engine_->processConcatenationTask(operation);

      // Submit all resulting operations to compilation queue
      for (auto* op : operations_to_compile) {
        try {
          if (op->cascading_operations.empty()) {
            SubmitCompilationTask(op);
          } else {
            // Own compilation for concatenated ops to catch failures
            ProcessCompilationTask(op);
            NotifyExecutionReady();
          }
          // Compilation for concatenated op succeeded.
          execution_queue_depth_.fetch_add(1);

          // TODO(apoorvgu,fhaolin): Implement changes to reflect concatenated ops in
          // StreamImpl::active_operations_.
          // 1. Find the sequence of operations in active_operations_
          // 2. Populate cascading_operations with shared_ptrs
          // 3. Replace the sequence with the concatenated operation

        } catch (const torch_neuronx::CompilationRuntimeException& e) {
          // Encountered compiler error fallback to op by op
          auto cascading_operations = concatenation_engine_->processCompilationFailure(op);

          // Increment for the number of ops being enqueued
          execution_queue_depth_.fetch_add(cascading_operations.size());
          for (auto& cascading_op : cascading_operations) {
            // Submit for compilation like normal flow
            SubmitCompilationTask(cascading_op);
          }
        }
        // Other exceptions are not caught since they cannot
        // be recovered from.
      }
    }
  }
  TORCH_NEURONX_DEBUG("Concatenation worker thread terminated");
}

bool OperationExecutionEngine::IsConcatenationEnabled() const {
  const char* concat_env = std::getenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  return concat_env && (std::string(concat_env) == "1" || std::string(concat_env) == "true" ||
                        std::string(concat_env) == "yes");
}

}  // namespace at::neuron
