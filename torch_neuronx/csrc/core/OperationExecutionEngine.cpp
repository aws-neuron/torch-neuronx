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
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron {

OperationExecutionEngine::OperationExecutionEngine(CompilationCache* compilation_cache,
                                                   ModelHandleCache* model_handle_cache)
    : compilation_cache_(compilation_cache), model_handle_cache_(model_handle_cache) {
  // Check if concatenation is enabled via environment variable (cache the result)
  // TODO(apoorvgu):Use torch.use_deterministic_algorithms instead
  const char* concat_env = std::getenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  concatenation_enabled_ = concat_env && (std::string(concat_env) == "1");

  // Override queue depth parameters from environment variables if set (cache the result)
  queue_low_threshold_ = DEFAULT_QUEUE_LOW_THRESHOLD;
  const char* queue_low_threshold_env =
      std::getenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD");
  if (queue_low_threshold_env) {
    try {
      int64_t parsed_value = std::stoll(queue_low_threshold_env);
      if (parsed_value > 0) {
        queue_low_threshold_ = parsed_value;
        TORCH_NEURONX_DEBUG("Using TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD=",
                            queue_low_threshold_);
      } else {
        TORCH_NEURONX_WARN("Invalid TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD value:",
                           queue_low_threshold_env,
                           "- must be positive, using default:", DEFAULT_QUEUE_LOW_THRESHOLD);
      }
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN("Failed to parse TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD:",
                         queue_low_threshold_env, "- using default:", DEFAULT_QUEUE_LOW_THRESHOLD);
    }
  }

  if (concatenation_enabled_) {
    TORCH_NEURONX_DEBUG("Concatenation is enabled, creating ConcatenationEngine",
                        "queue_low_threshold=", queue_low_threshold_);
    concatenation_engine_ = std::make_unique<ConcatenationEngine>();
    concatenation_thread_ = std::thread(&OperationExecutionEngine::ConcatenationWorker, this);
  }

  // Read max inflight per XU from environment variable
  if (const char* max_inflight_env = std::getenv("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS")) {
    try {
      int val = std::stoi(max_inflight_env);
      if (val > 0) {
        max_inflight_per_xu_ = static_cast<size_t>(val);
        TORCH_NEURONX_DEBUG("Using NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=",
                            max_inflight_per_xu_);
      }
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN("Invalid NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS value: ",
                         max_inflight_env, ". Using default: ", kDefaultMaxInflightPerXU);
    }
  }

  // Start worker threads
  compilation_thread_ = std::thread(&OperationExecutionEngine::CompilationWorker, this);
  // Start execution worker thread
  execution_thread_ = std::thread(&OperationExecutionEngine::AsyncExecutionWorker, this);

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
      OperationContext* operation = compilation_queue_.front();
      compilation_queue_.pop_front();
      lock.unlock();

      ProcessPrepareTask(operation);
    }
  }
  TORCH_NEURONX_DEBUG("Compilation worker thread terminated");
}

void OperationExecutionEngine::ProcessPrepareTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;

  try {
    if (operation->RequiresCompilation()) {
      if (!IsConcatenationEnabled()) {
        ProcessPreprocessingTask(operation);
      }
      ProcessCompilationTask(operation);
      if (compilation_cache_->IsServiceBackend()) {
        // Async path: NotifyExecutionReady called by the callback
        return;
      }
    } else {
      // Non-compilable operations (including collectives) are marked ready immediately
      operation->MarkExecutionReady();
    }
  } catch (const torch_neuronx::CompilationRuntimeException& e) {
    stream->HandleErrorWithCleanup(operation, e);
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(operation, torch_neuronx::CompilationRuntimeException(e.what()));
  }
  NotifyExecutionReady();
}

void OperationExecutionEngine::ProcessPreprocessingTask(OperationContext* operation) {
  // Validate operation before accessing any internal state
  TORCH_CHECK(operation != nullptr, "Received null operation");
  TORCH_CHECK(operation->kernel_execution != nullptr, "Operation has null kernel_execution");

  const std::string& op_name = operation->GetOpName();
  TORCH_NEURONX_DEBUG("Processing preprocessing task", "op=", op_name);
  c10_neuron::lazy::LazyTransformationManager::ProcessOperationInputs(operation);
}

void OperationExecutionEngine::HandleCompilationComplete(OperationContext* operation) {
  auto* stream = operation->stream;
  const std::string& op_name = operation->GetOpName();

  if (!operation->GetCompilableKernel().HasCachedNeff()) {
    stream->HandleErrorWithCleanup(operation,
                                   torch_neuronx::CompilationRuntimeException(
                                       "NEFF creation failed for operation: " + op_name));
    return;
  }

  operation->compile_end = std::chrono::steady_clock::now();
  operation->MarkExecutionReady();
  TORCH_NEURONX_DEBUG("Compilation task completed successfully", "stream_id=", stream->stream_id,
                      "op=", op_name);

  // Only trigger concatenated op compilation from individual cascading operations,
  // not from the concatenated operation itself (to prevent infinite recursion)
  if (!operation->IsConcatenatedOperation() && operation->HasConcatenatedOperation() &&
      operation->concatenation_state_->IncrementAndCheckCompiledCascadingOpsCount()) {
    auto* concat_op = operation->GetConcatenatedOperation();
    try {
      ProcessCompilationTask(concat_op);
    } catch (const torch_neuronx::CompilationRuntimeException& e) {
      // Encountered compiler error fallback to op by op
      TORCH_NEURONX_ERROR("Compilation failed for concatenated op:", concat_op, "op_name:", op_name,
                          "with exception: ", e.what());
      concat_op->stream->HandleErrorWithCleanup(concat_op, e);
    } catch (const std::exception& e) {
      // Encountered compiler error fallback to op by op
      TORCH_NEURONX_ERROR("Compilation failed for concatenated op:", concat_op, "op_name:", op_name,
                          "with exception: ", e.what());
      concat_op->stream->HandleErrorWithCleanup(
          concat_op, torch_neuronx::CompilationRuntimeException(e.what()));
    }
  }
}

void OperationExecutionEngine::ProcessCompilationTask(OperationContext* operation) {
  auto* stream = operation->stream;
  const std::string& op_name = operation->GetOpName();
  int stream_id = stream->stream_id;

  operation->compile_start = std::chrono::steady_clock::now();
  auto& kernel = operation->GetCompilableKernel();

  TORCH_NEURONX_DEBUG("Processing compilation task", "stream_id=", stream_id, "op=", op_name);

  if (compilation_cache_->IsServiceBackend()) {
    // Service active: async path with thread pool
    compilation_cache_->GetOrCompileNeffAsync(kernel, stream_id, [this, operation](bool success) {
      if (!success) {
        operation->stream->HandleErrorWithCleanup(
            operation, torch_neuronx::CompilationRuntimeException(
                           "NEFF creation failed for operation: " + operation->GetOpName()));
      } else {
        HandleCompilationComplete(operation);
      }
      NotifyExecutionReady();
    });
    return;
  }

  // No service: compile directly on this thread
  compilation_cache_->GetOrCompileNeff(kernel, stream_id);
  HandleCompilationComplete(operation);
}

void OperationExecutionEngine::AsyncExecutionWorker() {
  TORCH_NEURONX_DEBUG("NRT-Asynchronous execution worker thread started");
  auto devices = c10_neuron::get_visible_device_indices();

  while (true) {
    bool work_found;
    do {
      work_found = false;

      // First, check for completions
      CheckCompletions();

      // Then dispatch new ready operations (only iterate visible devices)
      for (auto device : devices) {
        ForEachStreamImpl(device, [&](StreamImpl* stream_impl) -> bool {
          OperationContext* operation = stream_impl->GetNextOperationToSchedule();
          if (operation != nullptr) {
            DispatchAsyncExecutionTask(operation);
            work_found = true;
          }
          return false;
        });
      }
    } while (work_found);

    // Only wait if there are no pending operations to poll
    if (!HasPendingAsyncOperations()) {
      std::unique_lock<std::mutex> lock(execution_mutex_);
      execution_cv_.wait(lock, [this] {
        return execution_pending_ || shutdown_requested_.load(std::memory_order_acquire);
      });
      execution_pending_ = false;
    }

    if (shutdown_requested_.load(std::memory_order_acquire) && !HasPendingAsyncOperations()) {
      break;
    }
  }
  TORCH_NEURONX_DEBUG("NRT-Asynchronous execution worker thread terminated");
}

void OperationExecutionEngine::NotifyExecutionReady() {
  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    execution_pending_ = true;
  }
  execution_cv_.notify_one();
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

  // Helper lambda to submit an operation to the next stage (compilation)
  auto submitToNextStage = [this](OperationContext* op) {
    if (IsDeviceKernelType(op->GetKernelType())) {
      execution_queue_depth_.fetch_add(1);
    }
    if (op->RequiresCompilation()) {
      ProcessPrepareTask(op);
    } else {
      op->MarkExecutionReady();
      NotifyExecutionReady();
    }
  };

  while (true) {
    OperationContext* operation = nullptr;
    std::list<OperationContext*> operations_to_compile;
    bool needs_flush = false;
    bool shutdown = false;

    // ========================================================================
    // CRITICAL SECTION: Wait and pop from queue
    // ========================================================================
    {
      std::unique_lock<std::mutex> lock(concatenation_mutex_);

      // Wait for: new operation, low queue depth (needs refill), or shutdown
      concatenation_cv_.wait(lock, [this] {
        return !concatenation_queue_.empty() ||
               execution_queue_depth_.load() <= queue_low_threshold_ ||
               shutdown_requested_.load(std::memory_order_acquire);
      });

      shutdown = shutdown_requested_.load(std::memory_order_acquire);

      // Check shutdown condition - only exit if queue is empty
      if (shutdown && concatenation_queue_.empty()) {
        // Flush any remaining buffered operations before exiting
        break;
      }

      // Check if we need to flush (queue depth below threshold or shutdown)
      needs_flush = (execution_queue_depth_.load() <= queue_low_threshold_) || shutdown;

      // Pop operation from queue if available
      // During flush/shutdown: pop to include in flush
      // During normal operation: pop for concatenation processing
      if (!concatenation_queue_.empty()) {
        operation = concatenation_queue_.front();
        concatenation_queue_.pop_front();
      }
    }
    // Lock released here - all queue operations complete

    // ========================================================================
    // Handle flush (low queue depth or shutdown)
    // ========================================================================
    int64_t current_depth = execution_queue_depth_.load();
    if (needs_flush) {
      // Flush ALL buffered operations without concatenation
      operations_to_compile = concatenation_engine_->Flush(SIZE_MAX);
      if (operations_to_compile.size() > 0) {
        TORCH_NEURONX_DEBUG("[CONCATENATION_WORKER] Flush triggered",
                            "reason=", shutdown ? "shutdown" : "low_queue_depth",
                            "current_depth=", current_depth, "threshold=", queue_low_threshold_,
                            "flushed_count=", operations_to_compile.size());
      }
    }

    if (operation) {
      // ========================================================================
      // Normal operation processing (not flushing)
      // ========================================================================

      // Validate operation integrity before processing
      TORCH_CHECK(operation->kernel_execution != nullptr,
                  "[CONCATENATION_WORKER] Operation has null kernel_execution, "
                  "this indicates memory corruption or use-after-free");
      TORCH_CHECK(operation->stream != nullptr,
                  "[CONCATENATION_WORKER] Operation has null stream field, "
                  "op_name=",
                  operation->GetOpName(),
                  " kernel_type=", static_cast<int>(operation->GetKernelType()));

      TORCH_NEURONX_DEBUG(
          "[CONCAT_WORKER] Processing task", "stream_id=", operation->stream->stream_id,
          "op_name=", operation->GetOpName(), "pytorch_seq_nr=", operation->pytorch_sequence_nr);

      // Process preprocessing task and concatenation with exception handling
      try {
        ProcessPreprocessingTask(operation);
        // If we have an operation from the queue, add it directly (no concatenation during low
        // queue depth)
        if (operations_to_compile.size() + current_depth <= queue_low_threshold_) {
          operation->CompleteConcatenation();
          operations_to_compile.push_back(operation);
        } else {
          // Process the operation through concatenation engine
          // Use splice to APPEND to existing flushed operations (don't overwrite!)
          auto concat_ops = concatenation_engine_->ProcessConcatenationTask(operation);
          operations_to_compile.splice(operations_to_compile.end(), concat_ops);
        }
      } catch (const std::exception& e) {
        // Exception during preprocessing - handle error and flush
        TORCH_NEURONX_ERROR("[CONCATENATION_WORKER] Exception during preprocessing",
                            "error=", e.what());
        operation->stream->HandleErrorWithCleanup(
            operation, torch_neuronx::CompilationRuntimeException(e.what()));
        // Use splice to APPEND flushed operations (don't overwrite!)
        auto flushed_ops = concatenation_engine_->Flush(static_cast<size_t>(SIZE_MAX));
        operations_to_compile.splice(operations_to_compile.end(), flushed_ops);
        operations_to_compile.push_back(operation);
      }
    }

    // Submit resulting operations to compilation queue
    for (auto* op : operations_to_compile) {
      submitToNextStage(op);
    }
  }

  // Final flush before terminating
  auto final_flush = concatenation_engine_->Flush(SIZE_MAX);
  for (auto* op : final_flush) {
    submitToNextStage(op);
  }

  TORCH_NEURONX_DEBUG("Concatenation worker thread terminated");
}

bool OperationExecutionEngine::IsConcatenationEnabled() const { return concatenation_enabled_; }

void OperationExecutionEngine::HandleConcatenationFailure(OperationContext* operation) {
  if (concatenation_enabled_ && concatenation_engine_) {
    concatenation_engine_->ProcessConcatenationFailure(operation);
  }
}

bool OperationExecutionEngine::HandleConcatenatedExecutionFailure(OperationContext* operation,
                                                                  const std::string& error_msg) {
  // Check if this is a concatenated operation failure
  OperationContext* concat_op = operation->GetConcatenatedOperation();

  if (concat_op) {
    // The concatenated operation was executed and failed
    TORCH_NEURONX_ERROR("Execution of concatenated operation failed with error: ", error_msg);
    TORCH_NEURONX_ERROR("Falling back to individual operation execution");

    // Mark the concatenated operation as failed
    concat_op->promise.set_value(OperationContextResult::CreateError(error_msg));

    // Invoke the failure callback via engine's method (which calls concatenation_engine_)
    HandleConcatenationFailure(concat_op);

    return true;  // Signal that retry is needed
  }

  return false;  // Not a concatenated operation failure, handle normally
}

bool OperationExecutionEngine::ScheduleOperation(OperationContext* op, StreamImpl* stream) {
  KernelTypeEnum kernel_type = op->GetKernelType();

  // Host-side ops: events and hints don't use NRT device scheduling.
  bool is_pending_host_op =
      kernel_type == KernelTypeEnum::kEvent || kernel_type == KernelTypeEnum::kHint;

  if (is_pending_host_op) {
    // Wait events need prior async ops to drain first
    if (kernel_type == KernelTypeEnum::kEvent &&
        op->GetKernel<KernelTypeEnum::kEvent>().GetEventAction() ==
            EventDirectKernelExecution::EventAction::kWait &&
        stream->HasPendingAsyncOperations()) {
      return false;
    }
    op->nrt_async_state_.is_pending_signal = true;
    pending_async_ops_count_.fetch_add(1);
    stream->AdvanceToNextOperation();
    return true;
  }

  // Device ops: XU inflight throttling and device scheduling
  TORCH_INTERNAL_ASSERT(stream->device_index >= 0 && stream->device_index < kMaxDevices,
                        "device_index out of bounds: ", stream->device_index);
  size_t xu_idx = GetXUIndex(kernel_type);
  auto& xu_inflight = xu_inflight_counts_[stream->device_index][xu_idx];
  if (static_cast<size_t>(xu_inflight) >= max_inflight_per_xu_) {
    return false;
  }

  ScheduleResult result = stream->ScheduleDeviceOperation(op, kernel_type);

  if (result == ScheduleResult::kScheduled) {
    xu_inflight_counts_[stream->device_index][xu_idx]++;
    pending_async_ops_count_.fetch_add(1);
    return true;
  }
  // kAdvanced: cascading op advanced (concat already scheduled). kNotReady: will retry.
  return result == ScheduleResult::kAdvanced;
}

void OperationExecutionEngine::DispatchAsyncExecutionTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;
  const std::string operation_name = operation->GetOpName();

  try {
    TORCH_NEURONX_DEBUG("Processing async execution task", "stream_id=", stream->stream_id,
                        "operation_name=", operation_name);
    bool scheduled = ScheduleOperation(operation, stream);

    if (scheduled && torch_neuronx::shouldLogExecuted(operation_name)) {
      torch_neuronx::NeuronLogger::getInstance().log(
          torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::OPERATOR_EXECUTED,
          "Operator '" + operation_name + "' executed on Neuron (async)");
      torch_neuronx::markExecutedLogged(operation_name);
    }
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    stream->HandleErrorWithCleanup(operation, e);
  } catch (const std::exception& e) {
    stream->HandleErrorWithCleanup(operation, torch_neuronx::ExecutionRuntimeException(e.what()));
  }
}

namespace {
// Helper to check if an operation is complete using explicit nrta_is_completed API.
bool IsOperationComplete(nrt::SequenceId seq_id, nrt::SequenceId completed_seq) {
  bool is_completed = false;
  NRT_STATUS status = nrt::IsNRTAsyncRequestCompleted(seq_id, &is_completed);
  if (status != NRT_SUCCESS) {
    std::string error_msg = "Operation completion check failed. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
  return is_completed;
}
}  // namespace

void OperationExecutionEngine::ProcessCompletions(StreamImpl* stream) {
  // Queue is determined per-operation based on its XU type
  std::vector<OperationContext*> ops_to_complete;

  {
    std::lock_guard<std::mutex> lock(stream->active_operations_mutex_);

    for (auto& op_ptr : stream->active_operations_) {
      // Cascading op - check if its concat op has completed
      if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation()) {
        auto* concat_op = op_ptr->GetConcatenatedOperation();
        if (concat_op->nrt_async_state_.is_scheduled) {
          size_t xu_idx = GetXUIndex(concat_op->GetKernelType());
          int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
          nrt::SequenceId concat_seq_id = concat_op->nrt_async_state_.sequence_id;
          nrt::SequenceId completed_seq =
              stream->xu_completed_seqs_.Get(static_cast<nrta_xu_t>(xu_idx), queue);
          if (IsOperationComplete(concat_seq_id, completed_seq)) {
            ops_to_complete.push_back(op_ptr.get());
            continue;
          }
        }
        break;  // Concat not done yet
      }

      // Regular operation
      if (op_ptr->nrt_async_state_.is_scheduled) {
        size_t xu_idx = GetXUIndex(op_ptr->GetKernelType());
        int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
        nrt::SequenceId op_seq_id = op_ptr->nrt_async_state_.sequence_id;
        nrt::SequenceId completed_seq =
            stream->xu_completed_seqs_.Get(static_cast<nrta_xu_t>(xu_idx), queue);
        if (IsOperationComplete(op_seq_id, completed_seq)) {
          ops_to_complete.push_back(op_ptr.get());
        } else {
          break;
        }
      } else if (op_ptr->nrt_async_state_.is_pending_signal) {
        ops_to_complete.push_back(op_ptr.get());
      } else {
        break;
      }
    }
  }

  // Complete operations
  for (auto* op : ops_to_complete) {
    if (op->nrt_async_state_.is_pending_signal) {
      try {
        op->execute_start = std::chrono::steady_clock::now();
        op->neuron_stream_os_tid = getOsTid();
        op->kernel_execution->Schedule(&op->nrt_status_, &op->nrt_async_state_.sequence_id,
                                       op->stream_id);
        op->nrt_status_ = NRT_SUCCESS;  // Host-side ops don't submit NRT work
        stream->CompleteOperation(op);
      } catch (const std::exception& e) {
        stream->HandleErrorWithCleanup(op, torch_neuronx::ExecutionRuntimeException(e.what()));
      }
      pending_async_ops_count_.fetch_sub(1);
    } else if (op->HasConcatenatedOperation() && !op->IsConcatenatedOperation()) {
      // Cascading op - complete it (concat op already ran on device)
      auto* concat_op = op->GetConcatenatedOperation();

      // Check if the concat op's promise has already been set by a previous cascading op
      // Use wait_for with zero timeout to check without blocking
      auto future_status = concat_op->result_future.wait_for(std::chrono::seconds(0));

      if (future_status != std::future_status::ready) {
        // First cascading op to complete - set the promise and decrement inflight count
        KernelTypeEnum concat_kernel_type = concat_op->GetKernelType();
        TORCH_CHECK(IsDeviceKernelType(concat_kernel_type),
                    "Concatenation operation is not a device kernel");
        size_t xu_idx = GetXUIndex(concat_kernel_type);
        xu_inflight_counts_[stream->device_index][xu_idx]--;
        concat_op->execute_end = std::chrono::steady_clock::now();
        // Check per-operation NRT status
        if (concat_op->HasNRTError()) {
          concat_op->promise.set_value(
              OperationContextResult::CreateError(concat_op->GetNRTError().what()));
        } else {
          concat_op->promise.set_value(OperationContextResult::CreateSuccess());
        }
        pending_async_ops_count_.fetch_sub(1);
      }
      // Complete each cascading op individually — error or success
      if (concat_op->HasNRTError()) {
        stream->HandleErrorWithCleanup(op, concat_op->GetNRTError());
      } else {
        stream->CompleteOperation(op);
      }
    } else {
      // Regular scheduled operation
      KernelTypeEnum kernel_type = op->GetKernelType();

      size_t xu_idx = GetXUIndex(kernel_type);
      xu_inflight_counts_[stream->device_index][xu_idx]--;

      // Check per-operation NRT status
      if (op->HasNRTError()) {
        stream->HandleErrorWithCleanup(op, op->GetNRTError());
      } else {
        stream->CompleteOperation(op);
      }
      pending_async_ops_count_.fetch_sub(1);
    }
  }
}

void OperationExecutionEngine::CheckCompletions() noexcept {
  auto devices = c10_neuron::get_visible_device_indices();
  for (auto device : devices) {
    ForEachStreamImpl(device, [&](StreamImpl* stream_impl) -> bool {
      if (stream_impl->HasPendingAsyncOperations()) {
        stream_impl->UpdateCompletionState();
        ProcessCompletions(stream_impl);
      }
      return false;
    });
  }
}

void OperationExecutionEngine::DecrementExecutionQueueDepth(OperationContext* operation) {
  TORCH_CHECK(operation != nullptr, "Received null operation pointer");

  if (!concatenation_enabled_) {
    return;
  }

  if (!IsDeviceKernelType(operation->GetKernelType())) {
    return;
  }

  // Do not decrement for concatenated ops - they are not individually tracked in queue depth.
  // Only cascading (individual) operations are counted in execution_queue_depth_.
  if (operation->IsConcatenatedOperation()) {
    return;
  }

  int64_t old_depth = execution_queue_depth_.fetch_sub(1);
  int64_t new_depth = old_depth - 1;

  // Assertion to catch increment/decrement imbalance early
  TORCH_CHECK(old_depth > 0, "[EXEC_QUEUE_DEPTH] execution_queue_depth went negative! ",
              "old_depth=", old_depth, " new_depth=", new_depth);

  bool will_notify = (new_depth <= queue_low_threshold_);
  if (will_notify) {
    concatenation_cv_.notify_one();
  }
}

}  // namespace at::neuron
