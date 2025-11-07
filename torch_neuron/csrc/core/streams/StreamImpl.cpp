#include "StreamImpl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>

#include <thread>

#include "torch_neuron/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuron/csrc/core/KernelExecution.h"
#include "torch_neuron/csrc/core/NeuronDevice.h"
#include "torch_neuron/csrc/core/NeuronLogging.h"
#include "torch_neuron/csrc/core/NeuronStorageImpl.h"
#include "torch_neuron/csrc/core/OperationExecutionEngine.h"
#include "torch_neuron/csrc/core/runtime/NRTHandler.h"
#include "torch_neuron/csrc/core/utils/CPUFallbackExecutor.h"
#include "torch_neuron/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuron/csrc/ops/InternalOps.h"
#include "torch_neuron/csrc/utils/CopyUtils.h"

extern "C" {
#include <nrt/nrt.h>
}

#include <atomic>
#include <chrono>
#include <iomanip>
#include <mutex>

// Stream configuration constants
namespace StreamConfig {
// Timeout configurations
constexpr auto kOperationTimeout = std::chrono::milliseconds(30000);
constexpr auto kShutdownTimeout = std::chrono::seconds(5);
constexpr auto kMonitoringInterval = std::chrono::seconds(5);
constexpr auto kStatsLogInterval = std::chrono::seconds(60);

// Thread configuration
constexpr int kTotalWorkerThreads = 2;

// Resource limits
constexpr int kDefaultNrtMaxInflightRequests = 0;

// Logging configuration
constexpr int kLogCounterCycles = 12;  // 12 * 5 seconds = 60 seconds
constexpr size_t kMaxLeakDetailsToLog = 5;
}  // namespace StreamConfig

namespace {
// NRT async availability check moved to NRTHandler
}  // anonymous namespace

namespace at::neuron {

StreamImpl::StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id, int prio,
                       OperationExecutionEngine* operation_execution_engine)
    : device_index(device_idx),
      stream_id(id),
      priority(prio),
      operation_execution_engine_(operation_execution_engine),
      creation_time_(std::chrono::steady_clock::now()),
      operation_timeout_(StreamConfig::kOperationTimeout) {
  TORCH_NEURON_STREAM_DEBUG("Creating StreamImpl", "device_index=", device_index,
                            "stream_id=", stream_id, "priority=", priority);

  // Initialize per-stream ErrorHandler
  error_handler_ = std::make_unique<ErrorHandler>();

  TORCH_NEURON_DEBUG("StreamImpl fully initialized", "stream_id=", stream_id,
                     "device_index=", device_index, "priority=", priority);
}

StreamImpl::~StreamImpl() {
  TORCH_NEURON_DEBUG("Destroying StreamImpl", "stream_id=", stream_id);

  // Request graceful shutdown with active cancellation
  TORCH_NEURON_DEBUG("Requesting graceful shutdown with active cancellation",
                     "stream_id=", stream_id);

  TORCH_NEURON_DEBUG("Synchronizing stream before destruction", "stream_id=", stream_id);
  Synchronize();

  shutdown_requested.store(true, std::memory_order_release);

  TORCH_NEURON_DEBUG("Stream destruction completed", "stream_id=", stream_id);
}

std::future<OperationContextResult> StreamImpl::SubmitOperationContext(
    OperationContext operation_context) {
  TORCH_NEURON_STREAM_DEBUG("Submitting operation context", "stream_id=", stream_id,
                            "op=", operation_context.GetOpName());
  auto op = std::make_shared<OperationContext>(std::move(operation_context));
  op->submit_time = std::chrono::steady_clock::now();

  // Track resource usage
  total_operations_submitted_++;

  // Add to unified operation tracking
  {
    std::lock_guard<std::mutex> lock(active_operations_mutex_);

    // Attach any pending event dependencies to this operation
    if (!pending_event_dependencies_.empty()) {
      op->events_to_wait_for = std::move(pending_event_dependencies_);
      pending_event_dependencies_.clear();
      TORCH_NEURON_STREAM_DEBUG("Attached event dependencies to operation", "stream_id=", stream_id,
                                "op=", op->GetOpName(),
                                "num_dependencies=", op->events_to_wait_for.size());
    }

    active_operations_.push_back(op);
    TORCH_NEURON_STREAM_DEBUG("Added operation to active tracking", "stream_id=", stream_id,
                              "op=", op->GetOpName(), "total_active=", active_operations_.size());
  }

  // Route to global execution manager via NeuronResourceManager
  TORCH_NEURON_STREAM_DEBUG("Routing to global execution manager", "stream_id=", stream_id,
                            "op=", op->GetOpName());

  // As of now, in order to guarantee serial set of executions, we enqueue
  // all tasks through the same sequence of workers. We skip the actual compilation
  // in case the operation does not require one, while retaining the original
  // order of execution.
  OperationExecutionTask task(op.get(), this, priority, TaskType::COMPILATION);
  operation_execution_engine_->SubmitTask(std::move(task));

  TORCH_NEURON_STREAM_DEBUG("Operation submitted to global execution manager",
                            "stream_id=", stream_id, "op=", op->GetOpName());

  TORCH_NEURON_STREAM_DEBUG("Operation context submitted successfully", "stream_id=", stream_id,
                            "op=", op->GetOpName());
  return op->promise.get_future();
}

bool StreamImpl::Query() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(active_operations_mutex_));
  return active_operations_.empty();
}

void StreamImpl::Synchronize() {
  TORCH_NEURON_DEBUG("Synchronizing stream", "stream_id=", stream_id);

  if (Py_IsInitialized() && PyGILState_Check()) {
    // Release GIL before waiting so worker threads can complete tensor cleanup
    Py_BEGIN_ALLOW_THREADS std::unique_lock<std::mutex> lock(active_operations_mutex_);
    completion_cv.wait(lock, [this] {
      TORCH_NEURON_STREAM_DEBUG("Waiting on synchronization, active_operations=",
                                active_operations_.size());
      return active_operations_.empty();
    });
    Py_END_ALLOW_THREADS
  } else {
    std::unique_lock<std::mutex> lock(active_operations_mutex_);
    completion_cv.wait(lock, [this] {
      TORCH_NEURON_STREAM_DEBUG("Waiting on synchronization, active_operations=",
                                active_operations_.size());
      return active_operations_.empty();
    });
  }

  // After synchronization completes, check for and propagate any pending exceptions
  error_handler_->CheckAndThrowPendingException(stream_id);

  TORCH_NEURON_DEBUG("Stream synchronization completed", "stream_id=", stream_id);
}

StreamImpl::StreamResourceStats StreamImpl::GetResourceStats() const {
  StreamResourceStats stats;
  stats.stream_id = stream_id;
  stats.total_submitted = total_operations_submitted_.load();
  stats.total_completed = total_operations_completed_.load();
  stats.total_failed = total_operations_failed_.load();
  stats.is_idle = Query();

  // Calculate uptime
  auto now = std::chrono::steady_clock::now();
  stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - creation_time_);

  stats.compilation_queue_size = 0;
  stats.execution_queue_size = 0;

  return stats;
}

void StreamImpl::RemoveActiveOperation(const OperationContext* operation) {
  std::shared_ptr<OperationContext> operation_to_destroy;
  {
    std::lock_guard<std::mutex> lock(active_operations_mutex_);
    auto it = std::find_if(
        active_operations_.begin(), active_operations_.end(),
        [operation](const auto& active_operation) { return active_operation.get() == operation; });
    if (it != active_operations_.end()) {
      std::string op_name = it->get()->GetOpName();
      TORCH_NEURON_STREAM_DEBUG("Removing completed operation from active tracking",
                                "stream_id=", stream_id, "operation=", op_name,
                                "remaining_active=", active_operations_.size() - 1);
      operation_to_destroy = std::move(*it);
      active_operations_.erase(it);

      completion_cv.notify_all();
      TORCH_NEURON_DEBUG("Operation removed and completion_cv notified", "stream_id=", stream_id,
                         "operation=", op_name, "new_active_count=", active_operations_.size());
    } else {
      TORCH_NEURON_ERROR("Completed operation not found in active tracking",
                         "stream_id=", stream_id, "operation=", operation->GetOpName());
    }
  }
}

void StreamImpl::CompleteOperation(OperationContext* operation) {
  TORCH_NEURON_DEBUG("Operation completion callback", "stream_id=", stream_id,
                     "operation=", operation->GetOpName());

  // Set execution end time
  operation->execute_end = std::chrono::steady_clock::now();

  TORCH_NEURON_DEBUG("Operation completed successfully", "stream_id=", stream_id,
                     "op=", operation->GetOpName(),
                     "execution_time_us=", operation->GetExecutionTime().count(),
                     "total_time_us=", operation->GetTotalTime().count());

  // Copy results to original tensors if needed (for dtype conversion)
  try {
    CopyResultsToOriginalTensors(operation);
  } catch (const std::exception& e) {
    TORCH_NEURON_ERROR("Failed to copy results to original tensors", "stream_id=", stream_id,
                       "op=", operation->GetOpName(), "error=", e.what());
    throw std::runtime_error("Failed to copy results to original tensors: " +
                             std::string(e.what()));
  }

  // Complete any events registered with this operation
  for (NeuronEvent* event : operation->events_to_complete) {
    if (event) {
      event->complete_event();
      TORCH_NEURON_STREAM_DEBUG("Completed event for operation", "stream_id=", stream_id,
                                "op=", operation->GetOpName(), "event=", static_cast<void*>(event));
    }
  }

  total_operations_completed_++;

  operation->promise.set_value(OperationContextResult::CreateSuccess());

  RemoveActiveOperation(operation);
}

void StreamImpl::RegisterEventWithLastOperation(NeuronEvent* event) {
  if (!event) {
    return;
  }

  std::lock_guard<std::mutex> lock(active_operations_mutex_);

  if (active_operations_.empty()) {
    // No operations at all - complete event immediately
    TORCH_NEURON_STREAM_DEBUG("No operations active, completing event immediately",
                              "stream_id=", stream_id, "event=", static_cast<void*>(event));
    event->complete_event();
    return;
  }

  // Attach event to the last operation (most recently submitted)
  auto& last_operation = active_operations_.back();
  last_operation->events_to_complete.push_back(event);

  TORCH_NEURON_STREAM_DEBUG("Registered event with last operation", "stream_id=", stream_id,
                            "op=", last_operation->GetOpName(),
                            "op_ptr=", static_cast<void*>(last_operation.get()),
                            "event=", static_cast<void*>(event));
}

void StreamImpl::RegisterEventDependencyForNextOperation(const NeuronEvent* event) {
  if (!event) {
    return;
  }

  std::lock_guard<std::mutex> lock(active_operations_mutex_);

  // Store the event to attach to the next submitted operation
  pending_event_dependencies_.push_back(event);

  TORCH_NEURON_STREAM_DEBUG("Registered event dependency for next operation",
                            "stream_id=", stream_id, "event=", static_cast<const void*>(event),
                            "total_pending=", pending_event_dependencies_.size());
}

void StreamImpl::HandleErrorWithCleanup(OperationContext* operation,
                                        const std::string& error_message, ErrorContext::Stage stage,
                                        int error_code) {
  TORCH_NEURON_DEBUG("Delegating error handling to ErrorHandler", "stream_id=", stream_id,
                     "op=", operation->GetOpName(), "stage=", static_cast<int>(stage));

  try {
    WaitForPriorOperationsToComplete(operation);
  } catch (const std::exception& sync_error) {
    TORCH_NEURON_ERROR("Failed to synchronize prior operations before CPU fallback",
                       "stream_id=", stream_id, "op=", operation->GetOpName(),
                       "sync_error=", sync_error.what());

    std::string full_error = "Synchronization failed: " + std::string(sync_error.what());
    operation->promise.set_value(OperationContextResult::CreateError(full_error));
    RemoveActiveOperation(operation);
    total_operations_failed_++;
    return;
  }

  auto status =
      error_handler_->HandleOperationError(operation, error_message, stage, stream_id, error_code);

  if (status.IsSuccess()) {
    // CPU fallback succeeded - complete the operation normally
    TORCH_NEURON_DEBUG("CPU fallback succeeded, completing operation normally",
                       "stream_id=", stream_id, "op=", operation->GetOpName());
    CompleteOperation(operation);
  } else {
    operation->promise.set_value(status);
    RemoveActiveOperation(operation);
    total_operations_failed_++;
  }
}

void StreamImpl::WaitForPriorOperationsToComplete(const OperationContext* failed_operation) {
  TORCH_NEURON_DEBUG("Waiting for prior operations to complete before CPU fallback",
                     "stream_id=", stream_id, "failed_op=", failed_operation->GetOpName());
  {
    std::unique_lock<std::mutex> lock(active_operations_mutex_);

    bool is_at_front =
        !active_operations_.empty() && active_operations_.front().get() == failed_operation;
    if (is_at_front) {
      return;
    }

    completion_cv.wait(lock, [this, failed_operation]() {
      // Since active_operations_ is FIFO, we only need to check if the failed operation
      // is at the front of the queue. If it is, there are no prior operations.
      return active_operations_.empty() || active_operations_.front().get() == failed_operation;
    });
  }
  error_handler_->CheckAndThrowPendingException(stream_id);

  TORCH_NEURON_DEBUG(
      "All prior operations completed successfully, safe to proceed with CPU fallback",
      "stream_id=", stream_id, "failed_op=", failed_operation->GetOpName());
}

void StreamImpl::CopyResultsToOriginalTensors(OperationContext* operation) {
  TORCH_NEURON_STREAM_DEBUG("Copying results to original tensors", "stream_id=", stream_id,
                            "op=", operation->GetOpName());
  const auto& outputs = operation->kernel_execution->GetOutputs();
  const auto& internal_outputs = operation->kernel_execution->GetInternalOutputs();

  // Copy results from internal tensors to original tensors using the mapping pairs
  TORCH_CHECK(outputs.size() == internal_outputs.size(), "Mismatch output tensor sizes (",
              outputs.size(), ") and (", internal_outputs.size(), ") for operation ",
              operation->GetOpName());

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    const auto& internal_output = internal_outputs[i];
    if (internal_output.data_ptr() != output.data_ptr()) {
      try {
        output.copy_(internal_output, /*non_blocking=*/true);
        TORCH_NEURON_STREAM_DEBUG("Copied result tensor", "stream_id=", stream_id,
                                  "op=", operation->GetOpName());
      } catch (const std::exception& e) {
        throw std::runtime_error("Failed to copy result tensor for op '" + operation->GetOpName() +
                                 "': " + e.what());
      }
    }
  }

  TORCH_NEURON_STREAM_DEBUG("Successfully copied results to original tensors",
                            "stream_id=", stream_id, "op=", operation->GetOpName());
}

}  // namespace at::neuron
