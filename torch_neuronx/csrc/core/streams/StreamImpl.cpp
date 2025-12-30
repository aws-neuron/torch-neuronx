#include "StreamImpl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>
#include <torch/extension.h>

#include <thread>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

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
      operation_timeout_(StreamConfig::kOperationTimeout) {
  TORCH_NEURONX_STREAM_DEBUG("Creating StreamImpl", "device_index=", device_index,
                             "stream_id=", stream_id, "priority=", priority);

  // Initialize per-stream ErrorHandler
  error_handler_ = std::make_unique<ErrorHandler>();

  TORCH_NEURONX_DEBUG("StreamImpl fully initialized", "stream_id=", stream_id,
                      "device_index=", device_index, "priority=", priority);
}

StreamImpl::~StreamImpl() {
  TORCH_NEURONX_DEBUG("Destroying StreamImpl", "stream_id=", stream_id);

  // Request graceful shutdown with active cancellation
  TORCH_NEURONX_DEBUG("Requesting graceful shutdown with active cancellation",
                      "stream_id=", stream_id);

  TORCH_NEURONX_DEBUG("Synchronizing stream before destruction", "stream_id=", stream_id);
  Synchronize();
  TORCH_NEURONX_DEBUG("Stream destruction completed", "stream_id=", stream_id);
}

std::future<OperationContextResult> StreamImpl::SubmitOperationContext(
    std::unique_ptr<OperationContext> operation) {
  TORCH_NEURONX_STREAM_DEBUG("Submitting operation context", "stream_id=", stream_id,
                             "op=", operation->GetOpName());
  operation->stream = this;
  std::string operation_name = operation->GetOpName();
  operation->MarkSubmitted();

  OperationContext* op_ptr = operation.get();
  auto future = op_ptr->promise.get_future();

  {
    std::lock_guard<std::mutex> lock(active_operations_mutex_);
    active_operations_.push_back(std::move(operation));
    TORCH_NEURONX_STREAM_DEBUG("Added operation to active tracking", "stream_id=", stream_id,
                               "op=", operation_name, "total_active=", active_operations_.size());
  }

  if (op_ptr->RequiresCompilation()) {
    operation_execution_engine_->SubmitOperation(op_ptr);
  } else {
    op_ptr->MarkExecutionReady();
    operation_execution_engine_->NotifyExecutionReady();
  }
  TORCH_NEURONX_STREAM_DEBUG("Operation submitted successfully", "stream_id=", stream_id,
                             "op=", operation_name);
  return future;
}

bool StreamImpl::Query() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(active_operations_mutex_));
  return active_operations_.empty();
}

void StreamImpl::Synchronize() {
  TORCH_NEURONX_DEBUG("Synchronizing stream", "stream_id=", stream_id);

  if (Py_IsInitialized() && PyGILState_Check()) {
    // Release GIL before waiting so worker threads can complete tensor cleanup
    Py_BEGIN_ALLOW_THREADS std::unique_lock<std::mutex> lock(active_operations_mutex_);
    completion_cv.wait(lock, [this] {
      TORCH_NEURONX_STREAM_DEBUG("Waiting on synchronization, active_operations=",
                                 active_operations_.size());
      return active_operations_.empty();
    });
    Py_END_ALLOW_THREADS
  } else {
    std::unique_lock<std::mutex> lock(active_operations_mutex_);
    completion_cv.wait(lock, [this] {
      TORCH_NEURONX_STREAM_DEBUG("Waiting on synchronization, active_operations=",
                                 active_operations_.size());
      return active_operations_.empty();
    });
  }

  // After synchronization completes, check for and propagate any pending exceptions
  error_handler_->CheckAndThrowPendingException(stream_id, /* clear exception */ true);

  TORCH_NEURONX_DEBUG("Stream synchronization completed", "stream_id=", stream_id);
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
      TORCH_NEURONX_STREAM_DEBUG("Removing completed operation from active tracking",
                                 "stream_id=", stream_id, "operation=", op_name,
                                 "remaining_active=", active_operations_.size() - 1);
      operation_to_destroy = std::move(*it);
      active_operations_.erase(it);

      completion_cv.notify_all();
      TORCH_NEURONX_DEBUG("Operation removed and completion_cv notified", "stream_id=", stream_id,
                          "operation=", op_name, "new_active_count=", active_operations_.size());
    } else {
      TORCH_NEURONX_ERROR("Completed operation not found in active tracking",
                          "stream_id=", stream_id, "operation=", operation->GetOpName());
    }
  }
}

void StreamImpl::CompleteOperation(OperationContext* operation) {
  TORCH_NEURONX_DEBUG("Operation completion callback", "stream_id=", stream_id,
                      "operation=", operation->GetOpName());

  // Set execution end time
  operation->execute_end = std::chrono::steady_clock::now();

  TORCH_NEURONX_DEBUG("Operation completed successfully", "stream_id=", stream_id,
                      "op=", operation->GetOpName(),
                      "execution_time_us=", operation->GetExecutionTime().count(),
                      "total_time_us=", operation->GetTotalTime().count());

  operation->promise.set_value(OperationContextResult::CreateSuccess());

  RemoveActiveOperation(operation);
}

// Template implementation for handling Neuron runtime exceptions
template <typename ExceptionType>
void StreamImpl::HandleErrorWithCleanup(OperationContext* operation, const ExceptionType& e) {
  static_assert(torch_neuronx::is_neuron_runtime_exception_v<ExceptionType>,
                "ExceptionType must be CompilationRuntimeException or ExecutionRuntimeException");
  auto status = error_handler_->HandleOperationError(operation, e);

  if (status.IsSuccess()) {
    TORCH_NEURONX_DEBUG("Recovery succeeded, completing operation normally",
                        "stream_id=", stream_id, "op=", operation->GetOpName());
    CompleteOperation(operation);
  } else {
    operation->promise.set_value(status);
    RemoveActiveOperation(operation);
  }
}

// Explicit template instantiations
template void StreamImpl::HandleErrorWithCleanup<torch_neuronx::CompilationRuntimeException>(
    OperationContext*, const torch_neuronx::CompilationRuntimeException&);
template void StreamImpl::HandleErrorWithCleanup<torch_neuronx::ExecutionRuntimeException>(
    OperationContext*, const torch_neuronx::ExecutionRuntimeException&);

void StreamImpl::WaitForPriorOperationsToComplete(const OperationContext* operation) {
  TORCH_NEURONX_DEBUG("Waiting for prior operations to complete before CPU fallback",
                      "stream_id=", stream_id, "failed_op=", operation->GetOpName());
  {
    std::unique_lock<std::mutex> lock(active_operations_mutex_);

    bool is_at_front = !active_operations_.empty() && active_operations_.front().get() == operation;
    if (is_at_front) {
      return;
    }

    completion_cv.wait(lock, [this, operation]() {
      // Since active_operations_ is FIFO, we only need to check if the failed operation
      // is at the front of the queue. If it is, there are no prior operations.
      return active_operations_.empty() || active_operations_.front().get() == operation;
    });
  }
  error_handler_->CheckAndThrowPendingException(stream_id, /* clear exception */ false);

  TORCH_NEURONX_DEBUG(
      "All prior operations completed successfully, safe to proceed with CPU fallback",
      "stream_id=", stream_id, "failed_op=", operation->GetOpName());
}

OperationContext* StreamImpl::GetNextReadyOperation() {
  std::lock_guard<std::mutex> lock(active_operations_mutex_);

  if (active_operations_.empty()) {
    return nullptr;
  }

  auto& first_op = active_operations_.front();
  // Check if operation is compiled (if needed) and ready to execute
  if (first_op->IsSchedulable()) {
    return first_op.get();
  }
  return nullptr;
}

}  // namespace at::neuron
