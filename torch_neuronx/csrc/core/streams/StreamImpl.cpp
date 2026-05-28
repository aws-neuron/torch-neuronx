#include "StreamImpl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>
#include <torch/extension.h>

#include <thread>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/ProfilerMappingCollector.h"
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

extern "C" {
#include <nrt/nrt.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <mutex>

// Stream configuration constants
namespace StreamConfig {
// Timeout configurations
constexpr auto kOperationTimeout = std::chrono::milliseconds(30000);

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

// Definition of ProfilerMappingCollector's static atomic unique event id
std::atomic<uint64_t> ProfilerMappingCollector::next_event_id_{1};

StreamImpl::StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id)
    : device_index(device_idx),
      stream_id(id),
      next_to_schedule_(active_operations_.end()),
      operation_timeout_(StreamConfig::kOperationTimeout) {
  int dev_idx = static_cast<int>(device_index);
  int str_id = static_cast<int>(stream_id);
  TORCH_NEURONX_STREAM_DEBUG("Creating StreamImpl", "device_index=", dev_idx, "stream_id=", str_id);

  // Initialize per-stream ErrorHandler
  error_handler_ = std::make_unique<ErrorHandler>();

  TORCH_NEURONX_DEBUG("StreamImpl fully initialized", "stream_id=", str_id,
                      "device_index=", dev_idx);
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

std::shared_future<OperationContextResult> StreamImpl::SubmitOperationContext(
    std::unique_ptr<OperationContext> operation) {
  TORCH_NEURONX_STREAM_DEBUG("Submitting operation context", "stream_id=", stream_id,
                             "op=", operation->GetOpName());
  operation->stream = this;
  std::string operation_name = operation->GetOpName();
  operation->MarkSubmitted();

  OperationContext* op_ptr = operation.get();
  auto future = op_ptr->result_future;

  {
    // Queue operations in the same order to prevent inconsistencies.
    std::lock_guard<std::mutex> lock(active_operations_mutex_);
    bool was_empty = (next_to_schedule_ == active_operations_.end());
    active_operations_.push_back(std::move(operation));
    // If next_to_schedule_ was at end (no ops to schedule), point to the new op
    if (was_empty) {
      next_to_schedule_ = std::prev(active_operations_.end());
    }
    TORCH_NEURONX_STREAM_DEBUG("Added operation to active tracking", "stream_id=", stream_id,
                               "op=", operation_name, "total_active=", active_operations_.size());

    // Ensure the operation is prepared, in case if consists of a staging step.
    PrepareOperation(op_ptr);

    // Always route through SubmitOperation to ensure consistent queue depth tracking
    // Non-compilable operations will be handled specially in the concatenation/execution pipeline
    auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
    if (engine.IsConcatenationEnabled() || op_ptr->RequiresCompilation()) {
      engine.SubmitOperation(op_ptr);
    } else {
      op_ptr->MarkExecutionReady();
      engine.NotifyExecutionReady();
    }
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

  // Rethrow stashed exception from runtime thread
  {
    std::lock_guard<std::mutex> lock(stashed_exception_mutex_);
    if (stashed_exception_) {
      auto eptr = stashed_exception_;
      stashed_exception_ = nullptr;
      std::rethrow_exception(eptr);
    }
  }

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
      // If we're erasing the element that next_to_schedule_ points to, advance it first
      // to avoid iterator invalidation
      if (it == next_to_schedule_) {
        ++next_to_schedule_;
      }
      operation_to_destroy = std::move(*it);
      active_operations_.erase(it);

      ReleaseDeferredAllocGates();

      completion_cv.notify_all();
      TORCH_NEURONX_DEBUG("Operation removed and completion_cv notified", "stream_id=", stream_id,
                          "operation=", op_name, "new_active_count=", active_operations_.size());
    } else {
      TORCH_NEURONX_ERROR("Completed operation not found in active tracking",
                          "stream_id=", stream_id, "operation=", operation->GetOpName());
    }
  }
}

bool StreamImpl::IsStreamDrainedWithLockHeld() const {
  // True iff no op is scheduled or pending-signal. Deferred kAllocs are
  // parked, not draining, so a stream holding only Deferred kAllocs counts
  // as drained (otherwise the drain hook could never fire).
  for (const auto& op_ptr : active_operations_) {
    if (op_ptr->nrt_async_state_.is_scheduled || op_ptr->nrt_async_state_.is_pending_signal) {
      return false;
    }
    if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation() &&
        op_ptr->GetConcatenatedOperation()->nrt_async_state_.is_scheduled) {
      return false;
    }
  }
  return true;
}

void StreamImpl::ReleaseDeferredAllocGates() {
  // Short circuit if no parked deferred kAlloc on this stream.
  if (deferred_alloc_count_.load(std::memory_order_relaxed) == 0) {
    return;
  }
  if (!IsStreamDrainedWithLockHeld()) {
    return;
  }
  // Stream drained: flip any deferred kAlloc to RetryReady and rewind
  // next_to_schedule_ to the earliest such kAlloc so the scheduler retries
  // it on the next pass.
  auto earliest_retry = active_operations_.end();
  for (auto it = active_operations_.begin(); it != active_operations_.end(); ++it) {
    auto& op_ptr = *it;
    if (op_ptr->GetKernelType() == KernelTypeEnum::kAlloc) {
      auto& alloc_kernel = op_ptr->GetKernel<KernelTypeEnum::kAlloc>();
      if (alloc_kernel.GetAllocState() ==
          at::neuron::AllocDirectKernelExecution::AllocState::Deferred) {
        alloc_kernel.SetAllocState(at::neuron::AllocDirectKernelExecution::AllocState::RetryReady);
        deferred_alloc_count_.fetch_sub(1, std::memory_order_relaxed);
        if (earliest_retry == active_operations_.end()) {
          earliest_retry = it;
        }
      }
    }
  }
  if (earliest_retry != active_operations_.end()) {
    next_to_schedule_ = earliest_retry;
    // Wake the runtime worker in case it is parked on completion_efd_.
    auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
    engine.NotifyExecutionReady();
  }
}

bool StreamImpl::TryDeferAlloc(AllocDirectKernelExecution& alloc_kernel) {
  std::lock_guard<std::mutex> lock(active_operations_mutex_);
  if (IsStreamDrainedWithLockHeld()) {
    return false;
  }
  alloc_kernel.SetAllocState(AllocDirectKernelExecution::AllocState::Deferred);
  deferred_alloc_count_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

namespace {
void RecordProfilerMapping(OperationContext* operation) {
  auto& collector = ProfilerMappingCollector::Instance();
  if (!collector.IsEnabled() || !operation->nrt_async_state_.is_scheduled) {
    return;
  }
  nrt::SequenceId seq_id = operation->nrt_async_state_.sequence_id;
  if (auto* concat_state = operation->GetConcatenationState()) {
    for (auto* cascading_op : concat_state->GetCascadingOperations()) {
      collector.Record(seq_id, cascading_op->pytorch_sequence_nr, cascading_op->pytorch_thread_id,
                       cascading_op->stream_id, operation->os_tid, cascading_op->event_id);
    }
  } else {
    collector.Record(seq_id, operation->pytorch_sequence_nr, operation->pytorch_thread_id,
                     operation->stream_id, operation->os_tid, operation->event_id);
  }
}
}  // anonymous namespace

void StreamImpl::CompleteOperation(OperationContext* operation) {
  TORCH_NEURONX_DEBUG("Operation completion callback", "stream_id=", stream_id,
                      "operation=", operation->GetOpName());

  // Decrement execution queue depth (skips concatenated ops internally)
  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  engine.DecrementExecutionQueueDepth(operation);

  // Set execution end time
  operation->execute_end = std::chrono::steady_clock::now();

  // Record operation snapshot for profiler if active
  if (auto* buffer = at::neuron::getActiveProfilerBuffer()) {
    at::neuron::ProfiledOperation snapshot;
    snapshot.op_name = operation->GetOpName();
    snapshot.submit_start_ns = at::neuron::toNs(operation->submit_start);
    snapshot.submit_time_ns = at::neuron::toNs(operation->submit_time);
    snapshot.execute_start_ns = at::neuron::toNs(operation->execute_start);
    snapshot.execute_end_ns = at::neuron::toNs(operation->execute_end);
    snapshot.nrt_sequence_id = operation->nrt_async_state_.sequence_id;
    snapshot.kineto_correlation_id = operation->kineto_correlation_id;
    snapshot.pytorch_thread_id = operation->pytorch_thread_id;
    snapshot.device_index = static_cast<int32_t>(device_index);
    snapshot.stream_id = static_cast<int32_t>(stream_id);
    snapshot.neuron_stream_os_tid = operation->neuron_stream_os_tid;
    snapshot.pytorch_os_tid = operation->os_tid;
    snapshot.dispatch_start_ns = at::neuron::toNs(operation->dispatch_start);
    snapshot.xu_type = operation->xu_type;
    snapshot.dispatch_slot = operation->dispatch_slot;
    snapshot.pytorch_sequence_nr = operation->pytorch_sequence_nr;
    snapshot.hierarchy_string = operation->hierarchy_string;
    if (operation->RequiresCompilation()) {
      auto& compilable_kernel = operation->GetCompilableKernel();
      snapshot.neff_cache_key = compilable_kernel.GetCacheKey();
      if (const auto* pk = compilable_kernel.GetCompilationCacheKey().GetPersistentKey()) {
        snapshot.persistent_cache_key = *pk;
      }
    }
    buffer->record(std::move(snapshot));
  }

  TORCH_NEURONX_DEBUG("Operation completed successfully", "stream_id=", stream_id,
                      "op=", operation->GetOpName(),
                      "execution_time_us=", operation->GetExecutionTime().count(),
                      "total_time_us=", operation->GetTotalTime().count());

  RecordProfilerMapping(operation);

  operation->promise.set_value(OperationContextResult::CreateSuccess());

  RemoveActiveOperation(operation);
}

// Template implementation for handling Neuron runtime exceptions
template <typename ExceptionType>
void StreamImpl::HandleErrorWithCleanup(OperationContext* operation, const ExceptionType& e) {
  static_assert(torch_neuronx::is_neuron_runtime_exception_v<ExceptionType>,
                "ExceptionType must be CompilationRuntimeException or ExecutionRuntimeException");

  bool is_concatenated = operation->IsConcatenatedOperation();

  OperationContextResult status;
  try {
    status = error_handler_->HandleOperationError(operation, e);
  } catch (...) {
    HandleErrorWithExceptionPtr(operation, std::current_exception());
    return;
  }

  if (status.IsSuccess()) {
    TORCH_NEURONX_DEBUG("Recovery succeeded", "stream_id=", stream_id,
                        "op=", operation->GetOpName());

    if (!is_concatenated) {
      CompleteOperation(operation);
    }
  } else {
    // Error path (no recovery) - decrement queue depth since CompleteOperation won't be called
    auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
    engine.DecrementExecutionQueueDepth(operation);
    operation->promise.set_value(status);
    RemoveActiveOperation(operation);
  }
}

// Explicit template instantiations
template void StreamImpl::HandleErrorWithCleanup<torch_neuronx::CompilationRuntimeException>(
    OperationContext*, const torch_neuronx::CompilationRuntimeException&);
template void StreamImpl::HandleErrorWithCleanup<torch_neuronx::ExecutionRuntimeException>(
    OperationContext*, const torch_neuronx::ExecutionRuntimeException&);

void StreamImpl::HandleErrorWithExceptionPtr(OperationContext* operation, std::exception_ptr eptr) {
  // Stash exception for rethrow on Synchronize() — don't call .what() here
  StashException(eptr);
  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  engine.DecrementExecutionQueueDepth(operation);
  operation->promise.set_value(
      OperationContextResult::CreateError("Operation failed on runtime thread"));
  RemoveActiveOperation(operation);
}

OperationContext* StreamImpl::GetNextOperationToSchedule() {
  std::lock_guard<std::mutex> lock(active_operations_mutex_);

  if (next_to_schedule_ == active_operations_.end()) {
    return nullptr;
  }

  // Pure FIFO: park if the head isn't ready. A Deferred kAlloc acts as an
  // in-band synchronize — preceding ops drain it (via ReleaseDeferredAllocGates)
  // and subsequent ops wait for the retry to resolve.
  auto& op = *next_to_schedule_;
  if (op->IsSchedulable()) {
    return op.get();
  }

  if (op->GetKernelType() == KernelTypeEnum::kEvent) {
    auto& event_kernel = op->GetKernel<KernelTypeEnum::kEvent>();
    if (event_kernel.GetEventAction() == EventDirectKernelExecution::EventAction::kWait &&
        event_kernel.GetSignalStreamId() == stream_id) {
      return op.get();
    }
  }
  return nullptr;
}

void StreamImpl::AdvanceToNextOperation(OperationContext* just_scheduled) {
  std::lock_guard<std::mutex> lock(active_operations_mutex_);

  if (next_to_schedule_ == active_operations_.end()) {
    return;
  }

  if (just_scheduled != nullptr && next_to_schedule_->get() != just_scheduled) {
    return;
  }
  ++next_to_schedule_;
}

bool StreamImpl::HasPendingAsyncOperations() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(active_operations_mutex_));

  if (active_operations_.empty()) {
    return false;
  }

  auto& front_op = active_operations_.front();
  // Check if front op is directly scheduled or pending
  if (front_op->nrt_async_state_.is_scheduled || front_op->nrt_async_state_.is_pending_signal) {
    return true;
  }
  // Check if front op is a cascading op whose concat op is scheduled
  if (front_op->HasConcatenatedOperation() && !front_op->IsConcatenatedOperation()) {
    auto* concat_op = front_op->GetConcatenatedOperation();
    return concat_op->nrt_async_state_.is_scheduled;
  }
  return false;
}

OperationContext* StreamImpl::FindOperationBySeq(nrt::SequenceId sequence_id) {
  std::lock_guard<std::mutex> lock(active_operations_mutex_);
  for (auto& op_ptr : active_operations_) {
    // Check regular scheduled ops
    if (op_ptr->nrt_async_state_.is_scheduled &&
        op_ptr->nrt_async_state_.sequence_id == sequence_id) {
      return op_ptr.get();
    }
    // Check if this is a cascading op whose concat op matches the sequence ID
    // This is needed because concat ops are not stored directly in active_operations_
    // - they are owned by ConcatenationState, and only cascading ops are tracked here
    if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation()) {
      auto* concat_op = op_ptr->GetConcatenatedOperation();
      if (concat_op->nrt_async_state_.is_scheduled &&
          concat_op->nrt_async_state_.sequence_id == sequence_id) {
        return concat_op;  // Return the concat op that failed
      }
    }
  }
  return nullptr;
}

void StreamImpl::UpdateCompletionState() {
  int vnc_id = c10_neuron::get_vnc_id(device_index);
  for (size_t xu = 0; xu < NRTA_XU_TYPE_NUM; ++xu) {
    auto xu_type = static_cast<nrta_xu_t>(xu);
    int num_queues = nrt::GetNumQueuesForXU(xu_type);
    for (int queue = 0; queue < num_queues; ++queue) {
      nrt::SequenceId seq = 0;
      if (nrt::GetLastCompletedRequest(vnc_id, xu_type, queue, &seq) == NRT_SUCCESS) {
        xu_completed_seqs_.Set(xu_type, queue, seq);
      }
    }
  }
}

ScheduleResult StreamImpl::ScheduleDeviceOperation(OperationContext* op,
                                                   KernelTypeEnum kernel_type) {
  // What next_to_schedule_ points at — `op` itself, except cascading concat
  // ops where it's the cascading op. Used for the idempotent advance below.
  OperationContext* scheduled_op = op;

  // Handle concatenated operations: cascading ops schedule their concat op first time,
  // subsequent cascading ops just advance since concat op is already scheduled.
  // Device ops require stream-specific hardware state (completion tracking, sequence IDs,
  // cross-XU sync), so concatenation handling lives here rather than in the engine.
  if (op->HasConcatenatedOperation() && !op->IsConcatenatedOperation()) {
    auto* concat_op = op->GetConcatenatedOperation();

    // Concat op already scheduled by a previous cascading op, just advance this one
    if (concat_op->nrt_async_state_.is_scheduled) {
      AdvanceToNextOperation(scheduled_op);
      return ScheduleResult::kAdvanced;
    }

    if (!concat_op->GetCompilableKernel().HasCachedNeff()) {
      TORCH_NEURONX_WARN("Concatenated op has no compiled NEFF, scheduling individual op",
                         "concat_op:", concat_op->GetOpName(), "cascading_op:", op->GetOpName());
      op->concatenation_state_ = nullptr;
      kernel_type = op->GetKernelType();
    } else {
      // First cascading op needs to schedule the concat op - use concat op's kernel type
      op = concat_op;
      kernel_type = concat_op->GetKernelType();
    }
  }

  // Cross-XU synchronization: ensure prior XU operation completes before switching XUs
  auto current_xu = static_cast<nrta_xu_t>(GetXUIndex(kernel_type));
  if (last_scheduled_info_.valid && last_scheduled_info_.xu != current_xu) {
    UpdateCompletionState();
    int queue = nrt::GetHardwareQueue(last_scheduled_info_.xu, stream_id);
    nrt::SequenceId completed_seq = xu_completed_seqs_.Get(last_scheduled_info_.xu, queue);
    if (NRTA_SEQ_GET_SEQ_NUM(last_scheduled_info_.sequence_id) >
        NRTA_SEQ_GET_SEQ_NUM(completed_seq)) {
      return ScheduleResult::kNotReady;
    }
  }

  op->execute_start = std::chrono::steady_clock::now();
  op->neuron_stream_os_tid = getOsTid();
  nrt::SequenceId sequence_id = 0;
  bool scheduled = op->kernel_execution->Schedule(&op->nrt_status_, &sequence_id, stream_id);

  if (scheduled) {
    op->MarkScheduled(sequence_id);
    last_scheduled_info_.xu = current_xu;
    last_scheduled_info_.sequence_id = sequence_id;
    last_scheduled_info_.valid = true;
  } else {
    // No NRT work — mark as immediately complete
    op->nrt_status_ = NRT_SUCCESS;
    op->MarkScheduled(sequence_id);
  }
  AdvanceToNextOperation(scheduled_op);

  TORCH_NEURONX_DEBUG("Operation scheduled", "stream_id=", stream_id, "op=", op->GetOpName(),
                      "sequence_id=", sequence_id);
  return ScheduleResult::kScheduled;
}

void StreamImpl::PrepareOperation(OperationContext* op) {
  if (!op->RequiresPrepare()) {
    return;
  }
  op->kernel_execution->Prepare();
  TORCH_NEURONX_DEBUG("Operation prepared", "stream_id=", stream_id, "op=", op->GetOpName());
}

}  // namespace at::neuron
