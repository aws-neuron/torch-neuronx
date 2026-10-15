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
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/streams/StreamConfig.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"

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

std::string ResolvePersistentCacheKey(const CompilableKernelExecution& kernel,
                                      CompilationCache& compilation_cache) {
  // Kernels without source IR (NeffDirectKernelExecution) can't recompute their
  // key, so read it from the CacheEntry the compile step stored.
  if (kernel.GetHloBytes().empty()) {
    if (auto entry = compilation_cache.GetEntry(kernel.GetCompileKey())) {
      return entry->persistent_cache_key;
    }
    return {};
  }
  // Kernels with HLO bytes recompute the key directly (no cache lookup).
  if (const auto* pk = kernel.GetCompilationCacheKey().GetPersistentKey()) {
    return *pk;
  }
  return {};
}

StreamImpl::StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id)
    : device_index(device_idx),
      stream_id(id),
      operation_timeout_(StreamConfig::kOperationTimeout),
      next_to_schedule_(runtime_queue_.end()) {
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

  // Drain, not Synchronize: an unconsumed operation error would rethrow out of
  // this destructor and call std::terminate.
  Drain();

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

  PrepareOperation(op_ptr);

  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  std::shared_ptr<OperationContext> shared = std::move(operation);
  // engine.SubmitOperation routes internally: dispatch_queue_ when concat is
  // on (ConcatenationWorker reorders + transfers to runtime_queue_ at flush),
  // straight to stream->runtime_queue_ when concat is off.
  engine.SubmitOperation(std::move(shared));

  TORCH_NEURONX_STREAM_DEBUG("Operation submitted successfully", "stream_id=", stream_id,
                             "op=", operation_name);
  return future;
}

bool StreamImpl::Query() const {
  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  return engine.IsStreamDrained(this);
}

void StreamImpl::Drain() {
  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  auto is_complete = [this, &engine] { return engine.IsStreamDrained(this); };

  if (Py_IsInitialized() && PyGILState_Check()) {
    // Release GIL before waiting so worker threads can complete tensor cleanup
    Py_BEGIN_ALLOW_THREADS std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, is_complete);
    Py_END_ALLOW_THREADS
  } else {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, is_complete);
  }
}

void StreamImpl::Synchronize() {
  TORCH_NEURONX_DEBUG("Synchronizing stream", "stream_id=", stream_id);

  Drain();

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
  std::shared_ptr<OperationContext> to_destroy;
  {
    std::lock_guard<std::mutex> lock(runtime_queue_mutex_);
    auto it = std::find_if(runtime_queue_.begin(), runtime_queue_.end(),
                           [operation](const auto& sp) { return sp.get() == operation; });
    if (it == runtime_queue_.end()) {
      TORCH_NEURONX_ERROR("Completed operation not found in active tracking",
                          "stream_id=", stream_id, "operation=", operation->GetOpName());
      return;
    }
    if (it == next_to_schedule_) {
      ++next_to_schedule_;
    }
    if (operation->GetKernelType() == KernelTypeEnum::kBarrier &&
        operation->nrt_async_state_.is_pending_signal && pending_barrier_count_ != 0) {
      --pending_barrier_count_;
    }
    to_destroy = std::move(*it);
    runtime_queue_.erase(it);
    // The removal may have drained the last running op; release any parked
    // kAllocs so the scheduler can retry them.
    ReleaseDeferredAllocGates();
  }
  // completion_mutex_ doesn't guard the predicate state — it serializes the
  // wait/notify pair against Synchronize() so the notify can't slip between
  // a waiter's predicate check and its park.
  std::lock_guard<std::mutex> lock(completion_mutex_);
  completion_cv_.notify_all();
}

bool StreamImpl::IsStreamDrainedWithLockHeld() const {
  // True iff no op is scheduled or pending-signal. Deferred kAllocs are
  // parked, not draining, so a stream holding only Deferred kAllocs counts
  // as drained (otherwise the drain hook could never fire).
  for (const auto& op_ptr : runtime_queue_) {
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
  // Caller must hold runtime_queue_mutex_.
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
  auto earliest_retry = runtime_queue_.end();
  for (auto it = runtime_queue_.begin(); it != runtime_queue_.end(); ++it) {
    auto& op_ptr = *it;
    if (op_ptr->GetKernelType() == KernelTypeEnum::kAlloc) {
      auto& alloc_kernel = op_ptr->GetKernel<KernelTypeEnum::kAlloc>();
      if (alloc_kernel.GetAllocState() ==
          at::neuron::AllocDirectKernelExecution::AllocState::Deferred) {
        alloc_kernel.SetAllocState(at::neuron::AllocDirectKernelExecution::AllocState::RetryReady);
        deferred_alloc_count_.fetch_sub(1, std::memory_order_relaxed);
        if (earliest_retry == runtime_queue_.end()) {
          earliest_retry = it;
        }
      }
    }
  }
  if (earliest_retry != runtime_queue_.end()) {
    next_to_schedule_ = earliest_retry;
    // Wake the runtime worker in case it is parked on completion_efd_.
    auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
    engine.NotifyExecutionReady();
  }
}

bool StreamImpl::TryDeferAlloc(AllocDirectKernelExecution& alloc_kernel) {
  std::lock_guard<std::mutex> lock(runtime_queue_mutex_);
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
    snapshot.host_worker_idx = operation->host_worker_idx;
    snapshot.hierarchy_string = operation->hierarchy_string;
    if (operation->RequiresCompilation()) {
      auto& compilable_kernel = operation->GetCompilableKernel();
      snapshot.neff_cache_key = compilable_kernel.GetCompileKey();
      snapshot.persistent_cache_key = ResolvePersistentCacheKey(
          compilable_kernel, NeuronResourceManager::Instance().GetCompilationCache());
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
    // Flattened operations retain the caller's logical ID on the default implementation.
    auto* error_stream =
        stream_id == 0 && operation->stream_id != 0
            ? NeuronStream(operation->stream_id, device_index, NeuronStream::UNCHECKED)
                  .get_stream_impl()
            : this;
    status = error_stream->error_handler_->HandleOperationError(operation, e);
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
  auto* error_stream =
      stream_id == 0 && operation->stream_id != 0
          ? NeuronStream(operation->stream_id, device_index, NeuronStream::UNCHECKED)
                .get_stream_impl()
          : this;
  error_stream->StashException(eptr);
  auto& engine = NeuronResourceManager::Instance().GetOperationExecutionEngine();
  engine.DecrementExecutionQueueDepth(operation);
  operation->promise.set_value(
      OperationContextResult::CreateError("Operation failed on runtime thread"));
  RemoveActiveOperation(operation);
}

bool StreamImpl::IsScheduleGatedByPriorOp(OperationContext* op, KernelTypeEnum kernel_type,
                                          bool prior_ops_in_flight, bool refresh_completion) {
  // (a) kHost gated on a prior device-XU op retiring on this stream: the kHost's
  //     CPU reads depend on the device op's output.
  if (kernel_type == KernelTypeEnum::kHost && last_scheduled_info_.valid &&
      last_scheduled_info_.kind == LastKind::kDeviceXU) {
    if (refresh_completion) {
      UpdateCompletionState();
    }
    int queue = nrt::GetHardwareQueue(last_scheduled_info_.xu, stream_id);
    nrt::SequenceId completed_seq = xu_completed_seqs_.Get(last_scheduled_info_.xu, queue);
    if (NRTA_SEQ_GET_SEQ_NUM(last_scheduled_info_.sequence_id) >
        NRTA_SEQ_GET_SEQ_NUM(completed_seq)) {
      return true;
    }
  }
  // (b) Device-XU gated on a prior kHost op completing on this stream: RunHost()
  //     must finish so its CPU output buffer is visible to the device op.
  if (IsDeviceKernelType(kernel_type) && kernel_type != KernelTypeEnum::kAlloc &&
      last_scheduled_info_.valid && last_scheduled_info_.kind == LastKind::kHost) {
    HostCompletionSeq completed =
        host_completion_state_.completed_seq_id.load(std::memory_order_acquire);
    if (last_scheduled_info_.host_completion_seq > completed) {
      return true;
    }
  }
  // (c) Cross-XU sync: a device-XU op on a different XU than the prior device op
  //     must wait for that prior op to retire.
  if (IsDeviceKernelType(kernel_type) && kernel_type != KernelTypeEnum::kAlloc &&
      last_scheduled_info_.valid && last_scheduled_info_.kind == LastKind::kDeviceXU) {
    auto current_xu = static_cast<nrta_xu_t>(GetXUIndex(kernel_type));
    if (last_scheduled_info_.xu != current_xu) {
      if (refresh_completion) {
        UpdateCompletionState();
      }
      int queue = nrt::GetHardwareQueue(last_scheduled_info_.xu, stream_id);
      nrt::SequenceId completed_seq = xu_completed_seqs_.Get(last_scheduled_info_.xu, queue);
      if (NRTA_SEQ_GET_SEQ_NUM(last_scheduled_info_.sequence_id) >
          NRTA_SEQ_GET_SEQ_NUM(completed_seq)) {
        return true;
      }
    }
  }
  // (d) HostCC cross-stream waits retain the consumer drain.
  //     Without HostCC, ready producer events and eligible compute fences may run ahead.
  if (kernel_type == KernelTypeEnum::kEvent && prior_ops_in_flight) {
    auto& event_kernel = op->GetKernel<KernelTypeEnum::kEvent>();
    if (event_kernel.GetEventAction() == EventDirectKernelExecution::EventAction::kWait &&
        event_kernel.GetSignalStreamImpl() != this &&
        (config::IsHostCCEnabled() || (!event_kernel.IsReady() && GetComputeQueueFence(op) == 0))) {
      return true;
    }
  }

  return false;
}

OperationContext* StreamImpl::GetNextOperationToSchedule() {
  std::lock_guard<std::mutex> lock(runtime_queue_mutex_);

  if (next_to_schedule_ == runtime_queue_.end()) {
    return nullptr;
  }

  // Pure FIFO: park if the head isn't ready. A Deferred kAlloc acts as an
  // in-band synchronize — preceding ops drain it (via ReleaseDeferredAllocGates)
  // and subsequent ops wait for the retry to resolve.
  auto& op = *next_to_schedule_;
  // Wait bypasses can skip event completion, but must still wait for preparation.
  if (!op->IsExecutionReady()) {
    return nullptr;
  }
  if (!op->IsSchedulable()) {
    // Queue-ordered waits can run ahead while completion remains pending.
    if ((IsSameStreamWaitBypass(op.get(), this) || GetComputeQueueFence(op.get()) != 0) &&
        IsRunAheadMemorySafe(device_index)) {
      return op.get();
    }
    return nullptr;
  }

  // Pre-check the same sync gates ScheduleOperation will apply. Returning null
  // here lets the worker sleep on completion_efd_ instead of busy-retrying an op
  // that ScheduleOperation would only reject. This is the *same* predicate the
  // authoritative path uses (single source of truth) — passed
  // refresh_completion=false because the loop-top CheckCompletions() already
  // refreshed completion state this pass, and prior_ops_in_flight computed under
  // the lock we already hold.
  KernelTypeEnum kernel_type = op->GetKernelType();
  if (IsScheduleGatedByPriorOp(op.get(), kernel_type,
                               /*prior_ops_in_flight=*/FrontOpInFlightWithLockHeld(),
                               /*refresh_completion=*/false)) {
    return nullptr;
  }

  return op.get();
}

nrt::SequenceId StreamImpl::GetComputeQueueFence(OperationContext* op) const {
  // The signal publishes a sequence only after the producer enters its compute queue.
  // A nonzero result permits consumer compute to follow it on the same hardware queue.
  // Import this sequence so later host and cross-XU work also waits for producer completion.
  // Return zero when unavailable. Event retirement and memory checks remain in force.
  if (op->GetKernelType() != KernelTypeEnum::kEvent || !op->IsExecutionReady() ||
      pending_barrier_count_ != 0 || !last_scheduled_info_.valid ||
      last_scheduled_info_.kind != LastKind::kDeviceXU ||
      last_scheduled_info_.xu != NRTA_XU_COMPUTE) {
    return 0;
  }
  auto& kernel = op->GetKernel<KernelTypeEnum::kEvent>();
  const auto& event = kernel.GetEventImpl();
  auto* producer = kernel.GetSignalStreamImpl();
  if (kernel.GetEventAction() != EventDirectKernelExecution::EventAction::kWait ||
      !event->allow_queue_ordering || producer == nullptr ||
      producer->device_index != device_index ||
      nrt::GetHardwareQueue(NRTA_XU_COMPUTE, producer->stream_id) !=
          nrt::GetHardwareQueue(NRTA_XU_COMPUTE, stream_id)) {
    return 0;
  }
  return event->queued_compute_sequence.load(std::memory_order_acquire);
}

bool StreamImpl::IsSameStreamWaitBypass(OperationContext* op, const StreamImpl* consumer) {
  if (op->GetKernelType() != KernelTypeEnum::kEvent) {
    return false;
  }
  auto& event_kernel = op->GetKernel<KernelTypeEnum::kEvent>();
  if (event_kernel.GetEventAction() != EventDirectKernelExecution::EventAction::kWait) {
    return false;
  }
  // StreamImpl* identity, not c10::StreamId — the latter is counter-encoded
  // by MakeStreamId so multiple ids can share a pool-slot StreamImpl.
  return event_kernel.GetSignalStreamImpl() == consumer;
}

bool StreamImpl::IsRunAheadMemorySafe(c10::DeviceIndex device_index) {
  // Default-on; Set TORCH_NEURONX_ENABLE_BOUNDED_RUNAHEAD=0
  // (or "false") to disable and restore unbounded same-stream kWait bypass.
  static const bool kBoundedRunaheadEnabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_BOUNDED_RUNAHEAD");
    if (env == nullptr) {
      return true;
    }
    const std::string value(env);
    return value != "0" && value != "false";
  }();
  if (!kBoundedRunaheadEnabled) {
    return true;
  }

  // Bound run-ahead by free device HBM so prefetched kAlloc bindings never
  // starve the next allocation. The headroom auto-sizes per model as the
  // largest observed single binding + slack.
  constexpr int64_t kSlack = int64_t{2} << 30;          // fragmentation slack-2GB
  constexpr int64_t kOverheadBound = int64_t{6} << 30;  // non-allocator HBM upper bound-6GB
  const int64_t margin =
      static_cast<int64_t>(c10_neuron::NeuronCachingAllocator::lazy::LargestAllocBytes()) + kSlack;
  const int64_t reserved = at::neuron::metrics::DeviceMemoryStatsRegistry::Instance()
                               .GetMemoryStats(device_index)
                               .reserved_bytes.current;
  static std::atomic<int64_t> cached_capacity{0};
  const int64_t cap = cached_capacity.load(std::memory_order_relaxed);

  // Cheap path: when the allocator is far from the ceiling there is ample room
  // even with worst-case non-allocator overhead, so advance without an NRT
  // syscall.
  if (cap > 0 && cap - reserved >= margin + kOverheadBound) {
    return true;
  }
  // Near the ceiling (or capacity not yet learned): precise free-HBM check via
  // NRT, whose bytes_used also accounts for scratchpad/code/DMA the allocator
  // cannot see. A query failure falls through to FIFO (park).
  uint64_t bytes_limit = 0, bytes_used = 0;
  if (c10_neuron::get_vnc_memory_stats(device_index, &bytes_limit, &bytes_used)) {
    cached_capacity.store(static_cast<int64_t>(bytes_limit), std::memory_order_relaxed);
    if (static_cast<int64_t>(bytes_limit - bytes_used) > margin) {
      return true;
    }
  }
  return false;
}

void StreamImpl::AdvanceToNextOperation(OperationContext* just_scheduled) {
  std::lock_guard<std::mutex> lock(runtime_queue_mutex_);

  if (next_to_schedule_ == runtime_queue_.end()) {
    return;
  }

  if (just_scheduled != nullptr && next_to_schedule_->get() != just_scheduled) {
    return;
  }
  ++next_to_schedule_;
}

bool StreamImpl::FrontOpInFlightWithLockHeld() const {
  if (runtime_queue_.empty()) {
    return false;
  }

  auto& front_op = runtime_queue_.front();
  // Front op is in flight if directly scheduled, pending (events/barriers/
  // allocs/deallocs), or running on the host pool (incl. RDMA P2P batches).
  if (front_op->nrt_async_state_.is_scheduled || front_op->nrt_async_state_.is_pending_signal ||
      front_op->nrt_async_state_.is_host_inflight.load(std::memory_order_acquire)) {
    return true;
  }
  if (front_op->HasConcatenatedOperation() && !front_op->IsConcatenatedOperation()) {
    auto* concat_op = front_op->GetConcatenatedOperation();
    return concat_op->nrt_async_state_.is_scheduled;
  }
  return false;
}

bool StreamImpl::HasPendingAsyncOperations() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(runtime_queue_mutex_));
  return FrontOpInFlightWithLockHeld();
}

OperationContext* StreamImpl::FindOperationBySeq(nrt::SequenceId sequence_id) {
  std::lock_guard<std::mutex> lock(runtime_queue_mutex_);
  for (auto& op_ptr : runtime_queue_) {
    if (op_ptr->nrt_async_state_.is_scheduled &&
        op_ptr->nrt_async_state_.sequence_id == sequence_id) {
      return op_ptr.get();
    }
    if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation()) {
      auto* concat_op = op_ptr->GetConcatenatedOperation();
      if (concat_op->nrt_async_state_.is_scheduled &&
          concat_op->nrt_async_state_.sequence_id == sequence_id) {
        return concat_op;
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

  // The first cascading op we reach schedules the merged op; the rest advance.
  // Ordering is already canonical (allocs → cascading → deallocs) because the
  // ConcatenationWorker reorders the partition before transferring to
  // runtime_queue_, so every required AllocBind has run by the time the
  // first cascading op is scheduled.
  if (op->HasConcatenatedOperation() && !op->IsConcatenatedOperation()) {
    auto* concat_op = op->GetConcatenatedOperation();

    // Concat op already scheduled by a previous cascading op, just advance this one
    if (concat_op->nrt_async_state_.is_scheduled) {
      AdvanceToNextOperation(scheduled_op);
      return ScheduleResult::kAdvanced;
    }

    // Dispatch handles failed merged compilations before device scheduling.
    op = concat_op;
    kernel_type = concat_op->GetKernelType();
  }

  // Authoritative cross-kind / cross-XU sync gate. Same predicate the runtime
  // worker pre-checks in GetNextOperationToSchedule (single source of truth),
  // but here with refresh_completion=true so the freshest NRT completion seqs
  // are read before we commit to scheduling. For a device op only gates (b)
  // device-after-kHost and (c) cross-XU can fire; (a)/(d) are kHost/kEvent-only,
  // so prior_ops_in_flight is irrelevant and passed false.
  auto current_xu = static_cast<nrta_xu_t>(GetXUIndex(kernel_type));
  if (IsScheduleGatedByPriorOp(op, kernel_type, /*prior_ops_in_flight=*/false,
                               /*refresh_completion=*/true)) {
    return ScheduleResult::kNotReady;
  }

  op->execute_start = std::chrono::steady_clock::now();
  op->neuron_stream_os_tid = getOsTid();
  nrt::SequenceId sequence_id = 0;
  bool scheduled = op->kernel_execution->Schedule(&op->nrt_status_, &sequence_id, stream_id);

  if (scheduled) {
    op->MarkScheduled(sequence_id);
    last_scheduled_info_.kind = LastKind::kDeviceXU;
    last_scheduled_info_.xu = current_xu;
    last_scheduled_info_.sequence_id = sequence_id;
    last_scheduled_info_.host_completion_seq = 0;
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
