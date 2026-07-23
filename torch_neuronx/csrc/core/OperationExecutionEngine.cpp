#include "OperationExecutionEngine.h"

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"

namespace at::neuron {

namespace {
// Release the device-after-kHost gate by advancing the stream's host-completion
// seq to this op's.
inline void AdvanceHostCompletion(OperationContext* op) {
  if (op == nullptr || op->stream == nullptr) {
    return;
  }
  op->stream->host_completion_state_.AdvanceCompleted(op->host_completion_seq_);
}
}  // namespace

HostWorkerPool::HostWorkerPool(size_t num_threads, CompletionCb on_complete)
    : on_complete_(std::move(on_complete)) {
  EXECUTION_RUNTIME_CHECK(num_threads > 0, "HostWorkerPool requires at least 1 thread");
  EXECUTION_RUNTIME_CHECK(static_cast<bool>(on_complete_),
                          "HostWorkerPool requires a non-null completion callback");
  workers_.reserve(num_threads);
  try {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this, i] { WorkerLoop(i); });
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
    throw;
  }
  TORCH_NEURONX_DEBUG("HostWorkerPool started", "num_threads=", num_threads);
}

HostWorkerPool::~HostWorkerPool() { Shutdown(); }

void HostWorkerPool::Submit(OperationContext* op) {
  EXECUTION_RUNTIME_CHECK(op != nullptr, "HostWorkerPool::Submit got null op");
  bool reject = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load(std::memory_order_acquire)) {
      reject = true;
    } else {
      jobs_.push_back(Job{op});
    }
  }
  if (reject) {
    // Shutting down — fail fast. Advance the stream's host-completion seq so
    // any device op gated on this kHost (RunHost will never run) unblocks
    // rather than hanging at Synchronize. Callback invoked outside the lock to
    // avoid lock-order inversion with engine state.
    AdvanceHostCompletion(op);
    auto eptr = std::make_exception_ptr(
        std::runtime_error("HostWorkerPool is shutting down; rejecting submission"));
    on_complete_(op, eptr);
    return;
  }
  cv_.notify_one();
}

void HostWorkerPool::Shutdown() {
  // Single-shot guard: only the first caller does the join+drain. CAS outside
  // the lock so repeat callers bail without contending.
  bool expected = false;
  if (!shutdown_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return;
  }
  // LOST-WAKEUP: re-store shutdown_ and notify_all under mutex_ so a worker
  // mid-cv_.wait() predicate eval can't miss the notify.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  // Drain remaining jobs: surface a shutdown exception (RunHost never runs) and
  // advance the host-completion seq so any gated device op unblocks.
  std::deque<Job> leftover;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    leftover.swap(jobs_);
  }
  for (auto& job : leftover) {
    AdvanceHostCompletion(job.op);
    auto eptr =
        std::make_exception_ptr(std::runtime_error("HostWorkerPool shut down before job ran"));
    on_complete_(job.op, eptr);
  }
  TORCH_NEURONX_DEBUG("HostWorkerPool shutdown complete", "drained=", leftover.size());
}

void HostWorkerPool::WorkerLoop(size_t worker_idx) {
  TORCH_NEURONX_DEBUG("HostWorkerPool worker started", "worker_idx=", worker_idx);
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock,
               [this] { return !jobs_.empty() || shutdown_.load(std::memory_order_acquire); });
      if (jobs_.empty() && shutdown_.load(std::memory_order_acquire)) {
        break;
      }
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }

    std::exception_ptr eptr = nullptr;
    try {
      EXECUTION_RUNTIME_CHECK(job.op != nullptr, "HostWorkerPool got null op in queue");
      EXECUTION_RUNTIME_CHECK(job.op->kernel_execution != nullptr,
                              "HostWorkerPool op has null kernel_execution");
      EXECUTION_RUNTIME_CHECK(job.op->GetKernelType() == KernelTypeEnum::kHost,
                              "HostWorkerPool got non-kHost op");

      // Stamp execute_start (profiler timing) and worker_idx (routes the DRIVER
      // activity to a per-worker pftrace track).
      job.op->execute_start = std::chrono::steady_clock::now();
      job.op->neuron_stream_os_tid = getOsTid();
      job.op->host_worker_idx = static_cast<int32_t>(worker_idx);

      auto* host_kernel = static_cast<HostKernelExecution*>(job.op->kernel_execution.get());
      host_kernel->RunHost();
    } catch (...) {
      eptr = std::current_exception();
    }

    // Advance the seq BEFORE on_complete_: the release store must happen-before
    // the queue push / worker wake, so a device op that observes the seq also
    // sees the kHost's output writes.
    try {
      AdvanceHostCompletion(job.op);
      on_complete_(job.op, eptr);
    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("HostWorkerPool: completion drain threw despite invariants: ", e.what(),
                          " — op may leak; downstream device ops gated on the host completion",
                          " seq may hang at Synchronize.");
    } catch (...) {
      TORCH_NEURONX_ERROR("HostWorkerPool: completion drain threw unknown exception",
                          " despite invariants — op may leak.");
    }
  }
  TORCH_NEURONX_DEBUG("HostWorkerPool worker exiting", "worker_idx=", worker_idx);
}

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

  // Start the runtime worker thread. The host worker pool is created lazily on
  // first kHost dispatch (EnsureHostWorkerPool), not here.
  try {
    // EFD_NONBLOCK so the worker can fully drain the counter before parking:
    // completion_efd_ is a coalescing counter shared by submission writes
    // (NotifyExecutionReady) and NRT completion signals. A single blocking read
    // drains an unknown number of accumulated wakes at once, so a coalesced read
    // could consume the wake meant for a still-pending op and the worker would
    // re-park forever (missed-wakeup deadlock seen at training steady state).
    completion_efd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    TORCH_CHECK(completion_efd_ >= 0, "Failed to create eventfd for runtime worker");

    runtime_thread_ = std::thread(&OperationExecutionEngine::RuntimeWorker, this);
  } catch (...) {
    shutdown_requested_.store(true, std::memory_order_release);
    if (host_worker_pool_) {
      host_worker_pool_->Shutdown();
      // Drain entries the pool's Shutdown pushed before reset (empty in practice
      // — no kHost ops can be submitted during ctor — but symmetric with Shutdown).
      DrainHostCompletions();
      host_worker_pool_.reset();
    }
    if (completion_efd_ >= 0) {
      ::close(completion_efd_);
      completion_efd_ = -1;
    }
    if (concatenation_enabled_) {
      concatenation_cv_.notify_all();
      if (concatenation_thread_.joinable()) concatenation_thread_.join();
    }
    throw;
  }

  TORCH_NEURONX_DEBUG("OperationExecutionEngine worker threads started",
                      "concatenation_enabled=", concatenation_enabled_, "host_pool=lazy",
                      "runtime_thread_id=", runtime_thread_.get_id());
}

OperationExecutionEngine::~OperationExecutionEngine() { Shutdown(); }

void OperationExecutionEngine::EnsureHostWorkerPool() {
  std::lock_guard<std::mutex> lock(host_worker_pool_init_mutex_);
  if (host_worker_pool_) {
    return;
  }
  host_worker_pool_ = std::make_unique<HostWorkerPool>(
      kHostWorkerThreads,
      [this](OperationContext* op, std::exception_ptr eptr) { OnHostJobComplete(op, eptr); });
  TORCH_NEURONX_DEBUG("Lazily initialized HostWorkerPool", "threads=", kHostWorkerThreads);
}

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
  WriteEventfd();

  if (concatenation_enabled_ && concatenation_thread_.joinable()) {
    concatenation_thread_.join();
  }
  if (runtime_thread_.joinable()) {
    runtime_thread_.join();
  }

  // Tear down the host worker pool AFTER the runtime thread has joined so any
  // kHost ops it scheduled have a chance to run. Pool::Shutdown() waits for
  // workers to exit and surfaces a shutdown error for any unrun jobs.
  if (host_worker_pool_) {
    host_worker_pool_->Shutdown();
    // Drain any completion entries the pool pushed while we were tearing
    // down — workers may have completed jobs after RuntimeWorker exited.
    DrainHostCompletions();
    host_worker_pool_.reset();
  }

  if (completion_efd_ >= 0) {
    ::close(completion_efd_);
    completion_efd_ = -1;
  }

  TORCH_NEURONX_DEBUG("OperationExecutionEngine shutdown complete");
}

bool OperationExecutionEngine::CompileNextPending() {
  OperationContext* operation = nullptr;
  {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    if (compilation_queue_.empty()) {
      return false;
    }
    operation = compilation_queue_.front();
    compilation_queue_.pop_front();
  }
  ProcessPrepareTask(operation);
  return true;
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
    operation->MarkExecutionReady();
  } catch (...) {
    stream->HandleErrorWithExceptionPtr(operation, std::current_exception());
  }
  NotifyExecutionReady();
}

void OperationExecutionEngine::ProcessPreprocessingTask(OperationContext* operation) {
  // Validate operation before accessing any internal state
  EXECUTION_RUNTIME_CHECK(operation != nullptr, "Received null operation");
  EXECUTION_RUNTIME_CHECK(operation->kernel_execution != nullptr,
                          "Operation has null kernel_execution");

  const std::string& op_name = operation->GetOpName();
  TORCH_NEURONX_DEBUG("Processing preprocessing task", "op=", op_name);
  c10_neuron::lazy::LazyTransformationManager::ProcessOperationInputs(operation);
  // Epilogue: merge output transformations into the operation MLIR (when enabled).
  // Safe to call unconditionally — ProcessOperationOutputs early-exits if no
  // output transforms are set (which is always the case when epilogue is OFF,
  // since DetectEpilogueTransforms returns empty and SetOutputTransformations
  // is never called).
  c10_neuron::lazy::LazyTransformationManager::ProcessOperationOutputs(operation);
}

void OperationExecutionEngine::HandleCompilationComplete(OperationContext* operation) {
  auto* stream = operation->stream;
  const std::string& op_name = operation->GetOpName();

  if (!operation->GetCompilableKernel().HasCachedNeff()) {
    operation->MarkExecutionReady();
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
      concat_op->stream->HandleErrorWithCleanup(concat_op, e);
    } catch (...) {
      concat_op->stream->HandleErrorWithExceptionPtr(concat_op, std::current_exception());
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
    compilation_cache_->GetOrCompileNeffAsync(
        kernel, stream_id, [this, operation](bool success, const std::string& error) {
          if (success) {
            HandleCompilationComplete(operation);
          } else {
            // Stash error on kernel for the execution worker to report.
            if (!error.empty()) {
              operation->GetCompilableKernel().SetCompilationError(error);
            }
            operation->MarkExecutionReady();
          }
          NotifyExecutionReady();
        });
    return;
  }

  // No service: compile directly on this thread
  compilation_cache_->GetOrCompileNeff(kernel, stream_id);
  HandleCompilationComplete(operation);
}

void OperationExecutionEngine::RuntimeWorker() {
  TORCH_NEURONX_DEBUG("Runtime worker thread started");
  auto devices = c10_neuron::get_visible_device_indices();

  while (true) {
    // Check shutdown at top of loop
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      break;
    }

    bool work_found;
    do {
      work_found = false;

      CheckCompletions();

      // Drain kHost completions right after CheckCompletions() so they flow
      // through the same FIFO pass as device completions — a device op blocked
      // on a kHost becomes schedulable next pass.
      {
        bool host_had_work;
        {
          std::lock_guard<std::mutex> lock(host_completed_mutex_);
          host_had_work = !host_completed_queue_.empty();
        }
        if (host_had_work) {
          DrainHostCompletions();
          work_found = true;
        }
      }

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

      if (CompileNextPending()) {
        work_found = true;
      }
    } while (work_found && !shutdown_requested_.load(std::memory_order_acquire));

    if (shutdown_requested_.load(std::memory_order_acquire)) {
      break;
    }

    if (HasPendingAsyncOperations()) {
      RegisterCompletionEvents();
    }

    // Block until NRT signals completion or new work arrives. completion_efd_ is
    // a coalescing counter (EFD_NONBLOCK) shared by submission writes and NRT
    // completion signals, so a single read drains an unknown number of wakes.
    // Drain it fully and only actually park (poll) when it is truly empty: a read
    // that returns a count means something happened — fall through to re-scan
    // rather than risk consuming a coalesced wake meant for a still-pending op.
    // Any WriteEventfd racing the check-then-park window leaves the counter > 0,
    // so the read returns it (no park) and the worker re-scans.
    auto sleep_start = std::chrono::steady_clock::now();
    uint64_t count;
    ssize_t r = ::read(completion_efd_, &count, sizeof(count));
    if (r == -1 && errno == EAGAIN) {
      // Counter empty: nothing pending right now. Park on poll() until a write
      // arrives, then loop back to re-scan + re-drain. Use a bounded timeout as
      // a safety net: NRT device-completion signals are registered one-shot per
      // stream (RegisterSeqCompletion's last_registered_efd_seq_ guard), so a
      // completion that fires for an already-registered seq, or any signal that
      // races/coalesces against the check-then-park window, could otherwise be
      // lost and hang an op FIFO-blocked behind it forever. Waking periodically
      // routes back through CheckCompletions() (which polls NRT state directly,
      // independent of the eventfd), bounding worst-case wake latency to the
      // timeout instead of deadlocking. The eventfd remains the fast path.
      static constexpr int kRuntimeWorkerPollTimeoutMs = 50;
      struct pollfd pfd {
        completion_efd_, POLLIN, 0
      };
      while (::poll(&pfd, 1, kRuntimeWorkerPollTimeoutMs) == -1) {
        if (errno == EINTR) continue;
        if (shutdown_requested_.load(std::memory_order_acquire)) break;
        TORCH_NEURONX_ERROR("poll on completion_efd_ failed: ", strerror(errno));
        break;
      }
    } else if (r == -1 && errno != EINTR && !shutdown_requested_.load(std::memory_order_acquire)) {
      TORCH_NEURONX_ERROR("read from completion_efd_ failed: ", strerror(errno));
    }
    if (auto* buffer = at::neuron::getActiveProfilerBuffer()) {
      at::neuron::RuntimeSleepEvent sleep_evt;
      sleep_evt.start_ns = at::neuron::toNs(sleep_start);
      sleep_evt.end_ns = at::neuron::toNs(std::chrono::steady_clock::now());
      buffer->recordSleep(sleep_evt);
    }
  }
  TORCH_NEURONX_DEBUG("Runtime worker thread terminated");
}

void OperationExecutionEngine::RegisterCompletionEvents() {
  // Register eventfd on the front-of-queue scheduled op per stream. The
  // inflight FIFO is the engine's per-stream runtime_queue_.
  auto devices = c10_neuron::get_visible_device_indices();
  for (auto device : devices) {
    ForEachStreamImpl(device, [&](StreamImpl* stream_impl) -> bool {
      if (!stream_impl->HasPendingAsyncOperations()) {
        return false;
      }
      std::lock_guard<std::mutex> qlock(stream_impl->runtime_queue_mutex_);
      for (auto& operation_context : stream_impl->runtime_queue_) {
        OperationContext* op_ptr = operation_context.get();
        if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation()) {
          auto* concat_op = op_ptr->GetConcatenatedOperation();
          if (concat_op->nrt_async_state_.is_scheduled) {
            RegisterSeqCompletion(stream_impl, concat_op->nrt_async_state_.sequence_id);
            break;
          }
        }
        if (op_ptr->nrt_async_state_.is_scheduled) {
          RegisterSeqCompletion(stream_impl, op_ptr->nrt_async_state_.sequence_id);
          break;
        }
        if (op_ptr->nrt_async_state_.is_pending_signal) {
          continue;  // skip this op, continue walking
        }
        break;  // not scheduled / pending — stop
      }
      return false;
    });
  }
}

void OperationExecutionEngine::RegisterSeqCompletion(StreamImpl* stream, nrt::SequenceId seq) {
  if (seq == stream->last_registered_efd_seq_) return;
  stream->last_registered_efd_seq_ = seq;
  NRT_STATUS status = nrt::RegisterSeqIdCompletion(seq, completion_efd_);
  if (status != NRT_SUCCESS) {
    TORCH_NEURONX_WARN("RegisterSeqIdCompletion failed, status=", status, " seq=", seq);
    WriteEventfd();
  }
}

void OperationExecutionEngine::WriteEventfd() {
  uint64_t val = 1;
  while (::write(completion_efd_, &val, sizeof(val)) == -1) {
    if (errno == EINTR) continue;
    TORCH_NEURONX_ERROR("write to completion_efd_ failed: ", strerror(errno));
    break;
  }
}

void OperationExecutionEngine::OnHostJobComplete(OperationContext* op,
                                                 std::exception_ptr eptr) noexcept {
  // Called from a HostWorkerPool worker thread: push to queue + wake the runtime
  // worker, which drains and routes through StreamImpl's completion paths.
  //
  // INVARIANT: must be noexcept. If a throw escapes, the op never reaches
  // host_completed_queue_, pending_async_ops_count_ is never decremented, and
  // Synchronize() hangs forever. The catch is the fallback: route the op
  // directly through HandleErrorWithExceptionPtr and decrement the counter, so
  // the op leaves active_operations_ even if the queue push (mutex/bad_alloc) fails.
  try {
    {
      std::lock_guard<std::mutex> lock(host_completed_mutex_);
      host_completed_queue_.emplace_back(op, eptr);
    }
    WriteEventfd();
  } catch (...) {
    // Fallback mirrors DrainHostCompletions. We are noexcept, so every leaf call
    // that can throw (e.g. TORCH_NEURONX_ERROR's ostringstream → bad_alloc) is
    // wrapped to swallow — a throw here → std::terminate, worse than a leak.
    // fetch_sub is noexcept by spec.
    int prior = pending_async_ops_count_.fetch_sub(1);
    if (prior <= 0) {
      // Underflow guard symmetric with DrainHostCompletions, but we cannot
      // throw here (noexcept). Try to log; swallow if logging itself throws.
      try {
        TORCH_NEURONX_WARN("OnHostJobComplete fallback: pending_async_ops_count_ underflow ",
                           "(prior=", prior, "); kHost increment/decrement asymmetry.");
      } catch (...) {  // swallow — noexcept invariant
      }
    }
    if (op && op->stream) {
      auto fallback_eptr = eptr ? eptr : std::current_exception();
      try {
        op->stream->HandleErrorWithExceptionPtr(op, fallback_eptr);
      } catch (...) {
        // Last resort: log (wrapped — noexcept) and accept the leak over a hang.
        try {
          TORCH_NEURONX_ERROR(
              "OnHostJobComplete fallback HandleErrorWithExceptionPtr threw — op may leak.");
        } catch (...) {  // swallow — noexcept invariant
        }
      }
    } else {
      try {
        TORCH_NEURONX_ERROR("OnHostJobComplete fallback: op has null stream — op leaked.");
      } catch (...) {  // swallow — noexcept invariant
      }
    }
  }
}

void OperationExecutionEngine::DrainHostCompletions() {
  // Move out under the lock — completion processing touches the stream's
  // active_operations_ which has its own mutex; we don't want nested locks.
  std::deque<std::pair<OperationContext*, std::exception_ptr>> drained;
  {
    std::lock_guard<std::mutex> lock(host_completed_mutex_);
    drained.swap(host_completed_queue_);
  }

  for (auto& entry : drained) {
    OperationContext* op = entry.first;
    std::exception_ptr eptr = entry.second;
    if (!op) continue;
    StreamImpl* stream = op->stream;
    EXECUTION_RUNTIME_CHECK(stream != nullptr, "DrainHostCompletions: kHost op has null stream");

    // Decrement the counter ScheduleOperation's host path incremented; a
    // non-positive prior means an increment/decrement asymmetry (loud, not silent).
    int prior = pending_async_ops_count_.fetch_sub(1);
    EXECUTION_RUNTIME_CHECK(prior > 0,
                            "DrainHostCompletions: pending_async_ops_count_ underflow "
                            "(prior=",
                            prior, "); kHost increment/decrement asymmetry");

    if (eptr) {
      stream->HandleErrorWithExceptionPtr(op, eptr);
    } else {
      try {
        stream->CompleteOperation(op);
      } catch (...) {
        stream->HandleErrorWithExceptionPtr(op, std::current_exception());
      }
    }
  }
}

void OperationExecutionEngine::SubmitOperation(
    std::shared_ptr<OperationContext> operation_context) {
  if (shutdown_requested_.load()) {
    TORCH_NEURONX_DEBUG("Rejecting operation submission during shutdown",
                        "op=", operation_context->GetOpName());
    return;
  }
  EXECUTION_RUNTIME_CHECK(
      operation_context->stream != nullptr,
      "SubmitOperation: operation has null stream, op_name=", operation_context->GetOpName());

  if (concatenation_enabled_) {
    // Concat-ON: park on dispatch_queue_; ConcatenationWorker reorders the
    // buffered window and transfers a partitioned prefix into
    // stream->runtime_queue_ at flush.
    {
      std::lock_guard<std::mutex> lock(dispatch_mutex_);
      dispatch_queue_.push_back(std::move(operation_context));
    }
    concatenation_cv_.notify_one();
    return;
  }

  // Two invariants: capture RequiresCompilation() before the move (op_ptr
  // becomes UAF-prone for kEvent kWait via the same-stream bypass), and
  // sequence every eventfd write strictly AFTER EmitToRuntimeQueue (else the
  // worker can drain the wake-up and re-park before the op is visible).
  OperationContext* op_ptr = operation_context.get();
  const bool requires_compilation = op_ptr->RequiresCompilation();
  if (!requires_compilation) {
    op_ptr->MarkExecutionReady();
  }
  EmitToRuntimeQueue(std::move(operation_context));
  if (requires_compilation) {
    SubmitCompilationTask(op_ptr);
  } else {
    NotifyExecutionReady();
  }
}

void OperationExecutionEngine::RecordHint(std::unique_ptr<OperationContext> hint,
                                          const NeuronStream& stream) {
  if (shutdown_requested_.load()) return;
  if (!concatenation_enabled_) return;

  // Hints need a valid StreamImpl* for the concatenation engine's stream-change detection.
  hint->stream = GetDefaultStreamImpl(stream.device_index());

  std::shared_ptr<OperationContext> shared = std::move(hint);
  {
    std::lock_guard<std::mutex> lock(dispatch_mutex_);
    dispatch_queue_.push_back(std::move(shared));
  }
  concatenation_cv_.notify_one();
}

void OperationExecutionEngine::SubmitCompilationTask(OperationContext* operation) {
  if (shutdown_requested_.load()) {
    TORCH_NEURONX_DEBUG("Rejecting compilation task submission during shutdown",
                        "op=", operation->GetOpName());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    compilation_queue_.push_back(operation);
  }
  WriteEventfd();
}

void OperationExecutionEngine::EmitToRuntimeQueue(
    std::shared_ptr<OperationContext> operation_context) {
  StreamImpl* stream = operation_context->stream;
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  bool was_empty = (stream->next_to_schedule_ == stream->runtime_queue_.end());
  stream->runtime_queue_.push_back(std::move(operation_context));
  if (was_empty) {
    stream->next_to_schedule_ = std::prev(stream->runtime_queue_.end());
  }
}

void OperationExecutionEngine::MoveToRuntimeQueueInOrder(
    const std::list<OperationContext*>& partitioned_raws,
    std::unordered_map<OperationContext*, std::shared_ptr<OperationContext>>& owners) {
  for (OperationContext* raw : partitioned_raws) {
    auto it = owners.find(raw);
    EXECUTION_RUNTIME_CHECK(
        it != owners.end(),
        "MoveToRuntimeQueueInOrder: op missing from owners map. op_name=", raw->GetOpName());
    if (raw->GetKernelType() == KernelTypeEnum::kHint) {
      // Hints don't enter runtime_queue_; releasing the shared_ptr destroys the op.
      owners.erase(it);
      continue;
    }
    EmitToRuntimeQueue(std::move(it->second));
    owners.erase(it);
  }
}

void OperationExecutionEngine::TestInjectIntoRuntimeQueue(StreamImpl* stream,
                                                          std::shared_ptr<OperationContext> op) {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  bool was_empty = (stream->next_to_schedule_ == stream->runtime_queue_.end());
  stream->runtime_queue_.push_back(std::move(op));
  if (was_empty) {
    stream->next_to_schedule_ = std::prev(stream->runtime_queue_.end());
  }
}

void OperationExecutionEngine::TestClearRuntimeQueue(StreamImpl* stream) {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  stream->runtime_queue_.clear();
  stream->next_to_schedule_ = stream->runtime_queue_.end();
}

size_t OperationExecutionEngine::TestRuntimeQueueSize(const StreamImpl* stream) const {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  return stream->runtime_queue_.size();
}

bool OperationExecutionEngine::TestRuntimeCursorAtEnd(const StreamImpl* stream) const {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  return stream->next_to_schedule_ == stream->runtime_queue_.end();
}

bool OperationExecutionEngine::TestRuntimeCursorAt(const StreamImpl* stream,
                                                   const OperationContext* op) const {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  if (stream->next_to_schedule_ == stream->runtime_queue_.end()) return false;
  return stream->next_to_schedule_->get() == op;
}

bool OperationExecutionEngine::TestSetRuntimeCursorTo(StreamImpl* stream,
                                                      const OperationContext* op) {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  auto qit = std::find_if(stream->runtime_queue_.begin(), stream->runtime_queue_.end(),
                          [op](const auto& p) { return p.get() == op; });
  if (qit == stream->runtime_queue_.end()) return false;
  stream->next_to_schedule_ = qit;
  return true;
}

OperationContext* OperationExecutionEngine::TestRuntimeQueueFront(const StreamImpl* stream) const {
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  return stream->runtime_queue_.empty() ? nullptr : stream->runtime_queue_.front().get();
}

std::vector<OperationContext*> OperationExecutionEngine::TestRuntimeQueueSnapshot(
    const StreamImpl* stream) const {
  std::vector<OperationContext*> out;
  std::lock_guard<std::mutex> lock(stream->runtime_queue_mutex_);
  for (auto& sp : stream->runtime_queue_) out.push_back(sp.get());
  return out;
}

bool OperationExecutionEngine::IsStreamDrained(const StreamImpl* stream) const {
  // drain_and_move holds dispatch_mutex_ across the pop + push, so the op is
  // visible in exactly one of these two scans.
  {
    std::lock_guard<std::mutex> dlock(dispatch_mutex_);
    for (const auto& op : dispatch_queue_) {
      if (op->stream == stream) return false;
    }
  }
  std::lock_guard<std::mutex> slock(stream->runtime_queue_mutex_);
  return stream->runtime_queue_.empty();
}

void OperationExecutionEngine::ConcatenationWorker() {
  TORCH_NEURONX_DEBUG("Concatenation worker thread started");

  // op's shared_ptr lives in stream->runtime_queue_ for the duration of this
  // call; RuntimeWorker won't schedule it until MarkExecutionReady() below.
  auto submit_compile_or_ready = [this](OperationContext* op) {
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

  // Drain `count` shared_ptrs off the front of dispatch_queue_ and move them
  // onto each owner's stream->runtime_queue_ in `partitioned_raws` order.
  // dispatch_mutex_ is held across both halves so IsStreamDrained's two-phase
  // scan stays atomic.
  auto drain_and_move = [this](size_t count, const std::list<OperationContext*>& partitioned_raws) {
    if (count == 0) return;
    std::unordered_map<OperationContext*, std::shared_ptr<OperationContext>> owners;
    owners.reserve(count);
    std::lock_guard<std::mutex> dlock(dispatch_mutex_);
    for (size_t i = 0; i < count && !dispatch_queue_.empty(); ++i) {
      auto sp = std::move(dispatch_queue_.front());
      dispatch_queue_.pop_front();
      OperationContext* raw = sp.get();
      owners.emplace(raw, std::move(sp));
    }
    MoveToRuntimeQueueInOrder(partitioned_raws, owners);
  };

  while (true) {
    // result_raws collects the partitioned, ready-to-commit raw pointers
    // returned by ConcatenationEngine for this iteration.
    std::list<OperationContext*> result_raws;
    OperationContext* new_op = nullptr;
    bool needs_flush = false;
    bool shutdown = false;

    // Wait for new work or flush trigger; identify the next not-yet-buffered
    // op (if any) at the head of the unconsumed suffix.
    {
      std::unique_lock<std::mutex> lock(dispatch_mutex_);

      concatenation_cv_.wait(lock, [this] {
        // Wake when:
        //  1) dispatch_queue_ holds at least one not-yet-buffered op; OR
        //  2) the engine has buffered ops and queue depth dropped low; OR
        //  3) shutdown was requested.
        size_t buffered = concatenation_engine_->GetBufferedOperationsCount();
        bool has_new = dispatch_queue_.size() > buffered;
        bool low_depth_with_buffer =
            (execution_queue_depth_.load() <= queue_low_threshold_) && (buffered > 0);
        return has_new || low_depth_with_buffer ||
               shutdown_requested_.load(std::memory_order_acquire);
      });

      shutdown = shutdown_requested_.load(std::memory_order_acquire);

      // Identify the next not-yet-buffered op without removing it: it lives
      // at offset == buffered_count() into dispatch_queue_.
      size_t buffered = concatenation_engine_->GetBufferedOperationsCount();
      if (dispatch_queue_.size() > buffered) {
        auto it = std::next(dispatch_queue_.begin(), buffered);
        new_op = it->get();
      }

      needs_flush = (execution_queue_depth_.load() <= queue_low_threshold_) || shutdown;

      if (shutdown && new_op == nullptr && buffered == 0) {
        // Nothing to drain and no new ops — exit.
        break;
      }
    }
    // Run flush + concat-engine processing outside dispatch_mutex_. The
    // engine's internal buffer is raw-pointer aliases into dispatch_queue_;
    // the worker is the sole consumer, so prefix pointers stay stable while
    // this thread doesn't mutate the queue.
    int64_t current_depth = execution_queue_depth_.load();

    if (needs_flush) {
      auto flushed = concatenation_engine_->Flush(SIZE_MAX);
      if (!flushed.empty()) {
        TORCH_NEURONX_DEBUG("[CONCATENATION_WORKER] Flush triggered",
                            "reason=", shutdown ? "shutdown" : "low_queue_depth",
                            "current_depth=", current_depth, "threshold=", queue_low_threshold_,
                            "flushed_count=", flushed.size());
      }
      result_raws.splice(result_raws.end(), flushed);
    }

    bool error_path = false;
    std::shared_ptr<OperationContext> failed_owner;
    std::string failed_what;

    if (new_op != nullptr) {
      EXECUTION_RUNTIME_CHECK(new_op->kernel_execution != nullptr,
                              "[CONCATENATION_WORKER] Operation has null kernel_execution");
      EXECUTION_RUNTIME_CHECK(
          new_op->stream != nullptr,
          "[CONCATENATION_WORKER] Operation has null stream field, op_name=", new_op->GetOpName(),
          " kernel_type=", static_cast<int>(new_op->GetKernelType()));

      TORCH_NEURONX_DEBUG("[CONCATENATION_WORKER] Processing task",
                          "stream_id=", new_op->stream->stream_id, "op_name=", new_op->GetOpName(),
                          "pytorch_seq_nr=", new_op->pytorch_sequence_nr);

      try {
        ProcessPreprocessingTask(new_op);
        if (result_raws.size() + current_depth <= static_cast<size_t>(queue_low_threshold_)) {
          // Skip concatenation when total queue depth is low — keep the
          // device fed.
          new_op->CompleteConcatenation();
          result_raws.push_back(new_op);
        } else {
          auto concat_ops = concatenation_engine_->ProcessConcatenationTask(new_op);
          result_raws.splice(result_raws.end(), concat_ops);
        }
      } catch (const std::exception& e) {
        TORCH_NEURONX_ERROR("[CONCATENATION_WORKER] Exception during preprocessing",
                            "error=", e.what());
        error_path = true;
        failed_what = e.what();

        // Pull new_op out of dispatch_queue_ so RemoveActiveOperation can find
        // it once HandleErrorWithCleanup runs.
        {
          std::lock_guard<std::mutex> dlock(dispatch_mutex_);
          auto it = std::find_if(dispatch_queue_.begin(), dispatch_queue_.end(),
                                 [new_op](const auto& sp) { return sp.get() == new_op; });
          if (it != dispatch_queue_.end()) {
            failed_owner = std::move(*it);
            dispatch_queue_.erase(it);
          }
        }

        // Drain anything the engine had buffered before new_op.
        auto flushed = concatenation_engine_->Flush(SIZE_MAX);
        result_raws.splice(result_raws.end(), flushed);
      }
    }

    // Snapshot non-hints before drain_and_move releases hint owners.
    if (!result_raws.empty()) {
      std::vector<OperationContext*> non_hint_ops;
      non_hint_ops.reserve(result_raws.size());
      for (auto* op : result_raws) {
        if (op->GetKernelType() != KernelTypeEnum::kHint) {
          non_hint_ops.push_back(op);
        }
      }
      drain_and_move(result_raws.size(), result_raws);

      for (auto* op : non_hint_ops) {
        submit_compile_or_ready(op);
      }
    }

    if (error_path && failed_owner) {
      OperationContext* op_ptr = failed_owner.get();
      EmitToRuntimeQueue(std::move(failed_owner));
      if (IsDeviceKernelType(op_ptr->GetKernelType())) {
        // Balances the DecrementExecutionQueueDepth in HandleErrorWithCleanup.
        execution_queue_depth_.fetch_add(1);
      }
      op_ptr->stream->HandleErrorWithCleanup(
          op_ptr, torch_neuronx::CompilationRuntimeException(failed_what));
    }
  }

  // Final shutdown flush.
  auto final_flush = concatenation_engine_->Flush(SIZE_MAX);
  if (!final_flush.empty()) {
    std::vector<OperationContext*> non_hint_ops;
    non_hint_ops.reserve(final_flush.size());
    for (auto* op : final_flush) {
      if (op->GetKernelType() != KernelTypeEnum::kHint) {
        non_hint_ops.push_back(op);
      }
    }
    drain_and_move(final_flush.size(), final_flush);
    for (auto* op : non_hint_ops) {
      submit_compile_or_ready(op);
    }
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

bool OperationExecutionEngine::ScheduleAllocOperation(OperationContext* op, StreamImpl* stream) {
  auto& alloc_kernel = op->GetKernel<KernelTypeEnum::kAlloc>();
  const bool is_retry = alloc_kernel.GetAllocState() ==
                        at::neuron::AllocDirectKernelExecution::AllocState::RetryReady;

  // Wall time captures the initial allocation, not retries.
  if (!is_retry) {
    op->execute_start = std::chrono::steady_clock::now();
    op->neuron_stream_os_tid = getOsTid();
  }

  NRT_STATUS alloc_status = NRT_SUCCESS;
  op->kernel_execution->Schedule(&alloc_status, &op->nrt_async_state_.sequence_id, op->stream_id);
  const int device_id = op->kernel_execution->GetDeviceId();

  if (alloc_status == NRT_RESOURCE) {
    const size_t alloc_size = alloc_kernel.GetSizeBytes();
    // RetryReady is sticky: a second NRT_RESOURCE escalates, never re-defers.
    if (is_retry) {
      metrics::RecordAllocDrainRetryFail(device_id, alloc_size, op->stream_id);
      throw torch_neuronx::ExecutionRuntimeException(
          "lazy::AllocBind: NRT_RESOURCE on retry after drain", NRT_RESOURCE);
    }
    if (stream->TryDeferAlloc(alloc_kernel)) {
      TORCH_NEURONX_DEBUG("TryDeferAlloc: requested_size=", alloc_size);
      metrics::RecordAllocDeferred(device_id, alloc_size, op->stream_id);
      return false;
    }
    throw torch_neuronx::ExecutionRuntimeException(
        "lazy::AllocBind: NRT_RESOURCE; no pending ops on stream to wait "
        "for (cannot defer)",
        NRT_RESOURCE);
  }

  if (is_retry) {
    TORCH_NEURONX_DEBUG("DrainRetryOk: requested_size=", alloc_kernel.GetSizeBytes());
    metrics::RecordAllocDrainRetryOk(device_id, alloc_kernel.GetSizeBytes(), op->stream_id);
  }
  op->nrt_status_ = NRT_SUCCESS;
  op->nrt_async_state_.is_pending_signal = true;
  pending_async_ops_count_.fetch_add(1);
  stream->AdvanceToNextOperation(op);
  return true;
}

bool OperationExecutionEngine::ScheduleOperation(OperationContext* op, StreamImpl* stream) {
  KernelTypeEnum kernel_type = op->GetKernelType();
  if (kernel_type == KernelTypeEnum::kAlloc) {
    return ScheduleAllocOperation(op, stream);
  }

  // ── kHost: cross-kind sync gate, then submit to HostWorkerPool ────────────
  // A kHost op may only start once any prior device op on this stream has
  // retired (completion polling has caught up). We don't gate on prior kHost
  // ops here — pool-internal FIFO + per-stream FIFO via active_operations_
  // already preserves ordering for kHost→kHost on the same stream.
  if (kernel_type == KernelTypeEnum::kHost) {
    auto& info = stream->GetLastScheduledInfo();
    // Authoritative gate (a): kHost waits for a prior device-XU op to retire.
    // Same predicate the runtime worker pre-checks in GetNextOperationToSchedule
    // (single source of truth); refresh_completion=true reads the freshest NRT
    // completion seqs. prior_ops_in_flight is irrelevant for kHost (only the
    // kEvent gate (d) consumes it), so pass false. On gate → runtime worker
    // retries next pass (woken by the NRT completion eventfd).
    if (stream->IsScheduleGatedByPriorOp(op, kernel_type, /*prior_ops_in_flight=*/false,
                                         /*refresh_completion=*/true)) {
      return false;
    }

    EnsureHostWorkerPool();
    EXECUTION_RUNTIME_CHECK(host_worker_pool_ != nullptr,
                            "Failed to lazily initialize HostWorkerPool");

    // Order matters: mark inflight + stamp the host-completion seq into
    // last_scheduled_info_ (so a following device op gates on it) + advance,
    // all BEFORE Submit — once submitted the worker may run RunHost on another
    // thread. is_host_inflight release-store pairs with the acquire-loads in
    // HasPendingAsyncOperations / ProcessCompletions.
    op->nrt_async_state_.is_host_inflight.store(true, std::memory_order_release);
    pending_async_ops_count_.fetch_add(1);
    op->host_completion_seq_ =
        stream->host_completion_state_.next_seq_id.fetch_add(1, std::memory_order_relaxed);
    info.kind = LastKind::kHost;
    info.host_completion_seq = op->host_completion_seq_;
    info.valid = true;
    stream->AdvanceToNextOperation();
    try {
      host_worker_pool_->Submit(op);
    } catch (...) {
      // Submit threw → roll back the bookkeeping (else a phantom inflight op),
      // and advance the seq so a downstream device op gated on this kHost (whose
      // RunHost will never run) doesn't hang. The Submit error surfaces via the
      // caller's HandleErrorWithExceptionPtr.
      op->nrt_async_state_.is_host_inflight.store(false, std::memory_order_release);
      pending_async_ops_count_.fetch_sub(1);
      AdvanceHostCompletion(op);
      throw;
    }
    return true;
  }

  // Host-side ops (events, barriers, hints, allocs, deallocs) don't use NRT device scheduling.
  bool is_pending_host_op = !IsDeviceKernelType(kernel_type);

  if (is_pending_host_op) {
    // Authoritative gate (d): a cross-stream event_wait must not run ahead of a
    // still-in-flight prior op on this stream. Same predicate the runtime worker
    // pre-checks (single source of truth). Same-stream kWait is redundant with
    // FIFO ordering and never gates. prior_ops_in_flight uses the front-op
    // in-flight check (HasPendingAsyncOperations); refresh_completion is
    // irrelevant for the kEvent gate (no seq compare), pass false.
    if (stream->IsScheduleGatedByPriorOp(
            op, kernel_type,
            /*prior_ops_in_flight=*/stream->HasPendingAsyncOperations(),
            /*refresh_completion=*/false)) {
      return false;
    }
    op->nrt_async_state_.is_pending_signal = true;
    pending_async_ops_count_.fetch_add(1);
    stream->AdvanceToNextOperation(op);
    return true;
  }

  // Device ops: XU inflight throttling and device scheduling
  EXECUTION_RUNTIME_CHECK(IsDeviceKernelType(kernel_type),
                          "ScheduleOperation reached device-op path with non-device kernel type: ",
                          static_cast<int>(kernel_type));
  EXECUTION_RUNTIME_CHECK(stream->device_index >= 0 && stream->device_index < kMaxDevices,
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
    op->xu_type = static_cast<int32_t>(xu_idx);
    // Assign a profiler dispatch slot from the bitmask (max 8 slots per XU)
    auto& bitmask = xu_slot_bitmask_[stream->device_index][xu_idx];
    for (int s = 0; s < std::min(static_cast<int>(max_inflight_per_xu_), 8); ++s) {
      if (!(bitmask & (1u << s))) {
        bitmask |= (1u << s);
        op->dispatch_slot = s;
        // Propagate slot to concat_op so ProcessCompletions can release it
        if (op->HasConcatenatedOperation() && !op->IsConcatenatedOperation()) {
          op->GetConcatenatedOperation()->dispatch_slot = s;
        }
        break;
      }
    }
    return true;
  }
  // kAdvanced: cascading op advanced (concat already scheduled). kNotReady: will retry.
  if (result == ScheduleResult::kAdvanced) {
    op->xu_type = static_cast<int32_t>(xu_idx);
    return true;
  }
  return false;
}

void OperationExecutionEngine::DispatchAsyncExecutionTask(OperationContext* operation) noexcept {
  auto* stream = operation->stream;
  const std::string operation_name = operation->GetOpName();

  try {
    // If the kernel is schedulable, requires compilation but does not have a
    // cached NEFF, it denotes a failed compilation. Propagate the exception for
    // recovery (e.g. CPU fallback). Ensure that prior ops complete first.
    if (operation->RequiresCompilation() && !operation->GetCompilableKernel().HasCachedNeff()) {
      if (!stream->HasPendingAsyncOperations()) {
        auto& kernel = operation->GetCompilableKernel();
        std::string error_msg = kernel.IsFailedCompilation()
                                    ? kernel.GetCompilationError()
                                    : "NEFF creation failed for operation: " + operation_name;
        stream->HandleErrorWithCleanup(operation,
                                       torch_neuronx::CompilationRuntimeException(error_msg));
      }
      return;
    }

    // Capture dispatch start for profiler on first attempt only — retries (e.g., event_wait
    // blocking on prior async ops) should not reset the start time. DISPATCH end = execute_end
    // (set in CompleteOperation), so the DISPATCH event shows the full wait + execution time.
    if (operation->dispatch_start.time_since_epoch().count() == 0) {
      operation->dispatch_start = std::chrono::steady_clock::now();
    }

    TORCH_NEURONX_DEBUG("Processing async execution task", "stream_id=", stream->stream_id,
                        "operation_name=", operation_name);
    bool scheduled = ScheduleOperation(operation, stream);

    if (scheduled) {
      torch_neuronx::logOperationExecuted(operation_name);
      // Also log the concatenated op name if this cascading op has one,
      // so it appears in the "Executed ops" report alongside individual ops
      if (operation->HasConcatenatedOperation() && !operation->IsConcatenatedOperation()) {
        auto* concat_op = operation->GetConcatenatedOperation();
        torch_neuronx::logOperationExecuted(concat_op->GetOpName());
      }
    }
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    if (operation->dispatch_slot >= 0 && operation->xu_type >= 0) {
      xu_slot_bitmask_[stream->device_index][operation->xu_type] &=
          ~(1u << operation->dispatch_slot);
    }
    stream->HandleErrorWithCleanup(operation, e);
  } catch (...) {
    if (operation->dispatch_slot >= 0 && operation->xu_type >= 0) {
      xu_slot_bitmask_[stream->device_index][operation->xu_type] &=
          ~(1u << operation->dispatch_slot);
    }
    stream->HandleErrorWithExceptionPtr(operation, std::current_exception());
  }
}

void OperationExecutionEngine::ProcessCompletions(StreamImpl* stream,
                                                  const XUCompletionState& completed_seqs) {
  // Queue is determined per-operation based on its XU type
  std::vector<OperationContext*> ops_to_complete;

  auto IsSeqCompleted = [&](nrt::SequenceId seq_id, KernelTypeEnum kernel_type) {
    size_t xu_idx = GetXUIndex(kernel_type);
    int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
    nrt::SequenceId completed_seq = completed_seqs.Get(static_cast<nrta_xu_t>(xu_idx), queue);
    return NRTA_SEQ_GET_SEQ_NUM(seq_id) <= NRTA_SEQ_GET_SEQ_NUM(completed_seq);
  };

  // Walk runtime_queue_ in submission order with front-of-queue early exit.
  {
    std::lock_guard<std::mutex> qlock(stream->runtime_queue_mutex_);
    for (auto& operation_context : stream->runtime_queue_) {
      OperationContext* op_ptr = operation_context.get();
      if (op_ptr->HasConcatenatedOperation() && !op_ptr->IsConcatenatedOperation()) {
        auto* concat_op = op_ptr->GetConcatenatedOperation();
        if (concat_op->nrt_async_state_.is_scheduled) {
          if (IsSeqCompleted(concat_op->nrt_async_state_.sequence_id, concat_op->GetKernelType())) {
            ops_to_complete.push_back(op_ptr);
            continue;
          }
        }
        break;  // concat not done yet — stop
      }
      if (op_ptr->nrt_async_state_.is_host_inflight.load(std::memory_order_acquire)) {
        // kHost: completion is the HostWorkerPool's responsibility, routed
        // through DrainHostCompletions. We must stop the FIFO walk here so
        // we don't complete a later op out of order.
        break;
      } else if (op_ptr->nrt_async_state_.is_scheduled) {
        if (IsSeqCompleted(op_ptr->nrt_async_state_.sequence_id, op_ptr->GetKernelType())) {
          ops_to_complete.push_back(op_ptr);
          continue;
        }
        break;
      }
      if (op_ptr->nrt_async_state_.is_pending_signal) {
        ops_to_complete.push_back(op_ptr);
        continue;
      }
      break;
    }
  }

  // Complete operations
  for (auto* op : ops_to_complete) {
    if (op->nrt_async_state_.is_pending_signal) {
      try {
        if (op->GetKernelType() != KernelTypeEnum::kAlloc) {
          op->execute_start = std::chrono::steady_clock::now();
          op->neuron_stream_os_tid = getOsTid();
          op->kernel_execution->Schedule(&op->nrt_status_, &op->nrt_async_state_.sequence_id,
                                         op->stream_id);
          op->nrt_status_ = NRT_SUCCESS;
        }
        stream->CompleteOperation(op);
      } catch (const torch_neuronx::ExecutionRuntimeException& e) {
        stream->HandleErrorWithCleanup(op, e);
      } catch (...) {
        stream->HandleErrorWithExceptionPtr(op, std::current_exception());
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
        EXECUTION_RUNTIME_CHECK(IsDeviceKernelType(concat_kernel_type),
                                "Concatenation operation is not a device kernel");
        size_t xu_idx = GetXUIndex(concat_kernel_type);
        xu_inflight_counts_[stream->device_index][xu_idx]--;
        if (concat_op->dispatch_slot >= 0) {
          xu_slot_bitmask_[stream->device_index][xu_idx] &= ~(1u << concat_op->dispatch_slot);
        }
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
      if (op->dispatch_slot >= 0) {
        xu_slot_bitmask_[stream->device_index][xu_idx] &= ~(1u << op->dispatch_slot);
      }

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
        ProcessCompletions(stream_impl, stream_impl->xu_completed_seqs_);
      }
      return false;
    });
  }
}

void OperationExecutionEngine::DecrementExecutionQueueDepth(OperationContext* operation) {
  EXECUTION_RUNTIME_CHECK(operation != nullptr, "Received null operation pointer");

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
  EXECUTION_RUNTIME_CHECK(old_depth > 0,
                          "[EXEC_QUEUE_DEPTH] execution_queue_depth went negative! "
                          "old_depth=",
                          old_depth, " new_depth=", new_depth);

  bool will_notify = (new_depth <= queue_low_threshold_);
  if (will_notify) {
    concatenation_cv_.notify_one();
  }
}

}  // namespace at::neuron
