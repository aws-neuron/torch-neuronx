#include "torch_neuronx/csrc/distributed/NeuronWork.h"

#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>

#include <sstream>
#include <thread>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace torch_neuronx {
namespace distributed {

// =============================================================================
// TensorShelf Implementation (thread-safe tensor container)
// =============================================================================

void TensorShelf::stash(const std::vector<at::Tensor>& tensors) {
  std::lock_guard<std::mutex> lock(mutex_);
  tensors_.insert(tensors_.end(), tensors.begin(), tensors.end());
}

void TensorShelf::unstash() { clear(); }

bool TensorShelf::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tensors_.empty();
}

void TensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tensors_.clear();
}

// =============================================================================
// Constructor
// =============================================================================

NeuronWork::NeuronWork(const std::string& pg_uid, const std::string& pg_desc, at::Device device,
                       int rank, const std::string& op_type, uint64_t seq_num,
                       std::vector<at::Tensor> outputs, bool enable_timing, float timeout_ms,
                       at::neuron::NeuronStream* stream, bool blocking_wait)
    : c10d::Work(rank, c10d::OpType::UNKNOWN),
      pg_uid_(pg_uid),
      pg_desc_(pg_desc),
      device_(device),
      rank_(rank),
      op_type_(op_type),
      seq_num_(seq_num),
      timing_enabled_(enable_timing),
      timeout_ms_(timeout_ms),
      work_start_time_(std::chrono::steady_clock::now()),
      start_event_(enable_timing),
      end_event_(enable_timing),
      stream_(stream != nullptr ? *stream : at::neuron::getCurrentNeuronStream(device.index())),
      outputs_(std::move(outputs)),
      stashed_for_allocator_safety_(std::make_unique<TensorShelf>()),
      blockingWait_(blocking_wait) {}

// =============================================================================
// Helper functions (independent - no dependencies on other NeuronWork methods)
// =============================================================================

std::string NeuronWork::logPrefix() const {
  std::ostringstream oss;
  oss << "[Rank " << rank_ << "] [PG " << pg_desc_ << "] [" << op_type_ << " Seq " << seq_num_
      << "] ";
  return oss.str();
}

void NeuronWork::stash(const std::vector<at::Tensor>& tensors) {
  // Thread-safe via TensorShelf's internal mutex
  stashed_for_allocator_safety_->stash(tensors);
}

void NeuronWork::unstashTensors() {
  std::lock_guard<std::mutex> lock(shelf_detach_mutex_);
  if (stashed_for_allocator_safety_ && !stashed_for_allocator_safety_->empty()) {
    stashed_for_allocator_safety_->unstash();
  }
}

void NeuronWork::checkAndSetException() {
  // TODO: Add Neuron-specific error checking when available
  // e.g., check if the stream/device has encountered errors
}

void NeuronWork::setException(std::exception_ptr exception_ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

void NeuronWork::markFutureWithError() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (future_ && !future_->completed() && exception_) {
    future_->setError(exception_);
  }
}

void NeuronWork::markFutureComplete() {
  if (future_ && !future_->completed()) {
    future_->markCompleted(at::IValue(outputs_));
  }
}

void NeuronWork::markFutureCompleteIfNeeded() {
  // Called by watchdog when work completes without wait() being called.
  if (isCompleted() && future_ && !future_->completed()) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (exception_) {
      // markFutureWithError() also locks, but that's fine - we can release and reacquire
      // Or just inline the logic here
      if (!future_->completed()) {
        future_->setError(exception_);
      }
    } else {
      if (!future_->completed()) {
        future_->markCompleted(at::IValue(outputs_));
      }
    }
  }
}

void NeuronWork::handleException() {
  if (!exception_) {
    return;
  }

  std::ostringstream oss;
  oss << logPrefix() << "Some Neuron collective operations have failed or timed out. "
      << "Due to the asynchronous nature of device execution, subsequent "
      << "operations might run on corrupted/incomplete data.";

  // Re-raise to avoid data inconsistency
  std::rethrow_exception(exception_);
}

// =============================================================================
// Primitive query/state functions
// =============================================================================

bool NeuronWork::isCompleted() {
  // TODO: we will eventually need this checkAndSetException();

  // Check exception first (requires mutex)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (exception_) {
      return true;
    }
  }

  // Query the end event directly (non-blocking).
  // Thread-safe without mutex because:
  // NeuronEvent::query() reads from std::atomic<bool> is_completed
  // Called from: main thread (wait()), watchdog thread (work iteration)
  return end_event_.query();
}

bool NeuronWork::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "NeuronWork::isSuccess() is deprecated");
}

// exception() uses base class c10d::Work::exception() - no override needed

bool NeuronWork::isStarted() const {
  // TODO: Implement properly when timing support is added
  return true;
}

bool NeuronWork::checkTimeout(std::optional<std::chrono::milliseconds> timeout) {
  auto current_time = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - work_start_time_);
  auto work_timeout =
      timeout.value_or(std::chrono::milliseconds(static_cast<int64_t>(timeout_ms_)));

  if (elapsed < work_timeout) {
    return false;
  }

  // Timed out - create exception message
  std::ostringstream oss;
  oss << logPrefix() << "Watchdog caught collective operation timeout: " << op_type_
      << " (seq=" << seq_num_ << ") ran for " << elapsed.count()
      << " milliseconds before timing out.";

  std::string exception_msg = oss.str();
  LOG(ERROR) << exception_msg;

  std::exception_ptr exception_ptr = std::make_exception_ptr(std::runtime_error(exception_msg));
  setException(exception_ptr);

  // This ensures code waiting on getFuture() sees the timeout error
  markFutureWithError();

  return true;
}

std::vector<at::Tensor> NeuronWork::result() { return outputs_; }

c10::intrusive_ptr<c10::ivalue::Future> NeuronWork::getFuture() { return future_; }

float NeuronWork::getDuration() const {
  TORCH_CHECK(timing_enabled_, "getDuration() only works if timing was enabled");
  return start_event_.elapsed_time(end_event_);
}

// =============================================================================
// Event recording functions
// =============================================================================

void NeuronWork::recordStartEvent(at::neuron::NeuronStream* stream) {
  // TODO(timing): Profiling support - not yet implemented.
  // When implementing, ensure start/end events are recorded on the same stream
  if (!timing_enabled_) {
    return;
  }
  at::neuron::NeuronStream& record_stream = (stream != nullptr) ? *stream : stream_;
  start_event_.record(record_stream);
}

void NeuronWork::recordEndEvent(at::neuron::NeuronStream* stream) {
  at::neuron::NeuronStream& record_stream = (stream != nullptr) ? *stream : stream_;
  end_event_.record(record_stream);

  // The MultiStreamGuard sets the current stream context so that when
  // markCompleted() is called, the Future records an event on this stream.
  // When user code later calls future.wait() or future.value(), PyTorch's
  // Future::synchronizeWithCurrentStreams() will make the user's current
  // stream wait for our collective's stream via event.block().
  {
    c10::Stream c10_stream = record_stream.unwrap();
    std::vector<c10::Stream> streams{c10_stream};
    c10::MultiStreamGuard streamGuard(streams);

    std::vector<at::Device> devices{device_};
    future_ = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    future_->markCompleted(at::IValue(outputs_));
  }
}

void NeuronWork::registerWithTensors() {
  // Register this work with output tensors for wait_tensor() support.
  if (c10d::allow_inflight_collective_as_graph_input()) {
    auto work_ptr = c10::intrusive_ptr<NeuronWork>::unsafe_reclaim_from_nonowning(this);
    for (const auto& tensor : outputs_) {
      c10d::register_work(tensor, work_ptr);
    }
  }
}

// =============================================================================
// Synchronization functions (called by wait())
// =============================================================================

void NeuronWork::synchronize() {
  auto current_stream = at::neuron::getCurrentNeuronStream(device_.index());
  end_event_.block(current_stream);

  // Clear stashed tensors directly - safe because synchronize() is called from
  // wait() on the main thread with GIL. Must hold shelf_detach_mutex_ to avoid
  // racing with watchdog's detachStashedTensorShelf() which can move the shelf.
  {
    std::lock_guard<std::mutex> lock(shelf_detach_mutex_);
    if (stashed_for_allocator_safety_) {
      stashed_for_allocator_safety_->unstash();
    }
  }

  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(c10::intrusive_ptr<NeuronWork>::unsafe_reclaim_from_nonowning(this));
  }
}

void NeuronWork::abort() {
  // Abort pattern matches NCCL's WorkNCCL::abort().
  //
  // Called from wait() AFTER synchronize() has already done cleanup:
  // - unstashTensors() already called
  // - unregister_work() already called
  //
  // This abort() only does:
  // 1. Force-complete the event (unblocks OTHER threads waiting on isCompleted())
  // 2. Set exception if not already set
  // 3. Mark future with error
  //
  // Note: This is a "soft" abort - the underlying collective on device
  // may continue running, but all CPU-side cleanup is done.

  LOG(WARNING) << logPrefix() << "Aborting work due to timeout";

  // Force-complete the event to unblock waiters
  // complete_event() sets is_completed=true and notifies the CV
  end_event_.complete_event();

  // If exception not already set, set it now
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!exception_) {
      std::ostringstream oss;
      oss << logPrefix() << "Operation aborted due to timeout after " << timeout_ms_
          << " milliseconds.";
      exception_ = std::make_exception_ptr(std::runtime_error(oss.str()));
    }
  }

  // Mark future with error if not already completed
  markFutureWithError();
}

// =============================================================================
// Main c10d::Work interface - wait() (depends on synchronize, abort, etc.)
// =============================================================================

bool NeuronWork::wait(std::chrono::milliseconds timeout) {
  // Make current stream wait for collective to complete (non-blocking to CPU)
  synchronize();

  // In case of blockingWait_ or a timeout value is specified by the user, we
  // block the CPU thread until the work is completed or timed out.
  if (blockingWait_ || timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timed_out =
          checkTimeout(timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      if (timed_out) {
        LOG(ERROR) << logPrefix() << "Work timed out in wait() - breaking out of wait loop";
        break;
      }
      // Yield
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } else if (op_type_ == "barrier" && !isCompleted()) {
    // This is to minimize the CPU barrier wait time in healthy path
    at::neuron::getCurrentNeuronStream(device_.index()).synchronize();
  }

  // If exception is detected, abort and throw from the main CPU thread
  if (exception()) {
    // Abort to unblock any pending operations and cleanup
    abort();
    // Throw exception (from main thread here)
    handleException();  // This will rethrow the exception
  }

  return true;
}
}  // namespace distributed
}  // namespace torch_neuronx
