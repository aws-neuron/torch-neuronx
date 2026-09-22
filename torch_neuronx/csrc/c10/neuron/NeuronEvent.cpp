#include "NeuronEvent.h"

#include <stdexcept>

#include "NeuronStream.h"

namespace at::neuron {

void NeuronEvent::record(const NeuronStream& stream) {
  impl_->device_index = stream.device_index();
  impl_->recorded_stream_impl.store(stream.get_stream_impl(), std::memory_order_relaxed);

  if (impl_->enable_timing) {
    impl_->record_time = std::chrono::steady_clock::now();
  }

  bool stream_has_no_work = stream.query();
  {
    std::lock_guard<std::mutex> lk(impl_->sync_mutex);
    mark_recorded();
    impl_->is_completed.store(false, std::memory_order_release);
    if (stream_has_no_work) {
      mark_completed();
    }
  }
  if (stream_has_no_work) {
    impl_->sync_cv.notify_all();
  } else {
    // Stream has work - submit event signal primitive
    stream.SubmitEventSignal(*this);
  }
}

void NeuronEvent::block(const NeuronStream& stream) {
  if (query()) {
    // Event already completed, no need to wait
    return;
  }
  // Submit event wait primitive
  stream.SubmitEventWait(*this);
}

bool NeuronEvent::query() const { return !is_recorded() || is_completed(); }

void NeuronEvent::synchronize() const {
  if (!is_recorded()) {
    return;
  }

  try {
    std::unique_lock<std::mutex> lock(impl_->sync_mutex);
    impl_->sync_cv.wait(lock, [this] { return is_completed(); });
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Event synchronization failed: ", e.what());
  }
}

float NeuronEvent::elapsed_time(const NeuronEvent& end_event) const {
  TORCH_CHECK(is_timing_enabled(), "Event was not created with timing enabled");
  TORCH_CHECK(end_event.is_timing_enabled(), "End event was not created with timing enabled");
  TORCH_CHECK(is_recorded(), "Event has not been recorded");
  TORCH_CHECK(end_event.is_recorded(), "End event has not been recorded");
  TORCH_CHECK(device_index() == end_event.device_index(), "Events must be on the same device");

  // Synchronize both events
  this->synchronize();
  end_event.synchronize();

  const auto duration = end_event.impl_->record_time - impl_->record_time;
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  return static_cast<float>(ns) / 1'000'000.0f;
}

}  // namespace at::neuron
