#include "NeuronEvent.h"

#include <stdexcept>

#include "NeuronStream.h"

namespace at::neuron {

void NeuronEvent::moveHelper(NeuronEvent&& other) noexcept {
  // Move state
  enable_timing_ = other.enable_timing_;
  blocking_ = other.blocking_;
  device_index_ = other.device_index_;
  record_time_ = other.record_time_;
  recorded_stream_id_ = other.recorded_stream_id_;

  // Use proper memory ordering for atomics
  is_recorded_.store(other.is_recorded_.load(std::memory_order_acquire), std::memory_order_release);
  is_completed_.store(other.is_completed_.load(std::memory_order_acquire),
                      std::memory_order_release);

  // Reset other object
  other.enable_timing_ = false;
  other.blocking_ = false;
  other.device_index_ = -1;
  other.record_time_ = {};
  other.recorded_stream_id_ = -1;
  other.is_recorded_.store(false, std::memory_order_release);
  other.is_completed_.store(false, std::memory_order_release);
}

void NeuronEvent::record() {
  try {
    record(getCurrentNeuronStream(device_index_ >= 0 ? device_index_ : -1));
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to record event on current stream: ", e.what());
  }
}

void NeuronEvent::record(const NeuronStream& stream) {
  device_index_ = stream.device_index();
  recorded_stream_id_ = stream.id();
  mark_recorded();
  is_completed_.store(false, std::memory_order_release);

  if (enable_timing_) {
    record_time_ = std::chrono::steady_clock::now();
  }

  if (stream.query()) {
    // Stream is idle - event completes immediately
    mark_completed();
    sync_cv_.notify_all();
  } else {
    // Stream has work - register event to complete after current operations
    stream.register_event_after_current_operations(this);
  }
}

void NeuronEvent::block(const NeuronStream& stream) { stream.wait_event(*this); }

bool NeuronEvent::query() const { return !is_recorded() || is_completed(); }

void NeuronEvent::synchronize() const {
  if (!is_recorded()) {
    return;
  }

  try {
    std::unique_lock<std::mutex> lock(sync_mutex_);
    sync_cv_.wait(lock, [this] { return is_completed(); });
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

  const auto duration = end_event.record_time_ - record_time_;
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  return static_cast<float>(ns) / 1'000'000.0f;
}

}  // namespace at::neuron
