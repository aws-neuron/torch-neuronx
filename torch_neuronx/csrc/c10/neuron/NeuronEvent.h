#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace at::neuron {

// Forward declarations
class NeuronStream;

// Internal event implementation (holds actual state)
struct EventImpl {
  mutable std::mutex sync_mutex;
  mutable std::condition_variable sync_cv;
  std::chrono::steady_clock::time_point record_time;
  std::atomic<bool> is_recorded{false};
  std::atomic<bool> is_completed{false};
  c10::DeviceIndex device_index = -1;
  int64_t recorded_stream_id = -1;
  bool enable_timing = false;
  bool blocking = false;
};

/// A Neuron event is a synchronization marker that can be used to monitor
/// device execution, accurately measure timing, and synchronize streams.
class NeuronEvent {
 public:
  /// @param enable_timing If true, enables high-precision timing measurements
  /// @param blocking If true, synchronization calls will block (unused currently)
  explicit NeuronEvent(bool enable_timing = false, bool blocking = false) noexcept
      : impl_(std::make_shared<EventImpl>()) {
    impl_->enable_timing = enable_timing;
    impl_->blocking = blocking;
  }

  // Copyable - creates another handle to the same event
  NeuronEvent(const NeuronEvent&) = default;
  NeuronEvent& operator=(const NeuronEvent&) = default;

  // Movable
  NeuronEvent(NeuronEvent&&) noexcept = default;
  NeuronEvent& operator=(NeuronEvent&&) noexcept = default;

  /// Record the event in a given stream
  void record(const NeuronStream& stream);

  /// Make all future work submitted to the given stream wait for this event
  void block(const NeuronStream& stream);

  /// Check if all work currently captured by event has completed
  bool query() const;

  /// Wait for the event to complete
  void synchronize() const;

  /// Return the time elapsed in milliseconds between this event and end_event
  float elapsed_time(const NeuronEvent& end_event) const;

  /// Check if the event was marked for timing
  bool is_timing_enabled() const { return impl_->enable_timing; }

  /// Get the device index where this event was recorded
  c10::DeviceIndex device_index() const { return impl_->device_index; }

  /// Complete the event (used by stream for async completion)
  void complete_event() const {
    mark_completed();
    impl_->sync_cv.notify_all();
  }

  /// Get the underlying shared_ptr
  std::shared_ptr<EventImpl> get_impl() const { return impl_; }

 private:
  std::shared_ptr<EventImpl> impl_;

  friend class NeuronStream;

  bool is_recorded() const noexcept { return impl_->is_recorded.load(std::memory_order_acquire); }
  void mark_recorded() const noexcept { impl_->is_recorded.store(true, std::memory_order_release); }
  bool is_completed() const noexcept { return impl_->is_completed.load(std::memory_order_acquire); }
  void mark_completed() const noexcept {
    impl_->is_completed.store(true, std::memory_order_release);
  }
};

// Forward declaration for getCurrentNeuronStream
NeuronStream getCurrentNeuronStream(c10::DeviceIndex device_index);

}  // namespace at::neuron
