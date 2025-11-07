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

/// A Neuron event is a synchronization marker that can be used to monitor
/// device execution, accurately measure timing, and synchronize streams.
class NeuronEvent {
 public:
  /// Default constructor
  NeuronEvent() noexcept = default;

  /// Constructor with timing and blocking options
  /// @param enable_timing If true, enables high-precision timing measurements
  /// @param blocking If true, synchronization calls will block (unused currently)
  explicit NeuronEvent(bool enable_timing, bool blocking = false) noexcept
      : enable_timing_(enable_timing), blocking_(blocking) {}

  NeuronEvent(const NeuronEvent&) = delete;
  NeuronEvent& operator=(const NeuronEvent&) = delete;
  NeuronEvent(NeuronEvent&& other) noexcept { moveHelper(std::move(other)); }
  NeuronEvent& operator=(NeuronEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  /// Record the event in the current stream
  void record();

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
  bool is_timing_enabled() const { return enable_timing_; }

  /// Get the device index where this event was recorded
  c10::DeviceIndex device_index() const { return device_index_; }

  /// Complete the event (used by stream for async completion)
  void complete_event() {
    mark_completed();
    sync_cv_.notify_all();
  }

 private:
  mutable std::mutex sync_mutex_;
  mutable std::condition_variable sync_cv_;

  std::chrono::steady_clock::time_point record_time_;

  std::atomic<bool> is_recorded_{false};
  std::atomic<bool> is_completed_{false};

  c10::DeviceIndex device_index_ = -1;
  int64_t recorded_stream_id_ = -1;

  bool enable_timing_ = false;
  bool blocking_ = false;

  friend class NeuronStream;

  void moveHelper(NeuronEvent&& other) noexcept;

  bool is_recorded() const noexcept { return is_recorded_.load(std::memory_order_acquire); }
  void mark_recorded() noexcept { is_recorded_.store(true, std::memory_order_release); }
  bool is_completed() const noexcept { return is_completed_.load(std::memory_order_acquire); }
  void mark_completed() noexcept { is_completed_.store(true, std::memory_order_release); }
};

// Forward declaration for getCurrentNeuronStream
NeuronStream getCurrentNeuronStream(c10::DeviceIndex device_index);

}  // namespace at::neuron
