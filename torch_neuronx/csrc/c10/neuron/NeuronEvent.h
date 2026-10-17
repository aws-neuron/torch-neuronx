#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace at::neuron {

// Forward declarations
class NeuronStream;
struct StreamImpl;

// Internal event implementation (holds actual state)
struct EventImpl {
  mutable std::mutex sync_mutex;
  mutable std::condition_variable sync_cv;
  // Timestamp idle records immediately and active records when they complete.
  std::chrono::steady_clock::time_point record_time;
  std::atomic<bool> is_recorded{false};
  std::atomic<bool> is_completed{false};
  c10::DeviceIndex device_index = -1;
  // Written by Python thread in record(), read by runtime worker. Pointer
  // identity is the canonical same-stream check (c10::StreamId is
  // counter-encoded; multiple ids can map to the same StreamImpl).
  std::atomic<StreamImpl*> recorded_stream_impl{nullptr};
  // Permit consumer compute after producer submission to the same hardware queue.
  // The signal publishes queued_compute_sequence only after the producer reaches that queue.
  // Software FIFO order and event completion stay unchanged. Enable only for streams=1, HostCC=0.
  bool allow_queue_ordering = false;
  std::atomic<uint64_t> queued_compute_sequence{0};
  bool enable_timing = false;
  bool blocking = false;

  bool query() const {
    return !is_recorded.load(std::memory_order_acquire) ||
           is_completed.load(std::memory_order_acquire);
  }

  void complete_event() {
    {
      std::lock_guard<std::mutex> lock(sync_mutex);
      if (enable_timing && !is_completed.load(std::memory_order_relaxed)) {
        record_time = std::chrono::steady_clock::now();
      }
      is_completed.store(true, std::memory_order_release);
    }
    sync_cv.notify_all();
  }
};

// Public copies share a handle; submitted operations retain one record.
struct EventHandle {
  explicit EventHandle(std::shared_ptr<EventImpl> record) : current_record(std::move(record)) {}
  // record() replaces this pointer; query(), capture(), and synchronize() copy it.
  // Protect those accesses through public aliases. Kernels retain a fixed record without this lock.
  mutable std::mutex mutex;
  std::shared_ptr<EventImpl> current_record;
};

/// A Neuron event is a synchronization marker that can be used to monitor
/// device execution, accurately measure timing, and synchronize streams.
class NeuronEvent {
 public:
  /// @param enable_timing If true, enables high-precision timing measurements
  /// @param blocking If true, synchronization calls will block (unused currently)
  explicit NeuronEvent(bool enable_timing = false, bool blocking = false) noexcept
      : NeuronEvent(std::make_shared<EventImpl>()) {
    auto record = get_impl();
    record->enable_timing = enable_timing;
    record->blocking = blocking;
  }

  // Copyable - creates another handle to the same event
  NeuronEvent(const NeuronEvent&) = default;
  NeuronEvent& operator=(const NeuronEvent&) = default;

  // Movable
  NeuronEvent(NeuronEvent&&) noexcept = default;
  NeuronEvent& operator=(NeuronEvent&&) noexcept = default;

  /// Record the event in a given stream
  void record(const NeuronStream& stream);

  // Submitted operations retain the record they observed.
  NeuronEvent capture() const { return NeuronEvent(get_impl()); }

  /// Make all future work submitted to the given stream wait for this event
  void block(const NeuronStream& stream);

  /// Check if all work currently captured by event has completed
  bool query() const;

  /// Wait for the event to complete
  void synchronize() const;

  /// Return the time elapsed in milliseconds between this event and end_event.
  /// Both events must have completed; call synchronize() first.
  float elapsed_time(const NeuronEvent& end_event) const;

  /// Check if the event was marked for timing
  bool is_timing_enabled() const { return get_impl()->enable_timing; }

  /// Get the device index where this event was recorded
  c10::DeviceIndex device_index() const { return get_impl()->device_index; }

  /// Stream that recorded this event, or nullptr if never recorded.
  StreamImpl* recorded_stream_impl() const {
    return get_impl()->recorded_stream_impl.load(std::memory_order_relaxed);
  }

  /// Complete the event (used by stream for async completion)
  void complete_event() const { get_impl()->complete_event(); }

  std::shared_ptr<EventImpl> get_impl() const {
    std::lock_guard<std::mutex> lock(handle_->mutex);
    return handle_->current_record;
  }

  // Python exposes a stable event identity, independent of its current record.
  void* get_handle() const noexcept { return handle_.get(); }

 private:
  friend class EventDirectKernelExecution;

  explicit NeuronEvent(std::shared_ptr<EventImpl> record)
      : handle_(std::make_shared<EventHandle>(std::move(record))) {}

  std::shared_ptr<EventHandle> handle_;
};

// Forward declaration for getCurrentNeuronStream
NeuronStream getCurrentNeuronStream(c10::DeviceIndex device_index);

}  // namespace at::neuron
