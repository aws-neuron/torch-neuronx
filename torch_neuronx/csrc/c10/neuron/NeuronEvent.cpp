#include "NeuronEvent.h"

#include <stdexcept>

#include "NeuronStream.h"
#include "torch_neuronx/csrc/core/streams/StreamConfig.h"

namespace at::neuron {
namespace {

void SynchronizeRecord(const std::shared_ptr<EventImpl>& record) {
  if (!record->is_recorded.load(std::memory_order_acquire)) return;
  try {
    std::unique_lock<std::mutex> lock(record->sync_mutex);
    record->sync_cv.wait(
        lock, [&record] { return record->is_completed.load(std::memory_order_acquire); });
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Event synchronization failed: ", e.what());
  }
}

}  // namespace

void NeuronEvent::record(const NeuronStream& stream) {
  const auto previous = get_impl();
  auto record = std::make_shared<EventImpl>();
  record->enable_timing = previous->enable_timing;
  record->blocking = previous->blocking;
  record->device_index = stream.device_index();
  record->recorded_stream_impl.store(stream.get_stream_impl(), std::memory_order_relaxed);
  record->allow_queue_ordering = config::AreStreamsEnabled() && !config::IsHostCCEnabled();
  const bool stream_has_no_work = stream.query();
  if (stream_has_no_work && record->enable_timing) {
    record->record_time = std::chrono::steady_clock::now();
  }
  record->is_recorded.store(true, std::memory_order_release);
  record->is_completed.store(stream_has_no_work, std::memory_order_release);

  // Construct the capture before submission; publication never retargets this signal.
  NeuronEvent captured(record);
  if (!stream_has_no_work) stream.SubmitEventSignal(captured);
  {
    std::lock_guard<std::mutex> lock(handle_->mutex);
    handle_->current_record = std::move(record);
  }
}

void NeuronEvent::block(const NeuronStream& stream) {
  const auto captured = capture();
  if (!captured.query()) stream.SubmitEventWait(captured);
}

bool NeuronEvent::query() const { return get_impl()->query(); }

void NeuronEvent::synchronize() const { SynchronizeRecord(get_impl()); }

float NeuronEvent::elapsed_time(const NeuronEvent& end_event) const {
  // Read the same records even if a public alias records either event again.
  const auto start = get_impl();
  const auto end = handle_ == end_event.handle_ ? start : end_event.get_impl();
  TORCH_CHECK(start->enable_timing, "Event was not created with timing enabled");
  TORCH_CHECK(end->enable_timing, "End event was not created with timing enabled");
  TORCH_CHECK(start->is_recorded.load(), "Event has not been recorded");
  TORCH_CHECK(end->is_recorded.load(), "End event has not been recorded");
  TORCH_CHECK(start->device_index == end->device_index, "Events must be on the same device");
  TORCH_CHECK(start->query() && end->query(), "Event has not completed. Call synchronize() first.");
  const auto duration = end->record_time - start->record_time;
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  return static_cast<float>(ns) / 1'000'000.0f;
}

}  // namespace at::neuron
