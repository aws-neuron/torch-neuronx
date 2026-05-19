#include "ProfilerEventBuffer.h"

namespace at::neuron {

namespace {
std::atomic<ProfilerEventBuffer*> g_activeBuffer{nullptr};
}  // namespace

ProfilerEventBuffer* getActiveProfilerBuffer() {
  return g_activeBuffer.load(std::memory_order_acquire);
}

void setActiveProfilerBuffer(ProfilerEventBuffer* buffer) {
  g_activeBuffer.store(buffer, std::memory_order_release);
}

void ProfilerEventBuffer::activate() {
  std::lock_guard<std::mutex> lock(mutex_);
  active_ = true;
}

void ProfilerEventBuffer::deactivate() {
  std::lock_guard<std::mutex> lock(mutex_);
  active_ = false;
}

// TODO: If profiling shows contention under multi-stream workloads, consider
// replacing this mutex with per-thread subqueues (modeled on PyTorch's
// RecordQueue pattern: TLS-cached lookup, lock-free writes, drain at stop).
void ProfilerEventBuffer::record(ProfiledOperation op) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) {
    return;
  }
  events_.push_back(std::move(op));
}

void ProfilerEventBuffer::recordSleep(RuntimeSleepEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) {
    return;
  }
  sleep_events_.push_back(event);
}

std::vector<ProfiledOperation> ProfilerEventBuffer::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ProfiledOperation> result;
  result.swap(events_);
  return result;
}

std::vector<RuntimeSleepEvent> ProfilerEventBuffer::drainSleepEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<RuntimeSleepEvent> result;
  result.swap(sleep_events_);
  return result;
}

bool ProfilerEventBuffer::isActive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

}  // namespace at::neuron
