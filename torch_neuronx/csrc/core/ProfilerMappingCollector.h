#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

namespace at::neuron {

struct FrameworkOpExecId {
  int64_t seq_nr = 0;
  uint64_t th_id = 0;
  int64_t stream_id = 0;
  uint64_t os_tid = 0;  // OS thread ID
  uint64_t event_id = 0;
};

class ProfilerMappingCollector {
 public:
  static ProfilerMappingCollector& Instance() {
    static ProfilerMappingCollector instance;
    return instance;
  }

  void SetEnabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }
  bool IsEnabled() const { return enabled_.load(std::memory_order_acquire); }

  // Generate a unique event ID for framework-to-NRT event linking.
  static uint64_t GenerateEventId() {
    return next_event_id_.fetch_add(1, std::memory_order_relaxed);
  }

  void Record(nrt::SequenceId nrta_seq_id, int64_t pytorch_seq_nr, uint64_t pytorch_thread_id,
              int64_t stream_id, uint64_t os_tid, uint64_t event_id) {
    if (!IsEnabled()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    mappings_[nrta_seq_id].push_back(
        {pytorch_seq_nr, pytorch_thread_id, stream_id, os_tid, event_id});
  }

  std::unordered_map<nrt::SequenceId, std::vector<FrameworkOpExecId>> GetAndClear() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = std::move(mappings_);
    mappings_.clear();
    return result;
  }

 private:
  ProfilerMappingCollector() = default;
  ProfilerMappingCollector(const ProfilerMappingCollector&) = delete;
  ProfilerMappingCollector& operator=(const ProfilerMappingCollector&) = delete;

  static std::atomic<uint64_t> next_event_id_;
  std::atomic<bool> enabled_{false};
  std::mutex mutex_;
  std::unordered_map<nrt::SequenceId, std::vector<FrameworkOpExecId>> mappings_;
};

}  // namespace at::neuron
