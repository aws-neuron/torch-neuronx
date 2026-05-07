#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace torch_neuronx {
namespace distributed {

/**
 * Tracks cleanup callbacks for NeuronWatchdog instances.
 * Allocator calls triggerShelvesCleanup() during allocation and OOM retry.
 */
class ShelvesCleanupRegistry {
 public:
  static ShelvesCleanupRegistry& instance() {
    static ShelvesCleanupRegistry registry;
    return registry;
  }

  void registerCleanup(void* key, std::function<void()> cleanup) {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanups_.push_back({key, std::move(cleanup)});
  }

  void unregisterCleanup(void* key) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, std::chrono::seconds(1), [this] { return cleanups_running_ == 0; });
    cleanups_.erase(std::remove_if(cleanups_.begin(), cleanups_.end(),
                                   [key](const auto& entry) { return entry.first == key; }),
                    cleanups_.end());
  }

  void triggerAllCleanups() {
    std::vector<std::function<void()>> callbacks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& entry : cleanups_) {
        callbacks.push_back(entry.second);
      }
      ++cleanups_running_;
      ++trigger_count_;
    }

    for (const auto& cb : callbacks) {
      cb();
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      --cleanups_running_;
    }
    cv_.notify_all();
  }

  size_t getTriggerCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trigger_count_;
  }

 private:
  ShelvesCleanupRegistry() = default;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::pair<void*, std::function<void()>>> cleanups_;
  int cleanups_running_ = 0;
  size_t trigger_count_ = 0;
};

inline void triggerShelvesCleanup() { ShelvesCleanupRegistry::instance().triggerAllCleanups(); }

}  // namespace distributed
}  // namespace torch_neuronx
