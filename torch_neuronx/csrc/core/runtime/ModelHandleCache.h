#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

namespace at::neuron {

/**
 * Model handle cache for sharing loaded NEFFs across streams.
 */
class ModelHandleCache {
 public:
  ModelHandleCache();
  ~ModelHandleCache();

  ModelHandleCache(const ModelHandleCache&) = delete;
  ModelHandleCache& operator=(const ModelHandleCache&) = delete;

  /**
   * Lookup only — returns nullptr on miss
   */
  std::shared_ptr<nrt::ModelHandle> GetModel(const std::string& cache_key) const;

  /**
   * Store a model in the cache. Returns the stored model
   */
  std::shared_ptr<nrt::ModelHandle> StoreModel(const std::string& cache_key,
                                               std::shared_ptr<nrt::ModelHandle> model,
                                               int device_index, bool has_collectives);

  /**
   * Compose the cache key for a (compile_key, device) pair.
   * Format: "dev{device_index}_{compile_key}".
   */
  std::string MakeCacheKey(const std::string& compile_key, int device_index) const;

  /**
   * Get detailed cache entry information for debugging
   */
  struct CacheEntryInfo {
    std::string cache_key;
    std::chrono::steady_clock::time_point created_time;
    int device_index;
    bool has_collectives;
  };

  std::vector<CacheEntryInfo> GetCacheEntries() const;

  /**
   * Clear the cache (for testing)
   */
  void Clear();

  /**
   * Get all cache keys
   */
  std::unordered_set<std::string> GetAllCacheKeys() const;

 private:
  struct CacheEntry {
    std::shared_ptr<nrt::ModelHandle> model;
    std::chrono::steady_clock::time_point created_time;
    int device_index;
    bool has_collectives;

    CacheEntry() = default;

    CacheEntry(std::shared_ptr<nrt::ModelHandle> model_ptr, int device, bool collectives)
        : model(std::move(model_ptr)),
          created_time(std::chrono::steady_clock::now()),
          device_index(device),
          has_collectives(collectives) {}
  };

  mutable std::shared_mutex cache_mutex_;
  std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache_;
};

}  // namespace at::neuron
