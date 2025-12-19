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

// Forward declarations
class NeuronKernelExecution;
class CompilableKernelExecution;

/**
 * Model handle cache for sharing loaded NEFFs across streams
 */
class ModelHandleCache {
 public:
  ModelHandleCache();
  ~ModelHandleCache();

  // Disable copy and move for now
  ModelHandleCache(const ModelHandleCache&) = delete;
  ModelHandleCache& operator=(const ModelHandleCache&) = delete;

  /**
   * Get or load a model for the given kernel
   *
   * @param kernel_execution The kernel execution context
   * @param device_index The device to load the model on
   * @param num_cores Number of cores to use (default 1)
   * @return Shared pointer to cached Model
   */
  std::shared_ptr<at::neuron::nrt::Model> GetOrLoadModel(
      const CompilableKernelExecution& kernel_execution, int device_index, int num_cores = 1);

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
  /**
   * Cache entry with Model and metadata
   */
  struct CacheEntry {
    std::shared_ptr<at::neuron::nrt::Model> model;
    std::chrono::steady_clock::time_point created_time;
    int device_index;
    bool has_collectives;

    CacheEntry() = default;

    CacheEntry(std::shared_ptr<at::neuron::nrt::Model> model_ptr, int device, bool collectives)
        : model(std::move(model_ptr)),
          created_time(std::chrono::steady_clock::now()),
          device_index(device),
          has_collectives(collectives) {}
  };

  mutable std::shared_mutex cache_mutex_;
  std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache_;

  // Helper functions
  std::string GenerateCacheKey(const std::vector<uint8_t>& neff_bytes, int device_index) const;

  std::shared_ptr<CacheEntry> TryCacheLookup(const std::string& cache_key) const;
};

}  // namespace at::neuron
