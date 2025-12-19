#include "ModelHandleCache.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron {

ModelHandleCache::ModelHandleCache() {}

ModelHandleCache::~ModelHandleCache() {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);
  cache_.clear();
}

std::string ModelHandleCache::GenerateCacheKey(const std::vector<uint8_t>& neff_bytes,
                                               int device_index) const {
  std::size_t hash_value = std::hash<std::string_view>{}(
      std::string_view(reinterpret_cast<const char*>(neff_bytes.data()), neff_bytes.size()));
  std::ostringstream oss;
  oss << "dev" << device_index << "_" << std::hex << hash_value;
  return oss.str();
}

std::shared_ptr<ModelHandleCache::CacheEntry> ModelHandleCache::TryCacheLookup(
    const std::string& cache_key) const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  const auto it = cache_.find(cache_key);
  if (it != cache_.end() && it->second) {
    TORCH_NEURONX_DEBUG("Model handle cache hit, cache_key=", cache_key);
    return it->second;
  }

  TORCH_NEURONX_DEBUG("Model handle cache miss, cache_key=", cache_key);
  return nullptr;
}

std::shared_ptr<at::neuron::nrt::Model> ModelHandleCache::GetOrLoadModel(
    const CompilableKernelExecution& kernel_execution, int device_index, int num_cores) {
  // Extract collective information from kernel execution
  bool has_collectives = kernel_execution.HasCollectives();
  const std::vector<uint8_t>& neff_bytes = kernel_execution.GetCachedNeff();
  std::string cache_key = GenerateCacheKey(neff_bytes, device_index);

  TORCH_NEURONX_DEBUG("Looking up model handle", "cache_key=", cache_key, "device=", device_index,
                      "cores=", num_cores, "collectives=", has_collectives);

  // Try cache lookup first
  auto cached_entry = TryCacheLookup(cache_key);
  if (cached_entry) {
    return cached_entry->model;
  }

  // Cache miss - need to load the model
  TORCH_NEURONX_DEBUG("Loading new model", "cache_key=", cache_key,
                      "neff_size=", neff_bytes.size());

  // Create a new Model instance and load the NEFF
  auto model = std::make_shared<at::neuron::nrt::Model>();

  try {
    if (has_collectives) {
      model->LoadCollectives(neff_bytes, device_index, num_cores);
    } else {
      model->Load(neff_bytes, device_index, num_cores);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to load model - " + std::string(e.what()) + " [cache_key=" + cache_key +
        ", device=" + std::to_string(device_index) + ", cores=" + std::to_string(num_cores) +
        ", has_collectives=" + (has_collectives ? "true" : "false") + "]");
  }

  TORCH_NEURONX_DEBUG("Model loaded successfully", "cache_key=", cache_key);

  // Store in cache with optimized double-checked locking
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    // Check if another thread loaded it in the meantime
    const auto it = cache_.find(cache_key);
    if (it != cache_.end() && it->second) {
      TORCH_NEURONX_DEBUG("Model was loaded by another thread, using cached version",
                          "cache_key=", cache_key);
      return it->second->model;
    }

    // Create and store new entry using emplace for efficiency
    auto entry = std::make_shared<CacheEntry>(model, device_index, has_collectives);
    cache_.emplace(cache_key, entry);

    TORCH_NEURONX_DEBUG("Model cached", "cache_key=", cache_key, "total_entries=", cache_.size());
    return model;
  }
}

std::vector<ModelHandleCache::CacheEntryInfo> ModelHandleCache::GetCacheEntries() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  std::vector<CacheEntryInfo> entries;
  entries.reserve(cache_.size());

  for (const auto& [key, entry] : cache_) {
    if (entry) {
      entries.emplace_back(CacheEntryInfo{.cache_key = key,
                                          .created_time = entry->created_time,
                                          .device_index = entry->device_index,
                                          .has_collectives = entry->has_collectives});
    }
  }

  return entries;
}

void ModelHandleCache::Clear() {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);

  TORCH_NEURONX_DEBUG("Clearing model cache", "entries=", cache_.size());

  cache_.clear();
}

std::unordered_set<std::string> ModelHandleCache::GetAllCacheKeys() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  std::unordered_set<std::string> keys;
  keys.reserve(cache_.size());

  for (const auto& [key, entry] : cache_) {
    keys.emplace(key);
  }

  return keys;
}

}  // namespace at::neuron
