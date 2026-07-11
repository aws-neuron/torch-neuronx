#include "ModelHandleCache.h"

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron {

ModelHandleCache::ModelHandleCache() {}

ModelHandleCache::~ModelHandleCache() {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);
  cache_.clear();
}

std::string ModelHandleCache::MakeCacheKey(const std::string& compile_key, int device_index) const {
  return "dev" + std::to_string(device_index) + "_" + compile_key;
}

std::shared_ptr<nrt::ModelHandle> ModelHandleCache::GetModel(const std::string& cache_key) const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  const auto it = cache_.find(cache_key);
  if (it != cache_.end() && it->second) {
    TORCH_NEURONX_DEBUG("Model handle cache hit, cache_key=", cache_key);
    return it->second->model;
  }

  TORCH_NEURONX_DEBUG("Model handle cache miss, cache_key=", cache_key);
  return nullptr;
}

std::shared_ptr<nrt::ModelHandle> ModelHandleCache::StoreModel(
    const std::string& cache_key, std::shared_ptr<nrt::ModelHandle> model, int device_index,
    bool has_collectives) {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);

  // Double-checked: another thread may have stored it first
  const auto it = cache_.find(cache_key);
  if (it != cache_.end() && it->second) {
    TORCH_NEURONX_DEBUG("Model was loaded by another thread, using cached version",
                        "cache_key=", cache_key);
    return it->second->model;
  }

  auto entry = std::make_shared<CacheEntry>(std::move(model), device_index, has_collectives);
  cache_.emplace(cache_key, entry);

  TORCH_NEURONX_DEBUG("Model cached", "cache_key=", cache_key, "total_entries=", cache_.size());
  return entry->model;
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
