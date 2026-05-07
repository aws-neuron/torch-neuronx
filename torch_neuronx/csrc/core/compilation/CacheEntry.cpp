#include "CacheEntry.h"

#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"

namespace at::neuron {

void CacheEntryGuard::operator()(const std::vector<uint8_t>* ptr) const {
  // Mark the cache entry as inactive if we have a valid cache and key.
  // This ensures proper cleanup when the unique_ptr is destroyed.
  if (cache != nullptr && !cache_key.empty()) {
    cache->MarkEntryInactive(cache_key);
  }
}

}  // namespace at::neuron
