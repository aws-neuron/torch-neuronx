#include "ConcatenationCore.h"

#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron {

// Constructor
ConcatenationCore::ConcatenationCore() = default;

// Destructor
ConcatenationCore::~ConcatenationCore() = default;

ConcatenationCore::ConcatenationResult ConcatenationCore::processBufferedOperations(
    const std::list<OperationContext*>& buffered_ops) {
  ConcatenationResult result;

  TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Mock processBufferedOperations called",
                      "buffered_ops=", buffered_ops.size());

  // Mock implementation: Just return all input operations unchanged
  // Convert to new ProcessedOperation structure
  for (auto* op : buffered_ops) {
    ConcatenationResult::ProcessedOperation proc_op;
    proc_op.operation = op;
    proc_op.cascading_ops = {};  // Empty = not concatenated
    result.processed_operations.push_back(proc_op);
  }

  result.original_operations_consumed = buffered_ops.size();
  result.success = true;

  TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Mock implementation returning operations unchanged",
                      "input_ops=", buffered_ops.size(),
                      "output_ops=", result.processed_operations.size());

  return result;
}

void ConcatenationCore::clearCache() {
  std::lock_guard<std::mutex> cache_lock(concatenation_cache_mutex_);
  concatenation_cache_.clear();
  invalidated_cache_keys_.clear();
  TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Concatenation cache cleared");
}

size_t ConcatenationCore::getCacheSize() const {
  std::lock_guard<std::mutex> cache_lock(concatenation_cache_mutex_);
  return concatenation_cache_.size();
}

void ConcatenationCore::invalidateCacheEntry(const std::string& key) {
  std::lock_guard<std::mutex> cache_lock(concatenation_cache_mutex_);
  invalidated_cache_keys_.insert(key);
  TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Cache entry invalidated", "key=", key);
}

bool ConcatenationCore::getCacheEntryIfValid(const std::string& key, std::string& value) const {
  std::lock_guard<std::mutex> cache_lock(concatenation_cache_mutex_);

  // Check if the entry is invalidated
  if (invalidated_cache_keys_.find(key) != invalidated_cache_keys_.end()) {
    TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Cache entry is invalidated", "key=", key);
    return false;
  }

  // Check if the entry exists in the cache
  auto it = concatenation_cache_.find(key);
  if (it == concatenation_cache_.end()) {
    TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Cache entry not found", "key=", key);
    return false;
  }

  // Entry exists and is valid
  value = it->second;
  TORCH_NEURONX_DEBUG("[CONCATENATION_CORE] Retrieved valid cache entry", "key=", key);
  return true;
}

}  // namespace at::neuron
