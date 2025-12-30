#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"

namespace at::neuron {

// Forward declarations
class OperationContext;

/**
 * @brief Core concatenation logic handler that manages StableHLO operations
 *
 * This class encapsulates all the concatenation logic including operation pattern
 * detection, tensor dependency analysis, and StableHLO module merging. It provides
 * a clean interface for the ConcatenationEngine to use without directly interacting
 * with StableHLO utilities.
 */
class ConcatenationCore {
 public:
  ConcatenationCore();
  ~ConcatenationCore();

  // Deleted copy/move constructors and assignment operators
  ConcatenationCore(const ConcatenationCore&) = delete;
  ConcatenationCore& operator=(const ConcatenationCore&) = delete;
  ConcatenationCore(ConcatenationCore&&) = delete;
  ConcatenationCore& operator=(ConcatenationCore&&) = delete;

  /**
   * @brief Result structure for concatenation processing
   */
  struct ConcatenationResult {
    /**
     * @brief Information about a processed operation
     *
     * For concatenated operations:
     * - operation points to the new concatenated operation
     * - cascading_ops contains raw pointers to the original operations being concatenated
     *
     * For non-concatenated operations:
     * - operation points to the original operation
     * - cascading_ops is empty
     */
    struct ProcessedOperation {
      OperationContext* operation;  // The raw ptr to new operation to execute
      std::vector<OperationContext*>
          cascading_ops;  // Raw ptrs to originals (empty if not concatenated)
    };

    std::vector<ProcessedOperation> processed_operations;  // Operations ready for execution
    size_t original_operations_consumed = 0;  // How many input operations were processed
    bool success = true;                      // Whether processing was successful
    std::string error_message;                // Error details if success is false
  };

  /**
   * @brief Process a list of buffered operations and attempt concatenation
   *
   * This is the main entry point that replaces the logic currently in
   * maybe_try_concatenate(). It analyzes the buffered operations, detects
   * concatenation opportunities, and returns processed operations ready for execution.
   *
   * @param buffered_ops List of buffered operations to process
   * @return ConcatenationResult containing processed operations and metadata
   */
  ConcatenationResult processBufferedOperations(const std::list<OperationContext*>& buffered_ops);

  /**
   * @brief Clear the concatenation cache
   */
  void clearCache();

  /**
   * @brief Get the current size of the concatenation cache
   * @return Number of cached concatenation results
   */
  size_t getCacheSize() const;

  /**
   * @brief Invalidate a cache entry for the given key
   *
   * This marks the cache entry as invalid so it will not be retrieved or used
   * even if the same key is encountered again. The entry itself remains in the
   * cache but is effectively disabled.
   *
   * @param key The cache key to invalidate
   */
  void invalidateCacheEntry(const std::string& key);

  /**
   * @brief Get a cache entry value and check if it's valid
   *
   * Retrieves the cache value for the given key and indicates whether the entry
   * is invalidated. If the entry doesn't exist or is invalidated, the function
   * returns false and value is not modified.
   *
   * @param key The cache key to retrieve
   * @param value Output parameter to store the cache value if valid
   * @return true if entry exists and is valid (not invalidated), false otherwise
   */
  bool getCacheEntryIfValid(const std::string& key, std::string& value) const;

 private:
  // Concatenation cache for storing merge results (currently unused in mock implementation)
  mutable std::mutex concatenation_cache_mutex_;
  std::unordered_map<std::string, std::string> concatenation_cache_;
  std::unordered_set<std::string> invalidated_cache_keys_;
};

}  // namespace at::neuron
