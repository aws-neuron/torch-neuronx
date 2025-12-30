#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationCore.h"

namespace at::neuron {

// Forward declarations
class StreamImpl;
class OperationContext;

/**
 * @brief Dedicated engine for handling operation concatenation optimization
 *
 * This class manages the buffering and concatenation of compatible operations using
 * a time-based buffering strategy. Operations are buffered until no new operations
 * arrive for a configurable timeout period, a synchronize is called, or a non-XLA
 * operation is encountered. It maintains per-stream buffers and a global concatenation cache.
 */
class ConcatenationEngine {
 public:
  ConcatenationEngine(std::atomic<int64_t>* execution_queue_depth = nullptr,
                      int64_t queue_low_threshold = 10, int64_t queue_target_depth = 50);
  ~ConcatenationEngine();

  // Deleted copy/move constructors and assignment operators
  ConcatenationEngine(const ConcatenationEngine&) = delete;
  ConcatenationEngine& operator=(const ConcatenationEngine&) = delete;
  ConcatenationEngine(ConcatenationEngine&&) = delete;
  ConcatenationEngine& operator=(ConcatenationEngine&&) = delete;

  /**
   * @brief Process a concatenation task and return operations to enqueue
   *
   * @param operation The operation context to process (nullptr for flush signal)
   * @return List of operations to be enqueued to compilation worker
   */
  std::list<OperationContext*> processConcatenationTask(OperationContext* operation);

  /**
   * @brief Maybe try to concatenate operations and return cutoff index
   *
   * @param buffered_ops List of buffered operations to attempt concatenation on
   * @param stream The stream for these operations
   * @param cutoff_index Output parameter indicating how many operations were processed
   * @return List of operations ready for execution
   */
  std::list<OperationContext*> maybe_try_concatenate(std::list<OperationContext*>& buffered_ops,
                                                     StreamImpl* stream, uint64_t* cutoff_index);

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
   * @brief Get statistics about buffered operations
   * @return Number of currently buffered operations across all streams
   */
  size_t getBufferedOperationsCount() const;

  /**
   * @brief Process a failed compilation of a concatenated operation
   *
   * When a concatenated operation fails to compile, this function handles the failure by:
   * - Extracting the original cascading operations that were concatenated
   * - Invalidating the cache entry to prevent retrying the same concatenation
   * - Cleaning up/deallocating the concatenated operation
   *
   * @param failed_concatenated_op The concatenated operation that failed to compile
   * @return List of original operations to be compiled individually
   */
  std::list<OperationContext*> processCompilationFailure(OperationContext* failed_concatenated_op);

 private:
  // Core concatenation logic handler
  ConcatenationCore concatenation_core_;

  class BufferingImpl;
  std::unique_ptr<BufferingImpl> buffering_impl_;

  // Execution queue depth management (shared with OperationExecutionEngine)
  std::atomic<int64_t>* execution_queue_depth_;
  int64_t queue_low_threshold_;
  int64_t queue_target_depth_;

  /**
   * @brief Flush operations in global order to maintain submission ordering
   * @param num_to_flush Maximum number of operations to flush
   * @return List of operations to be enqueued for compilation
   */
  std::list<OperationContext*> flushOperationsInGlobalOrder(size_t num_to_flush);

  /**
   * @brief Add an operation to the buffer for a stream
   * @param operation The operation context to buffer
   * @param stream The stream for this operation
   */
  void addOperationToBuffer(OperationContext* operation, StreamImpl* stream);
};

}  // namespace at::neuron
