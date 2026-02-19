#pragma once

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
#include "torch_neuronx/csrc/core/concatenation/IrConcatStrategy.h"

namespace at::neuron {

// Forward declarations
class StreamImpl;
class OperationContext;

// Default buffer size limit for concatenation engine
constexpr int DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT = 1024;

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
  ConcatenationEngine();
  ~ConcatenationEngine();

  // Deleted copy/move constructors and assignment operators
  ConcatenationEngine(const ConcatenationEngine&) = delete;
  ConcatenationEngine& operator=(const ConcatenationEngine&) = delete;
  ConcatenationEngine(ConcatenationEngine&&) = delete;
  ConcatenationEngine& operator=(ConcatenationEngine&&) = delete;

  /**
   * @brief Process a concatenation task and return operations to enqueue
   *
   * @param operation The operation context to process (must not be nullptr)
   * @return List of operations to be enqueued to compilation worker
   */
  std::list<OperationContext*> ProcessConcatenationTask(OperationContext* operation);

  /**
   * @brief Flush buffered operations without concatenation
   *
   * Disables accumulation mode and flushes buffered operations without
   * attempting concatenation.
   *
   * @param num_to_flush Maximum number of operations to flush
   * @return List of operations to be enqueued for compilation
   */
  std::list<OperationContext*> Flush(size_t num_to_flush);

  /**
   * @brief Maybe try to concatenate operations and return cutoff index
   *
   * @param buffered_ops List of buffered operations to attempt concatenation on
   * @param stream The stream for these operations
   * @param cutoff_index Output parameter indicating how many operations were processed
   * @return List of operations ready for execution
   */
  std::list<OperationContext*> MaybeTryConcatenate(std::list<OperationContext*>& buffered_ops,
                                                   StreamImpl* stream, uint64_t* cutoff_index);

  /**
   * @brief Clear the concatenation cache
   */
  void ClearCache();

  /**
   * @brief Get the current size of the concatenation cache
   * @return Number of cached concatenation results
   */
  size_t GetCacheSize() const;

  /**
   * @brief Get statistics about buffered operations
   * @return Number of currently buffered operations across all streams
   */
  size_t GetBufferedOperationsCount() const;

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
  void ProcessConcatenationFailure(OperationContext* failed_concatenated_op);

  int GetBufferSizeLimit() const { return buffer_size_limit; }

 private:
  class BufferingImpl;
  std::unique_ptr<BufferingImpl> buffering_impl_;

  // Core concatenation logic handler
  std::unique_ptr<torch_neuronx::ConcatenationCore> concatenation_core_;
  int buffer_size_limit = DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT;

  // Track the last processed stream for stream change detection
  StreamImpl* last_processed_stream_ = nullptr;

  // Accumulation state flag - controls whether we buffer operations for concatenation
  // When true: buffer operations and attempt concatenation (normal mode)
  // When false: execute operations directly without buffering (post-refill mode)
  bool accumulation_enabled_ = false;

  // Feature flag for skip intermediate optimization
  // Controlled by TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE environment variable
  // Default: enabled. Set TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 to disable
  bool skip_intermediate_enabled_ = true;

  /**
   * @brief Flush operations in global order to maintain submission ordering
   * Flushes buffered operations in global order (no concatenation)
   * @param num_to_flush Maximum number of operations to flush
   * @return List of operations to be enqueued for compilation
   */
  std::list<OperationContext*> FlushOperationsInGlobalOrder(size_t num_to_flush);

  /**
   * @brief Buffer operation and try normal concatenation
   * @param operation The operation to buffer
   * @return List of operations if concatenation triggers, empty otherwise
   */
  std::list<OperationContext*> BufferAndTryConcatenation(OperationContext* operation);

  /**
   * @brief Add an operation to the buffer for a stream
   * @param operation The operation context to buffer
   * @param stream The stream for this operation
   */
  void AddOperationToBuffer(OperationContext* operation, StreamImpl* stream);

  /**
   * @brief Check if an operation is a non-fusible boundary operation (memory/copy or buffer limit
   * exceeded)
   * @param operation The operation context to check
   * @param stream The stream for this operation (needed for buffer size check)
   * @return true if operation is a non-fusible boundary operation
   */
  bool IsNonFusibleBoundaryOperation(const OperationContext* operation, StreamImpl* stream) const;

  /**
   * @brief Check if an operation is a fusible boundary operation (e.g., matmul operations)
   * @param operation The operation context to check
   * @return true if operation can start a fusion region
   */
  bool IsFusibleBoundaryOperation(const OperationContext* operation) const;

  /**
   * @brief Check if an operation is in-place (any input address matches any output address)
   * @param operation The operation context to check
   * @return true if operation modifies its inputs in-place
   */
  bool IsInPlaceOperation(const OperationContext* operation) const;

  /**
   * @brief Flush all operations from a stream buffer with concatenation attempt
   * @param stream The stream to flush
   * @return List of operations to be enqueued for execution
   */
  std::list<OperationContext*> FlushStreamBuffer(StreamImpl* stream);

  /**
   * @brief Try normal concatenation on buffered operations
   * @param stream The stream to process
   * @return List of operations ready for execution
   */
  std::list<OperationContext*> TryNormalConcatenation(StreamImpl* stream);

  /**
   * @brief Check if an operation has output addresses that conflict with already tracked outputs
   * @param operation The operation to check
   * @param stream The stream for this operation
   * @return true if any output address is already tracked (conflict detected)
   */
  bool HasOutputConflict(const OperationContext* operation, StreamImpl* stream) const;

  /**
   * @brief Track output addresses for conflict detection
   * @param operation The operation whose outputs to track
   * @param stream The stream for this operation
   */
  void TrackOutputAddresses(const OperationContext* operation, StreamImpl* stream);
};

}  // namespace at::neuron
