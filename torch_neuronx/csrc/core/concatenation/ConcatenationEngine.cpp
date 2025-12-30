#include "ConcatenationEngine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationCore.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron {

// Pimpl implementation class to hide implementation details from header
class ConcatenationEngine::BufferingImpl {
 public:
  // Per-stream buffering system - using std::list for O(1) front removal
  std::unordered_map<StreamImpl*, std::unique_ptr<std::list<OperationContext*>>> operation_buffers_;

  // Global ordered buffer - maintains operations in arrival order across all streams
  // Using std::list for O(1) front removal operations
  std::list<OperationContext*> global_ordered_buffer_;

  mutable std::mutex buffered_operations_mutex_;
};

// Constructor
ConcatenationEngine::ConcatenationEngine(std::atomic<int64_t>* execution_queue_depth,
                                         int64_t queue_low_threshold, int64_t queue_target_depth)
    : buffering_impl_(std::make_unique<BufferingImpl>()),
      execution_queue_depth_(execution_queue_depth),
      queue_low_threshold_(queue_low_threshold),
      queue_target_depth_(queue_target_depth) {}

// Destructor
ConcatenationEngine::~ConcatenationEngine() = default;

std::list<OperationContext*> ConcatenationEngine::processConcatenationTask(
    OperationContext* operation) {
  std::list<OperationContext*> operations_to_enqueue;

  // Handle NULL operation - refill signal from execution worker
  if (!operation) {
    // NULL operation is a signal to flush operations to refill execution queue
    if (!execution_queue_depth_) {
      // No semaphore available (concatenation not using queue depth management)
      TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] NULL operation without queue depth management");
      return operations_to_enqueue;
    }

    int64_t current_depth = execution_queue_depth_->load();
    int64_t target_depth = queue_target_depth_;
    int64_t low_threshold = queue_low_threshold_;

    if (current_depth >= low_threshold) {
      // Already recovered, no need to flush
      TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Refill not needed",
                          "current_depth=", current_depth, "low_threshold=", low_threshold);
      return operations_to_enqueue;
    }

    int64_t ops_to_flush = target_depth - current_depth;
    if (ops_to_flush <= 0) {
      return operations_to_enqueue;
    }

    TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Refilling execution queue",
                        "current_depth=", current_depth, "target_depth=", target_depth,
                        "ops_to_flush=", ops_to_flush);

    // Flush operations in global order
    return flushOperationsInGlobalOrder(static_cast<size_t>(ops_to_flush));
  }

  auto* stream = operation->stream;
  c10::StreamId stream_id = stream->GetStreamId();

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Concatenation processing", "stream_id=", stream_id,
                      "op=", operation->GetOpName());
  try {
    // Check if we should do direct flush (low queue depth, no buffered ops)
    if (execution_queue_depth_) {
      int64_t current_depth = execution_queue_depth_->load();
      bool has_buffered = !buffering_impl_->global_ordered_buffer_.empty();

      if (current_depth < queue_low_threshold_ && !has_buffered) {
        TORCH_NEURONX_DEBUG(
            "[CONCATENATION_ENGINE] Direct flush - low queue depth, no buffered ops",
            "current_depth=", current_depth, "threshold=", queue_low_threshold_);
        return {operation};
      }
    }

    std::lock_guard<std::mutex> lock(buffering_impl_->buffered_operations_mutex_);
    addOperationToBuffer(operation, stream);

    // Check if we should try concatenation on the current buffer
    auto buffer_it = buffering_impl_->operation_buffers_.find(stream);
    if (buffer_it != buffering_impl_->operation_buffers_.end() && buffer_it->second &&
        !buffer_it->second->empty()) {
      uint64_t cutoff_index = 0;
      auto result = maybe_try_concatenate(*buffer_it->second, stream, &cutoff_index);

      if (!result.empty()) {
        TORCH_NEURONX_DEBUG(
            "[CONCATENATION_ENGINE] maybe_try_concatenate returned operations for execution",
            "stream_id=", stream_id, "returned_ops=", result.size(), "cutoff_index=", cutoff_index);

        // Append list to list
        operations_to_enqueue.splice(operations_to_enqueue.end(), result);

        // Remove the processed operations from buffer (up to cutoff_index)
        if (cutoff_index > 0 && cutoff_index <= buffer_it->second->size()) {
          // Remove cutoff_index elements from the front (O(1) per element with list)
          for (uint64_t idx = 0; idx < cutoff_index; ++idx) {
            buffer_it->second->pop_front();
          }

          // If buffer is now empty, clean up
          if (buffer_it->second->empty()) {
            buffering_impl_->operation_buffers_.erase(buffer_it);
          }
        }
      }
    }

    return operations_to_enqueue;
  } catch (const std::exception& e) {
    TORCH_NEURONX_ERROR(
        "[CONCATENATION_ENGINE] Error during concatenation processing", "stream_id=", stream_id,
        "op=", operation ? operation->GetOpName() : "flush_signal", "error=", e.what());
    // Return any operations we've already processed
    return operations_to_enqueue;
  }
}

// maybe_try_concatenate calls ConcatenationCore to perform any concatenation on a sequence of
// oeprations, the cutoff_index is populated based till what point buffered operations have been
// concatenated and ready for execution. the operations that are after the cutoff index are still
// kept in the buffer while operations before the cutoff index are returned for execution.
std::list<OperationContext*> ConcatenationEngine::maybe_try_concatenate(
    std::list<OperationContext*>& buffered_ops, StreamImpl* stream, uint64_t* cutoff_index) {
  c10::StreamId stream_id = stream->GetStreamId();

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] maybe_try_concatenate called",
                      "stream_id=", stream_id, "buffered_ops=", buffered_ops.size());

  // Initialize cutoff_index to 0 - will be updated as we process operations
  *cutoff_index = 0;
  if (buffered_ops.empty()) {
    return {};
  }

  // Delegate to ConcatenationCore for the actual concatenation logic
  auto result = concatenation_core_.processBufferedOperations(buffered_ops);

  if (!result.success) {
    TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] ConcatenationCore processing failed",
                        "stream_id=", stream_id, "error=", result.error_message);
    return {};
  }

  // Set the cutoff_index from ConcatenationCore result
  *cutoff_index = result.original_operations_consumed;

  // Replace sequential operations with concatenated operations in the stream's active_operations_
  // StreamImpl is responsible for populating cascading_operations with proper shared_ptrs
  std::list<OperationContext*> operations_to_execute;

  for (const auto& proc_op : result.processed_operations) {
    if (!proc_op.cascading_ops.empty()) {
      // This is a concatenated operation - let StreamImpl handle the replacement
      TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Replacing sequential operations in stream",
                          "stream_id=", stream_id,
                          "concatenated_op=", proc_op.operation->GetOpName(),
                          "original_ops_count=", proc_op.cascading_ops.size());
    }

    operations_to_execute.push_back(proc_op.operation);
  }

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] maybe_try_concatenate completed",
                      "stream_id=", stream_id, "input_ops=", buffered_ops.size(),
                      "output_ops=", operations_to_execute.size(),
                      "original_ops_consumed=", result.original_operations_consumed,
                      "cutoff_index=", *cutoff_index);

  return operations_to_execute;
}

void ConcatenationEngine::addOperationToBuffer(OperationContext* operation, StreamImpl* stream) {
  c10::StreamId stream_id = stream->GetStreamId();

  // Note: This method assumes the caller already holds buffered_operations_mutex_
  if (!buffering_impl_->operation_buffers_[stream]) {
    buffering_impl_->operation_buffers_[stream] = std::make_unique<std::list<OperationContext*>>();
  }
  buffering_impl_->operation_buffers_[stream]->push_back(operation);

  // Also add to global ordered buffer to maintain cross-stream ordering
  buffering_impl_->global_ordered_buffer_.push_back(operation);

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Operation added to buffer", "stream_id=", stream_id,
                      "op=", operation->GetOpName(), "per_stream_buffer_size=",
                      buffering_impl_->operation_buffers_[stream]->size(),
                      "global_buffer_size=", buffering_impl_->global_ordered_buffer_.size());
}

void ConcatenationEngine::clearCache() {
  concatenation_core_.clearCache();
  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Concatenation cache cleared");
}

size_t ConcatenationEngine::getCacheSize() const { return concatenation_core_.getCacheSize(); }

size_t ConcatenationEngine::getBufferedOperationsCount() const {
  std::lock_guard<std::mutex> lock(buffering_impl_->buffered_operations_mutex_);
  size_t total_count = 0;
  for (const auto& [stream, buffer] : buffering_impl_->operation_buffers_) {
    if (buffer) {
      total_count += buffer->size();
    }
  }
  return total_count;
}

std::list<OperationContext*> ConcatenationEngine::processCompilationFailure(
    OperationContext* failed_concatenated_op) {
  if (!failed_concatenated_op) {
    TORCH_NEURONX_ERROR(
        "[CONCATENATION_ENGINE] Null operation passed to processCompilationFailure");
    throw std::runtime_error(
        "[CONCATENATION_ENGINE] Null operation passed to processCompilationFailure");
  }

  auto* stream = failed_concatenated_op->stream;
  c10::StreamId stream_id = stream->GetStreamId();
  const std::string& op_name = failed_concatenated_op->GetOpName();

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Processing compilation failure",
                      "stream_id=", stream_id, "op=", op_name,
                      "cascading_ops_count=", failed_concatenated_op->cascading_operations.size());

  // Get the compilable kernel to extract cache key
  auto& kernel = failed_concatenated_op->GetCompilableKernel();
  auto* xla_kernel = dynamic_cast<XLACompilableKernelExecution*>(&kernel);

  if (xla_kernel) {
    // Generate cache key for this concatenated operation
    // The cache key should match what was used during concatenation
    std::string cache_key = xla_kernel->GetCacheKey();

    // Invalidate the cache entry to prevent retry for this concatenation
    concatenation_core_.invalidateCacheEntry(cache_key);

    TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Invalidated cache entry for failed concatenation",
                        "stream_id=", stream_id, "cache_key=", cache_key);
  }

  // Replace the failed concatenated operation with its original operations in the stream
  TORCH_NEURONX_DEBUG(
      "[CONCATENATION_ENGINE] Replacing failed concatenation with original operations",
      "stream_id=", stream_id, "failed_op=", op_name,
      "original_ops_count=", failed_concatenated_op->cascading_operations.size());

  // Extract raw pointers from cascading_operations (which are shared_ptrs)
  std::list<OperationContext*> raw_cascading_ops;
  for (const auto& op_shared : failed_concatenated_op->cascading_operations) {
    raw_cascading_ops.push_back(op_shared.get());
  }

  return raw_cascading_ops;
}

std::list<OperationContext*> ConcatenationEngine::flushOperationsInGlobalOrder(
    size_t num_to_flush) {
  std::lock_guard<std::mutex> lock(buffering_impl_->buffered_operations_mutex_);

  std::list<OperationContext*> operations_to_enqueue;

  if (buffering_impl_->global_ordered_buffer_.empty()) {
    TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] No buffered operations to flush");
    return operations_to_enqueue;
  }

  // Determine how many operations to actually flush
  size_t actual_flush_count =
      std::min(num_to_flush, buffering_impl_->global_ordered_buffer_.size());

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Direct flushing operations in global order",
                      "requested=", num_to_flush,
                      "available=", buffering_impl_->global_ordered_buffer_.size(),
                      "flushing=", actual_flush_count);

  // Directly flush operations in the order they were received (no concatenation)
  for (size_t i = 0; i < actual_flush_count; ++i) {
    OperationContext* op = buffering_impl_->global_ordered_buffer_.front();
    operations_to_enqueue.push_back(op);

    // Remove from per-stream buffer as well
    auto* stream = op->stream;
    auto buffer_it = buffering_impl_->operation_buffers_.find(stream);
    if (buffer_it != buffering_impl_->operation_buffers_.end() && buffer_it->second) {
      auto& stream_buffer = *buffer_it->second;
      // Remove the first operation in this stream
      stream_buffer.pop_front();

      // Clean up empty buffer
      if (stream_buffer.empty()) {
        buffering_impl_->operation_buffers_.erase(buffer_it);
      }
    }

    // Remove from global ordered buffer.
    buffering_impl_->global_ordered_buffer_.pop_front();
  }

  TORCH_NEURONX_DEBUG("[CONCATENATION_ENGINE] Direct flush completed",
                      "flushed_ops=", actual_flush_count,
                      "remaining_buffered=", buffering_impl_->global_ordered_buffer_.size());

  return operations_to_enqueue;
}

}  // namespace at::neuron
