#include "ConcatenationEngine.h"

#include <climits>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
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

  // Track output addresses per stream for conflict detection
  std::unordered_map<StreamImpl*, std::unordered_set<void*>> output_ptrs_;

  mutable std::mutex buffered_operations_mutex_;
};

// Constructor
ConcatenationEngine::ConcatenationEngine()
    : buffering_impl_(std::make_unique<BufferingImpl>()),
      concatenation_core_(torch_neuronx::ConcatenationCoreFactory::CreateInstance()) {
  // Load buffer size limit from environment variable
  const char* buffer_size_limit_env = std::getenv("TORCH_NEURONX_CONCATENATION_BUFFER_SIZE_LIMIT");
  if (buffer_size_limit_env) {
    try {
      int parsed_value = std::stoi(buffer_size_limit_env);
      if (parsed_value > 0) {
        buffer_size_limit = parsed_value;
      } else {
        TORCH_NEURONX_WARN(
            "Invalid TORCH_NEURONX_CONCATENATION_BUFFER_SIZE_LIMIT value:", buffer_size_limit_env,
            "- must be positive, using default:", DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT);
      }
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN(
          // Use default value on parse failure
          "Failed to parse TORCH_NEURONX_CONCATENATION_BUFFER_SIZE_LIMIT:", buffer_size_limit_env,
          "- using default:", DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT);
    }
  }

  // Load skip intermediate flag from environment variable
  // Default: enabled. Set TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 to disable
  const char* skip_intermediate_env = std::getenv("TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE");
  if (skip_intermediate_env && std::string(skip_intermediate_env) == "0") {
    skip_intermediate_enabled_ = false;
    TORCH_NEURONX_DEBUG("Using TORCH_NEURONX_APPLY_SKIP_INTERMEDIATE=0 (disabled)");
  }
}

// Destructor
ConcatenationEngine::~ConcatenationEngine() = default;

bool ConcatenationEngine::IsNonFusibleBoundaryOperation(const OperationContext* operation,
                                                        StreamImpl* stream) const {
  if (!operation || !operation->kernel_execution) {
    return false;
  }

  // Only kHLO operations can be concatenated
  // All other kernel types are boundary operations that trigger implicit flush
  // Exception: kHint operations (allocation/deallocation hints) are not boundaries
  auto kernel_type = operation->kernel_execution->GetKernelType();

  bool is_boundary_by_type =
      (kernel_type != KernelTypeEnum::kHLO && kernel_type != KernelTypeEnum::kHint);

  if (is_boundary_by_type) {
    return true;
  }

  // Check if kHLO operation contains collectives - treat as boundary if so
  // Collectives in kHLO operations should not be concatenated as they require
  // synchronization across devices and have different execution semantics
  if (kernel_type == KernelTypeEnum::kHLO) {
    auto* compilable_kernel =
        dynamic_cast<CompilableKernelExecution*>(operation->kernel_execution.get());
    if (compilable_kernel && compilable_kernel->HasCollectives()) {
      return true;
    }
  }

  // Check if buffer size limit would be exceeded - treat as boundary if so
  if (buffer_size_limit > 0 && stream) {
    auto buffer_it = buffering_impl_->operation_buffers_.find(stream);
    if (buffer_it != buffering_impl_->operation_buffers_.end() && buffer_it->second) {
      size_t current_buffer_size = buffer_it->second->size();
      if (current_buffer_size >= static_cast<size_t>(buffer_size_limit)) {
        c10::StreamId stream_id = stream->GetStreamId();

        TORCH_NEURONX_INFO(
            "[CONCATENATION_ENGINE] Buffer size limit exceeded - treating as boundary",
            "stream_id=", stream_id, "current_buffer_size=", current_buffer_size,
            "buffer_size_limit=", buffer_size_limit);
        return true;
      }
    }
  }

  return false;
}

std::list<OperationContext*> ConcatenationEngine::FlushStreamBuffer(StreamImpl* stream) {
  std::list<OperationContext*> result;

  auto buffer_it = buffering_impl_->operation_buffers_.find(stream);
  if (buffer_it == buffering_impl_->operation_buffers_.end() || !buffer_it->second ||
      buffer_it->second->empty()) {
    return result;
  }

  auto& buffer = buffer_it->second;

  // Attempt concatenation on buffered operations
  uint64_t cutoff_index = 0;
  MaybeTryConcatenate(*buffer, stream, &cutoff_index);

  // Mark ALL buffered operations as completed (regardless of concatenation result)
  for (auto* op : *buffer) {
    op->CompleteConcatenation();
    result.push_back(op);
  }

  // Clear buffer
  buffer->clear();
  buffering_impl_->operation_buffers_.erase(buffer_it);

  // Clear output address tracking for this stream
  auto output_it = buffering_impl_->output_ptrs_.find(stream);
  if (output_it != buffering_impl_->output_ptrs_.end()) {
    output_it->second.clear();
    buffering_impl_->output_ptrs_.erase(output_it);
  }

  return result;
}

bool ConcatenationEngine::IsFusibleBoundaryOperation(const OperationContext* operation) const {
  if (!operation) {
    return false;
  }
  return concatenation_core_->IsFusibleBoundaryOperation(operation->GetOpName());
}

bool ConcatenationEngine::IsInPlaceOperation(const OperationContext* operation) const {
  if (!operation || !operation->kernel_execution) {
    return false;
  }

  // Only kHLO operations can be in-place
  if (operation->GetKernelType() != KernelTypeEnum::kHLO) {
    return false;
  }

  const auto& src_ptrs = operation->kernel_execution->GetSrcPtrs();
  const auto& dst_ptrs = operation->kernel_execution->GetDstPtrs();

  // Build set from inputs (filtering out nulls) - O(n)
  std::unordered_set<void*> input_set;
  for (void* input : src_ptrs) {
    if (input != nullptr) {
      input_set.insert(input);
    }
  }

  // Check if any output exists in input set - O(m)
  for (void* output : dst_ptrs) {
    if (output != nullptr && input_set.count(output) > 0) {
      return true;  // In-place operation detected
    }
  }

  return false;
}

std::list<OperationContext*> ConcatenationEngine::BufferAndTryConcatenation(
    OperationContext* operation) {
  auto* stream = operation->stream;

  // Buffer the operation
  operation->StartConcatenation();
  AddOperationToBuffer(operation, stream);

  // Track output addresses for conflict detection
  TrackOutputAddresses(operation, stream);

  // Try normal concatenation
  return TryNormalConcatenation(stream);
}

std::list<OperationContext*> ConcatenationEngine::TryNormalConcatenation(StreamImpl* stream) {
  std::list<OperationContext*> operations_to_enqueue;

  auto buffer_it = buffering_impl_->operation_buffers_.find(stream);
  if (buffer_it == buffering_impl_->operation_buffers_.end() || !buffer_it->second ||
      buffer_it->second->empty()) {
    return operations_to_enqueue;
  }

  // Fast path: If the last operation is not a matmul, skip concatenation processing
  auto& op_contexts = *buffer_it->second;
  if (!op_contexts.empty()) {
    auto* last_op_context = op_contexts.back();
    if (last_op_context &&
        !concatenation_core_->IsFusibleBoundaryOperation(last_op_context->GetOpName())) {
      return {};
    }
  }

  return FlushStreamBuffer(stream);
}

std::list<OperationContext*> ConcatenationEngine::Flush(size_t num_to_flush) {
  // Disable accumulation mode
  if (accumulation_enabled_) {
    accumulation_enabled_ = false;
  }

  // Flush operations in global order (no concatenation)
  return FlushOperationsInGlobalOrder(num_to_flush);
}

std::list<OperationContext*> ConcatenationEngine::ProcessConcatenationTask(
    OperationContext* operation) {
  // ========================================================================
  // Validate operation and stream
  // ========================================================================
  TORCH_CHECK(operation != nullptr,
              "[CONCATENATION_ENGINE] Null operation passed to ProcessConcatenationTask");
  TORCH_CHECK(operation->stream != nullptr,
              "[CONCATENATION_ENGINE] Operation has null stream in ProcessConcatenationTask, "
              "op_name=",
              operation->GetOpName());

  // ========================================================================
  // Pre-checks (Stream Changes)
  // ========================================================================
  auto* stream = operation->stream;

  std::list<OperationContext*> result;

  // Handle stream change - flush previous stream
  bool is_stream_change = (last_processed_stream_ != nullptr && last_processed_stream_ != stream);
  if (is_stream_change) {
    auto flushed = FlushStreamBuffer(last_processed_stream_);
    result.splice(result.end(), flushed);
  }

  // ========================================================================
  // SECTION 4: Determine Operation Type
  // ========================================================================
  bool is_boundary;
  if (skip_intermediate_enabled_) {
    // New behavior with skip intermediate enabled
    is_boundary = IsNonFusibleBoundaryOperation(operation, stream) || IsInPlaceOperation(operation);
  } else {
    // Old behavior without skip intermediate
    is_boundary =
        IsNonFusibleBoundaryOperation(operation, stream) || HasOutputConflict(operation, stream);
  }
  bool is_fusible = IsFusibleBoundaryOperation(operation);

  // ========================================================================
  // SECTION 5: State Machine Logic
  // ========================================================================
  if (!accumulation_enabled_) {
    // ====== DIRECT MODE: Not buffering ======
    if (is_boundary || is_fusible) {
      // Hit a boundary/fusible op - resume accumulation
      accumulation_enabled_ = true;
      operation->CompleteConcatenation();
      result.splice(result.end(), {operation});
    } else {
      // Still in direct mode - execute without buffering
      operation->CompleteConcatenation();
      result.splice(result.end(), {operation});
    }
  } else {
    // ====== ACCUMULATION MODE: Buffering ======
    if (is_boundary) {
      // End of fusion region - flush buffer with concatenation
      auto flushed = FlushStreamBuffer(stream);
      result.splice(result.end(), flushed);

      // Add boundary operation directly for execution (don't buffer it)
      operation->CompleteConcatenation();
      result.push_back(operation);
    } else {
      // Continue accumulating - buffer and try concatenation
      auto ops = BufferAndTryConcatenation(operation);
      result.splice(result.end(), ops);
    }
  }

  // Update stream tracker
  last_processed_stream_ = stream;

  return result;
}

// maybe_try_concatenate calls ConcatenationCore to perform any concatenation on a sequence of
// oeprations, the cutoff_index is populated based till what point buffered operations have been
// concatenated and ready for execution. the operations that are after the cutoff index are still
// kept in the buffer while operations before the cutoff index are returned for execution.
std::list<OperationContext*> ConcatenationEngine::MaybeTryConcatenate(
    std::list<OperationContext*>& buffered_ops, StreamImpl* stream, uint64_t* cutoff_index) {
  // Initialize cutoff_index to 0 - will be updated as we process operations
  *cutoff_index = 0;
  if (buffered_ops.empty()) {
    return {};
  }

  // Delegate to ConcatenationCore for the actual concatenation logic
  auto result = concatenation_core_->ProcessBufferedOperations(buffered_ops);

  // Set the cutoff_index from ConcatenationCore result
  *cutoff_index = result.original_operations_consumed;

  // There's no need to replace the individual ops with concatenated ops in the active operation
  // queue, which is error prone. Instead, we can send the concatenated op to compilation and
  // directly mark individual ops ready for execution.
  return result.processed_operations;
}

void ConcatenationEngine::AddOperationToBuffer(OperationContext* operation, StreamImpl* stream) {
  // Note: This method assumes the caller already holds buffered_operations_mutex_
  // TODO: pre-allocate buffers to avoid hash map resize corruption
  auto [stream_buffer_it, inserted] = buffering_impl_->operation_buffers_.try_emplace(
      stream, std::move(std::make_unique<std::list<OperationContext*>>()));
  stream_buffer_it->second->push_back(operation);
}

void ConcatenationEngine::ClearCache() { concatenation_core_->ClearCache(); }

size_t ConcatenationEngine::GetCacheSize() const { return concatenation_core_->GetCacheSize(); }

size_t ConcatenationEngine::GetBufferedOperationsCount() const {
  std::lock_guard<std::mutex> lock(buffering_impl_->buffered_operations_mutex_);
  size_t total_count = 0;
  for (const auto& [stream, buffer] : buffering_impl_->operation_buffers_) {
    if (buffer) {
      total_count += buffer->size();
    }
  }
  return total_count;
}

void ConcatenationEngine::ProcessConcatenationFailure(OperationContext* failed_concatenated_op) {
  if (!failed_concatenated_op) {
    TORCH_NEURONX_ERROR(
        "[CONCATENATION_ENGINE] Null operation passed to processConcatenationFailure");
    throw std::runtime_error(
        "[CONCATENATION_ENGINE] Null operation passed to processConcatenationFailure");
  }

  // Get the concatenation state using the helper method
  // This handles both concatenated ops (raw ptr) and cascading ops (shared_ptr)
  auto* state = failed_concatenated_op->GetConcatenationState();
  TORCH_CHECK(state != nullptr,
              "[CONCATENATION_ENGINE] No concatenation state found for operation: ",
              failed_concatenated_op->GetOpName());

  // Invoke the failure callback stored in the ConcatenationState
  // The callback handles cache invalidation and clearing cascading ops' state
  // Note: callback is guaranteed to be set (required by ConcatenationState constructor)
  state->InvokeFailureCallback(failed_concatenated_op);
}

std::list<OperationContext*> ConcatenationEngine::FlushOperationsInGlobalOrder(
    size_t num_to_flush) {
  std::lock_guard<std::mutex> lock(buffering_impl_->buffered_operations_mutex_);

  std::list<OperationContext*> operations_to_enqueue;

  // If we have a last processed stream, prioritize flushing from it first
  if (last_processed_stream_ != nullptr) {
    auto buffer_it = buffering_impl_->operation_buffers_.find(last_processed_stream_);
    if (buffer_it != buffering_impl_->operation_buffers_.end() && buffer_it->second &&
        !buffer_it->second->empty()) {
      auto& buffer = buffer_it->second;
      size_t stream_ops_count = buffer->size();
      size_t to_flush_from_stream = std::min(num_to_flush, stream_ops_count);

      // Flush operations from last processed stream
      for (size_t i = 0; i < to_flush_from_stream; ++i) {
        OperationContext* op = buffer->front();
        op->CompleteConcatenation();
        operations_to_enqueue.push_back(op);
        buffer->pop_front();
      }

      // Clean up empty buffer
      if (buffer->empty()) {
        buffering_impl_->operation_buffers_.erase(buffer_it);
      }

      num_to_flush -= to_flush_from_stream;
    }
  }

  // If we still need to flush more, flush from other streams
  if (num_to_flush > 0) {
    std::vector<StreamImpl*> streams_to_erase;
    for (auto& [stream, buffer] : buffering_impl_->operation_buffers_) {
      if (num_to_flush <= 0) break;

      if (!buffer || buffer->empty()) continue;

      while (!buffer->empty() && num_to_flush > 0) {
        OperationContext* op = buffer->front();
        op->CompleteConcatenation();
        operations_to_enqueue.push_back(op);
        buffer->pop_front();
        num_to_flush--;
      }

      // Clean up empty buffer
      if (buffer->empty()) {
        streams_to_erase.push_back(stream);
      }
    }

    for (auto* stream : streams_to_erase) {
      buffering_impl_->operation_buffers_.erase(stream);
    }
  }

  // Clear ALL output address tracking across all streams after a global flush
  buffering_impl_->output_ptrs_.clear();

  return operations_to_enqueue;
}

bool ConcatenationEngine::HasOutputConflict(const OperationContext* operation,
                                            StreamImpl* stream) const {
  // Only check kHLO operations for output conflicts (they have dst pointers)
  if (!operation || operation->GetKernelType() != KernelTypeEnum::kHLO) {
    return false;
  }

  auto output_it = buffering_impl_->output_ptrs_.find(stream);
  if (output_it == buffering_impl_->output_ptrs_.end()) {
    return false;
  }

  // Get output pointers from the operation
  const auto& dst_ptrs = operation->kernel_execution->GetDstPtrs();

  // Check if any output pointer already exists in tracked outputs
  for (void* output_ptr : dst_ptrs) {
    if (output_ptr && output_it->second.count(output_ptr) > 0) {
      return true;
    }
  }

  return false;
}

void ConcatenationEngine::TrackOutputAddresses(const OperationContext* operation,
                                               StreamImpl* stream) {
  // Only track outputs for kHLO operations (they have dst pointers)
  if (!operation || operation->GetKernelType() != KernelTypeEnum::kHLO) {
    return;
  }

  // Get output pointers from the operation
  const auto& dst_ptrs = operation->kernel_execution->GetDstPtrs();

  // Track all output pointers
  for (void* output_ptr : dst_ptrs) {
    if (output_ptr) {
      buffering_impl_->output_ptrs_[stream].insert(output_ptr);
    }
  }
}

}  // namespace at::neuron
