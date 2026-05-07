#include "ConcatenationEngine.h"

#include <climits>
#include <cstdlib>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationCore.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron {

// Pimpl implementation class to hide implementation details from header
class ConcatenationEngine::BufferingImpl {
 public:
  // Single global buffer - using std::list for O(1) front removal
  std::list<OperationContext*> operation_buffer_;

  // Track output addresses for conflict detection
  std::unordered_set<void*> output_ptrs_;
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
  if (buffer_size_limit > 0) {
    size_t current_buffer_size = buffering_impl_->operation_buffer_.size();
    if (current_buffer_size >= static_cast<size_t>(buffer_size_limit)) {
      c10::StreamId stream_id = stream ? stream->GetStreamId() : -1;

      TORCH_NEURONX_INFO("[CONCATENATION_ENGINE] Buffer size limit exceeded - treating as boundary",
                         "stream_id=", stream_id, "current_buffer_size=", current_buffer_size,
                         "buffer_size_limit=", buffer_size_limit);
      // Buffer size limit is configurable via TORCH_NEURONX_CONCATENATION_BUFFER_SIZE_LIMIT env var
      TORCH_NEURONX_COUNTER("Concatenation.BufferLimitHits", 1);
      return true;
    }
  }

  return false;
}

std::list<OperationContext*> ConcatenationEngine::FlushStreamBuffer(StreamImpl* stream) {
  std::list<OperationContext*> result;

  auto& buffer = buffering_impl_->operation_buffer_;
  if (buffer.empty()) {
    return result;
  }

  TORCH_NEURONX_COUNTER("Concatenation.BoundaryFlushes", 1);
  TORCH_NEURONX_METRIC("Concatenation.BufferSize", static_cast<double>(buffer.size()));

  // Attempt concatenation on buffered operations
  uint64_t cutoff_index = 0;
  MaybeTryConcatenate(buffer, stream, &cutoff_index);

  // Mark ALL buffered operations as completed (regardless of concatenation result)
  for (auto* op : buffer) {
    op->CompleteConcatenation();
    result.push_back(op);
  }

  // Clear buffer
  buffer.clear();

  // Clear output address tracking
  buffering_impl_->output_ptrs_.clear();

  // ASSERTION: After flush, buffer must be empty
  TORCH_CHECK(buffering_impl_->operation_buffer_.empty(),
              "[CONCATENATION_ENGINE] Buffer not empty after FlushStreamBuffer! "
              "remaining_ops=",
              buffering_impl_->operation_buffer_.size(),
              " This indicates a bug in buffer management.");

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

  auto& buffer = buffering_impl_->operation_buffer_;
  if (buffer.empty()) {
    return operations_to_enqueue;
  }

  // Fast path: If the last operation is not a matmul, skip concatenation processing
  if (!buffer.empty()) {
    auto* last_op_context = buffer.back();
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
  auto concat_task_start = std::chrono::steady_clock::now();

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
    TORCH_NEURONX_COUNTER("Concatenation.StreamChangeFlushes", 1);
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

  auto concat_task_end = std::chrono::steady_clock::now();
  // Time-series metric: each sample is one invocation's duration (nanoseconds).
  // The metrics system stores all samples and computes accumulator (total), count, etc.
  TORCH_NEURONX_TIME_METRIC("Concatenation.ProcessingTime", concat_task_start, concat_task_end);

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

  // Concatenation.Attempts counts how many times we attempted concatenation.
  // Corresponding Concatenation.Successes and Concatenation.Failures are tracked
  // in ConcatenationCore::ProcessBufferedOperationsImpl and ProcessConcatenationFailure.
  TORCH_NEURONX_COUNTER("Concatenation.Attempts", 1);

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
  buffering_impl_->operation_buffer_.push_back(operation);
}

void ConcatenationEngine::ClearCache() { concatenation_core_->ClearCache(); }

size_t ConcatenationEngine::GetCacheSize() const { return concatenation_core_->GetCacheSize(); }

size_t ConcatenationEngine::GetBufferedOperationsCount() const {
  return buffering_impl_->operation_buffer_.size();
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
  TORCH_NEURONX_COUNTER("Concatenation.Failures", 1);
  state->InvokeFailureCallback(failed_concatenated_op);
}

std::list<OperationContext*> ConcatenationEngine::FlushOperationsInGlobalOrder(
    size_t num_to_flush) {
  std::list<OperationContext*> operations_to_enqueue;
  auto& buffer = buffering_impl_->operation_buffer_;
  size_t to_flush = std::min(num_to_flush, buffer.size());

  // Flush operations from the single global buffer
  for (size_t i = 0; i < to_flush; ++i) {
    OperationContext* op = buffer.front();
    op->CompleteConcatenation();
    operations_to_enqueue.push_back(op);
    buffer.pop_front();
  }

  // Clear output address tracking after flush
  if (buffer.empty()) {
    buffering_impl_->output_ptrs_.clear();
  }

  return operations_to_enqueue;
}

bool ConcatenationEngine::HasOutputConflict(const OperationContext* operation,
                                            StreamImpl* stream) const {
  // Only check kHLO operations for output conflicts (they have dst pointers)
  if (!operation || operation->GetKernelType() != KernelTypeEnum::kHLO) {
    return false;
  }

  const auto& tracked_outputs = buffering_impl_->output_ptrs_;
  if (tracked_outputs.empty()) {
    return false;
  }

  // Get output pointers from the operation
  const auto& dst_ptrs = operation->kernel_execution->GetDstPtrs();

  // Check if any output pointer already exists in tracked outputs
  for (void* output_ptr : dst_ptrs) {
    if (output_ptr && tracked_outputs.count(output_ptr) > 0) {
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
      buffering_impl_->output_ptrs_.insert(output_ptr);
    }
  }
}

}  // namespace at::neuron
