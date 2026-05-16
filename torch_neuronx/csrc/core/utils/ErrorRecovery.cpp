#include "ErrorRecovery.h"

#include <cstdlib>

#include "CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

namespace at::neuron {

ErrorRecovery::ErrorRecovery() : fallback_enabled_(true), oom_recovery_enabled_(true) {
  LoadConfiguration();
  cpu_fallback_executor_ = std::make_unique<CPUFallbackExecutor>();
}

ErrorRecovery::~ErrorRecovery() = default;

void ErrorRecovery::LoadConfiguration() {
  const char* fallback_env = std::getenv("NEURON_FALLBACK_ENABLED");
  if (fallback_env) {
    fallback_enabled_ = (std::string(fallback_env) == "1" || std::string(fallback_env) == "true");
  }
  const char* oom_env = std::getenv("NEURON_OOM_RECOVERY_ENABLED");
  if (oom_env) {
    oom_recovery_enabled_ = (std::string(oom_env) == "1" || std::string(oom_env) == "true");
  }
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* op, const torch_neuronx::CompilationRuntimeException& e) {
  auto* kernel = op->kernel_execution.get();
  EXECUTION_RUNTIME_CHECK(kernel->RequiresCompilation(),
                          "This operation requires a compilable kernel");
  return AttemptCompilableRecovery(op);
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* operation, const torch_neuronx::ExecutionRuntimeException& e) {
  if (ShouldAttemptOOMRecovery(e) && operation->GetKernelType() == KernelTypeEnum::kAlloc) {
    return AttemptOOMRecovery(operation);
  }
  return {};
}

OperationContextResult ErrorRecovery::AttemptCompilableRecovery(OperationContext* operation) {
  EXECUTION_RUNTIME_CHECK(operation->RequiresCompilation(),
                          "This operation requires a compilable kernel");

  // Handle concatenation failure first - invalidate cache and cleanup
  // For concatenated operations, we fall back to op-by-op execution
  if (operation->HasConcatenatedOperation() || operation->IsConcatenatedOperation()) {
    RecoverConcatenationFailure(operation);
    // For concatenated operations themselves, signal fallback and return success
    // so the individual ops can be retried
    if (operation->IsConcatenatedOperation() && !operation->HasConcatenatedOperation()) {
      TORCH_NEURONX_DEBUG("Concatenated operation compilation failed, falling back to op-by-op",
                          "stream_id=", operation->stream->stream_id,
                          "op=", operation->GetOpName());
      return OperationContextResult::CreateSuccess();
    }
  }

  if (!ShouldAttemptCpuFallback()) {
    return {};
  }

  return ExecuteCpuFallback(operation);
}

bool ErrorRecovery::ShouldAttemptCpuFallback() { return fallback_enabled_; }

OperationContextResult ErrorRecovery::ExecuteCpuFallback(OperationContext* op) {
  return cpu_fallback_executor_->ExecuteCpuFallback(op);
}

bool ErrorRecovery::ShouldAttemptOOMRecovery(const torch_neuronx::ExecutionRuntimeException& e) {
  return oom_recovery_enabled_ && e.HasNRTStatus() && e.GetNRTStatus() == NRT_RESOURCE;
}

OperationContextResult ErrorRecovery::AttemptOOMRecovery(OperationContext* op) {
  TORCH_NEURONX_DEBUG("Attempting OOM recovery via cache drain", "op=", op->GetOpName());

  const int device_id = op->kernel_execution->GetDeviceId();
  metrics::RecordOOMRetry(device_id);

  size_t requested_size = 0;
  if (op->GetKernelType() == KernelTypeEnum::kAlloc) {
    requested_size = op->GetKernel<KernelTypeEnum::kAlloc>().GetSizeBytes();
  }

  c10_neuron::NeuronCachingAllocator::emptyNrtBufferCache();
  try {
    NRT_STATUS nrt_status = NRT_SUCCESS;
    nrt::SequenceId seq_id = 0;
    op->kernel_execution->Schedule(&nrt_status, &seq_id, op->stream_id);
    if (nrt_status == NRT_RESOURCE) {
      TORCH_NEURONX_WARN("OOM recovery failed after cache drain", "op=", op->GetOpName(),
                         "nrt_status=NRT_RESOURCE");
      metrics::RecordOOMFailure(device_id, requested_size, op->stream_id, nullptr);
      return OperationContextResult::CreateError(
          "OOM recovery failed: nrt_tensor_allocate still failed after cache drain");
    }
    TORCH_NEURONX_DEBUG("OOM recovery succeeded", "op=", op->GetOpName());
    return OperationContextResult::CreateSuccess();
  } catch (const torch_neuronx::ExecutionRuntimeException& retry_e) {
    TORCH_NEURONX_WARN("OOM recovery failed after cache drain", "op=", op->GetOpName(),
                       "error=", retry_e.what());
    // Only count as an OOM if the retry's exception was actually NRT_RESOURCE.
    // Other exceptions (e.g. a downstream op throwing on the retry path) are
    // recovery failures but not OOMs and must not inflate num_ooms telemetry.
    if (retry_e.HasNRTStatus() && retry_e.GetNRTStatus() == NRT_RESOURCE) {
      metrics::RecordOOMFailure(device_id, requested_size, op->stream_id, nullptr);
    }
    return OperationContextResult::CreateError(std::string("OOM recovery failed: ") +
                                               retry_e.what());
  }
}

void RecoverConcatenationFailure(OperationContext* operation) {
  if (!operation) {
    TORCH_NEURONX_DEBUG("RecoverConcatenationFailure called with null operation");
    return;
  }

  // Determine which operation to use for the callback
  OperationContext* concat_op = nullptr;
  ConcatenationState* state = nullptr;

  if (operation->IsConcatenatedOperation()) {
    // This IS the concatenated operation - use raw pointer to get state
    // (concatenated ops use concatenation_state_raw_ to avoid circular reference)
    concat_op = operation;
    state = operation->concatenation_state_raw_;
  } else if (operation->HasConcatenatedOperation()) {
    // This is an individual cascading operation that has a concatenated operation
    // (cascading ops use concatenation_state_ shared_ptr)
    concat_op = operation->GetConcatenatedOperation();
    state = operation->concatenation_state_.get();
  }

  if (state && concat_op) {
    TORCH_NEURONX_DEBUG("Invoking concatenation failure callback",
                        "concat_op=", concat_op->GetOpName());
    state->InvokeFailureCallback(concat_op);
  } else {
    TORCH_NEURONX_DEBUG("No concatenation state or callback to invoke",
                        "op=", operation->GetOpName());
  }
}

}  // namespace at::neuron
