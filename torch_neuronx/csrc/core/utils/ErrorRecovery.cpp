#include "ErrorRecovery.h"

#include <cstdlib>

#include "CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron {

ErrorRecovery::ErrorRecovery() : fallback_enabled_(true) {
  LoadConfiguration();
  cpu_fallback_executor_ = std::make_unique<CPUFallbackExecutor>();
}

ErrorRecovery::~ErrorRecovery() = default;

void ErrorRecovery::LoadConfiguration() {
  const char* fallback_env = std::getenv("NEURON_FALLBACK_ENABLED");
  if (fallback_env) {
    fallback_enabled_ = (std::string(fallback_env) == "1" || std::string(fallback_env) == "true");
  }
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* op, const torch_neuronx::CompilationRuntimeException& e) {
  auto* kernel = op->kernel_execution.get();
  TORCH_CHECK(kernel->RequiresCompilation(), "This operation requires a compilable kernel");
  return AttemptCompilableRecovery(op);
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* operation, const torch_neuronx::ExecutionRuntimeException& e) {
  if (operation->RequiresCompilation()) {
    return AttemptCompilableRecovery(operation);
  }
  return {};
}

OperationContextResult ErrorRecovery::AttemptCompilableRecovery(OperationContext* operation) {
  TORCH_CHECK(operation->RequiresCompilation(), "This operation requires a compilable kernel");

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

  auto* stream = operation->stream;
  stream->WaitForPriorOperationsToComplete(operation);
  return ExecuteCpuFallback(operation);
}

bool ErrorRecovery::ShouldAttemptCpuFallback() { return fallback_enabled_; }

OperationContextResult ErrorRecovery::ExecuteCpuFallback(OperationContext* op) {
  return cpu_fallback_executor_->ExecuteCpuFallback(op);
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
