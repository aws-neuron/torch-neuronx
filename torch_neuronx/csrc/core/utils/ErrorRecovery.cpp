#include "ErrorRecovery.h"

#include <cstdlib>

#include "CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {

ErrorRecovery::ErrorRecovery() : fallback_enabled_(true), oom_recovery_enabled_(true) {
  LoadConfiguration();
  // Initialize CPU fallback executor
  cpu_fallback_executor_ = std::make_unique<CPUFallbackExecutor>();
}

ErrorRecovery::~ErrorRecovery() = default;

void ErrorRecovery::LoadConfiguration() {
  // Load configuration from environment variables
  const char* fallback_env = std::getenv("NEURON_FALLBACK_ENABLED");
  if (fallback_env) {
    fallback_enabled_ = (std::string(fallback_env) == "1" || std::string(fallback_env) == "true");
  }

  const char* oom_recovery_env = std::getenv("NEURON_OOM_RECOVERY_ENABLED");
  if (oom_recovery_env) {
    oom_recovery_enabled_ =
        (std::string(oom_recovery_env) == "1" || std::string(oom_recovery_env) == "true");
  }
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* op, const torch_neuronx::CompilationRuntimeException& e) {
  // Compilation errors should only be pertinent for compilable kernels.
  auto* kernel = op->kernel_execution.get();
  TORCH_CHECK(kernel->RequiresCompilation(), "This operation requires a compilable kernel");
  return AttemptCompilableRecovery(op);
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    OperationContext* operation, const torch_neuronx::ExecutionRuntimeException& e) {
  if (operation->RequiresCompilation()) {
    return AttemptCompilableRecovery(operation);
  }
  switch (operation->GetKernelType()) {
    case KernelTypeEnum::kAlloc: {
      auto& kernel = operation->GetKernel<KernelTypeEnum::kAlloc>();
      return AttemptRecovery(kernel, e);
    }
    default:
      return {};
  }
}

OperationContextResult ErrorRecovery::AttemptCompilableRecovery(OperationContext* operation) {
  TORCH_CHECK(operation->RequiresCompilation(), "This operation requires a compilable kernel");
  // As of now, we only support CPU fallback for compilable kernels.
  if (!ShouldAttemptCpuFallback()) {
    return {};
  }

  auto* stream = operation->stream;
  // In case of compilable failures, we need to ensure that all prior operations complete - as
  // preceding operations may be pending execution.
  stream->WaitForPriorOperationsToComplete(operation);
  return ExecuteCpuFallback(operation);
}

bool ErrorRecovery::ShouldAttemptCpuFallback() { return fallback_enabled_; }

bool ErrorRecovery::ShouldAttemptOOMRecovery(const torch_neuronx::ExecutionRuntimeException& e) {
  if (!oom_recovery_enabled_) {
    return false;
  }
  if (!e.HasNRTStatus()) {
    return false;
  }
  size_t cached_blocks = c10_neuron::NeuronCachingAllocator::getCachedBlocks();
  if (cached_blocks == 0 || e.GetNRTStatus() != NRT_RESOURCE) {
    return false;
  }
  return true;
}

OperationContextResult ErrorRecovery::ExecuteCpuFallback(OperationContext* op) {
  // Execute CPU fallback
  auto result = cpu_fallback_executor_->ExecuteCpuFallback(op);

  return result;
}

OperationContextResult ErrorRecovery::AttemptRecovery(
    const AllocDirectKernelExecution& kernel, const torch_neuronx::ExecutionRuntimeException& e) {
  if (!ShouldAttemptOOMRecovery(e)) {
    return {};
  }

  try {
    c10_neuron::NeuronCachingAllocator::emptyCache();
    kernel.Execute();
    return OperationContextResult::CreateSuccess();
  } catch (const std::exception& e) {
    return OperationContextResult::CreateError(e.what());
  }
}

}  // namespace at::neuron
