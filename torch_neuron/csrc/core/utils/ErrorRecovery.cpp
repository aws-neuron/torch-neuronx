#include "ErrorRecovery.h"

#include <cstdlib>

#include "CPUFallbackExecutor.h"

namespace at::neuron {

ErrorRecovery::ErrorRecovery() : stats_{0, 0, 0}, fallback_enabled_(true) {
  LoadConfiguration();
  // Initialize CPU fallback executor
  cpu_fallback_executor_ = std::make_unique<CPUFallbackExecutor>();
}

void ErrorRecovery::LoadConfiguration() {
  // Load configuration from environment variables
  const char* fallback_env = std::getenv("NEURON_FALLBACK_ENABLED");
  if (fallback_env) {
    fallback_enabled_ = (std::string(fallback_env) == "1" || std::string(fallback_env) == "true");
  }
}

OperationContextResult ErrorRecovery::AttemptRecovery(OperationContext* op) {
  if (ShouldAttemptCpuFallback()) {
    return ExecuteCpuFallback(op);
  }
  // Return an empty result, denoting a pending/unsubmitted execution.
  return {};
}

bool ErrorRecovery::ShouldAttemptCpuFallback() {
  if (!fallback_enabled_) {
    return false;
  }

  // Simple decision: if fallback is enabled, try it
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stats_.total_fallback_attempts++;
  return true;
}

OperationContextResult ErrorRecovery::ExecuteCpuFallback(OperationContext* op) {
  // Execute CPU fallback
  auto result = cpu_fallback_executor_->ExecuteCpuFallback(op);

  std::lock_guard<std::mutex> lock(stats_mutex_);
  if (result.status == 0) {  // Success
    stats_.successful_fallbacks++;
  } else {
    stats_.failed_fallbacks++;
  }

  return result;
}

ErrorRecovery::RecoveryStats ErrorRecovery::GetRecoveryStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

void ErrorRecovery::ResetRecoveryStats() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stats_ = {0, 0, 0};
}

}  // namespace at::neuron
