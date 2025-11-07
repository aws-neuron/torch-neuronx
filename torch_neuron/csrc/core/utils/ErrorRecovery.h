#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "torch_neuron/csrc/core/OperationContext.h"

namespace at::neuron {

// Forward declarations
class CPUFallbackExecutor;

/**
 * ErrorRecovery provides simple CPU fallback for failed Neuron operations
 * Per-stream instance that attempts CPU fallback when Neuron execution fails
 */
class ErrorRecovery {
 public:
  /**
   * Constructor for per-stream ErrorRecovery instance
   */
  ErrorRecovery();

  /**
   * Attempt to recover from an operation failure
   * This method encapsulates all recovery logic including:
   * - Deciding whether to attempt CPU fallback
   * - Executing CPU fallback if appropriate
   * - Setting the operation promise with the result
   *
   * @param op The failed operation
   * @return OperationContextResult
   */
  OperationContextResult AttemptRecovery(OperationContext* op);

  /**
   * Check if we should attempt CPU fallback for a failed operation
   * Simple decision: if fallback is enabled, try it; otherwise fail
   */
  bool ShouldAttemptCpuFallback();

  /**
   * Execute CPU fallback for a failed operation
   */
  OperationContextResult ExecuteCpuFallback(OperationContext* op);

  /**
   * Get fallback statistics for monitoring
   */
  struct RecoveryStats {
    uint64_t total_fallback_attempts;
    uint64_t successful_fallbacks;
    uint64_t failed_fallbacks;

    double fallback_success_rate() const {
      if (total_fallback_attempts == 0) return 0.0;
      return double(successful_fallbacks) / double(total_fallback_attempts);
    }
  };

  RecoveryStats GetRecoveryStats() const;

  /**
   * Reset recovery statistics
   */
  void ResetRecoveryStats();

  /**
   * Destructor
   */
  ~ErrorRecovery() = default;

 private:
  // Disable copy and assignment
  ErrorRecovery(const ErrorRecovery&) = delete;
  ErrorRecovery& operator=(const ErrorRecovery&) = delete;

  /**
   * Load configuration from environment variables
   */
  void LoadConfiguration();

  // Statistics
  mutable std::mutex stats_mutex_;
  RecoveryStats stats_;

  // Configuration flags
  bool fallback_enabled_;

  // CPU fallback executor
  std::unique_ptr<CPUFallbackExecutor> cpu_fallback_executor_;
};

}  // namespace at::neuron
