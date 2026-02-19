#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Include NRT headers for NRT_STATUS type
extern "C" {
#include <nrt/nrt.h>
}

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/ErrorRecovery.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

namespace at::neuron {

// Forward declarations
class CPUFallbackExecutor;

/**
 * ErrorContext provides detailed context information for error handling
 * This helps with debugging and error recovery decisions
 */
struct ErrorContext {
  // Operation identification
  std::string operation_name;
  uint32_t stream_id;

  // Pipeline stage information
  enum class Stage {
    SUBMISSION,
    COMPILATION,
    EXECUTION,
  };
  Stage current_stage;

  // Timing information
  std::chrono::steady_clock::time_point submit_time;
  std::chrono::steady_clock::time_point error_time;

  // Additional context
  std::string hlo_dump_path;  // Path to HLO dump for compilation errors
  std::string neff_path;      // Path to NEFF file if available

  std::string error_details;

  ErrorContext() : stream_id(0), current_stage(Stage::SUBMISSION) {
    error_time = std::chrono::steady_clock::now();
  }

  // Calculate time elapsed since submission
  std::chrono::milliseconds get_elapsed_time() const {
    if (submit_time.time_since_epoch().count() == 0) {
      return std::chrono::milliseconds(0);
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(error_time - submit_time);
  }

  // Get human-readable stage name
  std::string get_stage_name() const {
    switch (current_stage) {
      case Stage::SUBMISSION:
        return "SUBMISSION";
      case Stage::COMPILATION:
        return "COMPILATION";
      case Stage::EXECUTION:
        return "EXECUTION";
      default:
        return "UNKNOWN";
    }
  }
};

/**
 * ErrorHandler provides centralized error handling and recovery for the pipeline
 *
 * This class handles error propagation, classification, recovery attempts,
 * and debugging support throughout the multi-stream pipeline.
 */
class ErrorHandler {
 public:
  /**
   * Constructor for per-stream ErrorHandler instances
   */
  ErrorHandler();

  /**
   * Handle compilation errors
   * @param op The operation that failed
   * @param exception The compilation exception
   * @param context Additional error context
   * @return complete error message as std::string
   */
  std::string HandleCompilationError(OperationContext* op, const std::string& error_message,
                                     ErrorContext& context);

  /**
   * Handle execution errors
   * @param op The operation that failed
   * @param exception The execution exception
   * @param context Additional error context
   * @return complete error message as std::string
   */
  std::string HandleExecutionError(OperationContext* op, const std::string& error_message,
                                   ErrorContext& context);

  /**
   * Handle compilation errors with recovery strategies and exception tracking
   * @param op The operation that failed
   * @param e The compilation exception
   * @return OperationContextResult indicating success (recovery) or error
   */
  OperationContextResult HandleOperationError(OperationContext* op,
                                              const torch_neuronx::CompilationRuntimeException& e);

  /**
   * Handle execution errors with exception tracking
   * @param op The operation that failed
   * @param e The execution exception
   * @return OperationContextResult indicating success (recovery) or error
   */
  OperationContextResult HandleOperationError(OperationContext* op,
                                              const torch_neuronx::ExecutionRuntimeException& e);

  /**
   * Create enhanced error context for an operation
   * @param op The operation to create context for
   * @param stage The current pipeline stage
   * @return ErrorContext with populated information
   */
  ErrorContext CreateErrorContext(OperationContext* op, ErrorContext::Stage stage) const;

  /**
   * Classify an NRT error message to determine error handling strategy
   * @param e The ExecutionRuntimeException exception
   * @param context The error context
   * @return Readable error description
   */
  std::string ClassifyNrtError(const torch_neuronx::ExecutionRuntimeException& e) const;

  /**
   * Classify an NRT status code to human-readable error description
   * @param nrt_status The NRT status code
   * @return Readable error description
   */
  std::string ClassifyNrtError(NRT_STATUS nrt_status) const;

  /**
   * Generate comprehensive error message with context
   * @param base_message The base error message
   * @param context The error context
   * @return Enhanced error message with debugging information
   */
  std::string GenerateErrorMessage(const std::string& base_message,
                                   const ErrorContext& context) const;

  /**
   * Get recent error history for debugging
   * @param max_errors Maximum number of recent errors to return
   * @return Vector of recent error contexts
   */
  std::vector<ErrorContext> GetRecentErrors(size_t max_errors = 50) const;

  /**
   * Check if error handling is enabled via environment variables
   * @return true if enhanced error handling is enabled
   */
  bool IsEnhancedErrorHandlingEnabled() const;

  /**
   * Check if HLO dumping is enabled for debugging
   * @return true if HLO dumping is enabled
   */
  bool IsHloDumpingEnabled() const;

  /**
   * Check if execution context dumping is enabled for debugging
   * @return true if execution context dumping is enabled
   */
  bool IsExecutionDumpingEnabled() const;

  /**
   * Convert exception to human-readable string with type information
   * @param exception The exception to convert
   * @return Human-readable exception description
   */
  static std::string ExceptionToString(const std::exception& exception);

  /**
   * Clean error message by removing stacktraces and keeping only meaningful error
   * @param error_message The raw error message
   * @return Cleaned error message
   */
  static std::string CleanErrorMessage(const std::string& error_message);

  // Exception tracking helpers
  void CheckAndThrowPendingException(c10::StreamId stream_id, bool clear_exception);

  /**
   * Record first exception
   * @param stream_id The stream ID
   * @param exception The exception to record
   */
  void RecordFirstException(uint32_t stream_id, std::exception_ptr exception);

  // Exception tracking for proper error propagation, to avoid
  // cascading too many failures.
  bool has_pending_exception_ = false;
  mutable std::mutex exception_mutex_;
  std::exception_ptr first_exception_;

  /**
   * Destructor
   */
  ~ErrorHandler() = default;

  // Disable copy and assignment for safety
  ErrorHandler(const ErrorHandler&) = delete;
  ErrorHandler& operator=(const ErrorHandler&) = delete;

 private:
  /**
   * Record error in statistics and history
   * @param context The error context
   * @param processing_time Time spent handling this error
   */
  void RecordError(const ErrorContext& context, std::chrono::microseconds processing_time);

  /**
   * Get debug dump directory path
   * @return Path to directory for debug dumps
   */
  std::string GetDebugDumpDirectory() const;

  /**
   * Ensure debug dump directory exists
   * @param directory_path Path to create
   * @return true if directory exists or was created successfully
   */
  bool EnsureDebugDirectoryExists(const std::string& directory_path) const;

  /**
   * Generate unique filename for debug dumps
   * @param prefix Filename prefix
   * @param extension File extension
   * @param context Error context for unique identification
   * @return Unique filename
   */
  std::string GenerateDebugFilename(const std::string& prefix, const std::string& extension,
                                    const ErrorContext& context) const;

  // Recent error history
  mutable std::mutex history_mutex_;
  std::vector<ErrorContext> recent_errors_;
  static constexpr size_t MAX_RECENT_ERRORS = 100;

  // Configuration flags (cached from environment variables)
  bool enhanced_error_handling_enabled_;
  bool hlo_dumping_enabled_;
  bool execution_dumping_enabled_;
  std::string debug_dump_directory_;

  /**
   * Record error for monitoring only (does NOT set promise)
   * Consolidates all stage-specific error recording into a single method
   * @param op The operation context
   * @param context The error context (contains stage information)
   */
  void RecordErrorForMonitoring(OperationContext* op, const ErrorContext& context);
  // Per-stream error recovery (owns CPUFallbackExecutor)
  std::unique_ptr<ErrorRecovery> error_recovery_;
};

}  // namespace at::neuron
