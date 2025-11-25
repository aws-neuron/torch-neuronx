#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/ErrorRecovery.h"

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
  std::string hlo_dump_path;               // Path to HLO dump for compilation errors
  std::string neff_path;                   // Path to NEFF file if available
  std::vector<std::string> input_shapes;   // Input tensor shapes for debugging
  std::vector<std::string> output_shapes;  // Expected output tensor shapes

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
 * ErrorStatistics tracks error patterns for monitoring and debugging
 */
struct ErrorStatistics {
  // Error counts by type
  std::atomic<uint64_t> compilation_errors{0};
  std::atomic<uint64_t> execution_errors{0};
  std::atomic<uint64_t> timeout_errors{0};
  std::atomic<uint64_t> resource_errors{0};
  std::atomic<uint64_t> user_errors{0};
  std::atomic<uint64_t> transient_errors{0};

  // Error counts by stage
  std::atomic<uint64_t> submission_stage_errors{0};
  std::atomic<uint64_t> lowering_stage_errors{0};
  std::atomic<uint64_t> compilation_stage_errors{0};
  std::atomic<uint64_t> execution_stage_errors{0};

  // Default constructor
  ErrorStatistics() = default;

  // Copy constructor - copy atomic values
  ErrorStatistics(const ErrorStatistics& other)
      : compilation_errors(other.compilation_errors.load()),
        execution_errors(other.execution_errors.load()),
        timeout_errors(other.timeout_errors.load()),
        resource_errors(other.resource_errors.load()),
        user_errors(other.user_errors.load()),
        transient_errors(other.transient_errors.load()),
        submission_stage_errors(other.submission_stage_errors.load()),
        lowering_stage_errors(other.lowering_stage_errors.load()),
        compilation_stage_errors(other.compilation_stage_errors.load()),
        execution_stage_errors(other.execution_stage_errors.load()),
        retry_attempts(other.retry_attempts.load()),
        successful_retries(other.successful_retries.load()),
        failed_retries(other.failed_retries.load()),
        total_error_handling_time_us(other.total_error_handling_time_us.load()),
        error_handling_calls(other.error_handling_calls.load()) {}

  // Assignment operator - assign atomic values
  ErrorStatistics& operator=(const ErrorStatistics& other) {
    if (this != &other) {
      compilation_errors.store(other.compilation_errors.load());
      execution_errors.store(other.execution_errors.load());
      timeout_errors.store(other.timeout_errors.load());
      resource_errors.store(other.resource_errors.load());
      user_errors.store(other.user_errors.load());
      transient_errors.store(other.transient_errors.load());
      submission_stage_errors.store(other.submission_stage_errors.load());
      lowering_stage_errors.store(other.lowering_stage_errors.load());
      compilation_stage_errors.store(other.compilation_stage_errors.load());
      execution_stage_errors.store(other.execution_stage_errors.load());
      retry_attempts.store(other.retry_attempts.load());
      successful_retries.store(other.successful_retries.load());
      failed_retries.store(other.failed_retries.load());
      total_error_handling_time_us.store(other.total_error_handling_time_us.load());
      error_handling_calls.store(other.error_handling_calls.load());
    }
    return *this;
  }

  // Reset all counters to zero
  void reset() {
    compilation_errors.store(0);
    execution_errors.store(0);
    timeout_errors.store(0);
    resource_errors.store(0);
    user_errors.store(0);
    transient_errors.store(0);
    submission_stage_errors.store(0);
    lowering_stage_errors.store(0);
    compilation_stage_errors.store(0);
    execution_stage_errors.store(0);
    retry_attempts.store(0);
    successful_retries.store(0);
    failed_retries.store(0);
    total_error_handling_time_us.store(0);
    error_handling_calls.store(0);
  }

  // Recovery statistics
  std::atomic<uint64_t> retry_attempts{0};
  std::atomic<uint64_t> successful_retries{0};
  std::atomic<uint64_t> failed_retries{0};

  // Timing statistics
  std::atomic<uint64_t> total_error_handling_time_us{0};
  std::atomic<uint64_t> error_handling_calls{0};

  // Get average error handling time
  uint64_t get_avg_error_handling_time_us() const {
    uint64_t calls = error_handling_calls.load();
    if (calls == 0) return 0;
    return total_error_handling_time_us.load() / calls;
  }

  // Get total error count
  uint64_t get_total_errors() const {
    return compilation_errors.load() + execution_errors.load() + timeout_errors.load() +
           resource_errors.load() + user_errors.load() + transient_errors.load();
  }

  // Get error rate (errors per total operations)
  double get_error_rate(uint64_t total_operations) const {
    if (total_operations == 0) return 0.0;
    return double(get_total_errors()) / double(total_operations);
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
   * Handle operation errors with CPU fallback and exception tracking
   * This is the main entry point for StreamImpl error handling
   * @param op The operation that failed
   * @param error_message The error message
   * @param stage The pipeline stage where the error occurred
   * @param stream_id The stream ID for exception tracking
   * @param error_code Optional error code for execution errors
   * @return OperationContextResult indicating success (CPU fallback) or error
   */
  OperationContextResult HandleOperationError(OperationContext* op,
                                              const std::string& error_message,
                                              ErrorContext::Stage stage, uint32_t stream_id,
                                              int error_code = 0);

  /**
   * Create enhanced error context for an operation
   * @param op The operation to create context for
   * @param stage The current pipeline stage
   * @return ErrorContext with populated information
   */
  ErrorContext CreateErrorContext(OperationContext* op, ErrorContext::Stage stage) const;

  /**
   * Extract NRT status code from error message
   * @param error_message The error message containing "status: N"
   * @return The extracted status code, or -999 if not found
   */
  int ExtractNrtStatus(const std::string& error_message) const;

  /**
   * Classify an NRT error message to determine error handling strategy
   * @param error_message The NRT error message with error code
   * @param context The error context
   * @return Pair of updated context and readable error description
   */
  std::pair<ErrorContext, std::string> ClassifyNrtError(const std::string& error_message,
                                                        ErrorContext context) const;

  /**
   * Generate comprehensive error message with context
   * @param base_message The base error message
   * @param context The error context
   * @return Enhanced error message with debugging information
   */
  std::string GenerateErrorMessage(const std::string& base_message,
                                   const ErrorContext& context) const;

  /**
   * Get error statistics for monitoring
   * @return Current error statistics
   */
  ErrorStatistics GetErrorStatistics() const;

  /**
   * Reset error statistics (useful for testing)
   */
  void ResetErrorStatistics();

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

  /**
   * Check and rethrow any active pending exception
   * @param stream_id The stream ID
   * @param clear_exception Whether to clear the pending exception prior to throwing
   */
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

  // Error statistics
  mutable std::mutex stats_mutex_;
  ErrorStatistics stats_;

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
