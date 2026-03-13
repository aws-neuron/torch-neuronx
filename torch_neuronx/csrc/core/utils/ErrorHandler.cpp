#include "ErrorHandler.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"
#include "torch_neuronx/csrc/core/utils/ErrorRecovery.h"

namespace at::neuron {

ErrorHandler::ErrorHandler() {
  // Initialize configuration from environment variables
  const char* enhanced_env = std::getenv("NEURON_ENHANCED_ERROR_HANDLING");
  enhanced_error_handling_enabled_ = (enhanced_env && std::string(enhanced_env) == "1");

  const char* exec_dump_env = std::getenv("NEURON_DUMP_EXECUTION_ON_ERROR");
  execution_dumping_enabled_ = (exec_dump_env && std::string(exec_dump_env) == "1");

  const char* dump_dir_env = std::getenv("NEURON_ERROR_DUMP_DIR");
  if (dump_dir_env) {
    debug_dump_directory_ = std::string(dump_dir_env);
  } else {
    debug_dump_directory_ = "/tmp/neuron_error_dumps";
  }

  // Initialize per-stream error recovery
  error_recovery_ = std::make_unique<ErrorRecovery>();

  TORCH_NEURONX_DEBUG(
      "ErrorHandler initialized", "enhanced_error_handling=", enhanced_error_handling_enabled_,
      "hlo_dumping=", hlo_dumping_enabled_, "execution_dumping=", execution_dumping_enabled_,
      "dump_directory=", debug_dump_directory_);
}

std::string ErrorHandler::HandleCompilationError(OperationContext* op,
                                                 const std::string& error_message,
                                                 ErrorContext& context) {
  auto start_time = std::chrono::steady_clock::now();
  context.current_stage = ErrorContext::Stage::COMPILATION;

  std::string final_error_message =
      GenerateErrorMessage("COMPILATION FAILED: " + error_message, context);

  std::string complete_message =
      "Compilation error occurred on Neuron for operation=" + context.operation_name +
      ";\nerror message=\"" + final_error_message +
      "\""
      "\npython stack trace=\n" +
      op->python_stack_trace;

  std::string debug_message =
      "stream_id=" + std::to_string(context.stream_id) + "; " + complete_message;
  TORCH_NEURONX_DEBUG(debug_message);

  auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  RecordError(context, processing_time);
  return complete_message;
}

std::string ErrorHandler::HandleExecutionError(OperationContext* op,
                                               const std::string& error_message,
                                               ErrorContext& context) {
  auto start_time = std::chrono::steady_clock::now();
  context.current_stage = ErrorContext::Stage::EXECUTION;

  std::string final_error_message =
      GenerateErrorMessage("NRT EXECUTION FAILED: " + error_message, context);

  std::string complete_message =
      "NRT Execution error occurred on Neuron for operation=" + context.operation_name +
      "\nerror message=\"" + final_error_message +
      "\""
      "\npython stack trace=\n" +
      op->python_stack_trace;

  std::string debug_message =
      "stream_id=" + std::to_string(context.stream_id) + "; " + complete_message;
  TORCH_NEURONX_DEBUG(debug_message);

  auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  RecordError(context, processing_time);
  return complete_message;
}

std::string ErrorHandler::ClassifyNrtError(
    const torch_neuronx::ExecutionRuntimeException& e) const {
  if (e.HasNRTStatus()) {
    return std::string(e.what()) + ", " + ClassifyNrtError(e.GetNRTStatus());
  }
  return std::string(e.what());
}

std::string ErrorHandler::ClassifyNrtError(NRT_STATUS nrt_status) const {
  switch (nrt_status) {
    case 0:  // NRT_SUCCESS
      return "Success (unexpected in error context)";
    case 1:  // NRT_FAILURE
      return "Non-specific failure";
    case 2:  // NRT_INVALID
      return "Invalid NEFF, instruction, or input";
    case 3:  // NRT_INVALID_HANDLE
      return "Invalid handle";
    case 4:  // NRT_RESOURCE
      return "Failed to allocate resource";
    case 5:  // NRT_TIMEOUT
      return "Operation timed out";
    case 6:  // NRT_HW_ERROR
      return "Hardware failure";
    case 7:  // NRT_QUEUE_FULL
      return "Execution input queue full";
    case 9:  // NRT_LOAD_NOT_ENOUGH_NC
      return "Not enough NCs for loading NEFF";
    case 10:  // NRT_UNSUPPORTED_NEFF_VERSION
      return "Unsupported NEFF version";
    case 11:  // NRT_FAIL_HOST_MEM_ALLOC
      return "Failed to allocate host memory";
    case 13:  // NRT_UNINITIALIZED
      return "NRT not initialized";
    case 14:  // NRT_CLOSED
      return "NRT already closed";
    case 15:  // NRT_QUEUE_EMPTY
      return "Accessed an empty Queue";
    case 1002:  // NRT_EXEC_BAD_INPUT
      return "Invalid input to exec()";
    case 1003:  // NRT_EXEC_COMPLETED_WITH_NUM_ERR
      return "Execution completed with numerical errors (NaN)";
    case 1004:  // NRT_EXEC_COMPLETED_WITH_ERR
      return "Execution completed with errors";
    case 1005:  // NRT_EXEC_NC_BUSY
      return "Neuron core busy/locked by another model/process";
    case 1006:  // NRT_EXEC_OOB
      return "one or more indirect memcopies and/or embedding updates are out of bound";
    case 1100:  // NRT_COLL_PENDING
      return "Collective operation pending";
    case 1200:  // NRT_EXEC_HW_ERR_COLLECTIVES
      return "Stuck in collectives op (missing notification(s)). Possibly caused by a hardware "
             "error on another worker";
    case 1201:  // NRT_EXEC_HW_ERR_HBM_UE
      return "An HBM encountered an unrepairable uncorrectable error and produced incorrect "
             "results";
    case 1202:  // NRT_EXEC_HW_ERR_NC_UE
      return "An on-chip memory of Neuron Core encountered a parity error and produced incorrect "
             "results.";
    case 1203:  // NRT_EXEC_HW_ERR_DMA_ABORT
      return "A DMA engine encountered an unrecoverable error";
    case 1204:  // NRT_EXEC_SW_NQ_OVERFLOW
      return "Software notification queue overflow";
    case 1205:  // NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE
      return "HBM repairable uncorrectable error";
    case 1206:
      return "EFA network proxy failure";
    default:
      return "Unknown status (status: " + std::to_string(nrt_status) + ")";
  }
}

ErrorContext ErrorHandler::CreateErrorContext(OperationContext* op,
                                              ErrorContext::Stage stage) const {
  ErrorContext context;

  context.stream_id = op->stream->stream_id;
  context.operation_name = op->GetOpName();
  context.submit_time = op->submit_time;
  context.current_stage = stage;
  context.error_time = std::chrono::steady_clock::now();

  return context;
}

std::string ErrorHandler::GenerateErrorMessage(const std::string& base_message,
                                               const ErrorContext& context) const {
  std::ostringstream msg;
  msg << base_message;

  if (enhanced_error_handling_enabled_) {
    msg << "\n--- Error Context ---";
    msg << "\nOperation: " << context.operation_name;
    msg << "\nStream ID: " << context.stream_id;
    msg << "\nStage: " << context.get_stage_name();
    msg << "\nElapsed Time: " << context.get_elapsed_time().count() << "ms";

    if (!context.hlo_dump_path.empty()) {
      msg << "\nHLO Dump: " << context.hlo_dump_path;
    }

    if (!context.neff_path.empty()) {
      msg << "\nNEFF Path: " << context.neff_path;
    }
    msg << "\n--- End Context ---";
  }

  return msg.str();
}

std::vector<ErrorContext> ErrorHandler::GetRecentErrors(size_t max_errors) const {
  std::lock_guard<std::mutex> lock(history_mutex_);

  size_t start_index = 0;
  if (recent_errors_.size() > max_errors) {
    start_index = recent_errors_.size() - max_errors;
  }

  return std::vector<ErrorContext>(recent_errors_.begin() + start_index, recent_errors_.end());
}

bool ErrorHandler::IsEnhancedErrorHandlingEnabled() const {
  return enhanced_error_handling_enabled_;
}

bool ErrorHandler::IsHloDumpingEnabled() const { return hlo_dumping_enabled_; }

bool ErrorHandler::IsExecutionDumpingEnabled() const { return execution_dumping_enabled_; }

std::string ErrorHandler::ExceptionToString(const std::exception& exception) {
  std::ostringstream result;
  result << "Exception: " << typeid(exception).name() << " - " << exception.what();
  return result.str();
}

std::string ErrorHandler::CleanErrorMessage(const std::string& error_message) {
  // Remove stacktrace and keep only the last meaningful error line
  std::istringstream stream(error_message);
  std::string line;
  std::string last_meaningful_line;

  while (std::getline(stream, line)) {
    // Skip empty lines and common stacktrace patterns
    if (line.empty() || line.find("Traceback") != std::string::npos ||
        line.find("  File ") != std::string::npos || line.find("    at ") != std::string::npos ||
        line.find("\tat ") != std::string::npos) {
      continue;
    }

    // Keep lines that look like actual error messages
    if (line.find("Error:") != std::string::npos || line.find("Exception:") != std::string::npos ||
        line.find("Failed") != std::string::npos || line.find("failed") != std::string::npos ||
        (!line.empty() && line[0] != ' ' && line[0] != '\t')) {
      last_meaningful_line = line;
    }
  }

  // Return cleaned message or original if no meaningful line found
  return last_meaningful_line.empty() ? error_message : last_meaningful_line;
}

void ErrorHandler::RecordError(const ErrorContext& context,
                               std::chrono::microseconds processing_time) {
  // Add to recent error history
  {
    std::lock_guard<std::mutex> lock(history_mutex_);
    recent_errors_.push_back(context);

    // Keep only the most recent errors
    if (recent_errors_.size() > MAX_RECENT_ERRORS) {
      recent_errors_.erase(recent_errors_.begin());
    }
  }
}

std::string ErrorHandler::GetDebugDumpDirectory() const { return debug_dump_directory_; }

bool ErrorHandler::EnsureDebugDirectoryExists(const std::string& directory_path) const {
  try {
    std::filesystem::create_directories(directory_path);
    return std::filesystem::exists(directory_path);
  } catch (const std::exception& e) {
    TORCH_NEURONX_ERROR("Failed to create debug directory", "path=", directory_path,
                        "error=", e.what());
    return false;
  }
}

std::string ErrorHandler::GenerateDebugFilename(const std::string& prefix,
                                                const std::string& extension,
                                                const ErrorContext& context) const {
  std::ostringstream filename;
  filename << prefix << "_";
  filename << "stream" << context.stream_id << "_";
  filename << std::chrono::duration_cast<std::chrono::seconds>(
                  context.error_time.time_since_epoch())
                  .count();
  filename << "." << extension;

  return filename.str();
}

OperationContextResult ErrorHandler::HandleOperationError(
    OperationContext* op, const torch_neuronx::CompilationRuntimeException& e) {
  uint32_t stream_id = op->stream->stream_id;
  TORCH_NEURONX_DEBUG("Handling compilation error", "stream_id=", stream_id,
                      "op=", op->GetOpName());

  auto recovery_status = error_recovery_->AttemptRecovery(op, e);
  if (recovery_status.IsSuccess()) {
    return recovery_status;
  }

  // Create error context with inferred stage
  ErrorContext context = CreateErrorContext(op, ErrorContext::Stage::COMPILATION);

  std::string error_details;
  try {
    auto cleaned_error_message = CleanErrorMessage(e.what());
    error_details = HandleCompilationError(op, cleaned_error_message, context);
    context.error_details = error_details;
    RecordErrorForMonitoring(op, context);
  } catch (const std::exception& handler_error) {
    TORCH_NEURONX_WARN("Error monitoring threw exception", "stream_id=", stream_id,
                       "op=", op->GetOpName(), "handler_error=", handler_error.what());
  }
  auto exception_ptr = std::make_exception_ptr(std::runtime_error(error_details));
  RecordFirstException(stream_id, exception_ptr);

  return OperationContextResult::CreateError(error_details);
}

OperationContextResult ErrorHandler::HandleOperationError(
    OperationContext* op, const torch_neuronx::ExecutionRuntimeException& e) {
  uint32_t stream_id = op->stream->stream_id;
  TORCH_NEURONX_DEBUG("Handling execution error", "stream_id=", stream_id, "op=", op->GetOpName());

  // Create error context with inferred stage
  ErrorContext context = CreateErrorContext(op, ErrorContext::Stage::EXECUTION);

  // TODO: Use ExecutionRuntimeException properties to parse NRT statuses.
  std::string error_details;
  try {
    std::string error_msg;
    if (e.HasSeqId()) {
      // nrt-async execution error - include seq_id in message
      error_msg = "Async execution error: seq_id=" + std::to_string(e.GetSeqId());
      if (e.HasNRTStatus()) {
        error_msg += ", " + ClassifyNrtError(e.GetNRTStatus());
      }
    } else {
      // nrt-sync execution error
      error_msg = e.HasNRTStatus() ? ClassifyNrtError(e) : std::string(e.what());
    }
    error_details = HandleExecutionError(op, error_msg, context);
    context.error_details = error_details;
    RecordErrorForMonitoring(op, context);
  } catch (const std::exception& handler_error) {
    TORCH_NEURONX_WARN("Error monitoring threw exception", "stream_id=", stream_id,
                       "op=", op->GetOpName(), "handler_error=", handler_error.what());
  }
  auto exception_ptr = std::make_exception_ptr(std::runtime_error(error_details));
  RecordFirstException(stream_id, exception_ptr);

  return OperationContextResult::CreateError(error_details);
}

void ErrorHandler::RecordFirstException(uint32_t stream_id, std::exception_ptr exception) {
  std::lock_guard<std::mutex> lock(exception_mutex_);
  if (!has_pending_exception_) {
    first_exception_ = exception;
    has_pending_exception_ = true;
    TORCH_NEURONX_DEBUG("Recorded first exception for stream", "stream_id=", stream_id);
  } else {
    TORCH_NEURONX_DEBUG("Exception already recorded, ignoring subsequent exception",
                        "stream_id=", stream_id);
  }
}

// Consolidated monitoring method that does NOT set promises (promises already set by caller)
void ErrorHandler::RecordErrorForMonitoring(OperationContext* op, const ErrorContext& context) {
  TORCH_NEURONX_DEBUG("Recording error for monitoring only", "stream_id=", context.stream_id,
                      "op=", op->GetOpName(), "stage=", static_cast<int>(context.current_stage),
                      "error_details=", context.error_details);

  // Get the appropriate timing based on the error stage
  std::chrono::microseconds processing_time;
  switch (context.current_stage) {
    case ErrorContext::Stage::COMPILATION:
      processing_time = op->GetCompilationTime();
      break;
    case ErrorContext::Stage::EXECUTION:
      processing_time = op->GetExecutionTime();
      break;
    default:
      processing_time = std::chrono::microseconds(0);
      break;
  }

  // Record error statistics
  RecordError(context, processing_time);
}

void ErrorHandler::CheckAndThrowPendingException(c10::StreamId stream_id, bool clear_exception) {
  std::lock_guard<std::mutex> lock(exception_mutex_);
  if (has_pending_exception_) {
    TORCH_NEURONX_DEBUG("Propagating pending exception during synchronization",
                        "stream_id=", stream_id);

    // Clear the exception state before rethrowing
    auto exception_to_throw = first_exception_;
    if (clear_exception) {
      first_exception_ = nullptr;
      has_pending_exception_ = false;
    }

    // Rethrow the exception
    std::rethrow_exception(exception_to_throw);
  }
}

}  // namespace at::neuron
