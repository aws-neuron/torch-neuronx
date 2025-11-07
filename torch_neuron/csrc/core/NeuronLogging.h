#ifndef TORCH_NEURON_CSRC_CORE_NEURON_LOGGING_H
#define TORCH_NEURON_CSRC_CORE_NEURON_LOGGING_H

#include <c10/util/Logging.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace torch_neuron {

enum class LogLevel { ERROR = 0, WARNING = 1, INFO = 2, DEBUG = 3 };

enum class LogCategory {
  GENERAL = 0,
  OPERATOR_FALLBACK = 1,
  MEMORY = 2,
  KERNEL = 3,
  DEVICE = 4,
  OPERATOR_EXECUTED = 5,
  NEFF_CACHE = 6
};

class NeuronLogger {
 public:
  static NeuronLogger& getInstance() {
    static NeuronLogger instance;
    return instance;
  }

  void setLogLevel(LogLevel level) { log_level_ = level; }

  void setLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
      log_file_.close();
    }
    if (!filename.empty()) {
      log_file_.open(filename, std::ios::app);
      log_to_file_ = log_file_.is_open();
    } else {
      log_to_file_ = false;
    }
  }

  bool shouldLog(LogLevel level) const { return level <= log_level_; }

  void log(LogLevel level, LogCategory category, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);

    // OPERATOR_FALLBACK always goes to the fallback file (bypass log level check)
    if (category == LogCategory::OPERATOR_FALLBACK) {
      if (!fallback_file_.is_open()) {
        initializeDumpFiles();
      }
      if (fallback_file_.is_open()) {
        fallback_file_ << message << std::endl;
        fallback_file_.flush();
        return;  // Don't log to console or general log
      }
    }

    // OPERATOR_EXECUTED always goes to the executed ops file (bypass log level check)
    if (category == LogCategory::OPERATOR_EXECUTED) {
      if (!executed_ops_file_.is_open()) {
        initializeDumpFiles();
      }
      if (executed_ops_file_.is_open()) {
        executed_ops_file_ << message << std::endl;
        executed_ops_file_.flush();
        return;  // Don't log to console or general log
      }
    }

    // For general logging, check the log level
    if (!shouldLog(level)) {
      return;
    }

    std::string level_str;
    switch (level) {
      case LogLevel::ERROR:
        level_str = "[ERROR]";
        break;
      case LogLevel::WARNING:
        level_str = "[WARNING]";
        break;
      case LogLevel::INFO:
        level_str = "[INFO]";
        break;
      case LogLevel::DEBUG:
        level_str = "[DEBUG]";
        break;
    }

    std::string full_message = "[Neuron] " + level_str + " " + message;

    if (log_to_file_ && log_file_.is_open()) {
      log_file_ << full_message << std::endl;
      log_file_.flush();
    } else {
      // For warnings and errors, use PyTorch's warning system
      // For info and debug, use stdout
      if (level <= LogLevel::WARNING) {
        TORCH_WARN(full_message);
      } else {
        std::cout << full_message << std::endl;
      }
    }
  }

 private:
  NeuronLogger() {
    // Check environment variable for log level
    const char* env_level = std::getenv("TORCH_NEURON_LOG_LEVEL");
    if (env_level) {
      int level = std::atoi(env_level);
      if (level >= 0 && level <= 3) {
        log_level_ = static_cast<LogLevel>(level);
      }
    }

    // Check environment variable for log file
    const char* env_file = std::getenv("TORCH_NEURON_LOG_FILE");
    if (env_file) {
      setLogFile(env_file);
    }
  }

  void initializeDumpFiles() {
    // Get dump directory from environment variable or use default
    const char* env_dump_dir = std::getenv("TORCH_NEURON_DUMP");
    std::string base_dir = env_dump_dir ? env_dump_dir : "./torch_neuron_dump";

    // Create base directory if it doesn't exist
    mkdir(base_dir.c_str(), 0755);

    // Get current timestamp in MMDDHHMMSS format
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time_t);

    // Create timestamp-prefixed PID subdirectory (MMDDHHMMSS_PID)
    std::stringstream pid_dir_stream;
    pid_dir_stream << base_dir << "/" << std::setfill('0') << std::setw(2)
                   << (tm->tm_mon + 1)                                  // MM
                   << std::setfill('0') << std::setw(2) << tm->tm_mday  // DD
                   << std::setfill('0') << std::setw(2) << tm->tm_hour  // HH
                   << std::setfill('0') << std::setw(2) << tm->tm_min   // MM
                   << std::setfill('0') << std::setw(2) << tm->tm_sec   // SS
                   << "_" << getpid();                                  // _PID
    std::string pid_dir = pid_dir_stream.str();
    mkdir(pid_dir.c_str(), 0755);

    // Open the fallback file in offloaded_ops subdirectory
    std::stringstream fallback_filename_stream;
    fallback_filename_stream << pid_dir << "/offloaded_ops.txt";
    std::string fallback_filename = fallback_filename_stream.str();
    fallback_file_.open(fallback_filename, std::ios::app);

    // Open the executed ops file in used_ops subdirectory
    std::stringstream executed_filename_stream;
    executed_filename_stream << pid_dir << "/used_ops.txt";
    std::string executed_filename = executed_filename_stream.str();
    executed_ops_file_.open(executed_filename, std::ios::app);
  }

  LogLevel log_level_ = LogLevel::WARNING;
  bool log_to_file_ = false;
  std::ofstream log_file_;
  std::ofstream fallback_file_;
  std::ofstream executed_ops_file_;
  mutable std::mutex mutex_;
};

template <typename... Args>
void log_variadic(LogLevel level, LogCategory category, Args&&... args) {
  auto& logger = NeuronLogger::getInstance();
  if (logger.shouldLog(level)) {
    std::ostringstream oss;
    ((oss << args << " "), ...);
    logger.log(level, category, oss.str());
  }
}

// Convenience macros
#define TORCH_NEURON_LOG(level, category, ...) \
  torch_neuron::log_variadic(level, category, __VA_ARGS__)

// General logging macros (default to GENERAL category) - support variadic arguments
#define TORCH_NEURON_ERROR(...) \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::ERROR, torch_neuron::LogCategory::GENERAL, __VA_ARGS__)
#define TORCH_NEURON_WARN(...) \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::WARNING, torch_neuron::LogCategory::GENERAL, __VA_ARGS__)
#define TORCH_NEURON_INFO(...) \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::GENERAL, __VA_ARGS__)
#define TORCH_NEURON_DEBUG(...) \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::DEBUG, torch_neuron::LogCategory::GENERAL, __VA_ARGS__)

// Specialized logging macro for operator fallback
#define TORCH_NEURON_FALLBACK_WARN(op_name)                                                       \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::WARNING, torch_neuron::LogCategory::OPERATOR_FALLBACK, \
                   "Operator '", op_name, "' fell back to CPU")

// Specialized logging macro for executed operators
#define TORCH_NEURON_EXECUTED_OP(op_name)                                                      \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::OPERATOR_EXECUTED, \
                   "Operator '", op_name, "' executed on Neuron")

// Specialized logging macros for NEFF cache operations
#define TORCH_NEURON_NEFF_CACHE_HIT(cache_key)                                          \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::NEFF_CACHE, \
                   "[NEFF_CACHE] Cache HIT for key:", cache_key)

#define TORCH_NEURON_NEFF_CACHE_MISS(cache_key)                                         \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::NEFF_CACHE, \
                   "[NEFF_CACHE] Cache MISS for key:", cache_key)

#define TORCH_NEURON_NEFF_CACHE_STORE(cache_key)                                        \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::INFO, torch_neuron::LogCategory::NEFF_CACHE, \
                   "[NEFF_CACHE] Storing NEFF for key:", cache_key)

// Stream-specific debug logging
#define TORCH_NEURON_STREAM_DEBUG(...)                                                            \
  TORCH_NEURON_LOG(torch_neuron::LogLevel::DEBUG, torch_neuron::LogCategory::GENERAL, "[STREAM]", \
                   __VA_ARGS__)

// Function tracing macros
#define TORCH_NEURON_TRACE_FUNCTION() TORCH_NEURON_DEBUG("TRACE:", __FUNCTION__)

#define TORCH_NEURON_TRACE_FUNCTION_WITH_ARGS(...) \
  TORCH_NEURON_DEBUG("TRACE:", __FUNCTION__, __VA_ARGS__)

}  // namespace torch_neuron

#endif  // TORCH_NEURON_CSRC_CORE_NEURON_LOGGING_H
