#ifndef TORCH_NEURONX_CSRC_CORE_NEURON_LOGGING_H
#define TORCH_NEURONX_CSRC_CORE_NEURON_LOGGING_H

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

namespace torch_neuronx {

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

  static bool shouldLog(LogLevel level) { return level <= log_level_; }

  void log(LogLevel level, LogCategory category, const std::string& message,
           const char* file = nullptr, int line = 0) {
    std::lock_guard<std::mutex> lock(mutex_);

    // OPERATOR_FALLBACK and OPERATOR_EXECUTED - skip file dumping, just return
    if (category == LogCategory::OPERATOR_FALLBACK || category == LogCategory::OPERATOR_EXECUTED) {
      return;
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

    std::ostringstream full_message_stream;
    full_message_stream << "[Neuron] " << level_str;

    if (file != nullptr && line > 0) {
      full_message_stream << " [" << file << ":" << line << "]";
    }

    full_message_stream << " " << message;
    std::string full_message = full_message_stream.str();

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
    const char* env_level = std::getenv("TORCH_NEURONX_LOG_LEVEL");
    if (env_level) {
      int level = std::atoi(env_level);
      if (level >= 0 && level <= 3) {
        log_level_ = static_cast<LogLevel>(level);
      }
    }

    // Check environment variable for log file
    const char* env_file = std::getenv("TORCH_NEURONX_LOG_FILE");
    if (env_file) {
      setLogFile(env_file);
    }
  }

  inline static LogLevel log_level_ = LogLevel::WARNING;
  bool log_to_file_ = false;
  std::ofstream log_file_;
  mutable std::mutex mutex_;
};

template <typename... Args>
void log_variadic(LogLevel level, LogCategory category, const char* file, int line,
                  Args&&... args) {
  auto& logger = NeuronLogger::getInstance();
  std::ostringstream oss;
  ((oss << args << " "), ...);
  logger.log(level, category, oss.str(), file, line);
}

// Convenience macros
#define TORCH_NEURONX_LOG(level, category, ...)                                      \
  do {                                                                               \
    if (torch_neuronx::NeuronLogger::shouldLog(level)) {                             \
      torch_neuronx::log_variadic(level, category, __FILE__, __LINE__, __VA_ARGS__); \
    }                                                                                \
  } while (0)

// General logging macros (default to GENERAL category) - support variadic arguments
#define TORCH_NEURONX_ERROR(...)                                                         \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::ERROR, torch_neuronx::LogCategory::GENERAL, \
                    __VA_ARGS__)
#define TORCH_NEURONX_WARN(...)                                                            \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::WARNING, torch_neuronx::LogCategory::GENERAL, \
                    __VA_ARGS__)
#define TORCH_NEURONX_INFO(...) \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::GENERAL, __VA_ARGS__)
#define TORCH_NEURONX_DEBUG(...)                                                         \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::DEBUG, torch_neuronx::LogCategory::GENERAL, \
                    __VA_ARGS__)

// Specialized logging macro for operator fallback
#define TORCH_NEURONX_FALLBACK_WARN(op_name)                                              \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::WARNING,                                     \
                    torch_neuronx::LogCategory::OPERATOR_FALLBACK, "Operator '", op_name, \
                    "' fell back to CPU")

// Specialized logging macro for executed operators
#define TORCH_NEURONX_EXECUTED_OP(op_name)                                                        \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::OPERATOR_EXECUTED, \
                    "Operator '", op_name, "' executed on Neuron")

// Specialized logging macros for NEFF cache operations
#define TORCH_NEURONX_NEFF_CACHE_HIT(cache_key)                                            \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE, \
                    "[NEFF_CACHE] Cache HIT for key:", cache_key)

#define TORCH_NEURONX_NEFF_CACHE_MISS(cache_key)                                           \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE, \
                    "[NEFF_CACHE] Cache MISS for key:", cache_key)

#define TORCH_NEURONX_NEFF_CACHE_STORE(cache_key)                                          \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE, \
                    "[NEFF_CACHE] Storing NEFF for key:", cache_key)

// Stream-specific debug logging
#define TORCH_NEURONX_STREAM_DEBUG(...)                                                  \
  TORCH_NEURONX_LOG(torch_neuronx::LogLevel::DEBUG, torch_neuronx::LogCategory::GENERAL, \
                    "[STREAM]", __VA_ARGS__)

// Function tracing macros
#define TORCH_NEURONX_TRACE_FUNCTION() TORCH_NEURONX_DEBUG("TRACE:", __FUNCTION__)

#define TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS(...) \
  TORCH_NEURONX_DEBUG("TRACE:", __FUNCTION__, __VA_ARGS__)

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_NEURON_LOGGING_H
