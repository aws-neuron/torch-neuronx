#pragma once

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// Include NRT headers for NRT_STATUS type
extern "C" {
#include <nrt/nrt.h>
}

namespace torch_neuronx {

// Base exception for compilation-time errors (graph building, model loading, etc.)
class CompilationRuntimeException : public std::runtime_error {
 public:
  explicit CompilationRuntimeException(const std::string& message) : std::runtime_error(message) {}
};

// Base exception for execution-time errors (inference, NRT operations, etc.)
class ExecutionRuntimeException : public std::runtime_error {
  std::optional<NRT_STATUS> nrt_status_;
  std::optional<uint64_t> seq_id_;

 public:
  explicit ExecutionRuntimeException(const std::string& message)
      : std::runtime_error(message), nrt_status_(std::nullopt), seq_id_(std::nullopt) {}

  ExecutionRuntimeException(const std::string& message, NRT_STATUS status)
      : std::runtime_error(message), nrt_status_(status), seq_id_(std::nullopt) {}

  ExecutionRuntimeException(uint64_t seq_id, NRT_STATUS status)
      : std::runtime_error("Async execution error"), nrt_status_(status), seq_id_(seq_id) {}

  bool HasNRTStatus() const { return nrt_status_.has_value(); }
  bool HasSeqId() const { return seq_id_.has_value(); }

  // Ensure HasNRTStatus() prior to retrieving it.
  NRT_STATUS GetNRTStatus() const { return nrt_status_.value(); }
  uint64_t GetSeqId() const { return seq_id_.value(); }
};

// Type trait to check if a type is a Neuron runtime exception
template <typename T>
struct is_neuron_runtime_exception : std::false_type {};

template <>
struct is_neuron_runtime_exception<CompilationRuntimeException> : std::true_type {};

template <>
struct is_neuron_runtime_exception<ExecutionRuntimeException> : std::true_type {};

template <typename T>
inline constexpr bool is_neuron_runtime_exception_v = is_neuron_runtime_exception<T>::value;

}  // namespace torch_neuronx

#define EXECUTION_RUNTIME_CHECK(cond, ...)                                                      \
  do {                                                                                          \
    if (C10_UNLIKELY(!(cond))) {                                                                \
      throw torch_neuronx::ExecutionRuntimeException(                                           \
          std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #cond + " FAILED. " + \
          ::torch_neuronx::detail::ConcatMsg(__VA_ARGS__));                                     \
    }                                                                                           \
  } while (false)

// Like EXECUTION_RUNTIME_CHECK but preserves an NRT_STATUS so error recovery
#define EXECUTION_RUNTIME_CHECK_STATUS(cond, status, ...)                                       \
  do {                                                                                          \
    if (C10_UNLIKELY(!(cond))) {                                                                \
      throw torch_neuronx::ExecutionRuntimeException(                                           \
          std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #cond + " FAILED. " + \
              ::torch_neuronx::detail::ConcatMsg(__VA_ARGS__),                                  \
          (status));                                                                            \
    }                                                                                           \
  } while (false)

#define COMPILE_RUNTIME_CHECK(cond, ...)                                                        \
  do {                                                                                          \
    if (C10_UNLIKELY(!(cond))) {                                                                \
      throw torch_neuronx::CompilationRuntimeException(                                         \
          std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #cond + " FAILED. " + \
          ::torch_neuronx::detail::ConcatMsg(__VA_ARGS__));                                     \
    }                                                                                           \
  } while (false)

namespace torch_neuronx::detail {
// Helper to concatenate variadic message arguments into a single string.
inline std::string ConcatMsg() { return {}; }

template <typename T>
inline std::string ConcatMsg(const T& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

template <typename T, typename... Args>
inline std::string ConcatMsg(const T& first, const Args&... rest) {
  std::ostringstream oss;
  oss << first;
  return oss.str() + ConcatMsg(rest...);
}
}  // namespace torch_neuronx::detail
