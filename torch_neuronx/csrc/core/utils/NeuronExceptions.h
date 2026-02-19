#pragma once

#include <optional>
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
