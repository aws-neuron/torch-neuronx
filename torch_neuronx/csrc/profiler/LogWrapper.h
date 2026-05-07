#pragma once

#include <sstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"

#define NPROF_INFO(...) TORCH_NEURONX_INFO("[Profiler]", __VA_ARGS__)
#define NPROF_WARN(...) TORCH_NEURONX_WARN("[Profiler]", __VA_ARGS__)

template <typename... Args>
inline std::string nprof_make_error_string(Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

#define NPROF_ERROR(errors_vec, ...)                                   \
  do {                                                                 \
    std::string _nprof_err_msg = nprof_make_error_string(__VA_ARGS__); \
    TORCH_NEURONX_ERROR("[Profiler]", _nprof_err_msg);                 \
    errors_vec.push_back(_nprof_err_msg);                              \
  } while (0)
