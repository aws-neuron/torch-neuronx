#pragma once

#include <c10/core/SymIntArrayRef.h>

#include <vector>

namespace torch_neuronx {
namespace utils {

// Convert SymIntArrayRef to a vector of concrete int64_t sizes.
inline std::vector<int64_t> symint_to_sizes(c10::SymIntArrayRef sizes) {
  std::vector<int64_t> result;
  result.reserve(sizes.size());
  for (const auto& size : sizes) {
    result.push_back(size.expect_int());
  }
  return result;
}

}  // namespace utils
}  // namespace torch_neuronx
