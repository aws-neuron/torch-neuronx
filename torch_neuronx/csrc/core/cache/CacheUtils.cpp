#include "torch_neuronx/csrc/core/cache/CacheUtils.h"

#include <xxhash.h>

#include <iomanip>
#include <sstream>

namespace at::neuron {
namespace cache_utils {

std::string ComputeXXH3Hash128(const std::vector<uint8_t>& data) {
  if (data.empty()) {
    return std::string(32, '0');
  }

  XXH128_hash_t hash = XXH3_128bits(data.data(), data.size());

  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash.high64 << std::setw(16)
      << hash.low64;
  return oss.str();
}

}  // namespace cache_utils
}  // namespace at::neuron
