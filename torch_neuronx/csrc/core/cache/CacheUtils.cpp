#include "torch_neuronx/csrc/core/cache/CacheUtils.h"

#include <xxhash.h>

#include <iomanip>
#include <sstream>

namespace at::neuron {
namespace cache_utils {

std::string MakePersistentCacheKey(const std::string& base_key,
                                   const std::vector<uint8_t>& ir_bytes) {
  // Concatenate base_key and ir_bytes, then hash the result
  std::vector<uint8_t> combined;
  combined.reserve(base_key.size() + ir_bytes.size());

  // Add base_key bytes
  combined.insert(combined.end(), reinterpret_cast<const uint8_t*>(base_key.data()),
                  reinterpret_cast<const uint8_t*>(base_key.data()) + base_key.size());

  // Add IR bytes
  combined.insert(combined.end(), ir_bytes.begin(), ir_bytes.end());

  // Compute XXH3_128 hash
  if (combined.empty()) {
    return std::string(32, '0');  // Return zero hash for empty data (32 hex chars for 128-bit)
  }

  XXH128_hash_t hash = XXH3_128bits(combined.data(), combined.size());

  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash.high64 << std::setw(16)
      << hash.low64;
  return oss.str();
}

}  // namespace cache_utils
}  // namespace at::neuron
