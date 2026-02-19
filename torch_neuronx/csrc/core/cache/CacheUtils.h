#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace at::neuron {
namespace cache_utils {

// Creates a persistent cache key by combining the base cache key with an IR hash.
//
// The persistent cache key is an XXH3_128 hash of the concatenation of:
// - base_key: The in-memory cache key
// - ir_bytes: The HLO/StableHLO bytes
//
// This ensures:
// 1. The resulting key is filesystem-safe (hex characters only, fixed length)
// 2. When compiler toolchain versions change (causing different HLO to be generated),
//    the cached NEFFs are automatically invalidated because the HLO hash will differ.
//
// Returns a 32-character hex string (128-bit hash).
std::string MakePersistentCacheKey(const std::string& base_key,
                                   const std::vector<uint8_t>& ir_bytes);

}  // namespace cache_utils
}  // namespace at::neuron
