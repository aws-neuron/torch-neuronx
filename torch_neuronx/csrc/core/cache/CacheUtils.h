#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace at::neuron {
namespace cache_utils {

// Computes XXH3_128 hash of byte vector and returns a 32-character hex string.
//
// This is the hashing function used for persistent cache keys.
// Returns a string of 32 hex characters representing the 128-bit hash.
// For null or empty input, returns a 32-char string of zeros.
//
// Args:
//   data: bytes vector to hash
//
// Returns:
//   32-character hex string (128-bit hash)
std::string ComputeXXH3Hash128(const std::vector<uint8_t>& data);

}  // namespace cache_utils
}  // namespace at::neuron
