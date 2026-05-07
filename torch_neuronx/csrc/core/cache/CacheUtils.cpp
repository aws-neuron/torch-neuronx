#include "torch_neuronx/csrc/core/cache/CacheUtils.h"

#include <xxhash.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>

constexpr const char* kPreserveArtifactsEnvVar = "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS";
constexpr const char* kDebugDir = "TORCH_NEURONX_DEBUG_DIR";

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

std::string GetDebugDir() {
  const char* env = std::getenv(kDebugDir);
  if (!env) {
    return "/tmp/neuron_backend";
  }
  return std::string(env);
}

std::string GetCompilationWorkdir() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << dist(gen) << std::setw(16) << dist(gen);
  return GetDebugDir() + "/neuronx-cc/" + oss.str();
}

bool ShouldPreserveCompilationArtifacts() {
  const char* env = std::getenv(kPreserveArtifactsEnvVar);
  if (!env) {
    return false;
  }
  std::string env_str(env);
  std::transform(env_str.begin(), env_str.end(), env_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (env_str == "1" || env_str == "true" || env_str == "yes");
}

}  // namespace cache_utils
}  // namespace at::neuron
