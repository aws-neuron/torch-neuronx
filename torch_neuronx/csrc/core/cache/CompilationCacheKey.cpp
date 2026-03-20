#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"

#include <xxhash.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "torch_neuronx/csrc/core/cache/CacheUtils.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

namespace at::neuron {

namespace {

// Separator byte between fields in the hash input to prevent ambiguous concatenations.
// 0xFF is chosen because it cannot appear in valid UTF-8 strings.
constexpr uint8_t kFieldSeparator = 0xFF;

}  // namespace

std::string CompilationCacheKey::BuildCompilerExtension() const {
  return platform_target_ + "_" + logical_neuron_cores_ + "_" + optimization_level_ + "_" +
         compiler_args_;
}

const char* CompilationCacheKey::SourceName(CacheKeySource source) {
  switch (source) {
    case CacheKeySource::EAGER:
      return "EAGER";
    case CacheKeySource::TORCH_COMPILE:
      return "TORCH_COMPILE";
    default:
      return "UNKNOWN";
  }
}

CompilationCacheKey::CompilationCacheKey(const std::string& base_key,
                                         const std::string& compiler_args,
                                         const std::string& optimization_level,
                                         const std::vector<uint8_t>* hlo_bytes_ptr,
                                         CacheKeySource source)
    : base_key_(base_key),
      compiler_args_(compiler_args),
      optimization_level_(optimization_level),
      platform_target_(utils::GetPlatformTarget()),
      logical_neuron_cores_(utils::GetLogicalNeuronCores()),
      hlo_bytes_ptr_(hlo_bytes_ptr),
      source_(source) {}

CompilationCacheKey::CompilationCacheKey(const std::string& key, CacheKeySource source)
    : base_key_(key), source_(source) {}

const std::string* CompilationCacheKey::GetPersistentKey() const {
  if (base_key_.empty()) {
    return nullptr;
  }
  if (persistent_key_.empty()) {
    // Use base_key for persistent key
    persistent_key_ = ComputePersistentKey();
    key_str_.clear();
  }
  return &persistent_key_;
}

std::string CompilationCacheKey::ComputePersistentKey() const {
  // Build the hash input: base_key || 0xFF || compiler_config
  std::string compiler_config = BuildCompilerExtension();
  std::vector<uint8_t> combined;
  combined.reserve(base_key_.size() + 1 + compiler_config.size());

  combined.insert(combined.end(), reinterpret_cast<const uint8_t*>(base_key_.data()),
                  reinterpret_cast<const uint8_t*>(base_key_.data()) + base_key_.size());
  combined.push_back(kFieldSeparator);

  combined.insert(
      combined.end(), reinterpret_cast<const uint8_t*>(compiler_config.data()),
      reinterpret_cast<const uint8_t*>(compiler_config.data()) + compiler_config.size());

  // 32-char hex string
  return cache_utils::ComputeXXH3Hash128(combined);
}

const std::string& CompilationCacheKey::ToString() const {
  if (key_str_.empty() && !base_key_.empty()) {
    std::ostringstream oss;
    oss << "{key=" << base_key_ << ", platform=" << platform_target_
        << ", lnc=" << logical_neuron_cores_ << ", compiler_args=" << compiler_args_
        << ", optimization_level=" << optimization_level_ << ", source=" << SourceName(source_);
    if (!persistent_key_.empty()) {
      oss << ", persistent_key=" << persistent_key_;
    }
    oss << "}";
    key_str_ = oss.str();
  }
  return key_str_;
}

}  // namespace at::neuron
