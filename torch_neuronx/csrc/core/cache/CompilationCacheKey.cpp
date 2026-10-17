#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"

#include <xxhash.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "torch_neuronx/csrc/core/cache/CacheUtils.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

namespace at::neuron {

namespace {

// Separator byte between fields in the hash input to prevent ambiguous concatenations.
// 0xFF is chosen because it cannot appear in valid UTF-8 strings.
constexpr uint8_t kFieldSeparator = 0xFF;

// Marker that the frontend appends to a base key for a Weight Layout Optimization compile.
// WLO is inferred from this suffix in one place, so every CompilationCacheKey construction site
// (torch.compile, NeffDirectKernelExecution, put/get_neff_cache) derives the same identity.
constexpr std::string_view kWloSuffix = "|wlo";

bool WloEnabledFromKey(const std::string& base_key) {
  return base_key.size() >= kWloSuffix.size() &&
         base_key.compare(base_key.size() - kWloSuffix.size(), kWloSuffix.size(), kWloSuffix) == 0;
}

}  // namespace

std::string CompilationCacheKey::BuildCompilerExtension() const {
  return platform_target_ + "_" + logical_neuron_cores_ + "_" + optimization_level_ + "_" +
         compiler_args_ + "_v" + compiler_version_ + (wlo_enabled_ ? "_wlo" : "");
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
      compiler_version_(utils::GetCompilerVersion()),
      hlo_bytes_ptr_(hlo_bytes_ptr),
      source_(source),
      wlo_enabled_(WloEnabledFromKey(base_key_)) {}

CompilationCacheKey::CompilationCacheKey(const std::string& key, CacheKeySource source)
    : base_key_(key), source_(source), wlo_enabled_(WloEnabledFromKey(key)) {}

const std::string* CompilationCacheKey::GetPersistentKey() const {
  if (base_key_.empty()) {
    return nullptr;
  }
  // When HLO-content keying is enabled, the persistent key requires HLO
  // bytes — without them, persistent caching is not possible (e.g.,
  // NeffDirectKernelExecution paths). Caller must skip the persistent cache
  // lookup when this returns nullptr.
  // When disabled (legacy / production default), persistent key derives from
  // base_key, which is always present once we pass the IsEmpty check above.
  if (utils::IsHloContentKeyEnabled() && (hlo_bytes_ptr_ == nullptr || hlo_bytes_ptr_->empty())) {
    return nullptr;
  }
  if (persistent_key_.empty()) {
    persistent_key_ = ComputePersistentKey();
    key_str_.clear();
  }
  return &persistent_key_;
}

std::string CompilationCacheKey::ComputePersistentKey() const {
  // Persistent NEFF key composition is gated by TORCH_NEURONX_NEFF_KEY_FROM_HLO:
  //   - on  (dev/CI): hash(hlo_bytes || compiler_config) — content-addressed.
  //                   Lets benign Python source edits re-trace/re-lower (cheap)
  //                   but still hit the on-disk NEFF cache when HLO bytes are
  //                   unchanged, enabling cross-commit cache reuse.
  //   - off (default/prod): hash(base_key || compiler_config) — legacy.
  //                   Any source change in base_key invalidates the NEFF.
  // GetPersistentKey() guarantees hlo_bytes_ptr_ is valid when the flag is on.
  std::string compiler_config = BuildCompilerExtension();
  const bool use_hlo = utils::IsHloContentKeyEnabled();
  const size_t input_size = use_hlo ? hlo_bytes_ptr_->size() : base_key_.size();

  std::vector<uint8_t> combined;
  combined.reserve(input_size + 1 + compiler_config.size());

  if (use_hlo) {
    combined.insert(combined.end(), hlo_bytes_ptr_->begin(), hlo_bytes_ptr_->end());
  } else {
    combined.insert(combined.end(), reinterpret_cast<const uint8_t*>(base_key_.data()),
                    reinterpret_cast<const uint8_t*>(base_key_.data()) + base_key_.size());
  }
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
