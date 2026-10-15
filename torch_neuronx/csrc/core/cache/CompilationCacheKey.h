#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace at::neuron {

/**
 * @brief Identifies which frontend generated a cache key (for metrics/debugging only).
 */
enum class CacheKeySource : uint8_t {
  EAGER,          // Eager mode
  TORCH_COMPILE,  // torch.compile
  UNKNOWN
};

/**
 * @brief Unified cache key for compiled NEFFs.
 *
 * The key supports two levels optimized for different use cases:
 *
 * 1. In-memory key: `base_key`
 *    Used for in-memory cache lookup. The `base_key` is a string from the
 *    frontend (e.g., op_name + shapes + dtypes from eager, or a hash string
 *    from torch.compile). Includes Python source/version info so that any
 *    upstream change forces a re-trace/re-lower.
 *
 * 2. Persistent key: gated by TORCH_NEURONX_NEFF_KEY_FROM_HLO.
 *    Computed lazily when persistent cache access is needed.
 *
 *    - Flag on  (dev/CI):  hash(hlo_bytes || compiler_config)
 *      Content-derived from the lowered HLO/MLIR bytes plus compiler config
 *      (platform, lnc, optimization level, compiler args, neuronx-cc version).
 *      Does NOT include base_key — benign Python edits re-trace/re-lower
 *      (cheap) but still hit the on-disk NEFF cache when HLO bytes are
 *      unchanged, enabling cross-commit cache reuse.
 *      Returns nullptr if HLO bytes are not available (e.g., NeffDirect
 *      paths); callers gracefully skip the persistent cache in that case.
 *
 *    - Flag off (default/prod):  hash(base_key || compiler_config)
 *      Legacy behavior. Any change in base_key (source/version_hash) busts
 *      the persistent cache. Always available once the key is non-empty.
 *
 * ## In-memory key identity:  base_key
 * ## Persistent key identity (flag on):  hash(hlo_bytes || compiler_config)
 * ## Persistent key identity (flag off): hash(base_key || compiler_config)
 *
 */
class CompilationCacheKey {
 public:
  /**
   * @brief Default constructor (empty/invalid key).
   */
  CompilationCacheKey() = default;

  /**
   * @brief Construct from base key string + compiler metadata.
   *
   * @param base_key Frontend-provided computation identifier string
   * @param compiler_args Additional compiler arguments
   * @param optimization_level Optimization level
   * @param hlo_bytes_ptr Pointer to raw hlo bytes data
   * @param source Which frontend produced this (metadata only, not part of key)
   *
   * Weight Layout Optimization (WLO) is inferred from a trailing "|wlo" on base_key, so every
   * construction site (incl. NeffDirectKernelExecution / put_neff_cache / get_neff_cache) stays
   * consistent without threading a separate flag.
   */
  CompilationCacheKey(const std::string& base_key, const std::string& compiler_args,
                      const std::string& optimization_level,
                      const std::vector<uint8_t>* hlo_bytes_ptr = nullptr,
                      CacheKeySource source = CacheKeySource::UNKNOWN);

  /**
   * @brief Construct from a key string + source.
   *
   * For execution-only paths (e.g., NeffDirectKernelExecution).
   *
   * @param key The in-memory key string
   * @param source Which frontend produced this (metadata only, not part of key)
   */
  CompilationCacheKey(const std::string& key, CacheKeySource source);

  /**
   * @brief Get the in-memory key string (base_key).
   *
   * Used for in-memory map lookup and returned to Python for execute_compiled_graph.
   */
  const std::string& GetKey() const { return base_key_; }

  /** @brief Update the in-memory key (e.g., after merging prologues). Invalidates cached strings.
   */
  void SetKey(std::string new_key) {
    base_key_ = std::move(new_key);
    persistent_key_.clear();
    key_str_.clear();
  }

  /**
   * @brief Get persistent key, computing lazily on first call.
   *
   * Computes XXH3_128 over either (hlo_bytes || compiler_config) or
   * (base_key || compiler_config) depending on
   * TORCH_NEURONX_NEFF_KEY_FROM_HLO.
   *
   * Returns nullptr when:
   *   - The key is empty (default-constructed), OR
   *   - HLO-content keying is enabled and no HLO bytes were provided
   *     (e.g., the (key, source) constructor used by NeffDirectKernelExecution).
   * Callers must check for nullptr and skip the persistent cache when so.
   */
  const std::string* GetPersistentKey() const;

  /**
   * @brief Compute the persistent cache key.
   *
   * Hash input depends on TORCH_NEURONX_NEFF_KEY_FROM_HLO:
   *   - on:  hlo_bytes || compiler_config (caller must ensure hlo_bytes_ptr_
   *          is non-null/non-empty; GetPersistentKey() handles the guard)
   *   - off: base_key  || compiler_config
   *
   * @return 32-char hex string (XXH3_128)
   */
  std::string ComputePersistentKey() const;

  /**
   * @brief Check if this key is empty/invalid.
   */
  bool IsEmpty() const { return base_key_.empty(); }

  bool operator==(const CompilationCacheKey& other) const { return base_key_ == other.base_key_; }
  bool operator!=(const CompilationCacheKey& other) const { return base_key_ != other.base_key_; }

  /** @brief Compiler args included in the key. */
  const std::string& GetCompilerArgs() const { return compiler_args_; }

  /** @brief Optimization level included in the key. */
  const std::string& GetOptimizationLevel() const { return optimization_level_; }

  /** @brief Which frontend generated this key. */
  CacheKeySource GetSource() const { return source_; }

  /**
   * @brief Full string description for debugging.
   */
  const std::string& ToString() const;

  /**
   * @brief Build compiler config string from stored metadata.
   *
   * Combines platform_target, logical_neuron_cores, optimization_level, and
   * compiler_args. Used as hash input for persistent key computation.
   */
  std::string BuildCompilerExtension() const;

  /** @brief Get source name string for logging. */
  static const char* SourceName(CacheKeySource source);

  /** @brief Invalidate cached persistent key and ToString.
   *  Call after the underlying HLO bytes content has changed
   *  so that GetPersistentKey() recomputes from the updated bytes. */
  void InvalidateCachedStrings() const {
    persistent_key_.clear();
    key_str_.clear();
  }

  /** @brief Set the HLO bytes pointer for persistent key computation.
   *  Invalidates any previously cached persistent key. */
  void SetHloBytesPtr(const std::vector<uint8_t>* hlo_bytes_ptr) const {
    hlo_bytes_ptr_ = hlo_bytes_ptr;
    InvalidateCachedStrings();
  }

  /** @brief True when the key carries IR, so it can address its NEFF by content.
   *  False for NeffDirectKernelExecution, in both gate settings. */
  bool HasHloBytes() const { return hlo_bytes_ptr_ != nullptr && !hlo_bytes_ptr_->empty(); }

 private:
  // In-memory cache key
  std::string base_key_;

  // Compiler metadata
  std::string compiler_args_;
  std::string optimization_level_;

  // Platform metadata
  std::string platform_target_;
  std::string logical_neuron_cores_;

  // neuronx-cc compiler version (sourced from NEURONXCC_VERSION env var).
  // Included in BuildCompilerExtension so a compiler upgrade busts the cache.
  std::string compiler_version_;

  // Non-owning pointer to HLO bytes for persistent key computation.
  // Lifetime managed by the kernel execution object that owns this key.
  mutable const std::vector<uint8_t>* hlo_bytes_ptr_ = nullptr;

  // Source frontend (for logging/metrics only, not part of identity).
  CacheKeySource source_ = CacheKeySource::UNKNOWN;

  // WLO compile; folded into the cache key.
  bool wlo_enabled_ = false;

  // Persistent key (computed lazily on first GetPersistentKey() call).
  mutable std::string persistent_key_;

  // Cached ToString result (computed lazily).
  mutable std::string key_str_;
};

}  // namespace at::neuron

// std::hash specialization for unordered_map
namespace std {
template <>
struct hash<at::neuron::CompilationCacheKey> {
  size_t operator()(const at::neuron::CompilationCacheKey& key) const noexcept {
    return std::hash<std::string>{}(key.GetKey());
  }
};
}  // namespace std
