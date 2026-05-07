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
 *    from torch.compile).
 *
 * 2. Persistent key: `hash(hlo_bytes || compiler_config)`
 *    Computed on the fly only when persistent cache access is needed.
 *    Uses HLO content hash + compiler config to ensure correctness across
 *    compiler/toolchain versions and different process invocations.
 *
 * ## In-memory key identity:  base_key
 * ## Persistent key identity: hash(hlo_bytes || compiler_config)
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
   * @brief Compute the persistent cache key from stored HLO bytes.
   *
   * Computes: XXH3_128(hlo_bytes || compiler_config) → 32-char hex
   *
   * Uses HLO bytes passed to Create(). Returns empty string if no HLO
   * bytes were provided.
   *
   * @return 32-char hex string, or empty if HLO bytes not available
   */
  /**
   * @brief Get persistent key, computing lazily on first call.
   * Returns nullptr if no HLO bytes were provided to Create().
   */
  const std::string* GetPersistentKey() const;

  /**
   * @brief Compute the persistent cache key from base_key + compiler config.
   *
   * @return 32-char hex string (XXH3_128 of base_key || compiler_config)
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

 private:
  // In-memory cache key
  std::string base_key_;

  // Compiler metadata
  std::string compiler_args_;
  std::string optimization_level_;

  // Platform metadata
  std::string platform_target_;
  std::string logical_neuron_cores_;

  // Non-owning pointer to HLO bytes for persistent key computation.
  // Lifetime managed by the kernel execution object that owns this key.
  mutable const std::vector<uint8_t>* hlo_bytes_ptr_ = nullptr;

  // Source frontend (for logging/metrics only, not part of identity).
  CacheKeySource source_ = CacheKeySource::UNKNOWN;

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
