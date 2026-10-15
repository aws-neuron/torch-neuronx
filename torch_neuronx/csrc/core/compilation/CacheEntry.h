#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace at::neuron {

// Forward declaration to resolve circular dependency.
class CompilationCache;

// Result of a single HLO→NEFF compilation.
struct CompileResult {
  std::vector<uint8_t> neff_bytes;
  std::vector<uint8_t> wrapped_neff_hlo;
};

/**
 * @brief Cache entry containing compiled NEFF data with access metadata.
 *
 * This struct holds compiled neural network executable format (NEFF) bytes
 * along with timing and access statistics for cache management purposes.
 * Objects are designed to be managed via shared_ptr and are non-copyable.
 */
struct CacheEntry {
  // Compiled NEFF binary data.
  std::vector<uint8_t> neff_bytes;

  // WLO (Weight Layout Optimization) only.
  std::vector<uint8_t> wrapped_neff_hlo;

  // Key for persistent cache (disk). XXH3_128 hash of cache_key + hlo_bytes.
  std::string persistent_cache_key;

  // Timestamp when this entry was created.
  std::chrono::steady_clock::time_point created_time;

  // Timestamp of most recent access (mutable for const access tracking).
  mutable std::chrono::steady_clock::time_point last_used_time;

  // Total number of times this entry has been accessed.
  mutable std::atomic<uint64_t> access_count{0};

  // Time spent compiling this entry.
  std::chrono::milliseconds compilation_time{0};

  CacheEntry() = default;

  /**
   * @brief Constructs a cache entry with NEFF bytes, persistent key, and compilation time.
   * @param bytes Compiled NEFF binary data.
   * @param persistent_key Key for persistent cache storage.
   * @param compilation_time_ms Time spent compiling (defaults to 0).
   */
  CacheEntry(std::vector<uint8_t> bytes, std::string persistent_key,
             std::chrono::milliseconds compilation_time_ms = std::chrono::milliseconds{0})
      : neff_bytes(std::move(bytes)),
        persistent_cache_key(std::move(persistent_key)),
        compilation_time(compilation_time_ms) {
    const auto now = std::chrono::steady_clock::now();
    created_time = now;
    last_used_time = now;
  }

  /**
   * @brief Creates a failed cache entry that records a compilation failure.
   * @param cache_key The cache key for this entry.
   * @param error_msg The original compilation error message.
   */
  static std::shared_ptr<CacheEntry> Failed(std::string cache_key, std::string error_msg) {
    auto entry = std::make_shared<CacheEntry>();
    entry->persistent_cache_key = std::move(cache_key);
    entry->is_failed_ = true;
    entry->failure_message_ = std::move(error_msg);
    const auto now = std::chrono::steady_clock::now();
    entry->created_time = now;
    entry->last_used_time = now;
    return entry;
  }

  bool IsFailed() const { return is_failed_; }
  const std::string& GetFailureMessage() const { return failure_message_; }

  // Disable copy and move operations since objects are managed via shared_ptr.
  CacheEntry(const CacheEntry&) = delete;
  CacheEntry(CacheEntry&&) = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;
  CacheEntry& operator=(CacheEntry&&) = delete;

  /**
   * @brief Updates access metadata when entry is used.
   *
   * This method is const to allow updating access statistics even when
   * the entry itself is accessed through a const reference.
   */
  void UpdateAccess() const {
    last_used_time = std::chrono::steady_clock::now();
    access_count.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  // Whether this entry represents a failed compilation attempt.
  bool is_failed_ = false;

  // Original error message from the failed compilation.
  std::string failure_message_;
};

/**
 * @brief RAII guard that marks cache entries as inactive when released.
 *
 * This custom deleter ensures that when a unique_ptr to NEFF bytes is
 * destroyed, the corresponding cache entry is properly marked as inactive
 * in the compilation cache.
 */
struct CacheEntryGuard {
  // Pointer to the compilation cache (may be null).
  CompilationCache* cache = nullptr;

  // Key identifying the cache entry.
  std::string cache_key;

  // Default constructor for unique_ptr default initialization.
  CacheEntryGuard() = default;

  /**
   * @brief Constructs a guard for the specified cache and key.
   * @param cache_ptr Pointer to the compilation cache.
   * @param key Cache key for the entry.
   */
  CacheEntryGuard(CompilationCache* cache_ptr, std::string key)
      : cache(cache_ptr), cache_key(std::move(key)) {}

  /**
   * @brief Custom deleter function called when unique_ptr is destroyed.
   * @param ptr Pointer to the NEFF bytes being deleted.
   */
  void operator()(const std::vector<uint8_t>* ptr) const;
};

// Type alias for smart pointer to NEFF bytes with custom deleter.
using NeffBytesPtr = std::unique_ptr<const std::vector<uint8_t>, CacheEntryGuard>;

}  // namespace at::neuron
