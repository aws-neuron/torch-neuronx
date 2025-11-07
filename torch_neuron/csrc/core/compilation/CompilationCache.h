#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuron/csrc/core/OperationContext.h"
#include "torch_neuron/csrc/core/compilation/CacheEntry.h"

namespace at::neuron {

// Thread-safe compilation cache for sharing compiled NEFFs across streams.
//
// This cache prevents duplicate compilation work by coordinating compilation
// requests across multiple threads and streams. It manages memory efficiently
// through LRU eviction and tracks active entries to prevent premature eviction
// of in-use NEFFs.
//
// Key features:
// - Thread-safe concurrent access with shared/exclusive locking
// - Automatic LRU eviction based on memory and entry limits
// - Pending compilation coordination to avoid duplicate work
// - Comprehensive statistics and monitoring
// - Environment variable configuration support
class CompilationCache {
 public:
  // Default cache configuration constants.
  // TODO(rpsilva): Instance-specific heuristics for the limits. While configurable, we default
  // to unbounded values for now.
  static constexpr size_t kDefaultMaxCacheSizeBytes =
      std::numeric_limits<size_t>::max();  // unbounded
  static constexpr size_t kDefaultMaxCacheEntries =
      std::numeric_limits<size_t>::max();  // unbounded

  // Constructs a CompilationCache with default or environment-configured limits.
  //
  // Environment variables:
  // - NEURON_COMPILATION_CACHE_MAX_SIZE_MB: Maximum cache size in megabytes
  // - NEURON_COMPILATION_CACHE_MAX_ENTRIES: Maximum number of cache entries
  CompilationCache();

  // Default destructor is sufficient as all resources are managed by smart pointers.
  ~CompilationCache() = default;

  // CompilationCache is not copyable or movable to ensure singleton-like behavior
  // and prevent accidental duplication of cache state.
  CompilationCache(const CompilationCache&) = delete;
  CompilationCache& operator=(const CompilationCache&) = delete;

  // Checks cache and coordinates compilation work across threads.
  //
  // This method implements the core cache coordination logic:
  // - Cache hit: Sets cached NEFF on kernel_execution and returns true
  // - Pending compilation: Returns true (caller should wait)
  // - Cache miss: Registers pending compilation and returns false (caller should compile)
  //
  // Args:
  //   cache_key: Unique identifier for the compilation unit
  //   kernel_execution: Execution context to receive cached NEFF on hit
  //   stream_id: Stream identifier for logging and debugging
  //
  // Returns:
  //   true if work should be skipped (cache hit or pending compilation)
  //   false if work should proceed (cache miss, caller is first thread)
  bool CheckCacheAndCoordinateWork(const std::string& cache_key,
                                   const CompilableKernelExecution& kernel_execution,
                                   uint32_t stream_id);

  // Gets NEFF from cache or compiles IR to NEFF.
  //
  // This is the main entry point for cache operations. It handles the complete
  // flow of cache lookup, compilation coordination, and NEFF retrieval.
  // Compilation configuration is managed by the kernel execution.
  //
  // Args:
  //   kernel_execution: Execution context containing IR and receiving compiled NEFF
  //   stream_id: Stream identifier for logging and debugging
  void GetOrCompileNeff(const CompilableKernelExecution& kernel_execution, uint32_t stream_id);

  // Statistics about cache performance and resource usage.
  struct CacheStats {
    size_t total_entries;                              // Current number of cached entries
    size_t cache_hits;                                 // Total cache hits since creation
    size_t cache_misses;                               // Total cache misses since creation
    double hit_rate;                                   // Hit rate (hits / total_requests)
    size_t memory_usage_bytes;                         // Current memory usage in bytes
    std::chrono::milliseconds total_compilation_time;  // Cumulative compilation time
    size_t evictions_performed;                        // Total evictions performed
    size_t max_cache_size_bytes;                       // Maximum allowed cache size
    size_t max_cache_entries;                          // Maximum allowed cache entries
    double memory_utilization_ratio;                   // Current memory usage ratio
  };

  // Returns comprehensive cache statistics.
  //
  // This method provides a snapshot of cache performance metrics including
  // hit rates, memory usage, and eviction statistics. Thread-safe.
  CacheStats GetStatistics() const;

  // Detailed information about a single cache entry.
  struct CacheEntryInfo {
    std::string cache_key;                                 // Unique cache identifier
    size_t neff_size_bytes;                                // Size of compiled NEFF in bytes
    std::chrono::steady_clock::time_point created_time;    // When entry was created
    std::chrono::steady_clock::time_point last_used_time;  // When entry was last accessed
    uint64_t access_count;                                 // Total number of accesses
    std::chrono::milliseconds compilation_time;            // Time taken to compile
  };

  // Returns detailed information about all cache entries.
  //
  // Useful for debugging and monitoring cache behavior. Thread-safe.
  std::vector<CacheEntryInfo> GetCacheEntries() const;

  // Manually triggers LRU eviction to reduce cache size.
  //
  // Args:
  //   target_entries: Target number of entries after eviction (0 = evict all possible)
  //
  // Returns:
  //   Number of entries actually evicted
  size_t EvictLruEntries(size_t target_entries = 0);

  // Returns current memory usage in bytes.
  //
  // Includes both NEFF data and metadata overhead. Thread-safe.
  size_t GetMemoryUsageBytes() const;

  // Clears all cache entries and resets statistics.
  //
  // Primarily intended for testing. Cancels any pending compilations.
  // Thread-safe but should not be called during normal operation.
  void Clear();

  // Returns all cache keys currently in the cache.
  //
  // Useful for debugging and testing. Thread-safe.
  std::unordered_set<std::string> GetAllCacheKeys() const;

  // Creates a managed pointer to NEFF bytes with automatic lifecycle tracking.
  //
  // The returned pointer automatically marks the cache entry as active during
  // its lifetime and inactive when destroyed, preventing premature eviction.
  //
  // Args:
  //   neff_bytes: Reference to the NEFF data
  //   cache_key: Cache key for lifecycle tracking
  //
  // Returns:
  //   Smart pointer managing NEFF bytes lifecycle
  NeffBytesPtr CreateNeffBytesPtr(const std::vector<uint8_t>& neff_bytes,
                                  const std::string& cache_key);

  // Marks a cache entry as active to prevent eviction.
  //
  // Active entries are protected from LRU eviction. Should be called when
  // a NEFF is being used for execution.
  void MarkEntryActive(const std::string& cache_key);

  // Marks a cache entry as inactive, allowing eviction.
  //
  // Should be called when a NEFF is no longer being used for execution.
  void MarkEntryInactive(const std::string& cache_key);

 private:
  // Main cache storage and synchronization.
  mutable std::shared_mutex cache_mutex_;
  std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache_;

  // Tracks pending compilations to coordinate work across threads.
  struct PendingCompilation {
    std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise;
    std::shared_future<std::shared_ptr<CacheEntry>> future;

    explicit PendingCompilation(std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> p)
        : promise(std::move(p)), future(promise->get_future().share()) {}
  };
  std::unordered_map<std::string, PendingCompilation> pending_compilations_;

  // Tracks active entries to prevent eviction of in-use NEFFs.
  mutable std::shared_mutex active_entries_mutex_;
  std::unordered_set<std::string> active_cache_keys_;

  // Performance and usage statistics (thread-safe atomic counters).
  mutable std::atomic<uint64_t> cache_hits_{0};
  mutable std::atomic<uint64_t> cache_misses_{0};
  mutable std::atomic<uint64_t> total_compilation_time_ms_{0};
  mutable std::atomic<uint64_t> evictions_performed_{0};
  mutable std::atomic<uint64_t> total_compilations_{0};

  // Cache configuration limits (can be modified via environment variables).
  std::atomic<size_t> max_cache_size_bytes_{kDefaultMaxCacheSizeBytes};
  std::atomic<size_t> max_cache_entries_{kDefaultMaxCacheEntries};

  // Cache management helper methods.
  void CheckAndEvictIfNeeded();
  size_t CalculateMemoryUsage() const;
  std::vector<std::string> GetLruCandidates(size_t count) const;

  // Compilation coordination helper methods.
  void WaitForPendingCompilation(const std::string& cache_key,
                                 const CompilableKernelExecution& kernel_execution,
                                 uint32_t stream_id);
  void Compile(const std::string& cache_key, const CompilableKernelExecution& kernel_execution,
               uint32_t stream_id);
  std::shared_ptr<CacheEntry> CompileAndCache(const CompilableKernelExecution& kernel_execution,
                                              const std::string& cache_key);
  void HandleCompilationFailure(const std::string& cache_key,
                                std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise,
                                const std::string& error_message);
  void CleanupPendingCompilation(const std::string& cache_key);
  [[noreturn]] void ThrowCompilationError(const std::string& cache_key, const std::string& message);
};

}  // namespace at::neuron
