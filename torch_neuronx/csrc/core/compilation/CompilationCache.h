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

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/cache/FileLock.h"
#include "torch_neuronx/csrc/core/cache/PersistentCacheBackend.h"
#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

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

  // Returns the current number of entries in the in-memory cache.
  size_t GetCacheSize() const;

  // Returns maximum cache size in bytes.
  size_t GetMaxCacheSizeBytes() const { return max_cache_size_bytes_.load(); }

  // Returns maximum number of cache entries.
  size_t GetMaxCacheEntries() const { return max_cache_entries_.load(); }

  // Clears all cache entries and resets statistics.
  //
  // Primarily intended for testing. Cancels any pending compilations.
  // Thread-safe but should not be called during normal operation.
  void Clear();

  // Clears only the in-memory cache entries and resets statistics.
  //
  // Does not clear the persistent cache (NFS/disk). Useful for testing
  // scenarios where persistent cache data should be preserved.
  // Thread-safe but should not be called during normal operation.
  void ClearInMemoryCache();

  // Clears only the persistent cache (NFS/disk) and resets statistics.
  //
  // Does not clear the in-memory cache. Useful for testing scenarios
  // where in-memory cache data should be preserved.
  // Thread-safe but should not be called during normal operation.
  void ClearPersistentCache();

  // Returns all cache keys currently in the cache.
  //
  // Useful for debugging and testing. Thread-safe.
  std::unordered_set<std::string> GetAllCacheKeys() const;

  // Returns a cache entry for the given key, or nullptr if not found.
  //
  // Thread-safe. Used by MegaCache integration to retrieve NEFF info.
  std::shared_ptr<CacheEntry> GetEntry(const std::string& cache_key) const;

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

  // Local cache directory for lock files (initialized from environment or default).
  std::filesystem::path local_cache_dir_;

  // Persistent cache backend (NFS/local disk)
  std::unique_ptr<PersistentCacheBackend> persistent_cache_;

  // Checks if cache limits are exceeded and triggers LRU eviction if needed.
  // Called after adding new entries to maintain cache within configured limits.
  void CheckAndEvictIfNeeded();

  // Calculates current memory usage across all cache entries.
  // Includes both NEFF data and metadata overhead per entry.
  size_t CalculateMemoryUsage() const;

  // Returns cache keys of LRU candidates for eviction.
  // Excludes active entries (those with outstanding NeffBytesPtr references).
  //
  // Args:
  //   count: Maximum number of candidates to return
  //
  // Returns:
  //   Vector of cache keys sorted by last access time (oldest first)
  std::vector<std::string> GetLruCandidates(size_t count) const;

  // Gets the lock file path for a given cache key.
  // Lock files are stored in local_cache_dir_/locks/ to avoid NFS locking issues.
  //
  // Args:
  //   key: The persistent cache key (XXH3_64 hash, 16 hex chars)
  //
  // Returns:
  //   Filesystem path for the lock file
  std::filesystem::path GetLockPath(const std::string& key) const;

  // Checks in-memory cache and pending compilations.
  // Must be called with cache_mutex_ held.
  //
  // Args:
  //   cache_key: Unique identifier for the compilation unit
  //   kernel_execution: Execution context to receive cached NEFF on hit
  //
  // Returns:
  //   true if cache hit (NEFF set on kernel_execution) or pending compilation found
  //   false if cache miss (no entry found, no pending compilation)
  bool CheckInMemoryCacheAndPending(const std::string& cache_key,
                                    const CompilableKernelExecution& kernel_execution);

  // Checks both in-memory and persistent cache if enabled, coordinates compilation work across
  // threads. The persistent_cache_key is computed internally and returned via output parameter for
  // reuse in subsequent Put operations.
  //
  // Args:
  //   cache_key: Unique identifier for the compilation unit
  //   kernel_execution: Execution context to receive cached NEFF on hit
  //   stream_id: Stream identifier for logging
  //   persistent_cache_key: Output parameter to receive computed persistent key.
  //                         Pass nullptr if the key is not needed.
  //
  // Returns:
  //   true if cache hit or pending compilation (caller should not compile)
  //   false if cache miss (caller should compile)
  bool CheckCacheAndCoordinateWork(const std::string& cache_key,
                                   const CompilableKernelExecution& kernel_execution,
                                   uint32_t stream_id, std::string* persistent_cache_key);

  // Waits for another thread's pending compilation to complete.
  // Blocks until the compilation result is available, then sets NEFF on kernel_execution.
  //
  // Args:
  //   cache_key: Cache key of the pending compilation
  //   kernel_execution: Execution context to receive the compiled NEFF
  //   stream_id: Stream identifier for logging
  void WaitForPendingCompilation(const std::string& cache_key,
                                 const CompilableKernelExecution& kernel_execution,
                                 uint32_t stream_id);

  // Compiles IR to NEFF and notifies waiting threads.
  // Called when this thread is responsible for compilation (after CheckCacheAndCoordinateWork
  // returns false).
  //
  // Args:
  //   cache_key: In-memory cache key
  //   kernel_execution: Execution context containing IR to compile
  //   stream_id: Stream identifier for logging
  //   persistent_cache_key: Persistent cache key for file locking and cache storage.
  //                         If nullptr, will be computed internally.
  void Compile(const std::string& cache_key, const CompilableKernelExecution& kernel_execution,
               uint32_t stream_id, std::string* persistent_cache_key);

  // Performs actual compilation with file locking for cross-process coordination.
  // Acquires a file lock to prevent duplicate compilations across processes,
  // compiles the IR, stores result in both in-memory and persistent cache.
  //
  // Args:
  //   kernel_execution: Execution context containing IR to compile
  //   cache_key: In-memory cache key
  //   persistent_cache_key: Key for persistent cache storage and file lock.
  //                         If nullptr, will be computed from kernel_execution.
  //
  // Returns:
  //   Shared pointer to the newly created cache entry
  std::shared_ptr<CacheEntry> CompileAndCache(const CompilableKernelExecution& kernel_execution,
                                              const std::string& cache_key,
                                              std::string* persistent_cache_key);

  // Handles compilation failure by notifying waiting threads and cleaning up.
  // Sets an exception on the promise so waiting threads receive the error.
  //
  // Args:
  //   cache_key: Cache key of the failed compilation
  //   promise: Promise to set exception on (notifies waiting threads)
  //   error_message: Description of the compilation failure
  void HandleCompilationFailure(const std::string& cache_key,
                                std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise,
                                const std::string& error_message);

  // Removes a pending compilation entry from the tracking map.
  // Called after successful compilation or failure handling.
  void CleanupPendingCompilation(const std::string& cache_key);

  // Logs error and throws a runtime_error for compilation failures.
  // Marked [[noreturn]] as it always throws.
  //
  // Args:
  //   cache_key: Cache key associated with the error
  //   message: Error description
  [[noreturn]] void ThrowCompilationError(const std::string& cache_key, const std::string& message);
};

}  // namespace at::neuron
