#include "CompilationCache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "torch_neuron/csrc/core/NeuronLogging.h"

namespace at::neuron {

CompilationCache::CompilationCache() {
  // Configure maximum cache size from environment variable.
  const char* max_size_env = std::getenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
  if (max_size_env != nullptr) {
    try {
      size_t max_size_mb = std::stoul(max_size_env);
      max_cache_size_bytes_ = max_size_mb * 1024 * 1024;
    } catch (const std::exception& e) {
      throw std::runtime_error("Invalid NEURON_COMPILATION_CACHE_MAX_SIZE_MB value: " +
                               std::string(max_size_env) + " (" + e.what() + ")");
    }
  }

  // Configure maximum number of cache entries from environment variable.
  const char* max_entries_env = std::getenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
  if (max_entries_env != nullptr) {
    try {
      size_t max_entries = std::stoul(max_entries_env);
      max_cache_entries_ = max_entries;
    } catch (const std::exception& e) {
      TORCH_NEURON_ERROR("Invalid NEURON_COMPILATION_CACHE_MAX_ENTRIES value",
                         "value=", max_entries_env, "error=", e.what());
      throw std::runtime_error("Invalid NEURON_COMPILATION_CACHE_MAX_ENTRIES value: " +
                               std::string(max_entries_env) + " (" + e.what() + ")");
    }
  }
}

CompilationCache::CacheStats CompilationCache::GetStatistics() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  uint64_t hits = cache_hits_.load();
  uint64_t misses = cache_misses_.load();
  uint64_t total_requests = hits + misses;

  // Calculate total memory usage and count valid entries
  size_t total_memory = 0;
  size_t valid_entries = 0;

  for (const auto& [key, entry] : cache_) {
    if (entry) {
      total_memory += entry->neff_bytes.size();
      total_memory += sizeof(CacheEntry) + key.size();
      valid_entries++;
    }
  }

  CacheStats stats;
  stats.total_entries = valid_entries;
  stats.cache_hits = hits;
  stats.cache_misses = misses;
  stats.hit_rate = total_requests > 0 ? static_cast<double>(hits) / total_requests : 0.0;
  stats.memory_usage_bytes = total_memory;
  stats.total_compilation_time = std::chrono::milliseconds(total_compilation_time_ms_.load());
  stats.evictions_performed = evictions_performed_.load();
  stats.max_cache_size_bytes = max_cache_size_bytes_.load();
  stats.max_cache_entries = max_cache_entries_.load();
  stats.memory_utilization_ratio =
      stats.max_cache_size_bytes > 0
          ? static_cast<double>(stats.memory_usage_bytes) / stats.max_cache_size_bytes
          : 0.0;
  return stats;
}

std::vector<CompilationCache::CacheEntryInfo> CompilationCache::GetCacheEntries() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  std::vector<CacheEntryInfo> entries;
  entries.reserve(cache_.size());

  for (const auto& [key, entry] : cache_) {
    if (entry) {
      CacheEntryInfo info;
      info.cache_key = key;
      info.neff_size_bytes = entry->neff_bytes.size();
      info.created_time = entry->created_time;
      info.last_used_time = entry->last_used_time;
      info.access_count = entry->access_count.load();
      info.compilation_time = entry->compilation_time;

      entries.push_back(info);
    }
  }

  return entries;
}

void CompilationCache::Clear() {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);

  // Cancel any pending compilations by setting exceptions.
  for (auto& [key, pending] : pending_compilations_) {
    try {
      pending.promise->set_exception(std::make_exception_ptr(
          std::runtime_error("Cache cleared while compilation was pending")));
    } catch (...) {
      // Ignore exceptions if promise is already fulfilled.
    }
  }

  cache_.clear();
  pending_compilations_.clear();

  // Reset aggregate statistics
  cache_hits_ = 0;
  cache_misses_ = 0;
  total_compilation_time_ms_ = 0;
  evictions_performed_ = 0;
  total_compilations_ = 0;
}

bool CompilationCache::CheckCacheAndCoordinateWork(
    const std::string& cache_key, const CompilableKernelExecution& kernel_execution,
    uint32_t stream_id) {
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);

    // Check for completed NEFF in cache.
    auto cache_it = cache_.find(cache_key);
    if (cache_it != cache_.end()) {
      auto cached_entry = cache_it->second;
      TORCH_CHECK(cached_entry != nullptr, "Cache entry can not be null");
      // Record cache hit and update access statistics.
      cache_hits_.fetch_add(1, std::memory_order_relaxed);
      cached_entry->UpdateAccess();

      // Create managed pointer and set on kernel execution.
      auto neff_ptr = CreateNeffBytesPtr(cached_entry->neff_bytes, cache_key);
      kernel_execution.SetCachedNeff(std::move(neff_ptr));
      TORCH_NEURON_DEBUG("Cache hit", "key=", cache_key);
      return true;
    }

    // No completed NEFF found - check if compilation is already pending.
    auto pending_it = pending_compilations_.find(cache_key);
    if (pending_it != pending_compilations_.end()) {
      // Another thread is compiling, count as hit since we'll get the result.
      cache_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    } else {
      // Cache miss - this thread should compile.
      cache_misses_.fetch_add(1, std::memory_order_relaxed);
      auto compilation_promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
      pending_compilations_.emplace(cache_key, PendingCompilation(compilation_promise));
      TORCH_NEURON_DEBUG("Cache miss", "key=", cache_key);
      return false;
    }
  }
}

void CompilationCache::GetOrCompileNeff(const CompilableKernelExecution& kernel_execution,
                                        uint32_t stream_id) {
  const std::string& cache_key = kernel_execution.GetCacheKey();

  // Check cache and coordinate work scheduling across threads.
  if (CheckCacheAndCoordinateWork(cache_key, kernel_execution, stream_id)) {
    if (kernel_execution.HasCachedNeff()) {
      return;  // Cache hit - NEFF already set on kernel execution.
    }
    // Another thread is compiling - wait for completion.
    WaitForPendingCompilation(cache_key, kernel_execution, stream_id);
    return;
  }

  // Cache miss - this thread should perform compilation.
  Compile(cache_key, kernel_execution, stream_id);
}

size_t CompilationCache::GetMemoryUsageBytes() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);
  return CalculateMemoryUsage();
}

void CompilationCache::CheckAndEvictIfNeeded() {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);

  // Evaluate current cache usage against configured limits.
  size_t current_memory = CalculateMemoryUsage();
  size_t current_entries = cache_.size();
  size_t max_memory = max_cache_size_bytes_.load();
  size_t max_entries = max_cache_entries_.load();

  // Determine if eviction is necessary.
  bool memory_exceeded = max_memory > 0 && current_memory >= max_memory;
  bool entries_exceeded = max_entries > 0 && current_entries >= max_entries;

  if (!memory_exceeded && !entries_exceeded) {
    return;  // No eviction needed.
  }

  // Calculate target number of entries after eviction.
  size_t target_entries = current_entries;
  if (memory_exceeded) {
    // Target 80% of maximum memory usage to provide headroom.
    size_t target_memory = static_cast<size_t>(max_memory * 0.8);
    // Estimate entries to remove based on average entry size.
    size_t avg_entry_size = current_entries > 0 ? current_memory / current_entries : 0;
    if (avg_entry_size > 0) {
      size_t memory_to_free = current_memory - target_memory;
      size_t entries_to_remove = (memory_to_free + avg_entry_size - 1) / avg_entry_size;
      target_entries =
          current_entries > entries_to_remove ? current_entries - entries_to_remove : 0;
    }
  }
  if (entries_exceeded) {
    // When entry count is exceeded, target the configured maximum.
    target_entries = std::min(target_entries, max_entries);
  }

  // Perform LRU eviction to reach target size.
  size_t evicted = EvictLruEntries(target_entries);
  if (evicted > 0) {
    evictions_performed_ += evicted;
    TORCH_NEURON_DEBUG("Cache eviction performed", "evicted_entries=", evicted);
  }
}

size_t CompilationCache::EvictLruEntries(size_t target_entries) {
  if (cache_.empty() || target_entries >= cache_.size()) {
    return 0;  // Nothing to evict.
  }

  size_t entries_to_remove = cache_.size() - target_entries;
  std::shared_lock<std::shared_mutex> active_lock(active_entries_mutex_);
  auto lru_candidates = GetLruCandidates(entries_to_remove);

  size_t evicted = 0;
  for (const auto& key : lru_candidates) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      bool is_active = active_cache_keys_.find(key) != active_cache_keys_.end();

      if (!is_active) {
        cache_.erase(it);
        evicted++;
      }
    }
  }

  return evicted;
}

size_t CompilationCache::CalculateMemoryUsage() const {
  size_t total_memory = 0;
  for (const auto& [key, entry] : cache_) {
    if (entry != nullptr) {
      // Include both NEFF data and metadata overhead.
      total_memory += entry->neff_bytes.size();
      total_memory += sizeof(CacheEntry) + key.size();
    }
  }
  return total_memory;
}

std::vector<std::string> CompilationCache::GetLruCandidates(size_t count) const {
  if (count == 0 || cache_.empty()) {
    return {};
  }

  std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> candidates;
  candidates.reserve(cache_.size());

  for (const auto& [key, entry] : cache_) {
    bool is_active = active_cache_keys_.find(key) != active_cache_keys_.end();

    if (entry != nullptr && !is_active) {
      candidates.emplace_back(key, entry->last_used_time);
    }
  }

  // Sort by last_used_time (oldest entries first for LRU eviction).
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Extract keys of the oldest inactive entries up to requested count.
  std::vector<std::string> lru_keys;
  lru_keys.reserve(std::min(count, candidates.size()));

  for (size_t i = 0; i < std::min(count, candidates.size()); ++i) {
    lru_keys.push_back(candidates[i].first);
  }

  return lru_keys;
}

std::unordered_set<std::string> CompilationCache::GetAllCacheKeys() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);

  std::unordered_set<std::string> cache_keys;
  for (const auto& [key, entry] : cache_) {
    if (entry != nullptr) {
      cache_keys.insert(key);
    }
  }
  return cache_keys;
}

at::neuron::NeffBytesPtr CompilationCache::CreateNeffBytesPtr(
    const std::vector<uint8_t>& neff_bytes, const std::string& cache_key) {
  MarkEntryActive(cache_key);
  CacheEntryGuard guard(this, cache_key);
  return NeffBytesPtr(&neff_bytes, std::move(guard));
}

void CompilationCache::MarkEntryActive(const std::string& cache_key) {
  std::unique_lock<std::shared_mutex> lock(active_entries_mutex_);
  active_cache_keys_.insert(cache_key);
  TORCH_NEURON_DEBUG("Cache entry marked active", "cache_key=", cache_key,
                     "total_active=", active_cache_keys_.size());
}

void CompilationCache::MarkEntryInactive(const std::string& cache_key) {
  std::unique_lock<std::shared_mutex> lock(active_entries_mutex_);
  active_cache_keys_.erase(cache_key);
  TORCH_NEURON_DEBUG("Cache entry marked inactive", "cache_key=", cache_key,
                     "total_active=", active_cache_keys_.size());
}

void CompilationCache::WaitForPendingCompilation(const std::string& cache_key,
                                                 const CompilableKernelExecution& kernel_execution,
                                                 uint32_t stream_id) {
  TORCH_NEURON_DEBUG("Waiting for another thread's compilation", "stream_id=", stream_id,
                     "cache_key=", cache_key);

  // Retrieve the shared future for the pending compilation.
  std::shared_future<std::shared_ptr<CacheEntry>> compilation_future;
  {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    auto pending_it = pending_compilations_.find(cache_key);
    if (pending_it != pending_compilations_.end()) {
      compilation_future = pending_it->second.future;
      TORCH_NEURON_DEBUG("Found pending compilation to wait for", "cache_key=", cache_key);
    } else {
      TORCH_NEURON_DEBUG("No pending compilation found", "cache_key=", cache_key);
    }
  }

  if (!compilation_future.valid()) {
    ThrowCompilationError(cache_key, "No pending compilation found");
  }

  try {
    TORCH_NEURON_DEBUG("Waiting for compilation to complete", "stream_id=", stream_id,
                       "cache_key=", cache_key);
    auto cache_entry = compilation_future.get();
    auto neff_ptr = CreateNeffBytesPtr(cache_entry->neff_bytes, cache_key);
    kernel_execution.SetCachedNeff(std::move(neff_ptr));

    TORCH_NEURON_DEBUG("Compilation wait completed successfully", "stream_id=", stream_id,
                       "cache_key=", cache_key);
  } catch (const std::exception& e) {
    ThrowCompilationError(cache_key, "Pending compilation failed: " + std::string(e.what()));
  }
}

void CompilationCache::Compile(const std::string& cache_key,
                               const CompilableKernelExecution& kernel_execution,
                               uint32_t stream_id) {
  // Retrieve the compilation promise for this cache key.
  std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> compilation_promise;
  {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    auto pending_it = pending_compilations_.find(cache_key);
    if (pending_it != pending_compilations_.end()) {
      compilation_promise = pending_it->second.promise;
    }
  }

  if (!compilation_promise) {
    ThrowCompilationError(cache_key, "No pending compilation promise found");
  }

  try {
    TORCH_NEURON_DEBUG("Compiling IR to NEFF", "stream_id=", stream_id, "cache_key=", cache_key);

    auto compilation_result = CompileAndCache(kernel_execution, cache_key);

    // Notify all waiting threads that compilation is complete.
    compilation_promise->set_value(compilation_result);
    CleanupPendingCompilation(cache_key);

    // Set cached NEFF for the current execution context.
    auto neff_ptr = CreateNeffBytesPtr(compilation_result->neff_bytes, cache_key);
    kernel_execution.SetCachedNeff(std::move(neff_ptr));

    TORCH_NEURON_DEBUG("IR compilation complete", "cache_key=", cache_key);

  } catch (const std::exception& e) {
    HandleCompilationFailure(cache_key, compilation_promise, e.what());
    throw;
  }
}

std::shared_ptr<CacheEntry> CompilationCache::CompileAndCache(
    const CompilableKernelExecution& kernel_execution, const std::string& cache_key) {
  auto start_time = std::chrono::steady_clock::now();

  // Perform the actual IR to NEFF compilation.
  auto neff_bytes = kernel_execution.CompileToNeff();

  auto end_time = std::chrono::steady_clock::now();
  auto compilation_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Update aggregate compilation statistics.
  total_compilation_time_ms_ += compilation_time.count();
  total_compilations_++;

  // Create cache entry with compiled NEFF and timing information.
  auto cache_entry = std::make_shared<CacheEntry>(std::move(neff_bytes), compilation_time);

  // Store the cache entry atomically.
  {
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    cache_[cache_key] = cache_entry;
  }

  // Check if eviction is needed after adding new entry.
  CheckAndEvictIfNeeded();

  return cache_entry;
}

void CompilationCache::HandleCompilationFailure(
    const std::string& cache_key,
    std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise,
    const std::string& error_message) {
  std::string full_error_msg =
      "Compilation failed for cache key '" + cache_key + "': " + error_message;
  TORCH_NEURON_ERROR("Compilation error", "cache_key=", cache_key, "error=", error_message);

  // Notify all waiting threads of the compilation failure.
  if (promise) {
    promise->set_exception(std::make_exception_ptr(std::runtime_error(full_error_msg)));
  }

  CleanupPendingCompilation(cache_key);
}

void CompilationCache::CleanupPendingCompilation(const std::string& cache_key) {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);
  pending_compilations_.erase(cache_key);
}

[[noreturn]] void CompilationCache::ThrowCompilationError(const std::string& cache_key,
                                                          const std::string& message) {
  TORCH_NEURON_ERROR("Compilation error", "cache_key=", cache_key, "error=", message);
  throw std::runtime_error("Compilation failed for cache key '" + cache_key + "': " + message);
}

}  // namespace at::neuron
