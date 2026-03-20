#include "CompilationCache.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"
#include "torch_neuronx/csrc/core/compilation/CompilerDebugUtils.h"
#include "torch_neuronx/csrc/core/compilation/LocalCompilationBackend.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/compilation/service/ServiceCompilationBackend.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

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
      TORCH_NEURONX_ERROR("Invalid NEURON_COMPILATION_CACHE_MAX_ENTRIES value",
                          "value=", max_entries_env, "error=", e.what());
      throw std::runtime_error("Invalid NEURON_COMPILATION_CACHE_MAX_ENTRIES value: " +
                               std::string(max_entries_env) + " (" + e.what() + ")");
    }
  }

  // Initialize local cache directory for lock files.
  const char* neuron_local_cache_dir = std::getenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR");
  if (neuron_local_cache_dir != nullptr && neuron_local_cache_dir[0] != '\0') {
    local_cache_dir_ = std::filesystem::path(neuron_local_cache_dir);
    std::error_code ec;
    std::filesystem::create_directories(local_cache_dir_, ec);
    if (ec) {
      throw std::runtime_error("Invalid TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR value: path=" +
                               std::string(neuron_local_cache_dir) + ", error=" + ec.message());
    }
  } else {
    local_cache_dir_ = std::filesystem::path("/tmp/local_cache");
    std::error_code ec;
    std::filesystem::create_directories(local_cache_dir_, ec);
    if (ec) {
      throw std::runtime_error("Failed to create local cache directory: path=" +
                               local_cache_dir_.string() + ", error=" + ec.message());
    }
  }

  // Initialize persistent cache backend if not disabled.
  if (!PersistentCacheBackend::IsCachingDisabled()) {
    try {
      persistent_cache_ = std::make_unique<PersistentCacheBackend>();
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN(
          "Failed to initialize persistent cache backend, continuing without "
          "persistent caching",
          "error=", e.what());
      persistent_cache_ = nullptr;
    }
  }

  // Initialize compilation backend.
  // NEURON_COMPILE_SERVICE=0 disables the service (uses local compilation).
  // Default: service enabled.
  const char* service_env = std::getenv("NEURON_COMPILE_SERVICE");
  bool service_enabled = !(service_env && std::string(service_env) == "0");

  if (service_enabled) {
    std::string socket_name = "neuron_compile_service_" + std::to_string(getuid());
    // When distributed is initialized, all ranks share MASTER_PORT so they
    // connect to the same compilation service socket.
    // When not distributed, use PID to give each process its own socket
    // (uid is in the base name to separate users).
    if (c10_neuron::is_distributed_initialized()) {
      const char* master_port = std::getenv("MASTER_PORT");
      if (master_port && master_port[0] != '\0') {
        socket_name += "_";
        socket_name += master_port;
      } else {
        socket_name += "_" + std::to_string(getpid());
      }
    } else {
      socket_name += "_" + std::to_string(getpid());
    }
    std::string socket_path = "/tmp/" + socket_name + ".sock";
    try {
      backend_ = std::make_unique<ServiceCompilationBackend>(socket_path);
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN("Failed to connect to compilation service:", e.what(),
                         "— falling back to local backend");
      backend_ = std::make_unique<LocalCompilationBackend>();
    }
  } else {
    backend_ = std::make_unique<LocalCompilationBackend>();
  }
  TORCH_NEURONX_DEBUG("CompilationCache backend:", backend_->GetName());

  // Thread pool for parallel compilation via GetOrCompileNeffAsync.
  compilation_pool_ = std::make_unique<CompilationThreadPool>();
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

void CompilationCache::ClearInMemoryCache() {
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

  // Reset in-memory cache metrics
  auto* arena = metrics::NeuronMetricsArena::Get();
  if (auto* counter = arena->GetCounter("CompilationCache.InMemoryHits")) counter->Clear();
  if (auto* counter = arena->GetCounter("CompilationCache.InMemoryMisses")) counter->Clear();
  if (auto* counter = arena->GetCounter("CompilationCache.Evictions")) counter->Clear();
  if (auto* metric = arena->GetMetric("CompilationCache.MemoryUsage")) metric->Clear();
}

void CompilationCache::ClearPersistentCache() {
  if (persistent_cache_) {
    persistent_cache_->Clear();
    // Clear persistent cache metrics
    auto* arena = metrics::NeuronMetricsArena::Get();
    if (auto* counter = arena->GetCounter("CompilationCache.PersistentHits")) counter->Clear();
    if (auto* counter = arena->GetCounter("CompilationCache.PersistentMisses")) counter->Clear();
  }
}

void CompilationCache::Clear() {
  ClearInMemoryCache();
  ClearPersistentCache();

  // Clear total compilations and compilation time metrics
  auto* arena = metrics::NeuronMetricsArena::Get();
  if (auto* counter = arena->GetCounter("CompilationCache.TotalCompilations")) counter->Clear();
  if (auto* metric = arena->GetMetric("CompilationCache.CompilationTime")) metric->Clear();
}

bool CompilationCache::CheckInMemoryCacheAndPending(
    const CompilationCacheKey& key, const CompilableKernelExecution& kernel_execution) {
  const std::string& cache_key = key.GetKey();
  // Check for completed NEFF in in-memory cache.
  auto cache_it = cache_.find(cache_key);
  if (cache_it != cache_.end()) {
    auto cached_entry = cache_it->second;
    TORCH_CHECK(cached_entry != nullptr, "Cache entry can not be null");

    // Fast-fail if this entry records a previous compilation failure.
    if (cached_entry->IsFailed()) {
      ThrowCompilationError(key, cached_entry->GetFailureMessage());
    }

    // Record in-memory cache hit and increment access statistics.
    TORCH_NEURONX_COUNTER("CompilationCache.InMemoryHits", 1);
    cached_entry->UpdateAccess();

    // Create managed pointer and set on kernel execution.
    auto neff_ptr = CreateNeffBytesPtr(cached_entry->neff_bytes, cache_key);
    kernel_execution.SetCachedNeff(std::move(neff_ptr));
    TORCH_NEURONX_DEBUG("In-memory cache hit: ", key.ToString());
    return true;
  }

  // No completed NEFF found - check if compilation is already pending.
  auto pending_it = pending_compilations_.find(cache_key);
  if (pending_it != pending_compilations_.end()) {
    // Another thread is compiling, count as in-memory hit since we'll get the result.
    TORCH_NEURONX_COUNTER("CompilationCache.InMemoryHits", 1);
    return true;
  }

  return false;
}

bool CompilationCache::CheckCacheAndCoordinateWork(
    const CompilationCacheKey& key, const CompilableKernelExecution& kernel_execution,
    uint32_t stream_id) {
  const std::string& cache_key = key.GetKey();
  // First check in-memory cache and pending compilations with lock held.
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    if (CheckInMemoryCacheAndPending(key, kernel_execution)) {
      return true;
    }
    // Record in-memory cache miss.
    TORCH_NEURONX_COUNTER("CompilationCache.InMemoryMisses", 1);
    TORCH_NEURONX_DEBUG("In memory cache miss:", key.ToString());

    // No persistent cache — register pending now while lock is held.
    if (!persistent_cache_) {
      auto compilation_promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
      pending_compilations_.emplace(cache_key, PendingCompilation(compilation_promise));
      return false;
    }
  }  // Lock released before I/O persistent cache check.

  // Check persistent cache (NFS/disk) using the persistent key from CompilationCacheKey.
  const std::string* persistent_cache_key = key.GetPersistentKey();
  TORCH_CHECK(persistent_cache_key != nullptr,
              "No HLO bytes available to compute persistent key. "
              "Ensure HLO bytes are passed to CompilationCacheKey().");
  auto neff_bytes = persistent_cache_->Get(*persistent_cache_key);
  if (neff_bytes) {
    TORCH_NEURONX_DEBUG("Persistent cache hit:", key.ToString());

    // Record persistent cache hit.
    TORCH_NEURONX_COUNTER("CompilationCache.PersistentHits", 1);

    // Create cache entry and store in memory cache.
    auto cache_entry = std::make_shared<CacheEntry>(std::move(*neff_bytes), *persistent_cache_key,
                                                    std::chrono::milliseconds(0));

    {
      std::unique_lock<std::shared_mutex> lock(cache_mutex_);
      cache_[cache_key] = cache_entry;

      // Create managed pointer and set on kernel execution.
      auto neff_ptr = CreateNeffBytesPtr(cache_entry->neff_bytes, cache_key);
      kernel_execution.SetCachedNeff(std::move(neff_ptr));
    }
    // Check if eviction is needed after adding new entry.
    CheckAndEvictIfNeeded();
    return true;
  }

  // Persistent cache miss — double-check and register pending.
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    TORCH_NEURONX_COUNTER("CompilationCache.PersistentMisses", 1);
    TORCH_NEURONX_DEBUG("Persistent cache miss:", key.ToString());
    // Double-check: another thread may have populated cache or registered compilation
    // while we were checking persistent.
    if (CheckInMemoryCacheAndPending(key, kernel_execution)) {
      return true;
    }

    // Register pending compilation while still holding lock to prevent race.
    auto compilation_promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
    pending_compilations_.emplace(cache_key, PendingCompilation(compilation_promise));
    return false;
  }
}

void CompilationCache::GetOrCompileNeff(const CompilableKernelExecution& kernel_execution,
                                        uint32_t stream_id) {
  const auto& key = kernel_execution.GetCompilationCacheKey();

  // Check caches (memory + persistent) and coordinate work scheduling.
  if (CheckCacheAndCoordinateWork(key, kernel_execution, stream_id)) {
    if (kernel_execution.HasCachedNeff()) {
      return;  // Cache hit - NEFF already set.
    }
    // Another thread is compiling - wait for completion.
    WaitForPendingCompilation(key, kernel_execution, stream_id);
    return;
  }

  // Cache miss (both memory and persistent) - this thread should compile.
  Compile(key, kernel_execution, stream_id);
}

void CompilationCache::GetOrCompileNeffAsync(const CompilableKernelExecution& kernel_execution,
                                             uint32_t stream_id,
                                             std::function<void(bool success)> callback) {
  const auto& key = kernel_execution.GetCompilationCacheKey();
  const std::string& cache_key = key.GetKey();

  // Check caches (memory + persistent) and coordinate work scheduling.
  if (CheckCacheAndCoordinateWork(key, kernel_execution, stream_id)) {
    if (kernel_execution.HasCachedNeff()) {
      // Cache hit — invoke callback synchronously.
      callback(true);
      return;
    }
    // Another thread is compiling — get the future before it's cleaned up.
    std::shared_future<std::shared_ptr<CacheEntry>> compilation_future;
    {
      std::shared_lock<std::shared_mutex> lock(cache_mutex_);
      auto pending_it = pending_compilations_.find(cache_key);
      if (pending_it != pending_compilations_.end()) {
        compilation_future = pending_it->second.future;
      }
    }

    if (!compilation_future.valid()) {
      std::shared_lock<std::shared_mutex> lock(cache_mutex_);
      auto it = cache_.find(cache_key);
      if (it != cache_.end() && it->second) {
        auto neff_ptr = CreateNeffBytesPtr(it->second->neff_bytes, cache_key);
        kernel_execution.SetCachedNeff(std::move(neff_ptr));
        callback(true);
        return;
      }
      callback(false);
      return;
    }

    // Submit wait to thread pool (don't block compilation worker).
    std::string cache_key_copy = cache_key;
    compilation_pool_->Submit([this, cache_key_copy, &kernel_execution,
                               compilation_future = std::move(compilation_future),
                               callback = std::move(callback)]() {
      try {
        auto cache_entry = compilation_future.get();
        auto neff_ptr = CreateNeffBytesPtr(cache_entry->neff_bytes, cache_key_copy);
        kernel_execution.SetCachedNeff(std::move(neff_ptr));
        callback(true);
      } catch (...) {
        callback(false);
      }
    });
    return;
  }

  // Cache miss — submit compilation to thread pool.
  CompilationCacheKey key_copy = key;
  compilation_pool_->Submit(
      [this, key_copy, &kernel_execution, stream_id, callback = std::move(callback)]() {
        try {
          Compile(key_copy, kernel_execution, stream_id);
          callback(true);
        } catch (...) {
          callback(false);
        }
      });
}

std::string CompilationCache::GetBackendName() const {
  return backend_ ? backend_->GetName() : "none";
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
    TORCH_NEURONX_COUNTER("CompilationCache.Evictions", evicted);
    TORCH_NEURONX_DEBUG("Cache eviction performed", "evicted_entries=", evicted);
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

size_t CompilationCache::GetCacheSize() const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);
  return cache_.size();
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

std::shared_ptr<CacheEntry> CompilationCache::GetEntry(const std::string& cache_key) const {
  std::shared_lock<std::shared_mutex> lock(cache_mutex_);
  auto it = cache_.find(cache_key);
  // cache_ maps cache_key -> shared_ptr<CacheEntry>
  // Check both that key exists and entry is valid (not evicted)
  if (it != cache_.end() && it->second) {
    return it->second;
  }
  return nullptr;
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
  TORCH_NEURONX_DEBUG("Cache entry marked active", "cache_key=", cache_key,
                      "total_active=", active_cache_keys_.size());
}

void CompilationCache::MarkEntryInactive(const std::string& cache_key) {
  std::unique_lock<std::shared_mutex> lock(active_entries_mutex_);
  active_cache_keys_.erase(cache_key);
  TORCH_NEURONX_DEBUG("Cache entry marked inactive", "cache_key=", cache_key,
                      "total_active=", active_cache_keys_.size());
}

void CompilationCache::WaitForPendingCompilation(const CompilationCacheKey& key,
                                                 const CompilableKernelExecution& kernel_execution,
                                                 uint32_t stream_id) {
  const std::string& cache_key = key.GetKey();
  TORCH_NEURONX_DEBUG("Waiting for another thread's compilation", "stream_id=", stream_id,
                      "CompilationCacheKey=", key.ToString());

  // Retrieve the shared future for the pending compilation.
  std::shared_future<std::shared_ptr<CacheEntry>> compilation_future;
  {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    auto pending_it = pending_compilations_.find(cache_key);
    if (pending_it != pending_compilations_.end()) {
      compilation_future = pending_it->second.future;
      TORCH_NEURONX_DEBUG("Found pending compilation to wait for", key.ToString());
    } else {
      TORCH_NEURONX_DEBUG("No pending compilation found", key.ToString());
    }
  }

  if (!compilation_future.valid()) {
    ThrowCompilationError(key, "No pending compilation found");
  }

  try {
    TORCH_NEURONX_DEBUG("Waiting for compilation to complete", "stream_id=", stream_id,
                        "CompilationCacheKey=", key.ToString());
    auto cache_entry = compilation_future.get();
    auto neff_ptr = CreateNeffBytesPtr(cache_entry->neff_bytes, cache_key);
    kernel_execution.SetCachedNeff(std::move(neff_ptr));

    TORCH_NEURONX_DEBUG("Compilation wait completed successfully", "stream_id=", stream_id,
                        "CompilationCacheKey=", key.ToString());
  } catch (const std::exception& e) {
    ThrowCompilationError(key, "Pending compilation failed: " + std::string(e.what()));
  }
}

void CompilationCache::Compile(const CompilationCacheKey& key,
                               const CompilableKernelExecution& kernel_execution,
                               uint32_t stream_id) {
  const std::string& cache_key = key.GetKey();
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
    ThrowCompilationError(key, "No pending compilation promise found");
  }

  try {
    TORCH_NEURONX_DEBUG("Compiling IR to NEFF", "stream_id=", stream_id,
                        "CompilationCacheKey=", key.ToString());

    auto compilation_result = CompileAndCache(kernel_execution, key);

    // Notify all waiting threads that compilation is complete.
    compilation_promise->set_value(compilation_result);
    CleanupPendingCompilation(key);

    // Set cached NEFF for the current execution context.
    auto neff_ptr = CreateNeffBytesPtr(compilation_result->neff_bytes, cache_key);
    kernel_execution.SetCachedNeff(std::move(neff_ptr));

    TORCH_NEURONX_DEBUG("IR compilation complete:", key.ToString());

  } catch (const std::exception& e) {
    HandleCompilationFailure(key, compilation_promise, e.what());
    throw;
  }
}

std::shared_ptr<CacheEntry> CompilationCache::CompileAndCache(
    const CompilableKernelExecution& kernel_execution, const CompilationCacheKey& key) {
  const std::string& cache_key = key.GetKey();
  // Get persistent key for file lock and cache storage.
  const std::string* persistent_cache_key = key.GetPersistentKey();

  std::filesystem::path lock_path = GetLockPath(*persistent_cache_key);

  TORCH_NEURONX_DEBUG("Acquiring compilation lock:", "CompilationCacheKey=", key.ToString(),
                      "lock_path=", lock_path.string());

  // Acquire file lock to coordinate compilation across processes.
  FileLockGuard lock_guard(lock_path);

  TORCH_NEURONX_DEBUG("Compilation lock acquired:", key.ToString());

  // Double-check persistent cache after acquiring lock.
  // Another process may have completed compilation while waiting for lock.
  if (persistent_cache_) {
    auto neff_bytes = persistent_cache_->Get(*persistent_cache_key);
    // Persistent cache hit - entry populated while waiting for compilation lock.
    if (neff_bytes) {
      TORCH_NEURONX_DEBUG("Persistent cache hit after lock acquisition:", key.ToString());

      // Record persistent cache hit.
      TORCH_NEURONX_COUNTER("CompilationCache.PersistentHits", 1);

      // Create cache entry from persistent cache hit.
      auto cache_entry = std::make_shared<CacheEntry>(std::move(*neff_bytes), *persistent_cache_key,
                                                      std::chrono::milliseconds(0));

      // Store in memory cache using original cache_key.
      {
        std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
        cache_[cache_key] = cache_entry;
      }

      // Check if eviction is needed after adding new entry.
      CheckAndEvictIfNeeded();

      return cache_entry;
    }
  }

  auto start_time = std::chrono::steady_clock::now();

  // Dump IR details if requested (before compilation)
  std::string ir_type = (!kernel_execution.HasCollectives() && NeuronCompiler::IsStableHLOEnabled())
                            ? "StableHLO"
                            : "XLA";
  if (debug::IsDumpEnabled()) {
    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = kernel_execution.GetOptimizationLevel(),
                             .additional_args = kernel_execution.GetAdditionalArgs()};
    debug::DumpIRDetails(kernel_execution.GetHloBytes(), config, ir_type, cache_key,
                         "CompilableKernelExecution");
  }

  // Compile via backend: local (in-process neuronx-cc) or service (remote RPC).
  auto neff_bytes = backend_->Compile(
      kernel_execution.GetHloBytes(), *persistent_cache_key, ir_type,
      kernel_execution.GetOptimizationLevel(), kernel_execution.GetAdditionalArgs(),
      [&kernel_execution]() -> std::vector<uint8_t> { return kernel_execution.CompileToNeff(); });

  auto end_time = std::chrono::steady_clock::now();
  auto compilation_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Update centralized metrics
  TORCH_NEURONX_COUNTER("CompilationCache.TotalCompilations", 1);
  TORCH_NEURONX_BYTES_METRIC("CompilationCache.NeffSize", neff_bytes.size());
  TORCH_NEURONX_TIME_METRIC("CompilationCache.CompilationTime", start_time, end_time);

  // Create cache entry with compiled NEFF and timing information.
  auto cache_entry =
      std::make_shared<CacheEntry>(std::move(neff_bytes), *persistent_cache_key, compilation_time);

  // Store the cache entry in memory using original cache_key.
  {
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    cache_[cache_key] = cache_entry;
  }

  // Store to persistent cache (NFS/disk) for cross-process sharing.
  // Use persistent_cache_key.
  if (persistent_cache_) {
    if (persistent_cache_->Put(*persistent_cache_key, cache_entry->neff_bytes)) {
      TORCH_NEURONX_DEBUG("Stored to persistent cache", key.ToString());
    }
  }

  // Check if eviction is needed after adding new entry.
  CheckAndEvictIfNeeded();

  return cache_entry;
}

void CompilationCache::HandleCompilationFailure(
    const CompilationCacheKey& key,
    std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise,
    const std::string& error_message) {
  const std::string& cache_key = key.GetKey();
  std::string full_error_msg =
      "Compilation failed for cache key '" + cache_key + "': " + error_message;
  TORCH_NEURONX_DEBUG("Compilation error", "CompilationCacheKey=", key.ToString(),
                      "error=", error_message);

  // Cache the failed entry so subsequent lookups fail fast.
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_[cache_key] = CacheEntry::Failed(cache_key, full_error_msg);
  }
  CheckAndEvictIfNeeded();

  // Notify all waiting threads of the compilation failure.
  if (promise) {
    promise->set_exception(
        std::make_exception_ptr(torch_neuronx::CompilationRuntimeException(full_error_msg)));
  }

  CleanupPendingCompilation(key);
}

void CompilationCache::CleanupPendingCompilation(const CompilationCacheKey& key) {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);
  pending_compilations_.erase(key.GetKey());
}

std::filesystem::path CompilationCache::GetLockPath(const std::string& key) const {
  // Lock files are stored in local storage to avoid NFS locking issues.
  // The key is the persistent_cache_key.
  return local_cache_dir_ / "locks" / (key + ".lock");
}

[[noreturn]] void CompilationCache::ThrowCompilationError(const CompilationCacheKey& key,
                                                          const std::string& message) {
  TORCH_NEURONX_ERROR("Compilation error", "CompilationCacheKey=", key.ToString(),
                      "error=", message);
  throw torch_neuronx::CompilationRuntimeException("Compilation failed for CompilationCacheKey '" +
                                                   key.ToString() + "': " + message);
}

}  // namespace at::neuron
