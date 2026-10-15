#include "CompilationCache.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>

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

namespace {

constexpr uint8_t kWrappedHloSidecarVersion = 1;

std::vector<uint8_t> EncodeWrappedHloSidecar(const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> encoded;
  encoded.reserve(payload.size() + 1);
  encoded.push_back(kWrappedHloSidecarVersion);
  encoded.insert(encoded.end(), payload.begin(), payload.end());
  return encoded;
}

std::optional<std::vector<uint8_t>> DecodeWrappedHloSidecar(const std::vector<uint8_t>& stored) {
  if (stored.empty() || stored.front() != kWrappedHloSidecarVersion) {
    return std::nullopt;  // corrupt/legacy sidecar -> treat as absent (recompile)
  }
  return std::vector<uint8_t>(stored.begin() + 1, stored.end());
}

}  // namespace

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
      wrapped_hlo_cache_ =
          std::make_unique<PersistentCacheBackend>(std::string(".wrapped_neff_hlo"));
    } catch (const std::exception& e) {
      TORCH_NEURONX_WARN(
          "Failed to initialize persistent cache backend, continuing without "
          "persistent caching",
          "error=", e.what());
      persistent_cache_ = nullptr;
      wrapped_hlo_cache_ = nullptr;
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

  // Always-local backend for WLO compiles, which the shared service cannot serve yet. Reuse the
  // default backend when it is already local to avoid a redundant instance.
  if (backend_->GetName() == "local") {
    local_backend_ = nullptr;  // backend_ is already local; CompileAndCache falls back to it.
  } else {
    local_backend_ = std::make_unique<LocalCompilationBackend>();
  }

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
  persistent_key_hints_.clear();

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

std::string CompilationCache::ResolvePersistentKeyLocked(const CompilationCacheKey& key) const {
  // A key with no IR cannot name its NEFF by content, so a recorded mapping beats the
  // key's own derivation. This matters when the content-key gate is off: the key then
  // derives hash(base_key || config), which is not where put_neff_cache() put the NEFF.
  // A key WITH IR keeps its content-addressed name, so a stale hint cannot serve the
  // wrong NEFF.
  if (!key.HasHloBytes()) {
    auto it = persistent_key_hints_.find(key.GetKey());
    if (it != persistent_key_hints_.end()) {
      return it->second;
    }
  }
  if (const std::string* pk = key.GetPersistentKey()) {
    return *pk;
  }
  return std::string();
}

void CompilationCache::RegisterPersistentKeyHint(const std::string& base_key,
                                                 const std::string& persistent_key) {
  std::unique_lock<std::shared_mutex> lock(cache_mutex_);
  RecordPersistentKeyHintLocked(base_key, persistent_key);
}

void CompilationCache::RecordPersistentKeyHintLocked(const std::string& base_key,
                                                     const std::string& persistent_key) {
  if (base_key.empty() || persistent_key.empty()) {
    return;
  }
  persistent_key_hints_[base_key] = persistent_key;
}

bool CompilationCache::CheckCacheAndCoordinateWork(
    const CompilationCacheKey& key, const CompilableKernelExecution& kernel_execution,
    uint32_t stream_id) {
  const std::string& cache_key = key.GetKey();
  std::string persistent_cache_key;
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
      cache_cv_.notify_all();
      return false;
    }
    persistent_cache_key = ResolvePersistentKeyLocked(key);
    if (persistent_cache_key.empty()) {
      TORCH_NEURONX_DEBUG("Skipping persistent cache (no HLO bytes and no key hint):",
                          key.ToString());
      auto compilation_promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
      pending_compilations_.emplace(cache_key, PendingCompilation(compilation_promise));
      cache_cv_.notify_all();
      return false;
    }
  }

  const bool requires_wrapped_hlo = kernel_execution.RequiresWrappedNeffHlo();

  // Release the cache lock before disk or NFS access.
  auto neff_bytes = persistent_cache_->Get(persistent_cache_key);
  std::optional<std::vector<uint8_t>> wrapped_hlo_bytes;
  if (neff_bytes && requires_wrapped_hlo && wrapped_hlo_cache_) {
    if (auto stored = wrapped_hlo_cache_->Get(persistent_cache_key)) {
      wrapped_hlo_bytes = DecodeWrappedHloSidecar(*stored);
    }
  }
  if (neff_bytes && (!requires_wrapped_hlo || wrapped_hlo_bytes)) {
    TORCH_NEURONX_DEBUG("Persistent cache hit:", key.ToString());

    // Create cache entry and store in memory cache.
    auto cache_entry = std::make_shared<CacheEntry>(std::move(*neff_bytes), persistent_cache_key,
                                                    std::chrono::milliseconds(0));
    if (wrapped_hlo_bytes) {
      cache_entry->wrapped_neff_hlo = std::move(*wrapped_hlo_bytes);
    }

    {
      std::unique_lock<std::shared_mutex> lock(cache_mutex_);
      // Keep a successful entry because a kernel can still refer to its NEFF bytes.
      auto [it, inserted] = cache_.try_emplace(cache_key, cache_entry);
      if (!inserted && !it->second->IsFailed()) {
        it->second->UpdateAccess();
        TORCH_NEURONX_COUNTER("CompilationCache.InMemoryHits", 1);
      } else {
        // A failed entry has no NEFF pointer. Replace it with the valid disk result.
        if (!inserted) it->second = cache_entry;
        TORCH_NEURONX_COUNTER("CompilationCache.PersistentHits", 1);
      }
      RecordPersistentKeyHintLocked(cache_key, persistent_cache_key);
      cache_cv_.notify_all();

      // Create managed pointer and set on kernel execution.
      auto neff_ptr = CreateNeffBytesPtr(it->second->neff_bytes, cache_key);
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
    cache_cv_.notify_all();
    return false;
  }
}

std::shared_ptr<CacheEntry> CompilationCache::GetOrCompileNeff(
    const CompilableKernelExecution& kernel_execution, uint32_t stream_id) {
  const auto& key = kernel_execution.GetCompilationCacheKey();

  // Check caches (memory + persistent) and coordinate work scheduling.
  if (CheckCacheAndCoordinateWork(key, kernel_execution, stream_id)) {
    if (kernel_execution.HasCachedNeff()) {
      return GetEntry(key.GetKey());  // Cache hit - NEFF already set; return the cached entry.
    }
    // Another thread is compiling - wait for completion and return its entry.
    return WaitForPendingCompilation(key, kernel_execution, stream_id);
  }

  // Cache miss (both memory and persistent) - this thread compiles and returns the new entry.
  return Compile(key, kernel_execution, stream_id);
}

void CompilationCache::GetOrCompileNeffAsync(
    const CompilableKernelExecution& kernel_execution, uint32_t stream_id,
    std::function<void(bool success, const std::string& error)> callback) {
  const auto& key = kernel_execution.GetCompilationCacheKey();

  // Check caches (memory + persistent) and coordinate work scheduling.
  if (CheckCacheAndCoordinateWork(key, kernel_execution, stream_id)) {
    if (kernel_execution.HasCachedNeff()) {
      // Cache hit — invoke callback synchronously.
      callback(true, "");
      return;
    }
    ContinuePendingCompilationAsync(kernel_execution, std::move(callback));
    return;
  }

  // Cache miss — submit compilation to thread pool.
  CompilationCacheKey key_copy = key;
  compilation_pool_->Submit(
      [this, key_copy, &kernel_execution, stream_id, callback = std::move(callback)]() {
        try {
          Compile(key_copy, kernel_execution, stream_id);
          callback(true, "");
        } catch (const std::exception& e) {
          callback(false, e.what());
        } catch (...) {
          callback(false, "Unknown internal compilation error");
        }
      });
}

void CompilationCache::ContinuePendingCompilationAsync(
    const CompilableKernelExecution& kernel_execution,
    std::function<void(bool success, const std::string& error)> callback) {
  const std::string& cache_key = kernel_execution.GetCompileKey();
  std::shared_future<std::shared_ptr<CacheEntry>> compilation_future;
  bool success = false;
  std::string error;
  {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    auto pending_it = pending_compilations_.find(cache_key);
    if (pending_it != pending_compilations_.end()) {
      compilation_future = pending_it->second.future;
    } else {
      // The compilation can finish before this request captures its future.
      auto it = cache_.find(cache_key);
      if (it != cache_.end() && it->second) {
        if (it->second->IsFailed()) {
          error = it->second->GetFailureMessage();
        } else {
          auto neff_ptr = CreateNeffBytesPtr(it->second->neff_bytes, cache_key);
          kernel_execution.SetCachedNeff(std::move(neff_ptr));
          success = true;
        }
      }
    }
  }

  if (!compilation_future.valid()) {
    // Release the cache lock before the callback calls other cache methods.
    callback(success, error);
    return;
  }

  // Submit the wait to the thread pool so the caller can continue.
  std::string cache_key_copy = cache_key;
  compilation_pool_->Submit([this, cache_key_copy, &kernel_execution,
                             compilation_future = std::move(compilation_future),
                             callback = std::move(callback)]() {
    try {
      auto cache_entry = compilation_future.get();
      auto neff_ptr = CreateNeffBytesPtr(cache_entry->neff_bytes, cache_key_copy);
      kernel_execution.SetCachedNeff(std::move(neff_ptr));
      callback(true, "");
    } catch (const std::exception& e) {
      callback(false, e.what());
    } catch (...) {
      callback(false, "Unknown error while waiting for pending compilation");
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
      total_memory += entry->wrapped_neff_hlo.size();
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

std::vector<uint8_t> CompilationCache::GetWrappedNeffHlo(const std::string& cache_key,
                                                         std::chrono::milliseconds timeout) {
  std::shared_future<std::shared_ptr<CacheEntry>> fut;
  {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    const bool ready = cache_cv_.wait_until(lock, deadline, [&] {
      return pending_compilations_.count(cache_key) > 0 || cache_.count(cache_key) > 0;
    });
    if (!ready) {
      throw std::runtime_error(
          "get_wrapped_neff_hlo: timed out waiting for the submitted compile of key '" + cache_key +
          "'");
    }
    auto it = pending_compilations_.find(cache_key);
    if (it != pending_compilations_.end()) {
      fut = it->second.future;
    }
  }
  if (fut.valid()) {
    if (auto entry = fut.get()) {  // blocks until the compile completes; rethrows on failure
      return entry->wrapped_neff_hlo;
    }
  }

  // Compile already completed (pending entry cleaned up) — read from the in-memory cache.
  if (auto entry = GetEntry(cache_key)) {
    return entry->wrapped_neff_hlo;
  }

  // Registered/cached then evicted before we could read it — fail closed rather than return an
  // empty result, which would run the relaid NEFF against untransformed weights.
  throw std::runtime_error("get_wrapped_neff_hlo: no cached entry for key '" + cache_key +
                           "' after wait");
}

at::neuron::NeffBytesPtr CompilationCache::CreateNeffBytesPtr(
    const std::vector<uint8_t>& neff_bytes, const std::string& cache_key) {
  // TODO: Retain CacheEntry ownership in each NEFF handle to keep bytes alive during eviction.
  // Protect entries before activation and count active users per key.
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

std::shared_ptr<CacheEntry> CompilationCache::WaitForPendingCompilation(
    const CompilationCacheKey& key, const CompilableKernelExecution& kernel_execution,
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
    return cache_entry;
  } catch (const std::exception& e) {
    ThrowCompilationError(key, "Pending compilation failed: " + std::string(e.what()));
  }
}

std::shared_ptr<CacheEntry> CompilationCache::Compile(
    const CompilationCacheKey& key, const CompilableKernelExecution& kernel_execution,
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

    return compilation_result;
  } catch (const std::exception& e) {
    HandleCompilationFailure(key, compilation_promise, e.what());
    ThrowCompilationError(key, e.what());
  }
}

std::shared_ptr<CacheEntry> CompilationCache::CompileAndCache(
    const CompilableKernelExecution& kernel_execution, const CompilationCacheKey& key) {
  const std::string& cache_key = key.GetKey();
  // Get persistent key for the file lock and cache storage. When it is empty we compile
  // without persistent coordination: no cross-process file lock, no persistent put.
  std::string persistent_cache_key;
  {
    std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
    persistent_cache_key = ResolvePersistentKeyLocked(key);
  }

  // Optional file lock: only when we have a persistent key to coordinate on.
  std::optional<FileLockGuard> lock_guard;
  if (!persistent_cache_key.empty()) {
    std::filesystem::path lock_path = GetLockPath(persistent_cache_key);
    TORCH_NEURONX_DEBUG("Acquiring compilation lock:", "CompilationCacheKey=", key.ToString(),
                        "lock_path=", lock_path.string());
    lock_guard.emplace(lock_path);
    TORCH_NEURONX_DEBUG("Compilation lock acquired:", key.ToString());
  }

  const bool requires_wrapped_hlo = kernel_execution.RequiresWrappedNeffHlo();

  // Double-check persistent cache after acquiring lock.
  // Another process may have completed compilation while waiting for lock.
  if (persistent_cache_ && !persistent_cache_key.empty()) {
    auto neff_bytes = persistent_cache_->Get(persistent_cache_key);
    std::optional<std::vector<uint8_t>> wrapped_hlo_bytes;
    if (neff_bytes && requires_wrapped_hlo && wrapped_hlo_cache_) {
      if (auto stored = wrapped_hlo_cache_->Get(persistent_cache_key)) {
        wrapped_hlo_bytes = DecodeWrappedHloSidecar(*stored);
      }
    }
    // Persistent cache hit - entry populated while waiting for compilation lock.
    if (neff_bytes && (!requires_wrapped_hlo || wrapped_hlo_bytes)) {
      TORCH_NEURONX_DEBUG("Persistent cache hit after lock acquisition:", key.ToString());

      // Record persistent cache hit.
      TORCH_NEURONX_COUNTER("CompilationCache.PersistentHits", 1);

      // Create cache entry from persistent cache hit.
      auto cache_entry = std::make_shared<CacheEntry>(std::move(*neff_bytes), persistent_cache_key,
                                                      std::chrono::milliseconds(0));
      if (wrapped_hlo_bytes) {
        cache_entry->wrapped_neff_hlo = std::move(*wrapped_hlo_bytes);
      }

      // Store in memory cache using original cache_key.
      {
        std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
        auto [it, inserted] = cache_.try_emplace(cache_key, cache_entry);
        if (!inserted) cache_entry = it->second;
        RecordPersistentKeyHintLocked(cache_key, persistent_cache_key);
        cache_cv_.notify_all();
      }
      if (cache_entry->IsFailed()) {
        ThrowCompilationError(key, cache_entry->GetFailureMessage());
      }

      // Check if eviction is needed after adding new entry.
      CheckAndEvictIfNeeded();

      return cache_entry;
    }
  }

  auto start_time = std::chrono::steady_clock::now();

  // Dump IR details if requested (before compilation)
  std::string ir_type = "StableHLO";
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
  // Attempt to get op_name from NeuronKernelExecution if available.
  std::string op_name = "unknown_op";
  if (auto* nke = dynamic_cast<const NeuronKernelExecution*>(&kernel_execution)) {
    op_name = nke->GetOpName();
  }

  // WLO compiles must run on the local backend: they produce a wrapped_neff.hlo sidecar that the
  // shared compile service protocol cannot yet return (TODO(wlo-service)). local_backend_ is a
  // LocalCompilationBackend kept for exactly this case; backend_ is the configured default.
  CompilationBackend* compile_backend = backend_.get();
  if (requires_wrapped_hlo && IsServiceBackend()) {
    compile_backend = local_backend_.get();
  }

  auto compile_result = compile_backend->Compile(
      kernel_execution.GetHloBytes(), ir_type, op_name, cache_key,
      kernel_execution.GetOptimizationLevel(), kernel_execution.GetAdditionalArgs(),
      [&kernel_execution]() -> CompileResult { return kernel_execution.CompileToNeffResult(); });

  auto end_time = std::chrono::steady_clock::now();
  auto compilation_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Update centralized metrics
  TORCH_NEURONX_COUNTER("CompilationCache.TotalCompilations", 1);
  TORCH_NEURONX_BYTES_METRIC("CompilationCache.NeffSize", compile_result.neff_bytes.size());
  TORCH_NEURONX_TIME_METRIC("CompilationCache.CompilationTime", start_time, end_time);

  // Create cache entry with compiled NEFF and timing information.
  // Use persistent key when available, fall back to in-memory cache_key
  // for entries that can't be persistently cached (no HLO bytes).
  const std::string& entry_key = !persistent_cache_key.empty() ? persistent_cache_key : cache_key;
  auto cache_entry = std::make_shared<CacheEntry>(std::move(compile_result.neff_bytes), entry_key,
                                                  compilation_time);
  cache_entry->wrapped_neff_hlo = std::move(compile_result.wrapped_neff_hlo);

  // Store the cache entry in memory using original cache_key.
  {
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    auto [it, inserted] = cache_.try_emplace(cache_key, cache_entry);
    if (!inserted) cache_entry = it->second;
    RecordPersistentKeyHintLocked(cache_key, persistent_cache_key);
  }
  if (cache_entry->IsFailed()) {
    ThrowCompilationError(key, cache_entry->GetFailureMessage());
  }

  // Store to persistent cache (NFS/disk) for cross-process sharing.
  // Skip when no persistent key (no HLO bytes available).
  if (persistent_cache_ && !persistent_cache_key.empty()) {
    if (persistent_cache_->Put(persistent_cache_key, cache_entry->neff_bytes)) {
      TORCH_NEURONX_DEBUG("Stored to persistent cache", key.ToString());
    }
    // Always persist a version-tagged sidecar for WLO (even zero transforms) so a cross-process
    // WLO hit can tell "zero transforms" (valid) from "sidecar missing" (recompile).
    if (wrapped_hlo_cache_ && requires_wrapped_hlo) {
      wrapped_hlo_cache_->Put(persistent_cache_key,
                              EncodeWrappedHloSidecar(cache_entry->wrapped_neff_hlo));
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
    cache_.try_emplace(cache_key, CacheEntry::Failed(cache_key, full_error_msg));
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
