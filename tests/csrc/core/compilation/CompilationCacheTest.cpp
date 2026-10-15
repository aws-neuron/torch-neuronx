#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

using namespace at::neuron;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace at::neuron {

class CompilationCacheTestPeer {
 public:
  static auto SeedPending(CompilationCache& cache, const CompilationCacheKey& key) {
    auto promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
    std::unique_lock<std::shared_mutex> lock(cache.cache_mutex_);
    CompilationCache::PendingCompilation pending(promise);
    auto future = pending.future;
    cache.pending_compilations_.emplace(key.GetKey(), std::move(pending));
    return std::make_pair(promise, future);
  }

  static void FailPending(CompilationCache& cache, const CompilationCacheKey& key,
                          std::shared_ptr<std::promise<std::shared_ptr<CacheEntry>>> promise) {
    cache.HandleCompilationFailure(key, std::move(promise), "test compilation failed");
  }

  static std::shared_future<std::shared_ptr<CacheEntry>> SeedFailedPending(
      CompilationCache& cache, const CompilationCacheKey& key) {
    const std::string error = "test compilation failed";
    auto promise = std::make_shared<std::promise<std::shared_ptr<CacheEntry>>>();
    std::unique_lock<std::shared_mutex> lock(cache.cache_mutex_);
    cache.cache_.emplace(key.GetKey(), CacheEntry::Failed(key.GetKey(), error));
    CompilationCache::PendingCompilation pending(promise);
    auto future = pending.future;
    cache.pending_compilations_.emplace(key.GetKey(), std::move(pending));
    promise->set_exception(
        std::make_exception_ptr(torch_neuronx::CompilationRuntimeException(error)));
    return future;
  }

  static void CleanupPending(CompilationCache& cache, const CompilationCacheKey& key) {
    cache.CleanupPendingCompilation(key);
  }
};

}  // namespace at::neuron

class MockNeuronKernelExecution : public CompilableKernelExecution {
 public:
  // Static counter to track total compilations across all instances
  static std::atomic<int>& GetCompileCounter() {
    static std::atomic<int> compile_counter{0};
    return compile_counter;
  }

  static void ResetCompileCounter() { GetCompileCounter().store(0); }

  MockNeuronKernelExecution(const std::string& cache_key, bool wlo_enabled = false)
      // WLO identity rides on the "|wlo" suffix of the base key, which CompilationCacheKey infers.
      : CompilableKernelExecution(CompilationCacheKey(wlo_enabled ? cache_key + "|wlo" : cache_key,
                                                      "", "", nullptr, CacheKeySource::UNKNOWN),
                                  "", "", false),
        hlo_bytes_({'H', 'L', 'O', '_'}) {
    hlo_bytes_.insert(hlo_bytes_.end(), cache_key.begin(), cache_key.end());
    GetCompilationCacheKey().SetHloBytesPtr(&hlo_bytes_);
  }

  MockNeuronKernelExecution(const std::string& cache_key, const std::vector<uint8_t>& hlo_bytes)
      : CompilableKernelExecution(CompilationCacheKey(cache_key, "", ""), "", "", false),
        hlo_bytes_(hlo_bytes) {
    GetCompilationCacheKey().SetHloBytesPtr(&hlo_bytes_);
  }

  // Mock compilation that returns dummy NEFF data
  std::vector<uint8_t> CompileToNeff() const override {
    // Increment compile counter to track when actual compilation occurs
    GetCompileCounter()++;
    // Simulate compilation time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Return dummy NEFF data based on cache key for uniqueness
    std::vector<uint8_t> neff_data;
    std::string neff_content = "NEFF_" + GetCompileKey() + "_" + GetAdditionalArgs();
    neff_data.assign(neff_content.begin(), neff_content.end());
    return neff_data;
  }

  // Return HLO bytes for persistent cache key computation
  const std::vector<uint8_t>& GetHloBytes() const override { return hlo_bytes_; }

 private:
  std::vector<uint8_t> hlo_bytes_;
};

// Helper: generates the full cache key string from a base key.

class CompilationCacheTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  void SetUp() override {
    // Mock NRT instance info for CompilationCacheKey construction
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    // Enable metrics for testing
    at::neuron::metrics::SetMetricsEnabled(true);

    // Save and clear cache env vars to ensure clean test state
    SaveEnvVar("TORCH_NEURONX_NEFF_CACHE_DIR");
    SaveEnvVar("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR");
    SaveEnvVar("TORCH_NEURONX_NEFF_DISABLE_CACHE");
    SaveEnvVar("NEURON_COMPILE_SERVICE");
    SaveEnvVar("TORCH_NEURONX_NEFF_KEY_FROM_HLO");

    // Force local backend for unit tests (service backend calls real neuronx-cc)
    setenv("NEURON_COMPILE_SERVICE", "0", 1);

    // Pin the HLO-content keying flag on for these tests. The graceful-skip
    // and cross-commit-reuse cases below assert the new contract; the legacy
    // (flag-off) path is exercised in CompilationCacheKeyLegacyKeyTest.
    setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1", 1);

    // Create temp directories for persistent and local cache
    temp_dir_ = std::make_unique<TempDirectory>("compilation_cache_test_");
    local_cache_dir_ = std::make_unique<TempDirectory>("compilation_cache_local_test_");

    // Enable caching with temp directory
    setenv("TORCH_NEURONX_NEFF_CACHE_DIR", temp_dir_->path().c_str(), 1);
    setenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", local_cache_dir_->path().c_str(), 1);
    unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");

    cache_ = std::make_unique<CompilationCache>();
    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override {
    cache_->Clear();
    cache_.reset();
    temp_dir_.reset();
    local_cache_dir_.reset();
    RestoreEnvVars();
  }

  void SaveEnvVar(const char* name) {
    const char* value = std::getenv(name);
    if (value) {
      saved_env_vars_[name] = value;
    }
  }

  void RestoreEnvVars() {
    for (const auto& [name, value] : saved_env_vars_) {
      setenv(name.c_str(), value.c_str(), 1);
    }
    // Unset any vars that weren't originally set. Every name that SetUp() saves must
    // appear here, or SetUp()'s value leaks into a later suite in the same binary.
    for (const char* name : {"TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
                             "TORCH_NEURONX_NEFF_DISABLE_CACHE", "NEURON_COMPILE_SERVICE",
                             "TORCH_NEURONX_NEFF_KEY_FROM_HLO"}) {
      if (saved_env_vars_.find(name) == saved_env_vars_.end()) {
        unsetenv(name);
      }
    }
  }

  std::unique_ptr<TempDirectory> temp_dir_;
  std::unique_ptr<TempDirectory> local_cache_dir_;
  std::unique_ptr<CompilationCache> cache_;
  std::map<std::string, std::string> saved_env_vars_;
};

// Test basic cache functionality
TEST_F(CompilationCacheTest, BasicCacheHitMiss) {
  MockNeuronKernelExecution kernel_exec("test_key_1");
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;
  // Compile and store in cache - this should handle the cache coordination internally
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  EXPECT_TRUE(kernel_exec.HasCachedNeff());
  EXPECT_FALSE(kernel_exec.GetCachedNeff().empty());

  // Second call with same key should be a cache hit
  MockNeuronKernelExecution kernel_exec2("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  EXPECT_TRUE(kernel_exec2.HasCachedNeff());

  auto cached_neff = kernel_exec2.GetCachedNeff();
  auto original_neff = kernel_exec.GetCachedNeff();
  EXPECT_EQ(original_neff, cached_neff);
}

TEST_F(CompilationCacheTest, DifferentKeysProduceDifferentResults) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Compile with first key
  MockNeuronKernelExecution kernel_exec1("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);

  // Compile with second key
  MockNeuronKernelExecution kernel_exec2("test_key_2");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  // Results should be different
  EXPECT_NE(kernel_exec1.GetCachedNeff(), kernel_exec2.GetCachedNeff());
}
// Test basic cache operations
TEST_F(CompilationCacheTest, CacheBasicOperations) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Initial state - cache should be empty
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 0);
  EXPECT_EQ(cache_->GetMemoryUsageBytes(), 0);

  // Add one entry
  MockNeuronKernelExecution kernel_exec1("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);

  keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
  EXPECT_GT(cache_->GetMemoryUsageBytes(), 0);

  // Cache hit - same key should return cached result
  MockNeuronKernelExecution kernel_exec2("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  // Should still have only one entry
  keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(CompilationCacheTest, CacheEntryInfo) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  MockNeuronKernelExecution kernel_exec("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  // Call again to trigger a cache hit and increment access_count
  MockNeuronKernelExecution kernel_exec2("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  auto entries = cache_->GetCacheEntries();
  EXPECT_EQ(entries.size(), 1);

  const auto& entry = entries[0];
  EXPECT_EQ(entry.cache_key, "test_key_1");
  EXPECT_GT(entry.neff_size_bytes, 0);
  EXPECT_EQ(entry.access_count, 1);  // Should be 1 after the cache hit
  EXPECT_GT(entry.compilation_time.count(), 0);
}

TEST_F(CompilationCacheTest, ConcurrentAccess) {
  const int num_threads = 4;
  const std::string cache_key = "concurrent_test_key";
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};

  std::vector<std::future<std::vector<uint8_t>>> futures;
  std::atomic<int> compilation_count{0};

  // Launch multiple threads trying to compile the same key
  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      MockNeuronKernelExecution kernel_exec(cache_key);
      uint32_t stream_id = i + 1;

      // Use the proper API - GetOrCompileNeff handles coordination internally
      cache_->GetOrCompileNeff(kernel_exec, stream_id);

      return kernel_exec.GetCachedNeff();
    }));
  }

  // Wait for all threads and collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // Note: We can't easily count compilations with the simplified API,
  // but we can verify that all results are identical (indicating proper caching)

  // All results should be identical
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i]);
  }

  // Cache should have one entry
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

class LimitedCompilationCacheTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  void SetUpWithLimits(size_t max_entries, size_t max_size_mb) {
    // Mock NRT instance info for CompilationCacheKey construction
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    // Save original environment variables
    SaveEnvVar("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
    SaveEnvVar("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
    SaveEnvVar("TORCH_NEURONX_NEFF_DISABLE_CACHE");
    SaveEnvVar("NEURON_COMPILE_SERVICE");

    // Disable persistent caching for in-memory cache tests
    setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true", 1);

    // Force the local backend. The mock kernels return dummy NEFF bytes, so the
    // service backend would send the dummy IR to the real neuronx-cc and fail.
    setenv("NEURON_COMPILE_SERVICE", "0", 1);

    // Set test-specific limits
    setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", std::to_string(max_entries).c_str(), 1);
    setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", std::to_string(max_size_mb).c_str(), 1);

    // Create cache with new limits
    cache_ = std::make_unique<CompilationCache>();
    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override {
    cache_.reset();
    RestoreEnvVars();
  }

  void SaveEnvVar(const char* name) {
    const char* value = std::getenv(name);
    if (value) {
      saved_env_vars_[name] = value;
    }
  }

  void RestoreEnvVars() {
    for (const auto& [name, value] : saved_env_vars_) {
      setenv(name.c_str(), value.c_str(), 1);
    }
    // Unset any vars that weren't originally set. Every name that SetUpWithLimits()
    // saves must appear here, or its value leaks into a later suite in the same binary.
    for (const char* name :
         {"NEURON_COMPILATION_CACHE_MAX_SIZE_MB", "NEURON_COMPILATION_CACHE_MAX_ENTRIES",
          "TORCH_NEURONX_NEFF_DISABLE_CACHE", "NEURON_COMPILE_SERVICE"}) {
      if (saved_env_vars_.find(name) == saved_env_vars_.end()) {
        unsetenv(name);
      }
    }
  }

  std::unique_ptr<CompilationCache> cache_;
  std::map<std::string, std::string> saved_env_vars_;
};

TEST_F(LimitedCompilationCacheTest, SmallCacheEntriesEviction) {
  SetUpWithLimits(2, 1024);  // 2 entries max, 1GB size

  uint32_t stream_id = 1;

  // Add entries beyond the limit to trigger eviction
  std::vector<std::string> keys;
  for (int i = 0; i < 5; ++i) {
    std::string key = "eviction_test_key_" + std::to_string(i);
    keys.push_back(key);
    {
      MockNeuronKernelExecution kernel_exec(key);
      cache_->GetOrCompileNeff(kernel_exec, stream_id);
    }

    // Small delay to ensure different last_used_time
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Verify that the most recently used entries are kept
  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_LE(all_keys.size(), 2);  // Should not exceed limit
}

TEST_F(LimitedCompilationCacheTest, SmallCacheSizeEviction) {
  // Set very small memory limit to force eviction
  SetUpWithLimits(1000, 1);  // 1000 entries max, 1MB size limit

  uint32_t stream_id = 1;

  // Create entries with large NEFF data to exceed memory limit
  std::vector<std::string> keys;
  for (int i = 0; i < 12; ++i) {  // Increased from 10 to 12 to ensure we exceed 1MB
    std::string key = "memory_eviction_test_" + std::to_string(i);
    keys.push_back(key);

    // Create mock with large NEFF data (100KB each)
    class LargeNeffMockKernelExecution : public MockNeuronKernelExecution {
     public:
      LargeNeffMockKernelExecution(const std::string& key) : MockNeuronKernelExecution(key) {}

      std::vector<uint8_t> CompileToNeff() const override {
        // Return 100KB of data
        return std::vector<uint8_t>(100 * 1024, 0x42);
      }
    };

    {
      LargeNeffMockKernelExecution kernel_exec(key);
      cache_->GetOrCompileNeff(kernel_exec, stream_id);
      // kernel_exec goes out of scope here, releasing the NeffBytesPtr
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Should be under 1MB limit
  EXPECT_LT(cache_->GetMemoryUsageBytes(), 1024 * 1024);
}

TEST_F(LimitedCompilationCacheTest, LRUEvictionOrder) {
  SetUpWithLimits(3, 1024);  // 3 entries max, 1GB size

  uint32_t stream_id = 1;

  // Add entries with specific access patterns
  std::vector<std::string> keys = {"oldest", "middle", "newest", "trigger_eviction"};

  // Create first three entries
  for (int i = 0; i < 3; ++i) {
    {
      MockNeuronKernelExecution kernel_exec(keys[i]);
      cache_->GetOrCompileNeff(kernel_exec, stream_id);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Access "middle" entry to update its last_used_time
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  {
    MockNeuronKernelExecution middle_exec("middle");
    cache_->GetOrCompileNeff(middle_exec, stream_id);
  }

  // Add fourth entry to trigger eviction
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  {
    MockNeuronKernelExecution trigger_exec("trigger_eviction");
    cache_->GetOrCompileNeff(trigger_exec, stream_id);
  }

  // Verify that "oldest" was evicted (LRU), but "middle" and "newest" remain
  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(all_keys.size(), 3);
  EXPECT_EQ(all_keys.count("oldest"), 0);            // Should be evicted
  EXPECT_EQ(all_keys.count("middle"), 1);            // Should remain (recently accessed)
  EXPECT_EQ(all_keys.count("newest"), 1);            // Should remain (recently created)
  EXPECT_EQ(all_keys.count("trigger_eviction"), 1);  // Should remain (just added)
}

TEST_F(LimitedCompilationCacheTest, EvictionWithActiveEntries) {
  SetUpWithLimits(2, 1024);  // 2 entries max, 1GB size

  uint32_t stream_id = 1;

  // Create first entry and keep it active
  MockNeuronKernelExecution kernel_exec1("active_entry");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  auto active_neff = kernel_exec1.GetCachedNeff();  // Keep reference active

  // Create second entry
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  {
    MockNeuronKernelExecution kernel_exec2("second_entry");
    cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  }

  // Try to add third entry - should evict "second_entry" but not "active_entry"
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  {
    MockNeuronKernelExecution kernel_exec3("third_entry");
    cache_->GetOrCompileNeff(kernel_exec3, stream_id);
  }

  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(all_keys.size(), 2);
  EXPECT_EQ(all_keys.count("active_entry"), 1);  // Should remain (active)
  EXPECT_EQ(all_keys.count("second_entry"), 0);  // Should be evicted
  EXPECT_EQ(all_keys.count("third_entry"), 1);   // Should remain (just added)
}

TEST_F(LimitedCompilationCacheTest, NoEvictionWhenUnderLimits) {
  SetUpWithLimits(10, 1024);  // 10 entries max, 1GB size

  uint32_t stream_id = 1;

  // Add fewer entries than the limit
  for (int i = 0; i < 5; ++i) {
    MockNeuronKernelExecution kernel_exec("no_eviction_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(all_keys.size(), 5);  // All entries should be present
}

TEST_F(CompilationCacheTest, ManualEviction) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Add multiple entries
  for (int i = 0; i < 5; ++i) {
    MockNeuronKernelExecution kernel_exec("test_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto keys_before = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_before.size(), 5);

  // Manually evict to 2 entries
  cache_->EvictLruEntries(2);

  auto keys_after = cache_->GetAllCacheKeys();
  EXPECT_LE(keys_after.size(), 2);
}

TEST_F(CompilationCacheTest, ClearCache) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Add some entries
  for (int i = 0; i < 3; ++i) {
    MockNeuronKernelExecution kernel_exec("test_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto keys_before = cache_->GetAllCacheKeys();
  EXPECT_GT(keys_before.size(), 0);

  // Clear cache
  cache_->Clear();

  auto keys_after = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_after.size(), 0);
  EXPECT_EQ(cache_->GetMemoryUsageBytes(), 0);
}

TEST_F(CompilationCacheTest, GetAllCacheKeys) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  std::vector<std::string> expected_keys = {"key1", "key2", "key3"};

  // Add entries
  for (const auto& key : expected_keys) {
    MockNeuronKernelExecution kernel_exec(key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto cache_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(cache_keys.size(), expected_keys.size());

  for (const auto& key : expected_keys) {
    EXPECT_TRUE(cache_keys.find(key) != cache_keys.end());
  }
}

TEST_F(CompilationCacheTest, MemoryUsageTracking) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  size_t initial_memory = cache_->GetMemoryUsageBytes();
  EXPECT_EQ(initial_memory, 0);

  // Add an entry
  MockNeuronKernelExecution kernel_exec("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  size_t memory_after_add = cache_->GetMemoryUsageBytes();
  EXPECT_GT(memory_after_add, initial_memory);

  // Clear cache
  cache_->Clear();

  size_t memory_after_clear = cache_->GetMemoryUsageBytes();
  EXPECT_EQ(memory_after_clear, 0);
}

TEST_F(CompilationCacheTest, AdditionalArgsInCompilation) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  MockNeuronKernelExecution kernel_exec("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  // The mock implementation includes additional_args in the result
  auto neff_bytes = kernel_exec.GetCachedNeff();
  std::string result_str(neff_bytes.begin(), neff_bytes.end());
  EXPECT_FALSE(result_str.empty());
}

TEST_F(CompilationCacheTest, ErrorHandlingInCompilation) {
  // Create a mock that will throw during compilation
  class ThrowingMockKernelExecution : public MockNeuronKernelExecution {
   public:
    ThrowingMockKernelExecution(const std::string& cache_key)
        : MockNeuronKernelExecution(cache_key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("Compilation failed");
    }
  };

  ThrowingMockKernelExecution kernel_exec("failing_key");
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  EXPECT_THROW({ cache_->GetOrCompileNeff(kernel_exec, stream_id); }, std::runtime_error);
}

TEST_F(CompilationCacheTest, DifferentStreamsShareCache) {
  const std::string cache_key = "shared_key";

  // Compile on stream 1
  MockNeuronKernelExecution kernel_exec1(cache_key);
  uint32_t stream_id1 = 1;
  cache_->GetOrCompileNeff(kernel_exec1, stream_id1);
  auto neff_bytes1 = kernel_exec1.GetCachedNeff();

  // Access same key from stream 2 - should be cache hit
  MockNeuronKernelExecution kernel_exec2(cache_key);
  uint32_t stream_id2 = 2;
  cache_->GetOrCompileNeff(kernel_exec2, stream_id2);

  EXPECT_TRUE(kernel_exec2.HasCachedNeff());
  auto cached_neff = kernel_exec2.GetCachedNeff();
  EXPECT_EQ(neff_bytes1, cached_neff);

  // Should still have only one entry
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(CompilationCacheTest, MultipleStreamsCompilingSameKey) {
  const int num_streams = 8;
  const std::string cache_key = "multi_stream_key";

  std::vector<std::future<std::vector<uint8_t>>> futures;

  // Launch multiple streams trying to compile the same key
  for (int i = 0; i < num_streams; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      MockNeuronKernelExecution kernel_exec(cache_key);
      uint32_t stream_id = i + 10;  // Use different stream IDs

      cache_->GetOrCompileNeff(kernel_exec, stream_id);
      return kernel_exec.GetCachedNeff();
    }));
  }

  // Wait for all streams and collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should be identical regardless of stream
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i]);
  }

  // Should only have one cache entry
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

// Cache Key Edge Cases
TEST_F(CompilationCacheTest, LongCacheKeys) {
  uint32_t stream_id = 1;

  // Create long cache key (1000 characters)
  std::string long_key(1000, 'a');
  long_key += "_unique_suffix";

  MockNeuronKernelExecution kernel_exec(long_key);
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  EXPECT_TRUE(kernel_exec.HasCachedNeff());
  EXPECT_FALSE(kernel_exec.GetCachedNeff().empty());

  // Verify cache hit with same long key
  MockNeuronKernelExecution kernel_exec2(long_key);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());
  EXPECT_EQ(kernel_exec.GetCachedNeff(), kernel_exec2.GetCachedNeff());
}

TEST_F(CompilationCacheTest, SpecialCharactersInCacheKeys) {
  uint32_t stream_id = 1;

  // Test various special characters that might appear in cache keys
  std::vector<std::string> special_keys = {
      "key_with_underscores_and_123", "key-with-dashes-and-456",  "key.with.dots.and.789",
      "key:with:colons:and:abc",      "key/with/slashes/and/def", "key with spaces and ghi",
      "key\twith\ttabs\tand\tjkl"};

  std::vector<std::vector<uint8_t>> results;

  for (const auto& key : special_keys) {
    MockNeuronKernelExecution kernel_exec(key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    auto neff_bytes = kernel_exec.GetCachedNeff();
    results.push_back(neff_bytes);

    // Verify each key produces a result
    EXPECT_FALSE(neff_bytes.empty()) << "Failed for key: " << key;
  }

  // Verify all results are different (no key collisions)
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = i + 1; j < results.size(); ++j) {
      EXPECT_NE(results[i], results[j])
          << "Key collision between: " << special_keys[i] << " and " << special_keys[j];
    }
  }
}

TEST_F(CompilationCacheTest, EmptyAndMinimalCacheKeys) {
  uint32_t stream_id = 1;

  // Test minimal cache key (single char)
  MockNeuronKernelExecution kernel_exec_a("a");
  cache_->GetOrCompileNeff(kernel_exec_a, stream_id);
  EXPECT_TRUE(kernel_exec_a.HasCachedNeff());
  EXPECT_FALSE(kernel_exec_a.GetCachedNeff().empty());

  // Verify cache hit with same key
  MockNeuronKernelExecution kernel_exec_a2("a");
  cache_->GetOrCompileNeff(kernel_exec_a2, stream_id);
  EXPECT_TRUE(kernel_exec_a2.HasCachedNeff());
  EXPECT_EQ(kernel_exec_a.GetCachedNeff(), kernel_exec_a2.GetCachedNeff());

  // Different key should produce different NEFF
  MockNeuronKernelExecution kernel_exec_b("b");
  cache_->GetOrCompileNeff(kernel_exec_b, stream_id);
  EXPECT_NE(kernel_exec_a.GetCachedNeff(), kernel_exec_b.GetCachedNeff());
}

TEST_F(CompilationCacheTest, CacheStateAfterCompilationError) {
  uint32_t stream_id = 1;

  // Create a mock that will throw during compilation
  class ThrowingMockKernelExecution : public MockNeuronKernelExecution {
   public:
    ThrowingMockKernelExecution(const std::string& cache_key)
        : MockNeuronKernelExecution(cache_key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("Compilation failed");
    }
  };

  // Add a successful compilation first
  MockNeuronKernelExecution successful_kernel("successful_key");
  cache_->GetOrCompileNeff(successful_kernel, stream_id);

  auto keys_before = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_before.size(), 1);

  // Try to compile a failing kernel
  ThrowingMockKernelExecution failing_kernel("failing_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(failing_kernel, stream_id), std::runtime_error);

  // Cache should still be in a valid state — failed entry is now cached too
  auto keys_after = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_after.size(), 2);  // successful + failed entry

  // Should be able to access the successful entry
  MockNeuronKernelExecution successful_kernel2("successful_key");
  cache_->GetOrCompileNeff(successful_kernel2, stream_id);
  EXPECT_TRUE(successful_kernel2.HasCachedNeff());

  // Should be able to add new entries after error
  MockNeuronKernelExecution new_kernel("new_key");
  cache_->GetOrCompileNeff(new_kernel, stream_id);
  EXPECT_TRUE(new_kernel.HasCachedNeff());

  auto final_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(final_keys.size(), 3);  // successful + failed + new
}

TEST_F(LimitedCompilationCacheTest, CacheConsistencyAfterEviction) {
  SetUpWithLimits(3, 1024);  // 3 entries max, 1GB size
  uint32_t stream_id = 1;

  // Add entries to fill cache
  std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
  std::vector<std::vector<uint8_t>> expected_results;

  for (const auto& key : keys) {
    MockNeuronKernelExecution kernel_exec(key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    expected_results.push_back(kernel_exec.GetCachedNeff());
  }

  // Cache should have evicted some entries
  auto cache_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(cache_keys.size(), 3);

  // Verify cache consistency - entries that are still cached should return correct data
  for (const auto& cached_key : cache_keys) {
    MockNeuronKernelExecution kernel_exec(cached_key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    EXPECT_TRUE(kernel_exec.HasCachedNeff());

    // Find expected result for this key
    auto key_it = std::find(keys.begin(), keys.end(), cached_key);
    ASSERT_NE(key_it, keys.end());
    size_t key_index = std::distance(keys.begin(), key_it);
    EXPECT_EQ(kernel_exec.GetCachedNeff(), expected_results[key_index]);
  }
}

TEST_F(CompilationCacheTest, SlowCompilationHandling) {
  uint32_t stream_id = 1;

  // Create a mock that takes measurable time to compile
  class SlowMockKernelExecution : public MockNeuronKernelExecution {
   public:
    SlowMockKernelExecution(const std::string& cache_key) : MockNeuronKernelExecution(cache_key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      // Simulate longer compilation time
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return MockNeuronKernelExecution::CompileToNeff();
    }
  };

  // Compile multiple entries
  for (int i = 0; i < 3; ++i) {
    SlowMockKernelExecution kernel_exec("slow_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  // Verify all entries were cached
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 3);
}

TEST_F(LimitedCompilationCacheTest, MemoryUsageTracking) {
  SetUpWithLimits(1000, 1);  // 1000 entries max, 1MB limit
  uint32_t stream_id = 1;

  // Add some entries
  MockNeuronKernelExecution kernel_exec("memory_test_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  // Verify memory usage is tracked
  EXPECT_GT(cache_->GetMemoryUsageBytes(), 0);
  EXPECT_EQ(cache_->GetMaxCacheSizeBytes(), 1024 * 1024);  // 1MB
}

TEST_F(LimitedCompilationCacheTest, ConfigurationLimitsReflection) {
  SetUpWithLimits(100, 10);  // 100 entries max, 10MB limit

  EXPECT_EQ(cache_->GetMaxCacheSizeBytes(), 10 * 1024 * 1024);
  EXPECT_EQ(cache_->GetMaxCacheEntries(), 100);
}

TEST_F(LimitedCompilationCacheTest, ZeroCacheLimits) {
  SetUpWithLimits(0, 0);  // Zero limits (unlimited)

  uint32_t stream_id = 1;

  // Should still be able to compile (limits of 0 mean unlimited)
  MockNeuronKernelExecution kernel_exec("zero_limit_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());

  EXPECT_EQ(cache_->GetMaxCacheSizeBytes(), 0);
  EXPECT_EQ(cache_->GetMaxCacheEntries(), 0);
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(LimitedCompilationCacheTest, VeryLargeCacheLimits) {
  // Test behavior with very large limits
  size_t large_mb = 1024 * 1024;   // 1TB
  size_t large_entries = 1000000;  // 1M entries

  SetUpWithLimits(large_entries, large_mb);

  EXPECT_EQ(cache_->GetMaxCacheSizeBytes(), large_mb * 1024 * 1024);
  EXPECT_EQ(cache_->GetMaxCacheEntries(), large_entries);

  // Should still function normally
  uint32_t stream_id = 1;
  MockNeuronKernelExecution kernel_exec("large_limit_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());
}

TEST_F(CompilationCacheTest, SmartPointerLifecycle) {
  uint32_t stream_id = 1;

  // Create entry and get smart pointer
  MockNeuronKernelExecution kernel_exec("smart_ptr_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());

  // Get the cached NEFF (this creates a smart pointer)
  auto neff_bytes = kernel_exec.GetCachedNeff();
  EXPECT_FALSE(neff_bytes.empty());

  // The entry should be marked as active while smart pointer exists
  // Try to manually evict entries (should not evict active ones)
  cache_->EvictLruEntries(0);  // Try to evict all entries

  // Original entry should still exist because it's active
  MockNeuronKernelExecution kernel_exec3("smart_ptr_key");
  cache_->GetOrCompileNeff(kernel_exec3, stream_id);
  EXPECT_TRUE(kernel_exec3.HasCachedNeff());

  // The active entry should not have been evicted
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(CompilationCacheTest, CacheDestructionCleanup) {
  uint32_t stream_id = 1;

  // Add some entries
  for (int i = 0; i < 3; ++i) {
    MockNeuronKernelExecution kernel_exec("cleanup_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto keys_before = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_before.size(), 3);
  EXPECT_GT(cache_->GetMemoryUsageBytes(), 0);

  // Clear cache (simulates destruction cleanup)
  cache_->Clear();

  auto keys_after = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_after.size(), 0);
  EXPECT_EQ(cache_->GetMemoryUsageBytes(), 0);
}

class CompilationCacheEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original environment variables
    original_max_size_ = getenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
    original_max_entries_ = getenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
    original_service_env_ = getenv("NEURON_COMPILE_SERVICE");
    // Force local backend
    setenv("NEURON_COMPILE_SERVICE", "0", 1);
  }

  void TearDown() override {
    // Restore original environment variables
    if (original_max_size_) {
      setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", original_max_size_, 1);
    } else {
      unsetenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
    }

    if (original_max_entries_) {
      setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", original_max_entries_, 1);
    } else {
      unsetenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
    }

    if (original_service_env_) {
      setenv("NEURON_COMPILE_SERVICE", original_service_env_, 1);
    } else {
      unsetenv("NEURON_COMPILE_SERVICE");
    }
  }

  const char* original_max_size_ = nullptr;
  const char* original_max_entries_ = nullptr;
  const char* original_service_env_ = nullptr;
};

TEST_F(CompilationCacheEnvTest, EnvironmentVariableConfiguration) {
  // Set environment variables
  setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", "50", 1);
  setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", "200", 1);

  // Create new cache instance to pick up environment variables
  auto env_cache = std::make_unique<CompilationCache>();

  EXPECT_EQ(env_cache->GetMaxCacheSizeBytes(), 50 * 1024 * 1024);
  EXPECT_EQ(env_cache->GetMaxCacheEntries(), 200);
}

TEST_F(CompilationCacheEnvTest, InvalidEnvironmentVariables) {
  // Set invalid environment variables
  setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", "invalid", 1);
  setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", "not_a_number", 1);

  // Invalid environment variables should throw exceptions
  EXPECT_THROW({ auto env_cache = std::make_unique<CompilationCache>(); }, std::runtime_error);

  // Test each invalid variable separately
  unsetenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
  EXPECT_THROW({ auto env_cache = std::make_unique<CompilationCache>(); }, std::runtime_error);

  unsetenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
  setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", "not_a_number", 1);
  EXPECT_THROW({ auto env_cache = std::make_unique<CompilationCache>(); }, std::runtime_error);
}

TEST_F(CompilationCacheEnvTest, DefaultValuesWithoutEnvironment) {
  // Ensure no environment variables are set
  unsetenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
  unsetenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");

  // Create cache without environment variables
  auto default_cache = std::make_unique<CompilationCache>();

  // Should use default constants
  EXPECT_EQ(default_cache->GetMaxCacheSizeBytes(), CompilationCache::kDefaultMaxCacheSizeBytes);
  EXPECT_EQ(default_cache->GetMaxCacheEntries(), CompilationCache::kDefaultMaxCacheEntries);
}

TEST_F(CompilationCacheEnvTest, BackendSwitchingViaEnvVar) {
  // Disabled: NEURON_COMPILE_SERVICE=0 → local backend
  setenv("NEURON_COMPILE_SERVICE", "0", 1);
  auto local_cache = std::make_unique<CompilationCache>();
  EXPECT_FALSE(local_cache->IsServiceBackend());
  EXPECT_EQ(local_cache->GetBackendName(), "local");
  local_cache.reset();

  // Default (unset): service backend enabled (may fall back to local if service fails)
  unsetenv("NEURON_COMPILE_SERVICE");
  auto default_cache = std::make_unique<CompilationCache>();
  std::string name = default_cache->GetBackendName();
  EXPECT_TRUE(name == "service" || name == "local");
}

TEST_F(CompilationCacheTest, NestedLocks) {
  const std::string cache_key = "deadlock_test_key";
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};

  // Create entry and get smart pointer
  MockNeuronKernelExecution kernel_exec(cache_key);
  cache_->GetOrCompileNeff(kernel_exec, 1);

  auto neff_ptr = cache_->CreateNeffBytesPtr(ir_bytes, cache_key);

  // Launch thread that will trigger eviction (acquires cache_mutex_ then active_entries_mutex_)
  std::atomic<bool> eviction_started{false};
  std::atomic<bool> eviction_completed{false};

  auto eviction_thread = std::thread([&]() {
    eviction_started = true;
    // This should acquire locks in the correct order
    cache_->EvictLruEntries(0);  // Force eviction
    eviction_completed = true;
  });

  // Wait for eviction to start
  while (!eviction_started) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Simultaneously destroy smart pointer (acquires active_entries_mutex_ then potentially
  // cache_mutex_)
  neff_ptr.reset();

  // Wait for eviction to complete - should not deadlock
  eviction_thread.join();
  EXPECT_TRUE(eviction_completed);
}

TEST_F(CompilationCacheTest, ExceptionSafetyDuringCompilation) {
  class FailingKernelExecution : public MockNeuronKernelExecution {
   public:
    FailingKernelExecution(const std::string& key) : MockNeuronKernelExecution(key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("Compilation failed");
    }
  };

  const std::string cache_key = "exception_test_key";
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};

  // Launch multiple threads that will all fail compilation
  const int num_threads = 3;
  std::vector<std::future<void>> futures;
  std::atomic<int> exceptions_caught{0};

  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      try {
        FailingKernelExecution kernel_exec(cache_key);
        cache_->GetOrCompileNeff(kernel_exec, i + 1);
      } catch (const std::runtime_error&) {
        exceptions_caught++;
      }
    }));
  }

  // Wait for all threads
  for (auto& future : futures) {
    future.wait();
  }

  // All threads should have caught exceptions
  EXPECT_EQ(exceptions_caught.load(), num_threads);

  // Cache should have the failed entry cached
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);

  // Retrying the same key should fast-fail from cached failed entry
  FailingKernelExecution retry_kernel(cache_key);
  EXPECT_THROW(cache_->GetOrCompileNeff(retry_kernel, 999), std::runtime_error);

  // A different key should compile successfully
  MockNeuronKernelExecution successful_kernel("different_key");
  cache_->GetOrCompileNeff(successful_kernel, 999);

  auto final_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(final_keys.size(), 2);  // failed + successful
}

TEST_F(LimitedCompilationCacheTest, ConcurrentEvictionAndAccess) {
  SetUpWithLimits(1, 1024);  // 1 entry max, 1GB size

  const std::string cache_key = "eviction_race_test";
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};

  // Create first entry
  MockNeuronKernelExecution kernel_exec1(cache_key);
  cache_->GetOrCompileNeff(kernel_exec1, 1);

  // Get smart pointer to first entry
  auto neff_ptr = cache_->CreateNeffBytesPtr(ir_bytes, cache_key);

  // Launch thread that creates second entry (should trigger eviction)
  std::atomic<bool> second_compilation_done{false};
  auto eviction_thread = std::thread([&]() {
    const std::string cache_key2 = "eviction_race_test_2";
    MockNeuronKernelExecution kernel_exec2(cache_key2);
    cache_->GetOrCompileNeff(kernel_exec2, 2);
    second_compilation_done = true;
  });

  // Simultaneously access the first entry multiple times
  std::atomic<int> successful_accesses{0};
  for (int i = 0; i < 100; ++i) {
    if (neff_ptr && !neff_ptr->empty()) {
      successful_accesses++;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  eviction_thread.join();
  EXPECT_TRUE(second_compilation_done);

  // First entry should remain accessible through smart pointer even if evicted
  EXPECT_GT(successful_accesses.load(), 0);
  EXPECT_TRUE(neff_ptr && !neff_ptr->empty());
}

// Persistent Cache Integration Tests
// These tests verify persistent cache behavior integrated with CompilationCache

TEST_F(CompilationCacheTest, PersistentCacheWritesToDisk) {
  // Verify that compiled NEFFs are written to the persistent cache directory
  uint32_t stream_id = 1;
  const std::string cache_key = "persistent_write_test_key";

  MockNeuronKernelExecution kernel_exec(cache_key);
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());

  // Verify that the cache directory now contains files
  bool found_files = false;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(temp_dir_->path())) {
    if (entry.is_regular_file()) {
      found_files = true;
      break;
    }
  }
  EXPECT_TRUE(found_files) << "Expected persistent cache files in " << temp_dir_->path();
}

TEST_F(CompilationCacheTest, PersistentCacheHitAfterInMemoryCacheMiss) {
  // Test that data can be retrieved from persistent cache after in-memory cache is cleared
  uint32_t stream_id = 1;
  const std::string cache_key = "persistent_recovery_test_key";

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // First compile - should trigger actual compilation
  MockNeuronKernelExecution kernel_exec1(cache_key);
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  EXPECT_TRUE(kernel_exec1.HasCachedNeff());
  auto original_neff = kernel_exec1.GetCachedNeff();
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Clear in-memory cache only (preserve persistent cache)
  cache_->ClearInMemoryCache();

  // Request same key - should load from persistent cache
  MockNeuronKernelExecution kernel_exec2(cache_key);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());

  // Verify no additional compilation occurred - data came from disk
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Verify data matches what was originally compiled
  EXPECT_EQ(original_neff, kernel_exec2.GetCachedNeff());
}

TEST_F(CompilationCacheTest, ClearPersistentCachePreservesInMemoryCache) {
  using namespace at::neuron::metrics;
  // Test that ClearPersistentCache only clears persistent cache while preserving in-memory cache
  uint32_t stream_id = 1;
  const std::string cache_key = "clear_persistent_test_key";

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // First compile - populates both in-memory and persistent cache
  MockNeuronKernelExecution kernel_exec1(cache_key);
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  EXPECT_TRUE(kernel_exec1.HasCachedNeff());
  auto original_neff = kernel_exec1.GetCachedNeff();
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Access again to trigger in-memory cache hit
  MockNeuronKernelExecution kernel_exec1b(cache_key);
  cache_->GetOrCompileNeff(kernel_exec1b, stream_id);

  // Verify in-memory cache has the entry
  auto keys_before = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_before.size(), 1);
  EXPECT_TRUE(keys_before.count(cache_key) > 0);

  // Store in-memory cache metric value before clearing persistent cache
  auto* arena = NeuronMetricsArena::Get();
  auto* in_memory_hits_counter = arena->GetCounter("CompilationCache.InMemoryHits");
  ASSERT_NE(in_memory_hits_counter, nullptr);
  uint64_t in_memory_hits_before = in_memory_hits_counter->Value();
  EXPECT_GT(in_memory_hits_before, 0);

  // Clear persistent cache only (preserve in-memory cache)
  cache_->ClearPersistentCache();

  // Verify in-memory cache still has the entry
  auto keys_after = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys_after.size(), 1);
  EXPECT_TRUE(keys_after.count(cache_key) > 0);

  // Verify in-memory cache metrics are preserved
  uint64_t in_memory_hits_after = in_memory_hits_counter->Value();
  EXPECT_EQ(in_memory_hits_after, in_memory_hits_before);

  // Request same key - should hit in-memory cache (no recompilation)
  MockNeuronKernelExecution kernel_exec2(cache_key);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());
  EXPECT_EQ(original_neff, kernel_exec2.GetCachedNeff());

  // Verify no additional compilation - in-memory cache hit
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, ClearInMemoryCachePreservesPersistentCacheMetrics) {
  using namespace at::neuron::metrics;

  // Test that ClearInMemoryCache resets in-memory cache metrics but preserves persistent metrics
  uint32_t stream_id = 1;

  // Compile an entry
  MockNeuronKernelExecution kernel_exec1("inmem_clear_metrics_key");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);

  // Access again to trigger in-memory cache hit
  MockNeuronKernelExecution kernel_exec2("inmem_clear_metrics_key");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  // Verify in-memory cache metrics
  auto* arena = NeuronMetricsArena::Get();
  auto* in_memory_hits = arena->GetCounter("CompilationCache.InMemoryHits");
  ASSERT_NE(in_memory_hits, nullptr);
  ASSERT_GE(in_memory_hits->Value(), 0);

  // Clear in-memory to force persistent lookup
  cache_->ClearInMemoryCache();

  // Verify in-memory cache metrics are cleared
  EXPECT_EQ(in_memory_hits->Value(), 0);

  // Access again for persistent hit
  MockNeuronKernelExecution kernel_exec3("inmem_clear_metrics_key");
  cache_->GetOrCompileNeff(kernel_exec3, stream_id);

  // Store the persistent hits before in-memory cache clear
  auto* persistent_hits_counter = arena->GetCounter("CompilationCache.PersistentHits");
  ASSERT_NE(persistent_hits_counter, nullptr);
  uint64_t persistent_hits_value_before = persistent_hits_counter->Value();
  EXPECT_GT(persistent_hits_value_before, 0);

  // Clear in-memory caches
  cache_->ClearInMemoryCache();

  // Verify persistent cache metrics are not cleared by ClearInMemoryCache
  uint64_t persistent_hits_value_after = persistent_hits_counter->Value();
  EXPECT_EQ(persistent_hits_value_after, persistent_hits_value_before);
}

TEST_F(CompilationCacheTest, ClearAllCachesResetsAllMetrics) {
  using namespace at::neuron::metrics;

  // Test that Clear() resets all cache metrics
  uint32_t stream_id = 1;

  // Compile entries to populate metrics
  MockNeuronKernelExecution kernel_exec1("full_clear_key");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);

  // Access again for in-memory hit
  MockNeuronKernelExecution kernel_exec2("full_clear_key");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  // Clear in-memory to force persistent lookup
  cache_->ClearInMemoryCache();

  // Access again for persistent hit
  MockNeuronKernelExecution kernel_exec3("full_clear_key");
  cache_->GetOrCompileNeff(kernel_exec3, stream_id);

  auto* arena = NeuronMetricsArena::Get();

  // Verify metrics exist and have values
  auto* in_memory_hits = arena->GetCounter("CompilationCache.InMemoryHits");
  auto* persistent_hits = arena->GetCounter("CompilationCache.PersistentHits");
  auto* total_compilations = arena->GetCounter("CompilationCache.TotalCompilations");
  ASSERT_NE(in_memory_hits, nullptr);
  ASSERT_NE(persistent_hits, nullptr);
  ASSERT_NE(total_compilations, nullptr);

  // Clear all caches
  cache_->Clear();

  // Verify all metrics are cleared
  EXPECT_EQ(in_memory_hits->Value(), 0);
  EXPECT_EQ(persistent_hits->Value(), 0);
  EXPECT_EQ(total_compilations->Value(), 0);
}

TEST_F(CompilationCacheTest, PersistentCacheSurvivesCacheRecreation) {
  // Test that persistent cache survives CompilationCache destruction/recreation
  uint32_t stream_id = 1;
  const std::string cache_key = "persistent_survival_test_key";

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // Compile and store with first cache instance
  {
    MockNeuronKernelExecution kernel_exec(cache_key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    EXPECT_TRUE(kernel_exec.HasCachedNeff());
  }

  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Destroy and recreate cache (same temp directory)
  cache_.reset();
  cache_ = std::make_unique<CompilationCache>();

  // Request same key - should load from persistent cache (not recompile)
  MockNeuronKernelExecution kernel_exec2(cache_key);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());

  // Verify no additional compilation occurred - data came from disk
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, MultipleCacheKeysPersisted) {
  // Test that multiple cache keys are properly persisted
  uint32_t stream_id = 1;
  std::vector<std::string> cache_keys = {"persist_key_1", "persist_key_2", "persist_key_3"};
  std::vector<std::vector<uint8_t>> original_neffs;

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // Compile multiple entries
  for (const auto& key : cache_keys) {
    MockNeuronKernelExecution kernel_exec(key);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    EXPECT_TRUE(kernel_exec.HasCachedNeff());
    original_neffs.push_back(kernel_exec.GetCachedNeff());
  }

  // Verify 3 compilations occurred
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 3);

  // Clear in-memory cache only (preserve persistent cache)
  cache_->ClearInMemoryCache();

  // Verify all entries can be recovered from persistent cache
  for (size_t i = 0; i < cache_keys.size(); ++i) {
    MockNeuronKernelExecution kernel_exec(cache_keys[i]);
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    EXPECT_TRUE(kernel_exec.HasCachedNeff());
    EXPECT_EQ(original_neffs[i], kernel_exec.GetCachedNeff());
  }

  // Verify no additional compilations - all data came from persistent cache
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 3);
}

TEST_F(CompilationCacheTest, PersistentCacheDisabledWhenEnvVarSet) {
  // Test that persistent caching is disabled when env var is set

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // First, re-initialize with persistent caching disabled
  cache_.reset();
  setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "1", 1);
  cache_ = std::make_unique<CompilationCache>();

  uint32_t stream_id = 1;
  const std::string cache_key = "disabled_persist_test_key";

  // Compile
  MockNeuronKernelExecution kernel_exec1(cache_key);
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  EXPECT_TRUE(kernel_exec1.HasCachedNeff());

  auto cached_neff1 = kernel_exec1.GetCachedNeff();

  // Clear in-memory cache only (persistent cache is disabled anyway)
  cache_->ClearInMemoryCache();

  // Request same key - should trigger recompilation since persistent cache is disabled
  MockNeuronKernelExecution kernel_exec2(cache_key);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());

  // Both should still have matching data (same mock compilation)
  EXPECT_EQ(cached_neff1, kernel_exec2.GetCachedNeff());

  // Verify 2 compilations occurred
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 2);

  // Restore env var for other tests
  unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");
}

TEST_F(CompilationCacheTest, ConcurrentPersistentCacheAccess) {
  // Test concurrent access to persistent cache
  const int num_threads = 4;
  const std::string cache_key = "concurrent_persist_test_key";
  uint32_t stream_id = 1;

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  // First, compile and populate persistent cache
  MockNeuronKernelExecution initial_exec(cache_key);
  cache_->GetOrCompileNeff(initial_exec, stream_id);
  auto original_neff = initial_exec.GetCachedNeff();
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Clear in-memory cache only (preserve persistent cache)
  cache_->ClearInMemoryCache();

  // Launch multiple threads to access from persistent cache concurrently
  std::vector<std::future<std::vector<uint8_t>>> futures;
  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      MockNeuronKernelExecution kernel_exec(cache_key);
      cache_->GetOrCompileNeff(kernel_exec, i + 1);
      return kernel_exec.GetCachedNeff();
    }));
  }

  // Collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should match the original
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_EQ(original_neff, results[i]);
  }

  // Verify no additional compilations - all data came from persistent cache
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

// Pause the disk lookup after the initial memory check releases the lock.
class PausedPersistentLookupKernel : public MockNeuronKernelExecution {
 public:
  PausedPersistentLookupKernel(const std::string& key, std::promise<void>& lookup_started,
                               std::shared_future<void> resume_lookup)
      : MockNeuronKernelExecution(key),
        lookup_started_(lookup_started),
        resume_lookup_(std::move(resume_lookup)) {}

  bool RequiresWrappedNeffHlo() const override {
    lookup_started_.set_value();
    resume_lookup_.wait();
    return false;
  }

 private:
  std::promise<void>& lookup_started_;
  std::shared_future<void> resume_lookup_;
};

TEST_F(CompilationCacheTest, ConcurrentPersistentHitsDoNotReplaceActiveNeff) {
  const std::string cache_key = "concurrent_persistent_hit_lifetime";
  MockNeuronKernelExecution first(cache_key);
  const std::string* persistent_key = first.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(persistent_key, nullptr);

  const std::vector<uint8_t> neff_bytes{0x5a, 0x5b};
  PersistentCacheBackend disk_cache(std::filesystem::path(temp_dir_->path()));
  ASSERT_TRUE(disk_cache.Put(*persistent_key, neff_bytes));
  MockNeuronKernelExecution::ResetCompileCounter();

  std::promise<void> lookup_started;
  std::promise<void> resume_lookup;
  PausedPersistentLookupKernel second(cache_key, lookup_started,
                                      resume_lookup.get_future().share());
  auto reader = std::async(std::launch::async, [&] { return cache_->GetOrCompileNeff(second, 2); });
  lookup_started.get_future().get();

  // Install a successful entry while the other disk lookup is paused.
  std::shared_ptr<CacheEntry> first_entry;
  EXPECT_NO_THROW(first_entry = cache_->GetOrCompileNeff(first, 1));
  resume_lookup.set_value();
  std::shared_ptr<CacheEntry> second_entry;
  EXPECT_NO_THROW(second_entry = reader.get());

  ASSERT_NE(first_entry, nullptr);
  ASSERT_NE(second_entry, nullptr);
  EXPECT_EQ(first_entry, second_entry);
  EXPECT_EQ(cache_->GetEntry(cache_key), first_entry);
  EXPECT_EQ(first_entry->neff_bytes, neff_bytes);
  ASSERT_TRUE(first.HasCachedNeff());
  ASSERT_TRUE(second.HasCachedNeff());
  EXPECT_EQ(&first.GetCachedNeff(), &first_entry->neff_bytes);
  EXPECT_EQ(&second.GetCachedNeff(), &first_entry->neff_bytes);
  EXPECT_EQ(first_entry->access_count.load(), 1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 0);

  auto* arena = metrics::NeuronMetricsArena::Get();
  auto* persistent_hits = arena->GetCounter("CompilationCache.PersistentHits");
  auto* in_memory_hits = arena->GetCounter("CompilationCache.InMemoryHits");
  ASSERT_NE(persistent_hits, nullptr);
  ASSERT_NE(in_memory_hits, nullptr);
  EXPECT_EQ(persistent_hits->Value(), 1);
  EXPECT_EQ(in_memory_hits->Value(), 1);
}

TEST_F(CompilationCacheTest, PersistentHitReplacesConcurrentFailedEntry) {
  class ThrowingNoHloKernel : public CompilableKernelExecution {
   public:
    explicit ThrowingNoHloKernel(const std::string& key)
        : CompilableKernelExecution(key, "", "", false) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("test compilation failed");
    }
  };

  const std::string cache_key = "persistent_hit_after_failure";
  std::promise<void> lookup_started;
  std::promise<void> resume_lookup;
  PausedPersistentLookupKernel kernel(cache_key, lookup_started,
                                      resume_lookup.get_future().share());
  const std::string* persistent_key = kernel.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(persistent_key, nullptr);
  const std::vector<uint8_t> neff_bytes{0x5a, 0x5b};
  PersistentCacheBackend disk_cache(std::filesystem::path(temp_dir_->path()));
  ASSERT_TRUE(disk_cache.Put(*persistent_key, neff_bytes));
  MockNeuronKernelExecution::ResetCompileCounter();

  auto reader = std::async(std::launch::async, [&] { return cache_->GetOrCompileNeff(kernel, 1); });
  lookup_started.get_future().get();

  // The kernel with no HLO skips the disk cache and records a compilation failure.
  ThrowingNoHloKernel failing(cache_key);
  EXPECT_THROW(cache_->GetOrCompileNeff(failing, 2), std::runtime_error);
  auto failed_entry = cache_->GetEntry(cache_key);
  resume_lookup.set_value();
  std::shared_ptr<CacheEntry> entry;
  EXPECT_NO_THROW(entry = reader.get());

  ASSERT_NE(failed_entry, nullptr);
  EXPECT_TRUE(failed_entry->IsFailed());
  EXPECT_FALSE(failing.HasCachedNeff());
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->IsFailed());
  EXPECT_NE(entry, failed_entry);
  EXPECT_EQ(cache_->GetEntry(cache_key), entry);
  ASSERT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(kernel.GetCachedNeff(), neff_bytes);
  EXPECT_EQ(&kernel.GetCachedNeff(), &entry->neff_bytes);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 0);

  auto* arena = metrics::NeuronMetricsArena::Get();
  auto* persistent_hits = arena->GetCounter("CompilationCache.PersistentHits");
  ASSERT_NE(persistent_hits, nullptr);
  EXPECT_EQ(persistent_hits->Value(), 1);
  if (auto* in_memory_hits = arena->GetCounter("CompilationCache.InMemoryHits")) {
    EXPECT_EQ(in_memory_hits->Value(), 0);
  }
}

TEST_F(CompilationCacheTest, PersistentHitSucceedsBeforePendingCompilationFails) {
  for (bool asynchronous : {false, true}) {
    const std::string cache_key =
        asynchronous ? "persistent_hit_pending_async" : "persistent_hit_pending_sync";
    std::promise<void> lookup_started;
    std::promise<void> resume_lookup;
    PausedPersistentLookupKernel kernel(cache_key, lookup_started,
                                        resume_lookup.get_future().share());
    const auto& key = kernel.GetCompilationCacheKey();
    const std::string* persistent_key = key.GetPersistentKey();
    ASSERT_NE(persistent_key, nullptr);
    const std::vector<uint8_t> neff_bytes{0x5a, 0x5b};
    PersistentCacheBackend disk_cache(std::filesystem::path(temp_dir_->path()));
    ASSERT_TRUE(disk_cache.Put(*persistent_key, neff_bytes));

    auto reader = std::async(std::launch::async, [&] {
      if (!asynchronous) return cache_->GetOrCompileNeff(kernel, 1);
      std::promise<bool> completed;
      auto completion = completed.get_future();
      cache_->GetOrCompileNeffAsync(
          kernel, 1, [&](bool success, const std::string&) { completed.set_value(success); });
      EXPECT_TRUE(completion.get());
      return cache_->GetEntry(cache_key);
    });
    auto started = lookup_started.get_future();
    if (started.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      resume_lookup.set_value();
      EXPECT_NO_THROW(reader.get());
      FAIL() << "Disk lookup did not reach the pause point";
    }
    started.get();

    auto [pending, pending_future] = CompilationCacheTestPeer::SeedPending(*cache_, key);
    resume_lookup.set_value();

    // The disk reader must complete while the other compilation remains pending.
    EXPECT_EQ(reader.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(pending_future.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    CompilationCacheTestPeer::FailPending(*cache_, key, pending);
    EXPECT_THROW(pending_future.get(), torch_neuronx::CompilationRuntimeException);

    std::shared_ptr<CacheEntry> entry;
    EXPECT_NO_THROW(entry = reader.get());
    ASSERT_NE(entry, nullptr);
    EXPECT_FALSE(entry->IsFailed());
    EXPECT_EQ(cache_->GetEntry(cache_key), entry);
    ASSERT_TRUE(kernel.HasCachedNeff());
    EXPECT_EQ(kernel.GetCachedNeff(), neff_bytes);
    EXPECT_EQ(&kernel.GetCachedNeff(), &entry->neff_bytes);
  }
}

TEST_F(CompilationCacheTest, PersistentHitReplacesFailedEntryBeforePendingCleanup) {
  const std::string cache_key = "persistent_hit_before_failure_cleanup";
  std::promise<void> lookup_started;
  std::promise<void> resume_lookup;
  PausedPersistentLookupKernel kernel(cache_key, lookup_started,
                                      resume_lookup.get_future().share());
  const auto& key = kernel.GetCompilationCacheKey();
  const std::string* persistent_key = key.GetPersistentKey();
  ASSERT_NE(persistent_key, nullptr);
  const std::vector<uint8_t> neff_bytes{0x5a, 0x5b};
  PersistentCacheBackend disk_cache(std::filesystem::path(temp_dir_->path()));
  ASSERT_TRUE(disk_cache.Put(*persistent_key, neff_bytes));
  MockNeuronKernelExecution::ResetCompileCounter();

  auto reader = std::async(std::launch::async, [&] { return cache_->GetOrCompileNeff(kernel, 1); });
  lookup_started.get_future().get();

  // Publish the failure and its ready future before the pending entry is removed.
  auto pending_future = CompilationCacheTestPeer::SeedFailedPending(*cache_, key);
  auto failed_entry = cache_->GetEntry(cache_key);
  resume_lookup.set_value();
  std::shared_ptr<CacheEntry> entry;
  EXPECT_NO_THROW(entry = reader.get());
  CompilationCacheTestPeer::CleanupPending(*cache_, key);

  EXPECT_THROW(pending_future.get(), torch_neuronx::CompilationRuntimeException);
  ASSERT_NE(failed_entry, nullptr);
  EXPECT_TRUE(failed_entry->IsFailed());
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->IsFailed());
  EXPECT_NE(entry, failed_entry);
  EXPECT_EQ(cache_->GetEntry(cache_key), entry);
  ASSERT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(kernel.GetCachedNeff(), neff_bytes);
  EXPECT_EQ(&kernel.GetCachedNeff(), &entry->neff_bytes);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 0);

  auto* arena = metrics::NeuronMetricsArena::Get();
  auto* persistent_hits = arena->GetCounter("CompilationCache.PersistentHits");
  ASSERT_NE(persistent_hits, nullptr);
  EXPECT_EQ(persistent_hits->Value(), 1);
  if (auto* in_memory_hits = arena->GetCounter("CompilationCache.InMemoryHits")) {
    EXPECT_EQ(in_memory_hits->Value(), 0);
  }
}

TEST_F(CompilationCacheTest, CentralizedMetricsHitMissTracking) {
  using namespace at::neuron::metrics;

  // Test that the metrics framework can be used alongside CompilationCache
  // Create counters to track cache operations
  NeuronCounter hits_counter("CompilationCache.Test.Hits");
  NeuronCounter misses_counter("CompilationCache.Test.Misses");

  uint32_t stream_id = 1;

  // First compilation - simulate a cache miss
  MockNeuronKernelExecution kernel_exec1("metrics_test_key");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  misses_counter.AddValue(1);

  // Second compilation with same key - simulate a cache hit
  MockNeuronKernelExecution kernel_exec2("metrics_test_key");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  hits_counter.AddValue(1);

  // Verify counters work correctly
  EXPECT_EQ(misses_counter.Value(), 1);
  EXPECT_EQ(hits_counter.Value(), 1);

  // Verify counters are accessible from the arena
  auto* arena_hits = NeuronMetricsArena::Get()->GetCounter("CompilationCache.Test.Hits");
  auto* arena_misses = NeuronMetricsArena::Get()->GetCounter("CompilationCache.Test.Misses");
  ASSERT_NE(arena_hits, nullptr);
  ASSERT_NE(arena_misses, nullptr);
  EXPECT_EQ(arena_hits->Value(), 1);
  EXPECT_EQ(arena_misses->Value(), 1);
}

TEST_F(CompilationCacheTest, CentralizedMetricsCompilationTracking) {
  using namespace at::neuron::metrics;

  // Test that timing metrics work alongside CompilationCache
  NeuronMetric compile_time("CompilationCache.Test.CompilationTime", MetricFnTime);

  uint32_t stream_id = 1;

  // Compile a few entries and record timing
  for (int i = 0; i < 3; ++i) {
    auto start = GetCurrentTimeNs();
    MockNeuronKernelExecution kernel_exec("metrics_compile_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
    auto end = GetCurrentTimeNs();
    compile_time.AddSample(end, static_cast<double>(end - start));
  }

  // Verify compilation time metric has samples
  EXPECT_EQ(compile_time.Samples().size(), 3);
  EXPECT_GT(compile_time.Accumulator(), 0);  // Should have recorded some time

  // Verify metric is accessible from the arena
  auto* arena_metric =
      NeuronMetricsArena::Get()->GetMetric("CompilationCache.Test.CompilationTime");
  ASSERT_NE(arena_metric, nullptr);
  EXPECT_GE(arena_metric->TotalSamples(), 3);
}

TEST_F(CompilationCacheTest, SameCacheKeyDifferentIRBytesCacheMiss) {
  // Version changes are captured in the base_key
  // Different base_keys produce different persistent keys.
  uint32_t stream_id = 1;

  MockNeuronKernelExecution::ResetCompileCounter();

  // Simulate version 1 (different base_key)
  MockNeuronKernelExecution kernel_exec_v1("version_test_key_v1");
  cache_->GetOrCompileNeff(kernel_exec_v1, stream_id);
  EXPECT_TRUE(kernel_exec_v1.HasCachedNeff());
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Clear in-memory cache to force persistent cache lookup
  cache_->ClearInMemoryCache();

  // Simulate version 2 (different base_key from new global version hash)
  MockNeuronKernelExecution kernel_exec_v2("version_test_key_v2");
  cache_->GetOrCompileNeff(kernel_exec_v2, stream_id);
  EXPECT_TRUE(kernel_exec_v2.HasCachedNeff());

  // Should have compiled again because base_key is different
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 2);
}

TEST_F(CompilationCacheTest, SameCacheKeySameIRBytesCacheHit) {
  // Test that same cache key with same IR bytes produces persistent cache hit
  uint32_t stream_id = 1;
  const std::string cache_key = "same_ir_test_key";
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};

  MockNeuronKernelExecution::ResetCompileCounter();

  // First compile
  MockNeuronKernelExecution kernel_exec1(cache_key, ir_bytes);
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);
  EXPECT_TRUE(kernel_exec1.HasCachedNeff());
  auto original_neff = kernel_exec1.GetCachedNeff();
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Clear in-memory cache to force persistent cache lookup
  cache_->ClearInMemoryCache();

  // Second request with same IR bytes - should be persistent cache hit
  MockNeuronKernelExecution kernel_exec2(cache_key, ir_bytes);
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);
  EXPECT_TRUE(kernel_exec2.HasCachedNeff());

  // Should NOT have compiled again - persistent cache hit
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1)
      << "Expected persistent cache hit when IR bytes are the same";

  // Results should match
  EXPECT_EQ(original_neff, kernel_exec2.GetCachedNeff());
}

TEST_F(CompilationCacheTest, InMemoryCacheKeyDifferentFromPersistentKey) {
  // Test that in-memory cache uses simple key while persistent uses IR-aware key
  // Same cache_key should hit in-memory cache even with different IR bytes
  uint32_t stream_id = 1;
  const std::string cache_key = "memory_vs_persistent_key";
  std::vector<uint8_t> ir_v1 = {0xAA, 0xBB};
  std::vector<uint8_t> ir_v2 = {0xCC, 0xDD};

  MockNeuronKernelExecution::ResetCompileCounter();

  // Compile with IR version 1
  MockNeuronKernelExecution kernel_exec_v1(cache_key, ir_v1);
  cache_->GetOrCompileNeff(kernel_exec_v1, stream_id);
  EXPECT_TRUE(kernel_exec_v1.HasCachedNeff());
  auto v1_neff = kernel_exec_v1.GetCachedNeff();
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Without clearing in-memory cache, request with different IR bytes
  // Should still hit in-memory cache (keyed by cache_key only)
  MockNeuronKernelExecution kernel_exec_v2(cache_key, ir_v2);
  cache_->GetOrCompileNeff(kernel_exec_v2, stream_id);
  EXPECT_TRUE(kernel_exec_v2.HasCachedNeff());

  // Should not have compiled again - in-memory cache hit
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1)
      << "Expected in-memory cache hit (keyed by cache_key only)";

  // Results should match (from in-memory cache)
  EXPECT_EQ(v1_neff, kernel_exec_v2.GetCachedNeff());
}

TEST_F(CompilationCacheTest, VersionChangeInvalidatesPersistentCacheOnly) {
  // Toolchain version changes produce different base_keys
  uint32_t stream_id = 1;

  MockNeuronKernelExecution::ResetCompileCounter();

  // Simulate old toolchain compilation
  MockNeuronKernelExecution old_kernel("version_invalidation_test_old");
  cache_->GetOrCompileNeff(old_kernel, stream_id);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Simulate process restart (destroy and recreate cache)
  cache_.reset();
  cache_ = std::make_unique<CompilationCache>();

  // Simulate new toolchain version - different base_key for same operation
  MockNeuronKernelExecution new_kernel("version_invalidation_test_new");
  cache_->GetOrCompileNeff(new_kernel, stream_id);

  // Should recompile because base_key is different (version hash changed)
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 2)
      << "Expected recompilation after toolchain version change";
}

// =============================================================================
// Compilation File Lock Tests with Persistent Cache
// =============================================================================

TEST_F(CompilationCacheTest, FileLockPreventsMultipleCompilations) {
  // Test that file locking prevents multiple threads from compiling the same key
  // Only one thread should compile, others should wait and read from cache
  const std::string cache_key = "file_lock_test_key";
  const int num_threads = 4;

  // Reset compile counter for this test
  MockNeuronKernelExecution::ResetCompileCounter();

  std::vector<std::future<std::vector<uint8_t>>> futures;

  // Launch multiple threads trying to compile the same key concurrently
  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      MockNeuronKernelExecution kernel_exec(cache_key);
      cache_->GetOrCompileNeff(kernel_exec, i + 1);
      return kernel_exec.GetCachedNeff();
    }));
  }

  // Collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should be identical (same compilation result)
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i]);
  }

  // Only one compilation should have occurred due to file locking
  // Other threads should have waited for the lock, then read from persistent cache
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1)
      << "Expected exactly 1 compilation, but got "
      << MockNeuronKernelExecution::GetCompileCounter().load()
      << ". File locking should prevent duplicate compilations.";
}

TEST_F(CompilationCacheTest, FileLockWithSlowCompilation) {
  // Test that file locking works correctly with slow compilations
  // Threads should wait for the lock and then read from cache
  const std::string cache_key = "slow_compile_lock_test";
  const int num_threads = 3;

  // Create a mock that takes significant time to compile
  class SlowCompileMock : public MockNeuronKernelExecution {
   public:
    SlowCompileMock(const std::string& key) : MockNeuronKernelExecution(key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      GetCompileCounter()++;
      // Simulate long compilation (200ms)
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      std::vector<uint8_t> neff_data;
      std::string content = "SLOW_NEFF_" + GetCompileKey();
      neff_data.assign(content.begin(), content.end());
      return neff_data;
    }
  };

  MockNeuronKernelExecution::ResetCompileCounter();

  std::vector<std::future<std::vector<uint8_t>>> futures;
  std::atomic<int> threads_started{0};

  // Launch threads with slight stagger to ensure they all compete for the lock
  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      threads_started++;
      SlowCompileMock kernel_exec(cache_key);
      cache_->GetOrCompileNeff(kernel_exec, i + 1);
      return kernel_exec.GetCachedNeff();
    }));
    // Small delay to stagger thread starts (but they'll still compete for lock)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Wait for all threads to start
  while (threads_started.load() < num_threads) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should match
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i]);
  }

  // Only 1 compilation should occur
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, FileLockWithDifferentKeys) {
  // Test that different cache keys can compile concurrently (different lock files)
  const int num_keys = 4;

  MockNeuronKernelExecution::ResetCompileCounter();

  std::vector<std::future<std::vector<uint8_t>>> futures;

  // Launch threads with different keys - they should not block each other
  for (int i = 0; i < num_keys; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      std::string unique_key = "different_key_" + std::to_string(i);
      MockNeuronKernelExecution kernel_exec(unique_key);
      cache_->GetOrCompileNeff(kernel_exec, i + 1);
      return kernel_exec.GetCachedNeff();
    }));
  }

  // Collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should be different (different keys)
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = i + 1; j < results.size(); ++j) {
      EXPECT_NE(results[i], results[j]);
    }
  }

  // All 4 compilations should occur (different lock files)
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), num_keys);
}

TEST_F(CompilationCacheTest, FileLocksReadFromCacheAfterFirstCompile) {
  // Test second thead reads from persistent cache after first thread compiles with lock.

  const std::string cache_key = "sequential_lock_test";

  MockNeuronKernelExecution::ResetCompileCounter();

  std::atomic<bool> thread_a_acquired_lock{false};
  std::atomic<bool> thread_a_compilation_done{false};
  std::atomic<bool> thread_b_started{false};
  std::vector<uint8_t> thread_a_result;
  std::vector<uint8_t> thread_b_result;

  // Thread A: Acquires lock and compiles (with longer compilation time)
  class SlowFirstCompile : public MockNeuronKernelExecution {
   public:
    std::atomic<bool>& acquired_flag_;
    std::atomic<bool>& done_flag_;

    SlowFirstCompile(const std::string& key, std::atomic<bool>& acquired, std::atomic<bool>& done)
        : MockNeuronKernelExecution(key), acquired_flag_(acquired), done_flag_(done) {}

    std::vector<uint8_t> CompileToNeff() const override {
      acquired_flag_ = true;
      GetCompileCounter()++;
      // Simulate compilation time
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      done_flag_ = true;
      std::vector<uint8_t> neff_data;
      std::string content = "FIRST_COMPILE_" + GetCompileKey();
      neff_data.assign(content.begin(), content.end());
      return neff_data;
    }
  };

  std::thread thread_a([&]() {
    SlowFirstCompile kernel_exec(cache_key, thread_a_acquired_lock, thread_a_compilation_done);
    cache_->GetOrCompileNeff(kernel_exec, 1);
    thread_a_result = kernel_exec.GetCachedNeff();
  });

  // Wait for Thread A to start compilation (acquire lock)
  while (!thread_a_acquired_lock.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Thread B: Starts after Thread A has the lock, should wait then read from cache
  std::thread thread_b([&]() {
    thread_b_started = true;
    MockNeuronKernelExecution kernel_exec(cache_key);
    cache_->GetOrCompileNeff(kernel_exec, 2);
    thread_b_result = kernel_exec.GetCachedNeff();
  });

  // Wait for both threads to complete
  thread_a.join();
  thread_b.join();

  // Verify Thread B waited for Thread A (started before compilation was done)
  EXPECT_TRUE(thread_b_started.load());

  // Both should have the same result
  EXPECT_EQ(thread_a_result, thread_b_result);

  // Only 1 compilation should have occurred (Thread B read from cache)
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, FileLockRecoveryAfterCompilationFailure) {
  // Test that lock is properly released even if compilation fails
  // Next thread should be able to acquire lock and compile
  const std::string cache_key = "lock_failure_recovery_test";

  MockNeuronKernelExecution::ResetCompileCounter();

  // First, a compilation that fails
  class FailingCompile : public MockNeuronKernelExecution {
   public:
    FailingCompile(const std::string& key) : MockNeuronKernelExecution(key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("Compilation failed intentionally");
    }
  };

  // Thread 1: Fails compilation (should release lock on exception)
  std::thread thread_fail([&]() {
    FailingCompile kernel_exec(cache_key);
    try {
      cache_->GetOrCompileNeff(kernel_exec, 1);
    } catch (const std::runtime_error&) {
      // Expected
    }
  });
  thread_fail.join();

  // Thread 2: Same key should fast-fail from cached failed entry (no recompilation)
  std::thread thread_retry([&]() {
    MockNeuronKernelExecution kernel_exec(cache_key);
    EXPECT_THROW(cache_->GetOrCompileNeff(kernel_exec, 2), std::runtime_error);
  });
  thread_retry.join();

  // No successful compilations should have occurred
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 0);

  // Thread 3: A different key should compile successfully (lock was released)
  std::thread thread_success([&]() {
    MockNeuronKernelExecution kernel_exec("different_lock_key");
    cache_->GetOrCompileNeff(kernel_exec, 3);
    EXPECT_TRUE(kernel_exec.HasCachedNeff());
  });
  thread_success.join();

  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, FileLockStressTestManyThreadsSameKey) {
  // Stress test: Many threads trying to compile the same key
  const std::string cache_key = "stress_lock_test";
  const int num_threads = 16;

  MockNeuronKernelExecution::ResetCompileCounter();

  std::vector<std::future<std::vector<uint8_t>>> futures;

  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      MockNeuronKernelExecution kernel_exec(cache_key);
      cache_->GetOrCompileNeff(kernel_exec, i + 1);
      return kernel_exec.GetCachedNeff();
    }));
  }

  // Collect results
  std::vector<std::vector<uint8_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // All results should match
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i]);
  }

  // Only 1 compilation should occur
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1)
      << "Stress test: Expected 1 compilation, got "
      << MockNeuronKernelExecution::GetCompileCounter().load();
}

TEST_F(CompilationCacheTest, LockFilesWrittenToLocalCacheDir) {
  // Test that lock files are created in the local cache directory
  uint32_t stream_id = 1;
  const std::string cache_key = "local_cache_dir_test_key";

  // Verify local cache dir exists and is empty or contains only expected directories
  EXPECT_TRUE(std::filesystem::exists(local_cache_dir_->fs_path()));

  // Compile something to trigger lock file creation
  MockNeuronKernelExecution kernel_exec(cache_key);
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());

  // Verify that the locks subdirectory was created in local cache dir
  std::filesystem::path locks_dir = local_cache_dir_->fs_path() / "locks";
  EXPECT_TRUE(std::filesystem::exists(locks_dir))
      << "Expected locks directory at: " << locks_dir.string();

  // Check that lock files exist
  bool found_lock_related = false;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(local_cache_dir_->fs_path())) {
    if (entry.path().string().find("lock") != std::string::npos ||
        entry.path().string().find(".lock") != std::string::npos) {
      found_lock_related = true;
      break;
    }
  }
  EXPECT_TRUE(found_lock_related);
}

TEST_F(CompilationCacheTest, FailedEntryFastFailsWithoutRecompilation) {
  // Verify that a failed compilation is cached and subsequent attempts
  // fail immediately without triggering another compilation.
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    static std::atomic<int>& CompileCount() {
      static std::atomic<int> count{0};
      return count;
    }
    ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}
    std::vector<uint8_t> CompileToNeff() const override {
      CompileCount()++;
      throw std::runtime_error("bad input");
    }
  };

  ThrowingKernel::CompileCount().store(0);
  uint32_t stream_id = 1;

  // First attempt: compilation runs and fails
  ThrowingKernel kernel1("fail_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(kernel1, stream_id), std::runtime_error);
  EXPECT_EQ(ThrowingKernel::CompileCount().load(), 1);

  // Second attempt: should fast-fail from cache, no recompilation
  MockNeuronKernelExecution kernel2("fail_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(kernel2, stream_id), std::runtime_error);
  EXPECT_EQ(ThrowingKernel::CompileCount().load(), 1);  // Still 1
}

TEST_F(CompilationCacheTest, FailedEntryPreservesErrorMessage) {
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}
    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("specific error: invalid flag --xyz");
    }
  };

  ThrowingKernel kernel("msg_test_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(kernel, 1), std::runtime_error);

  // Verify the cached entry preserves the error message
  auto entry = cache_->GetEntry("msg_test_key");
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->IsFailed());
  EXPECT_THAT(entry->GetFailureMessage(),
              ::testing::HasSubstr("specific error: invalid flag --xyz"));
}

TEST_F(CompilationCacheTest, FailedEntryDoesNotAffectOtherKeys) {
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}
    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("compilation failed");
    }
  };

  uint32_t stream_id = 1;

  // Fail one key
  ThrowingKernel failing("bad_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(failing, stream_id), std::runtime_error);

  // A different key should compile successfully
  MockNeuronKernelExecution good("good_key");
  cache_->GetOrCompileNeff(good, stream_id);
  EXPECT_TRUE(good.HasCachedNeff());

  // The failed key should still fast-fail
  MockNeuronKernelExecution retry("bad_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(retry, stream_id), std::runtime_error);
}

TEST_F(CompilationCacheTest, FailedEntryClearedOnCacheClear) {
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}
    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("compilation failed");
    }
  };

  // Cache a failure
  ThrowingKernel kernel("clear_test_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(kernel, 1), std::runtime_error);
  EXPECT_EQ(cache_->GetAllCacheKeys().size(), 1);

  // Clear should remove it
  cache_->ClearInMemoryCache();
  EXPECT_EQ(cache_->GetAllCacheKeys().size(), 0);
}

// Test for race condition between double-check and pending registration.
TEST_F(CompilationCacheTest, RaceConditionDoubleCheckToRegistration) {
  // Disable persistent cache to isolate the in-memory coordination bug
  setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true", 1);
  cache_ = std::make_unique<CompilationCache>();

  const int num_threads = 8;
  const int num_iterations = 50;
  std::atomic<int> promise_errors{0};

  for (int iter = 0; iter < num_iterations; ++iter) {
    cache_->Clear();
    MockNeuronKernelExecution::ResetCompileCounter();

    std::string cache_key = "race_test_key_" + std::to_string(iter);
    std::atomic<int> threads_ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&, i]() {
        threads_ready++;
        while (!start.load()) {
          std::this_thread::yield();
        }
        try {
          MockNeuronKernelExecution kernel_exec(cache_key);
          cache_->GetOrCompileNeff(kernel_exec, i + 1);
        } catch (const std::future_error& e) {
          // "Promise already satisfied" indicates the race condition
          promise_errors++;
        }
      });
    }

    while (threads_ready.load() < num_threads) {
      std::this_thread::yield();
    }
    start.store(true);

    for (auto& t : threads) {
      t.join();
    }

    // Should have exactly 1 compilation per iteration
    if (MockNeuronKernelExecution::GetCompileCounter().load() != 1) {
      ADD_FAILURE() << "Iteration " << iter << ": Expected 1 compilation, got "
                    << MockNeuronKernelExecution::GetCompileCounter().load();
    }
  }

  EXPECT_EQ(promise_errors.load(), 0)
      << "Caught " << promise_errors.load()
      << " 'Promise already satisfied' errors - race condition detected!";

  unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");
}

// =============================================================================
// Dual-Path Routing Tests
// =============================================================================

// Verify that with local backend, sync GetOrCompileNeff works and
// IsServiceBackend returns false (ProcessCompilationTask takes sync path).
TEST_F(CompilationCacheTest, LocalBackendUsesSyncPath) {
  EXPECT_FALSE(cache_->IsServiceBackend());
  EXPECT_EQ(cache_->GetBackendName(), "local");

  MockNeuronKernelExecution::ResetCompileCounter();
  MockNeuronKernelExecution kernel("sync_path_test");

  // Sync path: compiles directly on this thread, returns with NEFF set
  cache_->GetOrCompileNeff(kernel, 1);
  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

// Verify that with local backend, async GetOrCompileNeffAsync also works
// (uses thread pool). This is the path taken when service is active.
TEST_F(CompilationCacheTest, AsyncPathWorksWithLocalBackend) {
  MockNeuronKernelExecution::ResetCompileCounter();
  MockNeuronKernelExecution kernel("async_path_test");

  std::atomic<bool> done{false};
  bool success = false;

  cache_->GetOrCompileNeffAsync(kernel, 1, [&](bool s, const std::string&) {
    success = s;
    done = true;
  });

  for (int i = 0; i < 100 && !done.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_TRUE(done.load());
  EXPECT_TRUE(success);
  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

// =============================================================================
// GetOrCompileNeffAsync Tests (local backend + thread pool)
// =============================================================================

TEST_F(CompilationCacheTest, AsyncCacheHit) {
  MockNeuronKernelExecution kernel("async_hit_test");
  cache_->GetOrCompileNeff(kernel, 1);
  ASSERT_TRUE(kernel.HasCachedNeff());

  MockNeuronKernelExecution kernel2("async_hit_test");
  std::atomic<bool> callback_invoked{false};
  bool callback_success = false;

  cache_->GetOrCompileNeffAsync(kernel2, 1, [&](bool success, const std::string&) {
    callback_success = success;
    callback_invoked = true;
  });

  for (int i = 0; i < 100 && !callback_invoked.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_TRUE(callback_invoked.load());
  EXPECT_TRUE(callback_success);
  EXPECT_TRUE(kernel2.HasCachedNeff());
}

TEST_F(CompilationCacheTest, AsyncCacheMiss) {
  MockNeuronKernelExecution::ResetCompileCounter();
  MockNeuronKernelExecution kernel("async_miss_test");

  std::atomic<bool> callback_invoked{false};
  bool callback_success = false;

  cache_->GetOrCompileNeffAsync(kernel, 1, [&](bool success, const std::string&) {
    callback_success = success;
    callback_invoked = true;
  });

  for (int i = 0; i < 100 && !callback_invoked.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_TRUE(callback_invoked.load());
  EXPECT_TRUE(callback_success);
  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, AsyncPendingCompilation) {
  MockNeuronKernelExecution::ResetCompileCounter();

  std::atomic<int> callbacks_invoked{0};
  std::atomic<int> callbacks_success{0};

  std::vector<std::unique_ptr<MockNeuronKernelExecution>> kernels;
  for (int i = 0; i < 4; ++i)
    kernels.push_back(std::make_unique<MockNeuronKernelExecution>("async_pending_test"));

  for (auto& kernel : kernels) {
    cache_->GetOrCompileNeffAsync(*kernel, 1, [&](bool success, const std::string&) {
      if (success) callbacks_success++;
      callbacks_invoked++;
    });
  }

  for (int i = 0; i < 100 && callbacks_invoked.load() < 4; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(callbacks_invoked.load(), 4);
  EXPECT_EQ(callbacks_success.load(), 4);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  for (auto& kernel : kernels) EXPECT_TRUE(kernel->HasCachedNeff());
}

TEST_F(CompilationCacheTest, AsyncPendingFailureAfterCleanupReportsErrorWithoutCacheLock) {
  class PendingCompilationTestCache : public CompilationCache {
   public:
    using CompilationCache::ContinuePendingCompilationAsync;
  };
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    explicit ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("test compilation failed");
    }
  };

  PendingCompilationTestCache cache;
  const std::string cache_key = "async_pending_failure_after_cleanup";
  ThrowingKernel failing(cache_key);
  EXPECT_THROW(cache.GetOrCompileNeff(failing, 1), torch_neuronx::CompilationRuntimeException);
  auto failed_entry = cache.GetEntry(cache_key);
  ASSERT_NE(failed_entry, nullptr);
  ASSERT_TRUE(failed_entry->IsFailed());

  // Resume the pending request after failure handling removes its future.
  MockNeuronKernelExecution waiter(cache_key);
  int callbacks = 0;
  cache.ContinuePendingCompilationAsync(waiter, [&](bool success, const std::string& error) {
    ++callbacks;
    EXPECT_FALSE(success);
    EXPECT_EQ(error, failed_entry->GetFailureMessage());
    EXPECT_THAT(error, ::testing::HasSubstr("test compilation failed"));
    EXPECT_FALSE(waiter.HasCachedNeff());
    // This method needs the exclusive cache lock.
    cache.RegisterPersistentKeyHint(cache_key, "callback_key_hint");
    EXPECT_EQ(cache.GetEntry(cache_key), failed_entry);
  });

  EXPECT_EQ(callbacks, 1);
  EXPECT_FALSE(waiter.HasCachedNeff());
}

TEST_F(CompilationCacheTest, AsyncMultipleDifferentKeys) {
  MockNeuronKernelExecution::ResetCompileCounter();

  std::atomic<int> callbacks_invoked{0};
  const int num_keys = 4;

  std::vector<std::unique_ptr<MockNeuronKernelExecution>> kernels;
  for (int i = 0; i < num_keys; ++i)
    kernels.push_back(
        std::make_unique<MockNeuronKernelExecution>("async_multi_" + std::to_string(i)));

  for (auto& kernel : kernels)
    cache_->GetOrCompileNeffAsync(*kernel, 1,
                                  [&](bool, const std::string&) { callbacks_invoked++; });

  for (int i = 0; i < 100 && callbacks_invoked.load() < num_keys; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(callbacks_invoked.load(), num_keys);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), num_keys);
}

TEST_F(CompilationCacheTest, AsyncFutureCaptureRace) {
  MockNeuronKernelExecution::ResetCompileCounter();

  std::atomic<int> callbacks_invoked{0};
  std::atomic<int> callbacks_success{0};

  std::vector<std::unique_ptr<MockNeuronKernelExecution>> kernels;
  for (int i = 0; i < 8; ++i)
    kernels.push_back(std::make_unique<MockNeuronKernelExecution>("async_race_test"));

  for (auto& kernel : kernels) {
    cache_->GetOrCompileNeffAsync(*kernel, 1, [&](bool success, const std::string&) {
      if (success) callbacks_success++;
      callbacks_invoked++;
    });
  }

  for (int i = 0; i < 200 && callbacks_invoked.load() < 8; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(callbacks_invoked.load(), 8);
  EXPECT_EQ(callbacks_success.load(), 8);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, AsyncErrorStashedOnKernel) {
  class ThrowingKernel : public MockNeuronKernelExecution {
   public:
    ThrowingKernel(const std::string& key) : MockNeuronKernelExecution(key) {}
    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("neuronx-cc: unsupported op xyz");
    }
  };

  ThrowingKernel kernel("async_error_stash_key");
  EXPECT_FALSE(kernel.IsFailedCompilation());

  std::atomic<bool> done{false};
  cache_->GetOrCompileNeffAsync(kernel, 1, [&](bool success, const std::string& error) {
    // Simulate what OperationExecutionEngine does on failure:
    if (!success && !error.empty()) {
      kernel.SetCompilationError(error);
    }
    done = true;
  });

  while (!done.load()) std::this_thread::yield();

  // Verify the new APIs work end-to-end
  EXPECT_FALSE(kernel.HasCachedNeff());
  EXPECT_TRUE(kernel.IsFailedCompilation());
  EXPECT_THAT(kernel.GetCompilationError(), ::testing::HasSubstr("neuronx-cc: unsupported op xyz"));
}

// ============================================================================
// Tests for graceful-skip behavior when HLO bytes are not available.
// Simulates NeffDirectKernelExecution-style paths where persistent caching
// is intentionally disabled.
// ============================================================================

class NoHloKernelExecution : public CompilableKernelExecution {
 public:
  static std::atomic<int>& GetCompileCounter() {
    static std::atomic<int> compile_counter{0};
    return compile_counter;
  }

  static void ResetCompileCounter() { GetCompileCounter().store(0); }

  explicit NoHloKernelExecution(const std::string& cache_key)
      : CompilableKernelExecution(cache_key, "", "", false) {
    // Intentionally do NOT call SetHloBytesPtr — persistent key is unavailable.
  }

  std::vector<uint8_t> CompileToNeff() const override {
    GetCompileCounter()++;
    std::string neff_content = "NEFF_NO_HLO_" + GetCompileKey();
    return std::vector<uint8_t>(neff_content.begin(), neff_content.end());
  }

  // Returning empty matches the contract for paths without HLO.
  const std::vector<uint8_t>& GetHloBytes() const override {
    static const std::vector<uint8_t> empty;
    return empty;
  }
};

TEST_F(CompilationCacheTest, GracefulSkip_NoHloBytesCompilesWithoutCrash) {
  // Without HLO bytes, GetPersistentKey() returns nullptr. Cache must
  // fall through to in-memory-only mode, compile the kernel, and cache
  // it without dereferencing the null persistent key.
  NoHloKernelExecution::ResetCompileCounter();
  NoHloKernelExecution kernel("no_hlo_key_1");

  // Sanity: persistent key really is unavailable.
  EXPECT_EQ(kernel.GetCompilationCacheKey().GetPersistentKey(), nullptr);

  cache_->GetOrCompileNeff(kernel, /*stream_id=*/1);

  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_FALSE(kernel.GetCachedNeff().empty());
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 1);
}

TEST_F(CompilationCacheTest, GracefulSkip_NoHloBytesHitsInMemoryCache) {
  // Even without persistent cache backing, in-memory cache should still
  // dedupe on base_key — second lookup should not trigger a recompile.
  NoHloKernelExecution::ResetCompileCounter();

  NoHloKernelExecution k1("no_hlo_key_2");
  cache_->GetOrCompileNeff(k1, /*stream_id=*/1);
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 1);

  NoHloKernelExecution k2("no_hlo_key_2");
  cache_->GetOrCompileNeff(k2, /*stream_id=*/1);
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 1);  // no recompile
  EXPECT_EQ(k1.GetCachedNeff(), k2.GetCachedNeff());
}

TEST_F(CompilationCacheTest, GracefulSkip_NoHloBytesNotPersistedAcrossCacheInstances) {
  // Without HLO bytes, the persistent cache layer is bypassed, so a fresh
  // CompilationCache instance does NOT see the previously-compiled NEFF.
  // This documents the cost of the graceful-skip path: in-memory only.
  NoHloKernelExecution::ResetCompileCounter();

  NoHloKernelExecution k1("no_hlo_key_3");
  cache_->GetOrCompileNeff(k1, /*stream_id=*/1);
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 1);

  // New cache instance — sharing the same persistent dir, but persistent
  // cache wasn't populated for this kernel.
  auto fresh_cache = std::make_unique<CompilationCache>();
  NoHloKernelExecution k2("no_hlo_key_3");
  fresh_cache->GetOrCompileNeff(k2, /*stream_id=*/1);
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 2);  // recompiled
}

TEST_F(CompilationCacheTest, KeyHint_NoHloKernelResolvesPersistentKeyFromCompileStep) {
  // NeffDirectKernelExecution carries no IR, so it cannot compute the content key.
  // The compile step records base_key -> persistent_key. A later IR-less execution
  // with the same base_key must resolve that key and read the NEFF from disk.
  MockNeuronKernelExecution::ResetCompileCounter();
  NoHloKernelExecution::ResetCompileCounter();
  const std::string base_key = "shared_base_key_for_hint";

  // Compile with IR. This writes the NEFF to disk and records the hint.
  // Scope the kernel: it holds the NEFF pointer, which keeps the entry active and
  // therefore not evictable.
  std::vector<uint8_t> compiled_neff;
  {
    MockNeuronKernelExecution with_hlo(base_key);
    cache_->GetOrCompileNeff(with_hlo, /*stream_id=*/1);
    EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
    compiled_neff = with_hlo.GetCachedNeff();
  }

  // Drop the NEFF but keep the hint, which is what LRU eviction does.
  EXPECT_EQ(cache_->EvictLruEntries(/*target_entries=*/0), 1u);

  // The IR-less kernel must not reach CompileToNeff().
  NoHloKernelExecution no_hlo(base_key);
  EXPECT_EQ(no_hlo.GetCompilationCacheKey().GetPersistentKey(), nullptr);
  cache_->GetOrCompileNeff(no_hlo, /*stream_id=*/1);

  EXPECT_TRUE(no_hlo.HasCachedNeff());
  EXPECT_EQ(NoHloKernelExecution::GetCompileCounter().load(), 0);
  EXPECT_EQ(no_hlo.GetCachedNeff(), compiled_neff);
}

TEST_F(CompilationCacheTest, CrossCommitReuse_SameHloDifferentBaseKeyHitsPersistentCache) {
  // The cross-commit reuse scenario: two kernels with different base_keys
  // (simulating a Python source edit that bumped global_version_hash) but
  // identical HLO bytes. After the first compile populates the persistent
  // cache, a fresh CompilationCache instance should see the entry via the
  // HLO-content-derived persistent key.
  MockNeuronKernelExecution::ResetCompileCounter();
  std::vector<uint8_t> shared_hlo = {0x42, 0x42, 0x42, 0x42};

  // First "commit": base_key v1, populates persistent cache.
  MockNeuronKernelExecution k1("base_key_v1_pre_edit", shared_hlo);
  cache_->GetOrCompileNeff(k1, /*stream_id=*/1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  // Second "commit" simulated via fresh cache (clears in-memory state).
  // Different base_key but same HLO → persistent key matches → cache hit.
  auto fresh_cache = std::make_unique<CompilationCache>();
  MockNeuronKernelExecution k2("base_key_v2_post_edit", shared_hlo);
  fresh_cache->GetOrCompileNeff(k2, /*stream_id=*/1);

  // No new compile — persistent cache served the previous NEFF.
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
  EXPECT_TRUE(k2.HasCachedNeff());
}

// WLO mock kernel: advertises RequiresWrappedNeffHlo and emits a wrapped_neff.hlo
// sidecar of a configurable size alongside the (base-class) NEFF, so we can assert
// the sidecar is stored on the cache entry and counted in memory accounting.
class WloMockKernelExecution : public MockNeuronKernelExecution {
 public:
  WloMockKernelExecution(const std::string& cache_key, size_t sidecar_size)
      : MockNeuronKernelExecution(cache_key, /*wlo_enabled=*/true), sidecar_size_(sidecar_size) {}

  bool RequiresWrappedNeffHlo() const override { return true; }
  CompileResult CompileToNeffResult() const override {
    return CompileResult{CompileToNeff(), std::vector<uint8_t>(sidecar_size_, 0xAB)};
  }

 private:
  size_t sidecar_size_;
};

TEST_F(CompilationCacheTest, WrappedNeffHloIsStoredAndCounted) {
  // A WLO compile stores the wrapped_neff.hlo sidecar on the cache entry...
  WloMockKernelExecution wlo_kernel("wlo_key", /*sidecar_size=*/4096);
  cache_->GetOrCompileNeff(wlo_kernel, /*stream_id=*/1);

  auto entry = cache_->GetEntry(wlo_kernel.GetCompileKey());
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->wrapped_neff_hlo.size(), 4096u);

  // ...and that sidecar counts toward the cache's memory footprint. Compare against
  // a fresh cache holding a non-WLO entry with the SAME NEFF (identical cache key ->
  // identical MockNeuronKernelExecution NEFF bytes), whose sidecar is empty. The only
  // difference is the 4096-byte sidecar, so the WLO cache must report more memory.
  auto plain_cache = std::make_unique<CompilationCache>();
  MockNeuronKernelExecution plain_kernel("wlo_key");
  plain_cache->GetOrCompileNeff(plain_kernel, /*stream_id=*/1);

  EXPECT_GT(cache_->GetMemoryUsageBytes(), plain_cache->GetMemoryUsageBytes());
}

TEST_F(CompilationCacheTest, ZeroTransformWloHitsPersistentCacheAcrossInstances) {
  // A WLO compile that yields NO transforms (empty wrapped_neff.hlo) must still be a
  // cacheable positive — a fresh process must hit the persistent cache, not recompile every
  // run. The version-tagged sidecar marker is what makes the empty case distinguishable from
  // "sidecar missing".
  MockNeuronKernelExecution::ResetCompileCounter();

  WloMockKernelExecution k1("zero_xform_wlo", /*sidecar_size=*/0);
  cache_->GetOrCompileNeff(k1, /*stream_id=*/1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);
  auto e1 = cache_->GetEntry(k1.GetCompileKey());
  ASSERT_NE(e1, nullptr);
  EXPECT_TRUE(e1->wrapped_neff_hlo.empty());  // zero transforms

  // Fresh instance on the same persistent dir simulates a new process.
  auto fresh_cache = std::make_unique<CompilationCache>();
  WloMockKernelExecution k2("zero_xform_wlo", /*sidecar_size=*/0);
  fresh_cache->GetOrCompileNeff(k2, /*stream_id=*/1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);  // hit, no recompile
  EXPECT_TRUE(k2.HasCachedNeff());
}

TEST_F(CompilationCacheTest, WloSidecarRecoveredAcrossInstances) {
  // A WLO compile with N transforms persists the sidecar; a fresh process recovers the
  // exact payload (version marker stripped) without recompiling.
  MockNeuronKernelExecution::ResetCompileCounter();

  WloMockKernelExecution k1("nonzero_xform_wlo", /*sidecar_size=*/128);
  cache_->GetOrCompileNeff(k1, /*stream_id=*/1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);

  auto fresh_cache = std::make_unique<CompilationCache>();
  WloMockKernelExecution k2("nonzero_xform_wlo", /*sidecar_size=*/128);
  fresh_cache->GetOrCompileNeff(k2, /*stream_id=*/1);
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 1);  // hit, no recompile

  auto e2 = fresh_cache->GetEntry(k2.GetCompileKey());
  ASSERT_NE(e2, nullptr);
  EXPECT_EQ(e2->wrapped_neff_hlo, std::vector<uint8_t>(128, 0xAB));  // payload recovered, no marker
}

TEST_F(CompilationCacheTest, WloAndNonWloSameCacheKeyDoNotCollide) {
  // Same base cache_key, one non-WLO and one WLO. Because WLO-ness is part of the key, they
  // occupy distinct entries — a WLO request is never served the non-WLO NEFF.
  MockNeuronKernelExecution::ResetCompileCounter();

  MockNeuronKernelExecution plain("shared_key");
  cache_->GetOrCompileNeff(plain, /*stream_id=*/1);

  WloMockKernelExecution wlo("shared_key", /*sidecar_size=*/64);
  cache_->GetOrCompileNeff(wlo, /*stream_id=*/1);

  // Two distinct entries, two compiles — no collision.
  EXPECT_EQ(MockNeuronKernelExecution::GetCompileCounter().load(), 2);
  EXPECT_NE(plain.GetCompileKey(), wlo.GetCompileKey());
  auto wlo_entry = cache_->GetEntry(wlo.GetCompileKey());
  ASSERT_NE(wlo_entry, nullptr);
  EXPECT_EQ(wlo_entry->wrapped_neff_hlo.size(), 64u);
}

TEST_F(CompilationCacheTest, GetWrappedNeffHloReturnsSidecar) {
  // After a WLO compile completes, GetWrappedNeffHlo returns the entry's wrapped_neff.hlo sidecar.
  WloMockKernelExecution wlo("wlo_fetch", /*sidecar_size=*/64);
  cache_->GetOrCompileNeff(wlo, /*stream_id=*/1);
  EXPECT_EQ(cache_->GetWrappedNeffHlo(wlo.GetCompileKey()), std::vector<uint8_t>(64, 0xAB));
}

TEST_F(CompilationCacheTest, GetWrappedNeffHloFailsClosedWhenAbsent) {
  // No pending compile and no cached entry -> after the bounded wait for an async submit to
  // register, fail closed rather than return an empty (wrong) sidecar that would run a relaid NEFF
  // against untransformed weights. A short timeout keeps the test fast.
  EXPECT_THROW(cache_->GetWrappedNeffHlo("no_such_key", std::chrono::milliseconds(50)),
               std::runtime_error);
}

TEST_F(CompilationCacheTest, GetWrappedNeffHloWaitsForAsyncRegistration) {
  // A' fix: the WLO fetch can run before the async submit has registered the pending compile.
  // GetWrappedNeffHlo must WAIT for the key to appear (rather than fail immediately). Here a
  // producer thread compiles shortly after we begin waiting; the fetch should block, then return
  // the sidecar once the compile is registered/completed.
  WloMockKernelExecution wlo("wlo_async_fetch", /*sidecar_size=*/32);
  std::thread producer([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cache_->GetOrCompileNeff(wlo, /*stream_id=*/2);
  });
  auto sidecar = cache_->GetWrappedNeffHlo(wlo.GetCompileKey(), std::chrono::seconds(10));
  producer.join();
  EXPECT_EQ(sidecar, std::vector<uint8_t>(32, 0xAB));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
