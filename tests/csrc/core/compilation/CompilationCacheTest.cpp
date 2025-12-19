#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"

using namespace at::neuron;
using ::testing::_;
using ::testing::Return;

class MockNeuronKernelExecution : public CompilableKernelExecution {
 public:
  MockNeuronKernelExecution(const std::string& cache_key)
      : CompilableKernelExecution(cache_key, "", "", false) {}

  // Mock compilation that returns dummy NEFF data
  std::vector<uint8_t> CompileToNeff() const override {
    // Simulate compilation time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Return dummy NEFF data based on cache key for uniqueness
    std::vector<uint8_t> neff_data;
    std::string neff_content = "NEFF_" + GetCacheKey() + "_" + GetAdditionalArgs();
    neff_data.assign(neff_content.begin(), neff_content.end());
    return neff_data;
  }
};

class CompilationCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cache_ = std::make_unique<CompilationCache>();
    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override { cache_->Clear(); }

  std::unique_ptr<CompilationCache> cache_;
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

TEST_F(CompilationCacheTest, CacheStats) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Initial stats
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 0);
  EXPECT_EQ(stats.cache_hits, 0);
  EXPECT_EQ(stats.cache_misses, 0);
  EXPECT_EQ(stats.hit_rate, 0.0);

  // Add one entry
  MockNeuronKernelExecution kernel_exec1("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec1, stream_id);

  stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 1);
  EXPECT_EQ(stats.cache_misses, 1);
  EXPECT_GT(stats.memory_usage_bytes, 0);

  // Cache hit
  MockNeuronKernelExecution kernel_exec2("test_key_1");
  cache_->GetOrCompileNeff(kernel_exec2, stream_id);

  stats = cache_->GetStatistics();
  EXPECT_EQ(stats.cache_hits, 1);
  EXPECT_EQ(stats.hit_rate, 0.5);  // 1 hit out of 2 total requests
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
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 1);
}

class LimitedCompilationCacheTest : public ::testing::Test {
 protected:
  void SetUpWithLimits(size_t max_entries, size_t max_size_mb) {
    // Save original environment variables
    original_max_size_ = getenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
    original_max_entries_ = getenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");

    // Set test-specific limits
    setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", std::to_string(max_entries).c_str(), 1);
    setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", std::to_string(max_size_mb).c_str(), 1);

    // Create cache with new limits
    cache_ = std::make_unique<CompilationCache>();
    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override {
    cache_.reset();

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
  }

  std::unique_ptr<CompilationCache> cache_;
  const char* original_max_size_ = nullptr;
  const char* original_max_entries_ = nullptr;
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

  auto stats = cache_->GetStatistics();
  EXPECT_LE(stats.total_entries, 2);        // Should not exceed limit
  EXPECT_GT(stats.evictions_performed, 0);  // Should have performed evictions

  // Verify that the most recently used entries are kept
  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_LE(all_keys.size(), 2);
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

  auto stats = cache_->GetStatistics();
  EXPECT_GT(stats.evictions_performed, 0);           // Should have performed evictions
  EXPECT_LT(stats.memory_usage_bytes, 1024 * 1024);  // Should be under 1MB limit
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

  auto stats = cache_->GetStatistics();
  EXPECT_GT(stats.evictions_performed, 0);
}

TEST_F(LimitedCompilationCacheTest, NoEvictionWhenUnderLimits) {
  SetUpWithLimits(10, 1024);  // 10 entries max, 1GB size

  uint32_t stream_id = 1;

  // Add fewer entries than the limit
  for (int i = 0; i < 5; ++i) {
    MockNeuronKernelExecution kernel_exec("no_eviction_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 5);
  EXPECT_EQ(stats.evictions_performed, 0);  // No evictions should occur

  auto all_keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(all_keys.size(), 5);
}

TEST_F(CompilationCacheTest, ManualEviction) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Add multiple entries
  for (int i = 0; i < 5; ++i) {
    MockNeuronKernelExecution kernel_exec("test_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto stats_before = cache_->GetStatistics();
  EXPECT_EQ(stats_before.total_entries, 5);

  // Manually evict to 2 entries
  cache_->EvictLruEntries(2);

  auto stats_after = cache_->GetStatistics();
  EXPECT_LE(stats_after.total_entries, 2);
}

TEST_F(CompilationCacheTest, ClearCache) {
  std::vector<uint8_t> ir_bytes = {0x01, 0x02, 0x03, 0x04};
  uint32_t stream_id = 1;

  // Add some entries
  for (int i = 0; i < 3; ++i) {
    MockNeuronKernelExecution kernel_exec("test_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto stats_before = cache_->GetStatistics();
  EXPECT_GT(stats_before.total_entries, 0);

  // Clear cache
  cache_->Clear();

  auto stats_after = cache_->GetStatistics();
  EXPECT_EQ(stats_after.total_entries, 0);
  EXPECT_EQ(stats_after.cache_hits, 0);
  EXPECT_EQ(stats_after.cache_misses, 0);
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
  class ThrowingMockKernelExecution : public CompilableKernelExecution {
   public:
    ThrowingMockKernelExecution(const std::string& cache_key)
        : CompilableKernelExecution(cache_key, "", "", false) {}

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
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 1);
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

  // Test empty cache key
  MockNeuronKernelExecution kernel_exec_empty("");
  cache_->GetOrCompileNeff(kernel_exec_empty, stream_id);
  EXPECT_TRUE(kernel_exec_empty.HasCachedNeff());
  EXPECT_FALSE(kernel_exec_empty.GetCachedNeff().empty());

  // Verify cache hit with same empty key
  MockNeuronKernelExecution kernel_exec_empty2("");
  cache_->GetOrCompileNeff(kernel_exec_empty2, stream_id);
  EXPECT_TRUE(kernel_exec_empty2.HasCachedNeff());
  EXPECT_EQ(kernel_exec_empty.GetCachedNeff(), kernel_exec_empty2.GetCachedNeff());

  // Test single character keys
  MockNeuronKernelExecution kernel_exec_single("a");
  cache_->GetOrCompileNeff(kernel_exec_single, stream_id);
  EXPECT_NE(kernel_exec_empty.GetCachedNeff(),
            kernel_exec_single.GetCachedNeff());  // Should be different from empty key
}

TEST_F(CompilationCacheTest, CacheStateAfterCompilationError) {
  uint32_t stream_id = 1;

  // Create a mock that will throw during compilation
  class ThrowingMockKernelExecution : public CompilableKernelExecution {
   public:
    ThrowingMockKernelExecution(const std::string& cache_key)
        : CompilableKernelExecution(cache_key, "", "", false) {}

    std::vector<uint8_t> CompileToNeff() const override {
      throw std::runtime_error("Compilation failed");
    }
  };

  // Add a successful compilation first
  MockNeuronKernelExecution successful_kernel("successful_key");
  cache_->GetOrCompileNeff(successful_kernel, stream_id);

  auto stats_before = cache_->GetStatistics();
  EXPECT_EQ(stats_before.total_entries, 1);

  // Try to compile a failing kernel
  ThrowingMockKernelExecution failing_kernel("failing_key");
  EXPECT_THROW(cache_->GetOrCompileNeff(failing_kernel, stream_id), std::runtime_error);

  // Cache should still be in a valid state
  auto stats_after = cache_->GetStatistics();
  EXPECT_EQ(stats_after.total_entries, 1);  // Should still have the successful entry

  // Should be able to access the successful entry
  MockNeuronKernelExecution successful_kernel2("successful_key");
  cache_->GetOrCompileNeff(successful_kernel2, stream_id);
  EXPECT_TRUE(successful_kernel2.HasCachedNeff());

  // Should be able to add new entries after error
  MockNeuronKernelExecution new_kernel("new_key");
  cache_->GetOrCompileNeff(new_kernel, stream_id);
  EXPECT_TRUE(new_kernel.HasCachedNeff());

  auto final_stats = cache_->GetStatistics();
  EXPECT_EQ(final_stats.total_entries, 2);
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
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 3);
  EXPECT_EQ(stats.evictions_performed, 2);

  // Verify cache consistency - entries that are still cached should return correct data
  auto cache_keys = cache_->GetAllCacheKeys();
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

// Additional Statistics and Configuration Tests
TEST_F(CompilationCacheTest, CompilationTimeTracking) {
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

  auto stats_before = cache_->GetStatistics();
  EXPECT_EQ(stats_before.total_compilation_time.count(), 0);

  // Compile multiple entries
  for (int i = 0; i < 3; ++i) {
    SlowMockKernelExecution kernel_exec("slow_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto stats_after = cache_->GetStatistics();
  EXPECT_GT(stats_after.total_compilation_time.count(), 150);  // At least 3 * 50ms
  EXPECT_EQ(stats_after.total_entries, 3);
}

TEST_F(LimitedCompilationCacheTest, MemoryUtilizationRatio) {
  SetUpWithLimits(1000, 1);  // 1000 entries max, 1MB limit
  uint32_t stream_id = 1;

  // Add some entries
  MockNeuronKernelExecution kernel_exec("memory_test_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);

  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.max_cache_size_bytes, 1024 * 1024);  // 1MB
  EXPECT_GE(stats.memory_utilization_ratio, 0.0);
  EXPECT_LE(stats.memory_utilization_ratio, 1.0);

  // If we have memory usage, ratio should be > 0
  if (stats.memory_usage_bytes > 0) {
    EXPECT_GT(stats.memory_utilization_ratio, 0.0);
  }
}

TEST_F(LimitedCompilationCacheTest, ConfigurationLimitsReflection) {
  SetUpWithLimits(100, 10);  // 100 entries max, 10MB limit

  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.max_cache_size_bytes, 10 * 1024 * 1024);
  EXPECT_EQ(stats.max_cache_entries, 100);
}

TEST_F(LimitedCompilationCacheTest, ZeroCacheLimits) {
  SetUpWithLimits(0, 0);  // Zero limits (unlimited)

  uint32_t stream_id = 1;

  // Should still be able to compile (limits of 0 mean unlimited)
  MockNeuronKernelExecution kernel_exec("zero_limit_key");
  cache_->GetOrCompileNeff(kernel_exec, stream_id);
  EXPECT_TRUE(kernel_exec.HasCachedNeff());

  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.max_cache_size_bytes, 0);
  EXPECT_EQ(stats.max_cache_entries, 0);
  EXPECT_EQ(stats.total_entries, 1);
}

TEST_F(LimitedCompilationCacheTest, VeryLargeCacheLimits) {
  // Test behavior with very large limits
  size_t large_mb = 1024 * 1024;   // 1TB
  size_t large_entries = 1000000;  // 1M entries

  SetUpWithLimits(large_entries, large_mb);

  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.max_cache_size_bytes, large_mb * 1024 * 1024);
  EXPECT_EQ(stats.max_cache_entries, large_entries);

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
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 1);
}

TEST_F(CompilationCacheTest, CacheDestructionCleanup) {
  uint32_t stream_id = 1;

  // Add some entries
  for (int i = 0; i < 3; ++i) {
    MockNeuronKernelExecution kernel_exec("cleanup_key_" + std::to_string(i));
    cache_->GetOrCompileNeff(kernel_exec, stream_id);
  }

  auto stats_before = cache_->GetStatistics();
  EXPECT_EQ(stats_before.total_entries, 3);
  EXPECT_GT(stats_before.memory_usage_bytes, 0);

  // Clear cache (simulates destruction cleanup)
  cache_->Clear();

  auto stats_after = cache_->GetStatistics();
  EXPECT_EQ(stats_after.total_entries, 0);
  EXPECT_EQ(stats_after.memory_usage_bytes, 0);
  EXPECT_EQ(stats_after.cache_hits, 0);
  EXPECT_EQ(stats_after.cache_misses, 0);
}

class CompilationCacheEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original environment variables
    original_max_size_ = getenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB");
    original_max_entries_ = getenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES");
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
  }

  const char* original_max_size_ = nullptr;
  const char* original_max_entries_ = nullptr;
};

TEST_F(CompilationCacheEnvTest, EnvironmentVariableConfiguration) {
  // Set environment variables
  setenv("NEURON_COMPILATION_CACHE_MAX_SIZE_MB", "50", 1);
  setenv("NEURON_COMPILATION_CACHE_MAX_ENTRIES", "200", 1);

  // Create new cache instance to pick up environment variables
  auto env_cache = std::make_unique<CompilationCache>();

  auto stats = env_cache->GetStatistics();
  EXPECT_EQ(stats.max_cache_size_bytes, 50 * 1024 * 1024);
  EXPECT_EQ(stats.max_cache_entries, 200);
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
  auto stats = default_cache->GetStatistics();

  // Should use default constants
  EXPECT_EQ(stats.max_cache_size_bytes, CompilationCache::kDefaultMaxCacheSizeBytes);
  EXPECT_EQ(stats.max_cache_entries, CompilationCache::kDefaultMaxCacheEntries);
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

  // Cache should be clean (no pending compilations left)
  auto stats = cache_->GetStatistics();
  EXPECT_EQ(stats.total_entries, 0);

  // Should be able to compile successfully after failures
  MockNeuronKernelExecution successful_kernel(cache_key);
  cache_->GetOrCompileNeff(successful_kernel, 999);

  auto final_stats = cache_->GetStatistics();
  EXPECT_EQ(final_stats.total_entries, 1);
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
