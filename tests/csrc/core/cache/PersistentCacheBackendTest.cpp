#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/cache/CacheUtils.h"
#include "torch_neuronx/csrc/core/cache/PersistentCacheBackend.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

namespace at::neuron {
namespace {

class PersistentCacheBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save and unset env var to ensure explicit path constructor is used
    const char* env_val = std::getenv("TORCH_NEURONX_NEFF_CACHE_DIR");
    if (env_val) {
      saved_cache_dir_env_ = env_val;
    }
    unsetenv("TORCH_NEURONX_NEFF_CACHE_DIR");

    // Create a unique test directory using RAII wrapper that uses nftw for cleanup
    // This avoids the segfault bug in std::filesystem::remove_all
    temp_dir_ = std::make_unique<TempDirectory>("test_cache_");
    persistent_cache_ = std::make_unique<PersistentCacheBackend>(temp_dir_->fs_path());
  }

  void TearDown() override {
    persistent_cache_.reset();
    temp_dir_.reset();  // RAII destructor handles cleanup using nftw

    // Restore env var
    if (!saved_cache_dir_env_.empty()) {
      setenv("TORCH_NEURONX_NEFF_CACHE_DIR", saved_cache_dir_env_.c_str(), 1);
    }
  }

  std::filesystem::path test_dir() const { return temp_dir_->fs_path(); }

  // Helper to create a valid XXH3_128 cache key (32 hex chars)
  std::string MakeCacheKey(const std::string& base_key, const std::vector<uint8_t>& ir_bytes = {}) {
    return cache_utils::MakePersistentCacheKey(base_key, ir_bytes);
  }

  std::unique_ptr<TempDirectory> temp_dir_;
  std::unique_ptr<PersistentCacheBackend> persistent_cache_;
  std::string saved_cache_dir_env_;
};

// ============================================================================
// PersistentCacheBackend Tests
// ============================================================================

TEST_F(PersistentCacheBackendTest, PutAndGet) {
  std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04, 0x05};
  std::string key = MakeCacheKey("test_key");

  ASSERT_TRUE(persistent_cache_->Put(key, data));
  ASSERT_TRUE(persistent_cache_->Exists(key));

  auto retrieved = persistent_cache_->Get(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data);
}

TEST_F(PersistentCacheBackendTest, GetNonExistent) {
  std::string key = MakeCacheKey("nonexistent_key");
  auto result = persistent_cache_->Get(key);
  EXPECT_FALSE(result.has_value());
}

TEST_F(PersistentCacheBackendTest, ExistsNonExistent) {
  std::string key = MakeCacheKey("nonexistent_key");
  EXPECT_FALSE(persistent_cache_->Exists(key));
}

TEST_F(PersistentCacheBackendTest, Remove) {
  std::vector<uint8_t> data = {0x01, 0x02, 0x03};
  std::string key = MakeCacheKey("test_key_remove");

  ASSERT_TRUE(persistent_cache_->Put(key, data));
  ASSERT_TRUE(persistent_cache_->Exists(key));

  EXPECT_TRUE(persistent_cache_->Remove(key));
  EXPECT_FALSE(persistent_cache_->Exists(key));
}

TEST_F(PersistentCacheBackendTest, RemoveNonExistent) {
  // Remove should return true even if key doesn't exist
  std::string key = MakeCacheKey("nonexistent_key");
  EXPECT_TRUE(persistent_cache_->Remove(key));
}

TEST_F(PersistentCacheBackendTest, ShardedPath) {
  // Key is an XXH3_128 hash (32 hex chars), first 4 chars used for 2-level sharding
  std::string key = MakeCacheKey("test_sharding");
  std::vector<uint8_t> data = {0xFF};

  ASSERT_TRUE(persistent_cache_->Put(key, data));

  // Verify a sharded directory structure was created (2 levels deep)
  std::error_code ec;
  bool found_nested_file = false;
  for (auto& level1_entry : std::filesystem::directory_iterator(test_dir(), ec)) {
    if (level1_entry.is_directory()) {
      for (auto& level2_entry : std::filesystem::directory_iterator(level1_entry.path(), ec)) {
        if (level2_entry.is_directory()) {
          for (auto& file_entry : std::filesystem::directory_iterator(level2_entry.path(), ec)) {
            if (file_entry.is_regular_file() && file_entry.path().extension() == ".neff") {
              found_nested_file = true;
              break;
            }
          }
        }
      }
    }
  }
  EXPECT_TRUE(found_nested_file) << "Expected to find .neff file in 2-level sharded directory";

  // Verify the data can be retrieved
  ASSERT_TRUE(persistent_cache_->Exists(key));
  auto retrieved = persistent_cache_->Get(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data);
}

TEST_F(PersistentCacheBackendTest, XXH3KeyFormat) {
  // Keys are expected to be 32-char XXH3_128 hex strings
  // MakeCacheKey ensures this format
  std::string key = MakeCacheKey("test_format");
  EXPECT_EQ(key.size(), 32);

  // All chars should be hex
  for (char c : key) {
    EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
  }

  std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04, 0x05};
  ASSERT_TRUE(persistent_cache_->Put(key, data));
  ASSERT_TRUE(persistent_cache_->Exists(key));

  auto retrieved = persistent_cache_->Get(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data);
}

TEST_F(PersistentCacheBackendTest, DifferentIRBytesDifferentKeys) {
  // Verify that different IR bytes produce different cache keys
  std::string base_key = "same_op_name";
  std::vector<uint8_t> ir1 = {0x01, 0x02, 0x03};  // Version 1 IR
  std::vector<uint8_t> ir2 = {0x04, 0x05, 0x06};  // Version 2 IR

  std::string key1 = cache_utils::MakePersistentCacheKey(base_key, ir1);
  std::string key2 = cache_utils::MakePersistentCacheKey(base_key, ir2);

  std::vector<uint8_t> data1 = {0xAA};
  std::vector<uint8_t> data2 = {0xBB};

  ASSERT_TRUE(persistent_cache_->Put(key1, data1));
  ASSERT_TRUE(persistent_cache_->Put(key2, data2));

  // Both keys should exist and return their respective data
  auto retrieved1 = persistent_cache_->Get(key1);
  auto retrieved2 = persistent_cache_->Get(key2);

  ASSERT_TRUE(retrieved1.has_value());
  ASSERT_TRUE(retrieved2.has_value());
  EXPECT_EQ(retrieved1.value(), data1);
  EXPECT_EQ(retrieved2.value(), data2);
}

TEST_F(PersistentCacheBackendTest, ConsistentKeyGeneration) {
  // Verify that the same inputs always produce the same key
  std::string base_key = "consistent_test";
  std::vector<uint8_t> ir = {0xDE, 0xAD, 0xBE, 0xEF};

  std::string key1 = cache_utils::MakePersistentCacheKey(base_key, ir);
  std::string key2 = cache_utils::MakePersistentCacheKey(base_key, ir);

  EXPECT_EQ(key1, key2);

  std::vector<uint8_t> data1 = {0x11, 0x22};
  ASSERT_TRUE(persistent_cache_->Put(key1, data1));

  // Overwrite with new data using same key
  std::vector<uint8_t> data2 = {0x33, 0x44, 0x55};
  ASSERT_TRUE(persistent_cache_->Put(key2, data2));

  // Should get the latest data (confirming it went to same file)
  auto retrieved = persistent_cache_->Get(key1);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data2);
}

TEST_F(PersistentCacheBackendTest, LargeData) {
  // Test with 1MB of data
  std::vector<uint8_t> data(1024 * 1024);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i % 256);
  }
  std::string key = MakeCacheKey("large_data_test");

  ASSERT_TRUE(persistent_cache_->Put(key, data));

  auto retrieved = persistent_cache_->Get(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data);
}

TEST_F(PersistentCacheBackendTest, EmptyData) {
  std::vector<uint8_t> data;
  std::string key = MakeCacheKey("empty_data_test");

  ASSERT_TRUE(persistent_cache_->Put(key, data));

  // Empty file should return nullopt (size <= 0)
  auto retrieved = persistent_cache_->Get(key);
  EXPECT_FALSE(retrieved.has_value());
}

TEST_F(PersistentCacheBackendTest, Overwrite) {
  std::string key = MakeCacheKey("overwrite_test");
  std::vector<uint8_t> data1 = {0x01, 0x02};
  std::vector<uint8_t> data2 = {0x03, 0x04, 0x05, 0x06};

  ASSERT_TRUE(persistent_cache_->Put(key, data1));
  ASSERT_TRUE(persistent_cache_->Put(key, data2));

  auto retrieved = persistent_cache_->Get(key);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data2);
}

TEST_F(PersistentCacheBackendTest, ConcurrentWrites) {
  const int num_threads = 4;
  const int writes_per_thread = 10;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, t, writes_per_thread]() {
      for (int i = 0; i < writes_per_thread; ++i) {
        std::string key = MakeCacheKey("concurrent_" + std::to_string(t) + "_" + std::to_string(i));
        std::vector<uint8_t> data(100, static_cast<uint8_t>(t * 10 + i));
        EXPECT_TRUE(persistent_cache_->Put(key, data));
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all data was written correctly
  for (int t = 0; t < num_threads; ++t) {
    for (int i = 0; i < writes_per_thread; ++i) {
      std::string key = MakeCacheKey("concurrent_" + std::to_string(t) + "_" + std::to_string(i));
      auto retrieved = persistent_cache_->Get(key);
      ASSERT_TRUE(retrieved.has_value());
      EXPECT_EQ(retrieved.value().size(), 100);
      EXPECT_EQ(retrieved.value()[0], static_cast<uint8_t>(t * 10 + i));
    }
  }
}

TEST_F(PersistentCacheBackendTest, GetCacheDir) {
  EXPECT_EQ(persistent_cache_->GetCacheDir(), test_dir());
}

TEST_F(PersistentCacheBackendTest, Clear_EmptyCache) {
  // Clearing an empty cache should not throw
  persistent_cache_->Clear();
  EXPECT_FALSE(persistent_cache_->Exists("any_key"));
}

TEST_F(PersistentCacheBackendTest, Clear_SingleEntry) {
  std::string key = MakeCacheKey("clear_test_key");
  std::vector<uint8_t> data = {0x01, 0x02, 0x03};

  ASSERT_TRUE(persistent_cache_->Put(key, data));
  ASSERT_TRUE(persistent_cache_->Exists(key));

  persistent_cache_->Clear();

  EXPECT_FALSE(persistent_cache_->Exists(key));
}

TEST_F(PersistentCacheBackendTest, Clear_MultipleEntries) {
  std::vector<std::string> keys = {
      MakeCacheKey("key1"),
      MakeCacheKey("key2"),
      MakeCacheKey("key3"),
      MakeCacheKey("key4"),
  };

  for (const auto& key : keys) {
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    ASSERT_TRUE(persistent_cache_->Put(key, data));
    ASSERT_TRUE(persistent_cache_->Exists(key));
  }

  persistent_cache_->Clear();

  for (const auto& key : keys) {
    EXPECT_FALSE(persistent_cache_->Exists(key));
  }
}

TEST_F(PersistentCacheBackendTest, Clear_RemovesEmptyShardDirectories) {
  // Create entries that will be sharded (keys are hashed internally)
  std::string key1 = "abcd1234test1";
  std::string key2 = "abcd5678test2";
  std::string key3 = "efgh1234test3";

  for (const auto& key : {key1, key2, key3}) {
    std::vector<uint8_t> data = {0x01};
    ASSERT_TRUE(persistent_cache_->Put(key, data));
  }

  // Verify that some shard directories were created
  std::error_code ec;
  size_t dir_count = 0;
  for (auto& entry : std::filesystem::directory_iterator(test_dir(), ec)) {
    if (entry.is_directory()) {
      dir_count++;
    }
  }
  EXPECT_GT(dir_count, 0) << "Expected shard directories to be created";

  persistent_cache_->Clear();

  // All cache entries should be cleared
  EXPECT_FALSE(persistent_cache_->Exists(key1));
  EXPECT_FALSE(persistent_cache_->Exists(key2));
  EXPECT_FALSE(persistent_cache_->Exists(key3));
}

TEST_F(PersistentCacheBackendTest, Clear_ThenAddNewEntries) {
  // Add some entries
  std::string key1 = "before_clear";
  std::vector<uint8_t> data1 = {0x01, 0x02};
  ASSERT_TRUE(persistent_cache_->Put(key1, data1));

  // Clear
  persistent_cache_->Clear();
  EXPECT_FALSE(persistent_cache_->Exists(key1));

  // Add new entries after clear
  std::string key2 = "after_clear";
  std::vector<uint8_t> data2 = {0x03, 0x04, 0x05};
  ASSERT_TRUE(persistent_cache_->Put(key2, data2));

  EXPECT_FALSE(persistent_cache_->Exists(key1));  // Still gone
  EXPECT_TRUE(persistent_cache_->Exists(key2));   // New entry exists

  auto retrieved = persistent_cache_->Get(key2);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved.value(), data2);
}

TEST_F(PersistentCacheBackendTest, Clear_LargeNumberOfEntries) {
  const int num_entries = 100;

  // Add many entries
  for (int i = 0; i < num_entries; ++i) {
    std::string key = "bulk_key_" + std::to_string(i) + "_padding";
    std::vector<uint8_t> data = {static_cast<uint8_t>(i % 256)};
    ASSERT_TRUE(persistent_cache_->Put(key, data));
  }

  // Verify they exist
  for (int i = 0; i < num_entries; ++i) {
    std::string key = "bulk_key_" + std::to_string(i) + "_padding";
    ASSERT_TRUE(persistent_cache_->Exists(key));
  }

  // Clear all
  persistent_cache_->Clear();

  // Verify all are gone
  for (int i = 0; i < num_entries; ++i) {
    std::string key = "bulk_key_" + std::to_string(i) + "_padding";
    EXPECT_FALSE(persistent_cache_->Exists(key));
  }
}

class PersistentCacheBackendEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original env vars
    SaveEnvVar("TORCH_NEURONX_NEFF_CACHE_DIR");
    SaveEnvVar("TORCH_NEURONX_NEFF_DISABLE_CACHE");
  }

  void TearDown() override {
    // Restore original env vars
    RestoreEnvVars();
  }

  void SaveEnvVar(const char* name) {
    const char* value = std::getenv(name);
    if (value) {
      saved_env_vars_[name] = value;
    }
    unsetenv(name);
  }

  void RestoreEnvVars() {
    // First, unset any env vars we may have modified
    unsetenv("TORCH_NEURONX_NEFF_CACHE_DIR");
    unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");

    // Then restore original values
    for (const auto& [name, value] : saved_env_vars_) {
      setenv(name.c_str(), value.c_str(), 1);
    }
  }

  std::map<std::string, std::string> saved_env_vars_;
};

TEST_F(PersistentCacheBackendEnvTest, DefaultConstructor_DefaultDir) {
  unsetenv("TORCH_NEURONX_NEFF_CACHE_DIR");
  PersistentCacheBackend cache;
  EXPECT_EQ(cache.GetCacheDir(), std::filesystem::path("/tmp/neff_cache"));
}

TEST_F(PersistentCacheBackendEnvTest, DefaultConstructor_EnvVarDir) {
  // Create a temp dir to use as the cache directory
  TempDirectory temp_dir("default_constructor_test_");
  setenv("TORCH_NEURONX_NEFF_CACHE_DIR", temp_dir.path().c_str(), 1);
  PersistentCacheBackend cache;
  EXPECT_EQ(cache.GetCacheDir(), temp_dir.fs_path());
}

TEST_F(PersistentCacheBackendEnvTest, IsCachingDisabled_Default) {
  EXPECT_FALSE(PersistentCacheBackend::IsCachingDisabled());
}

TEST_F(PersistentCacheBackendEnvTest, IsCachingDisabled_NeuronEnvVar) {
  setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "1", 1);
  EXPECT_TRUE(PersistentCacheBackend::IsCachingDisabled());
}

}  // namespace
}  // namespace at::neuron
