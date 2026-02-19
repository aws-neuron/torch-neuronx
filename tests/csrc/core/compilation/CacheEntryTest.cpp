#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"

using namespace at::neuron;

class CacheEntryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test data
    test_neff_data_ = {0x01, 0x02, 0x03, 0x04, 0x05};
    test_persistent_key_ = "test_persistent_key";
    compilation_time_ = std::chrono::milliseconds(100);
  }

  std::vector<uint8_t> test_neff_data_;
  std::string test_persistent_key_;
  std::chrono::milliseconds compilation_time_;
};

TEST_F(CacheEntryTest, DefaultConstruction) {
  CacheEntry entry;

  EXPECT_TRUE(entry.neff_bytes.empty());
  EXPECT_EQ(entry.access_count.load(), 0);
  EXPECT_EQ(entry.compilation_time.count(), 0);
}

TEST_F(CacheEntryTest, ConstructionWithData) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  EXPECT_EQ(entry.neff_bytes, test_neff_data_);
  EXPECT_EQ(entry.access_count.load(), 0);
  EXPECT_EQ(entry.compilation_time, compilation_time_);
  EXPECT_EQ(entry.created_time, entry.last_used_time);  // Should be same initially
}

TEST_F(CacheEntryTest, CopyAndMoveDisabled) {
  // Verify that copy and move operations are disabled
  EXPECT_FALSE(std::is_copy_constructible_v<CacheEntry>);
  EXPECT_FALSE(std::is_copy_assignable_v<CacheEntry>);
  EXPECT_FALSE(std::is_move_constructible_v<CacheEntry>);
  EXPECT_FALSE(std::is_move_assignable_v<CacheEntry>);
}

TEST_F(CacheEntryTest, AccessTracking) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  auto initial_last_used = entry.last_used_time;
  auto initial_access_count = entry.access_count.load();

  EXPECT_EQ(initial_access_count, 0);

  // Wait a bit to ensure time difference
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Update access
  entry.UpdateAccess();

  EXPECT_EQ(entry.access_count.load(), 1);
  EXPECT_GT(entry.last_used_time, initial_last_used);
}

TEST_F(CacheEntryTest, MultipleAccessUpdates) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  // Perform multiple access updates
  for (int i = 1; i <= 5; ++i) {
    entry.UpdateAccess();
    EXPECT_EQ(entry.access_count.load(), i);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Last used time should be more recent than created time
  EXPECT_GT(entry.last_used_time, entry.created_time);
}

TEST_F(CacheEntryTest, ConcurrentAccessUpdates) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  const int num_threads = 10;
  const int accesses_per_thread = 100;

  std::vector<std::thread> threads;

  // Launch threads that update access concurrently
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&entry, accesses_per_thread]() {
      for (int j = 0; j < accesses_per_thread; ++j) {
        entry.UpdateAccess();
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Total access count should be correct
  EXPECT_EQ(entry.access_count.load(), num_threads * accesses_per_thread);
}

TEST_F(CacheEntryTest, CacheEntryGuardDefaultConstruction) {
  CacheEntryGuard guard;

  EXPECT_EQ(guard.cache, nullptr);
  EXPECT_TRUE(guard.cache_key.empty());
}

TEST_F(CacheEntryTest, CacheEntryGuardConstructionWithRealCache) {
  // Create a real CompilationCache for testing
  auto cache = std::make_unique<CompilationCache>();
  std::string test_key = "test_key";

  CacheEntryGuard guard(cache.get(), test_key);

  EXPECT_EQ(guard.cache, cache.get());
  EXPECT_EQ(guard.cache_key, test_key);
}

TEST_F(CacheEntryTest, CacheEntryGuardOperatorCall) {
  // Test that the operator() can be called without crashing
  auto cache = std::make_unique<CompilationCache>();
  std::string test_key = "test_key";

  CacheEntryGuard guard(cache.get(), test_key);

  // Create some dummy data to pass to operator()
  std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03};

  // TODO(rpsilva):  This should not crash and should call mark_entry_inactive internally
  // We can't easily verify the call without mocking, but we can verify it doesn't crash
  EXPECT_NO_THROW(guard(&dummy_data));
}

TEST_F(CacheEntryTest, CacheEntryGuardWithNullCache) {
  // Test behavior with null cache pointer
  CacheEntryGuard guard(nullptr, "test_key");

  std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03};

  // Should handle null cache gracefully
  EXPECT_NO_THROW(guard(&dummy_data));
}

TEST_F(CacheEntryTest, CacheEntryGuardWithEmptyKey) {
  // Test behavior with empty cache key
  auto cache = std::make_unique<CompilationCache>();
  CacheEntryGuard guard(cache.get(), "");

  std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03};

  // Should handle empty key gracefully
  EXPECT_NO_THROW(guard(&dummy_data));
}

TEST_F(CacheEntryTest, NeffBytesPtrBasicFunctionality) {
  // Test that we can create and use NeffBytesPtr
  auto cache = std::make_unique<CompilationCache>();
  std::vector<uint8_t> test_data = {0x01, 0x02, 0x03, 0x04};

  // Create a NeffBytesPtr using the guard
  CacheEntryGuard guard(cache.get(), "test_key");
  NeffBytesPtr neff_ptr(&test_data, std::move(guard));

  // Verify we can access the data
  ASSERT_NE(neff_ptr.get(), nullptr);
  EXPECT_EQ(*neff_ptr, test_data);
  EXPECT_EQ(neff_ptr->size(), test_data.size());

  // Verify the guard is properly moved
  EXPECT_EQ(neff_ptr.get_deleter().cache, cache.get());
  EXPECT_EQ(neff_ptr.get_deleter().cache_key, "test_key");
}

class CacheEntryIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { cache_ = std::make_unique<CompilationCache>(); }

  void TearDown() override { cache_->Clear(); }

  std::unique_ptr<CompilationCache> cache_;
};

TEST_F(CacheEntryIntegrationTest, NeffBytesPtrLifecycle) {
  // TODO(rpsilva): Clean up to use the real construct over mocks
  std::vector<uint8_t> test_data = {0x01, 0x02, 0x03, 0x04};

  // Create a CacheEntry and verify it can be managed properly
  auto entry = std::make_shared<CacheEntry>(test_data, "test_key", std::chrono::milliseconds(50));

  EXPECT_EQ(entry->neff_bytes, test_data);
  EXPECT_EQ(entry->compilation_time.count(), 50);
  EXPECT_EQ(entry->access_count.load(), 0);

  // Simulate access
  entry->UpdateAccess();
  EXPECT_EQ(entry->access_count.load(), 1);
}

class CacheEntryDataSizeTest : public CacheEntryTest,
                               public ::testing::WithParamInterface<size_t> {};

TEST_P(CacheEntryDataSizeTest, DifferentDataSizes) {
  size_t data_size = GetParam();
  std::vector<uint8_t> data(data_size, 0xAB);

  CacheEntry entry(data, "test_key", std::chrono::milliseconds(10));

  EXPECT_EQ(entry.neff_bytes.size(), data_size);
  EXPECT_EQ(entry.neff_bytes, data);
  EXPECT_EQ(entry.compilation_time.count(), 10);
}

INSTANTIATE_TEST_SUITE_P(DataSizeVariants, CacheEntryDataSizeTest,
                         ::testing::Values(0,                // Empty data
                                           1,                // Single byte
                                           1024,             // 1KB
                                           1024 * 1024,      // 1MB
                                           10 * 1024 * 1024  // 10MB
                                           ));

class CacheEntryCompilationTimeTest : public CacheEntryTest,
                                      public ::testing::WithParamInterface<int> {};

TEST_P(CacheEntryCompilationTimeTest, DifferentCompilationTimes) {
  int time_ms = GetParam();
  std::chrono::milliseconds comp_time(time_ms);

  CacheEntry entry(test_neff_data_, test_persistent_key_, comp_time);

  EXPECT_EQ(entry.compilation_time, comp_time);
  EXPECT_EQ(entry.compilation_time.count(), time_ms);
}

INSTANTIATE_TEST_SUITE_P(CompilationTimeVariants, CacheEntryCompilationTimeTest,
                         ::testing::Values(0,     // No compilation time
                                           1,     // 1ms
                                           100,   // 100ms
                                           1000,  // 1 second
                                           10000  // 10 seconds
                                           ));

TEST_F(CacheEntryTest, LargeAccessCount) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  // Set access count to near maximum
  uint64_t large_count = std::numeric_limits<uint64_t>::max() - 10;
  entry.access_count.store(large_count);

  // Should handle overflow gracefully
  entry.UpdateAccess();
  EXPECT_GT(entry.access_count.load(), large_count);
}

TEST_F(CacheEntryTest, TimeConsistency) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  auto created_time = entry.created_time;
  auto initial_last_used = entry.last_used_time;

  // Initially, created_time and last_used_time should be the same
  EXPECT_EQ(created_time, initial_last_used);

  // After access update, last_used should be >= created_time
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  entry.UpdateAccess();

  EXPECT_GE(entry.last_used_time, created_time);
  EXPECT_GT(entry.last_used_time, initial_last_used);
}

TEST_F(CacheEntryTest, EmptyDataHandling) {
  std::vector<uint8_t> empty_data;
  CacheEntry entry(empty_data, "", std::chrono::milliseconds(0));

  EXPECT_TRUE(entry.neff_bytes.empty());
  EXPECT_EQ(entry.neff_bytes.size(), 0);
  EXPECT_EQ(entry.compilation_time.count(), 0);

  // Should still track access properly
  entry.UpdateAccess();
  EXPECT_EQ(entry.access_count.load(), 1);
}

TEST_F(CacheEntryTest, FailedEntryConstruction) {
  auto entry = CacheEntry::Failed("test_key", "compilation error: bad input");

  EXPECT_TRUE(entry->IsFailed());
  EXPECT_EQ(entry->GetFailureMessage(), "compilation error: bad input");
  EXPECT_TRUE(entry->neff_bytes.empty());
  EXPECT_EQ(entry->created_time, entry->last_used_time);
}

TEST_F(CacheEntryTest, SuccessfulEntryIsNotFailed) {
  CacheEntry entry(test_neff_data_, test_persistent_key_, compilation_time_);

  EXPECT_FALSE(entry.IsFailed());
  EXPECT_TRUE(entry.GetFailureMessage().empty());
}

TEST_F(CacheEntryTest, DefaultEntryIsNotFailed) {
  CacheEntry entry;

  EXPECT_FALSE(entry.IsFailed());
  EXPECT_TRUE(entry.GetFailureMessage().empty());
}

TEST_F(CacheEntryTest, FailedEntryAccessTracking) {
  auto entry = CacheEntry::Failed("test_key", "error");

  entry->UpdateAccess();
  EXPECT_EQ(entry->access_count.load(), 1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
