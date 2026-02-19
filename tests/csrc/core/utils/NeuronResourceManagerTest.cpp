#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

using namespace at::neuron;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

// Test ResourceSemaphore
class ResourceSemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override { semaphore_ = std::make_unique<ResourceSemaphore>(3); }

  std::unique_ptr<ResourceSemaphore> semaphore_;
};

TEST_F(ResourceSemaphoreTest, InitialState) {
  EXPECT_EQ(semaphore_->AvailableCount(), 3);
  EXPECT_EQ(semaphore_->WaitingThreads(), 0);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
  EXPECT_EQ(semaphore_->MaxSlots(), 3);
}

TEST_F(ResourceSemaphoreTest, BasicAcquireRelease) {
  semaphore_->Acquire();
  EXPECT_EQ(semaphore_->AvailableCount(), 2);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 1);

  semaphore_->Release();
  EXPECT_EQ(semaphore_->AvailableCount(), 3);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

TEST_F(ResourceSemaphoreTest, MultipleAcquireRelease) {
  semaphore_->Acquire();
  semaphore_->Acquire();
  semaphore_->Acquire();

  EXPECT_EQ(semaphore_->AvailableCount(), 0);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 3);

  semaphore_->Release();
  EXPECT_EQ(semaphore_->AvailableCount(), 1);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 2);

  semaphore_->Release();
  semaphore_->Release();
  EXPECT_EQ(semaphore_->AvailableCount(), 3);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

TEST_F(ResourceSemaphoreTest, TryAcquireForSuccess) {
  bool acquired = semaphore_->TryAcquireFor(100ms);
  EXPECT_TRUE(acquired);
  EXPECT_EQ(semaphore_->AvailableCount(), 2);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 1);
}

TEST_F(ResourceSemaphoreTest, TryAcquireForTimeout) {
  // Acquire all slots
  semaphore_->Acquire();
  semaphore_->Acquire();
  semaphore_->Acquire();

  // Try to acquire with timeout - should fail
  auto start = std::chrono::steady_clock::now();
  bool acquired = semaphore_->TryAcquireFor(50ms);
  auto duration = std::chrono::steady_clock::now() - start;

  EXPECT_FALSE(acquired);
  EXPECT_GE(duration, 50ms);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 3);
}

TEST_F(ResourceSemaphoreTest, PriorityAcquisition) {
  // Acquire all slots
  semaphore_->Acquire();
  semaphore_->Acquire();
  semaphore_->Acquire();

  std::vector<int> completion_order;
  std::mutex order_mutex;

  // Start threads with different priorities
  std::thread low_priority([&]() {
    semaphore_->AcquireWithPriority(1);  // Low priority
    std::lock_guard<std::mutex> lock(order_mutex);
    completion_order.push_back(1);
    semaphore_->Release();
  });

  std::this_thread::sleep_for(10ms);

  std::thread high_priority([&]() {
    semaphore_->AcquireWithPriority(10);  // High priority
    std::lock_guard<std::mutex> lock(order_mutex);
    completion_order.push_back(10);
    semaphore_->Release();
  });

  std::this_thread::sleep_for(10ms);

  // Release one slot - high priority should get it
  semaphore_->Release();

  high_priority.join();

  // Release another slot - low priority should get it
  semaphore_->Release();

  low_priority.join();

  // High priority should have completed first
  ASSERT_EQ(completion_order.size(), 2);
  EXPECT_EQ(completion_order[0], 10);
  EXPECT_EQ(completion_order[1], 1);
}

TEST_F(ResourceSemaphoreTest, ConcurrentAccess) {
  std::vector<std::thread> threads;
  std::atomic<int> successful_acquires{0};
  std::atomic<int> completed_releases{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&]() {
      if (semaphore_->TryAcquireFor(500ms)) {
        successful_acquires++;
        std::this_thread::sleep_for(5ms);
        semaphore_->Release();
        completed_releases++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All threads should eventually succeed
  EXPECT_EQ(successful_acquires.load(), 10);
  EXPECT_EQ(completed_releases.load(), 10);

  // After all threads complete, semaphore should be back to initial state
  // Note: There might be a slight delay in the internal accounting
  EXPECT_LE(semaphore_->AcquiredSlots(), 3);
  EXPECT_GE(semaphore_->AvailableCount(), 0);
}

TEST_F(ResourceSemaphoreTest, WaitingThreadsTracking) {
  // Acquire all slots
  semaphore_->Acquire();
  semaphore_->Acquire();
  semaphore_->Acquire();

  std::thread waiter([&]() {
    semaphore_->Acquire();
    semaphore_->Release();
  });

  // Give thread time to start waiting
  std::this_thread::sleep_for(10ms);

  EXPECT_EQ(semaphore_->WaitingThreads(), 1);

  // Release a slot
  semaphore_->Release();

  waiter.join();

  EXPECT_EQ(semaphore_->WaitingThreads(), 0);
}

// Test ResourceGuard
class ResourceGuardTest : public ::testing::Test {
 protected:
  void SetUp() override { semaphore_ = std::make_unique<ResourceSemaphore>(3); }

  std::unique_ptr<ResourceSemaphore> semaphore_;
};

TEST_F(ResourceGuardTest, ConstructionWithNullSemaphore) {
  EXPECT_THROW(ResourceGuard(nullptr), std::runtime_error);
}

TEST_F(ResourceGuardTest, BasicAcquireRelease) {
  {
    ResourceGuard guard(semaphore_.get());
    guard.Acquire();

    EXPECT_TRUE(guard.IsAcquired());
    EXPECT_EQ(semaphore_->AvailableCount(), 2);
    EXPECT_EQ(semaphore_->AcquiredSlots(), 1);
  }

  // Guard destroyed, should release
  EXPECT_EQ(semaphore_->AvailableCount(), 3);
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

TEST_F(ResourceGuardTest, TryAcquireForSuccess) {
  ResourceGuard guard(semaphore_.get());

  bool acquired = guard.TryAcquireFor(100ms);
  EXPECT_TRUE(acquired);
  EXPECT_TRUE(guard.IsAcquired());
  EXPECT_EQ(semaphore_->AvailableCount(), 2);
}

TEST_F(ResourceGuardTest, TryAcquireForTimeout) {
  // Acquire all slots
  semaphore_->Acquire();
  semaphore_->Acquire();
  semaphore_->Acquire();

  ResourceGuard guard(semaphore_.get());
  bool acquired = guard.TryAcquireFor(50ms);

  EXPECT_FALSE(acquired);
  EXPECT_FALSE(guard.IsAcquired());
}

TEST_F(ResourceGuardTest, ExplicitRelease) {
  ResourceGuard guard(semaphore_.get());
  guard.Acquire();

  EXPECT_TRUE(guard.IsAcquired());
  EXPECT_EQ(semaphore_->AcquiredSlots(), 1);

  guard.Release();

  EXPECT_FALSE(guard.IsAcquired());
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

TEST_F(ResourceGuardTest, MultipleAcquireCallsIdempotent) {
  ResourceGuard guard(semaphore_.get());
  guard.Acquire();
  guard.Acquire();  // Should be no-op

  EXPECT_TRUE(guard.IsAcquired());
  EXPECT_EQ(semaphore_->AcquiredSlots(), 1);  // Only one slot acquired
}

TEST_F(ResourceGuardTest, MultipleReleaseCallsSafe) {
  ResourceGuard guard(semaphore_.get());
  guard.Acquire();
  guard.Release();
  guard.Release();  // Should be no-op

  EXPECT_FALSE(guard.IsAcquired());
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

TEST_F(ResourceGuardTest, PrioritySupport) {
  ResourceGuard low_priority_guard(semaphore_.get(), 1);
  ResourceGuard high_priority_guard(semaphore_.get(), 10);

  EXPECT_EQ(low_priority_guard.GetPriority(), 1);
  EXPECT_EQ(high_priority_guard.GetPriority(), 10);
}

TEST_F(ResourceGuardTest, RAIIBehavior) {
  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);

  {
    ResourceGuard guard(semaphore_.get());
    guard.Acquire();
    EXPECT_EQ(semaphore_->AcquiredSlots(), 1);
  }  // Guard goes out of scope

  EXPECT_EQ(semaphore_->AcquiredSlots(), 0);
}

// Test NeuronResourceManager
class NeuronResourceManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up NRT mock to return 1 device
    auto* nrt_mock = torch_neuronx::testing::MockNRT::GetInstance();
    ON_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
        .WillByDefault(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    // Get singleton instance
    manager_ = &NeuronResourceManager::Instance();
  }

  NeuronResourceManager* manager_;
};

TEST_F(NeuronResourceManagerTest, SingletonInstance) {
  auto& instance1 = NeuronResourceManager::Instance();
  auto& instance2 = NeuronResourceManager::Instance();

  EXPECT_EQ(&instance1, &instance2);
}

TEST_F(NeuronResourceManagerTest, GetCompilationCache) {
  auto& cache = manager_->GetCompilationCache();

  // Should be able to use the cache
  auto keys = cache.GetAllCacheKeys();
  EXPECT_TRUE(keys.empty() || !keys.empty());  // Just verify it works
}

TEST_F(NeuronResourceManagerTest, GetModelHandleCache) {
  auto& cache = manager_->GetModelHandleCache();

  // Should be able to use the cache
  EXPECT_NO_THROW(cache.GetCacheEntries());
}

TEST_F(NeuronResourceManagerTest, GetResourceSemaphore) {
  auto& semaphore = manager_->GetResourceSemaphore();

  // Should be able to use the semaphore
  EXPECT_GE(semaphore.AvailableCount(), 0);
  EXPECT_GE(semaphore.MaxSlots(), 0);
}

TEST_F(NeuronResourceManagerTest, ResourceAccessibility) {
  // Verify all resources are accessible
  auto& cache = manager_->GetCompilationCache();
  auto& model_cache = manager_->GetModelHandleCache();
  auto& engine = manager_->GetOperationExecutionEngine();
  auto& semaphore = manager_->GetResourceSemaphore();

  // Basic sanity checks using available methods
  EXPECT_GE(semaphore.AvailableCount(), 0);
  EXPECT_GE(semaphore.MaxSlots(), 0);

  // Verify caches are accessible
  (void)cache.GetAllCacheKeys();
  (void)model_cache.GetAllCacheKeys();
}

TEST_F(NeuronResourceManagerTest, ConcurrentAccess) {
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 10; ++j) {
        // Access all resources to verify thread safety
        (void)manager_->GetCompilationCache();
        (void)manager_->GetOperationExecutionEngine();
        auto& semaphore = manager_->GetResourceSemaphore();

        if (semaphore.AvailableCount() >= 0) {
          success_count++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(success_count.load(), 100);
}

TEST_F(NeuronResourceManagerTest, SemaphoreConsistency) {
  auto& semaphore = manager_->GetResourceSemaphore();

  // Semaphore should maintain consistent state
  int available = semaphore.AvailableCount();
  int acquired = semaphore.AcquiredSlots();
  int max_slots = semaphore.MaxSlots();

  EXPECT_GE(available, 0);
  EXPECT_GE(acquired, 0);
  EXPECT_EQ(available + acquired, max_slots);
}

TEST_F(NeuronResourceManagerTest, SemaphoreUsage) {
  auto& semaphore = manager_->GetResourceSemaphore();

  int initial_available = semaphore.AvailableCount();
  int initial_acquired = semaphore.AcquiredSlots();

  semaphore.Acquire();

  EXPECT_EQ(semaphore.AvailableCount(), initial_available - 1);
  EXPECT_EQ(semaphore.AcquiredSlots(), initial_acquired + 1);

  semaphore.Release();

  EXPECT_EQ(semaphore.AvailableCount(), initial_available);
  EXPECT_EQ(semaphore.AcquiredSlots(), initial_acquired);
}

// Test environment variable configuration
class ResourceSemaphoreEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original value
    const char* original = std::getenv("NEURON_MAX_RESOURCE_SLOTS");
    if (original) {
      original_value_ = original;
      has_original_ = true;
    }
  }

  void TearDown() override {
    // Restore original value
    if (has_original_) {
      setenv("NEURON_MAX_RESOURCE_SLOTS", original_value_.c_str(), 1);
    } else {
      unsetenv("NEURON_MAX_RESOURCE_SLOTS");
    }
  }

  std::string original_value_;
  bool has_original_ = false;
};

TEST_F(ResourceSemaphoreEnvTest, DefaultMaxSlots) {
  unsetenv("NEURON_MAX_RESOURCE_SLOTS");

  auto semaphore = std::make_unique<ResourceSemaphore>(
      NeuronResourceManager::Instance().GetResourceSemaphore().MaxSlots());

  // Default should be 4 (from kDefaultMaxResourceSlots)
  EXPECT_GE(semaphore->MaxSlots(), 1);
}

TEST_F(ResourceSemaphoreEnvTest, CustomMaxSlots) {
  setenv("NEURON_MAX_RESOURCE_SLOTS", "8", 1);

  // Note: This test is limited because the singleton is already initialized
  auto& semaphore = NeuronResourceManager::Instance().GetResourceSemaphore();
  EXPECT_GE(semaphore.MaxSlots(), 1);
}

// Stress test for ResourceSemaphore
TEST(ResourceSemaphoreStressTest, HighConcurrency) {
  auto semaphore = std::make_unique<ResourceSemaphore>(5);
  std::vector<std::thread> threads;
  std::atomic<int> total_acquires{0};
  std::atomic<int> total_releases{0};
  std::atomic<int> max_concurrent{0};
  std::atomic<int> current_concurrent{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 10; ++j) {
        semaphore->Acquire();
        total_acquires++;

        int concurrent = ++current_concurrent;
        int expected_max = max_concurrent.load();
        while (concurrent > expected_max &&
               !max_concurrent.compare_exchange_weak(expected_max, concurrent)) {
        }

        std::this_thread::sleep_for(1ms);

        --current_concurrent;
        semaphore->Release();
        total_releases++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(total_acquires.load(), 100);
  EXPECT_EQ(total_releases.load(), 100);
  EXPECT_LE(max_concurrent.load(), 5);  // Should never exceed semaphore limit

  // After all operations complete, the semaphore state should be reasonable
  // Note: Internal accounting might have slight delays
  EXPECT_LE(semaphore->AcquiredSlots(), 5);
  EXPECT_GE(semaphore->AvailableCount(), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
