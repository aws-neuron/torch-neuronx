#include <Python.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/allocator/ShelvesCleanup.h"
#include "torch_neuronx/csrc/distributed/NeuronWatchdog.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch_neuronx {
namespace distributed {
namespace {

// =============================================================================
// NeuronWatchdog Tests
// =============================================================================

// Initialize Python once for all tests
class PythonEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
  }
  void TearDown() override {
    // Don't finalize - other things may need it
  }
};

// Register the environment
static ::testing::Environment* const python_env =
    ::testing::AddGlobalTestEnvironment(new PythonEnvironment);

class NeuronWatchdogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize NRT mocks
    nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    auto* nrt_mock = torch_neuronx::testing::MockNRT::GetInstance();

    ON_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
        .WillByDefault(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_allocate_tensor_set(_))
        .WillByDefault(DoAll(SetArgPointee<0>(reinterpret_cast<nrt::TensorSetHandle*>(0x1)),
                             Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_destroy_tensor_set(_)).WillByDefault(Return());
    ON_CALL(*nrt_mock, nrt_add_tensor_to_tensor_set(_, _, _)).WillByDefault(Return(NRT_SUCCESS));
    ON_CALL(*nrt_mock, nrt_execute(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    copy_utils_session_ = std::make_unique<torch_neuronx::utils::testing::MockCopyUtilsSession>();

    c10_neuron::reset_distributed_state();

    InitializeStreamPools();

    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();
    engine_ = mock_engine_wrapper_->GetEngine();
  }

  void TearDown() override {
    mock_engine_wrapper_.reset();
    copy_utils_session_.reset();
    nrt_session_.reset();
    CleanupStreamPools();
  }

  // Helper to create a NeuronWork instance for testing
  c10::intrusive_ptr<NeuronWork> createWork(const std::string& op_type = "test_op",
                                            std::vector<at::Tensor> outputs = {},
                                            float timeout_ms = 300000.0f,
                                            bool blocking_wait = false) {
    at::Device device(c10::DeviceType::PrivateUse1, 0);
    return c10::make_intrusive<NeuronWork>("test_pg_uid", "test_pg_desc", device,
                                           0,  // rank
                                           op_type, seq_num_++, std::move(outputs),
                                           false,  // enable_timing
                                           timeout_ms,
                                           nullptr,  // stream
                                           blocking_wait);
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuronx::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
  OperationExecutionEngine* engine_;
  uint64_t seq_num_ = 0;
};

// =============================================================================
// Basic Lifecycle Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, DefaultConstruction) {
  NeuronWatchdog watchdog;
  // Should not crash on construction
}

TEST_F(NeuronWatchdogTest, DestructorSafeWithoutStart) {
  // Watchdog should be safely destructible without ever calling start()
  NeuronWatchdog watchdog;
  // Destructor runs here - should not hang or crash
}

TEST_F(NeuronWatchdogTest, StartCreatesThread) {
  NeuronWatchdog watchdog;

  watchdog.start();
  // Give thread time to actually start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  watchdog.stop();
  // Should complete without hanging
}

TEST_F(NeuronWatchdogTest, StopWithoutStartIsSafe) {
  NeuronWatchdog watchdog;

  // stop() without start() should be a no-op
  EXPECT_NO_THROW(watchdog.stop());
}

TEST_F(NeuronWatchdogTest, MultipleStopsAreSafe) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // First stop
  EXPECT_NO_THROW(watchdog.stop());

  // Second stop should be safe (no-op)
  EXPECT_NO_THROW(watchdog.stop());

  // Third stop also safe
  EXPECT_NO_THROW(watchdog.stop());
}

TEST_F(NeuronWatchdogTest, StartAfterAlreadyStartedThrows) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // Second start should throw (enforced by TORCH_CHECK in implementation)
  EXPECT_THROW(watchdog.start(), c10::Error);

  watchdog.stop();
}

TEST_F(NeuronWatchdogTest, StopJoinsThreadCleanly) {
  NeuronWatchdog watchdog;
  watchdog.start();

  auto start = std::chrono::steady_clock::now();
  watchdog.stop();
  auto end = std::chrono::steady_clock::now();

  // Stop should complete in reasonable time (not hang)
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  EXPECT_LT(duration.count(), 500);  // Should complete well within 500ms
}

// =============================================================================
// Notification Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, NotifyWithoutStartIsSafe) {
  NeuronWatchdog watchdog;
  // notify() without start() should not crash
  EXPECT_NO_THROW(watchdog.notify());
}

TEST_F(NeuronWatchdogTest, NotifyWakesUpThread) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // notify() should wake up the thread (not block)
  EXPECT_NO_THROW(watchdog.notify());

  watchdog.stop();
}

// =============================================================================
// Work Enqueue Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, EnqueueWorkWithoutStartIsSafe) {
  NeuronWatchdog watchdog;

  auto work = createWork();
  // Should not crash even if watchdog thread isn't running
  EXPECT_NO_THROW(watchdog.enqueueWork(work));
}

TEST_F(NeuronWatchdogTest, EnqueueWorkAddsToWorkList) {
  NeuronWatchdog watchdog;
  watchdog.start();

  auto work = createWork();
  EXPECT_NO_THROW(watchdog.enqueueWork(work));

  watchdog.stop();
}

TEST_F(NeuronWatchdogTest, EnqueueMultipleWorks) {
  NeuronWatchdog watchdog;
  watchdog.start();

  for (int i = 0; i < 10; ++i) {
    auto work = createWork("op_" + std::to_string(i));
    EXPECT_NO_THROW(watchdog.enqueueWork(work));
  }

  watchdog.stop();
}

TEST_F(NeuronWatchdogTest, EnqueueWorkCallsNotify) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // Give the thread time to go into its wait state
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  auto start = std::chrono::steady_clock::now();

  // Enqueue work - this should wake up the thread via notify()
  auto work = createWork();
  watchdog.enqueueWork(work);

  // Stop - if notify worked, thread should wake up quickly
  watchdog.stop();

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should complete quickly, not wait for full sleep interval
  EXPECT_LT(duration.count(), 200);
}

// =============================================================================
// Cleanup Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, CleanupCompletedShelvesCanBeCalledMultipleTimes) {
  NeuronWatchdog watchdog;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_THROW(watchdog.cleanupCompletedShelves());
  }
}

TEST_F(NeuronWatchdogTest, CleanupCompletedShelvesOnEmptyWatchdog) {
  NeuronWatchdog watchdog;
  // Should not crash on empty watchdog
  EXPECT_NO_THROW(watchdog.cleanupCompletedShelves());
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, ConcurrentEnqueue) {
  NeuronWatchdog watchdog;
  watchdog.start();

  std::atomic<int> enqueue_count{0};
  std::vector<std::thread> threads;

  // Multiple threads enqueueing work concurrently
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([this, &watchdog, &enqueue_count]() {
      for (int j = 0; j < 25; ++j) {
        auto work = createWork();
        watchdog.enqueueWork(work);
        enqueue_count++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(enqueue_count.load(), 100);

  watchdog.stop();
}

TEST_F(NeuronWatchdogTest, ConcurrentNotify) {
  NeuronWatchdog watchdog;
  watchdog.start();

  std::vector<std::thread> threads;

  // Multiple threads calling notify concurrently
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&watchdog]() {
      for (int j = 0; j < 50; ++j) {
        watchdog.notify();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  watchdog.stop();
}

TEST_F(NeuronWatchdogTest, EnqueueWhileStopping) {
  NeuronWatchdog watchdog;
  watchdog.start();

  std::atomic<bool> stop_started{false};
  std::atomic<bool> enqueue_done{false};

  // Thread that will try to enqueue work while stop is in progress
  std::thread enqueue_thread([this, &watchdog, &stop_started, &enqueue_done]() {
    // Wait for stop to start
    while (!stop_started.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Try to enqueue work (might succeed or fail depending on timing, but shouldn't crash)
    for (int i = 0; i < 10; ++i) {
      try {
        auto work = createWork();
        watchdog.enqueueWork(work);
      } catch (...) {
        // Ignore exceptions during shutdown
      }
    }
    enqueue_done.store(true);
  });

  // Give enqueue thread time to start waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Signal that stop is starting
  stop_started.store(true);

  // Stop the watchdog
  watchdog.stop();

  // Wait for enqueue thread
  enqueue_thread.join();
}

// =============================================================================
// Completion Detection Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, DetectsCompletedWork) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // Create work and mark it as completed
  auto work = createWork();
  // Force complete by setting exception (simulates completion)
  auto exception = std::make_exception_ptr(std::runtime_error("test"));
  work->setException(exception);

  watchdog.enqueueWork(work);

  // Give watchdog time to process
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  watchdog.stop();
  // Should complete without hanging
}

// =============================================================================
// Timeout Detection Tests
// =============================================================================

TEST_F(NeuronWatchdogTest, DetectsTimedOutWork) {
  NeuronWatchdog watchdog;
  watchdog.start();

  // Create work with very short timeout
  auto work = createWork("timeout_test", {}, 1.0f);  // 1ms timeout

  watchdog.enqueueWork(work);

  // Give watchdog time to detect timeout
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Work should now have an exception set due to timeout
  EXPECT_TRUE(work->isCompleted());

  watchdog.stop();
}

// =============================================================================
// SLEEP_MILLIS Constant Test
// =============================================================================

TEST(NeuronWatchdogConstantsTest, SleepMillisConstant) {
  EXPECT_EQ(NeuronWatchdog::SLEEP_MILLIS, 100);
}

// =============================================================================
// ShelvesCleanupRegistry Tests
// =============================================================================

TEST(ShelvesCleanupRegistryTest, SingletonInstance) {
  auto& instance1 = ShelvesCleanupRegistry::instance();
  auto& instance2 = ShelvesCleanupRegistry::instance();
  EXPECT_EQ(&instance1, &instance2);
}

TEST(ShelvesCleanupRegistryTest, TriggerWithNoWatchdogs) {
  // Should not crash when no watchdogs registered
  EXPECT_NO_THROW(triggerShelvesCleanup());
}

TEST(ShelvesCleanupRegistryTest, WatchdogAutoRegisters) {
  // Creating a watchdog should auto-register it
  // triggerShelvesCleanup should call its cleanupCompletedShelves
  {
    NeuronWatchdog watchdog;
    EXPECT_NO_THROW(triggerShelvesCleanup());
  }
  // After destruction, should still work (watchdog unregistered)
  EXPECT_NO_THROW(triggerShelvesCleanup());
}

TEST(ShelvesCleanupRegistryTest, MultipleWatchdogs) {
  NeuronWatchdog watchdog1;
  NeuronWatchdog watchdog2;
  NeuronWatchdog watchdog3;

  // Should trigger cleanup on all three
  EXPECT_NO_THROW(triggerShelvesCleanup());
}

TEST(ShelvesCleanupRegistryTest, UnregisterDuringCleanup) {
  // Test that unregister waits for cleanup to finish (timeout prevents deadlock)
  auto watchdog = std::make_unique<NeuronWatchdog>();

  std::thread cleanup_thread([]() {
    for (int i = 0; i < 5; ++i) {
      triggerShelvesCleanup();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  // Destroy watchdog while cleanups may be running
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  watchdog.reset();

  cleanup_thread.join();
  // Should complete without crash or hang
}

}  // namespace
}  // namespace distributed
}  // namespace torch_neuronx
