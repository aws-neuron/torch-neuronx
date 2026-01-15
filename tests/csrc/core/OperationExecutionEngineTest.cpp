#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::Return;

class OperationExecutionEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    compilation_cache_ = std::make_unique<CompilationCache>();
    model_handle_cache_ = std::make_unique<ModelHandleCache>();

    // Create engine with our caches
    engine_ = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                         model_handle_cache_.get());

    mock_stream_ = std::make_unique<MockStreamImpl>();
    mock_stream_->SetDefaultBehavior();
  }

  void TearDown() override {
    if (engine_) {
      engine_->Shutdown();
    }
  }

  std::unique_ptr<CompilationCache> compilation_cache_;
  std::unique_ptr<ModelHandleCache> model_handle_cache_;
  std::unique_ptr<OperationExecutionEngine> engine_;
  std::unique_ptr<MockStreamImpl> mock_stream_;
};

// Test engine with null caches
TEST_F(OperationExecutionEngineTest, ConstructionWithNullCaches) {
  // This should work but operations will fail
  auto engine_with_nulls = std::make_unique<OperationExecutionEngine>(nullptr, nullptr);
  EXPECT_NE(engine_with_nulls, nullptr);

  engine_with_nulls->Shutdown();
}

// Test that TaskPriority enum has expected values
TEST_F(OperationExecutionEngineTest, TaskPriorityEnumOrdering) {
  // Verify the enum values are in the expected order
  EXPECT_EQ(static_cast<int>(TaskPriority::HIGH), 0);
  EXPECT_EQ(static_cast<int>(TaskPriority::DEFAULT), 1);
  EXPECT_EQ(static_cast<int>(TaskPriority::LOW), 2);
}

// Test BucketizePriority helper function
TEST_F(OperationExecutionEngineTest, BucketizePriorityFunction) {
  // Test the bucketing logic
  EXPECT_EQ(BucketizePriority(-1), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(-100), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(0), TaskPriority::DEFAULT);
  EXPECT_EQ(BucketizePriority(1), TaskPriority::LOW);
  EXPECT_EQ(BucketizePriority(100), TaskPriority::LOW);

  // Test edge cases
  EXPECT_EQ(BucketizePriority(INT_MIN), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(INT_MAX), TaskPriority::LOW);
}

TEST_F(OperationExecutionEngineTest, MultipleShutdownCallsAreSafe) {
  // First shutdown
  EXPECT_NO_THROW(engine_->Shutdown());

  // Multiple subsequent shutdowns should be no-ops and not crash
  EXPECT_NO_THROW(engine_->Shutdown());
  EXPECT_NO_THROW(engine_->Shutdown());
  EXPECT_NO_THROW(engine_->Shutdown());
}

TEST_F(OperationExecutionEngineTest, NotifyExecutionReadyDoesNotCrash) {
  // Should be safe to call even without pending operations
  EXPECT_NO_THROW(engine_->NotifyExecutionReady());
  EXPECT_NO_THROW(engine_->NotifyExecutionReady());
}

TEST_F(OperationExecutionEngineTest, DestructionWithoutExplicitShutdown) {
  // Create a new engine and let it be destroyed without calling Shutdown()
  auto temp_engine = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                                model_handle_cache_.get());
  EXPECT_NE(temp_engine, nullptr);

  // Destructor should handle shutdown gracefully
  EXPECT_NO_THROW(temp_engine.reset());
}

TEST_F(OperationExecutionEngineTest, ConcurrentNotifyExecutionReady) {
  std::vector<std::thread> threads;

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this]() { engine_->NotifyExecutionReady(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Should complete without issues
  EXPECT_NO_THROW(engine_->Shutdown());
}

TEST_F(OperationExecutionEngineTest, BucketizePriorityBoundaryValues) {
  // Test values around zero boundary
  EXPECT_EQ(BucketizePriority(-1), TaskPriority::HIGH);
  EXPECT_EQ(BucketizePriority(0), TaskPriority::DEFAULT);
  EXPECT_EQ(BucketizePriority(1), TaskPriority::LOW);

  // Verify the enum values are distinct
  EXPECT_NE(TaskPriority::HIGH, TaskPriority::DEFAULT);
  EXPECT_NE(TaskPriority::DEFAULT, TaskPriority::LOW);
  EXPECT_NE(TaskPriority::HIGH, TaskPriority::LOW);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
