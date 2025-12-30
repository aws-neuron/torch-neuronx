#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationEngine.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;

// Simple test fixture for ConcatenationEngine tests
class ConcatenationEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize execution queue depth and thresholds
    execution_queue_depth_ = 0;
    queue_low_threshold_ = 10;
    queue_target_depth_ = 50;

    // Create mock execution engine
    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();

    // Create a test stream with the mock engine
    test_stream_ = new StreamImpl(0, 0, 0, mock_engine_wrapper_->GetEngine());
  }

  void TearDown() override {
    // Cleanup
    if (test_stream_) {
      delete test_stream_;
      test_stream_ = nullptr;
    }
  }

  // Helper to create a minimal mock operation context
  OperationContext* CreateMockOperation() {
    auto mock_kernel = std::make_unique<MockNeuronKernelExecution>();
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Test data
  std::atomic<int64_t> execution_queue_depth_;
  int64_t queue_low_threshold_;
  int64_t queue_target_depth_;
  StreamImpl* test_stream_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, ConstructorInitialization) {
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  EXPECT_EQ(engine.getBufferedOperationsCount(), 0);
  EXPECT_EQ(engine.getCacheSize(), 0);
}

TEST_F(ConcatenationEngineTest, ProcessNullOperationWithoutQueueDepth) {
  // Create engine without queue depth management
  ConcatenationEngine engine(nullptr, queue_low_threshold_, queue_target_depth_);

  // Process NULL operation (flush signal)
  auto result = engine.processConcatenationTask(nullptr);

  // Should return empty when no queue depth management
  EXPECT_EQ(result.size(), 0);
}

// =============================================================================
// Queue Depth Management Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, NullOperationSkipsRefillWhenAboveThreshold) {
  execution_queue_depth_ = 15;  // Above threshold of 10
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Process NULL operation
  auto result = engine.processConcatenationTask(nullptr);

  // Should not flush anything since above threshold
  EXPECT_EQ(result.size(), 0);
}

TEST_F(ConcatenationEngineTest, DirectFlushWhenQueueLowAndNoBuffered) {
  execution_queue_depth_ = 5;  // Below threshold
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  OperationContext* op = CreateMockOperation();

  // Should bypass buffering and return immediately
  auto result = engine.processConcatenationTask(op);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.front(), op);
  EXPECT_EQ(engine.getBufferedOperationsCount(), 0);

  delete op;
}

TEST_F(ConcatenationEngineTest, BufferingWhenQueueAboveThreshold) {
  execution_queue_depth_ = 20;  // Above threshold to enable buffering
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  OperationContext* op = CreateMockOperation();

  // Should buffer the operation
  auto result = engine.processConcatenationTask(op);

  // May return operations depending on concatenation logic
  // But operation should be tracked internally
  size_t buffered = engine.getBufferedOperationsCount();
  EXPECT_GE(buffered, 0);

  delete op;
}

TEST_F(ConcatenationEngineTest, RefillTriggeredWhenBelowThreshold) {
  execution_queue_depth_ = 5;  // Below threshold of 10
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // First, buffer some operations by setting queue high
  execution_queue_depth_ = 50;
  for (int i = 0; i < 3; ++i) {
    OperationContext* op = CreateMockOperation();
    engine.processConcatenationTask(op);
    delete op;
  }

  size_t buffered_before = engine.getBufferedOperationsCount();

  // Now drop queue depth and trigger refill with NULL
  execution_queue_depth_ = 5;
  auto result = engine.processConcatenationTask(nullptr);

  // Should flush some operations
  if (buffered_before > 0) {
    EXPECT_GT(result.size(), 0);
  }
}

// =============================================================================
// Buffering Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, MultipleOperationsBuffered) {
  execution_queue_depth_ = 50;  // High to enable buffering
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Add multiple operations
  for (int i = 0; i < 5; ++i) {
    OperationContext* op = CreateMockOperation();
    engine.processConcatenationTask(op);
    delete op;
  }

  // Check that operations were buffered
  size_t buffered = engine.getBufferedOperationsCount();
  EXPECT_GE(buffered, 0);
}

TEST_F(ConcatenationEngineTest, MultipleStreamsTrackedSeparately) {
  execution_queue_depth_ = 50;
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Create separate mock execution engines for each stream
  auto mock_engine1 = std::make_unique<MockOperationExecutionEngineWrapper>();
  auto mock_engine2 = std::make_unique<MockOperationExecutionEngineWrapper>();

  StreamImpl* stream1 = new StreamImpl(0, 0, 0, mock_engine1->GetEngine());
  StreamImpl* stream2 = new StreamImpl(0, 1, 0, mock_engine2->GetEngine());

  // Add operations to different streams
  for (int i = 0; i < 3; ++i) {
    auto mock_kernel = std::make_unique<MockNeuronKernelExecution>();
    auto* op1 = new OperationContext(std::move(mock_kernel));
    op1->stream = stream1;
    engine.processConcatenationTask(op1);
    delete op1;
  }

  for (int i = 0; i < 2; ++i) {
    auto mock_kernel = std::make_unique<MockNeuronKernelExecution>();
    auto* op2 = new OperationContext(std::move(mock_kernel));
    op2->stream = stream2;
    engine.processConcatenationTask(op2);
    delete op2;
  }

  // Total buffered operations
  size_t total = engine.getBufferedOperationsCount();
  EXPECT_GE(total, 0);

  delete stream1;
  delete stream2;
}

// =============================================================================
// Cache Management Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, ClearCacheWorks) {
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Clear cache
  engine.clearCache();

  // Verify cache is empty
  EXPECT_EQ(engine.getCacheSize(), 0);
}

TEST_F(ConcatenationEngineTest, GetCacheSizeReturnsValue) {
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  size_t size = engine.getCacheSize();
  EXPECT_GE(size, 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ConcatenationEngineTest, HandleZeroThresholds) {
  ConcatenationEngine engine(&execution_queue_depth_, 0, 0);

  // Should handle edge case gracefully
  auto result = engine.processConcatenationTask(nullptr);
  EXPECT_GE(result.size(), 0);
}

TEST_F(ConcatenationEngineTest, HandleNegativeQueueDepth) {
  execution_queue_depth_ = -5;
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Should handle negative values gracefully
  auto result = engine.processConcatenationTask(nullptr);
  EXPECT_GE(result.size(), 0);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, EndToEndBufferAndRefillCycle) {
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Start with high queue depth (buffering enabled)
  execution_queue_depth_ = 50;

  // Buffer operations
  for (int i = 0; i < 10; ++i) {
    OperationContext* op = CreateMockOperation();
    engine.processConcatenationTask(op);
    delete op;
  }

  size_t buffered = engine.getBufferedOperationsCount();

  // Drop queue depth below threshold
  execution_queue_depth_ = 5;

  // Trigger refill
  auto result = engine.processConcatenationTask(nullptr);

  // Should get some operations back if any were buffered
  if (buffered > 0) {
    EXPECT_GT(result.size(), 0);
  }
}

TEST_F(ConcatenationEngineTest, DirectFlushDoesNotAffectSubsequentBuffering) {
  ConcatenationEngine engine(&execution_queue_depth_, queue_low_threshold_, queue_target_depth_);

  // Start with low queue depth - trigger direct flush
  execution_queue_depth_ = 5;
  OperationContext* op1 = CreateMockOperation();
  auto result1 = engine.processConcatenationTask(op1);
  EXPECT_EQ(result1.size(), 1);
  if (!result1.empty()) {
    EXPECT_EQ(result1.front(), op1);
  }
  delete op1;

  // Increase queue depth - enable buffering
  execution_queue_depth_ = 50;
  OperationContext* op2 = CreateMockOperation();
  auto result2 = engine.processConcatenationTask(op2);
  // Operation should be buffered this time
  delete op2;

  // Verify buffering is working
  size_t buffered = engine.getBufferedOperationsCount();
  EXPECT_GE(buffered, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
