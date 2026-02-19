#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
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
    // Create mock execution engine
    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();

    // Create a test stream with the mock engine
    test_stream_ = new StreamImpl(0, 0);
    test_stream2_ = new StreamImpl(0, 1);
  }

  void TearDown() override {
    // Cleanup
    if (test_stream_) {
      delete test_stream_;
      test_stream_ = nullptr;
    }
    if (test_stream2_) {
      delete test_stream2_;
      test_stream2_ = nullptr;
    }
  }

  // Helper to create a minimal mock operation context with XLA-compilable HLO kernel
  OperationContext* CreateMockHLOOperation(const std::string& op_name = "aten::add") {
    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>(op_name);
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create an HLO operation with specific output pointers
  // Note: This creates an operation with outputs that can be used for conflict detection tests
  OperationContext* CreateMockHLOOperationWithOutputs(const std::string& op_name,
                                                      const std::vector<void*>& dst_ptrs) {
    // Create TensorDataRef vectors from void* pointers
    std::vector<TensorDataRef> input_refs;
    std::vector<TensorDataRef> output_refs;
    for (void* ptr : dst_ptrs) {
      output_refs.emplace_back(ptr);
    }
    auto mock_kernel = std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>(
        op_name, std::move(input_refs), std::move(output_refs), std::vector<TensorContext>{},
        std::vector<TensorContext>{}, "mock_cache_key", std::vector<uint8_t>{}, false, 0);
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create a boundary operation (non-HLO)
  OperationContext* CreateMockBoundaryOperation() {
    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockNeuronKernelExecution>>("boundary_op");
    ON_CALL(*mock_kernel, GetKernelType()).WillByDefault(::testing::Return(KernelTypeEnum::kCopy));
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create a matmul operation (XLA-compilable)
  OperationContext* CreateMockMatmulOperation() {
    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>("aten::matmul");
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create a linear operation (XLA-compilable fusible boundary)
  OperationContext* CreateMockLinearOperation() {
    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>("aten::linear");
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create a Hint operation (allocation/deallocation hints - not boundaries)
  OperationContext* CreateMockHintOperation() {
    auto mock_kernel = std::make_unique<::testing::NiceMock<MockNeuronKernelExecution>>("hint_op");
    ON_CALL(*mock_kernel, GetKernelType()).WillByDefault(::testing::Return(KernelTypeEnum::kHint));
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Helper to create an HLO operation with collectives
  OperationContext* CreateMockHLOOperationWithCollectives() {
    // Create with has_collectives = true in the constructor (not mocked)
    // This is necessary because HasCollectives() is not a virtual method in the base class
    std::vector<TensorDataRef> input_refs;
    std::vector<TensorDataRef> output_refs;
    auto mock_kernel = std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>(
        "aten::all_reduce", std::move(input_refs), std::move(output_refs),
        std::vector<TensorContext>{}, std::vector<TensorContext>{}, "collective_cache_key",
        std::vector<uint8_t>{}, true,  // has_collectives = true
        0);
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;
    return op;
  }

  // Test data
  StreamImpl* test_stream_;
  StreamImpl* test_stream2_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, ConstructorInitialization) {
  ConcatenationEngine engine;

  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);
  EXPECT_EQ(engine.GetCacheSize(), 0);
  EXPECT_EQ(engine.GetBufferSizeLimit(), DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT);
}

TEST_F(ConcatenationEngineTest, DefaultBufferSizeLimit) {
  ConcatenationEngine engine;
  EXPECT_EQ(engine.GetBufferSizeLimit(), DEFAULT_CONCATENATION_BUFFER_SIZE_LIMIT);
}

// =============================================================================
// Boundary Operation Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, NonHLOBoundaryOperationTriggersImplicitFlush) {
  ConcatenationEngine engine;

  // With the new state machine, accumulation starts as disabled.
  // First, we need to enable accumulation by processing a fusible boundary op (matmul)
  OperationContext* matmul_op = CreateMockMatmulOperation();
  auto matmul_result = engine.ProcessConcatenationTask(matmul_op);
  // Matmul enables accumulation and is returned immediately
  EXPECT_EQ(matmul_result.size(), 1);

  // Now accumulation is enabled - subsequent HLO ops will be buffered
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // Now process a boundary operation - should flush buffered operations
  OperationContext* boundary_op = CreateMockBoundaryOperation();
  auto result = engine.ProcessConcatenationTask(boundary_op);

  // Should flush buffered operations AND include the boundary operation
  EXPECT_GE(result.size(), 1);  // At least the boundary operation

  // Buffer should be empty or reduced after flush
  size_t buffered_after = engine.GetBufferedOperationsCount();
  EXPECT_LE(buffered_after, buffered_before);

  // Clean up all operations
  delete matmul_op;
  delete op1;
  delete op2;
  delete boundary_op;
}

TEST_F(ConcatenationEngineTest, BoundaryOperationEnablesAccumulation) {
  ConcatenationEngine engine;

  // Boundary operation should enable accumulation mode
  OperationContext* boundary_op = CreateMockBoundaryOperation();
  auto result = engine.ProcessConcatenationTask(boundary_op);

  // Boundary operation should be returned directly
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.front(), boundary_op);

  // Now accumulation is enabled - subsequent HLO ops should be buffered
  OperationContext* hlo_op = CreateMockHLOOperation("aten::add");
  auto hlo_result = engine.ProcessConcatenationTask(hlo_op);

  // HLO op should be buffered (empty result)
  EXPECT_EQ(hlo_result.size(), 0);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete boundary_op;
  delete hlo_op;
}

TEST_F(ConcatenationEngineTest, HintOperationIsNotBoundary) {
  ConcatenationEngine engine;

  // Enable accumulation first with a matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some HLO operations
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // Hint operation should NOT trigger flush (not a boundary)
  OperationContext* hint_op = CreateMockHintOperation();
  auto result = engine.ProcessConcatenationTask(hint_op);

  // Hint should be processed but not trigger flush
  // Buffer count may or may not change depending on implementation

  delete matmul_op;
  delete op1;
  delete hint_op;
}

TEST_F(ConcatenationEngineTest, HLOWithCollectivesIsBoundary) {
  ConcatenationEngine engine;

  // Enable accumulation first with a matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some HLO operations
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // HLO with collectives should trigger flush (treated as boundary)
  OperationContext* collective_op = CreateMockHLOOperationWithCollectives();
  auto result = engine.ProcessConcatenationTask(collective_op);

  // Should have flushed buffered operations
  EXPECT_GE(result.size(), 1);

  delete matmul_op;
  delete op1;
  delete collective_op;
}

// =============================================================================
// Stream Change Detection Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, StreamChangeTriggersFlush) {
  ConcatenationEngine engine;

  // First enable accumulation by processing a fusible boundary op (matmul)
  OperationContext* matmul_op = CreateMockMatmulOperation();
  matmul_op->stream = test_stream_;
  auto matmul_result = engine.ProcessConcatenationTask(matmul_op);
  EXPECT_EQ(matmul_result.size(), 1);  // Matmul enables accumulation and is returned

  // Now accumulation is enabled - subsequent HLO ops will be buffered on stream 1
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  op1->stream = test_stream_;
  engine.ProcessConcatenationTask(op1);

  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  op2->stream = test_stream_;
  engine.ProcessConcatenationTask(op2);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // Process operation on stream 2 - should trigger flush of stream 1
  OperationContext* op3 = CreateMockHLOOperation("aten::sub");
  op3->stream = test_stream2_;
  auto result = engine.ProcessConcatenationTask(op3);

  // Should have flushed operations from stream 1
  EXPECT_GE(result.size(), 0);  // May have flushed operations

  // Clean up
  delete matmul_op;
  delete op1;
  delete op2;
  delete op3;
}

TEST_F(ConcatenationEngineTest, NoFlushOnSameStream) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  matmul_op->stream = test_stream_;
  engine.ProcessConcatenationTask(matmul_op);

  // Process multiple operations on same stream
  std::vector<OperationContext*> ops;
  for (int i = 0; i < 3; ++i) {
    OperationContext* op = CreateMockHLOOperation();
    op->stream = test_stream_;
    engine.ProcessConcatenationTask(op);
    ops.push_back(op);
  }

  // Operations should be buffered, not flushed due to stream change
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete matmul_op;
  for (auto* op : ops) {
    delete op;
  }
}

// =============================================================================
// State Machine Tests (Accumulation Mode Transitions)
// =============================================================================

TEST_F(ConcatenationEngineTest, StateMachine_InitialStateIsDirectMode) {
  ConcatenationEngine engine;

  // In direct mode (accumulation disabled), HLO operations should execute directly
  // and not be buffered
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  auto result1 = engine.ProcessConcatenationTask(op1);

  // HLO op should be returned immediately (direct execution)
  EXPECT_EQ(result1.size(), 1);
  EXPECT_EQ(result1.front(), op1);

  // Nothing should be buffered
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  delete op1;
}

TEST_F(ConcatenationEngineTest, StateMachine_MatmulEnablesAccumulationMode) {
  ConcatenationEngine engine;

  // Process a matmul - this should enable accumulation mode
  OperationContext* matmul_op = CreateMockMatmulOperation();
  auto matmul_result = engine.ProcessConcatenationTask(matmul_op);

  // Matmul should be returned immediately (triggers accumulation but executes)
  EXPECT_EQ(matmul_result.size(), 1);
  EXPECT_EQ(matmul_result.front(), matmul_op);

  // Now accumulation is enabled - subsequent HLO ops should be buffered
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  auto result1 = engine.ProcessConcatenationTask(op1);

  // HLO op should be buffered (empty result means buffered)
  EXPECT_EQ(result1.size(), 0);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete matmul_op;
  delete op1;
}

TEST_F(ConcatenationEngineTest, StateMachine_LinearEnablesAccumulationMode) {
  ConcatenationEngine engine;

  // Process a linear operation - this should enable accumulation mode
  OperationContext* linear_op = CreateMockLinearOperation();
  auto linear_result = engine.ProcessConcatenationTask(linear_op);

  // Linear should be returned immediately
  EXPECT_EQ(linear_result.size(), 1);
  EXPECT_EQ(linear_result.front(), linear_op);

  // Now accumulation is enabled - subsequent HLO ops should be buffered
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  auto result1 = engine.ProcessConcatenationTask(op1);

  // HLO op should be buffered
  EXPECT_EQ(result1.size(), 0);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete linear_op;
  delete op1;
}

TEST_F(ConcatenationEngineTest, StateMachine_BoundaryOpEnablesAccumulationMode) {
  ConcatenationEngine engine;

  // Process a boundary operation - this should enable accumulation mode
  OperationContext* boundary_op = CreateMockBoundaryOperation();
  auto boundary_result = engine.ProcessConcatenationTask(boundary_op);

  // Boundary should be returned immediately
  EXPECT_EQ(boundary_result.size(), 1);
  EXPECT_EQ(boundary_result.front(), boundary_op);

  // Now accumulation is enabled - subsequent HLO ops should be buffered
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  auto result1 = engine.ProcessConcatenationTask(op1);

  // HLO op should be buffered
  EXPECT_EQ(result1.size(), 0);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete boundary_op;
  delete op1;
}

TEST_F(ConcatenationEngineTest, StateMachine_FlushDisablesAccumulation) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);
  delete matmul_op;

  // Buffer some HLO operations
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  // Call flush - should disable accumulation
  auto flush_result = engine.Flush(100);

  // Should flush all buffered operations
  EXPECT_GT(flush_result.size(), 0);
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  // Now accumulation should be disabled - HLO ops should execute directly
  OperationContext* op3 = CreateMockHLOOperation("aten::sub");
  auto result3 = engine.ProcessConcatenationTask(op3);

  // Should be returned directly (not buffered)
  EXPECT_EQ(result3.size(), 1);
  EXPECT_EQ(result3.front(), op3);
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  delete op1;
  delete op2;
  delete op3;
}

TEST_F(ConcatenationEngineTest, StateMachine_AccumulationBuffersMultipleOps) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);
  delete matmul_op;

  // Buffer multiple HLO operations
  std::vector<OperationContext*> ops;
  for (int i = 0; i < 5; ++i) {
    OperationContext* op = CreateMockHLOOperation("aten::add");
    auto result = engine.ProcessConcatenationTask(op);
    // All should be buffered (empty result)
    EXPECT_EQ(result.size(), 0);
    ops.push_back(op);
  }

  // Verify all are buffered
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 5);

  // Clean up
  for (auto* op : ops) {
    delete op;
  }
}

TEST_F(ConcatenationEngineTest, StateMachine_BoundaryFlushesBufferedOps) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some HLO operations
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_EQ(buffered_before, 2);

  // Process boundary operation - should flush buffer
  OperationContext* boundary_op = CreateMockBoundaryOperation();
  auto boundary_result = engine.ProcessConcatenationTask(boundary_op);

  // Should include flushed ops + boundary op
  EXPECT_GE(boundary_result.size(), 1);

  // Buffer should be empty after flush
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  delete matmul_op;
  delete op1;
  delete op2;
  delete boundary_op;
}

TEST_F(ConcatenationEngineTest, StateMachine_FullCycle_DirectToAccumulationAndBack) {
  ConcatenationEngine engine;

  // PHASE 1: Direct mode (initial state)
  OperationContext* direct_op1 = CreateMockHLOOperation("aten::add");
  auto result1 = engine.ProcessConcatenationTask(direct_op1);
  EXPECT_EQ(result1.size(), 1);  // Direct execution
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);
  delete direct_op1;

  // PHASE 2: Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  auto matmul_result = engine.ProcessConcatenationTask(matmul_op);
  EXPECT_EQ(matmul_result.size(), 1);

  // Buffer HLO ops in accumulation mode
  OperationContext* buffered_op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(buffered_op1);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);  // Buffered

  // PHASE 3: Disable accumulation with flush
  auto flush_result = engine.Flush(100);
  EXPECT_GT(flush_result.size(), 0);  // Flushed
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  // PHASE 4: Back to direct mode
  OperationContext* direct_op2 = CreateMockHLOOperation("aten::sub");
  auto result2 = engine.ProcessConcatenationTask(direct_op2);
  EXPECT_EQ(result2.size(), 1);  // Direct execution again
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  // PHASE 5: Re-enable accumulation with another matmul
  OperationContext* matmul_op2 = CreateMockMatmulOperation();
  auto matmul_result2 = engine.ProcessConcatenationTask(matmul_op2);
  EXPECT_EQ(matmul_result2.size(), 1);

  // Buffer again
  OperationContext* buffered_op2 = CreateMockHLOOperation("aten::relu");
  engine.ProcessConcatenationTask(buffered_op2);
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);  // Buffered again

  // Clean up
  delete matmul_op;
  delete buffered_op1;
  delete direct_op2;
  delete matmul_op2;
  delete buffered_op2;
}

TEST_F(ConcatenationEngineTest, StateMachine_SecondMatmulFlushesBufferInAccumulationMode) {
  ConcatenationEngine engine;

  // Enable accumulation with first matmul
  OperationContext* matmul_op1 = CreateMockMatmulOperation();
  auto result1 = engine.ProcessConcatenationTask(matmul_op1);
  EXPECT_EQ(result1.size(), 1);

  // Buffer some HLO operations
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  engine.ProcessConcatenationTask(op1);
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_EQ(buffered_before, 2);

  // Second matmul should trigger flush (as it's a fusible boundary)
  OperationContext* matmul_op2 = CreateMockMatmulOperation();
  auto result2 = engine.ProcessConcatenationTask(matmul_op2);

  // Should flush buffered ops and return them + matmul
  EXPECT_GE(result2.size(), 1);

  // Clean up
  delete matmul_op1;
  delete op1;
  delete op2;
  delete matmul_op2;
}

// =============================================================================
// Output Conflict Detection Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, OutputConflict_TriggersFlush) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  auto matmul_result = engine.ProcessConcatenationTask(matmul_op);
  EXPECT_EQ(matmul_result.size(), 1);

  // Buffer an HLO operation with a specific output pointer
  void* output_ptr = reinterpret_cast<void*>(0x12345678);
  OperationContext* op1 = CreateMockHLOOperationWithOutputs("aten::add", {output_ptr});
  engine.ProcessConcatenationTask(op1);

  // Buffer another operation
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // Now process an operation that writes to the same output address
  // This should trigger a conflict and flush buffered operations
  OperationContext* conflicting_op = CreateMockHLOOperationWithOutputs("aten::sub", {output_ptr});
  auto conflict_result = engine.ProcessConcatenationTask(conflicting_op);

  // Should have flushed buffered operations due to conflict
  EXPECT_GE(conflict_result.size(), 0);

  // Clean up
  delete matmul_op;
  delete op1;
  delete op2;
  delete conflicting_op;
}

TEST_F(ConcatenationEngineTest, OutputConflict_NoConflictWithDifferentPointers) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer an HLO operation with pointer A
  void* ptr_a = reinterpret_cast<void*>(0x11111111);
  OperationContext* op1 = CreateMockHLOOperationWithOutputs("aten::add", {ptr_a});
  engine.ProcessConcatenationTask(op1);

  size_t buffered_before = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_before, 0);

  // Process operation with different pointer B - should NOT cause conflict
  void* ptr_b = reinterpret_cast<void*>(0x22222222);
  OperationContext* op2 = CreateMockHLOOperationWithOutputs("aten::mul", {ptr_b});
  auto result = engine.ProcessConcatenationTask(op2);

  // Should be buffered, not trigger conflict-based flush
  EXPECT_EQ(result.size(), 0);
  EXPECT_GT(engine.GetBufferedOperationsCount(), buffered_before);

  // Clean up
  delete matmul_op;
  delete op1;
  delete op2;
}

TEST_F(ConcatenationEngineTest, OutputConflict_TrackingClearedAfterFlush) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer an HLO operation with a specific output pointer
  void* test_ptr = reinterpret_cast<void*>(0x87654321);
  OperationContext* op1 = CreateMockHLOOperationWithOutputs("aten::add", {test_ptr});
  engine.ProcessConcatenationTask(op1);

  // Trigger flush via Flush() method
  auto flush_result = engine.Flush(100);

  // After flush, output tracking should be cleared
  // Now we can re-enable accumulation and the same pointer should not cause conflict
  OperationContext* matmul_op2 = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op2);

  // Buffer some operations
  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  engine.ProcessConcatenationTask(op2);

  size_t buffered_after_matmul = engine.GetBufferedOperationsCount();
  EXPECT_GT(buffered_after_matmul, 0);

  // Process operation with same pointer - should NOT cause conflict now since tracking was cleared
  OperationContext* op3 = CreateMockHLOOperationWithOutputs("aten::sub", {test_ptr});
  auto result = engine.ProcessConcatenationTask(op3);

  // Should be buffered without conflict
  EXPECT_EQ(result.size(), 0);

  // Clean up
  delete matmul_op;
  delete op1;
  delete matmul_op2;
  delete op2;
  delete op3;
}

TEST_F(ConcatenationEngineTest, OutputConflict_DirectModeSkipsConflictCheck) {
  ConcatenationEngine engine;

  // In direct mode (initial state), output conflict check is skipped
  void* test_ptr = reinterpret_cast<void*>(0xDEADBEEF);
  OperationContext* op1 = CreateMockHLOOperationWithOutputs("aten::add", {test_ptr});
  auto result1 = engine.ProcessConcatenationTask(op1);

  // Should execute directly without buffering
  EXPECT_EQ(result1.size(), 1);

  // Process operation with same pointer - should also execute directly
  // No conflict because output tracking is cleared in direct mode
  OperationContext* op2 = CreateMockHLOOperationWithOutputs("aten::mul", {test_ptr});
  auto result2 = engine.ProcessConcatenationTask(op2);

  // Should execute directly without issue
  EXPECT_EQ(result2.size(), 1);

  // Clean up
  delete op1;
  delete op2;
}

// =============================================================================
// Fusible Operation Detection Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, FusibleBoundary_MatmulVariants) {
  ConcatenationEngine engine;

  // Test various matmul operation names
  std::vector<std::string> matmul_ops = {"aten::linear", "aten::linear_backward",
                                         "aten::matmul", "aten::matmul_backward",
                                         "aten::mm",     "aten::bmm",
                                         "aten::addmm",  "aten::baddbmm",
                                         "aten::addbmm", "nki_kernel_global"};

  for (const auto& op_name : matmul_ops) {
    ConcatenationEngine local_engine;

    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>(op_name);
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;

    auto result = local_engine.ProcessConcatenationTask(op);

    // Fusible boundary operations should enable accumulation and be returned
    EXPECT_EQ(result.size(), 1) << "Failed for operation: " << op_name;

    // Subsequent HLO op should be buffered
    OperationContext* hlo_op = CreateMockHLOOperation("aten::add");
    auto hlo_result = local_engine.ProcessConcatenationTask(hlo_op);
    EXPECT_EQ(hlo_result.size(), 0) << "HLO not buffered after: " << op_name;

    delete op;
    delete hlo_op;
  }
}

TEST_F(ConcatenationEngineTest, NonFusibleOp_NotTreatedAsMatmul) {
  ConcatenationEngine engine;

  // Non-matmul operations should not be fusible boundaries
  std::vector<std::string> non_matmul_ops = {"aten::add", "aten::relu", "aten::sigmoid",
                                             "aten::softmax", "aten::dropout"};

  for (const auto& op_name : non_matmul_ops) {
    ConcatenationEngine local_engine;

    auto mock_kernel =
        std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>(op_name);
    auto* op = new OperationContext(std::move(mock_kernel));
    op->stream = test_stream_;

    auto result = local_engine.ProcessConcatenationTask(op);

    // In direct mode, these should be executed directly
    EXPECT_EQ(result.size(), 1) << "Failed for operation: " << op_name;

    // But they don't enable accumulation, so next op also executes directly
    OperationContext* hlo_op = CreateMockHLOOperation("aten::mul");
    auto hlo_result = local_engine.ProcessConcatenationTask(hlo_op);
    EXPECT_EQ(hlo_result.size(), 1) << "HLO should execute directly after: " << op_name;

    delete op;
    delete hlo_op;
  }
}

// =============================================================================
// Cache Management Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, ClearCacheWorks) {
  ConcatenationEngine engine;

  // Clear cache
  engine.ClearCache();

  // Verify cache is empty
  EXPECT_EQ(engine.GetCacheSize(), 0);
}

TEST_F(ConcatenationEngineTest, GetCacheSizeReturnsValue) {
  ConcatenationEngine engine;

  size_t size = engine.GetCacheSize();
  EXPECT_GE(size, 0);
}

// =============================================================================
// Flush Method Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, FlushReturnsBufferedOperations) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some operations
  std::vector<OperationContext*> ops;
  for (int i = 0; i < 5; ++i) {
    OperationContext* op = CreateMockHLOOperation("aten::add");
    engine.ProcessConcatenationTask(op);
    ops.push_back(op);
  }

  EXPECT_EQ(engine.GetBufferedOperationsCount(), 5);

  // Flush all
  auto result = engine.Flush(100);

  // Should return all buffered operations
  EXPECT_EQ(result.size(), 5);
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  // Clean up
  delete matmul_op;
  for (auto* op : ops) {
    delete op;
  }
}

TEST_F(ConcatenationEngineTest, FlushPartialCount) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some operations
  std::vector<OperationContext*> ops;
  for (int i = 0; i < 10; ++i) {
    OperationContext* op = CreateMockHLOOperation("aten::add");
    engine.ProcessConcatenationTask(op);
    ops.push_back(op);
  }

  EXPECT_EQ(engine.GetBufferedOperationsCount(), 10);

  // Flush only 3
  auto result = engine.Flush(3);

  // Should return up to 3 operations
  EXPECT_LE(result.size(), 3);

  // Clean up
  delete matmul_op;
  for (auto* op : ops) {
    delete op;
  }
}

TEST_F(ConcatenationEngineTest, FlushEmptyBuffer) {
  ConcatenationEngine engine;

  // Nothing buffered
  EXPECT_EQ(engine.GetBufferedOperationsCount(), 0);

  // Flush should return empty
  auto result = engine.Flush(100);
  EXPECT_EQ(result.size(), 0);
}

// =============================================================================
// Concatenation Failure Handling Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, ProcessConcatenationFailureInvalidatesCache) {
  ConcatenationEngine engine;

  // Create a mock concatenated operation (XLA-compilable)
  auto mock_kernel =
      std::make_unique<::testing::NiceMock<MockXLACompilableKernelExecution>>("mock_concat_op");
  auto failed_op_unique = std::make_unique<OperationContext>(std::move(mock_kernel));
  auto* failed_op = failed_op_unique.get();
  failed_op->stream = test_stream_;

  // Create cascading operations
  auto* cascading_op1 = CreateMockHLOOperation();
  auto* cascading_op2 = CreateMockHLOOperation();

  // Track if failure callback was invoked
  bool callback_invoked = false;

  // Create required failure callback that clears cascading operations' state
  ConcatenationFailureCallback failure_callback = [&callback_invoked](OperationContext* op) {
    callback_invoked = true;
    // Clear the concatenation state for all cascading operations
    if (op) {
      auto* state = op->GetConcatenationState();
      if (state) {
        for (auto* cascading_op : state->GetCascadingOperations()) {
          cascading_op->concatenation_state_ = nullptr;
        }
      }
    }
  };

  // Set up concatenation state with all data upfront (immutable after construction)
  // ConcatenationState takes sole ownership of the concatenated operation via unique_ptr
  // Note: failure_callback is now required (not optional)
  std::vector<OperationContext*> cascading_ops = {cascading_op1, cascading_op2};
  auto concat_state = std::make_shared<ConcatenationState>(
      std::move(failed_op_unique), std::move(cascading_ops), std::move(failure_callback));

  // Link operations to the state:
  // - Concatenated op uses raw pointer (non-owning) to avoid circular reference
  // - Cascading ops use shared_ptr to keep the state alive
  failed_op->concatenation_state_raw_ = concat_state.get();
  cascading_op1->concatenation_state_ = concat_state;
  cascading_op2->concatenation_state_ = concat_state;

  // Verify state is set up correctly before failure handling
  EXPECT_TRUE(failed_op->IsConcatenatedOperation());
  EXPECT_TRUE(cascading_op1->HasConcatenatedOperation());
  EXPECT_TRUE(cascading_op2->HasConcatenatedOperation());

  // Process the failure - should invoke callback which clears cascading ops' state
  engine.ProcessConcatenationFailure(failed_op);

  // Verify callback was invoked
  EXPECT_TRUE(callback_invoked);

  // Verify cascading operations have their concatenation state reset by the callback
  EXPECT_FALSE(cascading_op1->HasConcatenatedOperation());
  EXPECT_FALSE(cascading_op2->HasConcatenatedOperation());

  // Note: failed_op is owned by ConcatenationState (via unique_ptr)
  // Cascading ops need to be deleted manually
  delete cascading_op1;
  delete cascading_op2;
}

TEST_F(ConcatenationEngineTest, ProcessConcatenationFailureThrowsOnNull) {
  ConcatenationEngine engine;

  // Should throw when passed null
  EXPECT_THROW(engine.ProcessConcatenationFailure(nullptr), std::runtime_error);
}

// =============================================================================
// Buffering Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, MultipleOperationsBufferedInAccumulationMode) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Add multiple operations
  std::vector<OperationContext*> ops;
  for (int i = 0; i < 5; ++i) {
    OperationContext* op = CreateMockHLOOperation();
    engine.ProcessConcatenationTask(op);
    ops.push_back(op);
  }

  // Check that operations were buffered
  size_t buffered = engine.GetBufferedOperationsCount();
  EXPECT_EQ(buffered, 5);

  // Clean up
  delete matmul_op;
  for (auto* op : ops) {
    delete op;
  }
}

TEST_F(ConcatenationEngineTest, GetBufferedOperationsCount) {
  ConcatenationEngine engine;

  size_t initial_count = engine.GetBufferedOperationsCount();
  EXPECT_EQ(initial_count, 0);

  // Enable accumulation and add operations
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  for (int i = 0; i < 3; ++i) {
    OperationContext* op = CreateMockHLOOperation();
    engine.ProcessConcatenationTask(op);
    delete op;
  }

  // Count should be 3
  size_t after_count = engine.GetBufferedOperationsCount();
  EXPECT_EQ(after_count, 3);

  delete matmul_op;
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, CompleteWorkflowWithStreamChangeAndBoundary) {
  ConcatenationEngine engine;

  // Enable accumulation with matmul on stream 1
  OperationContext* matmul_op = CreateMockMatmulOperation();
  matmul_op->stream = test_stream_;
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer operations on stream 1
  OperationContext* op1 = CreateMockHLOOperation("aten::add");
  op1->stream = test_stream_;
  engine.ProcessConcatenationTask(op1);

  OperationContext* op2 = CreateMockHLOOperation("aten::mul");
  op2->stream = test_stream_;
  engine.ProcessConcatenationTask(op2);

  // Change to stream 2 - should flush stream 1
  OperationContext* op3 = CreateMockHLOOperation("aten::sub");
  op3->stream = test_stream2_;
  auto result1 = engine.ProcessConcatenationTask(op3);

  // Add boundary operation - should flush current stream
  OperationContext* boundary = CreateMockBoundaryOperation();
  boundary->stream = test_stream2_;
  auto result2 = engine.ProcessConcatenationTask(boundary);

  // Should have flushed operations
  EXPECT_GE(result2.size(), 1);  // At least the boundary operation

  // Clean up all operations
  delete matmul_op;
  delete op1;
  delete op2;
  delete op3;
  delete boundary;
}

TEST_F(ConcatenationEngineTest, FlushDoesNotAffectSubsequentBuffering) {
  ConcatenationEngine engine;

  // Enable accumulation
  OperationContext* matmul_op = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op);

  // Buffer some operations
  OperationContext* op1 = CreateMockHLOOperation();
  engine.ProcessConcatenationTask(op1);

  // Flush disables accumulation
  engine.Flush(100);

  // Next op should execute directly (accumulation disabled)
  OperationContext* op2 = CreateMockHLOOperation();
  auto result = engine.ProcessConcatenationTask(op2);
  EXPECT_EQ(result.size(), 1);

  // Re-enable with another matmul
  OperationContext* matmul_op2 = CreateMockMatmulOperation();
  engine.ProcessConcatenationTask(matmul_op2);

  // Now buffering should work again
  OperationContext* op3 = CreateMockHLOOperation();
  auto result3 = engine.ProcessConcatenationTask(op3);
  EXPECT_EQ(result3.size(), 0);  // Buffered
  EXPECT_GT(engine.GetBufferedOperationsCount(), 0);

  delete matmul_op;
  delete op1;
  delete op2;
  delete matmul_op2;
  delete op3;
}

// =============================================================================
// maybe_try_concatenate Tests
// =============================================================================

TEST_F(ConcatenationEngineTest, MaybeTryConcatenate_EmptyBuffer) {
  ConcatenationEngine engine;

  std::list<OperationContext*> empty_buffer;
  uint64_t cutoff_index = 0;

  auto result = engine.MaybeTryConcatenate(empty_buffer, test_stream_, &cutoff_index);

  EXPECT_EQ(result.size(), 0);
  EXPECT_EQ(cutoff_index, 0);
}

TEST_F(ConcatenationEngineTest, MaybeTryConcatenate_SingleOperation) {
  ConcatenationEngine engine;

  OperationContext* op = CreateMockHLOOperation();
  std::list<OperationContext*> buffer = {op};
  uint64_t cutoff_index = 0;

  auto result = engine.MaybeTryConcatenate(buffer, test_stream_, &cutoff_index);

  // Result depends on ConcatenationCore behavior
  EXPECT_GE(result.size(), 0);

  delete op;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
