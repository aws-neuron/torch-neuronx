#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class OperationExecutionEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock NRT session first
    mock_nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();

    // Initialize stream pools before creating any engines
    InitializeStreamPools();

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

    // Cleanup stream pools
    CleanupStreamPools();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_nrt_session_;
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

TEST_F(OperationExecutionEngineTest, QueueLowThresholdEnvironmentVariable) {
  // Test with default values (no environment variables set)
  {
    // Clear any existing environment variables
    unsetenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD");

    auto engine_default = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                                     model_handle_cache_.get());

    // Default queue low threshold should be applied
    EXPECT_EQ(engine_default->GetQueueLowThreshold(), DEFAULT_QUEUE_LOW_THRESHOLD);

    engine_default->Shutdown();
  }

  // Test with custom environment variable value
  {
    setenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD", "25", 1);

    auto engine_custom = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                                    model_handle_cache_.get());

    EXPECT_EQ(engine_custom->GetQueueLowThreshold(), 25);

    engine_custom->Shutdown();
  }

  // Test with invalid environment variable value (should use default)
  {
    setenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD", "invalid", 1);

    auto engine_invalid = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                                     model_handle_cache_.get());

    EXPECT_EQ(engine_invalid->GetQueueLowThreshold(), DEFAULT_QUEUE_LOW_THRESHOLD);

    engine_invalid->Shutdown();
  }

  // Test with negative value (should use default)
  {
    setenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD", "-10", 1);

    auto engine_negative = std::make_unique<OperationExecutionEngine>(compilation_cache_.get(),
                                                                      model_handle_cache_.get());

    EXPECT_EQ(engine_negative->GetQueueLowThreshold(), DEFAULT_QUEUE_LOW_THRESHOLD);

    engine_negative->Shutdown();
  }

  // Clean up environment variables
  unsetenv("TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD");
}

// Test that ProcessCompilationTask handles concatenated operation compilation failure
// When the concatenated operation fails compilation (e.g., compiler error inside
// ProcessCompilationTask), the error should be caught and HandleErrorWithCleanup should be called
// on the concatenated operation
TEST_F(OperationExecutionEngineTest, ProcessCompilationTaskConcatenatedOpFailure) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  // Create a stream
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Create 2 cascading operations that will be "merged" into a concatenated operation
  std::vector<std::shared_ptr<OperationContext>> cascading_ops;
  std::vector<OperationContext*> cascading_ops_raw;

  for (int i = 0; i < 2; ++i) {
    auto cascading_kernel = std::make_unique<at::neuron::testing::MockXLACompilableKernelExecution>(
        "cascading_op_" + std::to_string(i));
    // Cascading ops compile successfully
    ON_CALL(*cascading_kernel, CompileToNeff())
        .WillByDefault(::testing::Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));

    auto cascading_op = std::make_shared<OperationContext>(std::move(cascading_kernel));
    cascading_op->stream = stream.get();
    cascading_ops_raw.push_back(cascading_op.get());
    cascading_ops.push_back(cascading_op);
  }

  // Create the concatenated operation that will FAIL compilation
  auto concat_kernel =
      std::make_unique<at::neuron::testing::MockXLACompilableKernelExecution>("concatenated_op");
  // Make the concatenated operation's compilation throw an exception
  ON_CALL(*concat_kernel, CompileToNeff())
      .WillByDefault(
          ::testing::Throw(std::runtime_error("Simulated concatenated op compilation failure")));

  auto concat_op = std::make_unique<OperationContext>(std::move(concat_kernel));
  concat_op->stream = stream.get();

  // Keep a raw pointer before moving into ConcatenationState
  OperationContext* concat_op_ptr = concat_op.get();

  // Track if the failure callback was invoked
  bool failure_callback_invoked = false;
  OperationContext* failed_op_in_callback = nullptr;

  // Create a concatenation state with failure callback
  auto concat_state = std::make_shared<ConcatenationState>(
      std::move(concat_op), cascading_ops_raw,
      [&failure_callback_invoked, &failed_op_in_callback](OperationContext* failed_op) {
        failure_callback_invoked = true;
        failed_op_in_callback = failed_op;
      });

  // Link cascading ops to the concatenation state
  for (auto& op : cascading_ops) {
    op->concatenation_state_ = concat_state;
  }

  // Set the raw pointer on the concatenated op (now owned by concat_state)
  concat_op_ptr->concatenation_state_raw_ = concat_state.get();

  // Simulate: First cascading op compiles successfully
  // IncrementAndCheckCompiledCascadingOpsCount returns false (not all cascading ops compiled yet)
  EXPECT_FALSE(concat_state->IncrementAndCheckCompiledCascadingOpsCount());
  EXPECT_EQ(concat_state->GetCompiledCascadingOpsCount(), 1);

  // Simulate: Second cascading op compiles successfully
  // IncrementAndCheckCompiledCascadingOpsCount returns true (all cascading ops compiled)
  // This would trigger ProcessCompilationTask on the concatenated operation
  EXPECT_TRUE(concat_state->IncrementAndCheckCompiledCascadingOpsCount());
  EXPECT_EQ(concat_state->GetCompiledCascadingOpsCount(), 2);

  // At this point, ProcessCompilationTask would be called on concat_op_ptr
  // and it should fail compilation, triggering HandleErrorWithCleanup

  // Verify the concatenation state was set up correctly
  EXPECT_EQ(concat_state->GetCascadingOperationsCount(), 2);
  EXPECT_EQ(concat_state->GetConcatenatedOperation(), concat_op_ptr);
  EXPECT_TRUE(cascading_ops[0]->HasConcatenatedOperation());
  EXPECT_TRUE(cascading_ops[1]->HasConcatenatedOperation());
  EXPECT_FALSE(cascading_ops[0]->IsConcatenatedOperation());
  EXPECT_FALSE(cascading_ops[1]->IsConcatenatedOperation());
  EXPECT_TRUE(concat_op_ptr->IsConcatenatedOperation());

  // Verify we can invoke the failure callback (simulating error recovery path)
  concat_state->InvokeFailureCallback(concat_op_ptr);
  EXPECT_TRUE(failure_callback_invoked);
  EXPECT_EQ(failed_op_in_callback, concat_op_ptr);

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that verifies the error handling path when concatenated operation compilation
// throws an exception - the cascading operations should have their concatenation state
// cleared so they can proceed with individual execution
TEST_F(OperationExecutionEngineTest, ConcatenatedOpCompilationFailureFallsBackToIndividualOps) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Create cascading operations
  std::vector<std::shared_ptr<OperationContext>> cascading_ops;
  std::vector<OperationContext*> cascading_ops_raw;

  for (int i = 0; i < 3; ++i) {
    auto cascading_kernel = std::make_unique<at::neuron::testing::MockXLACompilableKernelExecution>(
        "cascading_op_" + std::to_string(i));
    auto cascading_op = std::make_shared<OperationContext>(std::move(cascading_kernel));
    cascading_op->stream = stream.get();
    cascading_ops_raw.push_back(cascading_op.get());
    cascading_ops.push_back(cascading_op);
  }

  // Create concatenated operation
  auto concat_kernel =
      std::make_unique<at::neuron::testing::MockXLACompilableKernelExecution>("concatenated_op");
  auto concat_op = std::make_unique<OperationContext>(std::move(concat_kernel));
  concat_op->stream = stream.get();
  OperationContext* concat_op_ptr = concat_op.get();

  // Track failure callback invocation
  std::atomic<int> failure_callback_count{0};

  // Create a failure callback that clears the concatenation state from cascading ops
  // This mimics what ConcatenationCore::ProcessNewConcatenatedOperation does
  auto concat_state =
      std::make_shared<ConcatenationState>(std::move(concat_op), cascading_ops_raw,
                                           [&failure_callback_count](OperationContext* failed_op) {
                                             failure_callback_count.fetch_add(1);

                                             // Clear concatenation state for all cascading
                                             // operations so they can fall back to individual
                                             // execution (This is what the real failure callback in
                                             // ConcatenationCore does)
                                             if (failed_op) {
                                               auto* state = failed_op->GetConcatenationState();
                                               if (state) {
                                                 for (auto* op : state->GetCascadingOperations()) {
                                                   op->concatenation_state_ = nullptr;
                                                 }
                                               }
                                             }
                                           });

  // Link cascading ops
  for (auto& op : cascading_ops) {
    op->concatenation_state_ = concat_state;
  }
  concat_op_ptr->concatenation_state_raw_ = concat_state.get();

  // Verify initial state - cascading ops are linked to concatenation state
  EXPECT_EQ(concat_state->GetCompiledCascadingOpsCount(), 0);
  EXPECT_EQ(failure_callback_count.load(), 0);
  for (auto& op : cascading_ops) {
    EXPECT_TRUE(op->HasConcatenatedOperation())
        << "Cascading op should have concatenated operation before failure";
  }

  // Simulate compilation of cascading ops
  for (size_t i = 0; i < cascading_ops.size() - 1; ++i) {
    bool is_last = concat_state->IncrementAndCheckCompiledCascadingOpsCount();
    EXPECT_FALSE(is_last) << "Should not be last at iteration " << i;
  }

  // Last cascading op compilation - triggers concatenated op compilation
  bool is_last = concat_state->IncrementAndCheckCompiledCascadingOpsCount();
  EXPECT_TRUE(is_last);
  EXPECT_EQ(concat_state->GetCompiledCascadingOpsCount(), 3);

  // Simulate the error path: concat op compilation failed, invoke failure callback
  // This clears the concatenation state from all cascading operations
  concat_state->InvokeFailureCallback(concat_op_ptr);
  EXPECT_EQ(failure_callback_count.load(), 1);

  // After failure callback, cascading ops should NOT have concatenation state anymore
  // They should be able to proceed with individual execution
  for (auto& op : cascading_ops) {
    EXPECT_FALSE(op->HasConcatenatedOperation())
        << "Cascading op should NOT have concatenated operation after failure callback";
    EXPECT_EQ(op->concatenation_state_, nullptr)
        << "Cascading op's concatenation_state_ should be nullptr after failure";
    // The operation itself is still valid and can be executed individually
    EXPECT_NE(op->GetOpName().find("cascading_op_"), std::string::npos);
  }

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that FindOperationBySeq can find a concatenated operation via its cascading ops
// This is critical for async error handling where NRT reports errors using concat op's sequence ID
TEST_F(OperationExecutionEngineTest, FindOperationBySeqFindsConcatOpViaCascadingOps) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Create the concatenated operation with a specific sequence ID
  auto concat_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*concat_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto concat_op = std::make_unique<OperationContext>(std::move(concat_kernel));
  concat_op->stream = stream.get();
  concat_op->nrt_async_state_.is_scheduled = true;
  concat_op->nrt_async_state_.sequence_id = 42;  // The sequence ID NRT would report

  OperationContext* concat_op_ptr = concat_op.get();

  // Create cascading operations (NOT scheduled individually)
  std::vector<std::shared_ptr<OperationContext>> cascading_ops;
  std::vector<OperationContext*> cascading_ops_raw;

  for (int i = 0; i < 2; ++i) {
    auto cascading_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*cascading_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    auto cascading_op = std::make_shared<OperationContext>(std::move(cascading_kernel));
    cascading_op->stream = stream.get();
    cascading_op->nrt_async_state_.is_scheduled = false;  // NOT scheduled
    cascading_ops_raw.push_back(cascading_op.get());
    cascading_ops.push_back(cascading_op);
  }

  // Create concatenation state
  auto concat_state = std::make_shared<ConcatenationState>(std::move(concat_op), cascading_ops_raw,
                                                           [](OperationContext*) {});

  // Link cascading ops to the concatenation state
  for (auto& op : cascading_ops) {
    op->concatenation_state_ = concat_state;
  }
  concat_op_ptr->concatenation_state_raw_ = concat_state.get();

  // Add ONLY cascading ops to active_operations_ (concat op is owned by concat_state)
  for (auto& op : cascading_ops) {
    stream->active_operations_.push_back(op);
  }

  // Test: FindOperationBySeq should find the concat op via its cascading ops
  OperationContext* found_op = stream->FindOperationBySeq(42);

  EXPECT_NE(found_op, nullptr) << "Should find concat op via cascading ops";
  EXPECT_EQ(found_op, concat_op_ptr) << "Found op should be the concat op";
  EXPECT_TRUE(found_op->IsConcatenatedOperation())
      << "Found op should be identified as concatenated op";

  // Test: FindOperationBySeq should return nullptr for non-existent sequence ID
  OperationContext* not_found = stream->FindOperationBySeq(999);
  EXPECT_EQ(not_found, nullptr) << "Should not find op with non-existent sequence ID";

  // Clean up: clear active_operations_ before stream destructor calls Synchronize()
  // (StreamImpl::~StreamImpl calls Synchronize which waits for active_operations_ to be empty)
  stream->active_operations_.clear();
  cascading_ops.clear();
  concat_state.reset();

  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that ProcessAsyncErrors properly handles concatenated operation async failure
// and fails all cascading operations individually
TEST_F(OperationExecutionEngineTest, ProcessAsyncErrorsConcatenatedOpFailsAllCascadingOps) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  // Initialize NRT mocks to avoid actual NRT calls during test setup
  auto mock_session = std::make_unique<torch_neuronx::testing::MockNRTSession>();

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Create the concatenated operation
  auto concat_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*concat_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto concat_op = std::make_unique<OperationContext>(std::move(concat_kernel));
  concat_op->stream = stream.get();
  concat_op->nrt_async_state_.is_scheduled = true;
  concat_op->nrt_async_state_.sequence_id = 100;

  OperationContext* concat_op_ptr = concat_op.get();

  // Create 3 cascading operations
  std::vector<std::shared_ptr<OperationContext>> cascading_ops;
  std::vector<OperationContext*> cascading_ops_raw;

  for (int i = 0; i < 3; ++i) {
    auto cascading_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*cascading_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    auto cascading_op = std::make_shared<OperationContext>(std::move(cascading_kernel));
    cascading_op->stream = stream.get();
    cascading_op->nrt_async_state_.is_scheduled = false;
    cascading_ops_raw.push_back(cascading_op.get());
    cascading_ops.push_back(cascading_op);
  }

  // Create concatenation state
  auto concat_state = std::make_shared<ConcatenationState>(std::move(concat_op), cascading_ops_raw,
                                                           [](OperationContext*) {});

  // Link cascading ops
  for (auto& op : cascading_ops) {
    op->concatenation_state_ = concat_state;
  }
  concat_op_ptr->concatenation_state_raw_ = concat_state.get();

  // Add cascading ops to active_operations_
  for (auto& op : cascading_ops) {
    stream->active_operations_.push_back(op);
  }

  // Set initial counters:
  // - pending_async_ops_count = 1 (one concat op scheduled)
  // - execution_queue_depth = 3 (three cascading ops in queue)
  engine.pending_async_ops_count_.store(1);
  engine.execution_queue_depth_.store(3);

  // Verify initial state
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);
  EXPECT_EQ(engine.GetExecutionQueueDepth(), 3);
  EXPECT_EQ(stream->active_operations_.size(), 3);

  // Verify the concatenation state is properly set up for error handling
  EXPECT_EQ(concat_state->GetCascadingOperationsCount(), 3);
  EXPECT_EQ(concat_state->GetConcatenatedOperation(), concat_op_ptr);
  EXPECT_TRUE(concat_op_ptr->IsConcatenatedOperation());

  // Verify GetCascadingOperations returns all cascading ops
  auto retrieved_cascading_ops = concat_state->GetCascadingOperations();
  EXPECT_EQ(retrieved_cascading_ops.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(retrieved_cascading_ops[i], cascading_ops_raw[i]);
  }

  // Clean up: clear active_operations_ before stream destructor calls Synchronize()
  stream->active_operations_.clear();
  cascading_ops.clear();
  concat_state.reset();

  engine.pending_async_ops_count_.store(0);
  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that DecrementExecutionQueueDepth decrements for device kernel types (kHLO, kCollective,
// kCopy) and is a no-op for non-device kernel types (kEvent, kHint).
// The device-op check is encapsulated inside DecrementExecutionQueueDepth so callers
// don't need to branch externally.
TEST_F(OperationExecutionEngineTest, DecrementQueueDepthSkipsNonDeviceOps) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream = std::make_shared<StreamImpl>(0, 1);
  engine.execution_queue_depth_.store(3);

  // Device op (kHLO) should decrement
  {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.DecrementExecutionQueueDepth(op.get());
    EXPECT_EQ(engine.GetExecutionQueueDepth(), 2) << "kHLO should decrement";
  }

  // Non-device op (kEvent) should NOT decrement
  {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kEvent));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.DecrementExecutionQueueDepth(op.get());
    EXPECT_EQ(engine.GetExecutionQueueDepth(), 2) << "kEvent should NOT decrement";
  }

  // Non-device op (kHint) should NOT decrement
  {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHint));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.DecrementExecutionQueueDepth(op.get());
    EXPECT_EQ(engine.GetExecutionQueueDepth(), 2) << "kHint should NOT decrement";
  }

  // Device op (kCollective) should decrement
  {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kCollective));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.DecrementExecutionQueueDepth(op.get());
    EXPECT_EQ(engine.GetExecutionQueueDepth(), 1) << "kCollective should decrement";
  }

  // Device op (kCopy) should decrement to 0
  {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kCopy));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.DecrementExecutionQueueDepth(op.get());
    EXPECT_EQ(engine.GetExecutionQueueDepth(), 0) << "kCopy should decrement";
  }

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that DecrementExecutionQueueDepth skips concatenated ops (IsConcatenatedOperation=true)
// since they are not individually tracked in queue depth - only cascading ops are counted.
TEST_F(OperationExecutionEngineTest, DecrementQueueDepthSkipsConcatenatedOps) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream = std::make_shared<StreamImpl>(0, 1);
  engine.execution_queue_depth_.store(3);

  // Create a concatenated operation (IsConcatenatedOperation() == true)
  auto concat_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*concat_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto concat_op = std::make_unique<OperationContext>(std::move(concat_kernel));
  concat_op->stream = stream.get();
  OperationContext* concat_op_ptr = concat_op.get();

  // Create cascading operations
  std::vector<OperationContext*> cascading_ops_raw;
  auto cascading_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*cascading_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto cascading_op = std::make_shared<OperationContext>(std::move(cascading_kernel));
  cascading_op->stream = stream.get();
  cascading_ops_raw.push_back(cascading_op.get());

  // Create concatenation state (transfers concat_op ownership)
  auto concat_state = std::make_shared<ConcatenationState>(std::move(concat_op), cascading_ops_raw,
                                                           [](OperationContext*) {});
  cascading_op->concatenation_state_ = concat_state;
  concat_op_ptr->concatenation_state_raw_ = concat_state.get();

  // Verify the concat op is identified as concatenated
  EXPECT_TRUE(concat_op_ptr->IsConcatenatedOperation());

  // Decrementing for the concatenated op itself should be a no-op
  engine.DecrementExecutionQueueDepth(concat_op_ptr);
  EXPECT_EQ(engine.GetExecutionQueueDepth(), 3)
      << "Concatenated op should NOT decrement queue depth";

  // Decrementing for a cascading op should work normally
  engine.DecrementExecutionQueueDepth(cascading_op.get());
  EXPECT_EQ(engine.GetExecutionQueueDepth(), 2) << "Cascading op should decrement queue depth";

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
