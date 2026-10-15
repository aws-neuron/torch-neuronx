#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/concatenation/ConcatenationEngine.h"
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
    c10_neuron::set_compile_only(false);

    // Initialize mock NRT session first
    mock_nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();

    // Configure mock to report 1 visible device so InitializeStreamPools()
    // creates a default stream for device 0. Without this, vnc_count() returns 0
    // and GetDefaultStreamImpl(0) returns nullptr, causing segfaults.
    auto* mock_nrt = torch_neuronx::testing::MockNRT::GetInstance();
    EXPECT_CALL(*mock_nrt, nrt_get_visible_vnc_count(_))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    // Mock NRT instance info for CompilationCacheKey construction
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    // Reset the cached vnc_count so it re-queries the mock
    c10_neuron::reset_vnc_count();

    // Default expectations for async NRT operations used by eventfd-based worker
    EXPECT_CALL(*torch_neuronx::testing::MockNRT::GetInstance(),
                nrta_event_register_seq_id_completion(_, _))
        .WillRepeatedly(::testing::Invoke([](nrta_seq_t, int fd) -> NRT_STATUS {
          uint64_t val = 1;
          ::write(fd, &val, sizeof(val));
          return NRT_SUCCESS;
        }));
    EXPECT_CALL(*torch_neuronx::testing::MockNRT::GetInstance(), nrta_get_sequence(_, _, _, _))
        .WillRepeatedly(DoAll(SetArgPointee<3>(UINT64_MAX), Return(NRT_SUCCESS)));

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
    c10_neuron::set_compile_only(false);

    if (engine_) {
      engine_->Shutdown();
    }

    // Cleanup stream pools
    CleanupStreamPools();

    // Reset the cached vnc_count so it doesn't leak between tests
    c10_neuron::reset_vnc_count();
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

TEST_F(OperationExecutionEngineTest, SubmitWhileRuntimeWorkerParkedMakesProgress) {
  auto* stream = GetDefaultStreamImpl(0);
  ASSERT_NE(stream, nullptr);

  for (nrt::SequenceId sequence = 1; sequence <= 100; ++sequence) {
    // With no queued work or pending completion, this gives RuntimeWorker time
    // to reach its blocking eventfd read before each fresh submission.
    std::this_thread::sleep_for(5ms);

    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
    EXPECT_CALL(*kernel, GetKernelType()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, RequiresCompilation()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, Schedule(_, _, _))
        .WillOnce(DoAll(SetArgPointee<0>(NRT_SUCCESS), SetArgPointee<1>(sequence), Return(true)));

    auto operation = std::make_shared<OperationContext>(std::move(kernel));
    operation->stream = stream;
    operation->stream_id = stream->stream_id;

    engine_->SubmitOperation(operation);
    ASSERT_EQ(operation->result_future.wait_for(2s), std::future_status::ready)
        << "submission did not wake the parked worker at sequence " << sequence;
    auto result = operation->result_future.get();
    ASSERT_TRUE(result.IsSuccess())
        << "submission failed at sequence " << sequence << ": " << result.GetError();
  }
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

  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
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

  // Add ONLY cascading ops to runtime_queue_ (concat op is owned by concat_state)
  for (auto& op : cascading_ops) {
    engine.InjectIntoRuntimeQueue(stream.get(), op);
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

  // Clean up: clear runtime_queue_ before stream destructor calls Synchronize()
  // (StreamImpl::~StreamImpl calls Synchronize which waits for runtime_queue_ to be empty)
  engine.ClearRuntimeQueue(stream.get());
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

  // Add cascading ops to runtime_queue_
  for (auto& op : cascading_ops) {
    engine.InjectIntoRuntimeQueue(stream.get(), op);
  }

  // Set initial counters:
  // - pending_async_ops_count = 1 (one concat op scheduled)
  // - execution_queue_depth = 3 (three cascading ops in queue)
  engine.pending_async_ops_count_.store(1);
  engine.execution_queue_depth_.store(3);

  // Verify initial state
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);
  EXPECT_EQ(engine.GetExecutionQueueDepth(), 3);
  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 3);

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

  // Clean up: clear runtime_queue_ before stream destructor calls Synchronize()
  engine.ClearRuntimeQueue(stream.get());
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

// Test that ProcessCompletions correctly uses seq watermark comparison:
// ops with NRTA_SEQ_GET_SEQ_NUM(seq) <= NRTA_SEQ_GET_SEQ_NUM(watermark) are completed,
// ops above the watermark are not.
//
// ScopedTestEngine swaps the mock in as the GLOBAL engine because
// stream->CompleteOperation → engine.DecrementExecutionQueueDepth runs against
// the global instance.
TEST_F(OperationExecutionEngineTest, ProcessCompletionsUsesSeqWatermark) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Submit 3 ops through the production concat-off path; then mark each as
  // post-schedule with increasing sequence IDs so ProcessCompletions has
  // something to compare against the watermark.
  std::vector<std::shared_ptr<OperationContext>> ops;
  for (int i = 0; i < 3; ++i) {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    engine.SubmitOperation(op);
    op->nrt_async_state_.is_scheduled = true;
    op->nrt_async_state_.sequence_id = static_cast<nrta_seq_t>(i + 1);  // seq 1, 2, 3
    op->nrt_status_ = NRT_SUCCESS;
    ops.push_back(op);
  }

  engine.pending_async_ops_count_.store(3);

  // Create watermark: only seq 1 and 2 are completed
  XUCompletionState completed_seqs;
  size_t xu_idx = GetXUIndex(KernelTypeEnum::kHLO);
  int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
  completed_seqs.Set(static_cast<nrta_xu_t>(xu_idx), queue, 2);  // watermark at seq 2

  // Process completions
  engine.ProcessCompletions(stream.get(), completed_seqs);

  // Ops with seq 1 and 2 should be completed (removed from runtime_queue_)
  // Op with seq 3 should still be pending
  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 1);
  EXPECT_EQ(engine.RuntimeQueueFront(stream.get())->nrt_async_state_.sequence_id, 3u);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);

  // Clean up
  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

// Test that ProcessCompletions stops at the first incomplete op (preserves ordering).
// Even if a later op's seq is below the watermark, it won't be completed if a prior op isn't.
TEST_F(OperationExecutionEngineTest, ProcessCompletionsStopsAtFirstIncomplete) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Op 1: seq 5 (above watermark — incomplete)
  auto kernel1 = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel1, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op1 = std::make_shared<OperationContext>(std::move(kernel1));
  op1->stream = stream.get();
  engine.SubmitOperation(op1);
  op1->nrt_async_state_.is_scheduled = true;
  op1->nrt_async_state_.sequence_id = 5;
  op1->nrt_status_ = NRT_SUCCESS;

  // Op 2: seq 1 (below watermark — but blocked by op1)
  auto kernel2 = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel2, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op2 = std::make_shared<OperationContext>(std::move(kernel2));
  op2->stream = stream.get();
  engine.SubmitOperation(op2);
  op2->nrt_async_state_.is_scheduled = true;
  op2->nrt_async_state_.sequence_id = 1;
  op2->nrt_status_ = NRT_SUCCESS;

  engine.pending_async_ops_count_.store(2);

  // Watermark at seq 3 — op2 (seq 1) would pass, but op1 (seq 5) blocks it
  XUCompletionState completed_seqs;
  size_t xu_idx = GetXUIndex(KernelTypeEnum::kHLO);
  int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
  completed_seqs.Set(static_cast<nrta_xu_t>(xu_idx), queue, 3);

  engine.ProcessCompletions(stream.get(), completed_seqs);

  // Neither op should be completed — op1 blocks op2
  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 2);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 2);

  // Clean up
  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

// Test that pending_signal ops (events/hints) are completed without seq comparison.
TEST_F(OperationExecutionEngineTest, ProcessCompletionsPendingSignalOps) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Create a host op (event/hint — no NRT scheduling)
  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kEvent));
  ON_CALL(*kernel, Schedule(_, _, _)).WillByDefault(Return(true));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  engine.SubmitOperation(op);
  op->nrt_async_state_.is_pending_signal = true;
  op->nrt_async_state_.is_scheduled = false;

  engine.pending_async_ops_count_.store(1);

  // Empty watermark — doesn't matter for host ops
  XUCompletionState completed_seqs;

  engine.ProcessCompletions(stream.get(), completed_seqs);

  // Pending signal op should be completed regardless of watermark
  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 0);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 0);
}

// Test that when watermark is 0 (no completions yet), no ops are completed.
TEST_F(OperationExecutionEngineTest, ProcessCompletionsZeroWatermarkCompletesNothing) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  engine.SubmitOperation(op);
  op->nrt_async_state_.is_scheduled = true;
  op->nrt_async_state_.sequence_id = 1;
  op->nrt_status_ = NRT_SUCCESS;

  engine.pending_async_ops_count_.store(1);

  // Zero watermark — nothing completed
  XUCompletionState completed_seqs;  // all zeros from value-init

  engine.ProcessCompletions(stream.get(), completed_seqs);

  // Op should NOT be completed (seq 1 > watermark 0)
  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 1);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);

  // Clean up
  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

// =============================================================================
// Hint Short-Circuit Tests (RecordHint)
// =============================================================================

// Test that RecordHint is a no-op when concatenation is disabled
TEST_F(OperationExecutionEngineTest, RecordHintNoOpWhenConcatenationDisabled) {
  // Engine created without TORCH_NEURONX_ENABLE_CONCATENATION, so concatenation_enabled_ = false
  EXPECT_FALSE(engine_->IsConcatenationEnabled());

  auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
      "neuron::hint::alloc", reinterpret_cast<void*>(0xDEAD), 1024, 0,
      at::neuron::HintDirectKernelExecution::HintType::kAllocation);
  auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));

  // RecordHint should silently return without crashing
  NeuronStream stream = NeuronStream::getDefaultStream(0);
  EXPECT_NO_THROW(engine_->RecordHint(std::move(hint_ctx), stream));

  // The unique_ptr should have been consumed (moved), but since concat is disabled,
  // RecordHint returns early — the unique_ptr is NOT moved (early return before move)
  // We just verify no crash occurred.
}

// Test that RecordHint is safe to call during shutdown
TEST_F(OperationExecutionEngineTest, RecordHintSafeDuringShutdown) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  // Shutdown first
  engine.Shutdown();

  // RecordHint after shutdown should be a safe no-op
  auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
      "neuron::hint::dealloc", reinterpret_cast<void*>(0x1234), 256, 0,
      at::neuron::HintDirectKernelExecution::HintType::kDeallocation);
  auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));

  NeuronStream stream = NeuronStream::getDefaultStream(0);
  EXPECT_NO_THROW(engine.RecordHint(std::move(hint_ctx), stream));

  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that hints submitted via RecordHint do NOT enter the per-stream runtime
// queue. Hints flow through dispatch_queue_, are consumed by the concatenation
// engine, and dropped at transfer time — they never reach runtime_queue_.
TEST_F(OperationExecutionEngineTest, RecordHintDoesNotEnterRuntimeQueue) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream_impl = GetDefaultStreamImpl(0);  // default stream on device 0

  EXPECT_EQ(engine.RuntimeQueueSize(stream_impl), 0u);

  // Submit multiple hints via RecordHint
  for (int i = 0; i < 5; ++i) {
    auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
        "neuron::hint::alloc", reinterpret_cast<void*>(0x1000 + i * 0x100), 1024, 0,
        at::neuron::HintDirectKernelExecution::HintType::kAllocation);
    auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));

    NeuronStream stream = NeuronStream::getDefaultStream(0);
    engine.RecordHint(std::move(hint_ctx), stream);
  }

  // Give the concatenation worker a moment to process
  std::this_thread::sleep_for(50ms);

  EXPECT_EQ(engine.RuntimeQueueSize(stream_impl), 0u)
      << "Hints submitted via RecordHint must NOT appear in runtime_queue_";

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that concurrent RecordHint calls are thread-safe
TEST_F(OperationExecutionEngineTest, RecordHintThreadSafety) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  std::vector<std::thread> threads;
  constexpr int kThreadCount = 10;
  constexpr int kHintsPerThread = 20;

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&engine, t]() {
      for (int i = 0; i < kHintsPerThread; ++i) {
        auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
            "neuron::hint::alloc", reinterpret_cast<void*>(0x1000 + t * 0x1000 + i * 0x100), 512, 0,
            at::neuron::HintDirectKernelExecution::HintType::kAllocation);
        auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));
        NeuronStream stream = NeuronStream::getDefaultStream(0);
        engine.RecordHint(std::move(hint_ctx), stream);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Should complete without crashes or deadlocks
  EXPECT_NO_THROW(engine.Shutdown());
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that execution_queue_depth is NOT incremented for hint operations
// Hints are not device operations and should not affect queue depth tracking
TEST_F(OperationExecutionEngineTest, RecordHintDoesNotAffectExecutionQueueDepth) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  int64_t initial_depth = engine.GetExecutionQueueDepth();

  // Submit hints
  for (int i = 0; i < 10; ++i) {
    auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
        "neuron::hint::dealloc", reinterpret_cast<void*>(0x2000 + i * 0x100), 256, 0,
        at::neuron::HintDirectKernelExecution::HintType::kDeallocation);
    auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));
    NeuronStream stream = NeuronStream::getDefaultStream(0);
    engine.RecordHint(std::move(hint_ctx), stream);
  }

  // Give concatenation worker time to process
  std::this_thread::sleep_for(100ms);

  // Queue depth should not have changed — hints are filtered out before submitToNextStage
  EXPECT_EQ(engine.GetExecutionQueueDepth(), initial_depth)
      << "Hint operations should not affect execution queue depth";

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// Test that hints are destroyed immediately after consumption by the
// concatenation worker, not accumulated until the queue drains.
TEST_F(OperationExecutionEngineTest, RecordHintDoesNotAccumulateOwnership) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  // Submit a batch of hints
  for (int i = 0; i < 100; ++i) {
    auto hint_kernel = std::make_unique<at::neuron::HintDirectKernelExecution>(
        "neuron::hint::alloc", reinterpret_cast<void*>(0x1000 + i * 0x100), 1024, 0,
        at::neuron::HintDirectKernelExecution::HintType::kAllocation);
    auto hint_ctx = std::make_unique<OperationContext>(std::move(hint_kernel));
    NeuronStream stream = NeuronStream::getDefaultStream(0);
    engine.RecordHint(std::move(hint_ctx), stream);
  }

  // Give the worker time to process all hints
  std::this_thread::sleep_for(200ms);

  // dispatch_queue_ should be empty — each hint destroyed on consumption (its
  // shared_ptr is dropped during transfer-to-runtime since hints never make
  // it to runtime_queue_).
  {
    std::lock_guard<std::mutex> lock(engine.dispatch_mutex_);
    EXPECT_EQ(engine.dispatch_queue_.size(), 0u)
        << "Hints should be destroyed immediately after consumption, not accumulated";
  }

  engine.Shutdown();
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

// --- Same-stream kWait bypass ---

namespace {

std::shared_ptr<OperationContext> MakeKWaitOp(StreamImpl* stream, StreamImpl* recorded_on) {
  NeuronEvent event(/*device_id=*/0);
  event.get_impl()->recorded_stream_impl.store(recorded_on, std::memory_order_relaxed);
  event.get_impl()->is_recorded.store(true, std::memory_order_relaxed);
  auto kernel = std::make_unique<EventDirectKernelExecution>(
      "test_event_wait", event, EventDirectKernelExecution::EventAction::kWait,
      /*device_id=*/0);
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->stream_id = stream->stream_id;
  op->MarkExecutionReady();
  return op;
}

std::shared_ptr<OperationContext> PushPendingDeviceOp(
    StreamImpl* stream, at::neuron::testing::MockOperationExecutionEngine& engine) {
  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->nrt_async_state_.is_scheduled = true;
  op->nrt_async_state_.sequence_id = 1;
  engine.InjectIntoRuntimeQueue(stream, op);
  return op;
}

}  // namespace

TEST_F(OperationExecutionEngineTest, ScheduleOperationKWaitSameStreamBypassesGuard) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);
  PushPendingDeviceOp(stream.get(), engine);
  ASSERT_TRUE(stream->HasPendingAsyncOperations());

  auto kwait = MakeKWaitOp(stream.get(), /*recorded_on=*/stream.get());
  engine.InjectIntoRuntimeQueue(stream.get(), kwait);
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), kwait.get()));
  ASSERT_FALSE(kwait->IsSchedulable());

  EXPECT_TRUE(engine.ScheduleOperation(kwait.get(), stream.get()));
  EXPECT_TRUE(kwait->nrt_async_state_.is_pending_signal);
  EXPECT_TRUE(engine.RuntimeCursorAtEnd(stream.get()));

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

TEST_F(OperationExecutionEngineTest, ScheduleOperationKWaitCrossStreamStillGates) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);
  PushPendingDeviceOp(stream.get(), engine);
  ASSERT_TRUE(stream->HasPendingAsyncOperations());

  // Event recorded on a different stream — must not bypass the gate.
  auto other_stream = std::make_shared<StreamImpl>(0, 2);
  auto kwait = MakeKWaitOp(stream.get(), /*recorded_on=*/other_stream.get());
  engine.InjectIntoRuntimeQueue(stream.get(), kwait);
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), kwait.get()));
  ASSERT_FALSE(kwait->IsSchedulable());

  EXPECT_FALSE(engine.ScheduleOperation(kwait.get(), stream.get()));
  EXPECT_FALSE(kwait->nrt_async_state_.is_pending_signal);
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), kwait.get()));

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

namespace {
// Build a non-compilation device op whose ownership we hand to SubmitOperation.
std::shared_ptr<OperationContext> MakeNonCompileDeviceOp(StreamImpl* stream) {
  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
  EXPECT_CALL(*kernel, RequiresCompilation()).Times(::testing::AnyNumber());
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->stream_id = stream->stream_id;
  return op;
}
}  // namespace

// A non-compilation op must come out of SubmitOperation already published
// to the runtime queue AND marked execution-ready. Because MarkExecutionReady()
// now runs pre-move, the flag is set regardless of whether the worker has since
// raced in — there is no post-move deref to observe a torn state.
TEST_F(OperationExecutionEngineTest, SubmitOperationMarksReadyBeforePublish) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);
  auto op = MakeNonCompileDeviceOp(stream.get());
  OperationContext* op_raw = op.get();

  engine.SubmitOperation(op);

  // The op (or its result) must be reachable: it was published to the queue.
  // We can't assume the worker hasn't drained it, so accept either "still
  // queued" or "already completed", but never "lost".
  EXPECT_TRUE(op_raw->IsExecutionReady())
      << "MarkExecutionReady() must be set pre-move; a lost flag implies the "
         "old post-move ordering.";

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

TEST_F(OperationExecutionEngineTest, SubmitOperationUsesCapturedExecutionRequirement) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);
  c10_neuron::set_compile_only(true);
  auto op = MakeNonCompileDeviceOp(stream.get());
  EXPECT_FALSE(op->RequiresExecution());

  // Submission must use the kernel's captured policy, not the current global mode.
  c10_neuron::set_compile_only(false);
  engine.SubmitOperation(op);

  EXPECT_EQ(engine.RuntimeQueueSize(stream.get()), 0);
  EXPECT_EQ(op->result_future.wait_for(0s), std::future_status::ready);
}

TEST_F(OperationExecutionEngineTest, DispatchReportsNoProgressAtXUCapacity) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  engine.max_inflight_per_xu_ = 2;
  auto stream = std::make_shared<StreamImpl>(0, 1);
  std::vector<std::shared_ptr<OperationContext>> operations;

  auto make_op = [&](nrt::SequenceId sequence_id, bool expect_schedule) {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
    EXPECT_CALL(*kernel, GetKernelType()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, RequiresCompilation()).Times(::testing::AnyNumber());
    if (expect_schedule) {
      EXPECT_CALL(*kernel, Schedule(_, _, _))
          .WillOnce(
              DoAll(SetArgPointee<0>(NRT_SUCCESS), SetArgPointee<1>(sequence_id), Return(true)));
    } else {
      EXPECT_CALL(*kernel, Schedule(_, _, _)).Times(0);
    }

    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    op->stream_id = stream->stream_id;
    op->MarkExecutionReady();
    engine.InjectIntoRuntimeQueue(stream.get(), op);
    return op;
  };

  for (size_t i = 0; i < engine.max_inflight_per_xu_; ++i) {
    auto op = make_op(static_cast<nrt::SequenceId>(i + 1), true);
    ASSERT_TRUE(engine.DispatchAsyncExecutionTask(op.get()));
    ASSERT_TRUE(op->nrt_async_state_.is_scheduled);
    operations.push_back(std::move(op));
  }

  auto blocked = make_op(static_cast<nrt::SequenceId>(engine.max_inflight_per_xu_ + 1), false);
  EXPECT_FALSE(engine.DispatchAsyncExecutionTask(blocked.get()));
  EXPECT_FALSE(blocked->nrt_async_state_.is_scheduled);
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), blocked.get()));
  EXPECT_EQ(engine.pending_async_ops_count_.load(), engine.max_inflight_per_xu_);

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
  size_t xu_idx = GetXUIndex(KernelTypeEnum::kHLO);
  engine.xu_inflight_counts_[0][xu_idx] = 0;
}

TEST_F(OperationExecutionEngineTest, DispatchAtXUCapacitySchedulesAfterCompletion) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  engine.max_inflight_per_xu_ = 2;
  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto make_op = [&](nrt::SequenceId sequence_id) {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
    EXPECT_CALL(*kernel, GetKernelType()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, RequiresCompilation()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, Schedule(_, _, _))
        .WillOnce(
            DoAll(SetArgPointee<0>(NRT_SUCCESS), SetArgPointee<1>(sequence_id), Return(true)));

    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    op->stream_id = stream->stream_id;
    op->MarkExecutionReady();
    engine.InjectIntoRuntimeQueue(stream.get(), op);
    return op;
  };

  std::vector<std::shared_ptr<OperationContext>> inflight;
  for (size_t i = 0; i < engine.max_inflight_per_xu_; ++i) {
    auto op = make_op(static_cast<nrt::SequenceId>(i + 1));
    ASSERT_TRUE(engine.DispatchAsyncExecutionTask(op.get()));
    inflight.push_back(std::move(op));
  }

  auto blocked = make_op(static_cast<nrt::SequenceId>(engine.max_inflight_per_xu_ + 1));
  ASSERT_FALSE(engine.DispatchAsyncExecutionTask(blocked.get()));
  ASSERT_TRUE(engine.RuntimeCursorAt(stream.get(), blocked.get()));

  XUCompletionState completed_seqs;
  size_t xu_idx = GetXUIndex(KernelTypeEnum::kHLO);
  int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
  completed_seqs.Set(static_cast<nrta_xu_t>(xu_idx), queue, 1);
  engine.ProcessCompletions(stream.get(), completed_seqs);

  EXPECT_TRUE(engine.DispatchAsyncExecutionTask(blocked.get()));
  EXPECT_TRUE(blocked->nrt_async_state_.is_scheduled);
  EXPECT_TRUE(engine.RuntimeCursorAtEnd(stream.get()));

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
  engine.xu_inflight_counts_[0][xu_idx] = 0;
}

TEST_F(OperationExecutionEngineTest, DispatchExceptionsReportProgress) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto dispatch_throwing_op = [&](auto exception) {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
    ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
    EXPECT_CALL(*kernel, GetKernelType()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, RequiresCompilation()).Times(::testing::AnyNumber());
    EXPECT_CALL(*kernel, Schedule(_, _, _)).WillOnce(::testing::Throw(exception));

    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    op->stream_id = stream->stream_id;
    op->MarkExecutionReady();
    engine.InjectIntoRuntimeQueue(stream.get(), op);

    EXPECT_TRUE(engine.DispatchAsyncExecutionTask(op.get()));
    EXPECT_TRUE(engine.RuntimeCursorAtEnd(stream.get()));
  };

  dispatch_throwing_op(torch_neuronx::ExecutionRuntimeException("execution failed"));
  dispatch_throwing_op(std::runtime_error("unexpected failure"));
}

TEST_F(OperationExecutionEngineTest, DispatchWithoutCachedNeffWaitsForPendingOperations) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto prior_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*prior_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  ON_CALL(*prior_kernel, RequiresCompilation()).WillByDefault(Return(false));
  EXPECT_CALL(*prior_kernel, GetKernelType()).Times(::testing::AnyNumber());
  EXPECT_CALL(*prior_kernel, RequiresCompilation()).Times(::testing::AnyNumber());
  EXPECT_CALL(*prior_kernel, Schedule(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(NRT_SUCCESS), SetArgPointee<1>(1), Return(true)));
  auto prior_op = std::make_shared<OperationContext>(std::move(prior_kernel));
  prior_op->stream = stream.get();
  prior_op->stream_id = stream->stream_id;
  prior_op->MarkExecutionReady();
  engine.InjectIntoRuntimeQueue(stream.get(), prior_op);
  ASSERT_TRUE(engine.DispatchAsyncExecutionTask(prior_op.get()));

  auto compile_kernel = std::make_unique<at::neuron::testing::MockCompilableKernelExecution>();
  EXPECT_CALL(*compile_kernel, Schedule(_, _, _)).Times(0);
  auto compile_op = std::make_shared<OperationContext>(std::move(compile_kernel));
  compile_op->stream = stream.get();
  compile_op->stream_id = stream->stream_id;
  compile_op->MarkExecutionReady();
  engine.InjectIntoRuntimeQueue(stream.get(), compile_op);

  EXPECT_FALSE(engine.DispatchAsyncExecutionTask(compile_op.get()));
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), compile_op.get()));

  XUCompletionState completed_seqs;
  size_t xu_idx = GetXUIndex(KernelTypeEnum::kHLO);
  int queue = nrt::GetHardwareQueue(static_cast<nrta_xu_t>(xu_idx), stream->stream_id);
  completed_seqs.Set(static_cast<nrta_xu_t>(xu_idx), queue, 1);
  engine.ProcessCompletions(stream.get(), completed_seqs);

  EXPECT_TRUE(engine.DispatchAsyncExecutionTask(compile_op.get()));
  EXPECT_TRUE(engine.RuntimeCursorAtEnd(stream.get()));
  auto result = compile_op->result_future.get();
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("NEFF creation failed"));
}

TEST_F(OperationExecutionEngineTest, AllocScheduleHappyPathNoRetryState) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  // Single inline attempt: sets NRT_SUCCESS.
  EXPECT_CALL(*kernel, Schedule(_, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgPointee<0>(NRT_SUCCESS), Return(false)));

  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  engine.InjectIntoRuntimeQueue(stream.get(), op);

  bool scheduled = engine.ScheduleOperation(op.get(), stream.get());

  EXPECT_TRUE(scheduled);
  EXPECT_TRUE(op->nrt_async_state_.is_pending_signal);
  EXPECT_EQ(op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Fresh);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);

  // Cleanup: clear before stream dtor
  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

TEST_F(OperationExecutionEngineTest, AllocScheduleDeferOnNrtResourceParksOp) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  // First attempt -> NRT_RESOURCE (defer), second -> NRT_SUCCESS after gate clears.
  EXPECT_CALL(*kernel, Schedule(_, _, _))
      .Times(2)
      .WillOnce(DoAll(SetArgPointee<0>(NRT_RESOURCE), Return(false)))
      .WillOnce(DoAll(SetArgPointee<0>(NRT_SUCCESS), Return(false)));

  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  // Prior scheduled op in front so TryDeferAlloc sees drainable work and
  // the "alone-on-stream" escalation does NOT fire.
  auto prior_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*prior_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto prior_op = std::make_shared<OperationContext>(std::move(prior_kernel));
  prior_op->stream = stream.get();
  prior_op->stream_id = stream->stream_id;
  prior_op->nrt_async_state_.is_scheduled = true;
  prior_op->nrt_async_state_.sequence_id = 1;
  engine.InjectIntoRuntimeQueue(stream.get(), prior_op);
  engine.InjectIntoRuntimeQueue(stream.get(), op);

  // First schedule -> defer.
  bool scheduled = engine.ScheduleOperation(op.get(), stream.get());
  EXPECT_FALSE(scheduled);
  EXPECT_EQ(op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Deferred);
  EXPECT_FALSE(op->nrt_async_state_.is_pending_signal);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 0);

  // IsSchedulable must return false while the gate is set.
  op->MarkExecutionReady();
  EXPECT_FALSE(op->IsSchedulable());

  // Simulate stream drain
  op->GetKernel<KernelTypeEnum::kAlloc>().SetAllocState(
      AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_TRUE(op->IsSchedulable());

  // Retry -> success.
  scheduled = engine.ScheduleOperation(op.get(), stream.get());
  EXPECT_TRUE(scheduled);
  EXPECT_TRUE(op->nrt_async_state_.is_pending_signal);
  // State remains RetryReady after a successful retry — the kernel is one-shot.
  EXPECT_EQ(op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 1);

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

TEST_F(OperationExecutionEngineTest, AllocScheduleSecondNrtResourceEscalates) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  // Both attempts hit NRT_RESOURCE.
  EXPECT_CALL(*kernel, Schedule(_, _, _))
      .Times(2)
      .WillRepeatedly(DoAll(SetArgPointee<0>(NRT_RESOURCE), Return(false)));

  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  // Prior scheduled op in front so TryDeferAlloc sees drainable work on the
  // first attempt and flips state to Deferred instead of escalating.
  auto prior_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*prior_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto prior_op = std::make_shared<OperationContext>(std::move(prior_kernel));
  prior_op->stream = stream.get();
  prior_op->stream_id = stream->stream_id;
  prior_op->nrt_async_state_.is_scheduled = true;
  prior_op->nrt_async_state_.sequence_id = 1;
  engine.InjectIntoRuntimeQueue(stream.get(), prior_op);
  engine.InjectIntoRuntimeQueue(stream.get(), op);

  // First schedule -> defer.
  bool scheduled = engine.ScheduleOperation(op.get(), stream.get());
  EXPECT_FALSE(scheduled);
  EXPECT_EQ(op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Deferred);

  // Simulate drain transitioning state to RetryReady.
  op->GetKernel<KernelTypeEnum::kAlloc>().SetAllocState(
      AllocDirectKernelExecution::AllocState::RetryReady);

  // Second schedule -> escalate via exception so ErrorHandler/AttemptOOMRecovery takes over.
  try {
    engine.ScheduleOperation(op.get(), stream.get());
    FAIL() << "Expected ExecutionRuntimeException";
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    EXPECT_TRUE(e.HasNRTStatus());
    EXPECT_EQ(e.GetNRTStatus(), NRT_RESOURCE);
  }

  engine.ClearRuntimeQueue(stream.get());
  engine.pending_async_ops_count_.store(0);
}

TEST_F(OperationExecutionEngineTest, AllocScheduleAloneOnStreamEscalates) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  // Single attempt -> NRT_RESOURCE. No retry should follow since the op is alone.
  EXPECT_CALL(*kernel, Schedule(_, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgPointee<0>(NRT_RESOURCE), Return(false)));

  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  // Only the kAlloc itself on the stream. TryDeferAlloc sees no drainable
  // work and returns false → we escalate with the "no pending ops" reason
  // so AttemptOOMRecovery gets a chance at the allocator cache.
  engine.InjectIntoRuntimeQueue(stream.get(), op);

  try {
    engine.ScheduleOperation(op.get(), stream.get());
    FAIL() << "Expected ExecutionRuntimeException for alone-on-stream kAlloc";
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    EXPECT_TRUE(e.HasNRTStatus());
    EXPECT_EQ(e.GetNRTStatus(), NRT_RESOURCE);
    EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("no pending ops on stream"));
  }

  // No parking: state remains Fresh since TryDeferAlloc didn't flip it.
  EXPECT_EQ(op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Fresh);
  EXPECT_FALSE(op->nrt_async_state_.is_pending_signal);
  EXPECT_EQ(engine.pending_async_ops_count_.load(), 0);

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, RemoveActiveOperationClearsAllocRetryPendingOnDrain) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Parked kAlloc op with AllocState::Deferred. Bypass TryDeferAlloc here, so
  // we mirror the deferred_alloc_count_ bookkeeping it would have done.
  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto alloc_op = std::make_shared<OperationContext>(std::move(alloc_kernel));
  alloc_op->stream = stream.get();
  alloc_op->stream_id = stream->stream_id;
  alloc_op->GetKernel<KernelTypeEnum::kAlloc>().SetAllocState(
      AllocDirectKernelExecution::AllocState::Deferred);
  stream->deferred_alloc_count_.fetch_add(1, std::memory_order_relaxed);
  alloc_op->MarkExecutionReady();

  // Prior scheduled op that will drain, triggering the hook.
  auto prior_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*prior_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto prior_op = std::make_shared<OperationContext>(std::move(prior_kernel));
  prior_op->stream = stream.get();
  prior_op->stream_id = stream->stream_id;

  engine.InjectIntoRuntimeQueue(stream.get(), prior_op);
  engine.InjectIntoRuntimeQueue(stream.get(), alloc_op);

  EXPECT_FALSE(alloc_op->IsSchedulable());

  // Drive the drain hook
  stream->RemoveActiveOperation(prior_op.get());

  EXPECT_EQ(alloc_op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_TRUE(alloc_op->IsSchedulable());

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, GetNextOperationParksOnDeferredKAllocAtHead) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Deferred kAlloc at the head.
  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto deferred_alloc = std::make_shared<OperationContext>(std::move(alloc_kernel));
  deferred_alloc->stream = stream.get();
  deferred_alloc->stream_id = stream->stream_id;
  deferred_alloc->GetKernel<KernelTypeEnum::kAlloc>().SetAllocState(
      AllocDirectKernelExecution::AllocState::Deferred);
  deferred_alloc->MarkExecutionReady();

  auto hlo_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*hlo_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto hlo_op = std::make_shared<OperationContext>(std::move(hlo_kernel));
  hlo_op->stream = stream.get();
  hlo_op->stream_id = stream->stream_id;
  hlo_op->MarkExecutionReady();

  engine.InjectIntoRuntimeQueue(stream.get(), deferred_alloc);
  engine.InjectIntoRuntimeQueue(stream.get(), hlo_op);
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), deferred_alloc.get()));

  EXPECT_EQ(stream->GetNextOperationToSchedule(), nullptr);
  // Cursor stays on the Deferred kAlloc — it does not advance.
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), deferred_alloc.get()));

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, GetNextOperationDoesNotSkipNonDeferredBlocker) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Non-schedulable, non-Deferred op at the head (kHLO with execution_ready=false).
  auto head_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*head_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto head_op = std::make_shared<OperationContext>(std::move(head_kernel));
  head_op->stream = stream.get();
  head_op->stream_id = stream->stream_id;
  // Deliberately NOT MarkExecutionReady() — IsSchedulable returns false.

  // Schedulable op behind it.
  auto back_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*back_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto back_op = std::make_shared<OperationContext>(std::move(back_kernel));
  back_op->stream = stream.get();
  back_op->stream_id = stream->stream_id;
  back_op->MarkExecutionReady();

  engine.InjectIntoRuntimeQueue(stream.get(), head_op);
  engine.InjectIntoRuntimeQueue(stream.get(), back_op);
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), head_op.get()));

  // Scheduler must return nullptr (head is a real blocker); back_op must not
  // be hoisted past the head.
  EXPECT_EQ(stream->GetNextOperationToSchedule(), nullptr);
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), head_op.get()));

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, AdvanceToNextOperationIsIdempotent) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Two ops; runtime cursor at the first.
  auto op1_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*op1_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op1 = std::make_shared<OperationContext>(std::move(op1_kernel));
  op1->stream = stream.get();
  op1->stream_id = stream->stream_id;

  auto op2_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*op2_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op2 = std::make_shared<OperationContext>(std::move(op2_kernel));
  op2->stream = stream.get();
  op2->stream_id = stream->stream_id;

  engine.InjectIntoRuntimeQueue(stream.get(), op1);
  engine.InjectIntoRuntimeQueue(stream.get(), op2);
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), op1.get()));

  stream->RemoveActiveOperation(op1.get());
  ASSERT_TRUE(engine.RuntimeCursorAt(stream.get(), op2.get()));

  // Re-advancing without a fresh op shouldn't mis-step the cursor.
  stream->AdvanceToNextOperation();
  EXPECT_TRUE(engine.RuntimeCursorAtEnd(stream.get()));

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, TryDeferAllocReturnsTrueWhenDrainableWorkExists) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Scheduled op that hasn't completed yet.
  auto inflight_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*inflight_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto inflight_op = std::make_shared<OperationContext>(std::move(inflight_kernel));
  inflight_op->stream = stream.get();
  inflight_op->stream_id = stream->stream_id;
  inflight_op->nrt_async_state_.is_scheduled = true;

  // Fresh kAlloc behind it.
  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto alloc_op = std::make_shared<OperationContext>(std::move(alloc_kernel));
  alloc_op->stream = stream.get();
  alloc_op->stream_id = stream->stream_id;
  ASSERT_EQ(alloc_op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Fresh);

  engine.InjectIntoRuntimeQueue(stream.get(), inflight_op);
  engine.InjectIntoRuntimeQueue(stream.get(), alloc_op);

  EXPECT_TRUE(stream->TryDeferAlloc(alloc_op->GetKernel<KernelTypeEnum::kAlloc>()));
  EXPECT_EQ(alloc_op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Deferred);

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, TryDeferAllocReturnsFalseWhenStreamDrained) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto alloc_op = std::make_shared<OperationContext>(std::move(alloc_kernel));
  alloc_op->stream = stream.get();
  alloc_op->stream_id = stream->stream_id;

  engine.InjectIntoRuntimeQueue(stream.get(), alloc_op);

  EXPECT_FALSE(stream->TryDeferAlloc(alloc_op->GetKernel<KernelTypeEnum::kAlloc>()));
  EXPECT_EQ(alloc_op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::Fresh)
      << "TryDeferAlloc must not mutate state on the no-defer path";

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, MultipleDeferredKAllocsDrainRewindsToEarliest) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Three Deferred kAllocs in order, then a single in-flight op that keeps
  // the stream from being considered drained until it's removed.
  auto make_deferred = [&]() {
    auto k = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
    auto op = std::make_shared<OperationContext>(std::move(k));
    op->stream = stream.get();
    op->stream_id = stream->stream_id;
    op->GetKernel<KernelTypeEnum::kAlloc>().SetAllocState(
        AllocDirectKernelExecution::AllocState::Deferred);
    // Bypass TryDeferAlloc; mirror its deferred_alloc_count_ bookkeeping.
    stream->deferred_alloc_count_.fetch_add(1, std::memory_order_relaxed);
    op->MarkExecutionReady();
    return op;
  };
  auto a = make_deferred();
  auto b = make_deferred();
  auto c = make_deferred();

  auto inflight_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*inflight_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto inflight = std::make_shared<OperationContext>(std::move(inflight_kernel));
  inflight->stream = stream.get();
  inflight->stream_id = stream->stream_id;
  inflight->nrt_async_state_.is_scheduled = true;

  engine.InjectIntoRuntimeQueue(stream.get(), a);
  engine.InjectIntoRuntimeQueue(stream.get(), b);
  engine.InjectIntoRuntimeQueue(stream.get(), c);
  engine.InjectIntoRuntimeQueue(stream.get(), inflight);
  // Simulate scheduler having scanned past all three Deferred kAllocs to
  // reach `inflight` on a previous pass — cursor is now on `inflight`.
  ASSERT_TRUE(engine.SetRuntimeCursorTo(stream.get(), inflight.get()));

  // Drain: remove the inflight op. All three Deferred kAllocs flip to
  // RetryReady, and the cursor rewinds to `a`.
  stream->RemoveActiveOperation(inflight.get());

  EXPECT_EQ(a->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_EQ(b->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_EQ(c->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);
  EXPECT_TRUE(engine.RuntimeCursorAt(stream.get(), a.get()))
      << "rewind should land on the earliest RetryReady kAlloc, not a later one";

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, DeferredAllocCountStaysZeroInSteadyState) {
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // A scheduled kHLO op being completed.
  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream.get();
  op->stream_id = stream->stream_id;
  op->nrt_async_state_.is_scheduled = true;
  engine.InjectIntoRuntimeQueue(stream.get(), op);

  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0);

  stream->RemoveActiveOperation(op.get());

  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0)
      << "counter must stay 0 in steady state";
}

TEST_F(OperationExecutionEngineTest, DeferredAllocCountTracksTryDeferAndDrain) {
  // TryDeferAlloc must increment; ReleaseDeferredAllocGates flip-out must
  // decrement; counter returns to 0 after a full park-then-drain cycle.
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  // Prior scheduled op so TryDeferAlloc sees drainable work.
  auto prior_kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*prior_kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  auto prior_op = std::make_shared<OperationContext>(std::move(prior_kernel));
  prior_op->stream = stream.get();
  prior_op->stream_id = stream->stream_id;
  prior_op->nrt_async_state_.is_scheduled = true;

  // Fresh kAlloc behind it.
  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto alloc_op = std::make_shared<OperationContext>(std::move(alloc_kernel));
  alloc_op->stream = stream.get();
  alloc_op->stream_id = stream->stream_id;
  alloc_op->MarkExecutionReady();

  engine.InjectIntoRuntimeQueue(stream.get(), prior_op);
  engine.InjectIntoRuntimeQueue(stream.get(), alloc_op);

  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0);

  // Park the kAlloc: counter -> 1.
  EXPECT_TRUE(stream->TryDeferAlloc(alloc_op->GetKernel<KernelTypeEnum::kAlloc>()));
  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 1);

  // Drain: removing the prior op triggers ReleaseDeferredAllocGates which
  // flips the kAlloc to RetryReady and decrements the counter.
  stream->RemoveActiveOperation(prior_op.get());
  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(alloc_op->GetKernel<KernelTypeEnum::kAlloc>().GetAllocState(),
            AllocDirectKernelExecution::AllocState::RetryReady);

  // Removing the now-RetryReady alloc op must NOT underflow the counter.
  // The drain hook should early-return on the zero counter.
  stream->RemoveActiveOperation(alloc_op.get());
  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0)
      << "second RemoveActiveOperation must not underflow the counter";

  engine.ClearRuntimeQueue(stream.get());
}

TEST_F(OperationExecutionEngineTest, DeferredAllocCountUnchangedWhenTryDeferFails) {
  // TryDeferAlloc must NOT increment the counter when it returns false
  // (no drainable work). State stays Fresh, counter stays 0.
  at::neuron::testing::ScopedTestEngine scoped(compilation_cache_.get(), model_handle_cache_.get());
  auto& engine = scoped.engine();
  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto alloc_kernel = std::make_unique<at::neuron::testing::MockAllocDirectKernelExecution>();
  auto alloc_op = std::make_shared<OperationContext>(std::move(alloc_kernel));
  alloc_op->stream = stream.get();
  alloc_op->stream_id = stream->stream_id;
  engine.InjectIntoRuntimeQueue(stream.get(), alloc_op);

  EXPECT_FALSE(stream->TryDeferAlloc(alloc_op->GetKernel<KernelTypeEnum::kAlloc>()));
  EXPECT_EQ(stream->deferred_alloc_count_.load(std::memory_order_relaxed), 0);

  engine.ClearRuntimeQueue(stream.get());
}

// =============================================================================
// MoveToRuntimeQueueInOrder mechanics.
//
// Drives a known interleaved partition (allocs + computes + deallocs + a hint)
// through an owners map, calls MoveToRuntimeQueueInOrder directly with the
// reordered raw-pointer list, and asserts:
//   1) stream->runtime_queue_ ends up exactly in the partition order given.
//   2) The hint's OperationContext is destroyed at the move boundary.
//   3) The owners map is empty afterward.
// =============================================================================
TEST_F(OperationExecutionEngineTest, MoveToRuntimeQueueInOrderCommitsPartitionAndDropsHint) {
  setenv("TORCH_NEURONX_ENABLE_CONCATENATION", "1", 1);

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  at::neuron::testing::MockOperationExecutionEngine engine(&compilation_cache, &model_handle_cache);

  auto stream = std::make_shared<StreamImpl>(0, 1);

  auto make_op = [&](KernelTypeEnum kt) {
    auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
    ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(kt));
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream.get();
    return op;
  };

  // Build an interleaved submission: alloc, hlo, hint, alloc, hlo, dealloc.
  auto a1 = make_op(KernelTypeEnum::kAlloc);
  auto h1 = make_op(KernelTypeEnum::kHLO);
  auto hint = make_op(KernelTypeEnum::kHint);
  auto a2 = make_op(KernelTypeEnum::kAlloc);
  auto h2 = make_op(KernelTypeEnum::kHLO);
  auto d1 = make_op(KernelTypeEnum::kDealloc);

  std::weak_ptr<OperationContext> hint_observer = hint;
  OperationContext* hint_raw = hint.get();

  // Build the owners map (raw → shared_ptr) — this is what the worker hands
  // to MoveToRuntimeQueueInOrder after draining dispatch_queue_.
  std::unordered_map<OperationContext*, std::shared_ptr<OperationContext>> owners;
  owners.emplace(a1.get(), a1);
  owners.emplace(h1.get(), h1);
  owners.emplace(hint_raw, std::move(hint));
  owners.emplace(a2.get(), a2);
  owners.emplace(h2.get(), h2);
  owners.emplace(d1.get(), d1);

  // The local `hint` shared_ptr was moved into owners, so dropping any other
  // refs leaves owners as the sole holder of the hint. The weak_ptr observer
  // must still be alive — ownership has not moved out of the map yet.
  ASSERT_FALSE(hint_observer.expired());

  // Reordered output as ConcatenationEngine would produce: allocs first,
  // computes, then deallocs, with the hint mixed in (typical post-flush shape).
  std::list<OperationContext*> partitioned_raws = {a1.get(), a2.get(), hint_raw,
                                                   h1.get(), h2.get(), d1.get()};

  engine.MoveToRuntimeQueueInOrder(partitioned_raws, owners);

  // 1) Runtime queue holds the non-hint ops in partition order.
  auto snapshot = engine.RuntimeQueueSnapshot(stream.get());
  ASSERT_EQ(snapshot.size(), 5u);
  EXPECT_EQ(snapshot[0], a1.get());
  EXPECT_EQ(snapshot[1], a2.get());
  EXPECT_EQ(snapshot[2], h1.get());
  EXPECT_EQ(snapshot[3], h2.get());
  EXPECT_EQ(snapshot[4], d1.get());

  // 2) The hint's OperationContext is destroyed: owners erased its entry,
  //    and no one else held a shared_ptr to it.
  EXPECT_TRUE(hint_observer.expired()) << "Hint shared_ptr should be released at the move boundary";

  // 3) Owners map is drained.
  EXPECT_TRUE(owners.empty());

  engine.ClearRuntimeQueue(stream.get());
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
