#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuron/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuron/csrc/core/OperationExecutionEngine.h"
#include "torch_neuron/csrc/core/compilation/CompilationCache.h"
#include "torch_neuron/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuron/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class StreamImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize NRT mocks
    nrt_session_ = std::make_unique<neuron::testing::MockNRTSession>();
    auto* nrt_mock = neuron::testing::MockNRT::GetInstance();

    // Set up default NRT behaviors for successful execution
    ON_CALL(*nrt_mock, nrt_allocate_tensor_set(_))
        .WillByDefault(
            DoAll(SetArgPointee<0>(reinterpret_cast<nrt_tensor_set_t*>(0x1)), Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_destroy_tensor_set(_)).WillByDefault(Return());

    ON_CALL(*nrt_mock, nrt_add_tensor_to_tensor_set(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    ON_CALL(*nrt_mock, nrt_execute(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    // Initialize copy utils mocks
    copy_utils_session_ = std::make_unique<torch_neuron::utils::testing::MockCopyUtilsSession>();

    // Use mock engine that skips device checks and actual execution
    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();
    engine_ = mock_engine_wrapper_->GetEngine();

    // Create stream with device 0, stream id 1, priority 0
    stream_ = std::make_shared<StreamImpl>(0, 1, 0, engine_);
  }

  void TearDown() override {
    if (stream_) {
      stream_->Synchronize();
      stream_.reset();
    }
    mock_engine_wrapper_.reset();
    copy_utils_session_.reset();
    nrt_session_.reset();
  }

  std::unique_ptr<neuron::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuron::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
  OperationExecutionEngine* engine_;
  std::shared_ptr<StreamImpl> stream_;

  // Helper to create a mock compilable kernel with default expectations
  std::unique_ptr<MockCompilableKernelExecution> CreateMockKernel(const std::string& name) {
    // Create CPU tensors - the mock will provide device info through GetInputs/GetOutputs
    auto inputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};
    auto outputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};

    auto kernel = std::make_unique<MockCompilableKernelExecution>(name, inputs, outputs);
    EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));
    EXPECT_CALL(*kernel, CompileToNeff())
        .WillRepeatedly(Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));
    return kernel;
  }
};

// Test basic construction
TEST_F(StreamImplTest, BasicConstruction) {
  EXPECT_NE(stream_, nullptr);
  EXPECT_EQ(stream_->device_index, 0);
  EXPECT_EQ(stream_->stream_id, 1);
  EXPECT_EQ(stream_->priority, 0);
  EXPECT_EQ(stream_->GetStreamId(), 1);
}

// Test Query on empty stream
TEST_F(StreamImplTest, QueryEmptyStream) { EXPECT_TRUE(stream_->Query()); }

// Test Synchronize on empty stream
TEST_F(StreamImplTest, SynchronizeEmptyStream) { EXPECT_NO_THROW(stream_->Synchronize()); }

// Test GetResourceStats
TEST_F(StreamImplTest, GetResourceStats) {
  auto stats = stream_->GetResourceStats();

  EXPECT_EQ(stats.stream_id, 1);
  EXPECT_EQ(stats.total_submitted, 0);
  EXPECT_EQ(stats.total_completed, 0);
  EXPECT_EQ(stats.total_failed, 0);
  EXPECT_TRUE(stats.is_idle);
  EXPECT_GE(stats.uptime.count(), 0);
}

// Test operation submission and tracking
TEST_F(StreamImplTest, OperationSubmissionTracking) {
  auto kernel = CreateMockKernel("test_op");

  auto op = OperationContext(std::move(kernel));
  auto future = stream_->SubmitOperationContext(std::move(op));

  // Stats should show submitted operation
  auto stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, 1);

  // Wait for completion
  EXPECT_NO_THROW(stream_->Synchronize());

  // Stats should show operation completed
  stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, 1);
  EXPECT_EQ(stats.total_completed, 1);
  EXPECT_TRUE(stats.is_idle);
}

// Test multiple operations
TEST_F(StreamImplTest, MultipleOperations) {
  const int num_ops = 5;
  std::vector<std::future<OperationContextResult>> futures;

  for (int i = 0; i < num_ops; ++i) {
    auto kernel = CreateMockKernel("test_op_" + std::to_string(i));
    auto op = OperationContext(std::move(kernel));
    futures.push_back(stream_->SubmitOperationContext(std::move(op)));
  }

  // Synchronize and check all completed
  EXPECT_NO_THROW(stream_->Synchronize());

  auto stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, num_ops);
  EXPECT_EQ(stats.total_completed, num_ops);
  EXPECT_TRUE(stats.is_idle);
}

// Test Query with active operations
TEST_F(StreamImplTest, QueryWithActiveOperations) {
  auto kernel = CreateMockKernel("test_op");
  auto op = OperationContext(std::move(kernel));
  auto future = stream_->SubmitOperationContext(std::move(op));

  // May or may not be idle depending on execution speed
  // Just verify Query doesn't crash
  stream_->Query();

  stream_->Synchronize();
  EXPECT_TRUE(stream_->Query());
}

// Test event registration with no operations
TEST_F(StreamImplTest, RegisterEventWithNoOperations) {
  NeuronEvent event;

  // Should complete immediately
  stream_->RegisterEventWithLastOperation(&event);

  EXPECT_TRUE(event.query());
}

// Test event registration with operations
TEST_F(StreamImplTest, RegisterEventWithOperations) {
  auto kernel = CreateMockKernel("test_op");
  auto op = OperationContext(std::move(kernel));
  auto future = stream_->SubmitOperationContext(std::move(op));

  NeuronEvent event;
  stream_->RegisterEventWithLastOperation(&event);

  // Event should complete after synchronization
  stream_->Synchronize();
  EXPECT_TRUE(event.query());
}

// Test multiple events
TEST_F(StreamImplTest, MultipleEvents) {
  const int num_ops = 3;
  std::vector<std::unique_ptr<NeuronEvent>> events;

  for (int i = 0; i < num_ops; ++i) {
    auto kernel = CreateMockKernel("test_op_" + std::to_string(i));
    auto op = OperationContext(std::move(kernel));
    stream_->SubmitOperationContext(std::move(op));

    auto event = std::make_unique<NeuronEvent>();
    stream_->RegisterEventWithLastOperation(event.get());
    events.push_back(std::move(event));
  }

  stream_->Synchronize();

  // All events should be completed
  for (const auto& event : events) {
    EXPECT_TRUE(event->query());
  }
}

// Test null event registration
TEST_F(StreamImplTest, RegisterNullEvent) {
  EXPECT_NO_THROW(stream_->RegisterEventWithLastOperation(nullptr));
}

// Test stream with different priorities
TEST_F(StreamImplTest, StreamWithDifferentPriorities) {
  auto high_priority_stream = std::make_shared<StreamImpl>(0, 2, 10, engine_);
  auto low_priority_stream = std::make_shared<StreamImpl>(0, 3, 1, engine_);

  EXPECT_EQ(high_priority_stream->priority, 10);
  EXPECT_EQ(low_priority_stream->priority, 1);

  high_priority_stream->Synchronize();
  low_priority_stream->Synchronize();
}

// Test stream on different devices
TEST_F(StreamImplTest, StreamOnDifferentDevices) {
  auto stream_device_0 = std::make_shared<StreamImpl>(0, 10, 0, engine_);
  auto stream_device_1 = std::make_shared<StreamImpl>(1, 11, 0, engine_);

  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);

  stream_device_0->Synchronize();
  stream_device_1->Synchronize();
}

// Test concurrent synchronization
TEST_F(StreamImplTest, ConcurrentSynchronization) {
  std::vector<std::thread> threads;

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([this]() { stream_->Synchronize(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(stream_->Query());
}

// Test resource stats during operation execution
TEST_F(StreamImplTest, ResourceStatsDuringExecution) {
  auto kernel = CreateMockKernel("test_op");
  auto op = OperationContext(std::move(kernel));
  auto future = stream_->SubmitOperationContext(std::move(op));

  // Get stats while operation may be running
  auto stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, 1);
  EXPECT_GE(stats.total_completed, 0);

  stream_->Synchronize();

  stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_completed, 1);
}

// Test uptime tracking
TEST_F(StreamImplTest, UptimeTracking) {
  std::this_thread::sleep_for(10ms);

  auto stats = stream_->GetResourceStats();
  EXPECT_GE(stats.uptime.count(), 0);
}

// Test shutdown flag
TEST_F(StreamImplTest, ShutdownFlag) {
  EXPECT_FALSE(stream_->shutdown_requested.load());

  // Trigger shutdown by destroying stream
  stream_->Synchronize();
  stream_.reset();

  // Create new stream to verify it starts with shutdown_requested = false
  stream_ = std::make_shared<StreamImpl>(0, 5, 0, engine_);
  EXPECT_FALSE(stream_->shutdown_requested.load());
}

// Test operation with compilable kernel
TEST_F(StreamImplTest, OperationWithCompilableKernel) {
  auto inputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};
  auto outputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};

  auto kernel = std::make_unique<MockCompilableKernelExecution>("compilable_op", inputs, outputs,
                                                                "test_key", false);
  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  auto op = OperationContext(std::move(kernel));
  auto future = stream_->SubmitOperationContext(std::move(op));

  stream_->Synchronize();

  auto stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, 1);
}

// Test stream destruction with pending operations
TEST_F(StreamImplTest, DestructionWithPendingOperations) {
  auto kernel = CreateMockKernel("test_op");
  auto op = OperationContext(std::move(kernel));
  stream_->SubmitOperationContext(std::move(op));

  // Destroy stream - should synchronize first
  EXPECT_NO_THROW(stream_.reset());

  // Recreate for teardown
  stream_ = std::make_shared<StreamImpl>(0, 1, 0, engine_);
}

// Test error handler initialization
TEST_F(StreamImplTest, ErrorHandlerInitialization) { EXPECT_NE(stream_->error_handler_, nullptr); }

// Test operation execution engine pointer
TEST_F(StreamImplTest, OperationExecutionEnginePointer) {
  EXPECT_NE(stream_->operation_execution_engine_, nullptr);
  EXPECT_EQ(stream_->operation_execution_engine_, engine_);
}

// Test stream with zero priority
TEST_F(StreamImplTest, StreamWithZeroPriority) {
  auto zero_priority_stream = std::make_shared<StreamImpl>(0, 100, 0, engine_);
  EXPECT_EQ(zero_priority_stream->priority, 0);
  zero_priority_stream->Synchronize();
}

// Test stream with negative priority
TEST_F(StreamImplTest, StreamWithNegativePriority) {
  auto negative_priority_stream = std::make_shared<StreamImpl>(0, 101, -5, engine_);
  EXPECT_EQ(negative_priority_stream->priority, -5);
  negative_priority_stream->Synchronize();
}

// Test concurrent operation submission
TEST_F(StreamImplTest, ConcurrentOperationSubmission) {
  std::vector<std::thread> threads;
  std::atomic<int> submitted{0};

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([this, &submitted, i]() {
      auto kernel = CreateMockKernel("test_op_" + std::to_string(i));
      auto op = OperationContext(std::move(kernel));
      stream_->SubmitOperationContext(std::move(op));
      submitted++;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  stream_->Synchronize();

  EXPECT_EQ(submitted.load(), 5);
  auto stats = stream_->GetResourceStats();
  EXPECT_EQ(stats.total_submitted, 5);
  EXPECT_EQ(stats.total_completed, 5);
}

// Test GetStreamId
TEST_F(StreamImplTest, GetStreamId) {
  EXPECT_EQ(stream_->GetStreamId(), 1);

  auto another_stream = std::make_shared<StreamImpl>(0, 42, 0, engine_);
  EXPECT_EQ(another_stream->GetStreamId(), 42);
  another_stream->Synchronize();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
