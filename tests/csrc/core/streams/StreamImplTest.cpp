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
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class StreamImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize NRT mocks
    nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    auto* nrt_mock = torch_neuronx::testing::MockNRT::GetInstance();

    // Set up default NRT behaviors for successful execution
    ON_CALL(*nrt_mock, nrt_allocate_tensor_set(_))
        .WillByDefault(
            DoAll(SetArgPointee<0>(reinterpret_cast<nrt_tensor_set_t*>(0x1)), Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_destroy_tensor_set(_)).WillByDefault(Return());

    ON_CALL(*nrt_mock, nrt_add_tensor_to_tensor_set(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    ON_CALL(*nrt_mock, nrt_execute(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    // Initialize copy utils mocks
    copy_utils_session_ = std::make_unique<torch_neuronx::utils::testing::MockCopyUtilsSession>();

    // Use mock engine that skips device checks and actual execution
    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();
    engine_ = mock_engine_wrapper_->GetEngine();

    // Create stream with device 0, stream id 1, priority 0
    stream_ = std::make_shared<StreamImpl>(0, 1, 0, engine_);
  }

  void TearDown() override {
    if (stream_) {
      stream_.reset();
    }
    mock_engine_wrapper_.reset();
    copy_utils_session_.reset();
    nrt_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuronx::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
  OperationExecutionEngine* engine_;
  std::shared_ptr<StreamImpl> stream_;
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

// Test basic stream functionality without statistics
TEST_F(StreamImplTest, BasicStreamFunctionality) {
  // Verify stream can handle operations without statistics tracking
  EXPECT_TRUE(stream_->Query());
  EXPECT_NO_THROW(stream_->Synchronize());
}

// Test stream with different priorities
TEST_F(StreamImplTest, StreamWithDifferentPriorities) {
  auto high_priority_stream = std::make_shared<StreamImpl>(0, 2, 10, engine_);
  auto low_priority_stream = std::make_shared<StreamImpl>(0, 3, 1, engine_);

  EXPECT_EQ(high_priority_stream->priority, 10);
  EXPECT_EQ(low_priority_stream->priority, 1);
}

// Test stream on different devices
TEST_F(StreamImplTest, StreamOnDifferentDevices) {
  auto stream_device_0 = std::make_shared<StreamImpl>(0, 10, 0, engine_);
  auto stream_device_1 = std::make_shared<StreamImpl>(1, 11, 0, engine_);

  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);
}

// Test concurrent synchronization on empty stream
TEST_F(StreamImplTest, ConcurrentSynchronizationEmptyStream) {
  std::vector<std::thread> threads;

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([this]() { stream_->Synchronize(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(stream_->Query());
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
}

// Test stream with negative priority
TEST_F(StreamImplTest, StreamWithNegativePriority) {
  auto negative_priority_stream = std::make_shared<StreamImpl>(0, 101, -5, engine_);
  EXPECT_EQ(negative_priority_stream->priority, -5);
}

// Test GetStreamId
TEST_F(StreamImplTest, GetStreamId) {
  EXPECT_EQ(stream_->GetStreamId(), 1);

  auto another_stream = std::make_shared<StreamImpl>(0, 42, 0, engine_);
  EXPECT_EQ(another_stream->GetStreamId(), 42);
}

// Test GetNextReadyOperation on empty stream returns nullptr
TEST_F(StreamImplTest, GetNextReadyOperationEmptyStream) {
  auto* result = stream_->GetNextReadyOperation();
  EXPECT_EQ(result, nullptr);
}

// Test active_operations_ starts empty
TEST_F(StreamImplTest, ActiveOperationsStartsEmpty) {
  // Query returns true when no active operations
  EXPECT_TRUE(stream_->Query());
}

// Test multiple Query calls are consistent
TEST_F(StreamImplTest, MultipleQueryCallsConsistent) {
  EXPECT_TRUE(stream_->Query());
  EXPECT_TRUE(stream_->Query());
  EXPECT_TRUE(stream_->Query());
}

// Test Synchronize is idempotent on empty stream
TEST_F(StreamImplTest, SynchronizeIdempotentOnEmptyStream) {
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_TRUE(stream_->Query());
}

// Test stream with large stream ID
TEST_F(StreamImplTest, StreamWithLargeStreamId) {
  auto large_id_stream = std::make_shared<StreamImpl>(0, 999999, 0, engine_);
  EXPECT_EQ(large_id_stream->GetStreamId(), 999999);
  EXPECT_TRUE(large_id_stream->Query());
}

// Test stream with various device indices
TEST_F(StreamImplTest, StreamWithVariousDeviceIndices) {
  for (c10::DeviceIndex device = 0; device < 4; ++device) {
    auto stream = std::make_shared<StreamImpl>(device, device + 100, 0, engine_);
    EXPECT_EQ(stream->device_index, device);
    EXPECT_EQ(stream->stream_id, device + 100);
  }
}

// Test stream priority range
TEST_F(StreamImplTest, StreamPriorityRange) {
  // Test various priority values
  std::vector<int> priorities = {-100, -10, -1, 0, 1, 10, 100};

  for (int prio : priorities) {
    auto stream = std::make_shared<StreamImpl>(0, 200 + prio, prio, engine_);
    EXPECT_EQ(stream->priority, prio);
  }
}

// Test concurrent Query calls
TEST_F(StreamImplTest, ConcurrentQueryCalls) {
  std::vector<std::thread> threads;
  std::atomic<int> true_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this, &true_count]() {
      if (stream_->Query()) {
        true_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All should return true for empty stream
  EXPECT_EQ(true_count.load(), 10);
}

// Test stream destruction is safe
TEST_F(StreamImplTest, StreamDestructionIsSafe) {
  auto temp_stream = std::make_shared<StreamImpl>(0, 500, 0, engine_);
  EXPECT_NE(temp_stream, nullptr);
  EXPECT_TRUE(temp_stream->Query());

  // Destruction should be safe
  EXPECT_NO_THROW(temp_stream.reset());
}

// Test multiple streams can coexist
TEST_F(StreamImplTest, MultipleStreamsCoexist) {
  std::vector<std::shared_ptr<StreamImpl>> streams;

  for (int i = 0; i < 5; ++i) {
    streams.push_back(std::make_shared<StreamImpl>(0, 300 + i, i - 2, engine_));
  }

  // All streams should be queryable
  for (const auto& s : streams) {
    EXPECT_TRUE(s->Query());
  }

  // Verify each has correct properties
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(streams[i]->stream_id, 300 + i);
    EXPECT_EQ(streams[i]->priority, i - 2);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
