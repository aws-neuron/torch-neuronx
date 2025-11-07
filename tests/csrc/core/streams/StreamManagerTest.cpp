#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuron/csrc/core/streams/StreamImpl.h"
#include "torch_neuron/csrc/core/streams/StreamManager.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class StreamManagerTest : public ::testing::Test {
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

    // StreamManager is a singleton, so we just get the instance
    manager_ = &StreamManager::Instance();
  }

  void TearDown() override {
    // Synchronize all streams to clean up
    manager_->SynchronizeDevice(0);
    copy_utils_session_.reset();
    nrt_session_.reset();
  }

  std::unique_ptr<neuron::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuron::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  StreamManager* manager_;

  // Helper to create a mock compilable kernel with default expectations
  std::unique_ptr<MockCompilableKernelExecution> CreateMockKernel(const std::string& name) {
    auto inputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};
    auto outputs = std::vector<torch::Tensor>{create_test_tensor({2, 3})};

    auto kernel = std::make_unique<MockCompilableKernelExecution>(name, inputs, outputs);
    EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));
    EXPECT_CALL(*kernel, CompileToNeff())
        .WillRepeatedly(Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));
    return kernel;
  }
};

// Test singleton instance
TEST_F(StreamManagerTest, SingletonInstance) {
  auto& instance1 = StreamManager::Instance();
  auto& instance2 = StreamManager::Instance();

  EXPECT_EQ(&instance1, &instance2);
}

// Test GetStream creates stream if not exists
TEST_F(StreamManagerTest, GetStreamCreatesIfNotExists) {
  auto stream = manager_->GetStream(0, 100);

  EXPECT_NE(stream, nullptr);
  EXPECT_EQ(stream->device_index, 0);
  EXPECT_EQ(stream->stream_id, 100);
}

// Test GetStream returns same stream for same device and id
TEST_F(StreamManagerTest, GetStreamReturnsSameStream) {
  auto stream1 = manager_->GetStream(0, 200);
  auto stream2 = manager_->GetStream(0, 200);

  EXPECT_EQ(stream1, stream2);
}

// Test GetStream creates different streams for different ids
TEST_F(StreamManagerTest, GetStreamCreatesDifferentStreams) {
  auto stream1 = manager_->GetStream(0, 300);
  auto stream2 = manager_->GetStream(0, 301);

  EXPECT_NE(stream1, stream2);
  EXPECT_EQ(stream1->stream_id, 300);
  EXPECT_EQ(stream2->stream_id, 301);
}

// Test GetStream creates different streams for different devices
TEST_F(StreamManagerTest, GetStreamCreatesDifferentDeviceStreams) {
  auto stream_device_0 = manager_->GetStream(0, 400);
  auto stream_device_1 = manager_->GetStream(1, 400);

  EXPECT_NE(stream_device_0, stream_device_1);
  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);
}

// Test CreateStream generates unique stream IDs
TEST_F(StreamManagerTest, CreateStreamGeneratesUniqueIds) {
  auto stream_id_1 = manager_->CreateStream(0, 0);
  auto stream_id_2 = manager_->CreateStream(0, 0);
  auto stream_id_3 = manager_->CreateStream(0, 0);

  EXPECT_NE(stream_id_1, stream_id_2);
  EXPECT_NE(stream_id_2, stream_id_3);
  EXPECT_NE(stream_id_1, stream_id_3);
}

// Test CreateStream with different priorities
TEST_F(StreamManagerTest, CreateStreamWithDifferentPriorities) {
  auto high_priority_id = manager_->CreateStream(0, 10);
  auto low_priority_id = manager_->CreateStream(0, 1);

  auto high_priority_stream = manager_->GetStream(0, high_priority_id);
  auto low_priority_stream = manager_->GetStream(0, low_priority_id);

  EXPECT_EQ(high_priority_stream->priority, 10);
  EXPECT_EQ(low_priority_stream->priority, 1);
}

// Test CreateStream on different devices
TEST_F(StreamManagerTest, CreateStreamOnDifferentDevices) {
  auto stream_id_device_0 = manager_->CreateStream(0, 0);
  auto stream_id_device_1 = manager_->CreateStream(1, 0);

  auto stream_device_0 = manager_->GetStream(0, stream_id_device_0);
  auto stream_device_1 = manager_->GetStream(1, stream_id_device_1);

  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);
}

// Test SynchronizeDevice with no streams
TEST_F(StreamManagerTest, SynchronizeDeviceWithNoStreams) {
  // Should not crash even if device has no streams
  EXPECT_NO_THROW(manager_->SynchronizeDevice(99));
}

// Test SynchronizeDevice with single stream

// Test SynchronizeDevice with multiple streams

// Test DestroyStream

// Test DestroyStream with non-existent stream
TEST_F(StreamManagerTest, DestroyNonExistentStream) {
  EXPECT_NO_THROW(manager_->DestroyStream(0, 99999));
}

// Test GetResourceStats
TEST_F(StreamManagerTest, GetResourceStats) {
  auto stats = manager_->GetResourceStats();

  EXPECT_GE(stats.total_streams, 0);
  EXPECT_GE(stats.total_pending_operations, 0);
  EXPECT_GE(stats.total_memory_usage_estimate, 0);
  EXPECT_GE(stats.nrt_available_slots, 0);
  EXPECT_GE(stats.nrt_acquired_slots, 0);
  EXPECT_GE(stats.nrt_waiting_threads, 0);
}

// Test GetResourceStats with active streams
TEST_F(StreamManagerTest, GetResourceStatsWithActiveStreams) {
  auto stream1 = manager_->GetStream(0, 700);
  auto stream2 = manager_->GetStream(0, 701);

  auto stats = manager_->GetResourceStats();
  EXPECT_GE(stats.total_streams, 2);
}

// Test GetResourceStats with pending operations

// Test thread-local current stream tracking
TEST_F(StreamManagerTest, ThreadLocalCurrentStreamTracking) {
  SetCurrentStreamId(0, 900);
  EXPECT_EQ(GetCurrentStreamId(0), 900);

  SetCurrentStreamId(0, 901);
  EXPECT_EQ(GetCurrentStreamId(0), 901);
}

// Test thread-local current stream for different devices
TEST_F(StreamManagerTest, ThreadLocalCurrentStreamDifferentDevices) {
  SetCurrentStreamId(0, 1000);
  SetCurrentStreamId(1, 1001);

  EXPECT_EQ(GetCurrentStreamId(0), 1000);
  EXPECT_EQ(GetCurrentStreamId(1), 1001);
}

// Test thread-local current stream default value
TEST_F(StreamManagerTest, ThreadLocalCurrentStreamDefault) {
  // For a device we haven't set, should return 0 (default stream)
  EXPECT_EQ(GetCurrentStreamId(99), 0);
}

// Test thread-local current stream isolation
TEST_F(StreamManagerTest, ThreadLocalCurrentStreamIsolation) {
  SetCurrentStreamId(0, 1100);

  std::thread other_thread([this]() {
    // In another thread, should get default value
    EXPECT_EQ(GetCurrentStreamId(0), 0);

    // Set in this thread
    SetCurrentStreamId(0, 1101);
    EXPECT_EQ(GetCurrentStreamId(0), 1101);
  });

  other_thread.join();

  // Original thread should still have its value
  EXPECT_EQ(GetCurrentStreamId(0), 1100);
}

// Test concurrent stream creation
TEST_F(StreamManagerTest, ConcurrentStreamCreation) {
  std::vector<std::thread> threads;
  std::vector<c10::StreamId> stream_ids(10);

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back(
        [this, &stream_ids, i]() { stream_ids[i] = manager_->CreateStream(0, 0); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All stream IDs should be unique
  std::set<c10::StreamId> unique_ids(stream_ids.begin(), stream_ids.end());
  EXPECT_EQ(unique_ids.size(), 10);
}

// Test concurrent GetStream calls
TEST_F(StreamManagerTest, ConcurrentGetStreamCalls) {
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<StreamImpl>> streams(10);

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this, &streams, i]() { streams[i] = manager_->GetStream(0, 1200); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All should point to the same stream
  for (int i = 1; i < 10; ++i) {
    EXPECT_EQ(streams[i], streams[0]);
  }
}

// Test concurrent SynchronizeDevice calls
TEST_F(StreamManagerTest, ConcurrentSynchronizeDeviceCalls) {
  auto stream = manager_->GetStream(0, 1300);

  std::vector<std::thread> threads;

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([this]() { manager_->SynchronizeDevice(0); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(stream->Query());
}

// Test GetResourceStats thread safety
TEST_F(StreamManagerTest, GetResourceStatsThreadSafety) {
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this, &success_count]() {
      for (int j = 0; j < 10; ++j) {
        auto stats = manager_->GetResourceStats();
        if (stats.total_streams >= 0) {
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

// Test default stream (stream_id 0) exists
TEST_F(StreamManagerTest, DefaultStreamExists) {
  auto default_stream = manager_->GetStream(0, 0);

  EXPECT_NE(default_stream, nullptr);
  EXPECT_EQ(default_stream->stream_id, 0);
  EXPECT_EQ(default_stream->device_index, 0);
}

// Test stream lifecycle

// Test multiple devices
TEST_F(StreamManagerTest, MultipleDevices) {
  auto stream_device_0 = manager_->GetStream(0, 1400);
  auto stream_device_1 = manager_->GetStream(1, 1401);
  auto stream_device_2 = manager_->GetStream(2, 1402);

  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);
  EXPECT_EQ(stream_device_2->device_index, 2);

  // Synchronize each device
  manager_->SynchronizeDevice(0);
  manager_->SynchronizeDevice(1);
  manager_->SynchronizeDevice(2);
}

// Test stream priority ordering
TEST_F(StreamManagerTest, StreamPriorityOrdering) {
  auto high_priority_id = manager_->CreateStream(0, 100);
  auto medium_priority_id = manager_->CreateStream(0, 50);
  auto low_priority_id = manager_->CreateStream(0, 1);

  auto high_priority_stream = manager_->GetStream(0, high_priority_id);
  auto medium_priority_stream = manager_->GetStream(0, medium_priority_id);
  auto low_priority_stream = manager_->GetStream(0, low_priority_id);

  EXPECT_GT(high_priority_stream->priority, medium_priority_stream->priority);
  EXPECT_GT(medium_priority_stream->priority, low_priority_stream->priority);
}

// Test DestroyStream removes stream from device map
TEST_F(StreamManagerTest, DestroyStreamRemovesFromMap) {
  auto stream_id = manager_->CreateStream(0, 0);
  auto stream_before = manager_->GetStream(0, stream_id);

  stream_before->Synchronize();
  manager_->DestroyStream(0, stream_id);

  // Getting stream again creates a new instance
  auto stream_after = manager_->GetStream(0, stream_id);
  EXPECT_NE(stream_before.get(), stream_after.get());
}

// Test concurrent operations on different streams

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
