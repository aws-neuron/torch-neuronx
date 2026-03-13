#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>
#include <vector>

// Include the mock and the header we're testing
#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"

using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace {

class NeuronDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock session
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

    c10_neuron::reset_distributed_state();

    // Clear any previous mock expectations
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
  }

  void TearDown() override {
    c10_neuron::reset_distributed_state();

    // Clear mock expectations
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);

    // Clean up mock session
    mock_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

// Test basic device operations
TEST_F(NeuronDeviceTest, CurrentDeviceInitialValue) {
  // Initially, current device should be 0
  EXPECT_EQ(c10_neuron::current_device(), 0);
}

TEST_F(NeuronDeviceTest, SetAndGetCurrentDevice) {
  // Mock vnc_count to return 2 devices
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(2), Return(NRT_SUCCESS)));

  // Test setting and getting current device
  c10_neuron::set_device(1);
  EXPECT_EQ(c10_neuron::current_device(), 1);

  c10_neuron::set_device(0);
  EXPECT_EQ(c10_neuron::current_device(), 0);
}

TEST_F(NeuronDeviceTest, SetDeviceInvalidRange) {
  // Mock vnc_count to return 2 devices
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(2), Return(NRT_SUCCESS)));

  // Test setting device with invalid indices
  EXPECT_THROW(c10_neuron::set_device(-1), std::invalid_argument);
  EXPECT_THROW(c10_neuron::set_device(10), std::invalid_argument);
}

TEST_F(NeuronDeviceTest, VncCount) {
  // Test vnc_count function - value is cached after first call
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillOnce(DoAll(SetArgPointee<0>(4), Return(NRT_SUCCESS)));
  EXPECT_EQ(c10_neuron::vnc_count(), 4);
  // Subsequent calls return cached value
  EXPECT_EQ(c10_neuron::vnc_count(), 4);
}

TEST_F(NeuronDeviceTest, VncCountZero) {
  // Test vnc_count with zero devices
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillOnce(DoAll(SetArgPointee<0>(0), Return(NRT_SUCCESS)));
  EXPECT_EQ(c10_neuron::vnc_count(), 0);
}

TEST_F(NeuronDeviceTest, VncCountFailure) {
  // Test vnc_count failure case - returns 0 on NRT failure
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_)).WillOnce(Return(NRT_FAILURE));
  EXPECT_EQ(c10_neuron::vnc_count(), 0);
}

TEST_F(NeuronDeviceTest, DeviceCountWithoutLocalWorldSize) {
  // When local_world_size is not set, device_count should return vnc_count
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillOnce(DoAll(SetArgPointee<0>(3), Return(NRT_SUCCESS)));
  EXPECT_EQ(c10_neuron::device_count(), 3);
}

// Tests that don't use local_world_size - these should work fine
TEST_F(NeuronDeviceTest, SetLocalWorldSizeInvalid) {
  // Test invalid local world size
  EXPECT_THROW(c10_neuron::set_local_world_size(-1), std::invalid_argument);
}

TEST_F(NeuronDeviceTest, SetLocalDeviceStartIndexWithoutWorldSize) {
  // Test setting start index without setting world size first
  EXPECT_THROW(c10_neuron::set_local_device_start_index(0), std::runtime_error);
}

TEST_F(NeuronDeviceTest, GetVncIdWithoutLocalWorldSize) {
  // When local_world_size is not set, get_vnc_id should return device_id
  EXPECT_EQ(c10_neuron::get_vnc_id(0), 0);
  EXPECT_EQ(c10_neuron::get_vnc_id(1), 1);
}

// Test edge cases
TEST_F(NeuronDeviceTest, ZeroDeviceCount) {
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(0), Return(NRT_SUCCESS)));

  EXPECT_EQ(c10_neuron::vnc_count(), 0);
  EXPECT_EQ(c10_neuron::device_count(), 0);

  // Setting any device should fail
  EXPECT_THROW(c10_neuron::set_device(0), std::invalid_argument);
}

TEST_F(NeuronDeviceTest, LargeDeviceCount) {
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(100), Return(NRT_SUCCESS)));

  EXPECT_EQ(c10_neuron::vnc_count(), 100);
  EXPECT_EQ(c10_neuron::device_count(), 100);

  // Should be able to set device 99
  c10_neuron::set_device(99);
  EXPECT_EQ(c10_neuron::current_device(), 99);
}

// Test error conditions
TEST_F(NeuronDeviceTest, NrtFailureHandling) {
  // Test when NRT calls fail
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_)).WillRepeatedly(Return(NRT_FAILURE));

  EXPECT_EQ(c10_neuron::vnc_count(), 0);
  EXPECT_EQ(c10_neuron::device_count(), 0);
}

// Test boundary conditions
TEST_F(NeuronDeviceTest, BoundaryConditions) {
  // Test with exactly 1 device
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

  EXPECT_EQ(c10_neuron::vnc_count(), 1);
  EXPECT_EQ(c10_neuron::device_count(), 1);

  // Should be able to set device 0
  c10_neuron::set_device(0);
  EXPECT_EQ(c10_neuron::current_device(), 0);

  // Should not be able to set device 1
  EXPECT_THROW(c10_neuron::set_device(1), std::invalid_argument);
}

// Test thread safety. Must run before LocalWorldSizeComprehensiveTest
TEST_F(NeuronDeviceTest, ThreadLocalCurrentDevice) {
  // Reset global_default_device so new threads will fallback to device 0
  c10_neuron::reset_distributed_state();

  // Mock vnc_count to return enough devices
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(4), Return(NRT_SUCCESS)));

  const int num_threads = 4;
  std::vector<std::thread> threads;
  std::vector<int> results(num_threads);

  // Create threads that each set a different current device
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([i, &results]() {
      // Each thread should start with device 0
      EXPECT_EQ(c10_neuron::current_device(), 0);

      // Set device to thread index
      c10_neuron::set_device(i);
      results[i] = c10_neuron::current_device();
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify that each thread had its own current device
  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(results[i], i);
  }

  // Main thread should still have device 0
  EXPECT_EQ(c10_neuron::current_device(), 0);
}

// Single comprehensive test for all local_world_size functionality
// This avoids the "Attempted to reset local_world_size" error by doing everything in one test
TEST_F(NeuronDeviceTest, LocalWorldSizeComprehensiveTest) {
  // Mock vnc_count to return 2 devices (for this process)
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(2), Return(NRT_SUCCESS)));

  // Test setting local world size
  c10_neuron::set_local_world_size(4);
  EXPECT_EQ(c10_neuron::device_count(), 4);

  // Test that setting local world size multiple times throws
  EXPECT_THROW(c10_neuron::set_local_world_size(3), std::runtime_error);

  // Test setting local device start index
  c10_neuron::set_local_device_start_index(1);
  EXPECT_EQ(c10_neuron::current_device(), 1);

  // Test invalid start index (should throw since world size is already set)
  EXPECT_THROW(c10_neuron::set_local_device_start_index(-1), std::invalid_argument);
  EXPECT_THROW(c10_neuron::set_local_device_start_index(4), std::invalid_argument);

  // Test VNC ID calculation
  EXPECT_EQ(c10_neuron::get_vnc_id(1), 0);  // device_id - start_index
  EXPECT_EQ(c10_neuron::get_vnc_id(2), 1);
  EXPECT_EQ(c10_neuron::get_vnc_id(3), 2);

  // Test invalid device IDs
  EXPECT_THROW(c10_neuron::get_vnc_id(0), std::invalid_argument);  // Below start index
  EXPECT_THROW(c10_neuron::get_vnc_id(4), std::invalid_argument);  // Above world size

  // Test setting device within valid range
  c10_neuron::set_device(2);
  EXPECT_EQ(c10_neuron::current_device(), 2);

  // Test setting device outside valid range
  EXPECT_THROW(c10_neuron::set_device(0), std::invalid_argument);  // Below start
  EXPECT_THROW(c10_neuron::set_device(5), std::invalid_argument);  // Above range
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
