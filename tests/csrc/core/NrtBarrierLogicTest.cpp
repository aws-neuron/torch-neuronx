#include <gtest/gtest.h>
#include <nrt/nrt.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronBindings.h"
#include "torch_neuronx/csrc/core/NeuronBarrier.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"

using ::testing::Return;

// Global to capture what device_index is passed to synchronize
static c10::DeviceIndex g_captured_sync_device = -1;

// Mock current_device to return 63
namespace c10_neuron {
int current_device() { return 63; }
}  // namespace c10_neuron

// Intercept at::neuron::synchronize to capture the device_index
namespace at::neuron {
void synchronize(c10::DeviceIndex device_index) { g_captured_sync_device = device_index; }
}  // namespace at::neuron

namespace {

class NrtBarrierLogicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_captured_sync_device = -1;

    mock_nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
  }

  void TearDown() override {
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
    mock_nrt_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_nrt_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
  torch_neuronx::testing::MockNeuronBindingsSession mock_bindings_session_;
};

TEST_F(NrtBarrierLogicTest, BarrierSynchronizesCurrentDeviceNotGlobalDeviceId) {
  EXPECT_CALL(*mock_nrt_, nrt_barrier(0, 255, 256)).WillOnce(Return(NRT_SUCCESS));

  nrt_barrier_impl(0, 255, 256);

  // Verify synchronize was called with current_device (63), NOT global_device_id (255)
  EXPECT_EQ(g_captured_sync_device, 63)
      << "synchronize should use current_device() (63), not global_device_id (255)";
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
