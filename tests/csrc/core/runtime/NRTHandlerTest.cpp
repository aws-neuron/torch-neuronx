#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class NRTHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();

    // Create test tensors
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};

    // Create test NEFF bytes
    neff_bytes_ = {0x4E, 0x45, 0x46, 0x46, 0x01, 0x02, 0x03, 0x04};
  }

  void TearDown() override { mock_session_.reset(); }

  std::unique_ptr<MockNRTSession> mock_session_;
  torch::Tensor input_tensor_;
  torch::Tensor output_tensor_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
  std::vector<uint8_t> neff_bytes_;
};

TEST_F(NRTHandlerTest, ExecutionConfigConstruction) {
  NRTHandler::ExecutionConfig config;
  EXPECT_EQ(config.device_id, 0);
  EXPECT_EQ(config.num_cores, 1);

  NRTHandler::ExecutionConfig config2(2, 4);
  EXPECT_EQ(config2.device_id, 2);
  EXPECT_EQ(config2.num_cores, 4);
}

// Note: Full integration tests for ExecuteCompilableKernel would require
// mocking NeuronResourceManager and the full tensor allocation stack.
// These tests focus on the ExecutionResult and ExecutionConfig structures.
// For full coverage, integration tests with actual NRT would be needed.

TEST_F(NRTHandlerTest, ExecutionConfigCopy) {
  NRTHandler::ExecutionConfig config1(1, 2);
  NRTHandler::ExecutionConfig config2 = config1;

  EXPECT_EQ(config2.device_id, 1);
  EXPECT_EQ(config2.num_cores, 2);
}

TEST_F(NRTHandlerTest, MultipleExecutionConfigs) {
  std::vector<NRTHandler::ExecutionConfig> configs;
  configs.emplace_back(0, 1);
  configs.emplace_back(1, 2);
  configs.emplace_back(2, 4);

  EXPECT_EQ(configs.size(), 3);
  EXPECT_EQ(configs[0].device_id, 0);
  EXPECT_EQ(configs[1].device_id, 1);
  EXPECT_EQ(configs[2].device_id, 2);
  EXPECT_EQ(configs[0].num_cores, 1);
  EXPECT_EQ(configs[1].num_cores, 2);
  EXPECT_EQ(configs[2].num_cores, 4);
}

TEST_F(NRTHandlerTest, ExecutionConfigDefaultValues) {
  NRTHandler::ExecutionConfig config;

  // Verify default values are sensible
  EXPECT_GE(config.device_id, 0);
  EXPECT_GT(config.num_cores, 0);
}

TEST_F(NRTHandlerTest, ExecutionConfigBoundaryValues) {
  NRTHandler::ExecutionConfig config1(0, 1);
  EXPECT_EQ(config1.device_id, 0);
  EXPECT_EQ(config1.num_cores, 1);

  NRTHandler::ExecutionConfig config2(15, 16);
  EXPECT_EQ(config2.device_id, 15);
  EXPECT_EQ(config2.num_cores, 16);
}

TEST_F(NRTHandlerTest, ExecutionConfigMultiCore) {
  NRTHandler::ExecutionConfig single_core(0, 1);
  NRTHandler::ExecutionConfig dual_core(0, 2);
  NRTHandler::ExecutionConfig quad_core(0, 4);

  EXPECT_EQ(single_core.num_cores, 1);
  EXPECT_EQ(dual_core.num_cores, 2);
  EXPECT_EQ(quad_core.num_cores, 4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
