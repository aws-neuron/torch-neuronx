#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::Truly;

class NRTHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*MockNRT::GetInstance(), nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    c10_neuron::reset_distributed_state();
    c10_neuron::set_world_size(4);
    c10_neuron::set_rank(1);
    NeuronResourceManager::Instance().GetModelHandleCache().Clear();

    // Create test tensors
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};

    // Create test NEFF bytes
    neff_bytes_ = {0x4E, 0x45, 0x46, 0x46, 0x01, 0x02, 0x03, 0x04};
  }

  void TearDown() override {
    NeuronResourceManager::Instance().GetModelHandleCache().Clear();
    c10_neuron::reset_distributed_state();
    mock_session_.reset();
  }

  void AttachCachedNeff(at::neuron::CompilableKernelExecution& kernel) {
    kernel.SetCachedNeff(NeffBytesPtr(&neff_bytes_, CacheEntryGuard{}));
  }

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

TEST_F(NRTHandlerTest, ScheduleTensorCastTranslatesOperands) {
  nrt_tensor_t* fake_src = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt_tensor_t* fake_dst = reinterpret_cast<nrt_tensor_t*>(0x5678);
  NRTHandler::TensorCastOperand src{fake_src, 3, c10::ScalarType::Float};
  NRTHandler::TensorCastOperand dst{fake_dst, 7, c10::ScalarType::BFloat16};
  nrta_seq_t expected_seq = 789;
  NRT_STATUS ret = NRT_SUCCESS;

  EXPECT_CALL(*MockNRT::GetInstance(),
              nrta_tensor_cast(Truly([fake_src](const nrt_compute_tensor_t* arg) {
                                 return arg != nullptr && arg->tensor == fake_src &&
                                        arg->offset_elem == 3 && arg->dtype == NRT_DTYPE_FLOAT32;
                               }),
                               Truly([fake_dst](const nrt_compute_tensor_t* arg) {
                                 return arg != nullptr && arg->tensor == fake_dst &&
                                        arg->offset_elem == 7 && arg->dtype == NRT_DTYPE_BFLOAT16;
                               }),
                               128, 1, 2, &ret, _))
      .WillOnce(DoAll(SetArgPointee<6>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = NRTHandler::ScheduleTensorCast(src, dst, 128, 1, 2, &ret, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTHandlerTest, PlainKernelUsesPlainModelLoader) {
  MockCompilableKernelExecution kernel("plain_kernel", "", "", false);
  AttachCachedNeff(kernel);
  auto* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes_.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load_collectives(_, _, _, _, _, _, _)).Times(0);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model = NRTHandler::GetOrLoadModel(kernel, NRTHandler::ExecutionConfig(0, 1));

  EXPECT_EQ(model.get(), fake_model);
}

TEST_F(NRTHandlerTest, CollectiveKernelUsesCollectiveModelLoader) {
  MockCompilableKernelExecution kernel("collective_kernel", "", "", true);
  AttachCachedNeff(kernel);
  auto* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xBCDE);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load_collectives(_, neff_bytes_.size(), 0, 1, 1, 4, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_model), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model = NRTHandler::GetOrLoadModel(kernel, NRTHandler::ExecutionConfig(0, 1));

  EXPECT_EQ(model.get(), fake_model);
}

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
