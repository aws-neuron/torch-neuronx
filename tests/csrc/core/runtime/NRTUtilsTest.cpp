#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

using namespace at::neuron::nrt;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class NRTUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    c10_neuron::reset_distributed_state();
  }

  void TearDown() override {
    mock_session_.reset();
    c10_neuron::reset_distributed_state();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
};

// TensorSet Tests
TEST_F(NRTUtilsTest, TensorSetConstruction) {
  nrt_tensor_set_t* fake_set = reinterpret_cast<nrt_tensor_set_t*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(1);

  TensorSet tensor_set;
  EXPECT_TRUE(tensor_set.IsValid());
  EXPECT_EQ(tensor_set.get(), fake_set);
}

TEST_F(NRTUtilsTest, TensorSetConstructionFailure) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_)).WillOnce(Return(NRT_FAILURE));

  EXPECT_THROW({ TensorSet tensor_set; }, std::runtime_error);
}

TEST_F(NRTUtilsTest, TensorSetMoveConstructor) {
  nrt_tensor_set_t* fake_set = reinterpret_cast<nrt_tensor_set_t*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(1);

  TensorSet tensor_set1;
  EXPECT_TRUE(tensor_set1.IsValid());

  TensorSet tensor_set2(std::move(tensor_set1));
  EXPECT_FALSE(tensor_set1.IsValid());
  EXPECT_TRUE(tensor_set2.IsValid());
  EXPECT_EQ(tensor_set2.get(), fake_set);
}

TEST_F(NRTUtilsTest, TensorSetMoveAssignment) {
  nrt_tensor_set_t* fake_set1 = reinterpret_cast<nrt_tensor_set_t*>(0x1234);
  nrt_tensor_set_t* fake_set2 = reinterpret_cast<nrt_tensor_set_t*>(0x5678);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(2);

  TensorSet tensor_set1;
  TensorSet tensor_set2;

  tensor_set2 = std::move(tensor_set1);
  EXPECT_FALSE(tensor_set1.IsValid());
  EXPECT_TRUE(tensor_set2.IsValid());
  EXPECT_EQ(tensor_set2.get(), fake_set1);
}

// Model Tests
TEST_F(NRTUtilsTest, ModelConstruction) {
  Model model;
  EXPECT_FALSE(model.IsValid());
  EXPECT_FALSE(model.IsLoaded());
}

TEST_F(NRTUtilsTest, ModelLoad) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};  // "NEFF"

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model;
  model.Load(neff_bytes, 0, 1);

  EXPECT_TRUE(model.IsValid());
  EXPECT_TRUE(model.IsLoaded());
  EXPECT_EQ(model.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelLoadFailure) {
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _)).WillOnce(Return(NRT_FAILURE));

  Model model;
  EXPECT_THROW(model.Load(neff_bytes, 0, 1), std::runtime_error);
  EXPECT_FALSE(model.IsValid());
}

TEST_F(NRTUtilsTest, ModelLoadCollectives) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  c10_neuron::set_rank(0);
  c10_neuron::set_world_size(2);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load_collectives(_, neff_bytes.size(), 0, 1, 0, 2, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model;
  model.LoadCollectives(neff_bytes, 0, 1);

  EXPECT_TRUE(model.IsValid());
  EXPECT_EQ(model.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelLoadCollectivesWithoutDistributedState) {
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  // It should throw if we have not set rank and world_size
  Model model;
  EXPECT_THROW(model.LoadCollectives(neff_bytes, 0, 1), std::runtime_error);
}

TEST_F(NRTUtilsTest, ModelExecute) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_tensor_set_t* fake_input = reinterpret_cast<nrt_tensor_set_t*>(0x1111);
  nrt_tensor_set_t* fake_output = reinterpret_cast<nrt_tensor_set_t*>(0x2222);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_execute(fake_model, fake_input, fake_output))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model;
  model.Load(neff_bytes, 0, 1);
  EXPECT_NO_THROW(model.Execute(fake_input, fake_output));
}

TEST_F(NRTUtilsTest, ModelExecuteFailure) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_tensor_set_t* fake_input = reinterpret_cast<nrt_tensor_set_t*>(0x1111);
  nrt_tensor_set_t* fake_output = reinterpret_cast<nrt_tensor_set_t*>(0x2222);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_execute(fake_model, fake_input, fake_output))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model;
  model.Load(neff_bytes, 0, 1);
  EXPECT_THROW(model.Execute(fake_input, fake_output), std::runtime_error);
}

TEST_F(NRTUtilsTest, ModelExecuteWithoutLoad) {
  nrt_tensor_set_t* fake_input = reinterpret_cast<nrt_tensor_set_t*>(0x1111);
  nrt_tensor_set_t* fake_output = reinterpret_cast<nrt_tensor_set_t*>(0x2222);

  Model model;
  EXPECT_THROW(model.Execute(fake_input, fake_output), std::runtime_error);
}

TEST_F(NRTUtilsTest, ModelExecuteWithNullTensorSets) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model;
  model.Load(neff_bytes, 0, 1);

  EXPECT_THROW(model.Execute(nullptr, nullptr), std::invalid_argument);
}

TEST_F(NRTUtilsTest, ModelMoveConstructor) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  Model model1;
  model1.Load(neff_bytes, 0, 1);
  EXPECT_TRUE(model1.IsValid());

  Model model2(std::move(model1));
  EXPECT_FALSE(model1.IsValid());
  EXPECT_TRUE(model2.IsValid());
  EXPECT_EQ(model2.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelMoveAssignment) {
  nrt_model_t* fake_model1 = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_model_t* fake_model2 = reinterpret_cast<nrt_model_t*>(0xDCBA);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  Model model1;
  model1.Load(neff_bytes, 0, 1);

  Model model2;
  model2.Load(neff_bytes, 1, 1);

  model2 = std::move(model1);
  EXPECT_FALSE(model1.IsValid());
  EXPECT_TRUE(model2.IsValid());
  EXPECT_EQ(model2.get(), fake_model1);
}

TEST_F(NRTUtilsTest, ModelReload) {
  nrt_model_t* fake_model1 = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_model_t* fake_model2 = reinterpret_cast<nrt_model_t*>(0xDCBA);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  Model model;
  model.Load(neff_bytes, 0, 1);
  EXPECT_EQ(model.get(), fake_model1);

  // Reload should unload the first model
  model.Load(neff_bytes, 1, 1);
  EXPECT_EQ(model.get(), fake_model2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
