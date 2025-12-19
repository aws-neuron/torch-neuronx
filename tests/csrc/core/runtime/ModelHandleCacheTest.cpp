#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class ModelHandleCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    cache_ = std::make_unique<ModelHandleCache>();
    c10_neuron::reset_distributed_state();

    // Create test tensors
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};
    input_ptrs_ = get_tensor_data_ptrs(inputs_);
    output_ptrs_ = get_tensor_data_ptrs(outputs_);

    // Create test NEFF bytes
    neff_bytes_ = {0x4E, 0x45, 0x46, 0x46, 0x01, 0x02, 0x03, 0x04};
  }

  void TearDown() override {
    cache_.reset();
    mock_session_.reset();
    c10_neuron::reset_distributed_state();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  std::unique_ptr<ModelHandleCache> cache_;
  torch::Tensor input_tensor_;
  torch::Tensor output_tensor_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
  std::vector<void*> input_ptrs_;
  std::vector<void*> output_ptrs_;
  std::vector<uint8_t> neff_bytes_;
};

TEST_F(ModelHandleCacheTest, GetOrLoadModelFirstTime) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);

  // Create a compilable kernel with cached NEFF
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  // Set cached NEFF
  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes_.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model = cache_->GetOrLoadModel(kernel, 0, 1);

  EXPECT_TRUE(model);
  EXPECT_TRUE(model->IsValid());
  EXPECT_EQ(model->get(), fake_model);
}

TEST_F(ModelHandleCacheTest, GetOrLoadModelCacheHit) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  // First load
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes_.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model1 = cache_->GetOrLoadModel(kernel, 0, 1);
  EXPECT_TRUE(model1);

  // Second load - should hit cache (no additional nrt_load call)
  auto model2 = cache_->GetOrLoadModel(kernel, 0, 1);
  EXPECT_TRUE(model2);
  EXPECT_EQ(model1, model2);  // Should be the same shared_ptr
}

TEST_F(ModelHandleCacheTest, GetOrLoadModelDifferentDevices) {
  nrt_model_t* fake_model1 = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_model_t* fake_model2 = reinterpret_cast<nrt_model_t*>(0xDCBA);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  // Load on device 0
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes_.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)));

  // Load on device 1
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes_.size(), 1, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  auto model1 = cache_->GetOrLoadModel(kernel, 0, 1);
  auto model2 = cache_->GetOrLoadModel(kernel, 1, 1);

  EXPECT_TRUE(model1);
  EXPECT_TRUE(model2);
  EXPECT_NE(model1, model2);  // Different devices = different models
}

TEST_F(ModelHandleCacheTest, GetOrLoadModelWithCollectives) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);

  c10_neuron::set_rank(0);
  c10_neuron::set_world_size(2);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      true, 0);  // has_collectives = true, device_id = 0

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load_collectives(_, neff_bytes_.size(), 0, 1, 0, 2, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model = cache_->GetOrLoadModel(kernel, 0, 1);

  EXPECT_TRUE(model);
  EXPECT_TRUE(model->IsValid());
}

TEST_F(ModelHandleCacheTest, GetOrLoadModelLoadFailure) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _)).WillOnce(Return(NRT_FAILURE));
  auto model = cache_->GetOrLoadModel(kernel, 0, 1);
  EXPECT_FALSE(model->IsValid());
}

TEST_F(ModelHandleCacheTest, ClearCache) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  auto model = cache_->GetOrLoadModel(kernel, 0, 1);
  EXPECT_TRUE(model);

  // Verify cache has entries by checking we can retrieve cached models
  auto entries_before = cache_->GetCacheEntries();
  EXPECT_FALSE(entries_before.empty());

  cache_->Clear();

  // Verify cache is empty after clear
  auto entries_after = cache_->GetCacheEntries();
  EXPECT_TRUE(entries_after.empty());
}

TEST_F(ModelHandleCacheTest, GetCacheEntries) {
  nrt_model_t* fake_model1 = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_model_t* fake_model2 = reinterpret_cast<nrt_model_t*>(0xDCBA);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, 1, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  cache_->GetOrLoadModel(kernel, 0, 1);
  cache_->GetOrLoadModel(kernel, 1, 1);

  auto entries = cache_->GetCacheEntries();
  EXPECT_EQ(entries.size(), 2);

  // Verify entry information
  for (const auto& entry : entries) {
    EXPECT_FALSE(entry.cache_key.empty());
    EXPECT_FALSE(entry.has_collectives);
    EXPECT_TRUE(entry.device_index == 0 || entry.device_index == 1);
  }
}

TEST_F(ModelHandleCacheTest, GetAllCacheKeys) {
  nrt_model_t* fake_model = reinterpret_cast<nrt_model_t*>(0xABCD);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("test_op", input_ptrs_, output_ptrs_, "cache_key", hlo_bytes,
                                      false, 0);

  auto neff_data = new std::vector<uint8_t>(neff_bytes_);
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});
  kernel.SetCachedNeff(std::move(neff_ptr));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  cache_->GetOrLoadModel(kernel, 0, 1);

  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
  EXPECT_FALSE(keys.begin()->empty());
}

TEST_F(ModelHandleCacheTest, MultipleNeffBytes) {
  nrt_model_t* fake_model1 = reinterpret_cast<nrt_model_t*>(0xABCD);
  nrt_model_t* fake_model2 = reinterpret_cast<nrt_model_t*>(0xDCBA);

  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Create two kernels with different NEFF bytes (different sizes to distinguish in mock)
  XLACompilableKernelExecution kernel1("test_op1", input_ptrs_, output_ptrs_, "cache_key1",
                                       hlo_bytes, false, 0);
  XLACompilableKernelExecution kernel2("test_op2", input_ptrs_, output_ptrs_, "cache_key2",
                                       hlo_bytes, false, 0);

  std::vector<uint8_t> neff_bytes1 = {0x4E, 0x45, 0x46, 0x46, 0x01};
  std::vector<uint8_t> neff_bytes2 = {0x4E, 0x45, 0x46, 0x46, 0x02, 0x03};  // Different size

  auto neff_data1 = new std::vector<uint8_t>(neff_bytes1);
  NeffBytesPtr neff_ptr1(neff_data1, CacheEntryGuard{});
  kernel1.SetCachedNeff(std::move(neff_ptr1));

  auto neff_data2 = new std::vector<uint8_t>(neff_bytes2);
  NeffBytesPtr neff_ptr2(neff_data2, CacheEntryGuard{});
  kernel2.SetCachedNeff(std::move(neff_ptr2));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes1.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes2.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  auto model1 = cache_->GetOrLoadModel(kernel1, 0, 1);
  auto model2 = cache_->GetOrLoadModel(kernel2, 0, 1);

  EXPECT_TRUE(model1);
  EXPECT_TRUE(model2);
  EXPECT_NE(model1, model2);  // Different NEFF = different models

  // Verify we have multiple cache entries by checking GetCacheEntries
  auto entries = cache_->GetCacheEntries();
  EXPECT_EQ(entries.size(), 2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
