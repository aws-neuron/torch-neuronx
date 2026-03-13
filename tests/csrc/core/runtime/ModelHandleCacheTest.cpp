#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using ::testing::_;

class ModelHandleCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    cache_ = std::make_unique<ModelHandleCache>();

    neff_bytes_ = {0x4E, 0x45, 0x46, 0x46, 0x01, 0x02, 0x03, 0x04};
  }

  void TearDown() override {
    cache_.reset();
    mock_session_.reset();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  std::unique_ptr<ModelHandleCache> cache_;
  std::vector<uint8_t> neff_bytes_;
};

TEST_F(ModelHandleCacheTest, InitialState) {
  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 0);
}

TEST_F(ModelHandleCacheTest, GetModelMiss) {
  std::string key = cache_->GenerateCacheKey(neff_bytes_, 0);
  auto model = cache_->GetModel(key);
  EXPECT_EQ(model, nullptr);
}

TEST_F(ModelHandleCacheTest, StoreAndGetModel) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  std::string key = cache_->GenerateCacheKey(neff_bytes_, 0);
  std::shared_ptr<nrt::ModelHandle> model(fake_model, nrt::ModelDeleter{});

  auto stored = cache_->StoreModel(key, model, 0, false);
  EXPECT_EQ(stored.get(), fake_model);

  auto retrieved = cache_->GetModel(key);
  EXPECT_EQ(retrieved.get(), fake_model);

  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(ModelHandleCacheTest, StoreModelRaceReturnsExisting) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  std::string key = cache_->GenerateCacheKey(neff_bytes_, 0);
  std::shared_ptr<nrt::ModelHandle> model1(fake_model1, nrt::ModelDeleter{});
  std::shared_ptr<nrt::ModelHandle> model2(fake_model2, nrt::ModelDeleter{});

  auto stored1 = cache_->StoreModel(key, model1, 0, false);
  EXPECT_EQ(stored1.get(), fake_model1);

  // Second store with same key returns the first one
  auto stored2 = cache_->StoreModel(key, model2, 0, false);
  EXPECT_EQ(stored2.get(), fake_model1);

  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 1);
}

TEST_F(ModelHandleCacheTest, DifferentDevicesAreDifferentKeys) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  std::string key0 = cache_->GenerateCacheKey(neff_bytes_, 0);
  std::string key1 = cache_->GenerateCacheKey(neff_bytes_, 1);
  EXPECT_NE(key0, key1);

  cache_->StoreModel(key0, std::shared_ptr<nrt::ModelHandle>(fake_model1, nrt::ModelDeleter{}), 0,
                     false);
  cache_->StoreModel(key1, std::shared_ptr<nrt::ModelHandle>(fake_model2, nrt::ModelDeleter{}), 1,
                     false);

  EXPECT_EQ(cache_->GetModel(key0).get(), fake_model1);
  EXPECT_EQ(cache_->GetModel(key1).get(), fake_model2);

  auto keys = cache_->GetAllCacheKeys();
  EXPECT_EQ(keys.size(), 2);
}

TEST_F(ModelHandleCacheTest, ClearCache) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  std::string key = cache_->GenerateCacheKey(neff_bytes_, 0);
  cache_->StoreModel(key, std::shared_ptr<nrt::ModelHandle>(fake_model, nrt::ModelDeleter{}), 0,
                     false);

  EXPECT_FALSE(cache_->GetCacheEntries().empty());

  cache_->Clear();

  EXPECT_TRUE(cache_->GetCacheEntries().empty());
}

TEST_F(ModelHandleCacheTest, GetCacheEntries) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  std::string key0 = cache_->GenerateCacheKey(neff_bytes_, 0);
  std::string key1 = cache_->GenerateCacheKey(neff_bytes_, 1);

  cache_->StoreModel(key0, std::shared_ptr<nrt::ModelHandle>(fake_model1, nrt::ModelDeleter{}), 0,
                     false);
  cache_->StoreModel(key1, std::shared_ptr<nrt::ModelHandle>(fake_model2, nrt::ModelDeleter{}), 1,
                     true);

  auto entries = cache_->GetCacheEntries();
  EXPECT_EQ(entries.size(), 2);

  for (const auto& entry : entries) {
    EXPECT_FALSE(entry.cache_key.empty());
    EXPECT_TRUE(entry.device_index == 0 || entry.device_index == 1);
  }
}

TEST_F(ModelHandleCacheTest, MultipleNeffBytes) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  std::vector<uint8_t> neff_bytes1 = {0x4E, 0x45, 0x46, 0x46, 0x01};
  std::vector<uint8_t> neff_bytes2 = {0x4E, 0x45, 0x46, 0x46, 0x02, 0x03};

  std::string key1 = cache_->GenerateCacheKey(neff_bytes1, 0);
  std::string key2 = cache_->GenerateCacheKey(neff_bytes2, 0);
  EXPECT_NE(key1, key2);

  cache_->StoreModel(key1, std::shared_ptr<nrt::ModelHandle>(fake_model1, nrt::ModelDeleter{}), 0,
                     false);
  cache_->StoreModel(key2, std::shared_ptr<nrt::ModelHandle>(fake_model2, nrt::ModelDeleter{}), 0,
                     false);

  EXPECT_EQ(cache_->GetModel(key1).get(), fake_model1);
  EXPECT_EQ(cache_->GetModel(key2).get(), fake_model2);
  EXPECT_EQ(cache_->GetCacheEntries().size(), 2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
