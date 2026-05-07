/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
==============================================================================*/
#include <gtest/gtest.h>

#include <cstring>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

namespace at::neuron::utils {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

// Helper to build a mock nrt_instance_info_t
nrt_instance_info_t MakeInstanceInfo(uint32_t family, const char* revision = "") {
  nrt_instance_info_t info{};
  info.family = family;
  std::strncpy(info.device_revision, revision, sizeof(info.device_revision) - 1);
  return info;
}

class PlatformUtilsTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
};

// --- GetInstanceType tests ---

TEST_F(PlatformUtilsTest, GetInstanceType_Trn3) {
  auto info = MakeInstanceInfo(NRT_INSTANCE_TRN3);
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  EXPECT_EQ(GetInstanceType(), "trn3");
}

TEST_F(PlatformUtilsTest, GetInstanceType_Trn3PDS98) {
  auto info = MakeInstanceInfo(NRT_INSTANCE_TRN3PDS98);
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  EXPECT_EQ(GetInstanceType(), "trn3u");
}

TEST_F(PlatformUtilsTest, GetInstanceType_NrtFailure) {
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _)).WillOnce(Return(NRT_FAILURE));
  EXPECT_THROW(GetInstanceType(), std::runtime_error);
}

// --- GetPlatformTarget tests ---

TEST_F(PlatformUtilsTest, GetPlatformTarget_Trn3_NoRevision) {
  auto info = MakeInstanceInfo(NRT_INSTANCE_TRN3);
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  EXPECT_EQ(GetPlatformTarget(), "trn3");
}

TEST_F(PlatformUtilsTest, GetPlatformTarget_Trn3_B1Revision) {
  auto info = MakeInstanceInfo(NRT_INSTANCE_TRN3, "b1");
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  EXPECT_EQ(GetPlatformTarget(), "trn3");
}

TEST_F(PlatformUtilsTest, GetPlatformTarget_Trn2_NoRevision) {
  auto info = MakeInstanceInfo(NRT_INSTANCE_TRN2);
  EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  EXPECT_EQ(GetPlatformTarget(), "trn2");
}

}  // namespace
}  // namespace at::neuron::utils
