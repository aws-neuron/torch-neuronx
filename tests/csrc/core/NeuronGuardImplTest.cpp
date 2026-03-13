#include <c10/core/DeviceCapability.h>
#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/NeuronGuardImpl.h"

namespace {

class NeuronGuardImplTest : public ::testing::Test {
 protected:
  c10_neuron::NeuronGuardImpl guard_impl_;
};

TEST_F(NeuronGuardImplTest, GetDeviceCapabilityReturnsExpectedDtypes) {
  c10::Device device(c10::DeviceType::PrivateUse1, 0);
  auto cap = guard_impl_.getDeviceCapability(device);

  // Verify supported dtypes
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Byte));      // uint8
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Char));      // int8
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Short));     // int16
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Int));       // int32
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Long));      // int64
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Float));     // float32
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Double));    // float64
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Half));      // float16
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_BFloat16));  // bfloat16
  EXPECT_TRUE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_Bool));
}

TEST_F(NeuronGuardImplTest, GetDeviceCapabilityExcludesComplexDtypes) {
  c10::Device device(c10::DeviceType::PrivateUse1, 0);
  auto cap = guard_impl_.getDeviceCapability(device);

  // Verify complex dtypes are not supported
  EXPECT_FALSE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_ComplexFloat));
  EXPECT_FALSE(cap.capability_data.capability_bits & (1ULL << c10::kIndex_ComplexDouble));
}

}  // namespace
