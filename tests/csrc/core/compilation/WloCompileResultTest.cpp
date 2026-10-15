#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"  // CompileResult

using namespace at::neuron;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace {

// Minimal non-WLO kernel: relies on the base class defaults for the WLO surface.
// Mirrors the base-ctor pattern used by MockNeuronKernelExecution in
// CompilationCacheTest.cpp.
class PlainKernel : public CompilableKernelExecution {
 public:
  explicit PlainKernel(const std::string& cache_key)
      : CompilableKernelExecution(cache_key, "", "", false) {}

  std::vector<uint8_t> CompileToNeff() const override { return {0xDE, 0xAD}; }
};

// WLO kernel: overrides the WLO surface to advertise + emit a wrapped_neff.hlo.
class WloKernel : public CompilableKernelExecution {
 public:
  explicit WloKernel(const std::string& cache_key)
      : CompilableKernelExecution(cache_key, "", "", false) {}

  std::vector<uint8_t> CompileToNeff() const override { return {0xDE, 0xAD}; }
  bool RequiresWrappedNeffHlo() const override { return true; }
  CompileResult CompileToNeffResult() const override {
    return CompileResult{CompileToNeff(), {0xBE, 0xEF}};
  }
};

}  // namespace

// Fixture mocks NRT instance info: constructing a CompilableKernelExecution builds
// a CompilationCacheKey, which queries NRT for the instance family. Without the mock
// that call fails ("Failed to get Neuron instance information") off a real device.
class WloCompileResultTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  void SetUp() override {
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  }
};

TEST_F(WloCompileResultTest, NonWloDefaultsToNoSidecar) {
  PlainKernel k("plain");
  // Default: not a WLO compile.
  EXPECT_FALSE(k.RequiresWrappedNeffHlo());
  // Default CompileToNeffResult wraps CompileToNeff() with an EMPTY sidecar.
  CompileResult r = k.CompileToNeffResult();
  EXPECT_EQ(r.neff_bytes, (std::vector<uint8_t>{0xDE, 0xAD}));
  EXPECT_TRUE(r.wrapped_neff_hlo.empty());
}

TEST_F(WloCompileResultTest, WloKernelAdvertisesAndEmitsSidecar) {
  WloKernel k("wlo");
  EXPECT_TRUE(k.RequiresWrappedNeffHlo());
  CompileResult r = k.CompileToNeffResult();
  EXPECT_EQ(r.neff_bytes, (std::vector<uint8_t>{0xDE, 0xAD}));
  EXPECT_EQ(r.wrapped_neff_hlo, (std::vector<uint8_t>{0xBE, 0xEF}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
