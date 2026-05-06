#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <libkineto.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/profiler/Profiler.h"

namespace at::neuron {
namespace {

class ProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
  }

  void TearDown() override { mock_session_.reset(); }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

TEST_F(ProfilerTest, ConfigureReturnsNullptrWhenCustomConfigEmpty) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
      libkineto::ActivityType::PRIVATEUSE1_DRIVER,
  };
  libkineto::Config config;
  // No custom config set -- default is empty

  auto session = profiler.configure(activityTypes, config);
  EXPECT_EQ(session, nullptr);
}

TEST_F(ProfilerTest, ConfigureReturnsNullptrWhenNoNeuronProfileKey) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  libkineto::Config config;
  std::string customConfig = "max_events_per_nc:100000";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  auto session = profiler.configure(activityTypes, config);
  EXPECT_EQ(session, nullptr);
}

TEST_F(ProfilerTest, ConfigureReturnsSessionWhenNeuronProfileKeyPresent) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device,runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  auto session = profiler.configure(activityTypes, config);
  EXPECT_NE(session, nullptr);
}

TEST_F(ProfilerTest, AsyncConfigureDelegatesToSyncConfigure) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  libkineto::Config config;
  // No neuron_profile key -- should return nullptr from async path too

  auto session = profiler.configure(0, 1000, activityTypes, config);
  EXPECT_EQ(session, nullptr);
}

TEST_F(ProfilerTest, AsyncConfigureReturnsSessionWhenNeuronProfileKeyPresent) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  auto session = profiler.configure(0, 1000, activityTypes, config);
  EXPECT_NE(session, nullptr);
}

TEST_F(ProfilerTest, ConfigureReturnsSessionWhenNeuronProfileKeyWithExtraConfig) {
  NeuronActivityProfiler profiler;

  std::set<libkineto::ActivityType> activityTypes = {
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device;max_events_per_nc:50000";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  auto session = profiler.configure(activityTypes, config);
  EXPECT_NE(session, nullptr);
}

}  // namespace
}  // namespace at::neuron
