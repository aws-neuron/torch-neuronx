#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <libkineto.h>
#include <output_base.h>

#include <chrono>
#include <set>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"
#include "torch_neuronx/csrc/profiler/Session.h"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrEq;

namespace at::neuron {
namespace {

// Fake nrt_inspect_config for testing
struct FakeNrtInspectConfig {
  int dummy;
};

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
  }

  void TearDown() override {
    setActiveProfilerBuffer(nullptr);
    mock_session_.reset();
  }

  // Helper: expects successful config allocation sequence.
  void expectConfigAllocation() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_allocate(_))
        .WillOnce(DoAll(SetArgPointee<0>(reinterpret_cast<nrt_inspect_config_t*>(&fake_config_)),
                        Return(NRT_SUCCESS)));
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_defaults(_)).WillOnce(Return(NRT_SUCCESS));
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_enable_inspect(_, true))
        .WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect activity type setup calls (baseline: all off)
  void expectActivityTypeSetup() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
        .WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect full successful start sequence
  void expectSuccessfulStart() {
    expectConfigAllocation();
    expectActivityTypeSetup();
    EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect cleanup
  void expectCleanup() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_free(_)).WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect stop
  void expectStop() { EXPECT_CALL(*mock_nrt_, nrt_inspect_stop()).WillOnce(Return(NRT_SUCCESS)); }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
  FakeNrtInspectConfig fake_config_;
};

TEST_F(SessionTest, StartStopSuccess) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileDeviceOnlyEnablesDeviceProfile) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));

  // Ensure system_profile is NOT enabled
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .Times(0);

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileRuntimeOnlyEnablesSystemProfile) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));

  // Ensure device_profile is NOT enabled
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .Times(0);

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileDeviceAndRuntimeEnablesBoth) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device,runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileCpuUtilEnablesCpuUtil) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:cpu_util";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("cpu_util"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileHostMemoryEnablesHostMemory) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:host_memory";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("host_memory"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileAllModesEnablesAll) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device,runtime,cpu_util,host_memory";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("cpu_util"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("host_memory"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, ErrorsReturnsAccumulatedErrors) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_allocate(_)).WillOnce(Return(NRT_FAILURE));

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  std::vector<std::string> errors = session.errors();
  EXPECT_FALSE(errors.empty());
}

TEST_F(SessionTest, StartInspectBeginFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectConfigAllocation();
  expectActivityTypeSetup();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_FAILURE));
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  EXPECT_FALSE(session.errors().empty());
}

TEST_F(SessionTest, StopInspectStopFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_stop()).WillOnce(Return(NRT_FAILURE));
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_FALSE(session.errors().empty());
}

// This establishes that starting another session when one is
// active is possible today.
TEST_F(SessionTest, StartDuplicateIgnored) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // Only expect one start sequence
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.start();  // Second start should be ignored
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StopWhenNotActive) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // No NRT calls expected - session never started

  NeuronActivityProfilerSession session(activityTypes, config);
  session.stop();  // Should be no-op

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, DestructorCleansUpActiveSession) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  // Destructor should call stop and cleanup
  expectStop();
  expectCleanup();

  {
    NeuronActivityProfilerSession session(activityTypes, config);
    session.start();
    // Session destroyed without explicit stop()
  }
}

TEST_F(SessionTest, DestructorDoesNotStopInactiveSession) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // No NRT calls expected for inactive session destruction

  {
    NeuronActivityProfilerSession session(activityTypes, config);
  }
}

TEST_F(SessionTest, StartWithCustomConfigProfileOutputDir) {
  // Custom config profile_output_dir should call nrt_inspect_config_set_output_dir.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "profile_output_dir:/tmp/neuron_traces";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  expectActivityTypeSetup();

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_output_dir(_, StrEq("/tmp/neuron_traces")))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileWithExtraConfig) {
  // neuron_profile key combined with other config keys
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device;max_events_per_nc:50000";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_sys_trace_max_events_per_nc(_, 50000))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithCustomConfigCaptureEnabledForNc) {
  // capture_enabled_for_nc should call nrt_inspect_config_set_capture_enabled_for_nc
  // for each expanded index.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "capture_enabled_for_nc:0,2-4";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  expectActivityTypeSetup();

  // Expect calls for NC indices 0, 2, 3, 4
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_capture_enabled_for_nc(_, 0, true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_capture_enabled_for_nc(_, 2, true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_capture_enabled_for_nc(_, 3, true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_capture_enabled_for_nc(_, 4, true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileDeviceActivityFailureContinues) {
  // When nrt_inspect_config_set_activity fails for a mode, the applier should
  // continue processing remaining modes and the session should still start.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device,runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  // device_profile fails
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_FAILURE));
  // device profile mode still attempted
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  // runtime should still be enabled
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  // Session still works despite partial failure in applier
  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeuronProfileUnknownModeIgnored) {
  // Unknown modes in neuron_profile should be silently ignored (logged as warning).
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:unknown_mode";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  // No additional activity calls expected for an unknown mode

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, AccessorsReturnDefaults) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  NeuronActivityProfilerSession session(activityTypes, config);

  auto deviceInfo = session.getDeviceInfo();
  EXPECT_NE(deviceInfo, nullptr);
  EXPECT_EQ(deviceInfo->id, 0);
  EXPECT_EQ(deviceInfo->sortIndex, libkineto::kExceedMaxPid);
  EXPECT_EQ(deviceInfo->label, "NeuronDevice 0");
  EXPECT_TRUE(session.getResourceInfos().empty());
  auto traceBuffer = session.getTraceBuffer();
  EXPECT_NE(traceBuffer, nullptr);
  EXPECT_TRUE(traceBuffer->activities.empty());
}

// Concurrent session starts.
TEST_F(SessionTest, ConcurrentStartCalls) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // Only expect one full start sequence
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);

  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&session]() { session.start(); });
  }

  for (auto& t : threads) {
    t.join();
  }

  session.stop();
  EXPECT_TRUE(session.errors().empty());
}

// Minimal no-op ActivityLogger for processTrace tests
class NoOpLogger : public libkineto::ActivityLogger {
 public:
  void handleDeviceInfo(const libkineto::DeviceInfo&, uint64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(const OverheadInfo&, int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override { activityCount_++; }
  void handleTraceStart(const std::unordered_map<std::string, std::string>&,
                        const std::string&) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&) override {}
  void finalizeTrace(const libkineto::Config&, std::unique_ptr<libkineto::ActivityBuffers>, int64_t,
                     std::unordered_map<std::string, std::vector<std::string>>&) override {}

  int activityCount() const { return activityCount_; }

 private:
  int activityCount_{0};
};

TEST_F(SessionTest, StartSetsGlobalBufferStopClearsIt) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);

  // Before start, global buffer should be null
  EXPECT_EQ(getActiveProfilerBuffer(), nullptr);

  session.start();

  // After start, global buffer should be set
  EXPECT_NE(getActiveProfilerBuffer(), nullptr);

  session.stop();

  // After stop, global buffer should be cleared
  EXPECT_EQ(getActiveProfilerBuffer(), nullptr);
}

TEST_F(SessionTest, DestructorClearsGlobalBuffer) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  {
    NeuronActivityProfilerSession session(activityTypes, config);
    session.start();
    EXPECT_NE(getActiveProfilerBuffer(), nullptr);
    // Session destroyed without explicit stop()
  }

  // After destruction, global buffer should be cleared
  EXPECT_EQ(getActiveProfilerBuffer(), nullptr);
}

TEST_F(SessionTest, ProcessTraceDrainsBufferAndPopulatesActivities) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  // Record a profiled operation into the global buffer
  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "test_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.dispatch_start_ns = 1500;
  op.kineto_correlation_id = 42;
  op.pytorch_thread_id = 100;
  op.device_index = 0;
  op.stream_id = 1;
  op.xu_type = 1;  // NRTA_XU_COMPUTE
  op.dispatch_slot = 0;
  op.neuron_stream_os_tid = 9999;  // simulated worker thread OS tid
  op.pytorch_os_tid = 8888;        // simulated PyTorch calling thread OS tid
  buf->record(std::move(op));

  session.stop();

  // processTrace should drain the buffer and create activities
  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  // processTrace emits 3 activities per device operation (submit + dispatch + driver)
  EXPECT_EQ(logger.activityCount(), 3);

  // getTraceBuffer should return the populated activities
  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 3u);

  // Activity 1: submit (PRIVATEUSE1_RUNTIME, submit_start -> submit_time)
  // Placed on CPU process with PyTorch calling thread's OS tid
  auto& submit_act = *traceBuffer->activities[0];
  EXPECT_EQ(submit_act.name(), "test_op");
  EXPECT_EQ(submit_act.type(), libkineto::ActivityType::PRIVATEUSE1_RUNTIME);
  EXPECT_EQ(submit_act.endTime - submit_act.startTime, 500);  // submit_time - submit_start
  EXPECT_GT(submit_act.startTime, 0);
  EXPECT_EQ(submit_act.id, 42);
  EXPECT_EQ(submit_act.device, libkineto::processId());  // CPU process PID
  EXPECT_EQ(submit_act.resource, 8888);                  // pytorch_os_tid
  EXPECT_EQ(submit_act.threadId, 8888);                  // pytorch_os_tid

  // Activity 2: dispatch (PRIVATEUSE1_RUNTIME, dispatch_start -> execute_end)
  // Placed on virtual slot tid
  auto& dispatch_act = *traceBuffer->activities[1];
  EXPECT_EQ(dispatch_act.name(), "test_op");
  EXPECT_EQ(dispatch_act.type(), libkineto::ActivityType::PRIVATEUSE1_RUNTIME);
  EXPECT_EQ(dispatch_act.endTime - dispatch_act.startTime, 1500);  // execute_end - dispatch_start

  // Verify flow arrow fields on dispatch activity (DISPATCH -> DRIVER)
  EXPECT_GT(dispatch_act.flow.id, 0u);
  EXPECT_EQ(dispatch_act.flow.type, libkineto::kLinkAsyncCpuGpu);
  EXPECT_EQ(dispatch_act.flow.start, 1u);

  // Activity 3: execution (PRIVATEUSE1_DRIVER, execute_start -> execute_end)
  auto& exec_act = *traceBuffer->activities[2];
  EXPECT_EQ(exec_act.name(), "test_op");
  EXPECT_EQ(exec_act.type(), libkineto::ActivityType::PRIVATEUSE1_DRIVER);
  EXPECT_EQ(exec_act.endTime - exec_act.startTime, 1000);  // execute_end - execute_start
  EXPECT_EQ(exec_act.id, 42);
  EXPECT_EQ(exec_act.device, 0);  // device_index

  // Verify flow arrow fields on execution activity (flow END, paired with dispatch)
  EXPECT_EQ(exec_act.flow.id, dispatch_act.flow.id);
  EXPECT_EQ(exec_act.flow.type, libkineto::kLinkAsyncCpuGpu);
  EXPECT_EQ(exec_act.flow.start, 0u);

  // Verify resource info: PyTorch thread + dispatch slot + device XU row
  auto resources = session.getResourceInfos();
  EXPECT_GE(resources.size(), 3u);
}

TEST_F(SessionTest, ProcessTraceMultipleOperations) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Record multiple operations
  for (int i = 0; i < 3; ++i) {
    ProfiledOperation op;
    op.op_name = "op_" + std::to_string(i);
    op.submit_start_ns = i * 1000;
    op.submit_time_ns = i * 1000 + 200;
    op.execute_start_ns = i * 1000 + 500;
    op.execute_end_ns = i * 1000 + 900;
    op.kineto_correlation_id = i + 1;
    op.pytorch_thread_id = 200;
    op.device_index = 0;
    op.stream_id = 0;
    op.neuron_stream_os_tid = 7777;
    op.pytorch_os_tid = 5000;
    buf->record(std::move(op));
  }

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  // 3 ops * 2 activities each (dispatch + execution) = 6
  EXPECT_EQ(logger.activityCount(), 6);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  EXPECT_EQ(traceBuffer->activities.size(), 6u);
}

TEST_F(SessionTest, ProcessTraceLinkedCpuActivity) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "linked_op";
  op.submit_start_ns = 4500;
  op.submit_time_ns = 5000;
  op.execute_start_ns = 6000;
  op.execute_end_ns = 7000;
  op.kineto_correlation_id = 99;
  op.pytorch_thread_id = 300;
  op.device_index = 0;
  op.stream_id = 2;
  op.neuron_stream_os_tid = 8888;
  op.pytorch_os_tid = 7777;
  buf->record(std::move(op));

  session.stop();

  // Create a fake CPU activity to be linked
  libkineto::TraceSpan cpuSpan(0, INT64_MAX, "cpu");
  libkineto::GenericTraceActivity cpuAct(cpuSpan, libkineto::ActivityType::CPU_OP, "aten::add");
  cpuAct.id = 99;

  NoOpLogger logger;
  auto linkedActivityCb = [&cpuAct](int32_t correlationId) -> const libkineto::ITraceActivity* {
    if (correlationId == 99) {
      return &cpuAct;
    }
    return nullptr;
  };
  session.processTrace(logger, linkedActivityCb, 0, INT64_MAX);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);

  // Dispatch activity should have the linked CPU activity set
  auto& dispatch_act = *traceBuffer->activities[0];
  EXPECT_NE(dispatch_act.linkedActivity(), nullptr);
  EXPECT_EQ(dispatch_act.linkedActivity()->correlationId(), 99);
}

TEST_F(SessionTest, GetTraceBufferEmptyWithoutProcessTrace) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  // getTraceBuffer without processTrace should return empty activities
  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  EXPECT_TRUE(traceBuffer->activities.empty());
}

// Verify that starting a second session while one is active is rejected
// by the cross-session s_sessionActive guard.
TEST_F(SessionTest, SecondSessionRejectedWhileFirstIsActive) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // First session starts successfully
  expectSuccessfulStart();
  expectStop();

  NeuronActivityProfilerSession session1(activityTypes, config);
  session1.start();
  EXPECT_TRUE(session1.errors().empty());

  // Second session: constructor allocates config and sets baseline activity (succeeds),
  // but start() is rejected by the cross-session guard. Config is freed in destructor.
  expectConfigAllocation();
  expectActivityTypeSetup();

  // Both sessions free their config in destructors
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_free(_)).Times(2).WillRepeatedly(Return(NRT_SUCCESS));

  NeuronActivityProfilerSession session2(activityTypes, config);
  session2.start();
  EXPECT_FALSE(session2.errors().empty());

  // Stop first session
  session1.stop();
}

// Verify that after a session fails to start (config allocation failure),
// the cross-session guard is cleared and a subsequent session can start.
TEST_F(SessionTest, SessionCanStartAfterPreviousFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // First session: config allocation fails in constructor
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_allocate(_)).WillOnce(Return(NRT_FAILURE));

  NeuronActivityProfilerSession session1(activityTypes, config);
  session1.start();
  EXPECT_FALSE(session1.errors().empty());

  // Second session should succeed (guard was cleared on failure)
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session2(activityTypes, config);
  session2.start();
  EXPECT_TRUE(session2.errors().empty());
  session2.stop();
}

// Verify that after nrt_inspect_begin_with_options fails, the guard is cleared.
TEST_F(SessionTest, SessionCanStartAfterBeginFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // First session: begin fails
  expectConfigAllocation();
  expectActivityTypeSetup();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_FAILURE));
  // Config still gets freed in destructor
  expectCleanup();

  {
    NeuronActivityProfilerSession session1(activityTypes, config);
    session1.start();
    EXPECT_FALSE(session1.errors().empty());
  }

  // Second session should succeed
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session2(activityTypes, config);
  session2.start();
  EXPECT_TRUE(session2.errors().empty());
  session2.stop();
}

TEST_F(SessionTest, CustomConfigCpuUtilEnablesActivity) {
  // cpu_util should be configurable via custom config, not via activity types.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "cpu_util:true";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  expectActivityTypeSetup();

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("cpu_util"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, ProcessTraceFiltersCaptureWindow) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Use real clock readings for timestamps.
  // steady_clock for op timestamps (as in production), system_clock for capture window.
  // The session's clockDomainOffset_ bridges the two domains in processTrace().
  auto steadyNow = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  auto sysNow = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

  // Op 1: entirely before window (ends before sysNow)
  ProfiledOperation op1;
  op1.op_name = "before_window";
  op1.submit_start_ns = steadyNow - 3500;
  op1.submit_time_ns = steadyNow - 3000;
  op1.execute_start_ns = steadyNow - 2000;
  op1.execute_end_ns = steadyNow - 1000;
  op1.kineto_correlation_id = 1;
  op1.device_index = 0;
  op1.stream_id = 0;
  op1.neuron_stream_os_tid = 5555;
  op1.pytorch_os_tid = 5000;
  buf->record(std::move(op1));

  // Op 2: inside window
  ProfiledOperation op2;
  op2.op_name = "inside_window";
  op2.submit_start_ns = steadyNow + 500;
  op2.submit_time_ns = steadyNow + 1000;
  op2.execute_start_ns = steadyNow + 2000;
  op2.execute_end_ns = steadyNow + 3000;
  op2.kineto_correlation_id = 2;
  op2.device_index = 0;
  op2.stream_id = 0;
  op2.neuron_stream_os_tid = 5555;
  op2.pytorch_os_tid = 5000;
  buf->record(std::move(op2));

  // Op 3: entirely after window (starts well after sysNow + 5000)
  ProfiledOperation op3;
  op3.op_name = "after_window";
  op3.submit_start_ns = steadyNow + 7500;
  op3.submit_time_ns = steadyNow + 8000;
  op3.execute_start_ns = steadyNow + 9000;
  op3.execute_end_ns = steadyNow + 10000;
  op3.kineto_correlation_id = 3;
  op3.device_index = 0;
  op3.stream_id = 0;
  op3.neuron_stream_os_tid = 5555;
  op3.pytorch_os_tid = 5000;
  buf->record(std::move(op3));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  // Capture window in system_clock domain: sysNow to sysNow + 5000.
  // After offset conversion in processTrace():
  //   op1 ends at ~sysNow-1000 (before window) -> filtered
  //   op2 spans ~sysNow+1000 to ~sysNow+3000 (inside window) -> kept
  //   op3 starts at ~sysNow+8000 (after window) -> filtered
  session.processTrace(logger, noLinkedActivity, sysNow, sysNow + 5000);

  // 1 op * 2 activities each = 2
  EXPECT_EQ(logger.activityCount(), 2);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);
  EXPECT_EQ(traceBuffer->activities[0]->name(), "inside_window");
}

TEST_F(SessionTest, ProcessTraceCaptureWindowBoundaryOverlap) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Use real clock readings (see ProcessTraceFiltersCaptureWindow for rationale).
  auto steadyNow = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  auto sysNow = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

  // Op that straddles the start boundary: submit before window, execute_end inside window
  ProfiledOperation op1;
  op1.op_name = "straddles_start";
  op1.submit_start_ns = steadyNow - 1500;
  op1.submit_time_ns = steadyNow - 1000;
  op1.execute_start_ns = steadyNow + 500;
  op1.execute_end_ns = steadyNow + 1500;
  op1.kineto_correlation_id = 1;
  op1.device_index = 0;
  op1.stream_id = 0;
  op1.neuron_stream_os_tid = 6666;
  op1.pytorch_os_tid = 5000;
  buf->record(std::move(op1));

  // Op that straddles the end boundary: submit inside window, execute_end after window
  ProfiledOperation op2;
  op2.op_name = "straddles_end";
  op2.submit_start_ns = steadyNow + 1500;
  op2.submit_time_ns = steadyNow + 2000;
  op2.execute_start_ns = steadyNow + 3000;
  op2.execute_end_ns = steadyNow + 5000;
  op2.kineto_correlation_id = 2;
  op2.device_index = 0;
  op2.stream_id = 0;
  op2.neuron_stream_os_tid = 6666;
  op2.pytorch_os_tid = 5000;
  buf->record(std::move(op2));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  // Capture window: sysNow to sysNow + 4000.
  // After offset conversion:
  //   op1: submit ~sysNow-1000 (before), execute_end ~sysNow+1500 (inside) -> kept
  //   op2: submit ~sysNow+2000 (inside), execute_end ~sysNow+5000 (after) -> kept
  session.processTrace(logger, noLinkedActivity, sysNow, sysNow + 4000);

  // 2 ops * 2 activities each = 4
  EXPECT_EQ(logger.activityCount(), 4);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 4u);
  EXPECT_EQ(traceBuffer->activities[0]->name(), "straddles_start");
  EXPECT_EQ(traceBuffer->activities[2]->name(), "straddles_end");
}

TEST_F(SessionTest, GetDeviceInfoReflectsPrimaryDevice) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Record operation on device 3
  ProfiledOperation op;
  op.op_name = "device3_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.kineto_correlation_id = 1;
  op.device_index = 3;
  op.stream_id = 0;
  op.neuron_stream_os_tid = 4444;
  op.pytorch_os_tid = 5000;
  buf->record(std::move(op));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  // After processTrace, getDeviceInfo should reflect device_index from first operation
  auto deviceInfo = session.getDeviceInfo();
  EXPECT_NE(deviceInfo, nullptr);
  EXPECT_EQ(deviceInfo->id, 3);
  EXPECT_EQ(deviceInfo->label, "NeuronDevice 3");
}

TEST_F(SessionTest, ProcessTraceSyncOverloadUsesFullCaptureWindow) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Use real clock timestamps so the operation falls within the capture window.
  // profileStartTime_ and profileEndTime_ are in system_clock domain after the
  // clock domain fix; op timestamps are in steady_clock and get converted in processTrace.
  auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
  ProfiledOperation op;
  op.op_name = "sync_trace_op";
  op.submit_start_ns = now_ns - 500;
  op.submit_time_ns = now_ns;
  op.execute_start_ns = now_ns + 1000;
  op.execute_end_ns = now_ns + 2000;
  op.kineto_correlation_id = 1;
  op.device_index = 0;
  op.stream_id = 0;
  op.neuron_stream_os_tid = 3333;
  op.pytorch_os_tid = 5000;
  buf->record(std::move(op));

  session.stop();

  // Use the sync overload (no callback, no explicit window)
  NoOpLogger logger;
  session.processTrace(logger);

  // The sync overload should use profileStartTime_..profileEndTime_ as window,
  // which should include the operation recorded during the active session.
  EXPECT_EQ(logger.activityCount(), 2);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);
  EXPECT_EQ(traceBuffer->activities[0]->name(), "sync_trace_op");
}

TEST_F(SessionTest, GetTraceBufferConsumedOnSecondCall) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;
  std::string customConfig = "neuron_profile:runtime";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "consumed_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.kineto_correlation_id = 1;
  op.device_index = 0;
  op.stream_id = 0;
  op.neuron_stream_os_tid = 2222;
  op.pytorch_os_tid = 5000;
  buf->record(std::move(op));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  // First call returns populated buffer
  auto traceBuffer1 = session.getTraceBuffer();
  ASSERT_NE(traceBuffer1, nullptr);
  EXPECT_EQ(traceBuffer1->activities.size(), 2u);

  // Second call returns empty buffer (ownership was moved)
  auto traceBuffer2 = session.getTraceBuffer();
  ASSERT_NE(traceBuffer2, nullptr);
  EXPECT_TRUE(traceBuffer2->activities.empty());
}

TEST_F(SessionTest, ProcessTraceWithEmptyBuffer) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  // processTrace with no recorded operations
  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  EXPECT_EQ(logger.activityCount(), 0);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  EXPECT_TRUE(traceBuffer->activities.empty());

  // getDeviceInfo should still return defaults (no ops to set primaryDevice_)
  auto deviceInfo = session.getDeviceInfo();
  EXPECT_NE(deviceInfo, nullptr);
  EXPECT_EQ(deviceInfo->id, 0);

  // getResourceInfos should be empty
  EXPECT_TRUE(session.getResourceInfos().empty());
}

TEST_F(SessionTest, ProcessTraceVerifiesActivityMetadata) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "meta_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.nrt_sequence_id = 777;
  op.kineto_correlation_id = 1;
  op.pytorch_thread_id = 42;
  op.device_index = 0;
  op.stream_id = 2;
  op.neuron_stream_os_tid = 5555;
  op.pytorch_os_tid = 6666;
  buf->record(std::move(op));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);

  // RUNTIME activity metadata
  auto& runtime_act = *traceBuffer->activities[0];
  EXPECT_EQ(runtime_act.type(), libkineto::ActivityType::PRIVATEUSE1_RUNTIME);
  EXPECT_EQ(runtime_act.getMetadataValue("nrt_sequence_id"), "777");
  EXPECT_EQ(runtime_act.getMetadataValue("stream_id"), "2");
  EXPECT_EQ(runtime_act.getMetadataValue("pytorch_logical_tid"), "42");
  EXPECT_EQ(runtime_act.getMetadataValue("neuron_worker_tid"), "5555");

  // DRIVER activity metadata — only nrt_sequence_id is emitted
  auto& driver_act = *traceBuffer->activities[1];
  EXPECT_EQ(driver_act.type(), libkineto::ActivityType::PRIVATEUSE1_DRIVER);
  EXPECT_EQ(driver_act.getMetadataValue("nrt_sequence_id"), "777");
  EXPECT_EQ(driver_act.getMetadataValue("stream_id"), "");
  EXPECT_EQ(driver_act.getMetadataValue("pytorch_logical_tid"), "");
}

TEST_F(SessionTest, ProcessTraceEmitsCacheAndHierarchyMetadata) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "meta_cache_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.kineto_correlation_id = 1;
  op.device_index = 0;
  op.stream_id = 0;
  op.neuron_stream_os_tid = 5555;
  op.pytorch_os_tid = 6666;
  op.hierarchy_string = "Linear[fc1][1]/";
  op.neff_cache_key = "linear--_[8,4096]:BFloat16:abc123";
  op.persistent_cache_key = "00813ff53bb82d57870f6335f192ae50";
  buf->record(std::move(op));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);

  // RUNTIME (dispatch) activity metadata
  auto& runtime_act = *traceBuffer->activities[0];
  EXPECT_EQ(runtime_act.getMetadataValue("hierarchy"), "Linear[fc1][1]/");
  EXPECT_EQ(runtime_act.getMetadataValue("neff_cache_key"), "linear--_[8,4096]:BFloat16:abc123");
  EXPECT_EQ(runtime_act.getMetadataValue("persistent_cache_key"),
            "00813ff53bb82d57870f6335f192ae50");

  // DRIVER (execution) activity metadata
  auto& driver_act = *traceBuffer->activities[1];
  EXPECT_EQ(driver_act.getMetadataValue("hierarchy"), "Linear[fc1][1]/");
  EXPECT_EQ(driver_act.getMetadataValue("neff_cache_key"), "linear--_[8,4096]:BFloat16:abc123");
  EXPECT_EQ(driver_act.getMetadataValue("persistent_cache_key"),
            "00813ff53bb82d57870f6335f192ae50");
}

TEST_F(SessionTest, ProcessTraceOmitsEmptyCacheAndHierarchyMetadata) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "no_cache_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.kineto_correlation_id = 1;
  op.device_index = 0;
  op.stream_id = 0;
  op.neuron_stream_os_tid = 5555;
  op.pytorch_os_tid = 6666;
  // hierarchy_string, neff_cache_key, persistent_cache_key left empty
  buf->record(std::move(op));

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 2u);

  // Empty fields should not be emitted as metadata
  auto& runtime_act = *traceBuffer->activities[0];
  EXPECT_EQ(runtime_act.getMetadataValue("hierarchy"), "");
  EXPECT_EQ(runtime_act.getMetadataValue("neff_cache_key"), "");
  EXPECT_EQ(runtime_act.getMetadataValue("persistent_cache_key"), "");

  auto& driver_act = *traceBuffer->activities[1];
  EXPECT_EQ(driver_act.getMetadataValue("hierarchy"), "");
  EXPECT_EQ(driver_act.getMetadataValue("neff_cache_key"), "");
  EXPECT_EQ(driver_act.getMetadataValue("persistent_cache_key"), "");
}

TEST_F(SessionTest, ProcessTraceMultiDeviceMultiStreamCorrelation) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Op 0: device 0, stream 0, corr 10, worker 1001, pytorch 5000
  // Op 1: device 1, stream 1, corr 20, worker 2002, pytorch 5000
  // Op 2: device 0, stream 2, corr 30, worker 1001, pytorch 5000 (same worker as op 0)
  const int32_t devices[] = {0, 1, 0};
  const int32_t streams[] = {0, 1, 2};
  const uint64_t corr_ids[] = {10, 20, 30};
  const int32_t worker_tids[] = {1001, 2002, 1001};
  const int32_t pytorch_tid = 5000;

  for (int i = 0; i < 3; ++i) {
    ProfiledOperation op;
    op.op_name = "op_" + std::to_string(i);
    op.submit_start_ns = i * 10000;
    op.submit_time_ns = i * 10000 + 2000;
    op.execute_start_ns = i * 10000 + 5000;
    op.execute_end_ns = i * 10000 + 9000;
    op.dispatch_start_ns = i * 10000 + 3000;
    op.kineto_correlation_id = corr_ids[i];
    op.pytorch_thread_id = 100;
    op.device_index = devices[i];
    op.stream_id = streams[i];
    op.xu_type = 1;  // NRTA_XU_COMPUTE
    op.dispatch_slot = i;
    op.neuron_stream_os_tid = worker_tids[i];
    op.pytorch_os_tid = pytorch_tid;
    buf->record(std::move(op));
  }

  session.stop();

  NoOpLogger logger;
  auto noLinkedActivity = [](int32_t) -> const libkineto::ITraceActivity* { return nullptr; };
  session.processTrace(logger, noLinkedActivity, 0, INT64_MAX);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  // 3 activities per op: submit + dispatch + driver
  ASSERT_EQ(traceBuffer->activities.size(), 9u);

  for (int i = 0; i < 3; ++i) {
    auto& submit = *traceBuffer->activities[3 * i];
    auto& dispatch = *traceBuffer->activities[3 * i + 1];
    auto& driver = *traceBuffer->activities[3 * i + 2];

    // Correlation pairing
    EXPECT_EQ(submit.id, static_cast<int32_t>(corr_ids[i]));
    EXPECT_EQ(driver.id, static_cast<int32_t>(corr_ids[i]));

    // Flow arrows: DISPATCH -> DRIVER
    EXPECT_GT(dispatch.flow.id, 0u);
    EXPECT_EQ(driver.flow.id, dispatch.flow.id);
    EXPECT_EQ(dispatch.flow.start, 1u);
    EXPECT_EQ(driver.flow.start, 0u);

    // Device placement
    EXPECT_EQ(submit.device, libkineto::processId());
    EXPECT_EQ(driver.device, devices[i]);

    // Submit on PyTorch thread
    EXPECT_EQ(submit.resource, pytorch_tid);
    EXPECT_EQ(submit.threadId, pytorch_tid);
  }

  // Verify all flow IDs are unique across operations
  std::set<uint32_t> flowIds;
  for (int i = 0; i < 3; ++i) {
    flowIds.insert(traceBuffer->activities[3 * i + 1]->flow.id);
  }
  EXPECT_EQ(flowIds.size(), 3u);

  // Resource deduplication: 1 PyTorch thread + 3 dispatch slots + 3 XU rows = 7
  auto resources = session.getResourceInfos();
  EXPECT_GE(resources.size(), 4u);

  // primaryDevice_ set from first op's device_index
  auto deviceInfo = session.getDeviceInfo();
  EXPECT_NE(deviceInfo, nullptr);
  EXPECT_EQ(deviceInfo->id, 0);
}

TEST_F(SessionTest, ProcessTraceZeroCorrelationIdSkipsLinking) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  ProfiledOperation op;
  op.op_name = "zero_corr_op";
  op.submit_start_ns = 500;
  op.submit_time_ns = 1000;
  op.execute_start_ns = 2000;
  op.execute_end_ns = 3000;
  op.dispatch_start_ns = 1500;
  op.kineto_correlation_id = 0;
  op.pytorch_thread_id = 50;
  op.device_index = 0;
  op.stream_id = 0;
  op.xu_type = 1;
  op.dispatch_slot = 0;
  op.neuron_stream_os_tid = 4444;
  op.pytorch_os_tid = 5000;
  buf->record(std::move(op));

  session.stop();

  // Provide a callback that tracks invocations — it should never be called
  bool callbackInvoked = false;
  libkineto::GenericTraceActivity fakeCpuAct;
  auto trackingCallback = [&](int32_t) -> const libkineto::ITraceActivity* {
    callbackInvoked = true;
    return &fakeCpuAct;
  };

  NoOpLogger logger;
  session.processTrace(logger, trackingCallback, 0, INT64_MAX);

  // Callback must not have been invoked (correlation_id == 0 skips linking)
  EXPECT_FALSE(callbackInvoked);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 3u);  // submit + dispatch + driver

  auto& submit = *traceBuffer->activities[0];
  auto& dispatch = *traceBuffer->activities[1];
  auto& driver = *traceBuffer->activities[2];

  // All activities should have no linked activity (correlation_id == 0)
  EXPECT_EQ(submit.linkedActivity(), nullptr);
  EXPECT_EQ(dispatch.linkedActivity(), nullptr);
  EXPECT_EQ(driver.linkedActivity(), nullptr);

  // Flow arrows still connect DISPATCH -> DRIVER even with zero correlation
  EXPECT_GT(dispatch.flow.id, 0u);
  EXPECT_EQ(driver.flow.id, dispatch.flow.id);
}

// Verify that multiple ops sharing the same kineto_correlation_id get unique flow IDs.
// This is the exact scenario that caused the TORCH_INTERNAL_ASSERT failure in
// collection.cpp:1152 before the monotonic counter fix: aten::mm decomposes into
// multiple NRT dispatches within the same RecordFunction scope, all sharing one
// correlation ID.
TEST_F(SessionTest, ProcessTraceSharedCorrelationIdGetsUniqueFlowIds) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  ProfilerEventBuffer* buf = getActiveProfilerBuffer();
  ASSERT_NE(buf, nullptr);

  // Three NRT ops all sharing kineto_correlation_id = 42 (same RecordFunction scope)
  const uint64_t sharedCorrId = 42;
  for (int i = 0; i < 3; ++i) {
    ProfiledOperation op;
    op.op_name = "nrt_dispatch_" + std::to_string(i);
    op.submit_start_ns = i * 1000;
    op.submit_time_ns = i * 1000 + 200;
    op.execute_start_ns = i * 1000 + 400;
    op.execute_end_ns = i * 1000 + 800;
    op.dispatch_start_ns = i * 1000 + 300;
    op.kineto_correlation_id = sharedCorrId;
    op.pytorch_thread_id = 100;
    op.device_index = 0;
    op.stream_id = 0;
    op.xu_type = 1;
    op.dispatch_slot = i;
    op.neuron_stream_os_tid = 7777;
    op.pytorch_os_tid = 5000;
    buf->record(std::move(op));
  }

  session.stop();

  // Set up a fake CPU activity for the shared correlation ID
  libkineto::TraceSpan cpuSpan(0, INT64_MAX, "cpu");
  libkineto::GenericTraceActivity cpuAct(cpuSpan, libkineto::ActivityType::CPU_OP, "aten::mm");
  cpuAct.id = static_cast<int32_t>(sharedCorrId);

  NoOpLogger logger;
  auto linkedActivityCb = [&cpuAct](int32_t correlationId) -> const libkineto::ITraceActivity* {
    if (correlationId == 42) {
      return &cpuAct;
    }
    return nullptr;
  };
  session.processTrace(logger, linkedActivityCb, 0, INT64_MAX);

  // 3 ops * 3 activities each = 9
  EXPECT_EQ(logger.activityCount(), 9);

  auto traceBuffer = session.getTraceBuffer();
  ASSERT_NE(traceBuffer, nullptr);
  ASSERT_EQ(traceBuffer->activities.size(), 9u);

  // Collect all DISPATCH flow IDs and verify uniqueness
  std::set<uint32_t> dispatchFlowIds;
  for (int i = 0; i < 3; ++i) {
    auto& submit = *traceBuffer->activities[3 * i];
    auto& dispatch = *traceBuffer->activities[3 * i + 1];
    auto& driver = *traceBuffer->activities[3 * i + 2];

    // All activities should share the same correlation ID
    EXPECT_EQ(submit.id, static_cast<int32_t>(sharedCorrId));
    EXPECT_EQ(driver.id, static_cast<int32_t>(sharedCorrId));

    // Each DISPATCH->DRIVER pair has matching flow IDs
    EXPECT_GT(dispatch.flow.id, 0u);
    EXPECT_EQ(driver.flow.id, dispatch.flow.id);
    EXPECT_EQ(dispatch.flow.start, 1u);
    EXPECT_EQ(driver.flow.start, 0u);

    dispatchFlowIds.insert(dispatch.flow.id);

    // All three ops should link to the same CPU activity
    EXPECT_NE(submit.linkedActivity(), nullptr);
    EXPECT_EQ(submit.linkedActivity()->correlationId(), static_cast<int32_t>(sharedCorrId));
  }

  // Critical assertion: all 3 flow IDs must be unique despite shared correlation ID.
  EXPECT_EQ(dispatchFlowIds.size(), 3u);
}

TEST_F(SessionTest, NeffCacheDirSetViaCustomConfig) {
  // When neff_cache_dir is in the custom config string, initNrtConfig should call
  // nrt_inspect_config_set_neff_cache_dir with the configured path.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "neuron_profile:device;neff_cache_dir:/tmp/neff_cache";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_neff_cache_dir(_, StrEq("/tmp/neff_cache")))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, NeffCacheDirNotSetWhenOmittedFromConfig) {
  // When neff_cache_dir is not in the custom config, nrt_inspect_config_set_neff_cache_dir
  // should never be called.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_neff_cache_dir(_, _)).Times(0);

  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

}  // namespace
}  // namespace at::neuron
