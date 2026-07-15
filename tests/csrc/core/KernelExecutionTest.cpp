#include <gtest/gtest.h>

#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class KernelExecutionTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  void SetUp() override {
    // Mock NRT instance info for CompilationCacheKey construction
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    // Set up world size for collective kernel tests
    c10_neuron::set_world_size(2);
    c10_neuron::set_rank(0);

    // Create test tensors
    input_tensor_ = create_test_tensor({2, 3});
    output_tensor_ = create_test_tensor({2, 3});
    inputs_ = {input_tensor_};
    outputs_ = {output_tensor_};
    input_ptrs_ = get_tensor_data_ptrs(inputs_);
    output_ptrs_ = get_tensor_data_ptrs(outputs_);
  }

  void TearDown() override {
    // Reset distributed state so next test can set it again
    c10_neuron::reset_distributed_state();
  }

  torch::Tensor input_tensor_;
  torch::Tensor output_tensor_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
  std::vector<void*> input_ptrs_;
  std::vector<void*> output_ptrs_;
};

TEST_F(KernelExecutionTest, CompilableKernelConstruction) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("compilable_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {},
                                      "cache_key_123", hlo_bytes, false, 0);

  EXPECT_EQ(kernel.GetOpName(), "compilable_op");
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_TRUE(kernel.GetCompileKey().find("cache_key_123") == 0);
  EXPECT_FALSE(kernel.HasCachedNeff());
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompilableKernelWithCollectives) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("collective_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "cache_key",
                                      hlo_bytes, true, 0);

  EXPECT_TRUE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompilableKernelCompilation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("compilable_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "cache_key",
                                      hlo_bytes, false, 0);

  // Test compilation - will throw since we don't have real compiler
  EXPECT_THROW(kernel.CompileToNeff(), std::exception);
}

TEST_F(KernelExecutionTest, CompilableKernelValidation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Test with empty cache key (should be invalid)
  XLACompilableKernelExecution kernel1("op", create_fake_tensor_refs(input_ptrs_),
                                       create_fake_tensor_refs(output_ptrs_), {}, {}, "", hlo_bytes,
                                       false, 0);
  EXPECT_FALSE(kernel1.IsValid());

  // Test with empty op name (should be invalid)
  XLACompilableKernelExecution kernel2("", create_fake_tensor_refs(input_ptrs_),
                                       create_fake_tensor_refs(output_ptrs_), {}, {}, "cache_key",
                                       hlo_bytes, false, 0);
  EXPECT_FALSE(kernel2.IsValid());
}

// DirectKernelExecution is abstract - tested through CollectiveDirectKernelExecution below

TEST_F(KernelExecutionTest, XLAKernelConstruction) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};  // "HLO"

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {},
                                      "xla_cache_key", hlo_bytes, false, 0);

  EXPECT_EQ(kernel.GetOpName(), "xla_op");
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_TRUE(kernel.GetCompileKey().find("xla_cache_key") == 0);
  EXPECT_EQ(kernel.GetHloBytes(), hlo_bytes);
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, XLAKernelWithEmptyHLO) {
  std::vector<uint8_t> empty_hlo;

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "cache_key",
                                      empty_hlo, false, 0);

  EXPECT_FALSE(kernel.IsValid());  // Should be invalid with empty HLO
}

TEST_F(KernelExecutionTest, XLAKernelCompilation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F, 0x01, 0x02};

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "cache_key",
                                      hlo_bytes, false, 0);

  // TODO(rpsilva): Integrate or mock the compiler.
  EXPECT_THROW(kernel.CompileToNeff(), std::exception);
}

TEST_F(KernelExecutionTest, CollectiveKernelConstruction) {
  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  CollectiveDirectKernelExecution kernel(
      "allreduce_op", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::Float, c10d::ReduceOp::SUM, replica_groups, 0);

  EXPECT_EQ(kernel.GetOpName(), "allreduce_op");
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kCollective);
  EXPECT_EQ(kernel.GetCollectiveType(),
            CollectiveDirectKernelExecution::CollectiveType::kAllReduce);
  EXPECT_EQ(kernel.GetReduceOp(), c10d::ReduceOp::SUM);
  EXPECT_EQ(kernel.GetDtype(), c10::ScalarType::Float);
  EXPECT_EQ(kernel.GetReplicaGroups(), replica_groups);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CollectiveKernelAllTypes) {
  // Test supported collective types (kAllToAll, kBroadcast, kReduce not yet implemented)
  std::vector<CollectiveDirectKernelExecution::CollectiveType> types = {
      CollectiveDirectKernelExecution::CollectiveType::kAllGather,
      CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      CollectiveDirectKernelExecution::CollectiveType::kReduceScatter};

  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  for (auto type : types) {
    CollectiveDirectKernelExecution kernel("collective", type, create_fake_tensor_refs(input_ptrs_),
                                           create_fake_tensor_refs(output_ptrs_), {}, {},
                                           c10::ScalarType::Float, c10d::ReduceOp::SUM,
                                           replica_groups, 0);
    EXPECT_EQ(kernel.GetCollectiveType(), type);
    EXPECT_TRUE(kernel.IsValid());
  }
}

TEST_F(KernelExecutionTest, CollectiveKernelReduceOps) {
  // Test different reduce operations
  std::vector<c10d::ReduceOp> reduce_ops = {c10d::ReduceOp::SUM, c10d::ReduceOp::MIN,
                                            c10d::ReduceOp::MAX};

  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  for (auto reduce_op : reduce_ops) {
    CollectiveDirectKernelExecution kernel(
        "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
        create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
        c10::ScalarType::Float, reduce_op, replica_groups, 0);
    EXPECT_EQ(kernel.GetReduceOp(), reduce_op);
    EXPECT_TRUE(kernel.IsValid());
  }
}

TEST_F(KernelExecutionTest, CollectiveKernelUsesGlobalRankForLookup) {
  // Simulate multi-node: global rank 1, but local device_id 0
  c10_neuron::reset_distributed_state();
  c10_neuron::set_world_size(2);
  c10_neuron::set_rank(1);

  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  CollectiveDirectKernelExecution kernel(
      "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::Float, c10d::ReduceOp::SUM, replica_groups, /*device_id=*/0);

  // Should find rank 1 at index 1 in the replica group, not index 0
  EXPECT_EQ(kernel.GetMyRank(), 1);
  const auto& comm = kernel.GetComm();
  EXPECT_EQ(comm.ctx_device_id, 1u);  // global rank, not local device_id
  EXPECT_EQ(comm.rank, 1u);
}

TEST_F(KernelExecutionTest, CollectiveKernelMultiNodeRank0) {
  // Simulate multi-node: global rank 0, local device_id 0
  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  CollectiveDirectKernelExecution kernel(
      "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::Float, c10d::ReduceOp::SUM, replica_groups, /*device_id=*/0);

  EXPECT_EQ(kernel.GetMyRank(), 0);
  const auto& comm = kernel.GetComm();
  EXPECT_EQ(comm.ctx_device_id, 0u);
  EXPECT_EQ(comm.rank, 0u);
  EXPECT_EQ(comm.rank_n, 2u);
  EXPECT_EQ(comm.ctx_device_count, 2u);
}

TEST_F(KernelExecutionTest, CollectiveKernelMultiGroupLookup) {
  // Multiple replica groups — device should be found in the correct one
  c10_neuron::reset_distributed_state();
  c10_neuron::set_world_size(4);
  c10_neuron::set_rank(3);

  std::vector<std::vector<int>> replica_groups = {{0, 1}, {2, 3}};
  CollectiveDirectKernelExecution kernel(
      "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::Float, c10d::ReduceOp::SUM, replica_groups, /*device_id=*/0);

  // Global rank 3 is at index 1 in group {2, 3}
  EXPECT_EQ(kernel.GetMyRank(), 1);
  const auto& comm = kernel.GetComm();
  EXPECT_EQ(comm.ctx_device_id, 3u);
  EXPECT_EQ(comm.rank, 1u);
  EXPECT_EQ(comm.rank_n, 2u);
}

TEST_F(KernelExecutionTest, PolymorphicBehavior) {
  // Create different kernel types and verify polymorphic behavior through base pointer
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  std::vector<std::vector<int>> replica_groups = {{0, 1}};

  std::vector<std::unique_ptr<NeuronKernelExecution>> kernels;
  kernels.push_back(std::make_unique<XLACompilableKernelExecution>(
      "xla", create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_),
      std::vector<TensorContext>{}, std::vector<TensorContext>{}, "key", hlo_bytes, false, 0));
  kernels.push_back(std::make_unique<CollectiveDirectKernelExecution>(
      "collective", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_),
      std::vector<TensorContext>{}, std::vector<TensorContext>{}, c10::ScalarType::Float,
      c10d::ReduceOp::SUM, replica_groups, 0));

  // Verify polymorphic behavior
  EXPECT_TRUE(kernels[0]->RequiresCompilation());   // XLA kernel
  EXPECT_FALSE(kernels[1]->RequiresCompilation());  // Collective kernel

  for (const auto& kernel : kernels) {
    EXPECT_TRUE(kernel->IsValid());
    EXPECT_FALSE(kernel->GetOpName().empty());
  }
}

TEST_F(KernelExecutionTest, CacheKeyIsBaseKey) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "base_key",
                                      hlo_bytes, false, 0);

  // In-memory cache key is just the base key (compiler config is only in persistent key)
  EXPECT_EQ(kernel.GetCompileKey(), "base_key");
}

TEST_F(KernelExecutionTest, CompilableKernelAccessors) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // Test that we can access compilation configuration
  EXPECT_FALSE(kernel.GetAdditionalArgs().empty());
  EXPECT_FALSE(kernel.GetOptimizationLevel().empty());

  // XLA should have default args
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--model-type") != std::string::npos);
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--auto-cast=none") != std::string::npos);

  // Default optimization level should be -O1
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O1");
}

TEST_F(KernelExecutionTest, XLAEnvironmentVariableSupport) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  // Set environment variables
  setenv("NEURON_CC_FLAGS", "--custom-flag=value", 1);
  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);

  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // Check that environment variables are reflected
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--custom-flag=value") != std::string::npos);
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O2");

  // Cache key is just the base key (compiler config is in persistent key only)
  EXPECT_EQ(kernel.GetCompileKey(), "key");

  // Clean up
  unsetenv("NEURON_CC_FLAGS");
  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, SameBaseKeyDifferentArgsEqualInMemoryKey) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  setenv("NEURON_COMPILER_OPT_LEVEL", "-O1", 1);
  XLACompilableKernelExecution kernel1("op", create_fake_tensor_refs(input_ptrs_),
                                       create_fake_tensor_refs(output_ptrs_), {}, {}, "same_base",
                                       hlo_bytes, false, 0);

  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);
  XLACompilableKernelExecution kernel2("op", create_fake_tensor_refs(input_ptrs_),
                                       create_fake_tensor_refs(output_ptrs_), {}, {}, "same_base",
                                       hlo_bytes, false, 0);

  // In-memory keys are equal (both just "same_base")
  EXPECT_EQ(kernel1.GetCompileKey(), kernel2.GetCompileKey());
  EXPECT_EQ(kernel1.GetCompileKey(), "same_base");

  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, CachedNeffManagement) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // Initially no cached NEFF
  EXPECT_FALSE(kernel.HasCachedNeff());

  // Set a cached NEFF - create a NeffBytesPtr (unique_ptr with CacheEntryGuard deleter)
  auto neff_data = new std::vector<uint8_t>{0x4E, 0x45, 0x46, 0x46};  // "NEFF"
  NeffBytesPtr neff_ptr(neff_data, CacheEntryGuard{});

  // Store expected data before moving the pointer
  std::vector<uint8_t> expected_data = *neff_data;

  kernel.SetCachedNeff(std::move(neff_ptr));

  // Now should have cached NEFF
  EXPECT_TRUE(kernel.HasCachedNeff());
  EXPECT_EQ(kernel.GetCachedNeff(), expected_data);
}

TEST_F(KernelExecutionTest, IoNamesDefaultEmpty) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // By default a compilable kernel carries no I/O names — NRTHandler will
  // fall back to ``input{i}`` / ``output{i}`` when scheduling.
  EXPECT_TRUE(kernel.GetInputNames().empty());
  EXPECT_TRUE(kernel.GetOutputNames().empty());
}

TEST_F(KernelExecutionTest, IoNamesSetAndGet) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  std::vector<std::string> in_names = {"q", "k", "v"};
  std::vector<std::string> out_names = {"attn_out"};
  kernel.SetInputNames(in_names);
  kernel.SetOutputNames(out_names);

  EXPECT_EQ(kernel.GetInputNames(), in_names);
  EXPECT_EQ(kernel.GetOutputNames(), out_names);
}

TEST_F(KernelExecutionTest, IoNamesAcceptAliasSlots) {
  // NKI kernel-builder produces aliased input ports named
  // ``<output>.must_alias_input`` — verify those round-trip without any
  // special handling in the base class.
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  kernel.SetInputNames({"a", "b", "out.must_alias_input"});
  kernel.SetOutputNames({"out"});

  const auto& ins = kernel.GetInputNames();
  ASSERT_EQ(ins.size(), 3u);
  EXPECT_EQ(ins[2], "out.must_alias_input");
}

TEST_F(KernelExecutionTest, XLACompilationConfigPropagation) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  setenv("NEURON_COMPILER_OPT_LEVEL", "-O2", 1);
  setenv("NEURON_CC_FLAGS", "--test-flag", 1);

  XLACompilableKernelExecution kernel("op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // Verify that compilation config would include these settings
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O2");
  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--test-flag") != std::string::npos);

  unsetenv("NEURON_COMPILER_OPT_LEVEL");
  unsetenv("NEURON_CC_FLAGS");
}

TEST_F(KernelExecutionTest, EventSignalKernelConstruction) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_signal", event,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  EXPECT_EQ(kernel.GetOpName(), "event_signal");
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kEvent);
  EXPECT_EQ(kernel.GetEventAction(), EventDirectKernelExecution::EventAction::kSignal);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, EventWaitKernelConstruction) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_wait", event,
                                    EventDirectKernelExecution::EventAction::kWait, 0);

  EXPECT_EQ(kernel.GetOpName(), "event_wait");
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kEvent);
  EXPECT_EQ(kernel.GetEventAction(), EventDirectKernelExecution::EventAction::kWait);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, EventSignalKernelIsAlwaysReady) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_signal", event,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  // Signal kernels are always ready to execute
  EXPECT_TRUE(kernel.IsReady());
}

TEST_F(KernelExecutionTest, EventWaitKernelReadinessFollowsEvent) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_wait", event,
                                    EventDirectKernelExecution::EventAction::kWait, 0);

  // Wait kernel is ready when event is not recorded (query returns true)
  EXPECT_TRUE(kernel.IsReady());

  // Simulate event being recorded but not completed
  event.get_impl()->is_recorded.store(true, std::memory_order_release);
  event.get_impl()->is_completed.store(false, std::memory_order_release);

  // Now wait kernel should not be ready
  EXPECT_FALSE(kernel.IsReady());

  // Complete the event
  event.complete_event();

  // Now wait kernel should be ready
  EXPECT_TRUE(kernel.IsReady());
}

TEST_F(KernelExecutionTest, EventSignalKernelExecuteCompletesEvent) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_signal", event,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  // Event should not be completed initially
  EXPECT_FALSE(event.get_impl()->is_completed.load());

  // Execute the signal kernel
  kernel.Schedule(nullptr, nullptr, 0);

  // Event should now be completed
  EXPECT_TRUE(event.get_impl()->is_completed.load());
  EXPECT_TRUE(event.query());
}

TEST_F(KernelExecutionTest, EventWaitKernelExecuteWhenReady) {
  NeuronEvent event;
  event.complete_event();  // Pre-complete the event

  EventDirectKernelExecution kernel("event_wait", event,
                                    EventDirectKernelExecution::EventAction::kWait, 0);

  EXPECT_TRUE(kernel.IsReady());

  // Execute should not throw when event is ready
  EXPECT_NO_THROW(kernel.Schedule(nullptr, nullptr, 0));
}

TEST_F(KernelExecutionTest, EventKernelGetEventReturnsCorrectEvent) {
  NeuronEvent event;

  EventDirectKernelExecution kernel("event_op", event,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  // The kernel should hold a copy of the event that shares the same impl
  const NeuronEvent& kernel_event = kernel.GetEvent();
  EXPECT_EQ(kernel_event.get_impl(), event.get_impl());
}

TEST_F(KernelExecutionTest, EventKernelSharedEventState) {
  NeuronEvent original_event;
  NeuronEvent event_copy = original_event;

  EventDirectKernelExecution kernel("event_signal", event_copy,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  // Execute the kernel (signals the event)
  kernel.Schedule(nullptr, nullptr, 0);

  // Both the original event and the copy should see the completion
  EXPECT_TRUE(original_event.query());
  EXPECT_TRUE(event_copy.query());
  EXPECT_TRUE(kernel.GetEvent().query());
}

TEST_F(KernelExecutionTest, EventKernelDifferentDevices) {
  NeuronEvent event;

  // Create event kernels on different devices
  EventDirectKernelExecution kernel_dev0("event_signal_0", event,
                                         EventDirectKernelExecution::EventAction::kSignal, 0);
  EventDirectKernelExecution kernel_dev1("event_signal_1", event,
                                         EventDirectKernelExecution::EventAction::kSignal, 1);

  EXPECT_TRUE(kernel_dev0.IsValid());
  EXPECT_TRUE(kernel_dev1.IsValid());
}

TEST_F(KernelExecutionTest, CompileOnlyKernelConstruction) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  CompileOnlyKernelExecution kernel("base_cache_key", stablehlo_bytes, false);

  EXPECT_TRUE(kernel.IsValid());
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHLO);
  EXPECT_FALSE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.GetCompileKey().find("base_cache_key") == 0);
}

TEST_F(KernelExecutionTest, CompileOnlyKernelWithCollectives) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  CompileOnlyKernelExecution kernel("cache_key", stablehlo_bytes, true);

  EXPECT_TRUE(kernel.HasCollectives());
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompileOnlyKernelValidation) {
  // Empty StableHLO bytes should be invalid
  std::vector<uint8_t> empty_bytes;
  CompileOnlyKernelExecution kernel("cache_key", empty_bytes, false);

  EXPECT_FALSE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CompileOnlyKernelCacheKey) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  CompileOnlyKernelExecution kernel("base_key", stablehlo_bytes, false);

  // In-memory cache key is just the base key
  EXPECT_EQ(kernel.GetCompileKey(), "base_key");
}

TEST_F(KernelExecutionTest, CompileOnlyKernelEnvironmentVariables) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};

  setenv("NEURON_CC_FLAGS", "--compile-only-flag", 1);
  setenv("NEURON_COMPILER_OPT_LEVEL", "-O3", 1);

  CompileOnlyKernelExecution kernel("key", stablehlo_bytes, false);

  EXPECT_TRUE(kernel.GetAdditionalArgs().find("--compile-only-flag") != std::string::npos);
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O3");

  unsetenv("NEURON_CC_FLAGS");
  unsetenv("NEURON_COMPILER_OPT_LEVEL");
}

TEST_F(KernelExecutionTest, CompileOnlyKernelDefaultOptLevel) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  unsetenv("NEURON_COMPILER_OPT_LEVEL");

  CompileOnlyKernelExecution kernel("key", stablehlo_bytes, false);

  // Default optimization level for torch.compile should be -O2
  EXPECT_EQ(kernel.GetOptimizationLevel(), "-O2");
}

TEST_F(KernelExecutionTest, CompileOnlyKernelCompilation) {
  std::vector<uint8_t> stablehlo_bytes = {0x53, 0x48, 0x4C, 0x4F};
  CompileOnlyKernelExecution kernel("key", stablehlo_bytes, false);

  // CompileToNeff should throw since we don't have real compiler
  EXPECT_THROW(kernel.CompileToNeff(), std::exception);
}

// NeffDirectKernelExecution tests

TEST_F(KernelExecutionTest, NeffDirectKernelConstruction) {
  NeffDirectKernelExecution kernel("graph_name", "cache_key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);

  EXPECT_EQ(kernel.GetOpName(), "graph_name");
  EXPECT_EQ(kernel.GetCompileKey(), "cache_key");
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHLO);
  EXPECT_FALSE(kernel.HasCollectives());
}

TEST_F(KernelExecutionTest, NeffDirectKernelWithCollectives) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, true);

  EXPECT_TRUE(kernel.HasCollectives());
}

TEST_F(KernelExecutionTest, NeffDirectKernelValidation) {
  // Valid kernel with inputs and outputs
  NeffDirectKernelExecution valid_kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                         create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);
  EXPECT_TRUE(valid_kernel.IsValid());
}

TEST_F(KernelExecutionTest, NeffDirectKernelCompileToNeffThrows) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);

  // CompileToNeff should throw logic_error since NEFF is pre-compiled
  EXPECT_THROW(kernel.CompileToNeff(), std::logic_error);
}

TEST_F(KernelExecutionTest, NeffDirectKernelRequiresCompilation) {
  NeffDirectKernelExecution kernel("graph", "key", create_fake_tensor_refs(input_ptrs_),
                                   create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);

  // RequiresCompilation returns true to force cache lookup
  EXPECT_TRUE(kernel.RequiresCompilation());
}

TEST_F(KernelExecutionTest, WriteKernelBoolConstruction) {
  bool src_value = true;
  void* dst_ptr = output_ptrs_[0];

  WriteDirectKernelExecution kernel("write_bool", &src_value, TensorDataRef{dst_ptr}, 0,
                                    sizeof(bool), 0);

  EXPECT_EQ(kernel.GetOpName(), "write_bool");
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, WriteKernelAllIntTypes) {
  void* dst_ptr = output_ptrs_[0];
  int64_t src_value = 123;

  WriteDirectKernelExecution kernel("write_int", &src_value, TensorDataRef{dst_ptr}, 0,
                                    sizeof(int64_t), 0);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, WriteKernelAllFloatTypes) {
  void* dst_ptr = output_ptrs_[0];
  double src_value = 1.5;

  WriteDirectKernelExecution kernel("write_float", &src_value, TensorDataRef{dst_ptr}, 0,
                                    sizeof(double), 0);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, WriteKernelBufferConstruction) {
  void* src_ptr = input_ptrs_[0];
  void* dst_ptr = output_ptrs_[0];
  size_t size_bytes = 24;  // 2*3 floats * 4 bytes

  WriteDirectKernelExecution kernel("write_buffer", src_ptr, TensorDataRef{dst_ptr}, 0, size_bytes,
                                    0);

  EXPECT_EQ(kernel.GetOpName(), "write_buffer");
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kWrite);
  EXPECT_TRUE(kernel.IsValid());
}

TEST_F(KernelExecutionTest, CollectiveKernelRequiresPrepare) {
  std::vector<std::vector<int>> replica_groups = {{0, 1}};
  CollectiveDirectKernelExecution kernel(
      "allreduce_op", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::Float, c10d::ReduceOp::SUM, replica_groups, 0);

  // Collectives require preparation before scheduling
  EXPECT_TRUE(kernel.RequiresPrepare());
}

TEST_F(KernelExecutionTest, NonCollectiveKernelsDoNotRequirePrepare) {
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};

  XLACompilableKernelExecution xla_kernel("xla", create_fake_tensor_refs(input_ptrs_),
                                          create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                          hlo_bytes, false, 0);
  EXPECT_FALSE(xla_kernel.RequiresPrepare());

  NeffDirectKernelExecution neff_kernel("neff", "key", create_fake_tensor_refs(input_ptrs_),
                                        create_fake_tensor_refs(output_ptrs_), {}, {}, 0, false);
  EXPECT_FALSE(neff_kernel.RequiresPrepare());
}

TEST_F(KernelExecutionTest, EventKernelDoesNotRequirePrepare) {
  NeuronEvent event;
  EventDirectKernelExecution kernel("event_signal", event,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);

  EXPECT_FALSE(kernel.RequiresPrepare());
}

// IsDeviceKernelType and GetXUIndex tests

TEST_F(KernelExecutionTest, IsDeviceKernelTypeReturnsTrue) {
  // Device kernel types should return true
  EXPECT_TRUE(IsDeviceKernelType(KernelTypeEnum::kHLO));
  EXPECT_TRUE(IsDeviceKernelType(KernelTypeEnum::kCollective));
  EXPECT_TRUE(IsDeviceKernelType(KernelTypeEnum::kCopy));
  EXPECT_TRUE(IsDeviceKernelType(KernelTypeEnum::kWrite));
  EXPECT_TRUE(IsDeviceKernelType(KernelTypeEnum::kRead));
}

TEST_F(KernelExecutionTest, IsDeviceKernelTypeReturnsFalse) {
  // Host-side kernel types should return false
  EXPECT_FALSE(IsDeviceKernelType(KernelTypeEnum::kEvent));
  EXPECT_FALSE(IsDeviceKernelType(KernelTypeEnum::kHint));
}

TEST_F(KernelExecutionTest, GetXUIndexMapping) {
  // All tensor ops map to NRTA_XU_TENSOR_OP
  EXPECT_EQ(GetXUIndex(KernelTypeEnum::kHLO), NRTA_XU_COMPUTE);
  EXPECT_EQ(GetXUIndex(KernelTypeEnum::kCollective), NRTA_XU_COLLECTIVES);
  EXPECT_EQ(GetXUIndex(KernelTypeEnum::kCopy), NRTA_XU_TENSOR_OP);
  EXPECT_EQ(GetXUIndex(KernelTypeEnum::kWrite), NRTA_XU_TENSOR_OP);
  EXPECT_EQ(GetXUIndex(KernelTypeEnum::kRead), NRTA_XU_TENSOR_OP);
}

TEST_F(KernelExecutionTest, GetXUIndexThrowsForNonDevice) {
  // Non-device kernel types should throw
  EXPECT_THROW(GetXUIndex(KernelTypeEnum::kEvent), torch_neuronx::ExecutionRuntimeException);
  EXPECT_THROW(GetXUIndex(KernelTypeEnum::kHint), torch_neuronx::ExecutionRuntimeException);
}

TEST_F(KernelExecutionTest, BaseKernelPrepareDoesNotThrow) {
  // Base NeuronKernelExecution::Prepare() should complete without throwing
  std::vector<uint8_t> hlo_bytes = {0x48, 0x4C, 0x4F};
  XLACompilableKernelExecution kernel("xla_op", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "key",
                                      hlo_bytes, false, 0);

  // Non-collective kernels should not throw from Prepare()
  EXPECT_NO_THROW(kernel.Prepare());
}

TEST_F(KernelExecutionTest, CollectiveKernelNewAccessors) {
  std::vector<std::vector<int>> replica_groups = {{0, 1, 2}, {3, 4, 5}};
  CollectiveDirectKernelExecution kernel(
      "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
      create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {}, {},
      c10::ScalarType::BFloat16, c10d::ReduceOp::MAX, replica_groups, 0);

  // Test new accessors added for host-cc
  EXPECT_EQ(kernel.GetDtype(), c10::ScalarType::BFloat16);
  EXPECT_EQ(kernel.GetReplicaGroups().size(), 2);
  EXPECT_EQ(kernel.GetReplicaGroups()[0].size(), 3);
  EXPECT_EQ(kernel.GetReplicaGroups()[1][2], 5);
}

TEST_F(KernelExecutionTest, CollectiveKernelValidationRequiresReplicaGroups) {
  // Empty replica_groups should throw in constructor
  std::vector<std::vector<int>> empty_replica_groups = {};
  EXPECT_THROW(CollectiveDirectKernelExecution(
                   "allreduce", CollectiveDirectKernelExecution::CollectiveType::kAllReduce,
                   create_fake_tensor_refs(input_ptrs_), create_fake_tensor_refs(output_ptrs_), {},
                   {}, c10::ScalarType::Float, c10d::ReduceOp::SUM, empty_replica_groups, 0),
               c10::Error);
}

// NRT type mapping tests

TEST_F(KernelExecutionTest, MapScalarTypeToNRTFloat) {
  using namespace at::neuron::nrt;

  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Float), NRTDType::kFloat32);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Half), NRTDType::kFloat16);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::BFloat16), NRTDType::kBFloat16);
}

TEST_F(KernelExecutionTest, MapScalarTypeToNRTInt) {
  using namespace at::neuron::nrt;

  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Int), NRTDType::kInt32);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Long), NRTDType::kInt64);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Short), NRTDType::kInt16);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Char), NRTDType::kInt8);
}

TEST_F(KernelExecutionTest, MapScalarTypeToNRTUnsigned) {
  using namespace at::neuron::nrt;

  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::Byte), NRTDType::kUInt8);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::UInt16), NRTDType::kUInt16);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::UInt32), NRTDType::kUInt32);
  EXPECT_EQ(MapScalarTypeToNRT(c10::ScalarType::UInt64), NRTDType::kUInt64);
}

TEST_F(KernelExecutionTest, MapScalarTypeToNRTUnsupportedThrows) {
  using namespace at::neuron::nrt;

  // Complex types are not supported
  EXPECT_THROW(MapScalarTypeToNRT(c10::ScalarType::ComplexFloat), c10::Error);
  EXPECT_THROW(MapScalarTypeToNRT(c10::ScalarType::ComplexDouble), c10::Error);
}

TEST_F(KernelExecutionTest, MapReduceOpToNRT) {
  using namespace at::neuron::nrt;

  EXPECT_EQ(MapReduceOpToNRT(c10d::ReduceOp::SUM), NRTReduceOp::kAdd);
  EXPECT_EQ(MapReduceOpToNRT(c10d::ReduceOp::MAX), NRTReduceOp::kMax);
  EXPECT_EQ(MapReduceOpToNRT(c10d::ReduceOp::MIN), NRTReduceOp::kMin);
}

TEST_F(KernelExecutionTest, MapReduceOpToNRTUnsupportedThrows) {
  using namespace at::neuron::nrt;

  // PRODUCT, BAND, BOR, BXOR are not supported
  EXPECT_THROW(MapReduceOpToNRT(c10d::ReduceOp::PRODUCT), c10::Error);
  EXPECT_THROW(MapReduceOpToNRT(c10d::ReduceOp::BAND), c10::Error);
}

TEST_F(KernelExecutionTest, NRTCollectiveOpEnumValues) {
  using namespace at::neuron::nrt;

  // Verify enum values match NRT constants
  EXPECT_EQ(static_cast<int>(NRTCollectiveOp::kAllGather), NRT_CC_ALLGATHER);
  EXPECT_EQ(static_cast<int>(NRTCollectiveOp::kReduceScatter), NRT_CC_REDUCESCATTER);
  EXPECT_EQ(static_cast<int>(NRTCollectiveOp::kAllReduce), NRT_CC_ALLREDUCE);
}

TEST_F(KernelExecutionTest, NRTReduceOpEnumValues) {
  using namespace at::neuron::nrt;

  // Verify enum values match NRT constants
  EXPECT_EQ(static_cast<int>(NRTReduceOp::kAdd), NRT_OP_ADD);
  EXPECT_EQ(static_cast<int>(NRTReduceOp::kMax), NRT_OP_MAX);
  EXPECT_EQ(static_cast<int>(NRTReduceOp::kMin), NRT_OP_MIN);
}

TEST_F(KernelExecutionTest, CompileOnlyKernelWithTensorizer) {
  std::vector<uint8_t> mlir_bytes = {0x4D, 0x4C, 0x49, 0x52};
  CompileOnlyKernelExecution kernel("cache_key", mlir_bytes, false, true);

  EXPECT_TRUE(kernel.IsValid());
  EXPECT_TRUE(kernel.RequiresCompilation());
  EXPECT_FALSE(kernel.HasCollectives());
}

TEST_F(KernelExecutionTest, CompileOnlyKernelTensorizerCompilationThrows) {
  std::vector<uint8_t> mlir_bytes = {0x4D, 0x4C, 0x49, 0x52};
  CompileOnlyKernelExecution kernel("key", mlir_bytes, false, true);

  // CompileToNeff should throw since we don't have real compiler
  try {
    kernel.CompileToNeff();
    FAIL() << "Expected exception";
  } catch (const std::exception& e) {
    EXPECT_TRUE(std::string(e.what()).find("Linalg") != std::string::npos)
        << "Error message should mention Linalg, got: " << e.what();
  }
}

TEST_F(KernelExecutionTest, CompileOnlyKernelTensorizerWithEmptyBytes) {
  std::vector<uint8_t> empty_bytes;
  CompileOnlyKernelExecution kernel("cache_key", empty_bytes, false, true);

  EXPECT_FALSE(kernel.IsValid());
}

// =============================================================================
// Compilation Error Tracking Tests
// =============================================================================

TEST_F(KernelExecutionTest, CompilationErrorDefaultState) {
  XLACompilableKernelExecution kernel("error_test", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "error_key",
                                      {0x01}, false, 0);
  EXPECT_FALSE(kernel.IsFailedCompilation());
  EXPECT_TRUE(kernel.GetCompilationError().empty());
}

TEST_F(KernelExecutionTest, SetAndGetCompilationError) {
  XLACompilableKernelExecution kernel("error_test", create_fake_tensor_refs(input_ptrs_),
                                      create_fake_tensor_refs(output_ptrs_), {}, {}, "error_key",
                                      {0x01}, false, 0);
  kernel.SetCompilationError("neuronx-cc: invalid HLO format");

  EXPECT_TRUE(kernel.IsFailedCompilation());
  EXPECT_EQ(kernel.GetCompilationError(), "neuronx-cc: invalid HLO format");
  EXPECT_FALSE(kernel.HasCachedNeff());
}

// =============================================================================
// Lazy Allocation Kernel Tests
// =============================================================================

TEST_F(KernelExecutionTest, AllocDirectKernelExecutionIsTyped) {
  void* va = output_ptrs_[0];
  TensorDataRef dst_ref(va);
  AllocDirectKernelExecution kernel("alloc_op", std::move(dst_ref), 1024, 0);

  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kAlloc);
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetSizeBytes(), 1024u);
  EXPECT_TRUE(kernel.IsValid());

  // dst_ptrs should contain the VA
  auto dst_ptrs = kernel.GetDstPtrs();
  EXPECT_EQ(dst_ptrs.size(), 1u);
  EXPECT_EQ(dst_ptrs[0], va);
}

TEST_F(KernelExecutionTest, DeallocDirectKernelExecutionIsTyped) {
  void* va = input_ptrs_[0];
  TensorDataRef src_ref(va);
  DeallocDirectKernelExecution kernel("dealloc_op", std::move(src_ref), 1024, 0);

  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kDealloc);
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_EQ(kernel.GetSizeBytes(), 1024u);
  EXPECT_TRUE(kernel.IsValid());

  // src_ptrs should contain the VA
  auto src_ptrs = kernel.GetSrcPtrs();
  EXPECT_EQ(src_ptrs.size(), 1u);
  EXPECT_EQ(src_ptrs[0], va);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
