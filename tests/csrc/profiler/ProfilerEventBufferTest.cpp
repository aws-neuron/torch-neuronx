#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"

namespace at::neuron {
namespace {

TEST(ProfilerEventBufferTest, InactiveBufferIgnoresRecords) {
  ProfilerEventBuffer buffer;
  EXPECT_FALSE(buffer.isActive());

  ProfiledOperation op;
  op.op_name = "test_op";
  buffer.record(std::move(op));

  auto events = buffer.drain();
  EXPECT_TRUE(events.empty());
}

TEST(ProfilerEventBufferTest, ActiveBufferRecordsEvents) {
  ProfilerEventBuffer buffer;
  buffer.activate();
  EXPECT_TRUE(buffer.isActive());

  ProfiledOperation op;
  op.op_name = "test_op";
  op.submit_start_ns = 50;
  op.submit_time_ns = 100;
  op.execute_start_ns = 200;
  op.execute_end_ns = 300;
  op.neuron_stream_os_tid = 9999;
  op.pytorch_os_tid = 8888;
  buffer.record(std::move(op));

  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].op_name, "test_op");
  EXPECT_EQ(events[0].submit_start_ns, 50);
  EXPECT_EQ(events[0].submit_time_ns, 100);
  EXPECT_EQ(events[0].execute_start_ns, 200);
  EXPECT_EQ(events[0].execute_end_ns, 300);
  EXPECT_EQ(events[0].neuron_stream_os_tid, 9999);
  EXPECT_EQ(events[0].pytorch_os_tid, 8888);
}

TEST(ProfilerEventBufferTest, DrainClearsBuffer) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  ProfiledOperation op;
  op.op_name = "op1";
  buffer.record(std::move(op));

  auto events1 = buffer.drain();
  ASSERT_EQ(events1.size(), 1);

  auto events2 = buffer.drain();
  EXPECT_TRUE(events2.empty());
}

TEST(ProfilerEventBufferTest, DeactivateStopsRecording) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  ProfiledOperation op1;
  op1.op_name = "before_deactivate";
  buffer.record(std::move(op1));

  buffer.deactivate();
  EXPECT_FALSE(buffer.isActive());

  ProfiledOperation op2;
  op2.op_name = "after_deactivate";
  buffer.record(std::move(op2));

  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].op_name, "before_deactivate");
}

TEST(ProfilerEventBufferTest, ConcurrentRecordFromMultipleThreads) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&buffer, t]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        ProfiledOperation op;
        op.op_name = "thread_" + std::to_string(t) + "_op_" + std::to_string(i);
        op.device_index = static_cast<int32_t>(t);
        buffer.record(std::move(op));
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto events = buffer.drain();
  EXPECT_EQ(events.size(), kThreads * kOpsPerThread);
}

TEST(ProfilerEventBufferTest, ActiveBufferRecordsCacheAndHierarchyFields) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  ProfiledOperation op;
  op.op_name = "test_op";
  op.submit_start_ns = 50;
  op.submit_time_ns = 100;
  op.execute_start_ns = 200;
  op.execute_end_ns = 300;
  op.hierarchy_string = "Linear[fc1][1]/";
  op.neff_cache_key = "linear--_[8,4096]:BFloat16:abc123";
  op.persistent_cache_key = "00813ff53bb82d57870f6335f192ae50";
  buffer.record(std::move(op));

  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].hierarchy_string, "Linear[fc1][1]/");
  EXPECT_EQ(events[0].neff_cache_key, "linear--_[8,4096]:BFloat16:abc123");
  EXPECT_EQ(events[0].persistent_cache_key, "00813ff53bb82d57870f6335f192ae50");
}

TEST(ProfilerEventBufferTest, ActiveBufferRecordsDispatchFields) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  ProfiledOperation op;
  op.op_name = "dispatch_test_op";
  op.submit_start_ns = 100;
  op.submit_time_ns = 150;
  op.execute_start_ns = 300;
  op.execute_end_ns = 500;
  op.dispatch_start_ns = 250;
  op.xu_type = 1;  // NRTA_XU_COMPUTE
  op.dispatch_slot = 3;
  op.pytorch_sequence_nr = 42;
  buffer.record(std::move(op));

  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].op_name, "dispatch_test_op");
  EXPECT_EQ(events[0].dispatch_start_ns, 250);
  EXPECT_EQ(events[0].xu_type, 1);
  EXPECT_EQ(events[0].dispatch_slot, 3);
  EXPECT_EQ(events[0].pytorch_sequence_nr, 42);
  // Verify dispatch_start < execute_start (dispatch encapsulates execution)
  EXPECT_LT(events[0].dispatch_start_ns, events[0].execute_start_ns);
}

TEST(ProfilerEventBufferTest, DefaultDispatchFieldsAreUnset) {
  ProfilerEventBuffer buffer;
  buffer.activate();

  ProfiledOperation op;
  op.op_name = "no_dispatch_op";
  buffer.record(std::move(op));

  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].dispatch_start_ns, 0);
  EXPECT_EQ(events[0].xu_type, -1);
  EXPECT_EQ(events[0].dispatch_slot, -1);
  EXPECT_EQ(events[0].pytorch_sequence_nr, -1);
}

TEST(ProfilerEventBufferTest, GlobalPointerDefaultsToNull) {
  EXPECT_EQ(getActiveProfilerBuffer(), nullptr);
}

TEST(ProfilerEventBufferTest, GlobalPointerSetAndClear) {
  ProfilerEventBuffer buffer;

  setActiveProfilerBuffer(&buffer);
  EXPECT_EQ(getActiveProfilerBuffer(), &buffer);

  setActiveProfilerBuffer(nullptr);
  EXPECT_EQ(getActiveProfilerBuffer(), nullptr);
}

TEST(ProfilerEventBufferTest, ToNsConvertsTimePoint) {
  auto now = std::chrono::steady_clock::now();
  int64_t ns = toNs(now);
  EXPECT_GT(ns, 0);
}

}  // namespace
}  // namespace at::neuron
