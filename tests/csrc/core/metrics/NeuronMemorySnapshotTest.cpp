#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/metrics/NeuronMemorySnapshot.h"

using namespace c10_neuron::NeuronMemorySnapshot;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace {

class NeuronMemorySnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock session
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

    // Clear any previous mock expectations
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);

    // Disable recording and clear history before each test
    RecordHistory(false, RecordContext::NEVER, false, 1, true);
  }

  void TearDown() override {
    // Disable recording and clear history after each test
    RecordHistory(false, RecordContext::NEVER, false, 1, true);

    // Clear mock expectations
    ::testing::Mock::VerifyAndClearExpectations(mock_nrt_);
    mock_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

// =============================================================================
// TraceAction String Conversion Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringAlloc) {
  EXPECT_STREQ(TraceActionToString(TraceAction::ALLOC), "alloc");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringFreeRequested) {
  EXPECT_STREQ(TraceActionToString(TraceAction::FREE_REQUESTED), "free_requested");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringFreeCompleted) {
  EXPECT_STREQ(TraceActionToString(TraceAction::FREE_COMPLETED), "free_completed");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringSegmentAlloc) {
  EXPECT_STREQ(TraceActionToString(TraceAction::SEGMENT_ALLOC), "segment_alloc");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringSegmentFree) {
  EXPECT_STREQ(TraceActionToString(TraceAction::SEGMENT_FREE), "segment_free");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringSegmentMap) {
  EXPECT_STREQ(TraceActionToString(TraceAction::SEGMENT_MAP), "segment_map");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringSegmentUnmap) {
  EXPECT_STREQ(TraceActionToString(TraceAction::SEGMENT_UNMAP), "segment_unmap");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringSnapshot) {
  EXPECT_STREQ(TraceActionToString(TraceAction::SNAPSHOT), "snapshot");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringOom) {
  EXPECT_STREQ(TraceActionToString(TraceAction::OOM), "oom");
}

TEST_F(NeuronMemorySnapshotTest, TraceActionToStringUnknown) {
  // Test with an invalid enum value (cast to simulate unknown)
  EXPECT_STREQ(TraceActionToString(static_cast<TraceAction>(999)), "unknown");
}

// =============================================================================
// RecordContext String Conversion Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextNever) {
  EXPECT_EQ(StringToRecordContext("never"), RecordContext::NEVER);
}

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextNone) {
  EXPECT_EQ(StringToRecordContext("none"), RecordContext::NEVER);
}

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextState) {
  EXPECT_EQ(StringToRecordContext("state"), RecordContext::STATE);
}

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextAlloc) {
  EXPECT_EQ(StringToRecordContext("alloc"), RecordContext::ALLOC);
}

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextAll) {
  EXPECT_EQ(StringToRecordContext("all"), RecordContext::ALL);
}

TEST_F(NeuronMemorySnapshotTest, StringToRecordContextInvalid) {
  EXPECT_THROW(StringToRecordContext("invalid"), std::invalid_argument);
  EXPECT_THROW(StringToRecordContext(""), std::invalid_argument);
  EXPECT_THROW(StringToRecordContext("NEVER"), std::invalid_argument);  // Case sensitive
}

// =============================================================================
// Recording State Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, RecordingInitiallyDisabled) { EXPECT_FALSE(IsRecordingEnabled()); }

TEST_F(NeuronMemorySnapshotTest, EnableRecording) {
  RecordHistory(true, RecordContext::ALLOC, false, 100, false);
  EXPECT_TRUE(IsRecordingEnabled());
  EXPECT_EQ(GetRecordContext(), RecordContext::ALLOC);
}

TEST_F(NeuronMemorySnapshotTest, DisableRecording) {
  RecordHistory(true, RecordContext::ALL, false, 100, false);
  EXPECT_TRUE(IsRecordingEnabled());

  RecordHistory(false, RecordContext::NEVER, false, 1, false);
  EXPECT_FALSE(IsRecordingEnabled());
}

TEST_F(NeuronMemorySnapshotTest, RecordContextLevels) {
  RecordHistory(true, RecordContext::NEVER, false, 100, false);
  EXPECT_EQ(GetRecordContext(), RecordContext::NEVER);

  RecordHistory(true, RecordContext::STATE, false, 100, false);
  EXPECT_EQ(GetRecordContext(), RecordContext::STATE);

  RecordHistory(true, RecordContext::ALLOC, false, 100, false);
  EXPECT_EQ(GetRecordContext(), RecordContext::ALLOC);

  RecordHistory(true, RecordContext::ALL, false, 100, false);
  EXPECT_EQ(GetRecordContext(), RecordContext::ALL);
}

TEST_F(NeuronMemorySnapshotTest, IncludeCppFrames) {
  RecordHistory(true, RecordContext::ALL, false, 100, false);
  EXPECT_FALSE(ShouldIncludeCppFrames());

  RecordHistory(true, RecordContext::ALL, true, 100, false);
  EXPECT_TRUE(ShouldIncludeCppFrames());
}

// =============================================================================
// Trace Recording Tests (via public API)
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, RecordTraceWhenDisabled) {
  // Recording is disabled by default
  EXPECT_FALSE(IsRecordingEnabled());

  // Record a trace - should be ignored
  RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x1000, 4096, 0, nullptr, 0, 0);

  // Take snapshot - should have no traces
  auto snap = snapshot();
  // device_traces may be empty or have empty vectors
  size_t total_traces = 0;
  for (const auto& traces : snap.device_traces) {
    total_traces += traces.size();
  }
  EXPECT_EQ(total_traces, 0);
}

TEST_F(NeuronMemorySnapshotTest, RecordTraceWhenEnabled) {
  // Mock vnc_count for device operations
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

  // Enable recording
  RecordHistory(true, RecordContext::ALLOC, false, 100, true);
  EXPECT_TRUE(IsRecordingEnabled());

  // Record some traces
  RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x1000, 4096, 0, nullptr, 0, 0);
  RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x2000, 8192, 0, nullptr, 0, 0);
  RecordTrace(TraceAction::FREE_REQUESTED, 0, 0x1000, 4096, 0, nullptr, 0, 0);

  // Take snapshot
  auto snap = snapshot();

  // Should have traces for device 0
  ASSERT_GE(snap.device_traces.size(), 1u);
  // 3 recorded + 1 SNAPSHOT event
  EXPECT_GE(snap.device_traces[0].size(), 3u);
}

TEST_F(NeuronMemorySnapshotTest, ClearHistory) {
  // Mock vnc_count
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

  // Enable recording and add traces
  RecordHistory(true, RecordContext::ALLOC, false, 100, true);
  RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x1000, 4096, 0, nullptr, 0, 0);
  RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x2000, 8192, 0, nullptr, 0, 0);

  // Clear history
  RecordHistory(true, RecordContext::ALLOC, false, 100, true);

  // Take snapshot - should have no traces (except possibly SNAPSHOT event)
  auto snap = snapshot();
  size_t total_traces = 0;
  for (const auto& traces : snap.device_traces) {
    for (const auto& trace : traces) {
      if (trace.action != TraceAction::SNAPSHOT) {
        total_traces++;
      }
    }
  }
  EXPECT_EQ(total_traces, 0);
}

// =============================================================================
// Ring Buffer Capacity Tests (via public API)
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, RingBufferCapacity) {
  // Mock vnc_count
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

  // Enable recording with max_entries = 5
  RecordHistory(true, RecordContext::ALLOC, false, 5, true);

  // Record 10 traces (more than capacity)
  for (int i = 0; i < 10; i++) {
    RecordTrace(TraceAction::SEGMENT_ALLOC, 0, 0x1000 + i * 0x1000, 4096, 0, nullptr, 0, 0);
  }

  // Take snapshot
  auto snap = snapshot();

  // Should have at most 5 traces (ring buffer capacity)
  // Note: snapshot() also adds a SNAPSHOT event, so we check <= 6
  ASSERT_GE(snap.device_traces.size(), 1u);
  EXPECT_LE(snap.device_traces[0].size(), 6u);
}

// =============================================================================
// TraceEntry Structure Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, TraceEntryDefaultConstructor) {
  TraceEntry entry;
  EXPECT_EQ(entry.device, 0);
  EXPECT_EQ(entry.addr, 0u);
  EXPECT_EQ(entry.size, 0u);
  EXPECT_EQ(entry.stream, 0);
  EXPECT_EQ(entry.timestamp_us, 0);
  EXPECT_EQ(entry.context, nullptr);
  EXPECT_EQ(entry.device_free, 0u);
}

TEST_F(NeuronMemorySnapshotTest, TraceEntryParameterizedConstructor) {
  auto ctx = std::make_shared<StackContext>();
  TraceEntry entry(TraceAction::OOM, 1, 0x5000, 1024, 42, 123456, ctx, 999, 0);

  EXPECT_EQ(entry.action, TraceAction::OOM);
  EXPECT_EQ(entry.device, 1);
  EXPECT_EQ(entry.addr, 0x5000u);
  EXPECT_EQ(entry.size, 1024u);
  EXPECT_EQ(entry.stream, 42);
  EXPECT_EQ(entry.timestamp_us, 123456);
  EXPECT_EQ(entry.context, ctx);
  EXPECT_EQ(entry.device_free, 999u);
  EXPECT_EQ(entry.block_size, 1024u);  // defaults to size when blk_sz=0
}

// =============================================================================
// StackFrame and StackContext Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, StackFrameDefaultConstructor) {
  StackFrame frame;
  EXPECT_TRUE(frame.filename.empty());
  EXPECT_TRUE(frame.funcname.empty());
  EXPECT_EQ(frame.lineno, 0);
}

TEST_F(NeuronMemorySnapshotTest, StackFrameParameterizedConstructor) {
  StackFrame frame("test.py", "test_func", 42);
  EXPECT_EQ(frame.filename, "test.py");
  EXPECT_EQ(frame.funcname, "test_func");
  EXPECT_EQ(frame.lineno, 42);
}

TEST_F(NeuronMemorySnapshotTest, StackContextEmptyFrames) {
  StackContext ctx;
  EXPECT_TRUE(ctx.frames.empty());
}

TEST_F(NeuronMemorySnapshotTest, StackContextWithFrames) {
  StackContext ctx;
  ctx.frames.emplace_back("file1.py", "func1", 10);
  ctx.frames.emplace_back("file2.py", "func2", 20);

  EXPECT_EQ(ctx.frames.size(), 2u);
  EXPECT_EQ(ctx.frames[0].filename, "file1.py");
  EXPECT_EQ(ctx.frames[1].funcname, "func2");
}

// =============================================================================
// BlockInfo and SegmentInfo Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, BlockInfoDefaults) {
  BlockInfo block;
  EXPECT_EQ(block.size, 0u);
  EXPECT_EQ(block.requested_size, 0u);
  EXPECT_EQ(block.address, 0u);
  EXPECT_FALSE(block.allocated);
  EXPECT_EQ(block.context, nullptr);
}

TEST_F(NeuronMemorySnapshotTest, SegmentInfoDefaults) {
  SegmentInfo seg;
  EXPECT_EQ(seg.device, 0);
  EXPECT_EQ(seg.address, 0u);
  EXPECT_EQ(seg.total_size, 0u);
  EXPECT_EQ(seg.allocated_size, 0u);
  EXPECT_EQ(seg.active_size, 0u);
  EXPECT_EQ(seg.stream, 0);
  EXPECT_FALSE(seg.is_small);
  EXPECT_TRUE(seg.blocks.empty());
}

// =============================================================================
// Timestamp Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, GetCurrentTimestampUs) {
  int64_t ts1 = GetCurrentTimestampUs();
  // Small delay
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  int64_t ts2 = GetCurrentTimestampUs();

  // Timestamps should be positive and increasing
  EXPECT_GT(ts1, 0);
  EXPECT_GT(ts2, 0);
  EXPECT_GE(ts2, ts1);
}

TEST_F(NeuronMemorySnapshotTest, ClockDomainOffsetAlignsTimestampToSystemClock) {
  // Enabling recording computes the clock domain offset so memory timestamps
  // align with the runtime system trace (both use system_clock).
  RecordHistory(true, RecordContext::ALLOC, false, 100, true);

  int64_t sys_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  int64_t ts = GetCurrentTimestampUs();

  // The offset shifts the steady-clock reading into the system-clock domain.
  // Allow 1 second of slack for scheduling jitter.
  EXPECT_NEAR(static_cast<double>(ts), static_cast<double>(sys_now_us), 1000000.0);
}

// =============================================================================
// Trace ID Correlation Tests
// =============================================================================

TEST_F(NeuronMemorySnapshotTest, TraceEntryStoresTraceId) {
  TraceEntry entry(TraceAction::ALLOC, 0, 0x1000, 4096, 0, 123, nullptr, 0, 4096, 0xDEADBEEF);
  EXPECT_EQ(entry.trace_id, 0xDEADBEEFu);
}

TEST_F(NeuronMemorySnapshotTest, TraceEntryTraceIdDefaultsToZero) {
  TraceEntry entry;
  EXPECT_EQ(entry.trace_id, 0u);
}

TEST_F(NeuronMemorySnapshotTest, RecordTracePropagatesTraceId) {
  EXPECT_CALL(*mock_nrt_, nrt_get_visible_vnc_count(_))
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

  RecordHistory(true, RecordContext::ALLOC, false, 100, true);

  const uint64_t kTraceId = 0xABCDEF;
  RecordTrace(TraceAction::ALLOC, 0, 0x3000, 4096, 0, nullptr, 0, 4096, kTraceId);

  auto snap = snapshot();
  ASSERT_GE(snap.device_traces.size(), 1u);

  bool found = false;
  for (const auto& entry : snap.device_traces[0]) {
    if (entry.addr == 0x3000u) {
      EXPECT_EQ(entry.trace_id, kTraceId);
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
