#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/NeuronOpTracking.h"

using namespace torch_neuronx;

class NeuronOpTrackingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    clearOpTracking();
    ConcatenationProfiler::getInstance().set_enabled(false);
  }
  void TearDown() override {
    clearOpTracking();
    ConcatenationProfiler::getInstance().set_enabled(false);
  }
};

// ============================================================================
// ConcatenatedOpTracker Tests
// ============================================================================

TEST_F(NeuronOpTrackingTest, TrackerEmptyOnStart) {
  EXPECT_EQ(getConcatenatedOps().size(), 0);
  EXPECT_EQ(getConcatenatedOpExecutionCounts().size(), 0);
}

TEST_F(NeuronOpTrackingTest, TrackerRecordAndRetrieve) {
  std::vector<std::string> ops = {"aten::mm", "aten::add.Tensor"};
  logConcatenatedOpExecuted("aten::mm|aten::add.Tensor", ops);

  auto result = getConcatenatedOps();
  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(result.count("aten::mm|aten::add.Tensor"));
  EXPECT_EQ(result["aten::mm|aten::add.Tensor"].size(), 2);
  EXPECT_EQ(result["aten::mm|aten::add.Tensor"][0], "aten::mm");
  EXPECT_EQ(result["aten::mm|aten::add.Tensor"][1], "aten::add.Tensor");
}

TEST_F(NeuronOpTrackingTest, TrackerDeduplicationAndExecutionCounts) {
  // Same group recorded 3 times (simulates 3 iterations)
  logConcatenatedOpExecuted("op_a|op_b", {"op_a", "op_b"});
  logConcatenatedOpExecuted("op_a|op_b", {"op_a", "op_b"});
  logConcatenatedOpExecuted("op_a|op_b", {"op_a", "op_b"});
  // Different group recorded once
  logConcatenatedOpExecuted("op_c|op_d", {"op_c", "op_d"});

  // Verify deduplication: ops map should have 2 entries, not 4
  auto ops = getConcatenatedOps();
  ASSERT_EQ(ops.size(), 2);
  EXPECT_EQ(ops["op_a|op_b"].size(), 2);
  EXPECT_EQ(ops["op_c|op_d"].size(), 2);

  // Verify execution counts track first-time vs cached
  auto counts = getConcatenatedOpExecutionCounts();
  ASSERT_EQ(counts.size(), 2);
  EXPECT_EQ(counts["op_a|op_b"], 3);  // 1 compiled + 2 cache hits
  EXPECT_EQ(counts["op_c|op_d"], 1);  // 1 compiled, no cache hits
}

TEST_F(NeuronOpTrackingTest, TrackerClear) {
  logConcatenatedOpExecuted("op_a|op_b", {"op_a", "op_b"});
  ASSERT_GT(getConcatenatedOps().size(), 0);

  clearOpTracking();
  EXPECT_EQ(getConcatenatedOps().size(), 0);
  EXPECT_EQ(getConcatenatedOpExecutionCounts().size(), 0);
}

// ============================================================================
// ConcatenationProfiler Tests
// ============================================================================

TEST_F(NeuronOpTrackingTest, ProfilerDisabledByDefault) {
  EXPECT_FALSE(ConcatenationProfiler::getInstance().is_enabled());
}

TEST_F(NeuronOpTrackingTest, ProfilerNoRecordWhenDisabled) {
  logConcatenationProfile("op_a|op_b", 1000000.0, 2);
  EXPECT_EQ(getConcatenationProfiles().size(), 0);
}

TEST_F(NeuronOpTrackingTest, ProfilerRecordWhenEnabled) {
  ConcatenationProfiler::getInstance().set_enabled(true);

  // Two invocations of same type, one of different type
  logConcatenationProfile("op_a|op_b", 1000000.0, 2);
  logConcatenationProfile("op_a|op_b", 500000.0, 2);
  logConcatenationProfile("op_c|op_d|op_e", 2000000.0, 3);

  auto profiles = getConcatenationProfiles();
  ASSERT_EQ(profiles.size(), 2);

  auto& ab = profiles["op_a|op_b"];
  EXPECT_EQ(ab.invocation_count, 2);
  ASSERT_EQ(ab.merge_times_ns.size(), 2);
  EXPECT_DOUBLE_EQ(ab.merge_times_ns[0], 1000000.0);
  EXPECT_DOUBLE_EQ(ab.merge_times_ns[1], 500000.0);
  EXPECT_EQ(ab.group_sizes[0], 2);

  auto& cde = profiles["op_c|op_d|op_e"];
  EXPECT_EQ(cde.invocation_count, 1);
  EXPECT_EQ(cde.group_sizes[0], 3);
}

TEST_F(NeuronOpTrackingTest, ProfilerClearedByOpTrackingClear) {
  ConcatenationProfiler::getInstance().set_enabled(true);
  logConcatenationProfile("op_a|op_b", 1000000.0, 2);
  ASSERT_EQ(getConcatenationProfiles().size(), 1);

  clearOpTracking();  // Should also clear profiler data
  EXPECT_EQ(getConcatenationProfiles().size(), 0);
}
