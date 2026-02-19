#include <gtest/gtest.h>

#include <vector>

#include "torch_neuronx/csrc/core/ProfilerMappingCollector.h"

using namespace at::neuron;

class ProfilerMappingCollectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& collector = ProfilerMappingCollector::Instance();
    collector.SetEnabled(false);
    collector.GetAndClear();
  }

  void TearDown() override {
    auto& collector = ProfilerMappingCollector::Instance();
    collector.SetEnabled(false);
    collector.GetAndClear();
  }
};

TEST_F(ProfilerMappingCollectorTest, DisabledByDefault) {
  auto& collector = ProfilerMappingCollector::Instance();
  EXPECT_FALSE(collector.IsEnabled());
}

TEST_F(ProfilerMappingCollectorTest, EnableDisable) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  EXPECT_TRUE(collector.IsEnabled());
  collector.SetEnabled(false);
  EXPECT_FALSE(collector.IsEnabled());
}

TEST_F(ProfilerMappingCollectorTest, RecordWhenDisabled) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.Record(123, 42, 7890, 1, 12345);
  auto mappings = collector.GetAndClear();
  EXPECT_TRUE(mappings.empty());
}

TEST_F(ProfilerMappingCollectorTest, RecordWhenEnabled) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  collector.Record(123, 42, 7890, 0, 12345);
  auto mappings = collector.GetAndClear();

  ASSERT_EQ(mappings.size(), 1);
  ASSERT_EQ(mappings[123].size(), 1);
  EXPECT_EQ(mappings[123][0].seq_nr, 42);
  EXPECT_EQ(mappings[123][0].th_id, 7890);
  EXPECT_EQ(mappings[123][0].stream_id, 0);
  EXPECT_EQ(mappings[123][0].os_tid, 12345);
}

TEST_F(ProfilerMappingCollectorTest, MultipleFrameworkIdsPerSequenceId) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  collector.Record(123, 42, 7890, 0, 12345);
  collector.Record(123, 43, 7890, 0, 12345);
  collector.Record(123, 44, 7890, 0, 12345);
  auto mappings = collector.GetAndClear();

  ASSERT_EQ(mappings.size(), 1);
  ASSERT_EQ(mappings[123].size(), 3);
  EXPECT_EQ(mappings[123][0].seq_nr, 42);
  EXPECT_EQ(mappings[123][1].seq_nr, 43);
  EXPECT_EQ(mappings[123][2].seq_nr, 44);
}

TEST_F(ProfilerMappingCollectorTest, MultipleSequenceIds) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  collector.Record(123, 42, 7890, 0, 12345);
  collector.Record(124, 43, 7890, 1, 12345);
  auto mappings = collector.GetAndClear();

  ASSERT_EQ(mappings.size(), 2);
  EXPECT_EQ(mappings[123][0].seq_nr, 42);
  EXPECT_EQ(mappings[123][0].stream_id, 0);
  EXPECT_EQ(mappings[124][0].seq_nr, 43);
  EXPECT_EQ(mappings[124][0].stream_id, 1);
}

TEST_F(ProfilerMappingCollectorTest, GetAndClearClears) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  collector.Record(123, 42, 7890, 0, 12345);
  collector.GetAndClear();
  auto mappings = collector.GetAndClear();
  EXPECT_TRUE(mappings.empty());
}

TEST_F(ProfilerMappingCollectorTest, SequenceNrOrdering) {
  auto& collector = ProfilerMappingCollector::Instance();
  collector.SetEnabled(true);
  collector.Record(100, 1, 1000, 0, 12345);
  collector.Record(200, 2, 1000, 0, 12345);
  collector.Record(300, 3, 1000, 0, 12345);
  auto mappings = collector.GetAndClear();

  // Verify entries exist and preserve insertion order within each seq_id
  ASSERT_EQ(mappings.count(100), 1);
  ASSERT_EQ(mappings.count(200), 1);
  ASSERT_EQ(mappings.count(300), 1);
  ASSERT_EQ(mappings[100].size(), 1);
  ASSERT_EQ(mappings[200].size(), 1);
  ASSERT_EQ(mappings[300].size(), 1);
  EXPECT_EQ(mappings[100][0].seq_nr, 1);
  EXPECT_EQ(mappings[200][0].seq_nr, 2);
  EXPECT_EQ(mappings[300][0].seq_nr, 3);
}
