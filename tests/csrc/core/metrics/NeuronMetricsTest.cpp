#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"

using namespace at::neuron::metrics;
using namespace std::chrono_literals;

class NeuronMetricsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Enable metrics for testing
    SetMetricsEnabled(true);

    // Clear all metrics and counters before each test
    NeuronMetricsArena::Get()->ClearCounters();
    NeuronMetricsArena::Get()->ClearMetrics();
  }

  void TearDown() override {
    // Clean up after each test
    NeuronMetricsArena::Get()->ClearCounters();
    NeuronMetricsArena::Get()->ClearMetrics();
  }
};

// ============================================================================
// NeuronCounter Tests
// ============================================================================

TEST_F(NeuronMetricsTest, CounterBasicOperations) {
  NeuronCounter counter("test.counter.basic");

  // Initial value should be 0
  EXPECT_EQ(counter.Value(), 0);

  // Add values
  counter.AddValue(1);
  EXPECT_EQ(counter.Value(), 1);

  counter.AddValue(5);
  EXPECT_EQ(counter.Value(), 6);

  counter.AddValue(10);
  EXPECT_EQ(counter.Value(), 16);
}

TEST_F(NeuronMetricsTest, CounterNegativeValues) {
  NeuronCounter counter("test.counter.negative");

  counter.AddValue(100);
  EXPECT_EQ(counter.Value(), 100);

  // Adding negative values should work (for decrement scenarios)
  counter.AddValue(-30);
  EXPECT_EQ(counter.Value(), 70);
}

TEST_F(NeuronMetricsTest, CounterThreadSafety) {
  NeuronCounter counter("test.counter.threadsafe");
  const int num_threads = 10;
  const int increments_per_thread = 1000;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&counter, increments_per_thread]() {
      for (int j = 0; j < increments_per_thread; ++j) {
        counter.AddValue(1);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(counter.Value(), num_threads * increments_per_thread);
}

TEST_F(NeuronMetricsTest, CounterRegistration) {
  NeuronCounter counter1("test.counter.reg1");
  NeuronCounter counter2("test.counter.reg2");

  counter1.AddValue(10);
  counter2.AddValue(20);

  // Verify counters are registered in the arena
  auto* arena = NeuronMetricsArena::Get();
  auto counter_names = arena->GetCounterNames();

  EXPECT_TRUE(std::find(counter_names.begin(), counter_names.end(), "test.counter.reg1") !=
              counter_names.end());
  EXPECT_TRUE(std::find(counter_names.begin(), counter_names.end(), "test.counter.reg2") !=
              counter_names.end());

  // Verify values through arena
  auto* c1 = arena->GetCounter("test.counter.reg1");
  auto* c2 = arena->GetCounter("test.counter.reg2");
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);
  EXPECT_EQ(c1->Value(), 10);
  EXPECT_EQ(c2->Value(), 20);
}

// ============================================================================
// NeuronMetric Tests
// ============================================================================

TEST_F(NeuronMetricsTest, MetricBasicOperations) {
  NeuronMetric metric("test.metric.basic");

  // Add samples
  metric.AddSample(10.0);
  metric.AddSample(20.0);
  metric.AddSample(30.0);

  // Accumulator should be sum of all samples
  EXPECT_DOUBLE_EQ(metric.Accumulator(), 60.0);

  // Get samples
  double accumulator;
  size_t total_samples;
  auto samples = metric.Samples(&accumulator, &total_samples);

  EXPECT_EQ(total_samples, 3);
  EXPECT_DOUBLE_EQ(accumulator, 60.0);
  EXPECT_EQ(samples.size(), 3);
}

TEST_F(NeuronMetricsTest, MetricWithTimestamp) {
  NeuronMetric metric("test.metric.timestamp");

  int64_t ts1 = 1000000000;  // 1 second in ns
  int64_t ts2 = 2000000000;  // 2 seconds in ns

  metric.AddSample(ts1, 100.0);
  metric.AddSample(ts2, 200.0);

  auto samples = metric.Samples();
  ASSERT_EQ(samples.size(), 2);
  EXPECT_EQ(samples[0].timestamp_ns, ts1);
  EXPECT_DOUBLE_EQ(samples[0].value, 100.0);
  EXPECT_EQ(samples[1].timestamp_ns, ts2);
  EXPECT_DOUBLE_EQ(samples[1].value, 200.0);
}

TEST_F(NeuronMetricsTest, MetricThreadSafety) {
  NeuronMetric metric("test.metric.threadsafe");
  const int num_threads = 10;
  const int samples_per_thread = 100;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&metric, samples_per_thread]() {
      for (int j = 0; j < samples_per_thread; ++j) {
        metric.AddSample(1.0);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  size_t total_samples;
  metric.Samples(nullptr, &total_samples);
  EXPECT_EQ(total_samples, num_threads * samples_per_thread);
  EXPECT_DOUBLE_EQ(metric.Accumulator(), num_threads * samples_per_thread * 1.0);
}

TEST_F(NeuronMetricsTest, MetricRegistration) {
  NeuronMetric metric1("test.metric.reg1");
  NeuronMetric metric2("test.metric.reg2", MetricFnBytes);

  metric1.AddSample(100.0);
  metric2.AddSample(1024.0);

  // Verify metrics are registered in the arena
  auto* arena = NeuronMetricsArena::Get();
  auto metric_names = arena->GetMetricNames();

  EXPECT_TRUE(std::find(metric_names.begin(), metric_names.end(), "test.metric.reg1") !=
              metric_names.end());
  EXPECT_TRUE(std::find(metric_names.begin(), metric_names.end(), "test.metric.reg2") !=
              metric_names.end());
}

// ============================================================================
// Representation Function Tests
// ============================================================================

TEST_F(NeuronMetricsTest, MetricFnValueRepresentation) {
  std::string result = MetricFnValue(123.456);
  EXPECT_THAT(result, ::testing::HasSubstr("123"));
}

TEST_F(NeuronMetricsTest, MetricFnBytesRepresentation) {
  // Test bytes formatting
  EXPECT_THAT(MetricFnBytes(500),
              ::testing::AnyOf(::testing::HasSubstr("500"), ::testing::HasSubstr("B")));
  EXPECT_THAT(MetricFnBytes(1024),
              ::testing::AnyOf(::testing::HasSubstr("1"), ::testing::HasSubstr("K")));
  EXPECT_THAT(MetricFnBytes(1024 * 1024),
              ::testing::AnyOf(::testing::HasSubstr("1"), ::testing::HasSubstr("M")));
  EXPECT_THAT(MetricFnBytes(1024 * 1024 * 1024),
              ::testing::AnyOf(::testing::HasSubstr("1"), ::testing::HasSubstr("G")));
}

TEST_F(NeuronMetricsTest, MetricFnTimeRepresentation) {
  // MetricFnTime expects nanoseconds
  EXPECT_THAT(MetricFnTime(1000),
              ::testing::AnyOf(::testing::HasSubstr("us"), ::testing::HasSubstr("ns"),
                               ::testing::HasSubstr("1")));
  EXPECT_THAT(MetricFnTime(1000000),
              ::testing::AnyOf(::testing::HasSubstr("ms"), ::testing::HasSubstr("us"),
                               ::testing::HasSubstr("1")));
  EXPECT_THAT(MetricFnTime(1000000000),
              ::testing::AnyOf(::testing::HasSubstr("s"), ::testing::HasSubstr("ms"),
                               ::testing::HasSubstr("1")));
}

// ============================================================================
// TimedSection Tests
// ============================================================================

TEST_F(NeuronMetricsTest, TimedSectionBasic) {
  NeuronMetric metric("test.metric.timed", MetricFnTime);

  {
    TimedSection section(&metric);
    std::this_thread::sleep_for(10ms);
  }

  // Should have recorded one sample
  size_t total_samples;
  metric.Samples(nullptr, &total_samples);
  EXPECT_EQ(total_samples, 1);

  // Accumulator should be > 0 (time elapsed)
  EXPECT_GT(metric.Accumulator(), 0);
}

TEST_F(NeuronMetricsTest, TimedSectionElapsed) {
  NeuronMetric metric("test.metric.elapsed", MetricFnTime);

  TimedSection section(&metric);
  std::this_thread::sleep_for(50ms);
  double elapsed = section.Elapsed();

  // Elapsed should be approximately 50ms (0.05 seconds)
  EXPECT_GT(elapsed, 0.04);
  EXPECT_LT(elapsed, 0.2);  // Allow some variance
}

// ============================================================================
// NeuronMetricsArena Tests
// ============================================================================

TEST_F(NeuronMetricsTest, ArenaForEachCounter) {
  NeuronCounter counter1("test.arena.counter1");
  NeuronCounter counter2("test.arena.counter2");

  counter1.AddValue(10);
  counter2.AddValue(20);

  int counter_count = 0;
  int64_t total_value = 0;

  NeuronMetricsArena::Get()->ForEachCounter(
      [&counter_count, &total_value](const std::string& name, NeuronCounterData* data) {
        if (name.find("test.arena.counter") != std::string::npos) {
          counter_count++;
          total_value += data->Value();
        }
      });

  EXPECT_EQ(counter_count, 2);
  EXPECT_EQ(total_value, 30);
}

TEST_F(NeuronMetricsTest, ArenaForEachMetric) {
  NeuronMetric metric1("test.arena.metric1");
  NeuronMetric metric2("test.arena.metric2");

  metric1.AddSample(100.0);
  metric2.AddSample(200.0);

  int metric_count = 0;
  double total_accumulator = 0;

  NeuronMetricsArena::Get()->ForEachMetric(
      [&metric_count, &total_accumulator](const std::string& name, NeuronMetricData* data) {
        if (name.find("test.arena.metric") != std::string::npos) {
          metric_count++;
          total_accumulator += data->Accumulator();
        }
      });

  EXPECT_EQ(metric_count, 2);
  EXPECT_DOUBLE_EQ(total_accumulator, 300.0);
}

TEST_F(NeuronMetricsTest, ArenaClearCounters) {
  NeuronCounter counter("test.arena.clear.counter");
  counter.AddValue(100);
  EXPECT_EQ(counter.Value(), 100);

  NeuronMetricsArena::Get()->ClearCounters();

  // After clearing, counter should be 0
  auto* data = NeuronMetricsArena::Get()->GetCounter("test.arena.clear.counter");
  if (data) {
    EXPECT_EQ(data->Value(), 0);
  }
}

TEST_F(NeuronMetricsTest, ArenaClearMetrics) {
  NeuronMetric metric("test.arena.clear.metric");
  metric.AddSample(100.0);
  EXPECT_DOUBLE_EQ(metric.Accumulator(), 100.0);

  NeuronMetricsArena::Get()->ClearMetrics();

  // After clearing, metric should have no samples
  auto* data = NeuronMetricsArena::Get()->GetMetric("test.arena.clear.metric");
  if (data) {
    EXPECT_DOUBLE_EQ(data->Accumulator(), 0.0);
  }
}

// ============================================================================
// CreateMetricReport Tests
// ============================================================================

TEST_F(NeuronMetricsTest, CreateMetricReportBasic) {
  NeuronCounter counter("test.report.counter");
  NeuronMetric metric("test.report.metric");

  counter.AddValue(42);
  metric.AddSample(123.0);

  std::string report = CreateMetricReport();

  // Report should contain our counter and metric
  EXPECT_THAT(report, ::testing::HasSubstr("test.report.counter"));
  EXPECT_THAT(report, ::testing::HasSubstr("test.report.metric"));
  EXPECT_THAT(report, ::testing::HasSubstr("42"));
}

TEST_F(NeuronMetricsTest, CreateMetricReportFiltered) {
  NeuronCounter counter1("test.filter.counter1");
  NeuronCounter counter2("test.filter.counter2");
  NeuronMetric metric1("test.filter.metric1");
  NeuronMetric metric2("test.filter.metric2");

  counter1.AddValue(10);
  counter2.AddValue(20);
  metric1.AddSample(100.0);
  metric2.AddSample(200.0);

  // Create report with only specific counters and metrics
  std::string report = CreateMetricReport({"test.filter.counter1"}, {"test.filter.metric1"});

  EXPECT_THAT(report, ::testing::HasSubstr("test.filter.counter1"));
  EXPECT_THAT(report, ::testing::HasSubstr("test.filter.metric1"));
  // Should not contain the filtered out ones
  EXPECT_THAT(report, ::testing::Not(::testing::HasSubstr("test.filter.counter2")));
  EXPECT_THAT(report, ::testing::Not(::testing::HasSubstr("test.filter.metric2")));
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(NeuronMetricsTest, GetCurrentTimeNs) {
  int64_t time1 = GetCurrentTimeNs();
  std::this_thread::sleep_for(1ms);
  int64_t time2 = GetCurrentTimeNs();

  // time2 should be greater than time1
  EXPECT_GT(time2, time1);

  // Difference should be at least 1ms (1,000,000 ns)
  EXPECT_GE(time2 - time1, 1000000);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(NeuronMetricsTest, EmptyMetricName) {
  // Empty name should still work
  NeuronCounter counter("");
  counter.AddValue(1);
  EXPECT_EQ(counter.Value(), 1);
}

TEST_F(NeuronMetricsTest, LargeValues) {
  NeuronCounter counter("test.large.counter");
  NeuronMetric metric("test.large.metric");

  int64_t large_value = 1000000000000LL;  // 1 trillion
  counter.AddValue(large_value);
  metric.AddSample(static_cast<double>(large_value));

  EXPECT_EQ(counter.Value(), large_value);
  EXPECT_DOUBLE_EQ(metric.Accumulator(), static_cast<double>(large_value));
}

TEST_F(NeuronMetricsTest, ZeroValues) {
  NeuronCounter counter("test.zero.counter");
  NeuronMetric metric("test.zero.metric");

  counter.AddValue(0);
  metric.AddSample(0.0);

  EXPECT_EQ(counter.Value(), 0);
  EXPECT_DOUBLE_EQ(metric.Accumulator(), 0.0);
}

TEST_F(NeuronMetricsTest, NegativeMetricValues) {
  NeuronMetric metric("test.negative.metric");

  metric.AddSample(-10.0);
  metric.AddSample(30.0);
  metric.AddSample(-5.0);

  EXPECT_DOUBLE_EQ(metric.Accumulator(), 15.0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
