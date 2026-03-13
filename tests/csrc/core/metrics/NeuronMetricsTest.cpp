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

    // Reset device memory stats for device 0 (used in tests)
    auto& registry = DeviceMemoryStatsRegistry::Instance();
    auto& stats = registry.GetDeviceStats(0);
    stats.reset_all();  // Reset to default state
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

// ============================================================================
// MemoryStatistics Tests
// ============================================================================

TEST_F(NeuronMetricsTest, MemoryStatIncreaseUpdatesCurrent) {
  MemoryStatistics stat;
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 0);

  stat.increase(100);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 100);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);

  stat.increase(50);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 150);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 150);
}

TEST_F(NeuronMetricsTest, MemoryStatDecreaseUpdatesCurrent) {
  MemoryStatistics stat;
  stat.increase(100);

  stat.decrease(30);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 70);
  // Peak should remain unchanged
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);
}

TEST_F(NeuronMetricsTest, MemoryStatPeakTracking) {
  MemoryStatistics stat;

  // Increase to 100
  stat.increase(100);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);

  // Decrease to 50 - peak should stay at 100
  stat.decrease(50);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 50);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);

  // Increase to 80 - peak should stay at 100
  stat.increase(30);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 80);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);

  // Increase to 120 - peak should update to 120
  stat.increase(40);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 120);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 120);
}

TEST_F(NeuronMetricsTest, MemoryStatResetPeak) {
  MemoryStatistics stat;
  stat.increase(100);
  stat.decrease(60);

  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 40);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 100);

  stat.reset_peak();
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 40);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 40);

  // After reset, new peak should track from current
  stat.increase(30);
  EXPECT_EQ(stat.current.load(std::memory_order_relaxed), 70);
  EXPECT_EQ(stat.peak.load(std::memory_order_relaxed), 70);
}

// ============================================================================
// DeviceMemoryStatsRegistry Tests
// ============================================================================

TEST_F(NeuronMetricsTest, RegistryGetDeviceStatsLazyInit) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();

  // Getting stats for a device should initialize it
  auto& stats = registry.GetDeviceStats(0);

  // Initial values should be zero
  EXPECT_EQ(stats.allocated_bytes.current.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.reserved_bytes.current.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.active_bytes.current.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_alloc_retries.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_ooms.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_tensor_frees.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.allocation_requests.load(std::memory_order_relaxed), 0);
}

TEST_F(NeuronMetricsTest, RegistryGetMemoryStatsReturnsCorrectValues) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();

  // Modify stats directly
  auto& stats = registry.GetDeviceStats(0);
  stats.allocated_bytes.increase(1000);
  stats.active_bytes.increase(1000);
  stats.num_alloc_retries.store(2, std::memory_order_relaxed);
  stats.num_ooms.store(1, std::memory_order_relaxed);
  stats.num_tensor_frees.store(5, std::memory_order_relaxed);
  stats.allocation_requests.store(10, std::memory_order_relaxed);

  // Get a copy via GetMemoryStats
  auto info = registry.GetMemoryStats(0);

  EXPECT_EQ(info.allocated_bytes.current, 1000);
  EXPECT_EQ(info.allocated_bytes.peak, 1000);
  EXPECT_EQ(info.reserved_bytes.current, 0);  // No caching, always 0
  EXPECT_EQ(info.active_bytes.current, 1000);
  EXPECT_EQ(info.num_alloc_retries, 2);
  EXPECT_EQ(info.num_ooms, 1);
  EXPECT_EQ(info.num_tensor_frees, 5);
  EXPECT_EQ(info.allocation_requests, 10);
}

TEST_F(NeuronMetricsTest, RegistryResetPeakStats) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();
  auto& stats = registry.GetDeviceStats(0);

  // Set up some values with peaks
  stats.allocated_bytes.increase(1000);
  stats.allocated_bytes.decrease(400);
  stats.active_bytes.increase(1000);
  stats.active_bytes.decrease(400);

  EXPECT_EQ(stats.allocated_bytes.current.load(std::memory_order_relaxed), 600);
  EXPECT_EQ(stats.allocated_bytes.peak.load(std::memory_order_relaxed), 1000);
  EXPECT_EQ(stats.active_bytes.current.load(std::memory_order_relaxed), 600);
  EXPECT_EQ(stats.active_bytes.peak.load(std::memory_order_relaxed), 1000);

  // Reset peaks
  registry.ResetPeakStats(0);

  EXPECT_EQ(stats.allocated_bytes.current.load(std::memory_order_relaxed), 600);
  EXPECT_EQ(stats.allocated_bytes.peak.load(std::memory_order_relaxed), 600);
  EXPECT_EQ(stats.active_bytes.current.load(std::memory_order_relaxed), 600);
  EXPECT_EQ(stats.active_bytes.peak.load(std::memory_order_relaxed), 600);
}

TEST_F(NeuronMetricsTest, RegistryResetAccumulatedStats) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();
  auto& stats = registry.GetDeviceStats(0);

  stats.num_alloc_retries.store(5, std::memory_order_relaxed);
  stats.num_ooms.store(3, std::memory_order_relaxed);
  stats.num_tensor_frees.store(10, std::memory_order_relaxed);
  stats.allocation_requests.store(50, std::memory_order_relaxed);

  registry.ResetAccumulatedStats(0);

  EXPECT_EQ(stats.num_alloc_retries.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_ooms.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_tensor_frees.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.allocation_requests.load(std::memory_order_relaxed), 0);
}

TEST_F(NeuronMetricsTest, RegistryMultipleDevices) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();

  // Get stats for multiple devices
  auto& stats0 = registry.GetDeviceStats(0);
  auto& stats1 = registry.GetDeviceStats(1);

  stats0.allocated_bytes.increase(100);
  stats1.allocated_bytes.increase(200);

  auto info0 = registry.GetMemoryStats(0);
  auto info1 = registry.GetMemoryStats(1);

  EXPECT_EQ(info0.allocated_bytes.current, 100);
  EXPECT_EQ(info1.allocated_bytes.current, 200);
}

TEST_F(NeuronMetricsTest, RegistryUninitializedDeviceReturnsZeros) {
  auto& registry = DeviceMemoryStatsRegistry::Instance();

  // Get stats for a device that hasn't been initialized
  // (using a high device index that likely hasn't been used)
  auto info = registry.GetMemoryStats(99);

  EXPECT_EQ(info.allocated_bytes.current, 0);
  EXPECT_EQ(info.allocated_bytes.peak, 0);
  EXPECT_EQ(info.reserved_bytes.current, 0);
  EXPECT_EQ(info.active_bytes.current, 0);
  EXPECT_EQ(info.num_alloc_retries, 0);
  EXPECT_EQ(info.num_ooms, 0);
}

TEST_F(NeuronMetricsTest, DeviceMemoryStatsResetPeak) {
  DeviceMemoryStats stats;

  stats.allocated_bytes.increase(1000);
  stats.active_bytes.increase(1000);

  stats.allocated_bytes.decrease(200);
  stats.active_bytes.decrease(200);

  EXPECT_EQ(stats.allocated_bytes.peak.load(std::memory_order_relaxed), 1000);
  EXPECT_EQ(stats.active_bytes.peak.load(std::memory_order_relaxed), 1000);

  stats.reset_peak();

  EXPECT_EQ(stats.allocated_bytes.peak.load(std::memory_order_relaxed), 800);
  EXPECT_EQ(stats.active_bytes.peak.load(std::memory_order_relaxed), 800);
}

TEST_F(NeuronMetricsTest, DeviceMemoryStatsResetCounters) {
  DeviceMemoryStats stats;

  stats.num_alloc_retries.store(10, std::memory_order_relaxed);
  stats.num_ooms.store(5, std::memory_order_relaxed);
  stats.num_tensor_frees.store(20, std::memory_order_relaxed);
  stats.allocation_requests.store(100, std::memory_order_relaxed);

  stats.reset_counters();

  EXPECT_EQ(stats.num_alloc_retries.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_ooms.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.num_tensor_frees.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.allocation_requests.load(std::memory_order_relaxed), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
