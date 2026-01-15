/*
 * LICENSE -
 * Last updated December 19, 2025
 *
 * ** PyTorch XLA; version 2.9 -- https://github.com/pytorch/xla
 * Copyright (c) 2018 Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Metrics infrastructure for torch-neuronx.
// Design inspired by PyTorch/XLA's metrics system (torch_xla/csrc/runtime/metrics.h).
// Adapted for Neuron with runtime enable/disable support and simplified API.

#ifndef TORCH_NEURON_CSRC_CORE_METRICS_NEURON_METRICS_H_
#define TORCH_NEURON_CSRC_CORE_METRICS_NEURON_METRICS_H_

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace at::neuron::metrics {

// Sample represents a timestamped value
struct Sample {
  Sample() = default;
  Sample(int64_t timestamp_ns, double value) : timestamp_ns(timestamp_ns), value(value) {}

  int64_t timestamp_ns = 0;
  double value = 0;
};

// Function type for formatting metric values
using MetricReprFn = std::function<std::string(double)>;

// Runtime metrics control
// Check if metrics collection is enabled (can be toggled at runtime)
bool IsMetricsEnabled();

// Representation functions (forward declarations)
std::string MetricFnValue(double value);  // Plain numeric value
std::string MetricFnBytes(double value);  // Human-readable bytes (KB, MB, etc.)
std::string MetricFnTime(
    double value);  // Human-readable time - IMPORTANT: expects value in NANOSECONDS

// NeuronMetricData: Thread-safe storage for time-series metric data
class NeuronMetricData {
 public:
  explicit NeuronMetricData(MetricReprFn repr_fn, size_t max_samples = 1024);

  // Add a new sample (thread-safe)
  void AddSample(int64_t timestamp_ns, double value);

  // Get current accumulator value (sum of all samples)
  double Accumulator() const;

  // Get total number of samples recorded
  size_t TotalSamples() const;

  // Get all current samples (thread-safe copy)
  std::vector<Sample> Samples(double* accumulator = nullptr, size_t* total_samples = nullptr) const;

  // Format a value using the representation function
  std::string Repr(double value) const { return repr_fn_(value); }

  // Clear all data
  void Clear();

 private:
  mutable std::mutex lock_;
  MetricReprFn repr_fn_;
  size_t count_ = 0;
  std::vector<Sample> samples_;  // Circular buffer
  double accumulator_ = 0.0;
};

// NeuronCounterData: Thread-safe atomic counter
class NeuronCounterData {
 public:
  NeuronCounterData() : value_(0) {}

  // Add value to counter (thread-safe, lock-free)
  void AddValue(int64_t value) { value_.fetch_add(value, std::memory_order_relaxed); }

  // Get current counter value (thread-safe, lock-free)
  int64_t Value() const { return value_.load(std::memory_order_relaxed); }

  // Reset counter to zero
  void Clear() { value_.store(0, std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> value_;
};

// NeuronMetricsArena: Global registry for all metrics and counters
class NeuronMetricsArena {
 public:
  static NeuronMetricsArena* Get();

  // Register a new metric (thread-safe)
  void RegisterMetric(const std::string& name, MetricReprFn repr_fn, size_t max_samples,
                      std::shared_ptr<NeuronMetricData>* data);

  // Register a new counter (thread-safe)
  void RegisterCounter(const std::string& name, std::shared_ptr<NeuronCounterData>* data);

  // Iterate over all metrics (thread-safe)
  void ForEachMetric(const std::function<void(const std::string&, NeuronMetricData*)>& metric_func);

  // Iterate over all counters (thread-safe)
  void ForEachCounter(
      const std::function<void(const std::string&, NeuronCounterData*)>& counter_func);

  // Get metric names that have data
  std::vector<std::string> GetMetricNames();

  // Get specific metric by name
  NeuronMetricData* GetMetric(const std::string& name);

  // Get counter names that have data
  std::vector<std::string> GetCounterNames();

  // Get specific counter by name
  NeuronCounterData* GetCounter(const std::string& name);

  // Clear all counters
  void ClearCounters();

  // Clear all metrics
  void ClearMetrics();

 private:
  NeuronMetricsArena() = default;

  std::mutex lock_;
  std::map<std::string, std::shared_ptr<NeuronMetricData>> metrics_;
  std::map<std::string, std::shared_ptr<NeuronCounterData>> counters_;
};

// NeuronMetric: High-level interface for recording metric samples
class NeuronMetric {
 public:
  explicit NeuronMetric(std::string name, MetricReprFn repr_fn = MetricFnValue,
                        size_t max_samples = 1024);

  const std::string& Name() const { return name_; }

  // Add sample with explicit timestamp
  void AddSample(int64_t timestamp_ns, double value);

  // Add sample with current timestamp
  void AddSample(double value);

  // Get current accumulator value
  double Accumulator() const;

  // Get all samples
  std::vector<Sample> Samples(double* accumulator = nullptr, size_t* total_samples = nullptr) const;

  // Format value
  std::string Repr(double value) const;

  // Clear/reset the metric (clears all samples and accumulator)
  void Clear() { GetData()->Clear(); }

 private:
  NeuronMetricData* GetData() const;

  std::string name_;
  MetricReprFn repr_fn_;
  size_t max_samples_;
  mutable std::shared_ptr<NeuronMetricData> data_ptr_;
  mutable std::once_flag init_flag_;
};

// NeuronCounter: High-level interface for counting operations
class NeuronCounter {
 public:
  explicit NeuronCounter(std::string name);

  const std::string& Name() const { return name_; }

  // Add value to counter (only if metrics are enabled)
  void AddValue(int64_t value) {
    if (IsMetricsEnabled()) {
      GetData()->AddValue(value);
    }
  }

  // Get current counter value
  int64_t Value() const { return GetData()->Value(); }

  // Clear/reset the counter to zero
  void Clear() { GetData()->Clear(); }

 private:
  NeuronCounterData* GetData() const;

  std::string name_;
  mutable std::shared_ptr<NeuronCounterData> data_ptr_;
  mutable std::once_flag init_flag_;
};

// Utility class for automatic timing measurement
class TimedSection {
 public:
  explicit TimedSection(NeuronMetric* metric);
  ~TimedSection();

  // Get elapsed time in seconds
  double Elapsed() const;

 private:
  NeuronMetric* metric_;
  int64_t start_time_ns_;
};

// Utility functions
int64_t GetCurrentTimeNs();

// Reporting functions
std::string CreateMetricReport();
std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names);

// =============================================================================
// Runtime metrics control
// Metrics can be enabled/disabled at runtime via environment variable or API.
// Default: disabled. Set TORCH_NEURONX_METRICS_ENABLED=1 to enable.
// =============================================================================

// Check if metrics collection is enabled (can be toggled at runtime)
bool IsMetricsEnabled();

// Enable or disable metrics collection at runtime
void SetMetricsEnabled(bool enabled);

// Convenience macros for easy usage
#define TORCH_NEURONX_COUNTER(name, value)                     \
  do {                                                         \
    if (::at::neuron::metrics::IsMetricsEnabled()) {           \
      static ::at::neuron::metrics::NeuronCounter* __counter = \
          new ::at::neuron::metrics::NeuronCounter(name);      \
      __counter->AddValue(value);                              \
    }                                                          \
  } while (0)

#define TORCH_NEURONX_METRIC(name, value)                                                      \
  do {                                                                                         \
    if (::at::neuron::metrics::IsMetricsEnabled()) {                                           \
      static ::at::neuron::metrics::NeuronMetric* __metric =                                   \
          new ::at::neuron::metrics::NeuronMetric(name, ::at::neuron::metrics::MetricFnValue); \
      __metric->AddSample(value);                                                              \
    }                                                                                          \
  } while (0)

#define TORCH_NEURONX_BYTES_METRIC(name, value)                                                \
  do {                                                                                         \
    if (::at::neuron::metrics::IsMetricsEnabled()) {                                           \
      static ::at::neuron::metrics::NeuronMetric* __metric =                                   \
          new ::at::neuron::metrics::NeuronMetric(name, ::at::neuron::metrics::MetricFnBytes); \
      __metric->AddSample(value);                                                              \
    }                                                                                          \
  } while (0)

#define TORCH_NEURONX_TIME_METRIC(name, start_time, end_time)                                      \
  do {                                                                                             \
    if (::at::neuron::metrics::IsMetricsEnabled()) {                                               \
      static ::at::neuron::metrics::NeuronMetric* __metric =                                       \
          new ::at::neuron::metrics::NeuronMetric(name, ::at::neuron::metrics::MetricFnTime);      \
      auto __duration_ns =                                                                         \
          std::chrono::duration_cast<std::chrono::nanoseconds>((end_time) - (start_time)).count(); \
      __metric->AddSample(__duration_ns);                                                          \
    }                                                                                              \
  } while (0)

#define TORCH_NEURONX_TIMED_SECTION(name)                                                      \
  static ::at::neuron::metrics::NeuronMetric* __timed_metric =                                 \
      ::at::neuron::metrics::IsMetricsEnabled()                                                \
          ? new ::at::neuron::metrics::NeuronMetric(name, ::at::neuron::metrics::MetricFnTime) \
          : nullptr;                                                                           \
  ::at::neuron::metrics::TimedSection __timed_section(__timed_metric)

}  // namespace at::neuron::metrics

#endif  // TORCH_NEURON_CSRC_CORE_METRICS_NEURON_METRICS_H_
