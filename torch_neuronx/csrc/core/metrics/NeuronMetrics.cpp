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
// Design inspired by PyTorch/XLA's metrics system (torch_xla/csrc/runtime/metrics.cc).
// Adapted for Neuron with runtime enable/disable support and simplified API.

#include "NeuronMetrics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace at::neuron::metrics {

// =============================================================================
// Runtime metrics enable/disable flag
// =============================================================================

namespace {

// Global flag for metrics enabled state
// Initialized from environment variable TORCH_NEURONX_METRICS_ENABLED (default: false)
std::atomic<bool> g_metrics_enabled{false};
std::once_flag g_metrics_init_flag;

void InitMetricsEnabledFromEnv() {
  const char* env_value = std::getenv("TORCH_NEURONX_METRICS_ENABLED");
  if (env_value != nullptr) {
    std::string value(env_value);
    // Enable if explicitly set to "1", "true", or "yes" (case-insensitive)
    if (value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES") {
      g_metrics_enabled.store(true, std::memory_order_relaxed);
    }
  }
}

}  // namespace

bool IsMetricsEnabled() {
  std::call_once(g_metrics_init_flag, InitMetricsEnabledFromEnv);
  return g_metrics_enabled.load(std::memory_order_relaxed);
}

void SetMetricsEnabled(bool enabled) {
  std::call_once(g_metrics_init_flag, InitMetricsEnabledFromEnv);
  g_metrics_enabled.store(enabled, std::memory_order_relaxed);
}

namespace {

// Helper function to emit metric information to stringstream
void EmitMetricInfo(const std::string& name, NeuronMetricData* data, std::stringstream* ss) {
  double accumulator = 0.0;
  size_t total_samples = 0;
  std::vector<Sample> samples = data->Samples(&accumulator, &total_samples);

  (*ss) << "Metric: " << name << std::endl;
  (*ss) << "  TotalSamples: " << total_samples << std::endl;
  (*ss) << "  Accumulator: " << data->Repr(accumulator) << std::endl;

  if (!samples.empty()) {
    double total = 0.0;
    for (const auto& sample : samples) {
      total += sample.value;
    }

    // Calculate basic statistics
    double mean = total / samples.size();
    (*ss) << "  Mean: " << data->Repr(mean) << std::endl;

    // Find min/max
    auto minmax =
        std::minmax_element(samples.begin(), samples.end(),
                            [](const Sample& a, const Sample& b) { return a.value < b.value; });
    (*ss) << "  Min: " << data->Repr(minmax.first->value) << std::endl;
    (*ss) << "  Max: " << data->Repr(minmax.second->value) << std::endl;
  }
}

// Helper function to emit counter information to stringstream
void EmitCounterInfo(const std::string& name, NeuronCounterData* data, std::stringstream* ss) {
  (*ss) << "Counter: " << name << std::endl;
  (*ss) << "  Value: " << data->Value() << std::endl;
}

}  // namespace

// NeuronMetricData implementation
NeuronMetricData::NeuronMetricData(MetricReprFn repr_fn, size_t max_samples)
    : repr_fn_(std::move(repr_fn)), samples_(max_samples) {}

void NeuronMetricData::AddSample(int64_t timestamp_ns, double value) {
  std::lock_guard<std::mutex> lock(lock_);
  size_t position = count_ % samples_.size();
  ++count_;
  accumulator_ += value;
  samples_[position] = Sample(timestamp_ns, value);
}

double NeuronMetricData::Accumulator() const {
  std::lock_guard<std::mutex> lock(lock_);
  return accumulator_;
}

size_t NeuronMetricData::TotalSamples() const {
  std::lock_guard<std::mutex> lock(lock_);
  return count_;
}

std::vector<Sample> NeuronMetricData::Samples(double* accumulator, size_t* total_samples) const {
  std::lock_guard<std::mutex> lock(lock_);

  std::vector<Sample> result;
  if (count_ <= samples_.size()) {
    result.insert(result.end(), samples_.begin(), samples_.begin() + count_);
  } else {
    size_t position = count_ % samples_.size();
    result.insert(result.end(), samples_.begin() + position, samples_.end());
    result.insert(result.end(), samples_.begin(), samples_.begin() + position);
  }

  if (accumulator != nullptr) {
    *accumulator = accumulator_;
  }
  if (total_samples != nullptr) {
    *total_samples = count_;
  }

  return result;
}

void NeuronMetricData::Clear() {
  std::lock_guard<std::mutex> lock(lock_);
  count_ = 0;
  accumulator_ = 0.0;
  samples_ = std::vector<Sample>(samples_.size());
}

// NeuronMetricsArena implementation
NeuronMetricsArena* NeuronMetricsArena::Get() {
  static NeuronMetricsArena* arena = new NeuronMetricsArena();
  return arena;
}

void NeuronMetricsArena::RegisterMetric(const std::string& name, MetricReprFn repr_fn,
                                        size_t max_samples,
                                        std::shared_ptr<NeuronMetricData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
      *data = it->second;
    } else {
      *data = std::make_shared<NeuronMetricData>(std::move(repr_fn), max_samples);
      metrics_[name] = *data;
    }
  }
}

void NeuronMetricsArena::RegisterCounter(const std::string& name,
                                         std::shared_ptr<NeuronCounterData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    auto it = counters_.find(name);
    if (it != counters_.end()) {
      *data = it->second;
    } else {
      *data = std::make_shared<NeuronCounterData>();
      counters_[name] = *data;
    }
  }
}

void NeuronMetricsArena::ForEachMetric(
    const std::function<void(const std::string&, NeuronMetricData*)>& metric_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (const auto& name_data : metrics_) {
    metric_func(name_data.first, name_data.second.get());
  }
}

void NeuronMetricsArena::ForEachCounter(
    const std::function<void(const std::string&, NeuronCounterData*)>& counter_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (const auto& name_data : counters_) {
    counter_func(name_data.first, name_data.second.get());
  }
}

std::vector<std::string> NeuronMetricsArena::GetMetricNames() {
  std::vector<std::string> names;
  std::lock_guard<std::mutex> lock(lock_);
  for (const auto& name_data : metrics_) {
    names.push_back(name_data.first);
  }
  return names;
}

NeuronMetricData* NeuronMetricsArena::GetMetric(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = metrics_.find(name);
  return (it != metrics_.end()) ? it->second.get() : nullptr;
}

std::vector<std::string> NeuronMetricsArena::GetCounterNames() {
  std::vector<std::string> names;
  std::lock_guard<std::mutex> lock(lock_);
  for (const auto& name_data : counters_) {
    names.push_back(name_data.first);
  }
  return names;
}

NeuronCounterData* NeuronMetricsArena::GetCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counters_.find(name);
  return (it != counters_.end()) ? it->second.get() : nullptr;
}

void NeuronMetricsArena::ClearCounters() {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& counter : counters_) {
    counter.second->Clear();
  }
}

void NeuronMetricsArena::ClearMetrics() {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& metric : metrics_) {
    metric.second->Clear();
  }
}

// NeuronMetric implementation
NeuronMetric::NeuronMetric(std::string name, MetricReprFn repr_fn, size_t max_samples)
    : name_(std::move(name)), repr_fn_(std::move(repr_fn)), max_samples_(max_samples) {}

void NeuronMetric::AddSample(int64_t timestamp_ns, double value) {
  if (IsMetricsEnabled()) {
    GetData()->AddSample(timestamp_ns, value);
  }
}

void NeuronMetric::AddSample(double value) {
  if (IsMetricsEnabled()) {
    GetData()->AddSample(GetCurrentTimeNs(), value);
  }
}

double NeuronMetric::Accumulator() const { return GetData()->Accumulator(); }

std::vector<Sample> NeuronMetric::Samples(double* accumulator, size_t* total_samples) const {
  return GetData()->Samples(accumulator, total_samples);
}

std::string NeuronMetric::Repr(double value) const { return GetData()->Repr(value); }

NeuronMetricData* NeuronMetric::GetData() const {
  std::call_once(init_flag_, [this]() {
    NeuronMetricsArena* arena = NeuronMetricsArena::Get();
    arena->RegisterMetric(name_, repr_fn_, max_samples_, &data_ptr_);
  });
  return data_ptr_.get();
}

// NeuronCounter implementation
NeuronCounter::NeuronCounter(std::string name) : name_(std::move(name)) {}

NeuronCounterData* NeuronCounter::GetData() const {
  std::call_once(init_flag_, [this]() {
    NeuronMetricsArena* arena = NeuronMetricsArena::Get();
    arena->RegisterCounter(name_, &data_ptr_);
  });
  return data_ptr_.get();
}

// TimedSection implementation
TimedSection::TimedSection(NeuronMetric* metric)
    : metric_(metric), start_time_ns_((metric && IsMetricsEnabled()) ? GetCurrentTimeNs() : 0) {}

TimedSection::~TimedSection() {
  if (metric_ && IsMetricsEnabled() && start_time_ns_ > 0) {
    int64_t end_time_ns = GetCurrentTimeNs();
    metric_->AddSample(end_time_ns, static_cast<double>(end_time_ns - start_time_ns_));
  }
}

double TimedSection::Elapsed() const {
  if (!metric_) return 0.0;
  return static_cast<double>(GetCurrentTimeNs() - start_time_ns_) * 1e-9;
}

// Representation functions
std::string MetricFnValue(double value) {
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value;
  return ss.str();
}

std::string MetricFnBytes(double value) {
  const int kNumSuffixes = 6;
  static const char* const kSizeSuffixes[kNumSuffixes] = {"B", "KB", "MB", "GB", "TB", "PB"};

  int suffix_idx = 0;
  for (; (suffix_idx + 1) < kNumSuffixes && value >= 1024.0; ++suffix_idx) {
    value /= 1024.0;
  }

  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value << kSizeSuffixes[suffix_idx];
  return ss.str();
}

std::string MetricFnTime(double value) {
  // Input value is in nanoseconds
  static struct TimePart {
    const char* suffix;
    double scaler;
    int width;
    int precision;
    char fill;
  } const time_parts[] = {
      {"d", 86400.0 * 1e9, 2, 0, '0'},  // days
      {"h", 3600.0 * 1e9, 2, 0, '0'},   // hours
      {"m", 60.0 * 1e9, 2, 0, '0'},     // minutes
      {"s", 1e9, 2, 0, '0'},            // seconds
      {"ms", 1e6, 3, 0, '0'},           // milliseconds
      {"us", 1e3, 7, 3, '0'},           // microseconds
  };

  int count = 0;
  std::stringstream ss;
  for (size_t i = 0; i < sizeof(time_parts) / sizeof(time_parts[0]); ++i) {
    const TimePart& part = time_parts[i];
    double ctime = value / part.scaler;
    if (ctime >= 1.0 || count > 0 || i + 1 == sizeof(time_parts) / sizeof(time_parts[0])) {
      ss.precision(part.precision);
      ss.width(part.width);
      ss.fill(part.fill);
      ss << std::fixed << ctime << part.suffix;
      value -= std::floor(ctime) * part.scaler;
      ++count;
    }
  }
  return ss.str();
}

// Utility functions
int64_t GetCurrentTimeNs() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

// Reporting functions
std::string CreateMetricReport() {
  NeuronMetricsArena* arena = NeuronMetricsArena::Get();
  std::stringstream ss;

  ss << "=== Neuron Metrics Report ===" << std::endl;

  arena->ForEachMetric([&ss](const std::string& name, NeuronMetricData* data) {
    if (data->TotalSamples() > 0) {
      EmitMetricInfo(name, data, &ss);
      ss << std::endl;
    }
  });

  arena->ForEachCounter([&ss](const std::string& name, NeuronCounterData* data) {
    if (data->Value() > 0) {
      EmitCounterInfo(name, data, &ss);
      ss << std::endl;
    }
  });

  return ss.str();
}

std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names) {
  NeuronMetricsArena* arena = NeuronMetricsArena::Get();
  std::stringstream ss;

  ss << "=== Neuron Metrics Report (Filtered) ===" << std::endl;

  for (const std::string& metric_name : metric_names) {
    NeuronMetricData* data = arena->GetMetric(metric_name);
    if (data && data->TotalSamples() > 0) {
      EmitMetricInfo(metric_name, data, &ss);
      ss << std::endl;
    }
  }

  for (const std::string& counter_name : counter_names) {
    NeuronCounterData* data = arena->GetCounter(counter_name);
    if (data && data->Value() > 0) {
      EmitCounterInfo(counter_name, data, &ss);
      ss << std::endl;
    }
  }

  return ss.str();
}

// =============================================================================
// Memory Statistics Helper Functions implementation
// =============================================================================

void RecordNewAllocation(int device_id, size_t size_bytes) {
  auto& stats = DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device_id);
  stats.allocated_bytes.increase(size_bytes);
  stats.active_bytes.increase(size_bytes);
}

void RecordDeallocation(int device_id, size_t size_bytes) {
  auto& stats = DeviceMemoryStatsRegistry::Instance().GetDeviceStats(device_id);
  stats.num_tensor_frees.fetch_add(1, std::memory_order_relaxed);
  stats.active_bytes.decrease(size_bytes);
  stats.allocated_bytes.decrease(size_bytes);
}

// =============================================================================
// DeviceMemoryStatsRegistry implementation
// =============================================================================

DeviceMemoryStatsRegistry& DeviceMemoryStatsRegistry::Instance() {
  static DeviceMemoryStatsRegistry instance;
  return instance;
}

DeviceMemoryStats& DeviceMemoryStatsRegistry::GetDeviceStats(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Grow vector if needed
  if (device_index >= static_cast<int>(device_stats_.size())) {
    device_stats_.resize(device_index + 1);
  }

  // Create stats for device if not exists
  if (!device_stats_[device_index]) {
    device_stats_[device_index] = std::make_unique<DeviceMemoryStats>();
  }

  return *device_stats_[device_index];
}

DeviceMemoryStatsInfo DeviceMemoryStatsRegistry::GetMemoryStats(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Return zeros if device not initialized
  if (device_index >= static_cast<int>(device_stats_.size()) || !device_stats_[device_index]) {
    return DeviceMemoryStatsInfo{};
  }

  const auto& stats = *device_stats_[device_index];

  return DeviceMemoryStatsInfo{{stats.allocated_bytes.current.load(std::memory_order_relaxed),
                                stats.allocated_bytes.peak.load(std::memory_order_relaxed),
                                stats.allocated_bytes.allocated.load(std::memory_order_relaxed),
                                stats.allocated_bytes.freed.load(std::memory_order_relaxed)},
                               {stats.reserved_bytes.current.load(std::memory_order_relaxed),
                                stats.reserved_bytes.peak.load(std::memory_order_relaxed),
                                stats.reserved_bytes.allocated.load(std::memory_order_relaxed),
                                stats.reserved_bytes.freed.load(std::memory_order_relaxed)},
                               {stats.active_bytes.current.load(std::memory_order_relaxed),
                                stats.active_bytes.peak.load(std::memory_order_relaxed),
                                stats.active_bytes.allocated.load(std::memory_order_relaxed),
                                stats.active_bytes.freed.load(std::memory_order_relaxed)},
                               stats.num_alloc_retries.load(std::memory_order_relaxed),
                               stats.num_ooms.load(std::memory_order_relaxed),
                               stats.num_tensor_frees.load(std::memory_order_relaxed),
                               stats.allocation_requests.load(std::memory_order_relaxed)};
}

void DeviceMemoryStatsRegistry::ResetPeakStats(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_index < static_cast<int>(device_stats_.size()) && device_stats_[device_index]) {
    device_stats_[device_index]->reset_peak();
  }
}

void DeviceMemoryStatsRegistry::ResetAccumulatedStats(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_index < static_cast<int>(device_stats_.size()) && device_stats_[device_index]) {
    device_stats_[device_index]->reset_counters();
    device_stats_[device_index]->reset_accumulated();
  }
}

}  // namespace at::neuron::metrics
