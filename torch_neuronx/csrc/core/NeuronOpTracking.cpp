#include "NeuronOpTracking.h"

#include <atomic>
#include <cstdlib>

namespace torch_neuronx {

// OpTrackerBase implementation
bool OpTrackerBase::has_been_tracked(const std::string& op_name) const {
  std::lock_guard<std::mutex> lock(tracker_mutex);
  return tracked_ops.find(op_name) != tracked_ops.end();
}

void OpTrackerBase::mark_as_tracked(const std::string& op_name) {
  std::lock_guard<std::mutex> lock(tracker_mutex);
  tracked_ops.insert(op_name);
}

std::unordered_set<std::string> OpTrackerBase::get_tracked_ops() const {
  std::lock_guard<std::mutex> lock(tracker_mutex);
  return tracked_ops;
}

void OpTrackerBase::clear() {
  std::lock_guard<std::mutex> lock(tracker_mutex);
  tracked_ops.clear();
}

// FallbackOpTracker singleton implementation
FallbackOpTracker& FallbackOpTracker::getInstance() {
  static FallbackOpTracker instance;
  return instance;
}

// ExecutedOpTracker singleton implementation
ExecutedOpTracker& ExecutedOpTracker::getInstance() {
  static ExecutedOpTracker instance;
  return instance;
}

// Helper functions implementation
bool shouldLogFallback(const std::string& op_name) {
  return !FallbackOpTracker::getInstance().has_been_tracked(op_name);
}

void markFallbackLogged(const std::string& op_name) {
  FallbackOpTracker::getInstance().mark_as_tracked(op_name);
}

bool shouldLogExecuted(const std::string& op_name) {
  return !ExecutedOpTracker::getInstance().has_been_tracked(op_name);
}

void markExecutedLogged(const std::string& op_name) {
  ExecutedOpTracker::getInstance().mark_as_tracked(op_name);
}

void logOperationExecuted(const std::string& op_name) {
  if (shouldLogExecuted(op_name)) {
    markExecutedLogged(op_name);
  }
}

// ConcatenatedOpTracker singleton implementation
ConcatenatedOpTracker& ConcatenatedOpTracker::getInstance() {
  static ConcatenatedOpTracker instance;
  return instance;
}

void ConcatenatedOpTracker::record(const std::string& concat_op_name,
                                   const std::vector<std::string>& original_ops) {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  // Only insert on first occurrence to avoid redundant vector copies in
  // steady-state (same concatenated group executed many times; constituent
  // ops don't change).
  if (concatenated_ops_.find(concat_op_name) == concatenated_ops_.end()) {
    concatenated_ops_.emplace(concat_op_name, original_ops);
  }
  // Always increment execution count (tracks first-time vs cached executions)
  execution_counts_[concat_op_name]++;
}

std::unordered_map<std::string, std::vector<std::string>>
ConcatenatedOpTracker::get_concatenated_ops() const {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  return concatenated_ops_;
}

std::unordered_map<std::string, int64_t> ConcatenatedOpTracker::get_execution_counts() const {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  return execution_counts_;
}

void ConcatenatedOpTracker::clear() {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  concatenated_ops_.clear();
  execution_counts_.clear();
}

void logConcatenatedOpExecuted(const std::string& concat_op_name,
                               const std::vector<std::string>& original_op_names) {
  ConcatenatedOpTracker::getInstance().record(concat_op_name, original_op_names);
}

// ConcatenationProfiler singleton implementation
// Default disabled. Enable via TORCH_NEURONX_CONCATENATION_PROFILING=1 or set_enabled(true).
static std::atomic<bool> g_concatenation_profiling_enabled{false};

ConcatenationProfiler::ConcatenationProfiler() {
  // Check env var on construction
  const char* env = std::getenv("TORCH_NEURONX_CONCATENATION_PROFILING");
  if (env && std::string(env) == "1") {
    g_concatenation_profiling_enabled.store(true, std::memory_order_relaxed);
  }
}

ConcatenationProfiler& ConcatenationProfiler::getInstance() {
  static ConcatenationProfiler instance;
  return instance;
}

bool ConcatenationProfiler::is_enabled() const {
  return g_concatenation_profiling_enabled.load(std::memory_order_relaxed);
}

void ConcatenationProfiler::set_enabled(bool enabled) {
  g_concatenation_profiling_enabled.store(enabled, std::memory_order_relaxed);
}

void ConcatenationProfiler::record(const std::string& concat_op_name, double merge_time_ns,
                                   int64_t group_size) {
  if (!is_enabled()) return;
  std::lock_guard<std::mutex> lock(profiler_mutex_);
  auto& profile = profiles_[concat_op_name];
  profile.merge_times_ns.push_back(merge_time_ns);
  profile.group_sizes.push_back(group_size);
  profile.invocation_count++;
}

std::unordered_map<std::string, ConcatTypeProfile> ConcatenationProfiler::get_profiles() const {
  std::lock_guard<std::mutex> lock(profiler_mutex_);
  return profiles_;
}

void ConcatenationProfiler::clear() {
  std::lock_guard<std::mutex> lock(profiler_mutex_);
  profiles_.clear();
}

void logConcatenationProfile(const std::string& concat_op_name, double merge_time_ns,
                             int64_t group_size) {
  ConcatenationProfiler::getInstance().record(concat_op_name, merge_time_ns, group_size);
}

std::unordered_map<std::string, ConcatTypeProfile> getConcatenationProfiles() {
  return ConcatenationProfiler::getInstance().get_profiles();
}

std::unordered_set<std::string> getFallbackOps() {
  return FallbackOpTracker::getInstance().get_tracked_ops();
}

std::unordered_set<std::string> getExecutedOps() {
  return ExecutedOpTracker::getInstance().get_tracked_ops();
}

std::unordered_map<std::string, std::vector<std::string>> getConcatenatedOps() {
  return ConcatenatedOpTracker::getInstance().get_concatenated_ops();
}

std::unordered_map<std::string, int64_t> getConcatenatedOpExecutionCounts() {
  return ConcatenatedOpTracker::getInstance().get_execution_counts();
}

void clearOpTracking() {
  FallbackOpTracker::getInstance().clear();
  ExecutedOpTracker::getInstance().clear();
  ConcatenatedOpTracker::getInstance().clear();
  ConcatenationProfiler::getInstance().clear();
}

}  // namespace torch_neuronx
