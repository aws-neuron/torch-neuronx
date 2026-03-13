#ifndef TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H
#define TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch_neuronx {

// Base class for deduplication tracking of operations
class OpTrackerBase {
 protected:
  std::unordered_set<std::string> tracked_ops;
  mutable std::mutex tracker_mutex;

 public:
  // Check if an operation has been tracked
  bool has_been_tracked(const std::string& op_name) const;

  // Mark an operation as tracked
  void mark_as_tracked(const std::string& op_name);

  // Get all tracked operations
  std::unordered_set<std::string> get_tracked_ops() const;

  // Clear all tracked operations
  void clear();
};

// Singleton tracker for CPU fallback operations
class FallbackOpTracker : public OpTrackerBase {
 public:
  static FallbackOpTracker& getInstance();

 private:
  FallbackOpTracker() = default;
  FallbackOpTracker(const FallbackOpTracker&) = delete;
  FallbackOpTracker& operator=(const FallbackOpTracker&) = delete;
};

// Singleton tracker for executed operations
class ExecutedOpTracker : public OpTrackerBase {
 public:
  static ExecutedOpTracker& getInstance();

 private:
  ExecutedOpTracker() = default;
  ExecutedOpTracker(const ExecutedOpTracker&) = delete;
  ExecutedOpTracker& operator=(const ExecutedOpTracker&) = delete;
};

// Singleton tracker for concatenated operations.
// Maps concatenated op names to their constituent original op names.
// Does NOT inherit from OpTrackerBase because the data model is different:
// OpTrackerBase uses an unordered_set<string> for deduplication tracking,
// while this tracker uses an unordered_map<string, vector<string>> to preserve
// the mapping between concatenated groups and their original ops, plus
// per-op execution counts for distinguishing first-time vs cached executions.
class ConcatenatedOpTracker {
 public:
  static ConcatenatedOpTracker& getInstance();

  // Record a concatenated op and its original constituent ops.
  // Uses try_emplace to avoid redundant copies in steady-state (same group
  // executed repeatedly). Also increments the execution count for this op.
  void record(const std::string& concat_op_name, const std::vector<std::string>& original_ops);

  // Get the full map of concatenated ops -> original ops
  std::unordered_map<std::string, std::vector<std::string>> get_concatenated_ops() const;

  // Get per-op execution counts (how many times each concatenated op was executed).
  // First execution is always uncached (compilation required); subsequent executions
  // are cache hits. Count of 1 = uncached, count > 1 = (count - 1) cache hits.
  std::unordered_map<std::string, int64_t> get_execution_counts() const;

  // Clear all tracked concatenated operations and execution counts
  void clear();

 private:
  ConcatenatedOpTracker() = default;
  ConcatenatedOpTracker(const ConcatenatedOpTracker&) = delete;
  ConcatenatedOpTracker& operator=(const ConcatenatedOpTracker&) = delete;

  mutable std::mutex tracker_mutex_;
  std::unordered_map<std::string, std::vector<std::string>> concatenated_ops_;
  std::unordered_map<std::string, int64_t> execution_counts_;
};

// Per-concatenation-type profiling data
struct ConcatTypeProfile {
  std::vector<double> merge_times_ns;  // IR merge time per invocation (nanoseconds)
  std::vector<int64_t> group_sizes;    // Number of ops per concat group
  int64_t invocation_count = 0;        // Total invocations of this concat type
};

// Singleton profiler for per-concatenation-type metrics.
// Stores detailed timing and size data keyed by concatenated op name,
// enabling analysis like "matmul|add takes 5ms avg to merge" vs
// "linear|dropout|norm takes 12ms avg to merge".
class ConcatenationProfiler {
 public:
  static ConcatenationProfiler& getInstance();

  // Enable/disable detailed per-type profiling.
  // Default: disabled. Enable via set_enabled(true) or
  // TORCH_NEURONX_CONCATENATION_PROFILING=1 env var.
  // This is separate from TORCH_NEURONX_METRICS_ENABLED because per-type
  // profiling stores unbounded per-invocation vectors (memory grows with
  // unique concat types and invocations), unlike the fixed-size metrics.
  bool is_enabled() const;
  void set_enabled(bool enabled);

  // Record merge time and group size for a specific concatenation type.
  // Called from ConcatenationCore::ProcessBufferedOperationsImpl.
  // No-op if profiling is not enabled.
  void record(const std::string& concat_op_name, double merge_time_ns, int64_t group_size);

  // Get all per-type profiles
  std::unordered_map<std::string, ConcatTypeProfile> get_profiles() const;

  // Clear all profiling data
  void clear();

 private:
  ConcatenationProfiler();
  ConcatenationProfiler(const ConcatenationProfiler&) = delete;
  ConcatenationProfiler& operator=(const ConcatenationProfiler&) = delete;

  mutable std::mutex profiler_mutex_;
  std::unordered_map<std::string, ConcatTypeProfile> profiles_;
};

// Helper to record per-type profiling data
void logConcatenationProfile(const std::string& concat_op_name, double merge_time_ns,
                             int64_t group_size);

// Get all per-type concatenation profiles
std::unordered_map<std::string, ConcatTypeProfile> getConcatenationProfiles();

// Helper functions for cleaner API
bool shouldLogFallback(const std::string& op_name);
void markFallbackLogged(const std::string& op_name);
bool shouldLogExecuted(const std::string& op_name);
void markExecutedLogged(const std::string& op_name);

// Helper function to log operation execution once
void logOperationExecuted(const std::string& op_name);

// Log a concatenated op execution with its original ops
void logConcatenatedOpExecuted(const std::string& concat_op_name,
                               const std::vector<std::string>& original_op_names);

// Functions to get tracked operations for testing
std::unordered_set<std::string> getFallbackOps();
std::unordered_set<std::string> getExecutedOps();
std::unordered_map<std::string, std::vector<std::string>> getConcatenatedOps();
std::unordered_map<std::string, int64_t> getConcatenatedOpExecutionCounts();
void clearOpTracking();  // Clear fallback, executed, and concatenated ops

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H
