#ifndef TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H
#define TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H

#include <mutex>
#include <string>
#include <unordered_set>

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

// Helper functions for cleaner API
bool shouldLogFallback(const std::string& op_name);
void markFallbackLogged(const std::string& op_name);
bool shouldLogExecuted(const std::string& op_name);
void markExecutedLogged(const std::string& op_name);

// Helper function to log operation execution once
void logOperationExecuted(const std::string& op_name);

// Functions to get tracked operations for testing
std::unordered_set<std::string> getFallbackOps();
std::unordered_set<std::string> getExecutedOps();
void clearOpTracking();  // Clear both fallback and executed ops

}  // namespace torch_neuronx

#endif  // TORCH_NEURONX_CSRC_CORE_NEURON_OP_TRACKING_H
