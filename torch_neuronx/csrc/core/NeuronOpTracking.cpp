#include "NeuronOpTracking.h"

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

std::unordered_set<std::string> getFallbackOps() {
  return FallbackOpTracker::getInstance().get_tracked_ops();
}

std::unordered_set<std::string> getExecutedOps() {
  return ExecutedOpTracker::getInstance().get_tracked_ops();
}

void clearOpTracking() {
  FallbackOpTracker::getInstance().clear();
  ExecutedOpTracker::getInstance().clear();
}

}  // namespace torch_neuronx
