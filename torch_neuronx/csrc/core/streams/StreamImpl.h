#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/utils/ErrorHandler.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

// Include NRT headers for NRT_STATUS type
extern "C" {
#include <nrt/nrt_status.h>
}

namespace at::neuron {

// Forward declarations
class NeuronEvent;
class ModelHandleCache;
class OperationExecutionEngine;

// Stream-level event state
struct EventState {
  NeuronEvent event;

  enum EventType {
    BLOCKING,   // Blocks next operations
    COMPLETING  // Will be completed by an operation
  };

  EventType type;
  OperationContext* completing_operation;  // Only applicable for COMPLETING events

  EventState(const NeuronEvent& e, EventType t, OperationContext* op = nullptr)
      : event(e), type(t), completing_operation(op) {}

  bool can_be_removed() const {
    return event.query() && (type == BLOCKING || completing_operation == nullptr);
  }
};

// StreamImpl provides the core implementation for Neuron stream execution.
// This class implements a multi-threaded pipeline for async operation execution:
// - Build IR thread: Handles JAX lowering and NEFF compilation
// - Execution thread: Handles NRT async execution
// - Unified operation tracking: Manages operation lifecycle and cleanup
struct StreamImpl : public std::enable_shared_from_this<StreamImpl> {
  friend class OperationExecutionEngine;

  // Stream identification
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
  int priority;

  // Operation tracking
  std::list<std::shared_ptr<OperationContext>> active_operations_;
  std::mutex active_operations_mutex_;

  // Stream-level event tracking
  std::vector<EventState> tracked_events_;

  // Event state specific mutex
  mutable std::mutex event_state_mutex_;

  // Shared resources accessed via NeuronResourceManager
  OperationExecutionEngine* operation_execution_engine_;

  // Per-stream error handling
  std::unique_ptr<ErrorHandler> error_handler_;

  // Resource tracking and cleanup
  std::atomic<size_t> total_operations_submitted_{0};
  std::atomic<size_t> total_operations_completed_{0};
  std::atomic<size_t> total_operations_failed_{0};
  std::chrono::steady_clock::time_point creation_time_;

  // Cleanup timeout configuration
  std::chrono::milliseconds operation_timeout_;

  // States
  std::atomic<bool> shutdown_requested{false};

  // Completion tracking for synchronization
  std::condition_variable completion_cv;

  StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id, int prio,
             OperationExecutionEngine* execution_manager);
  ~StreamImpl();

  // Main interface methods
  std::future<OperationContextResult> SubmitOperationContext(OperationContext operation_context);

  bool Query() const;
  void Synchronize();

  // Event management
  void RegisterEventWithLastOperation(NeuronEvent* event);
  void RegisterEventDependencyForNextOperation(const NeuronEvent& event);

  void CompleteOperation(OperationContext* operation);

  c10::StreamId GetStreamId() const { return stream_id; }

  // Resource statistics
  struct StreamResourceStats {
    c10::StreamId stream_id;
    size_t total_submitted;
    size_t total_completed;
    size_t total_failed;
    size_t pending_operations;
    size_t compilation_queue_size;
    size_t execution_queue_size;
    std::chrono::seconds uptime;
    bool is_idle;
  };

  StreamResourceStats GetResourceStats() const;

 private:
  // Initialization helpers
  void TrackMemoryAllocation();

  // Operation management
  void RemoveActiveOperation(const OperationContext* operation);

  // Handle the respective error for the current operation
  void HandleErrorWithCleanup(OperationContext* operation, const std::string& error_message,
                              ErrorContext::Stage stage, int error_code = -1);

  // Ensure all prior operations preceding the given one are completed.
  void WaitForPriorOperationsToComplete(const OperationContext* failed_operation);

  // Event management helpers
  void CompleteEventsForOperation(OperationContext* operation);

  // Result copying helper
  void CopyResultsToOriginalTensors(OperationContext* operation);
};

}  // namespace at::neuron
