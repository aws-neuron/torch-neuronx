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

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/utils/ErrorHandler.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
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

  // Shared resources accessed via NeuronResourceManager
  OperationExecutionEngine* operation_execution_engine_;

  // Per-stream error handling
  std::unique_ptr<ErrorHandler> error_handler_;

  // Cleanup timeout configuration
  std::chrono::milliseconds operation_timeout_;

  // Completion tracking for synchronization
  std::condition_variable completion_cv;

  StreamImpl(c10::DeviceIndex device_idx, c10::StreamId id, int prio,
             OperationExecutionEngine* execution_manager);
  ~StreamImpl();

  // Main interface methods
  std::future<OperationContextResult> SubmitOperationContext(
      std::unique_ptr<OperationContext> operation_context);

  bool Query() const;
  void Synchronize();

  void CompleteOperation(OperationContext* operation);

  // Ensure all prior operations preceding the given one are completed.
  void WaitForPriorOperationsToComplete(const OperationContext* operation);

  c10::StreamId GetStreamId() const { return stream_id; }

  // For execution worker polling
  OperationContext* GetNextReadyOperation();

 private:
  // Operation management
  void RemoveActiveOperation(const OperationContext* operation);

  // Handle the respective error for the current operation
  template <typename ExceptionType>
  void HandleErrorWithCleanup(OperationContext* operation, const ExceptionType& e);
};

}  // namespace at::neuron
