#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace at::neuron {

// Forward declarations
struct StreamImpl;
class NeuronResourceManager;
class OperationExecutionEngine;

/**
 * StreamManager manages all Neuron streams across devices.
 * It provides centralized stream creation, lifecycle management, and resource coordination.
 *
 * Key responsibilities:
 * - Stream creation and destruction
 * - Device-wide synchronization
 * - Resource statistics and monitoring
 * - Shared resource management (compilation cache, NRT semaphore)
 *
 * Thread Safety: All public methods are thread-safe.
 */
class StreamManager {
 public:
  /// Constructor - takes OperationExecutionEngine to avoid mutex recursion
  explicit StreamManager(OperationExecutionEngine* operation_execution_engine);
  ~StreamManager() = default;

  /// Get or create a stream for the given device and stream ID
  StreamImpl* GetStream(c10::DeviceIndex device_index, c10::StreamId stream_id);

  /// Create a new stream on the specified device
  c10::StreamId CreateStream(c10::DeviceIndex device_index = -1, int priority = 0);

  /// Synchronize all streams on a device
  void SynchronizeDevice(c10::DeviceIndex device_index);

  /// Resource statistics for monitoring and debugging
  struct ManagerResourceStats {
    size_t total_streams;                ///< Total number of active streams
    size_t total_pending_operations;     ///< Total pending operations across all streams
    size_t total_memory_usage_estimate;  ///< Estimated memory usage in bytes
    size_t nrt_available_slots;          ///< Available NRT execution slots
    size_t nrt_acquired_slots;           ///< Currently acquired NRT slots
    size_t nrt_waiting_threads;          ///< Threads waiting for NRT slots
  };

  /// Get comprehensive resource statistics
  ManagerResourceStats GetResourceStats() const;

  /// Iterate through all streams on a device with a callback
  /// @param device_index The device to iterate streams on
  /// @param callback Function to call for each stream. Return true to stop iteration.
  void ForEachStreamImpl(c10::DeviceIndex device_index,
                         const std::function<bool(StreamImpl*)>& callback);

 private:
  // Non-copyable, non-movable
  StreamManager(const StreamManager&) = delete;
  StreamManager& operator=(const StreamManager&) = delete;
  StreamManager(StreamManager&&) = delete;
  StreamManager& operator=(StreamManager&&) = delete;

  // Thread synchronization
  mutable std::mutex mutex_;

  OperationExecutionEngine* operation_execution_engine_;

  // Stream storage: device_index -> stream_id -> StreamImpl
  std::unordered_map<c10::DeviceIndex,
                     std::unordered_map<c10::StreamId, std::unique_ptr<StreamImpl>>>
      streams_;
};

// Thread-local current stream tracking functions
c10::StreamId GetCurrentStreamId(c10::DeviceIndex device_index);
void SetCurrentStreamId(c10::DeviceIndex device_index, c10::StreamId stream_id);

}  // namespace at::neuron
