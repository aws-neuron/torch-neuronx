#include "StreamManager.h"

#include <chrono>
#include <thread>
#include <unordered_map>

#include "StreamImpl.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"

namespace at::neuron {

// Thread-local current stream tracking per device
thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId> current_streams_;

c10::StreamId GetCurrentStreamId(c10::DeviceIndex device_index) {
  auto it = current_streams_.find(device_index);
  if (it != current_streams_.end()) {
    return it->second;
  }
  return 0;  // Default stream
}

void SetCurrentStreamId(c10::DeviceIndex device_index, c10::StreamId stream_id) {
  current_streams_[device_index] = stream_id;
}

StreamManager& StreamManager::Instance() {
  static StreamManager manager;
  return manager;
}

std::shared_ptr<StreamImpl> StreamManager::GetStream(c10::DeviceIndex device_index,
                                                     c10::StreamId stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto device_it = streams_.find(device_index);
  if (device_it == streams_.end()) {
    device_it =
        streams_
            .emplace(device_index, std::unordered_map<c10::StreamId, std::shared_ptr<StreamImpl>>{})
            .first;
  }

  auto& operation_execution_engine =
      NeuronResourceManager::Instance().GetOperationExecutionEngine();
  auto stream_it = device_it->second.find(stream_id);
  if (stream_it == device_it->second.end()) {
    auto stream_impl =
        std::make_shared<StreamImpl>(device_index, stream_id, 0, &operation_execution_engine);
    device_it->second[stream_id] = stream_impl;
    return stream_impl;
  }

  return stream_it->second;
}

c10::StreamId StreamManager::CreateStream(c10::DeviceIndex device_index, int priority) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Generate new stream ID
  static std::atomic<c10::StreamId> next_stream_id{1};
  c10::StreamId stream_id = next_stream_id++;

  auto device_it = streams_.find(device_index);
  if (device_it == streams_.end()) {
    device_it =
        streams_
            .emplace(device_index, std::unordered_map<c10::StreamId, std::shared_ptr<StreamImpl>>{})
            .first;
  }
  auto& operation_execution_engine =
      NeuronResourceManager::Instance().GetOperationExecutionEngine();
  auto stream_impl =
      std::make_shared<StreamImpl>(device_index, stream_id, priority, &operation_execution_engine);
  device_it->second[stream_id] = stream_impl;
  return stream_id;
}

void StreamManager::SynchronizeDevice(c10::DeviceIndex device_index) {
  std::vector<std::shared_ptr<StreamImpl>> streams_to_sync;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto device_it = streams_.find(device_index);
    if (device_it != streams_.end()) {
      streams_to_sync.reserve(device_it->second.size());
      for (auto& [stream_id, stream_impl] : device_it->second) {
        streams_to_sync.push_back(stream_impl);
      }
    }
  }

  // Synchronize without holding the lock
  for (auto& stream_impl : streams_to_sync) {
    stream_impl->Synchronize();
  }
}

StreamManager::ManagerResourceStats StreamManager::GetResourceStats() const {
  ManagerResourceStats stats;
  stats.total_streams = 0;
  stats.total_pending_operations = 0;
  stats.total_memory_usage_estimate = 0;

  auto& resource_manager = NeuronResourceManager::Instance();
  auto resource_stats = resource_manager.GetResourceStats();
  stats.nrt_available_slots = resource_stats.resource_available_slots;
  stats.nrt_acquired_slots = resource_stats.resource_acquired_slots;
  stats.nrt_waiting_threads = resource_stats.resource_waiting_threads;

  {
    std::lock_guard<std::mutex> streams_lock(mutex_);
    for (const auto& [device_index, device_streams] : streams_) {
      for (const auto& [stream_id, stream_impl] : device_streams) {
        if (stream_impl) {
          stats.total_streams++;

          try {
            auto stream_stats = stream_impl->GetResourceStats();
            stats.total_pending_operations += stream_stats.pending_operations;
            stats.total_memory_usage_estimate +=
                (stream_stats.pending_operations * sizeof(OperationContext)) +
                (stream_stats.compilation_queue_size * sizeof(std::shared_ptr<OperationContext>)) +
                (stream_stats.execution_queue_size * sizeof(std::shared_ptr<OperationContext>));
          } catch (const std::exception& e) {
            TORCH_NEURONX_ERROR("Exception getting stream stats", "stream_id=", stream_id,
                                "error=", e.what());
          }
        }
      }
    }
  }

  return stats;
}

void StreamManager::DestroyStream(c10::DeviceIndex device_index, c10::StreamId stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto device_it = streams_.find(device_index);
  if (device_it != streams_.end()) {
    device_it->second.erase(stream_id);
    if (device_it->second.empty()) {
      streams_.erase(device_it);
    }
  }
}

StreamManager::StreamManager() {
  try {
    auto current_device = c10_neuron::current_device();
    GetStream(current_device, 0);
  } catch (const std::exception& e) {
    throw std::runtime_error("StreamManager initialization failed: " + std::string(e.what()));
  }
}

StreamManager::~StreamManager() {
  TORCH_NEURONX_DEBUG("~StreamManager: Stop monitoring thread");
  // Stop monitoring thread
  monitoring_enabled_.store(false);
  TORCH_NEURONX_DEBUG("~StreamManager: Stopped monitoring thread");
}

}  // namespace at::neuron
