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

StreamImpl* StreamManager::GetStream(c10::DeviceIndex device_index, c10::StreamId stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto device_it = streams_.find(device_index);
  if (device_it == streams_.end()) {
    device_it =
        streams_
            .emplace(device_index, std::unordered_map<c10::StreamId, std::unique_ptr<StreamImpl>>{})
            .first;
  }

  auto stream_it = device_it->second.find(stream_id);
  if (stream_it == device_it->second.end()) {
    auto stream_impl =
        std::make_unique<StreamImpl>(device_index, stream_id, 0, operation_execution_engine_);
    auto* stream_ptr = stream_impl.get();
    device_it->second[stream_id] = std::move(stream_impl);
    return stream_ptr;
  }

  return stream_it->second.get();
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
            .emplace(device_index, std::unordered_map<c10::StreamId, std::unique_ptr<StreamImpl>>{})
            .first;
  }
  auto stream_impl =
      std::make_unique<StreamImpl>(device_index, stream_id, priority, operation_execution_engine_);
  device_it->second[stream_id] = std::move(stream_impl);
  return stream_id;
}

void StreamManager::SynchronizeDevice(c10::DeviceIndex device_index) {
  ForEachStreamImpl(device_index, [](StreamImpl* stream_impl) -> bool {
    stream_impl->Synchronize();
    return false;  // Continue to all streams
  });
}

void StreamManager::ForEachStreamImpl(c10::DeviceIndex device_index,
                                      const std::function<bool(StreamImpl*)>& callback) {
  std::vector<StreamImpl*> streams_to_iterate;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto device_it = streams_.find(device_index);
    if (device_it != streams_.end()) {
      streams_to_iterate.reserve(device_it->second.size());
      for (auto& [stream_id, stream_impl] : device_it->second) {
        if (stream_impl) {
          streams_to_iterate.push_back(stream_impl.get());
        }
      }
    }
  }
  for (auto& stream_impl : streams_to_iterate) {
    if (callback(stream_impl)) {
      return;
    }
  }
}

StreamManager::StreamManager(OperationExecutionEngine* operation_execution_engine)
    : operation_execution_engine_(operation_execution_engine) {
  try {
    auto current_device = c10_neuron::current_device();
    TORCH_CHECK(GetStream(current_device, 0) != nullptr, "Default stream initialization failed");
  } catch (const std::exception& e) {
    throw std::runtime_error("StreamManager initialization failed: " + std::string(e.what()));
  }
}

}  // namespace at::neuron
