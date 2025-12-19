#include "NeuronStream.h"

#include <c10/core/DeviceGuard.h>

#include "NeuronEvent.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/streams/StreamManager.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace at::neuron {

namespace {
c10::DeviceIndex resolve_device_index(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  TORCH_CHECK(device_index >= 0, "Invalid device index: ", device_index);
  return device_index;
}
}  // namespace

StreamImpl* NeuronStream::get_stream_impl() const {
  auto& manager = at::neuron::NeuronResourceManager::Instance().GetStreamManager();
  auto stream_impl = manager.GetStream(device_index(), id());
  TORCH_CHECK(stream_impl != nullptr, "Stream implementation not found for stream ", id(),
              " on device ", device_index());
  return stream_impl;
}

int NeuronStream::priority() const { return get_stream_impl()->priority; }

bool NeuronStream::query() const {
  TORCH_NEURONX_TRACE_FUNCTION();
  return get_stream_impl()->Query();
}

void NeuronStream::synchronize() {
  TORCH_NEURONX_TRACE_FUNCTION();
  get_stream_impl()->Synchronize();
}

void NeuronStream::wait_event(const NeuronEvent& event) const {
  TORCH_NEURONX_TRACE_FUNCTION();

  // If event is already complete, nothing to do
  if (event.query()) {
    TORCH_NEURONX_DEBUG("Event already complete, no wait needed", "stream_id=", id(),
                        "event=", static_cast<const void*>(&event));
    return;
  }

  // Submit event wait primitive
  SubmitEventWait(event);
  TORCH_NEURONX_DEBUG("Submitted event wait primitive", "stream_id=", id());
}

void NeuronStream::wait_stream(const NeuronStream& other) const {
  TORCH_NEURONX_TRACE_FUNCTION();

  if (*this == other) {
    return;
  }
  // Different devices should use explicit synchronization.
  // TODO: Validate how it works outside of Neuron
  if (device_index() != other.device_index()) {
    TORCH_CHECK(false, "Cross-device stream synchronization not supported: stream ", id(),
                " on device ", device_index(), " cannot wait for stream ", other.id(),
                " on device ", other.device_index());
  }

  try {
    NeuronEvent event;
    // Record on producer stream
    event.record(other);
    // Wait on consumer stream
    wait_event(event);
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to synchronize stream ", id(), " with stream ", other.id(), ": ",
                e.what());
  }
}

NeuronStream NeuronStream::getDefaultStream(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  return NeuronStream(0, device_index, UNCHECKED);
}

NeuronStream NeuronStream::getCurrentStream(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  c10::StreamId stream_id = GetCurrentStreamId(device_index);
  return NeuronStream(stream_id, device_index, UNCHECKED);
}

void NeuronStream::setCurrentStream(const NeuronStream& stream) {
  SetCurrentStreamId(stream.device_index(), stream.id());
}

NeuronStream NeuronStream::createStream(c10::DeviceIndex device_index, int priority) {
  device_index = resolve_device_index(device_index);
  auto& manager = at::neuron::NeuronResourceManager::Instance().GetStreamManager();
  c10::StreamId stream_id = manager.CreateStream(device_index, priority);
  return NeuronStream(stream_id, device_index, UNCHECKED);
}

void NeuronStream::SubmitEventSignal(const NeuronEvent& event) const {
  TORCH_NEURONX_DEBUG("Submitting event signal primitive", "stream_id=", id(),
                      "device=", device_index());

  auto event_kernel = std::make_unique<EventDirectKernelExecution>(
      "event_signal", event, EventDirectKernelExecution::EventAction::kSignal, device_index());

  auto operation = std::make_unique<OperationContext>(std::move(event_kernel));
  SubmitOperationContext(*this, std::move(operation));
}

void NeuronStream::SubmitEventWait(const NeuronEvent& event) const {
  TORCH_NEURONX_DEBUG("Submitting event wait primitive", "stream_id=", id(),
                      "device=", device_index());

  auto event_kernel = std::make_unique<EventDirectKernelExecution>(
      "event_wait", event, EventDirectKernelExecution::EventAction::kWait, device_index());

  auto operation = std::make_unique<OperationContext>(std::move(event_kernel));
  SubmitOperationContext(*this, std::move(operation));
}

// Global functions
NeuronStream getCurrentNeuronStream(c10::DeviceIndex device_index) {
  return NeuronStream::getCurrentStream(device_index);
}

NeuronStream getDefaultNeuronStream(c10::DeviceIndex device_index) {
  return NeuronStream::getDefaultStream(device_index);
}

void setCurrentNeuronStream(const NeuronStream& stream) { NeuronStream::setCurrentStream(stream); }

c10::DeviceIndex current_device() { return c10_neuron::current_device(); }

void synchronize(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  auto& manager = at::neuron::NeuronResourceManager::Instance().GetStreamManager();
  manager.SynchronizeDevice(device_index);
}

void SubmitOperationContext(const NeuronStream& stream,
                            std::unique_ptr<OperationContext> operation) {
  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("op_name=", operation->GetOpName());
  TORCH_NEURONX_DEBUG("Submit operation context", "stream_id=", stream.id(),
                      "op=", operation->GetOpName(), "device=", stream.device_index());

  auto& stream_manager = NeuronResourceManager::Instance().GetStreamManager();
  auto stream_impl = stream_manager.GetStream(stream.device_index(), stream.id());
  auto future = stream_impl->SubmitOperationContext(std::move(operation));
  auto shared_future = future.share();

  static const bool neuron_launch_blocking = []() {
    const char* env_val = std::getenv("NEURON_LAUNCH_BLOCKING");
    if (!env_val || std::strlen(env_val) == 0) return false;
    return std::string(env_val) == "1";
  }();

  if (neuron_launch_blocking) {
    at::neuron::OperationContextResult result = shared_future.get();
    if (!result.IsSuccess()) {
      throw std::runtime_error(result.GetError());
    }
  }
}

}  // namespace at::neuron
