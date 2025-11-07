#include "NeuronStream.h"

#include <c10/core/DeviceGuard.h>

#include "NeuronEvent.h"
#include "torch_neuron/csrc/core/NeuronDevice.h"
#include "torch_neuron/csrc/core/NeuronLogging.h"
#include "torch_neuron/csrc/core/OperationContext.h"
#include "torch_neuron/csrc/core/streams/StreamImpl.h"
#include "torch_neuron/csrc/core/streams/StreamManager.h"

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

std::shared_ptr<StreamImpl> NeuronStream::get_stream_impl() const {
  auto stream_impl = StreamManager::Instance().GetStream(device_index(), id());
  TORCH_CHECK(stream_impl != nullptr, "Stream implementation not found for stream ", id(),
              " on device ", device_index());
  return stream_impl;
}

bool NeuronStream::query() const {
  TORCH_NEURON_TRACE_FUNCTION();
  return get_stream_impl()->Query();
}

void NeuronStream::synchronize() {
  TORCH_NEURON_TRACE_FUNCTION();
  get_stream_impl()->Synchronize();
}

void NeuronStream::wait_event(const NeuronEvent& event) const {
  TORCH_NEURON_TRACE_FUNCTION();

  // If event is already complete, nothing to do
  if (event.query()) {
    TORCH_NEURON_DEBUG("Event already complete, no wait needed", "stream_id=", id(),
                       "event=", static_cast<const void*>(&event));
    return;
  }

  // Register this event as a dependency for the next operation on this stream
  try {
    get_stream_impl()->RegisterEventDependencyForNextOperation(&event);
    TORCH_NEURON_DEBUG("Registered event dependency for stream", "stream_id=", id(),
                       "event=", static_cast<const void*>(&event));
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to register event dependency on stream ", id(), ": ", e.what());
  }
}

void NeuronStream::wait_stream(const NeuronStream& other) {
  TORCH_NEURON_TRACE_FUNCTION();
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
    event.record(other);
    wait_event(event);
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to synchronize stream ", id(), " with stream ", other.id(), ": ",
                e.what());
  }
}

void NeuronStream::register_event_after_current_operations(NeuronEvent* event) const {
  TORCH_NEURON_TRACE_FUNCTION();
  TORCH_CHECK(event != nullptr, "Cannot register null event on stream ", id());

  try {
    auto stream_impl = get_stream_impl();
    stream_impl->RegisterEventWithLastOperation(event);
  } catch (const std::exception& e) {
    TORCH_NEURON_ERROR("Failed to register event with stream", "stream_id=", id(),
                       "error=", e.what());
    // Avoid hanging the event
    event->complete_event();
    throw;
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
  c10::StreamId stream_id = StreamManager::Instance().CreateStream(device_index, priority);
  return NeuronStream(stream_id, device_index, UNCHECKED);
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
  StreamManager::Instance().SynchronizeDevice(device_index);
}

std::future<OperationContextResult> submit_operation_context(const NeuronStream& stream,
                                                             OperationContext operation_context) {
  TORCH_NEURON_TRACE_FUNCTION_WITH_ARGS("op_name=", operation_context.GetOpName());
  TORCH_NEURON_DEBUG("Submit operation context", "stream_id=", stream.id(),
                     "op=", operation_context.GetOpName(), "device=", stream.device_index());
  auto stream_impl = StreamManager::Instance().GetStream(stream.device_index(), stream.id());
  return stream_impl->SubmitOperationContext(std::move(operation_context));
}

}  // namespace at::neuron
