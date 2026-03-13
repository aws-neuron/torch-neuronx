#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "NeuronDevice.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"

namespace c10_neuron {

struct NeuronGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  NeuronGuardImpl() = default;

  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      c10_neuron::set_device(d.index());
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, c10_neuron::current_device());
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    c10_neuron::set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    c10_neuron::set_device(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    return at::neuron::getCurrentNeuronStream(d.index()).unwrap();
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    TORCH_INTERNAL_ASSERT(s.device_type() == c10::DeviceType::PrivateUse1);
    c10::Stream old_stream = getStream(s.device());
    at::neuron::setCurrentNeuronStream(at::neuron::NeuronStream(s));
    return old_stream;
  }

  c10::Stream getNewStream(c10::Device device, int priority = 0) const override {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::PrivateUse1);
    return at::neuron::NeuronStream::createStream(device.index(), priority).unwrap();
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(c10_neuron::device_count());
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
    return at::neuron::getDefaultNeuronStream(d.index()).unwrap();
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);

    // Neuron doesn't have a stream pool like CUDA, so we create a new stream.

    // This is called by Future::invokeCallback when .then() is used on a device-aware Future.
    int priority = isHighPriority ? -1 : 0;
    return at::neuron::NeuronStream::createStream(d.index(), priority).unwrap();
  }

  bool queryStream(const c10::Stream& stream) const override {
    TORCH_INTERNAL_ASSERT(stream.device_type() == c10::DeviceType::PrivateUse1);
    return at::neuron::NeuronStream(stream).query();
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    TORCH_INTERNAL_ASSERT(stream.device_type() == c10::DeviceType::PrivateUse1);
    at::neuron::NeuronStream(stream).synchronize();
  }

  void synchronizeEvent(void* event) const override {
    TORCH_INTERNAL_ASSERT(event != nullptr);
    static_cast<at::neuron::NeuronEvent*>(event)->synchronize();
  }

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    at::neuron::synchronize(device_index);
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                             const c10::Stream& stream) const override {
    TORCH_INTERNAL_ASSERT(stream.device_type() == c10::DeviceType::PrivateUse1);
    c10_neuron::NeuronCachingAllocator::recordStream(data_ptr, stream);
  }

  double elapsedTime(void* event1, void* event2,
                     const c10::DeviceIndex device_index) const override {
    TORCH_INTERNAL_ASSERT(event1 != nullptr);
    TORCH_INTERNAL_ASSERT(event2 != nullptr);
    // TODO(rpsilva): Phase0 interfaces: minimal no-op logic
    (void)device_index;  // Suppress unused parameter warning

    auto* start_event = static_cast<at::neuron::NeuronEvent*>(event1);
    auto* end_event = static_cast<at::neuron::NeuronEvent*>(event2);

    return static_cast<double>(start_event->elapsed_time(*end_event));
  }

  // Event-related methods
  void record(void** event, const c10::Stream& stream, const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    TORCH_INTERNAL_ASSERT(stream.device_type() == c10::DeviceType::PrivateUse1);
    // Create a new NeuronEvent if needed
    if (*event == nullptr) {
      bool enable_timing = (flag == c10::EventFlag::PYTORCH_DEFAULT);
      *event = new at::neuron::NeuronEvent(enable_timing);
    }
    // Record the event on the stream
    auto* neuron_event = static_cast<at::neuron::NeuronEvent*>(*event);
    at::neuron::NeuronStream neuron_stream(stream);
    neuron_event->record(neuron_stream);
  }

  void block(void* event, const c10::Stream& stream) const override {
    TORCH_INTERNAL_ASSERT(stream.device_type() == c10::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(event != nullptr, "Event cannot be null");

    auto* neuron_event = static_cast<at::neuron::NeuronEvent*>(event);
    neuron_event->block(at::neuron::NeuronStream(stream));
  }

  bool queryEvent(void* event) const override {
    TORCH_INTERNAL_ASSERT(event != nullptr, "Event cannot be null");

    auto* neuron_event = static_cast<at::neuron::NeuronEvent*>(event);
    return neuron_event->query();
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override {
    if (event != nullptr) {
      auto* neuron_event = static_cast<at::neuron::NeuronEvent*>(event);
      delete neuron_event;
    }
  }
};

}  // namespace c10_neuron
