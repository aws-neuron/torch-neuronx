/*
 * Neuron Profiler Stubs Implementation
 *
 * Implements torch::profiler::impl::ProfilerStubs interface for
 * PrivateUse1 device (Neuron).
 *
 * Note: This is separate from the Kineto activity profiler
 * (Profiler.cpp).
 * PyTorch has two profiler registration systems:
 * 1. Kineto Activity Profiler - for trace collection (Profiler.cpp)
 * 2. ProfilerStubs interface - for device-specific timing/sync (this file)
 */

#include <c10/core/DeviceType.h>
#include <torch/csrc/profiler/stubs/base.h>

#include <chrono>
#include <functional>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"

namespace at::neuron {

struct TorchStubs : public torch::profiler::impl::ProfilerStubs {
  void record(c10::DeviceIndex* device, torch::profiler::impl::ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override {
    if (device) {
      *device = static_cast<c10::DeviceIndex>(c10_neuron::current_device());
    }

    if (cpu_ns) {
      *cpu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
    }

    if (event) {
      *event = nullptr;
    }
  }

  // TODO: Implement device-side elapsed time once NRT exposes event timestamps.
  // Currently a no-op because record() does not create real device events.
  float elapsed(const torch::profiler::impl::ProfilerVoidEventStub* /*event*/,
                const torch::profiler::impl::ProfilerVoidEventStub* /*event2*/) const override {
    return 0.0f;
  }

  void mark(const char* /*name*/) const override {}
  void rangePush(const char* /*name*/) const override {}
  void rangePop() const override {}

  bool enabled() const override { return true; }

  void onEachDevice(std::function<void(int)> op) const override {
    for (auto device_idx : c10_neuron::get_visible_device_indices()) {
      op(static_cast<int>(device_idx));
    }
  }

  void synchronize() const override {
    for (auto device_idx : c10_neuron::get_visible_device_indices()) {
      at::neuron::synchronize(device_idx);
    }
  }
};

static TorchStubs neuron_profiler_stubs;

// Auto-register profiler stubs at module load time
struct RegisterTorchStubs {
  RegisterTorchStubs() {
    torch::profiler::impl::registerPrivateUse1Methods(&neuron_profiler_stubs);
  }
} register_neuron_profiler_stubs;

}  // namespace at::neuron
