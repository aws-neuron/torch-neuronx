#include "NeuronStream.h"

#include <ATen/SequenceNumber.h>
#include <ATen/record_function.h>
#include <c10/core/DeviceGuard.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdlib>

#include "NeuronEvent.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/ProfilerMappingCollector.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/profiler/CorrelationTracker.h"

namespace at::neuron {

// Stream ID encoding layout:
// Bits 0-2:  Pool index (0-7)
// Bits 3-4:  Priority (-1, 0, 1 as 2-bit two's complement)
// Bits 5+:   Unique counter (for distinguishing streams)
constexpr int kPoolIndexBits = 3;
constexpr int kPoolIndexMask = (1 << kPoolIndexBits) - 1;
constexpr int kPriorityBits = 2;
constexpr int kPriorityMask = (1 << kPriorityBits) - 1;
constexpr int kPriorityShift = kPoolIndexBits;
constexpr int kCounterShift = kPoolIndexBits + kPriorityBits;

inline c10::StreamId MakeStreamId(int pool_index, int priority, uint32_t counter) {
  return (static_cast<c10::StreamId>(counter) << kCounterShift) |
         (static_cast<c10::StreamId>(priority & kPriorityMask) << kPriorityShift) |
         (static_cast<c10::StreamId>(pool_index) & kPoolIndexMask);
}

inline int GetPoolIndexFromStreamId(c10::StreamId id) {
  if (id == 0) return -1;  // Default stream
  return static_cast<int>(id & kPoolIndexMask);
}

inline int GetPriorityFromStreamId(c10::StreamId id) {
  if (id == 0) return 0;  // Default stream
  int encoded = static_cast<int>((id >> kPriorityShift) & kPriorityMask);
  // Sign-extend from 2 bits
  return (encoded & 0x2) ? (encoded | ~0x3) : encoded;
}

// Use raw pointers to avoid static destruction order issues
// These are explicitly managed by InitializeStreamPools/CleanupStreamPools
std::array<std::array<StreamImpl*, kStreamPoolSize>, kMaxDevices> stream_pools{};
std::array<StreamImpl*, kMaxDevices> default_streams{};
std::array<std::atomic<uint32_t>, kMaxDevices> stream_counters{};
std::atomic<uint32_t> round_robin_counter{0};

// Check if host-cc is enabled via environment variable.
bool IsHostCCEnabled() {
  const char* host_cc = std::getenv("TORCH_NEURONX_ENABLE_HOST_CC");
  return host_cc && std::string(host_cc) == "1";
}

void EnableHostCC(int stream_pool_size) {
  nrt::HostCollectivesEnable(stream_pool_size);
  TORCH_NEURONX_DEBUG("Host Collectives is enabled for non-default stream support");
}

// Current streams for each device
thread_local std::array<c10::StreamId, kMaxDevices> current_streams{};

// Initialize streams for a single device
void InitDeviceStreamState(c10::DeviceIndex device) {
  // Create default stream (stream_id = 0)
  default_streams[device] = new StreamImpl(device, 0);

  // Create pool streams (stream_id = 1 to kStreamPoolSize)
  for (int i = 0; i < kStreamPoolSize; i++) {
    stream_pools[device][i] = new StreamImpl(device, i + 1);
  }
  stream_counters[device].store(1);
  TORCH_NEURONX_DEBUG("Initialized streams for device", "device=", static_cast<int>(device));
}

void InitializeStreamPools() {
  if (IsHostCCEnabled()) {
    EnableHostCC(kStreamPoolSize);
  }
  auto devices = c10_neuron::get_visible_device_indices();
  for (auto device : devices) {
    InitDeviceStreamState(device);
  }
  TORCH_NEURONX_DEBUG("Initialized stream pools for", "device_count=", devices.size());
}

void CleanupStreamPools() {
  for (auto& device_streams : stream_pools) {
    for (auto& stream : device_streams) {
      delete stream;
      stream = nullptr;
    }
  }
  for (auto& stream : default_streams) {
    delete stream;
    stream = nullptr;
  }
}

StreamImpl* GetStreamImplById(c10::DeviceIndex device_index, c10::StreamId stream_id) {
  TORCH_CHECK(device_index >= 0 && device_index < kMaxDevices, "Device index ",
              static_cast<int>(device_index), " out of range [0, ", kMaxDevices, ")");
  TORCH_CHECK(default_streams[device_index] != nullptr, "Stream pool not initialized for device ",
              static_cast<int>(device_index));

  if (stream_id == 0) {
    return default_streams[device_index];
  }
  int pool_index = GetPoolIndexFromStreamId(stream_id);
  TORCH_CHECK(pool_index >= 0 && pool_index < kStreamPoolSize, "Pool index ", pool_index,
              " out of range [0, ", kStreamPoolSize, ") for stream_id ", stream_id);

  return stream_pools[device_index][pool_index];
}

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
  return GetStreamImplById(device_index(), id());
}

int NeuronStream::priority() const { return GetPriorityFromStreamId(id()); }

bool NeuronStream::query() const {
  TORCH_NEURONX_TRACE_FUNCTION();
  return get_stream_impl()->Query();
}

void NeuronStream::synchronize() {
  TORCH_NEURONX_TRACE_FUNCTION();
  if (!c10_neuron::is_initialized()) {
    return;
  }
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
                " on device ", static_cast<int>(device_index()), " cannot wait for stream ",
                other.id(), " on device ", static_cast<int>(other.device_index()));
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
  c10::StreamId stream_id = current_streams[device_index];
  return NeuronStream(stream_id, device_index, UNCHECKED);
}

void NeuronStream::setCurrentStream(const NeuronStream& stream) {
  current_streams[stream.device_index()] = stream.id();
}

NeuronStream NeuronStream::createStream(c10::DeviceIndex device_index, int priority) {
  device_index = resolve_device_index(device_index);

  if (!IsHostCCEnabled()) {
    return getDefaultStream(device_index);
  }

  // Round-robin pool assignment
  uint32_t pool_index = round_robin_counter.fetch_add(1) % kStreamPoolSize;
  uint32_t counter = stream_counters[device_index].fetch_add(1);

  c10::StreamId stream_id = MakeStreamId(pool_index, priority, counter);
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

  if (default_streams[device_index] == nullptr) {
    return;
  }

  // Synchronize default stream
  default_streams[device_index]->Synchronize();

  // Synchronize all pool streams
  for (int i = 0; i < kStreamPoolSize; i++) {
    auto* stream = stream_pools[device_index][i];
    if (stream == nullptr) continue;
    stream->Synchronize();
  }
}

void SubmitOperationContext(const NeuronStream& stream,
                            std::unique_ptr<OperationContext> operation) {
  // Capture submission start time + PyTorch thread OS tid for profiler flow arrows.
  // This is the cudaLaunchKernel equivalent — a synchronous call on the PyTorch thread.
  operation->submit_start = std::chrono::steady_clock::now();

  // Capture PyTorch RecordFunction tracing identifiers at submission time.
  operation->pytorch_sequence_nr = at::sequence_number::peek();

  // Capture OS thread ID (matches "tid" field in PyTorch chrome trace)
  operation->os_tid = at::neuron::getOsTid();

  operation->pytorch_thread_id = at::RecordFunction::currentThreadId();
  operation->kineto_correlation_id = at::neuron::CorrelationTracker::current();
  operation->event_id = ProfilerMappingCollector::GenerateEventId();

  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("op_name=", operation->GetOpName());
  TORCH_NEURONX_DEBUG(
      "Submit operation context", "stream_id=", stream.id(), "op=", operation->GetOpName(),
      "device=", static_cast<int>(stream.device_index()),
      "pytorch_seq_nr=", operation->pytorch_sequence_nr,
      "pytorch_tid=", operation->pytorch_thread_id, "event_id=", operation->event_id);

  // Capture priority from stream_id before routing to HW queue
  operation->stream_priority = GetPriorityFromStreamId(stream.id());
  // Capture the full encoded stream ID for profiler mapping
  operation->stream_id = stream.id();

  auto* stream_impl = GetStreamImplById(stream.device_index(), stream.id());
  auto shared_future = stream_impl->SubmitOperationContext(std::move(operation));

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

// Direct stream accessors for performance-critical paths
StreamImpl* GetDefaultStreamImpl(c10::DeviceIndex device_index) {
  return default_streams[device_index];
}

StreamImpl* GetPoolStreamImpl(c10::DeviceIndex device_index, int pool_index) {
  return stream_pools[device_index][pool_index];
}

// Helper for ExecutionWorker to iterate all streams on a device
void ForEachStreamImpl(c10::DeviceIndex device_index,
                       const std::function<bool(StreamImpl*)>& callback) {
  auto* default_stream = default_streams[device_index];
  if (default_stream == nullptr) return;

  // Default stream first
  if (callback(default_stream)) return;

  // Then pool streams
  for (int i = 0; i < kStreamPoolSize; i++) {
    auto* pool_stream = stream_pools[device_index][i];
    if (pool_stream == nullptr) continue;
    if (callback(pool_stream)) return;
  }
}

}  // namespace at::neuron
