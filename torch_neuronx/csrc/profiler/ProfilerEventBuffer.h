#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace at::neuron {

// Convert steady_clock time_point to nanoseconds since epoch.
inline int64_t toNs(std::chrono::steady_clock::time_point tp) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

// Snapshot of a completed operation, captured in CompleteOperation().
// All timing values are in nanoseconds (steady_clock epoch).
struct ProfiledOperation {
  std::string op_name;

  int64_t submit_start_ns{0};  // Top of SubmitOperationContext (PyTorch thread)
  int64_t submit_time_ns{0};   // End of SubmitOperationContext (enqueue complete)
  int64_t execute_start_ns{0};
  int64_t execute_end_ns{0};
  int64_t dispatch_start_ns{0};  // RuntimeWorker picks up op (dispatch end = execute_end_ns)

  uint64_t nrt_sequence_id{0};        // NRT async sequence ID (hardware-level)
  uint64_t kineto_correlation_id{0};  // Links device activity → originating CPU op
  uint64_t pytorch_thread_id{0};      // PyTorch logical thread counter (NOT an OS tid)

  int32_t device_index{0};  // Neuron device index (0, 1, ...)
  int32_t stream_id{0};     // Logical Neuron stream (0, 1, 2)
  int32_t xu_type{-1};  // XU type index (NRTA_XU_TENSOR_OP, NRTA_XU_COMPUTE, NRTA_XU_COLLECTIVES)

  int32_t neuron_stream_os_tid{-1};  // -1 = not yet assigned; valid tids are positive (gettid())
  int32_t pytorch_os_tid{0};         // OS tid (gettid) of the PyTorch calling thread
  int32_t dispatch_slot{-1};         // Inflight slot index for parallel row visualization

  // For kHost operations: index of the HostWorkerPool worker.
  int32_t host_worker_idx{-1};

  std::string hierarchy_string;      // nn.Module hierarchy (e.g. "Linear[fc1][1]/")
  std::string neff_cache_key;        // NEFF compilation cache key
  std::string persistent_cache_key;  // Persistent cache key (matches NEFF filename on disk)
  int64_t pytorch_sequence_nr{-1};   // PyTorch sequence number for cross-reference
};

// Lightweight event for RuntimeWorker idle/sleep periods.
struct RuntimeSleepEvent {
  int64_t start_ns{0};
  int64_t end_ns{0};
};

// Thread-safe buffer for profiled operation snapshots.
// Session::start() activates the buffer and publishes it via the global atomic pointer.
// Session::stop() clears the pointer and deactivates the buffer.
// CompleteOperation() checks the global pointer and records snapshots.
class ProfilerEventBuffer {
 public:
  ProfilerEventBuffer() = default;

  // Activate the buffer — subsequent record() calls will store events.
  void activate();

  // Deactivate the buffer — subsequent record() calls are no-ops.
  void deactivate();

  // Record a profiled operation snapshot. No-op if not active.
  void record(ProfiledOperation op);

  // Record a RuntimeWorker sleep event. No-op if not active.
  void recordSleep(RuntimeSleepEvent event);

  // Drain all recorded events. Returns the events and clears the internal buffer.
  std::vector<ProfiledOperation> drain();

  // Drain sleep events.
  std::vector<RuntimeSleepEvent> drainSleepEvents();

  // Returns true if the buffer is active.
  bool isActive() const;

 private:
  mutable std::mutex mutex_;
  bool active_{false};
  std::vector<ProfiledOperation> events_;
  std::vector<RuntimeSleepEvent> sleep_events_;
};

// Global atomic pointer to the active profiler event buffer.
// Set by Session::start(), cleared by Session::stop().
// Checked by CompleteOperation() to decide whether to record.
ProfilerEventBuffer* getActiveProfilerBuffer();
void setActiveProfilerBuffer(ProfilerEventBuffer* buffer);

}  // namespace at::neuron
