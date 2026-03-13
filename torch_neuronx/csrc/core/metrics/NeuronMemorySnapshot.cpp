#include "NeuronMemorySnapshot.h"

// Python.h must be included before any standard headers
#include <Python.h>
#include <frameobject.h>
#include <torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h>
#include <torch_neuronx/csrc/core/metrics/NeuronMetrics.h>

#include <algorithm>
#include <stdexcept>

namespace c10_neuron {
namespace NeuronMemorySnapshot {

// =============================================================================
// TraceBuffer - Thread-safe ring buffer for trace entries
// =============================================================================

class TraceBuffer {
 public:
  // Maximum reasonable pre-allocation size (1M entries)
  // Beyond this, we grow dynamically to avoid memory issues with sys.maxsize
  static constexpr size_t kMaxPreallocSize = 1024 * 1024;

  explicit TraceBuffer(size_t max_entries = 1) : max_entries_(max_entries) {
    // Only pre-allocate up to a reasonable size
    // For very large max_entries (like sys.maxsize), we grow dynamically
    size_t prealloc = std::min(max_entries, kMaxPreallocSize);
    if (prealloc > 0) {
      buffer_.reserve(prealloc);
    }
  }

  void SetMaxEntries(size_t max_entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_entries_ = max_entries;
    // If buffer is larger than new max, trim oldest entries
    if (buffer_.size() > max_entries_) {
      // Keep only the most recent max_entries_ entries
      size_t to_remove = buffer_.size() - max_entries_;
      buffer_.erase(buffer_.begin(), buffer_.begin() + to_remove);
      head_ = 0;
      count_ = buffer_.size();
    }
  }

  void insert(TraceEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_entries_ == 0) {
      return;
    }

    if (buffer_.size() < max_entries_) {
      // Buffer not full yet, just append
      buffer_.push_back(std::move(entry));
      count_ = buffer_.size();
    } else {
      // Buffer full, overwrite oldest entry
      buffer_[head_] = std::move(entry);
      head_ = (head_ + 1) % max_entries_;
    }
  }

  std::vector<TraceEntry> getEntries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.size() < max_entries_) {
      // Buffer not full, entries are in order
      return buffer_;
    }
    // Buffer is full, need to reorder from head
    std::vector<TraceEntry> result;
    result.reserve(buffer_.size());
    for (size_t i = 0; i < buffer_.size(); ++i) {
      size_t idx = (head_ + i) % buffer_.size();
      result.push_back(buffer_[idx]);
    }
    return result;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    head_ = 0;
    count_ = 0;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
  }

  size_t maxEntries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_entries_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<TraceEntry> buffer_;
  size_t max_entries_;
  size_t head_ = 0;   // Next write position (for circular buffer)
  size_t count_ = 0;  // Total entries ever written
};

// =============================================================================
// Global State
// =============================================================================

namespace {

// Recording configuration
std::atomic<bool> g_record_enabled{false};
std::atomic<RecordContext> g_record_context{RecordContext::NEVER};
std::atomic<bool> g_include_cpp_frames{false};

// Per-device trace buffers (lazily initialized)
std::mutex g_buffers_mutex;
std::vector<std::unique_ptr<TraceBuffer>> g_device_buffers;

// Default max entries for trace buffer
size_t g_max_entries = 1;

// Get or create trace buffer for a device
TraceBuffer& getDeviceBuffer(int device) {
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  if (device >= static_cast<int>(g_device_buffers.size())) {
    g_device_buffers.resize(device + 1);
  }
  if (!g_device_buffers[device]) {
    g_device_buffers[device] = std::make_unique<TraceBuffer>(g_max_entries);
  }
  return *g_device_buffers[device];
}

// Clear all device buffers
void clearAllBuffers() {
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  for (auto& buffer : g_device_buffers) {
    if (buffer) {
      buffer->clear();
    }
  }
}

// Set max entries for all buffers
void setAllBuffersMaxEntries(size_t max_entries) {
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  g_max_entries = max_entries;
  for (auto& buffer : g_device_buffers) {
    if (buffer) {
      buffer->SetMaxEntries(max_entries);
    }
  }
}

}  // namespace

// =============================================================================
// Public API Implementation
// =============================================================================

int64_t GetCurrentTimestampUs() { return at::neuron::metrics::GetCurrentTimeNs() / 1000; }

const char* TraceActionToString(TraceAction action) {
  switch (action) {
    case TraceAction::ALLOC:
      return "alloc";
    case TraceAction::FREE_REQUESTED:
      return "free_requested";
    case TraceAction::FREE_COMPLETED:
      return "free_completed";
    case TraceAction::SEGMENT_ALLOC:
      return "segment_alloc";
    case TraceAction::SEGMENT_FREE:
      return "segment_free";
    case TraceAction::SEGMENT_MAP:
      return "segment_map";
    case TraceAction::SEGMENT_UNMAP:
      return "segment_unmap";
    case TraceAction::SNAPSHOT:
      return "snapshot";
    case TraceAction::OOM:
      return "oom";
    default:
      return "unknown";
  }
}

RecordContext StringToRecordContext(const std::string& str) {
  if (str == "never" || str == "none") {
    return RecordContext::NEVER;
  } else if (str == "state") {
    return RecordContext::STATE;
  } else if (str == "alloc") {
    return RecordContext::ALLOC;
  } else if (str == "all") {
    return RecordContext::ALL;
  }
  throw std::invalid_argument("Invalid RecordContext string: " + str);
}

void RecordHistory(bool enabled, RecordContext when, bool include_cpp_frames, size_t max_entries,
                   bool clear_history) {
  g_record_enabled.store(enabled, std::memory_order_release);
  g_record_context.store(when, std::memory_order_release);
  g_include_cpp_frames.store(include_cpp_frames, std::memory_order_release);

  if (max_entries > 0) {
    setAllBuffersMaxEntries(max_entries);
  }

  if (clear_history) {
    clearAllBuffers();
  }
}

void RecordTrace(TraceAction action, int device, size_t addr, size_t size, c10::StreamId stream,
                 std::shared_ptr<StackContext> context, size_t device_free) {
  if (!g_record_enabled.load(std::memory_order_acquire)) {
    return;
  }

  TraceEntry entry(action, device, addr, size, stream, GetCurrentTimestampUs(), std::move(context),
                   device_free);

  getDeviceBuffer(device).insert(std::move(entry));
}

bool IsRecordingEnabled() { return g_record_enabled.load(std::memory_order_acquire); }

RecordContext GetRecordContext() { return g_record_context.load(std::memory_order_acquire); }

bool ShouldIncludeCppFrames() { return g_include_cpp_frames.load(std::memory_order_acquire); }

// StackContext::capture - Capture Python stack trace
//
// GIL Acquisition Strategy:
// -------------------------
// We acquire the GIL here to access Python frame objects. This is safe because:
// 1. This function is called BEFORE any allocator locks are held
// 2. The call originates from Python, so we're just re-acquiring what was released
// 3. We use PyGILState_Ensure/Release which handles nested acquisition correctly
//
// The allocator calls StackContext::capture() before inserting into the trace
// buffer, so there's no lock ordering issue.
//
std::shared_ptr<StackContext> StackContext::capture(bool include_cpp) {
  // Note: include_cpp is not implemented - C++ stack traces require
  // platform-specific unwinding (e.g., libunwind) which adds complexity.
  // For now, we only capture Python frames which is the most useful for debugging.
  (void)include_cpp;

  // Check if Python is initialized
  if (!Py_IsInitialized()) {
    return nullptr;
  }

  // Acquire the GIL to safely access Python frame objects
  PyGILState_STATE gstate = PyGILState_Ensure();

  auto context = std::make_shared<StackContext>();

  // Get the current Python frame
  PyFrameObject* first_frame = PyEval_GetFrame();
  if (!first_frame) {
    PyGILState_Release(gstate);
    return context;  // Return empty context if no Python frame
  }

  // Walk up the frame stack
  PyFrameObject* frame = first_frame;
  while (frame != nullptr) {
    // Get frame info using Python 3.9+ API
    PyCodeObject* code = PyFrame_GetCode(frame);
    if (code) {
      // Get filename
      const char* filename = "";
      if (code->co_filename) {
        filename = PyUnicode_AsUTF8(code->co_filename);
        if (!filename) {
          PyErr_Clear();
          filename = "<unknown>";
        }
      }

      // Get function name
      const char* funcname = "";
      if (code->co_name) {
        funcname = PyUnicode_AsUTF8(code->co_name);
        if (!funcname) {
          PyErr_Clear();
          funcname = "<unknown>";
        }
      }

      // Get line number
      int lineno = PyFrame_GetLineNumber(frame);

      context->frames.emplace_back(filename ? filename : "<unknown>",
                                   funcname ? funcname : "<unknown>", lineno);

      Py_DECREF(code);
    }

    // Move to the next frame (caller)
    PyFrameObject* back = PyFrame_GetBack(frame);
    // Only decref frames obtained from PyFrame_GetBack (new references)
    // Don't decref first_frame as it's a borrowed reference from PyEval_GetFrame
    if (frame != first_frame) {
      Py_DECREF(frame);
    }
    frame = back;
  }

  // Release the GIL
  PyGILState_Release(gstate);

  return context;
}

// snapshot() - Capture current memory state
SnapshotInfo snapshot() {
  SnapshotInfo info;

  // Get trace entries from all device buffers
  {
    std::lock_guard<std::mutex> lock(g_buffers_mutex);
    info.device_traces.resize(g_device_buffers.size());
    for (size_t i = 0; i < g_device_buffers.size(); ++i) {
      if (g_device_buffers[i]) {
        info.device_traces[i] = g_device_buffers[i]->getEntries();
      }
    }
  }

  // Record a SNAPSHOT trace event for correlation
  if (g_record_enabled.load(std::memory_order_acquire)) {
    // Record on device 0 by default
    RecordTrace(TraceAction::SNAPSHOT, 0, 0, 0, 0, nullptr, 0);
  }

  // Get segments from the allocator - already in unified SegmentInfo format
  // The allocator now returns NeuronMemorySnapshot::SegmentInfo directly,
  // including stack traces in blocks and recorded_streams at segment level
  info.segments = NeuronCachingAllocator::getSnapshotSegments();

  return info;
}

}  // namespace NeuronMemorySnapshot
}  // namespace c10_neuron
