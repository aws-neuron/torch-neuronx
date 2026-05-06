#ifndef TORCH_NEURON_CSRC_CORE_ALLOCATOR_NEURON_MEMORY_SNAPSHOT_H_
#define TORCH_NEURON_CSRC_CORE_ALLOCATOR_NEURON_MEMORY_SNAPSHOT_H_

#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10_neuron {
namespace NeuronMemorySnapshot {

// =============================================================================
// Enums
// =============================================================================

/**
 * TraceAction - Types of memory events that can be recorded.
 */
enum class TraceAction {
  ALLOC,           // Memory allocated to user (adds to active memory)
  FREE_REQUESTED,  // Tensor freed by user (subtracts from active memory)
  FREE_COMPLETED,  // Memory freed to NRT (nrt_tensor_free completed)
  SEGMENT_ALLOC,   // Reserved for future use (segment-level tracking)
  SEGMENT_FREE,    // Reserved for future use (segment-level tracking)
  SEGMENT_MAP,     // Reserved for future use
  SEGMENT_UNMAP,   // Reserved for future use
  SNAPSHOT,        // Snapshot taken (for correlation with traces)
  OOM              // Out of memory error
};

/**
 * RecordContext - Controls when stack traces are captured.
 */
enum class RecordContext {
  NEVER,  // No tracebacks recorded
  STATE,  // Tracebacks for currently allocated memory only
  ALLOC,  // Tracebacks for allocation events
  ALL     // Tracebacks for both allocation and free events
};

// =============================================================================
// Stack Trace Structures
// =============================================================================

/**
 * StackFrame - A single frame in a stack trace.
 */
struct StackFrame {
  std::string filename;
  std::string funcname;
  int lineno = 0;

  StackFrame() = default;
  StackFrame(std::string fn, std::string func, int line)
      : filename(std::move(fn)), funcname(std::move(func)), lineno(line) {}
};

/**
 * StackContext - Captured stack trace (Python and optionally C++).
 */
struct StackContext {
  std::vector<StackFrame> frames;

  /**
   * Capture current stack trace.
   * @param include_cpp If true, include C++ frames (not implemented initially)
   * @return Shared pointer to captured context, or nullptr if capture fails
   */
  static std::shared_ptr<StackContext> capture(bool include_cpp = false);
};

// =============================================================================
// Trace Entry
// =============================================================================

/**
 * TraceEntry - A single memory event record.
 */
struct TraceEntry {
  TraceAction action;
  int device = 0;
  size_t addr = 0;                        // Memory address (for OOM: requested size that failed)
  size_t size = 0;                        // Allocation size
  c10::StreamId stream = 0;               // NeuronStream ID
  int64_t timestamp_us = 0;               // Timestamp in microseconds
  std::shared_ptr<StackContext> context;  // Optional stack trace
  size_t device_free = 0;                 // Only for OOM: free memory at failure time

  TraceEntry() = default;
  TraceEntry(TraceAction act, int dev, size_t address, size_t sz, c10::StreamId str, int64_t ts,
             std::shared_ptr<StackContext> ctx = nullptr, size_t dev_free = 0)
      : action(act),
        device(dev),
        addr(address),
        size(sz),
        stream(str),
        timestamp_us(ts),
        context(std::move(ctx)),
        device_free(dev_free) {}
};

// =============================================================================
// Snapshot Structures
// =============================================================================

/**
 * BlockInfo - Information about a memory block within a segment.
 */
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  size_t address = 0;
  bool allocated = false;  // true = active_allocated, false = inactive
  std::shared_ptr<StackContext> context;
};

/**
 * SegmentInfo - Information about a memory segment (NRT super tensor).
 * Each segment contains one or more blocks from splitting.
 */
struct SegmentInfo {
  int device = 0;
  size_t address = 0;
  size_t total_size = 0;
  size_t allocated_size = 0;
  size_t active_size = 0;
  c10::StreamId stream = 0;
  bool is_small = false;  // true for small pool (< kPoolBoundary)
  std::vector<BlockInfo> blocks;
};

/**
 * SnapshotInfo - Complete memory snapshot.
 */
struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;  // Per-device trace history
};

// =============================================================================
// Public API
// =============================================================================

/**
 * Enable or disable memory history recording.
 *
 * @param enabled Whether to enable recording
 * @param when When to record stack traces (NEVER, STATE, ALLOC, ALL)
 * @param include_cpp_frames Whether to include C++ frames in traces
 * @param max_entries Maximum trace entries in ring buffer
 * @param clear_history Whether to clear existing history
 */
void RecordHistory(bool enabled, RecordContext when = RecordContext::NEVER,
                   bool include_cpp_frames = false, size_t max_entries = 1,
                   bool clear_history = false);

/**
 * Record a trace entry for a memory event.
 *
 * @param action Type of memory event
 * @param device Device index
 * @param addr Memory address
 * @param size Allocation size
 * @param stream NeuronStream ID
 * @param context Optional stack trace (nullptr if not recording context)
 * @param device_free Free memory (only for OOM events)
 */
void RecordTrace(TraceAction action, int device, size_t addr, size_t size, c10::StreamId stream,
                 std::shared_ptr<StackContext> context = nullptr, size_t device_free = 0);

/**
 * Capture current memory snapshot.
 * Note: This is a placeholder - actual implementation requires access to
 * AllocatorRegistry and SimpleMemoryPool internals.
 *
 * @return SnapshotInfo containing segments and device traces
 */
SnapshotInfo snapshot();

/**
 * Check if recording is currently enabled.
 */
bool IsRecordingEnabled();

/**
 * Get current recording context level.
 */
RecordContext GetRecordContext();

/**
 * Check if C++ frames should be included in traces.
 */
bool ShouldIncludeCppFrames();

/**
 * Get current timestamp in microseconds.
 * Reuses GetCurrentTimeNs() from NeuronMetrics and converts to microseconds.
 */
int64_t GetCurrentTimestampUs();

/**
 * Convert TraceAction to string for serialization.
 */
const char* TraceActionToString(TraceAction action);

/**
 * Convert string to RecordContext enum.
 * @throws std::invalid_argument if string is not valid
 */
RecordContext StringToRecordContext(const std::string& str);

}  // namespace NeuronMemorySnapshot
}  // namespace c10_neuron

#endif  // TORCH_NEURON_CSRC_CORE_ALLOCATOR_NEURON_MEMORY_SNAPSHOT_H_
