#pragma once

#include <nrt/nrt_status.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace at::neuron {

// Captures the status export-request-submission - the async version.
// Keep members and values in lock-step with torch_neuronx.profiling.ExportStatus.
enum class ExportStatus {
  Submitted = 1,         // request accepted, will be executed later.
  NoActiveSession = 2,   // no NRT inspect session is active
  ExportNotEnabled = 3,  // async worker unavailable: not requested via
                         // ContinuousExportConfig() at session creation.
  UnsupportedMask = 4,   // mask has bits the async path does not support
};

// Selects what nrt_inspect_export(mask) drains. Values are static_assert'ed against
// NRT's nrt_inspect_export_mask_t (ContinuousExporter.cpp); keep members in lock-step
// with torch_neuronx.profiling.InspectExportMask. When adding a bit, also widen
// kSupportedInspectExportMask if the async path accepts it.
enum class InspectExportMask : uint32_t {
  SystemProfile = 1U,
};

// container for parsed from custom_profiler_config, specific to profile export API.
struct ExportConfig {
  bool enabled{false};  // continuous_export:true — construct + start the async worker
};

// Owns the background export thread that services asynchronous drain requests.
// requestExport() latches a request; the thread calls doExport() and clears the drained
// mask bits AFTER the drain, so same-mask requests landing mid-drain coalesce into it
// (collapse-to-0, no pileup) while bits not covered by the drain stay latched.
// One-shot lifecycle, enforced by the state machine: exactly one start() and one stop()
// per exporter (the Session constructs a fresh exporter per profiling session); requests
// are accepted only while Running; stop() joins (idempotent; dtor calls it).
class ContinuousExporter {
 public:
  ContinuousExporter() = default;
  virtual ~ContinuousExporter();

  ContinuousExporter(const ContinuousExporter&) = delete;
  ContinuousExporter& operator=(const ContinuousExporter&) = delete;
  ContinuousExporter(ContinuousExporter&&) = delete;
  ContinuousExporter& operator=(ContinuousExporter&&) = delete;

  // Spawns the export thread (NotStarted -> Running). Returns false and does nothing
  // unless the exporter is in NotStarted (one-shot: a stopped exporter cannot restart).
  bool start();
  // Running -> Stopped: signals the thread and joins it. A latched-but-unserviced
  // request is NOT drained — the final flush belongs to nrt_inspect_stop().
  // Idempotent for SEQUENTIAL callers (e.g. explicit stop() then dtor); concurrent
  // stop() calls require external serialization. The join runs outside exportMutex_,
  // so two simultaneous stops could double-join. The Session serializes via its mutex_.
  void stop();
  // Non-blocking: latches an async drain request for the export thread. Accepted only
  // while Running (ExportNotEnabled otherwise). Submitted covers both a fresh wakeup
  // and coalescing into an in-flight drain.
  ExportStatus requestExport(uint32_t mask);

 protected:
  // The drain: a passthrough to nrt_inspect_export, reporting NRT's own status
  // verbatim. Overrides may return any NRT_STATUS (or throw); the export loop
  // logs non-success and survives — it never translates or retries.
  //
  // virtual is deliberate — this is the TEST SEAM that lets the export thread run
  // without hardware: tests override it to gate/count drains and to throw (a C++
  // exception cannot safely cross MockNRT's extern "C" boundary, so exception
  // recovery is testable only through here). Do not de-virtualize or inline into
  // the loop even while the body is a one-liner.
  virtual NRT_STATUS doExport(uint32_t mask);

 private:
  // Lifecycle: strictly forward, one-shot. Guarded by exportMutex_.
  enum class State {
    NotStarted,  // constructed; start() not called yet
    Running,     // export thread live; requests accepted
    Stopped,     // stop() ran (or start() was never called); terminal
  };

  void exportLoop();

  // Background thread running exportLoop(); joined by stop().
  std::thread exportThread_;

  // Guards state_, exportRequested_, and pendingMask_; exportCv_ wakes the export
  // thread on a new request or on the Running -> Stopped transition. The blocking
  // drain itself runs with the mutex RELEASED.
  std::mutex exportMutex_;
  std::condition_variable exportCv_;
  State state_{State::NotStarted};
  // Request latch: set by requestExport(), cleared by the export thread only AFTER the
  // drain returns — the ordering that implements collapse-to-0 coalescing.
  bool exportRequested_{false};
  // Accumulated mask bits (OR of all latched requests); drained and cleared with the latch.
  uint32_t pendingMask_{0};
};

// Process-global bridge between the Python bindings and the (single) active session.
// Publish AFTER nrt_inspect_begin succeeds; exporter may be null (session active, async
// export not enabled).
void setActiveExportBridgeForSession(ContinuousExporter* exporter);
// Unpublish (async requests then report NoActiveSession).
void clearActiveExportBridgeForSession();
// Async path: returns a frontend gating status; the eventual drain outcome is log-only.
ExportStatus requestActiveExport(uint32_t mask);
// Sync path: pure NRT passthrough on the caller's thread — returns nrt_inspect_export's
// status verbatim. NRT owns mask validation, the no-session no-op, and serialization
// against begin/stop/other exports (its internal global_lock).
NRT_STATUS exportNow(uint32_t mask);

}  // namespace at::neuron
