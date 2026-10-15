// Mid-session system and device export through nrt_inspect_export.
//
// The new runtime API supports mid-session exports. End-of-session exports are the default.
// `ContinuousExporter` makes that accessible to the python layer.
// Two flavors:
//   - exportNow (sync):        caller's thread blocks on NRT function call.
//                              Returns the NRT_STATUS verbatim.
//   - requestActiveExport (async): Non-blocking and fire-and-forget. Returns after submitting
//                                  an export request which background thread full-fills
//                                  later on. The return errors out if the submission
//                                  validation fails. Export failure does not crash the
//                                  thread but only logs the error.
// Teardown invariant (Session::stop()/dtor): clear the bridge -> stop the exporter
// (joins the async drain) -> nrt_inspect_stop() runs as the session's FINAL drain.
// Sync exportNow drains need no frontend fence: NRT's global_lock (wrlock in both
// nrt_inspect_export and nrt_inspect_stop) serializes them, and an export landing
// after stop drains torn-down rings -- a benign NRT_SUCCESS no-op that writes nothing.
#include "ContinuousExporter.h"

#include <nrt/nrt_profile.h>

#include <exception>
#include <mutex>
#include <type_traits>

#include "LogWrapper.h"

namespace at::neuron {

namespace {

// Mask bits the ASYNC path accepts today (the sync path forwards masks verbatim and
// lets Runtime validate them). Pinned to NRT's contract by the static_assert below.
// When InspectExportMask grows, add a per-member assert: with several bits, an
// OR-equality alone cannot catch swapped values, and the async-supported set may
// deliberately lag the enum.
constexpr uint32_t kSupportedInspectExportMask = static_cast<uint32_t>(InspectExportMask::All);

static_assert(static_cast<uint32_t>(InspectExportMask::SystemProfile) ==
                  NRT_INSPECT_EXPORT_MASK_SYSTEM_PROFILE,
              "InspectExportMask::SystemProfile diverged from nrt_inspect_export_mask_t");
static_assert(static_cast<uint32_t>(InspectExportMask::DeviceProfile) ==
                  NRT_INSPECT_EXPORT_MASK_DEVICE_PROFILE,
              "InspectExportMask::DeviceProfile diverged from nrt_inspect_export_mask_t");
static_assert(kSupportedInspectExportMask == NRT_INSPECT_EXPORT_MASK_ALL,
              "InspectExportMask::All diverged from nrt_inspect_export_mask_t");

// The bridge: process-global rendezvous between requestActiveExport() -- reached from
// the module-level Python API via its pybind lambda, which holds no session handle
// (kineto owns the session) -- and the single active profiler session.
// Session::start() publishes; stop()/dtor clears. At most one profile session is active
// process-wide (s_sessionActive in Session.cpp), so one slot suffices.
struct ExportBridgeState {
  bool sessionActive{false};                    // an NRT inspect session is live
  ContinuousExporter* activeExporter{nullptr};  // non-null iff continuous_export opted in
};

// Guards g_exportBridge.
std::mutex g_exporterMutex;

ExportBridgeState g_exportBridge;

}  // namespace

// Async requests and configured policies share the same supported mask set.
bool isSupportedInspectExportMask(const uint32_t mask) {
  return mask != 0 && (mask & ~kSupportedInspectExportMask) == 0;
}

TimeExportPolicy::TimeExportPolicy(const TimeExportConfig& config)
    : interval_(config.interval), mask_(config.mask) {}

void TimeExportPolicy::start(const TimePoint& now) {
  nextDeadline_.reset();
  if (interval_ <= std::chrono::nanoseconds::zero() || now > TimePoint::max() - interval_) {
    NPROF_WARN("Time export policy disabled: interval or first deadline is out of range");
    return;
  }
  nextDeadline_ = now + interval_;
}

// Advance the timer on its original schedule and update pending requests.
uint32_t TimeExportPolicy::updateScheduleAndPendingMask(const TimePoint& now,
                                                        const uint32_t pendingMask,
                                                        const uint32_t completedMask) {
  uint32_t requests = pendingMask;
  if (nextDeadline_ && now >= *nextDeadline_) {
    // Coalesce all elapsed ticks into one request for the policy mask.
    requests |= mask_;

    // Unsigned subtraction represents even a span across the full signed clock range.
    using Tick = std::chrono::nanoseconds::rep;
    using UnsignedTick = std::make_unsigned_t<Tick>;
    const auto elapsed = static_cast<UnsignedTick>(now.time_since_epoch().count()) -
                         static_cast<UnsignedTick>(nextDeadline_->time_since_epoch().count());
    const auto period = static_cast<UnsignedTick>(interval_.count());
    // Skip missed ticks while preserving the original schedule.
    // For ticks at 100, 200, 300 ms, now=350 ms advances the deadline to 400 ms.
    const auto delay = std::chrono::nanoseconds(static_cast<Tick>(period - elapsed % period));
    if (now > TimePoint::max() - delay) {
      // Prevent a wrapped deadline from causing an immediate export loop.
      nextDeadline_.reset();
      NPROF_WARN("Time export policy disabled: next deadline is out of range");
    } else {
      nextDeadline_ = now + delay;
    }
  }
  // Clear attempted bits last to coalesce requests received during the export.
  // Other bits remain pending. completedMask records attempted bits regardless of success.
  return requests & ~completedMask;
}

ContinuousExporter::ContinuousExporter(const ExportConfig& config) {
  if (config.timePolicy) {
    if (isSupportedInspectExportMask(config.timePolicy->mask)) {
      timePolicy_.emplace(*config.timePolicy);
    } else {
      NPROF_WARN("Time export policy disabled: unsupported export mask");
    }
  }
}

ContinuousExporter::~ContinuousExporter() { stop(); }

bool ContinuousExporter::start() {
  std::lock_guard<std::mutex> lock(exportMutex_);
  // One-shot: only NotStarted -> Running spawns the thread; a stopped exporter
  // stays stopped (the Session constructs a fresh exporter per profiling session).
  if (state_ != State::NotStarted) {
    return false;
  }
  if (timePolicy_) {
    timePolicy_->start(
        std::chrono::time_point_cast<std::chrono::nanoseconds>(TimeExportPolicy::Clock::now()));
  }
  state_ = State::Running;
  exportThread_ = std::thread(&ContinuousExporter::exportLoop, this);
  return true;
}

void ContinuousExporter::stop() {
  {
    std::lock_guard<std::mutex> lock(exportMutex_);
    state_ = State::Stopped;
  }
  exportCv_.notify_one();

  // Idempotent: a second stop() (e.g. explicit stop then dtor) finds nothing to join.
  if (!exportThread_.joinable()) {
    return;
  }

  exportThread_.join();
}

ExportStatus ContinuousExporter::requestExport(uint32_t mask) {
  // The drain runs later on the export thread, so the caller can only be told about
  // frontend gates that are checkable right now; the mask is validated here because
  // NRT's verdict during the background drain would otherwise be log-only.
  if (!isSupportedInspectExportMask(mask)) {
    return ExportStatus::UnsupportedMask;
  }

  {
    std::lock_guard<std::mutex> lock(exportMutex_);
    // Requests are accepted only while the worker can service them.
    if (state_ != State::Running) {
      return ExportStatus::ExportNotEnabled;
    }

    exportRequested_ = true;
    pendingMask_ |= mask;
  }
  exportCv_.notify_one();
  return ExportStatus::Submitted;
}

NRT_STATUS ContinuousExporter::doExport(uint32_t mask) { return nrt_inspect_export(mask); }

void ContinuousExporter::updatePendingRequests(const uint32_t completedMask) {
  pendingMask_ = timePolicy_ ? timePolicy_->updateScheduleAndPendingMask(
                                   std::chrono::time_point_cast<std::chrono::nanoseconds>(
                                       TimeExportPolicy::Clock::now()),
                                   pendingMask_, completedMask)
                             : pendingMask_ & ~completedMask;
  exportRequested_ = pendingMask_ != 0;
}

// Wait for an explicit request, a timer deadline, or stop. Both request sources use one worker.
// The latch is deliberately cleared AFTER the drain -- requests arriving mid-drain are
// considered satisfied by it and coalesce to zero extra drains.
void ContinuousExporter::exportLoop() {
  while (true) {
    uint32_t mask = 0;
    {
      std::unique_lock<std::mutex> lock(exportMutex_);
      const auto ready = [this]() { return exportRequested_ || state_ == State::Stopped; };
      if (timePolicy_ && timePolicy_->nextDeadline()) {
        exportCv_.wait_until(lock, *timePolicy_->nextDeadline(), ready);
      } else {
        exportCv_.wait(lock, ready);
      }
      // Stop wins over a pending request: exit WITHOUT draining -- the final flush
      // belongs to nrt_inspect_stop(), which Session runs right after joining us.
      if (state_ == State::Stopped) {
        break;
      }
      updatePendingRequests();
      if (!exportRequested_) {
        continue;
      }
      // Snapshot the mask; the drain below runs with the mutex released so
      // requestExport() stays non-blocking while we are inside NRT.
      mask = pendingMask_;
    }

    // The async caller was already answered at requestExport() time; the drain's outcome
    // is log-only. Warn with NRT's verbatim errno so failures are diagnosable from logs.
    // An escaping exception would std::terminate the process (thread body), so swallow:
    // the profiler is an observer -- the workload must outlive its failures.
    try {
      NRT_STATUS status = doExport(mask);
      if (status != NRT_SUCCESS) {
        NPROF_WARN("nrt_inspect_export failed (status:", status, ") - continuing");
      }
    } catch (const std::exception& e) {
      NPROF_WARN("nrt_inspect_export threw:", e.what(), "- continuing");
    } catch (...) {
      NPROF_WARN("nrt_inspect_export threw unknown exception - continuing");
    }

    // Clear ONLY the drained bits, AFTER the drain (collapse-to-0 per mask bit).
    // Clearing before the drain would let a mid-drain request immediately re-fire --
    // a perpetual export loop under load. Bits latched mid-drain for masks NOT in
    // this drain's snapshot survive and re-arm the latch, so their Submitted
    // acknowledgement is honored with a follow-up drain.
    {
      std::lock_guard<std::mutex> lock(exportMutex_);
      updatePendingRequests(mask);
    }
  }
}

// Publish the session to the bridge (called by Session::start() after
// nrt_inspect_begin succeeds). exporter is null when continuous_export was not
// requested and async requests then report ExportNotEnabled instead of NoActiveSession.
void setActiveExportBridgeForSession(ContinuousExporter* exporter) {
  std::lock_guard<std::mutex> lock(g_exporterMutex);
  g_exportBridge.sessionActive = true;
  g_exportBridge.activeExporter = exporter;
}

// Unpublish: async requests report NoActiveSession afterwards. Sync exportNow needs
// no rendezvous here -- NRT's global_lock orders any in-flight drain against the
// caller's upcoming nrt_inspect_stop(), and a drain landing after stop is a benign
// NRT_SUCCESS no-op on torn-down rings.
void clearActiveExportBridgeForSession() {
  std::lock_guard<std::mutex> lock(g_exporterMutex);
  g_exportBridge = ExportBridgeState{};
}

// Async entry point (Python request_export): frontend gating checks, then submit a
// request on the active exporter. Holding g_exporterMutex through requestExport() is
// safe -- it only sets a flag under the exporter's own small mutex, never blocks --
// and it pins the exporter alive for the call (teardown clears the bridge under this
// same mutex before destroying the exporter).
ExportStatus requestActiveExport(uint32_t mask) {
  std::lock_guard<std::mutex> lock(g_exporterMutex);
  if (!g_exportBridge.sessionActive) {
    return ExportStatus::NoActiveSession;
  }
  if (!g_exportBridge.activeExporter) {
    return ExportStatus::ExportNotEnabled;
  }
  return g_exportBridge.activeExporter->requestExport(mask);
}

// Sync entry point (Python export_now): a pure NRT passthrough on the caller's
// thread. No session gate, no mask pre-check, no frontend synchronization --
// Runtime owns all of it: nrt_inspect_export validates the mask (NRT_INVALID),
// no-ops safely with no session, and serializes against begin/stop/other exports
// on its own global_lock.
NRT_STATUS exportNow(uint32_t mask) { return nrt_inspect_export(mask); }

}  // namespace at::neuron
