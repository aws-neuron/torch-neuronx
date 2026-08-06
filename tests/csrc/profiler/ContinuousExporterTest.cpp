// Unit tests for ContinuousExporter and its process-global bridge (the frontend of
// mid-session profile export, csrc/profiler/ContinuousExporter.{h,cpp}).
//
// Two seams drive the code without hardware:
//   - TestableExporter overrides the virtual doExport() with a gate/counter, for
//     lifecycle, coalescing, and error-recovery tests of the export thread;
//   - MockNRT's strong nrt_inspect_export symbol (gmock), for tests of the REAL
//     doExport()/exportNow() passthroughs.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/profiler/ContinuousExporter.h"

using ::testing::_;
using ::testing::Return;
using namespace std::chrono_literals;

namespace at::neuron {
namespace {

constexpr uint32_t kSystemProfile = static_cast<uint32_t>(InspectExportMask::SystemProfile);
constexpr uint32_t kUnknownMask = 1U << 2;

// Polls cond every 10ms for up to 2s; returns whether it became true.
bool pollUntil(const std::function<bool()>& cond) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!cond() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  return cond();
}

// Re-requests until countReached reports true. A single follow-up request could
// coalesce into an in-flight drain and never produce a new one, so "thread survived"
// tests must keep requesting; each request must be accepted (Submitted).
template <typename Exporter, typename CountReached>
bool requestUntil(Exporter& exporter, const CountReached& countReached) {
  return pollUntil([&] {
    EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
    return countReached();
  });
}

class TestableExporter : public ContinuousExporter {
 public:
  void setBlockExports(bool block) {
    std::lock_guard<std::mutex> lock(mutex_);
    blockExports_ = block;
    cv_.notify_all();
  }

  void setThrowOnExport(bool shouldThrow) {
    std::lock_guard<std::mutex> lock(mutex_);
    throwOnExport_ = shouldThrow;
  }

  bool waitForExportCount(int count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 2s, [&] { return exportCount_ >= count; });
  }

  bool waitForInFlight() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 2s, [&] { return inFlight_; });
  }

  int exportCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exportCount_;
  }

  uint32_t lastMask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastMask_;
  }

 protected:
  NRT_STATUS doExport(uint32_t mask) override {
    bool shouldThrow;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ++exportCount_;
      lastMask_ = mask;
      inFlight_ = true;
      cv_.notify_all();
      cv_.wait(lock, [&] { return !blockExports_; });
      inFlight_ = false;
      shouldThrow = throwOnExport_;
      cv_.notify_all();
    }
    if (shouldThrow) {
      throw std::runtime_error("test export failure");
    }
    return NRT_SUCCESS;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int exportCount_{0};
  uint32_t lastMask_{0};
  bool inFlight_{false};
  bool blockExports_{false};
  bool throwOnExport_{false};
};

class ContinuousExporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
    clearActiveExportBridgeForSession();
  }

  void TearDown() override {
    clearActiveExportBridgeForSession();
    mock_session_.reset();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
};

TEST_F(ContinuousExporterTest, SingleRequestExportsOnce) {
  TestableExporter exporter;
  EXPECT_TRUE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);

  ASSERT_TRUE(exporter.waitForExportCount(1));
  EXPECT_EQ(exporter.exportCount(), 1);
  EXPECT_EQ(exporter.lastMask(), kSystemProfile);
  exporter.stop();
}

TEST_F(ContinuousExporterTest, CoalescesRequestsWhileExportInFlight) {
  // Collapse-to-0: same-mask requests landing while a drain is in flight are
  // satisfied by that drain — they must not queue follow-up drains.
  TestableExporter exporter;
  exporter.setBlockExports(true);
  ASSERT_TRUE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForInFlight());

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  }
  EXPECT_EQ(exporter.exportCount(), 1);
  exporter.setBlockExports(false);

  // Negative assertion: a queued pileup would fire extra drains immediately after
  // the unblock, so a short settle window then an exact count catches it. (A
  // follow-up-request probe cannot work here: it could legitimately coalesce into
  // drain #1's clear window and time out — that IS the feature under test.)
  ASSERT_TRUE(exporter.waitForExportCount(1));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(exporter.exportCount(), 1);
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestAfterCompletedExportFiresAgain) {
  // The thread is reusable: after a drain completes, later requests drain again.
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForExportCount(1));

  // requestUntil, not a single request: waitForExportCount(1) can return while the
  // loop is still between drain #1 and its latch clear, and a lone request landing
  // in that window is (correctly) coalesced into drain #1.
  EXPECT_TRUE(requestUntil(exporter, [&] { return exporter.exportCount() >= 2; }));
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestBeforeStartIsRejected) {
  TestableExporter exporter;
  // Requests are accepted only while Running; the pre-start rejection keeps the
  // one-shot state machine free of latched-before-start bookkeeping.
  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::ExportNotEnabled);

  ASSERT_TRUE(exporter.start());
  EXPECT_EQ(exporter.exportCount(), 0);
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestAfterStopIsRejected) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());
  exporter.stop();

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::ExportNotEnabled);
  EXPECT_EQ(exporter.exportCount(), 0);
}

TEST_F(ContinuousExporterTest, StoppedExporterCannotRestart) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());
  exporter.stop();

  // One-shot: the Session constructs a fresh exporter per session; a stopped
  // exporter stays stopped.
  EXPECT_FALSE(exporter.start());
}

TEST_F(ContinuousExporterTest, StopWaitsForInFlightExport) {
  // stop() joins the thread, so it must block until an in-flight drain returns —
  // Session relies on this: after stop() no async drain can race nrt_inspect_stop().
  TestableExporter exporter;
  exporter.setBlockExports(true);
  ASSERT_TRUE(exporter.start());
  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForInFlight());

  std::atomic<bool> stopped{false};
  std::thread stopper([&] {
    exporter.stop();
    stopped.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(stopped.load(std::memory_order_acquire));
  exporter.setBlockExports(false);
  stopper.join();
  EXPECT_TRUE(stopped.load(std::memory_order_acquire));
  EXPECT_EQ(exporter.exportCount(), 1);
}

TEST_F(ContinuousExporterTest, StopIsIdempotent) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());

  exporter.stop();
  exporter.stop();
}

TEST_F(ContinuousExporterTest, DoubleStartDoesNotSpawnSecondThread) {
  TestableExporter exporter;
  EXPECT_TRUE(exporter.start());
  EXPECT_FALSE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForExportCount(1));
  EXPECT_EQ(exporter.exportCount(), 1);
  exporter.stop();
}

TEST_F(ContinuousExporterTest, ThrowingExportDoesNotKillThread) {
  // A doExport() that THROWS is caught by the loop (an escaping exception from a
  // thread body would std::terminate the process); the thread must keep servicing.
  // The non-success-RETURN recovery path is covered on the real doExport by
  // RuntimeRejectedExportDoesNotKillThread below.
  TestableExporter exporter;
  exporter.setThrowOnExport(true);
  ASSERT_TRUE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForExportCount(1));

  exporter.setThrowOnExport(false);
  EXPECT_TRUE(requestUntil(exporter, [&] { return exporter.exportCount() >= 2; }));
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestExportRejectsUnsupportedMaskSynchronously) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());

  // The drain outcome is log-only, so a bad mask must be rejected at requestExport()
  // time rather than acknowledged and then rejected by NRT in the background.
  EXPECT_EQ(exporter.requestExport(kUnknownMask), ExportStatus::UnsupportedMask);
  EXPECT_EQ(exporter.requestExport(0), ExportStatus::UnsupportedMask);

  // Deterministic negative proof: a valid follow-up request drains exactly once with
  // only its own mask — a latched rejected mask would show up as an extra drain or a
  // kUnknownMask bit in lastMask().
  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForExportCount(1));
  EXPECT_EQ(exporter.exportCount(), 1);
  EXPECT_EQ(exporter.lastMask(), kSystemProfile);
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestBridgeForwardsToActiveExporter) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());
  setActiveExportBridgeForSession(&exporter);

  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForExportCount(1));
  exporter.stop();
}

TEST_F(ContinuousExporterTest, RequestBridgeReportsInactiveAndMissingExporterStates) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());

  clearActiveExportBridgeForSession();
  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::NoActiveSession);

  setActiveExportBridgeForSession(nullptr);
  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::ExportNotEnabled);

  exporter.stop();
}

TEST_F(ContinuousExporterTest, ConcurrentRequestsRaceWithSlowExportAndStop) {
  // Stress: 8 requester threads hammer the bridge while one drain is held in flight
  // and stop() races in. Hunts crashes/deadlocks/lost-wakeups in the latch + state
  // machine; run with --runs_per_test=N for flake-hunting.
  TestableExporter exporter;
  exporter.setBlockExports(true);
  ASSERT_TRUE(exporter.start());
  setActiveExportBridgeForSession(&exporter);

  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(exporter.waitForInFlight());

  std::atomic<bool> keepRequesting{true};
  std::atomic<int> submitted{0};
  std::atomic<int> exportNotEnabled{0};
  std::atomic<int> unexpected{0};
  std::vector<std::thread> requesters;
  for (int i = 0; i < 8; ++i) {
    requesters.emplace_back([&] {
      while (keepRequesting.load(std::memory_order_acquire)) {
        ExportStatus status = requestActiveExport(kSystemProfile);
        if (status == ExportStatus::Submitted) {
          submitted.fetch_add(1, std::memory_order_relaxed);
        } else if (status == ExportStatus::ExportNotEnabled) {
          exportNotEnabled.fetch_add(1, std::memory_order_relaxed);
        } else {
          unexpected.fetch_add(1, std::memory_order_relaxed);
        }
        std::this_thread::yield();
      }
    });
  }

  std::this_thread::sleep_for(50ms);
  std::thread stopper([&] { exporter.stop(); });
  std::this_thread::sleep_for(50ms);
  exporter.setBlockExports(false);
  stopper.join();

  keepRequesting.store(false, std::memory_order_release);
  for (auto& requester : requesters) {
    requester.join();
  }
  clearActiveExportBridgeForSession();

  // Requests observe only the two legal statuses for this interleaving: Submitted
  // (before stop landed) or ExportNotEnabled (after). Everything coalesced into the
  // single blocked drain, and stop-wins means no drain ran after it: count stays 1.
  EXPECT_EQ(unexpected.load(std::memory_order_relaxed), 0);
  EXPECT_GT(submitted.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(exporter.exportCount(), 1);
}

TEST_F(ContinuousExporterTest, ExportNowIsAPurePassthroughWithoutSessionOrExporter) {
  // No bridge state published at all: export_now does not gate on the session
  // (NRT no-ops safely) and does not need the async worker.
  EXPECT_CALL(*mock_nrt_, nrt_inspect_export(kSystemProfile)).WillOnce(Return(NRT_SUCCESS));
  EXPECT_EQ(exportNow(kSystemProfile), NRT_SUCCESS);
}

TEST_F(ContinuousExporterTest, ExportNowReturnsNrtStatusVerbatim) {
  EXPECT_CALL(*mock_nrt_, nrt_inspect_export(kSystemProfile)).WillOnce(Return(NRT_FAILURE));
  EXPECT_EQ(exportNow(kSystemProfile), NRT_FAILURE);

  // Sync path forwards unsupported masks verbatim; Runtime owns validation.
  EXPECT_CALL(*mock_nrt_, nrt_inspect_export(kUnknownMask)).WillOnce(Return(NRT_INVALID));
  EXPECT_EQ(exportNow(kUnknownMask), NRT_INVALID);
}

TEST_F(ContinuousExporterTest, RuntimeRejectedExportDoesNotKillThread) {
  // Real (non-testable) exporter: NRT rejecting the drain (NRT_INVALID) is
  // log-only and must leave the thread alive for later requests.
  std::atomic<int> calls{0};
  EXPECT_CALL(*mock_nrt_, nrt_inspect_export(kSystemProfile))
      .WillOnce([&](uint32_t) {
        calls.fetch_add(1, std::memory_order_relaxed);
        return NRT_INVALID;
      })
      .WillRepeatedly([&](uint32_t) {
        calls.fetch_add(1, std::memory_order_relaxed);
        return NRT_SUCCESS;
      });

  ContinuousExporter exporter;
  ASSERT_TRUE(exporter.start());

  EXPECT_EQ(exporter.requestExport(kSystemProfile), ExportStatus::Submitted);
  ASSERT_TRUE(pollUntil([&] { return calls.load(std::memory_order_relaxed) >= 1; }));

  EXPECT_TRUE(requestUntil(exporter, [&] { return calls.load(std::memory_order_relaxed) >= 2; }));
  exporter.stop();
}

TEST_F(ContinuousExporterTest, SyncExportDoesNotBlockConcurrentRequestOrTeardown) {
  TestableExporter exporter;
  ASSERT_TRUE(exporter.start());
  setActiveExportBridgeForSession(&exporter);

  // Gate the NRT call so the sync drain can be held "in flight" mid-call: the mock
  // signals drainStarted, then blocks until the test fulfils release.
  std::promise<void> drainStarted;
  std::promise<void> release;
  auto drainStartedFuture = drainStarted.get_future();
  auto releaseFuture = release.get_future().share();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_export(kSystemProfile)).WillOnce([&](uint32_t) {
    drainStarted.set_value();
    releaseFuture.wait();
    return NRT_SUCCESS;
  });

  // Run a synchronous export on a background thread; it parks inside the gated
  // NRT call, holding the export "in flight" WITHOUT holding g_exporterMutex
  // (exportNow is a lock-free passthrough).
  std::thread syncExporter([] { EXPECT_EQ(exportNow(kSystemProfile), NRT_SUCCESS); });
  ASSERT_EQ(drainStartedFuture.wait_for(2s), std::future_status::ready);

  // With the sync drain in flight, the async request path must remain responsive
  // (a passthrough holding g_exporterMutex through the drain would block this call).
  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::Submitted);

  // Bridge teardown must NOT wait on the in-flight sync drain: ordering against
  // nrt_inspect_stop() is NRT's job (global_lock), not the frontend's.
  clearActiveExportBridgeForSession();
  EXPECT_EQ(requestActiveExport(kSystemProfile), ExportStatus::NoActiveSession);

  release.set_value();
  syncExporter.join();
  exporter.stop();
}

}  // namespace
}  // namespace at::neuron
