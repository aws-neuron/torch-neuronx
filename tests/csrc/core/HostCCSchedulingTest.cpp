// Unit tests for the two cross-kind sync gates wiring kHost into per-stream FIFO:
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace {

// Closure that blocks in RunHost() on a shared release_future until the test
// calls Release(), so the worker can't race the test to advance completed_seq.
// Bounded wait (not forever) so a forgotten release fails the test, not hangs.
HostKernelExecution::HostFn MakeBlockingFn(std::shared_future<void> release) {
  return [release = std::move(release)](const std::vector<TensorDataRef>&,
                                        const std::vector<TensorDataRef>&) {
    release.wait_for(std::chrono::seconds(10));
  };
}

// Per-op state: holds both the test-controlled release gate AND the
// OperationContext, so the test fixture can release workers after assertions.
struct BlockingHostOp {
  std::shared_ptr<OperationContext> op;
  std::shared_ptr<std::promise<void>> release_promise;

  void Release() {
    try {
      release_promise->set_value();
    } catch (const std::future_error&) {
      // Already released (shutdown drain path) — ignore.
    }
  }
};

// Build a blocking kHost OperationContext: once submitted, the worker enters
// RunHost() and blocks until the test calls Release().
BlockingHostOp MakeBlockingHostOp(StreamImpl* stream, const std::string& name = "host") {
  auto release_promise = std::make_shared<std::promise<void>>();
  std::shared_future<void> release_future = release_promise->get_future().share();
  auto kernel = std::make_unique<HostKernelExecution>(
      name, /*src_refs=*/std::vector<TensorDataRef>{}, /*dst_refs=*/std::vector<TensorDataRef>{},
      MakeBlockingFn(std::move(release_future)), /*device_id=*/0);
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->stream_id = stream->stream_id;
  op->MarkExecutionReady();
  return BlockingHostOp{std::move(op), std::move(release_promise)};
}

// Closure that sleeps — used by the multi-worker test to force completion order
// to differ from submission order.
HostKernelExecution::HostFn MakeSleepFn(int sleep_ms) {
  return [sleep_ms](const std::vector<TensorDataRef>&, const std::vector<TensorDataRef>&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  };
}

// kHost OperationContext for negative-path tests (never submitted). Pre-released
// closure so an accidental Submit doesn't hang the runner.
std::shared_ptr<OperationContext> MakeHostOp(StreamImpl* stream, const std::string& name = "host") {
  auto p = std::make_shared<std::promise<void>>();
  p->set_value();  // pre-released
  auto kernel = std::make_unique<HostKernelExecution>(
      name, /*src_refs=*/std::vector<TensorDataRef>{}, /*dst_refs=*/std::vector<TensorDataRef>{},
      MakeBlockingFn(p->get_future().share()), /*device_id=*/0);
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->stream_id = stream->stream_id;
  op->MarkExecutionReady();
  return op;
}

// Build a kHLO device OperationContext with a Schedule() that immediately
// returns a synthetic sequence id. The runtime worker calls Schedule via
// StreamImpl::ScheduleDeviceOperation on the device-op path.
std::shared_ptr<OperationContext> MakeDeviceOp(StreamImpl* stream, nrt::SequenceId seq_to_assign,
                                               const std::string& name = "device") {
  auto kernel = std::make_unique<MockNeuronKernelExecution>(name);
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(KernelTypeEnum::kHLO));
  ON_CALL(*kernel, RequiresCompilation()).WillByDefault(Return(false));
  ON_CALL(*kernel, ValidateImpl()).WillByDefault(Return(true));
  ON_CALL(*kernel, Schedule(_, _, _))
      .WillByDefault(
          DoAll(SetArgPointee<0>(NRT_SUCCESS), SetArgPointee<1>(seq_to_assign), Return(true)));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->stream = stream;
  op->stream_id = stream->stream_id;
  op->MarkExecutionReady();
  return op;
}

}  // namespace

class HostCCSchedulingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // NRT mock setup — the engine's ctor and the gate's UpdateCompletionState
    // call into nrta_get_sequence / nrt_get_visible_vnc_count.
    nrt_session_ = std::make_unique<MockNRTSession>();
    auto* nrt_mock = MockNRT::GetInstance();

    EXPECT_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*nrt_mock, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    // Default: the prior device seq is retired. Tests that need the
    // "still inflight" path override with a per-test EXPECT_CALL.
    EXPECT_CALL(*nrt_mock, nrta_get_sequence(_, _, _, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<3>(UINT64_MAX), Return(NRT_SUCCESS)));

    c10_neuron::reset_vnc_count();
    InitializeStreamPools();

    compilation_cache_ = std::make_unique<CompilationCache>();
    model_handle_cache_ = std::make_unique<ModelHandleCache>();
    engine_ = std::make_unique<MockOperationExecutionEngine>(compilation_cache_.get(),
                                                             model_handle_cache_.get());
    stream_ = std::make_shared<StreamImpl>(/*device=*/0, /*stream_id=*/1);
  }

  void TearDown() override {
    if (engine_) {
      // Poll the pending counter to zero before asserting: the WorkerLoop
      // advances completed_seq before on_complete_ → DrainHostCompletions →
      // fetch_sub, so a test waking on the seq can briefly see a non-zero
      // counter. A timeout here is a real leak (op stuck / counter asymmetry).
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (engine_->pending_async_ops_count_.load() > 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      EXPECT_EQ(engine_->pending_async_ops_count_.load(), 0)
          << "Test left pending kHost ops in flight after 10s; BlockingHostOp tests must "
             "Release() and WaitHostSeqReached() before TearDown.";
      engine_->Shutdown();
      // Sweep orphaned ops before stream destruction. Tests that inject directly
      // into runtime_queue_ (bypassing SubmitOperation) leave ops the engine
      // never removes; ~StreamImpl → Synchronize() would block forever on them.
      // Must run before engine_.reset() — clearing goes through the engine.
      // Safe unconditionally: engine is shut down, and engine-driven tests have
      // already drained (queue empty → no-op).
      if (stream_) {
        engine_->ClearRuntimeQueue(stream_.get());
      }
      engine_.reset();
    }
    stream_.reset();
    CleanupStreamPools();
    c10_neuron::reset_vnc_count();
  }

  // Wait until the runtime worker has fully drained kHost completions and is
  // done dereferencing each op, so the test can safely drop OperationContexts.
  // Polling the pending counter alone is NOT enough: DrainHostCompletions does
  // fetch_sub(1) BEFORE CompleteOperation, which dereferences *op for tens of µs
  // after. The real barrier is the runtime queue being empty —
  // RemoveActiveOperation is CompleteOperation's last line.
  // Tests must call this before clearing the queue or dropping ops.
  void DrainEngineHostCompletions(std::chrono::seconds timeout = std::chrono::seconds(2)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      // Counter==0 is the fast-path gate; the empty-queue check is the real barrier.
      if (engine_->pending_async_ops_count_.load() == 0 &&
          engine_->RuntimeQueueSize(stream_.get()) == 0) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Poll until completed_seq reaches `seq` (a submitted kHost op's stamped
  // value), failing on timeout. Reaching the seq does NOT imply the runtime
  // worker drained the completion queue — call DrainEngineHostCompletions()
  // after this before touching the runtime queue.
  void WaitHostSeqReached(HostCompletionSeq seq,
                          std::chrono::seconds timeout = std::chrono::seconds(2)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (stream_->host_completion_state_.completed_seq_id.load(std::memory_order_acquire) < seq &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(stream_->host_completion_state_.completed_seq_id.load(std::memory_order_acquire), seq)
        << "host completion seq did not reach " << seq << " within " << timeout.count() << "s";
  }

  std::unique_ptr<MockNRTSession> nrt_session_;
  std::unique_ptr<CompilationCache> compilation_cache_;
  std::unique_ptr<ModelHandleCache> model_handle_cache_;
  std::unique_ptr<MockOperationExecutionEngine> engine_;
  std::shared_ptr<StreamImpl> stream_;
};

// ── HasPendingAsyncOperations reflects is_host_inflight ─────────────────────
// Without this, the engine's drain logic would think the stream is idle while
// a kHost op is still running on the worker pool — causing premature
// shutdown / wrong synchronization.
TEST_F(HostCCSchedulingTest, HasPendingAsyncOperationsReflectsHostInflight) {
  auto op = MakeHostOp(stream_.get(), "host");

  engine_->InjectIntoRuntimeQueue(stream_.get(), op);

  // Initially: not yet submitted to host pool — neither is_scheduled nor
  // is_host_inflight is set, so HasPendingAsyncOperations is false.
  EXPECT_FALSE(stream_->HasPendingAsyncOperations());

  // Use release-store to match the production write site in
  // OperationExecutionEngine::ScheduleOperation kHost path.
  op->nrt_async_state_.is_host_inflight.store(true, std::memory_order_release);
  EXPECT_TRUE(stream_->HasPendingAsyncOperations());

  // Reset clears the flag.
  op->nrt_async_state_.Reset();
  EXPECT_FALSE(stream_->HasPendingAsyncOperations());
}

// ── ScheduleOperation: kHost gates on prior device-XU retire (NOT retired) ──
// Setup: stream's last_scheduled_info_ marks a kDeviceXU op with seq=10. NRT
// reports last completed seq < 10. The kHost op is being considered for
// scheduling — the engine should return false (try again later) and NOT
// submit it to the HostWorkerPool.
TEST_F(HostCCSchedulingTest, ScheduleHostGatesOnPriorDeviceSeqNotRetired) {
  // Override the default "fully retired" seq mock with one that returns a
  // value smaller than the last_scheduled_info_.sequence_id we set.
  auto* nrt_mock = MockNRT::GetInstance();
  ::testing::Mock::VerifyAndClearExpectations(nrt_mock);
  EXPECT_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
      .Times(::testing::AnyNumber())
      .WillRepeatedly(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));
  EXPECT_CALL(*nrt_mock, nrta_get_sequence(_, _, _, _))
      .Times(::testing::AnyNumber())
      .WillRepeatedly(DoAll(SetArgPointee<3>(static_cast<nrta_seq_t>(5)), Return(NRT_SUCCESS)));

  // Mark a prior device op on this stream with seq=10 — newer than the
  // 5 the mock reports as completed.
  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kDeviceXU;
  info.xu = NRTA_XU_COMPUTE;
  info.sequence_id = 10;
  info.valid = true;

  auto host_op = MakeHostOp(stream_.get(), "gated_host");
  engine_->InjectIntoRuntimeQueue(stream_.get(), host_op);
  engine_->SetRuntimeCursorTo(stream_.get(), host_op.get());

  // Gate must reject: prior device seq not retired.
  EXPECT_FALSE(engine_->ScheduleOperation(host_op.get(), stream_.get()));

  // Verify no engine bookkeeping happened — host op must NOT have been
  // submitted to the pool.
  EXPECT_FALSE(host_op->nrt_async_state_.is_host_inflight.load());
  EXPECT_EQ(engine_->pending_async_ops_count_.load(), 0);
  // last_scheduled_info_ must still describe the prior device op (gate didn't
  // overwrite it).
  EXPECT_EQ(info.kind, LastKind::kDeviceXU);
  EXPECT_EQ(info.sequence_id, 10u);

  // Cleanup
  engine_->ClearRuntimeQueue(stream_.get());
}

// ── ScheduleOperation: kHost passes once prior device-XU retired ────────────
// Same setup, but NRT now reports the prior seq as completed. Engine should
// submit to the pool, advance bookkeeping, and stamp last_scheduled_info_ to
// kHost (with a host completion seq) so subsequent device ops gate on it.
//
// Uses a blocking kernel so the worker thread that picks up the submitted
// op cannot race the test to advance the stream's completed_seq. The test
// releases the gate AFTER it has finished asserting on the not-ready state.
TEST_F(HostCCSchedulingTest, ScheduleHostProceedsOncePriorDeviceRetired) {
  // Default mock returns UINT64_MAX as last completed → any seq we set is
  // retired.
  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kDeviceXU;
  info.xu = NRTA_XU_COMPUTE;
  info.sequence_id = 10;
  info.valid = true;

  BlockingHostOp blocking = MakeBlockingHostOp(stream_.get(), "passes_host");
  auto host_op = blocking.op;
  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), host_op);
    engine_->SetRuntimeCursorTo(stream_.get(), host_op.get());
  }

  EXPECT_TRUE(engine_->ScheduleOperation(host_op.get(), stream_.get()));

  // is_host_inflight set, pending counter bumped, last_scheduled_info_ now
  // carries a host completion seq (so a downstream device op will gate).
  EXPECT_TRUE(host_op->nrt_async_state_.is_host_inflight.load());
  EXPECT_EQ(engine_->pending_async_ops_count_.load(), 1);
  EXPECT_EQ(info.kind, LastKind::kHost);
  EXPECT_GT(info.host_completion_seq, 0u);
  EXPECT_EQ(info.host_completion_seq, host_op->host_completion_seq_);

  // completed_seq has NOT reached the op's seq — the worker is blocked in the
  // closure waiting on our release gate.
  EXPECT_LT(stream_->host_completion_state_.completed_seq_id.load(std::memory_order_acquire),
            host_op->host_completion_seq_);

  // Now release the worker so it can finish RunHost() and advance completed_seq
  // (via WorkerLoop's AdvanceHostCompletion). DrainHostCompletions on TearDown's
  // engine_->Shutdown() will sweep the completion through CompleteOperation.
  blocking.Release();

  // Seq-reached ≠ completion drained: wait the seq, then drain so the runtime
  // worker flushes the completion before we touch the runtime queue (else UAF
  // on a still-queued OperationContext* at test exit).
  WaitHostSeqReached(host_op->host_completion_seq_);
  DrainEngineHostCompletions();

  engine_->ClearRuntimeQueue(stream_.get());
}

// ── ScheduleOperation: kHost with no prior op is unconditionally scheduled ──
// Empty last_scheduled_info_ → no gate → straight submit.
TEST_F(HostCCSchedulingTest, ScheduleHostNoPriorOpProceeds) {
  auto& info = stream_->GetLastScheduledInfo();
  EXPECT_FALSE(info.valid);

  BlockingHostOp blocking = MakeBlockingHostOp(stream_.get(), "first_host");
  auto host_op = blocking.op;
  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), host_op);
    engine_->SetRuntimeCursorTo(stream_.get(), host_op.get());
  }

  EXPECT_TRUE(engine_->ScheduleOperation(host_op.get(), stream_.get()));
  EXPECT_TRUE(host_op->nrt_async_state_.is_host_inflight.load());
  EXPECT_EQ(info.kind, LastKind::kHost);
  EXPECT_GT(info.host_completion_seq, 0u);
  EXPECT_TRUE(info.valid);

  blocking.Release();
  WaitHostSeqReached(host_op->host_completion_seq_);
  DrainEngineHostCompletions();

  engine_->ClearRuntimeQueue(stream_.get());
}

// ── ScheduleDeviceOperation: device-after-kHost gate (host not done) ────────
// Setup: stream's last_scheduled_info_ marks a kHost with host_completion_seq=5
// while the stream's completed_seq is still 0 (default). The gate must return
// kNotReady and NOT invoke kernel Schedule.
TEST_F(HostCCSchedulingTest, ScheduleDeviceGatesOnHostSeqNotComplete) {
  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kHost;
  info.host_completion_seq = 5;
  info.valid = true;
  // completed_seq defaults to 0, so 5 > 0 → not complete.
  ASSERT_LT(stream_->host_completion_state_.completed_seq_id.load(), info.host_completion_seq);

  auto device_op = MakeDeviceOp(stream_.get(), /*seq=*/1, "gated_dev");
  // Schedule must NOT be called — the gate aborts before it.
  EXPECT_CALL(static_cast<MockNeuronKernelExecution&>(*device_op->kernel_execution),
              Schedule(_, _, _))
      .Times(0);

  ScheduleResult result = stream_->ScheduleDeviceOperation(device_op.get(), KernelTypeEnum::kHLO);
  EXPECT_EQ(result, ScheduleResult::kNotReady);
  EXPECT_FALSE(device_op->nrt_async_state_.is_scheduled);
  // last_scheduled_info_ must NOT have been overwritten.
  EXPECT_EQ(info.kind, LastKind::kHost);
  EXPECT_EQ(info.host_completion_seq, 5u);
}

// ── ScheduleDeviceOperation: device-after-kHost passes once host done ───────
// Same setup but the stream's completed_seq has reached the kHost's
// host_completion_seq when the gate runs → device op proceeds, NRT Schedule is
// called, and last_scheduled_info_ flips to kDeviceXU (the kind flip is the
// staleness guard — no later read can mis-interpret the stale host seq).
TEST_F(HostCCSchedulingTest, ScheduleDeviceProceedsOnceHostSeqComplete) {
  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kHost;
  info.host_completion_seq = 5;
  info.valid = true;
  // completed_seq >= 5 → host done.
  stream_->host_completion_state_.completed_seq_id.store(5, std::memory_order_release);

  auto device_op = MakeDeviceOp(stream_.get(), /*seq=*/42, "passes_dev");
  // Place op into the runtime queue so AdvanceToNextOperation has something
  // to walk past.
  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), device_op);
    engine_->SetRuntimeCursorTo(stream_.get(), device_op.get());
  }

  ScheduleResult result = stream_->ScheduleDeviceOperation(device_op.get(), KernelTypeEnum::kHLO);
  EXPECT_EQ(result, ScheduleResult::kScheduled);
  EXPECT_TRUE(device_op->nrt_async_state_.is_scheduled);

  // After scheduling: last_scheduled_info_ flips back to kDeviceXU (the
  // authoritative discriminator) and the stale host seq is zeroed for hygiene.
  EXPECT_EQ(info.kind, LastKind::kDeviceXU);
  EXPECT_EQ(info.host_completion_seq, 0u);

  engine_->ClearRuntimeQueue(stream_.get());
}

// ── ScheduleDeviceOperation: no prior kHost is unaffected ───────────────────
// Sanity: with last_scheduled_info_ unset, the kHost gate is bypassed
// entirely. Device op proceeds normally.
TEST_F(HostCCSchedulingTest, ScheduleDeviceNoPriorHostUnaffected) {
  auto& info = stream_->GetLastScheduledInfo();
  ASSERT_FALSE(info.valid);

  auto device_op = MakeDeviceOp(stream_.get(), /*seq=*/7, "no_prior");
  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), device_op);
    engine_->SetRuntimeCursorTo(stream_.get(), device_op.get());
  }

  ScheduleResult result = stream_->ScheduleDeviceOperation(device_op.get(), KernelTypeEnum::kHLO);
  EXPECT_EQ(result, ScheduleResult::kScheduled);
  EXPECT_EQ(info.kind, LastKind::kDeviceXU);
  EXPECT_TRUE(info.valid);

  engine_->ClearRuntimeQueue(stream_.get());
}

// ── End-to-end gate symmetry: H→D→H sequence ────────────────────────────────
// host[0] schedules unconditionally; device[1] gates on host[0]'s completion
// seq (kNotReady until Release, then kScheduled); host[2] gates on device[1]'s
// retire (mock NRT). Blocking closures keep the worker from racing the asserts.
TEST_F(HostCCSchedulingTest, EndToEndHostDeviceHostSequence) {
  BlockingHostOp blocking_h0 = MakeBlockingHostOp(stream_.get(), "h0");
  auto host0 = blocking_h0.op;
  auto dev1 = MakeDeviceOp(stream_.get(), /*seq=*/100, "d1");
  BlockingHostOp blocking_h2 = MakeBlockingHostOp(stream_.get(), "h2");
  auto host2 = blocking_h2.op;

  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), host0);
    engine_->InjectIntoRuntimeQueue(stream_.get(), dev1);
    engine_->InjectIntoRuntimeQueue(stream_.get(), host2);
    engine_->SetRuntimeCursorTo(stream_.get(), host0.get());
  }

  // Step 1: host0 schedules (no prior op).
  ASSERT_TRUE(engine_->ScheduleOperation(host0.get(), stream_.get()));
  ASSERT_TRUE(host0->nrt_async_state_.is_host_inflight.load());

  // Step 2: dev1 sees last_scheduled_info_.kind == kHost, completed_seq below
  // host0's seq (worker blocked in the closure) → kNotReady.
  ASSERT_EQ(stream_->ScheduleDeviceOperation(dev1.get(), KernelTypeEnum::kHLO),
            ScheduleResult::kNotReady);
  ASSERT_FALSE(dev1->nrt_async_state_.is_scheduled);

  // Step 3: release host0's worker so RunHost() returns and the worker pool
  // advances completed_seq (via WorkerLoop's AdvanceHostCompletion).
  blocking_h0.Release();
  WaitHostSeqReached(host0->host_completion_seq_);

  // Step 4: dev1 retried — now scheduled.
  ASSERT_EQ(stream_->ScheduleDeviceOperation(dev1.get(), KernelTypeEnum::kHLO),
            ScheduleResult::kScheduled);
  ASSERT_TRUE(dev1->nrt_async_state_.is_scheduled);
  ASSERT_EQ(stream_->GetLastScheduledInfo().kind, LastKind::kDeviceXU);

  // Step 5: host2 sees last_scheduled_info_.kind == kDeviceXU and uses the
  // default "fully retired" mock → proceeds.
  ASSERT_TRUE(engine_->ScheduleOperation(host2.get(), stream_.get()));
  ASSERT_TRUE(host2->nrt_async_state_.is_host_inflight.load());
  ASSERT_EQ(stream_->GetLastScheduledInfo().kind, LastKind::kHost);

  // Cleanup: release host2, wait its seq, then drain before teardown.
  blocking_h2.Release();
  WaitHostSeqReached(host2->host_completion_seq_);
  DrainEngineHostCompletions();
  engine_->ClearRuntimeQueue(stream_.get());
}

// ── kHost-after-kHost on same stream: pool-internal FIFO preserves order ────
// The engine's gate does NOT block kHost on a prior kHost — the pool/stream
// FIFOs preserve order. Verify ScheduleOperation accepts back-to-back host ops.
// Blocking closures keep the worker from racing the asserts on completed_seq.
TEST_F(HostCCSchedulingTest, ScheduleHostAfterHostNotGated) {
  BlockingHostOp blocking_h0 = MakeBlockingHostOp(stream_.get(), "h0");
  auto host0 = blocking_h0.op;
  BlockingHostOp blocking_h1 = MakeBlockingHostOp(stream_.get(), "h1");
  auto host1 = blocking_h1.op;

  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), host0);
    engine_->InjectIntoRuntimeQueue(stream_.get(), host1);
    engine_->SetRuntimeCursorTo(stream_.get(), host0.get());
  }

  ASSERT_TRUE(engine_->ScheduleOperation(host0.get(), stream_.get()));
  // host0 set kind=kHost; host1's gate should still pass without blocking on
  // host0's future.
  ASSERT_EQ(stream_->GetLastScheduledInfo().kind, LastKind::kHost);

  ASSERT_TRUE(engine_->ScheduleOperation(host1.get(), stream_.get()));
  ASSERT_TRUE(host0->nrt_async_state_.is_host_inflight.load());
  ASSERT_TRUE(host1->nrt_async_state_.is_host_inflight.load());

  // Cleanup: release both, wait their seqs, then drain before teardown.
  blocking_h0.Release();
  blocking_h1.Release();
  WaitHostSeqReached(host0->host_completion_seq_);
  WaitHostSeqReached(host1->host_completion_seq_);
  DrainEngineHostCompletions();
  engine_->ClearRuntimeQueue(stream_.get());
}

// ── Out-of-order host completion is monotonic (fetch_max never regresses) ───
// With >1 worker, same-stream kHost ops can finish out of order; AdvanceCompleted
// (fetch_max/CAS) must never regress completed_seq. Deterministic: advance 5
// then 3, assert it stays 5, and that the device gate reads the monotonic value.
TEST_F(HostCCSchedulingTest, OutOfOrderHostCompletionIsMonotonic) {
  auto& hcs = stream_->host_completion_state_;
  ASSERT_EQ(hcs.completed_seq_id.load(), 0u);

  // A later op completes first.
  hcs.AdvanceCompleted(5);
  EXPECT_EQ(hcs.completed_seq_id.load(), 5u);

  // An earlier op completes second — must NOT regress.
  hcs.AdvanceCompleted(3);
  EXPECT_EQ(hcs.completed_seq_id.load(), 5u) << "fetch_max regressed on a stale (lower) advance";

  // Re-advancing to the same value is idempotent.
  hcs.AdvanceCompleted(5);
  EXPECT_EQ(hcs.completed_seq_id.load(), 5u);

  // A strictly newer completion advances.
  hcs.AdvanceCompleted(7);
  EXPECT_EQ(hcs.completed_seq_id.load(), 7u);

  // Gate semantics on the monotonic value: a device op gated on seq 5 sees the
  // host done (5 <= 7 → complete); one gated on seq 8 does not.
  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kHost;
  info.host_completion_seq = 5;
  info.valid = true;
  auto dev_done = MakeDeviceOp(stream_.get(), /*seq=*/200, "gated_complete");
  {
    engine_->InjectIntoRuntimeQueue(stream_.get(), dev_done);
    engine_->SetRuntimeCursorTo(stream_.get(), dev_done.get());
  }
  EXPECT_EQ(stream_->ScheduleDeviceOperation(dev_done.get(), KernelTypeEnum::kHLO),
            ScheduleResult::kScheduled);
  {
    engine_->ClearRuntimeQueue(stream_.get());
  }

  // Reset gate for the "not complete" half: seq 8 > completed (7).
  info.kind = LastKind::kHost;
  info.host_completion_seq = 8;
  info.valid = true;
  auto dev_pending = MakeDeviceOp(stream_.get(), /*seq=*/201, "gated_pending");
  EXPECT_CALL(static_cast<MockNeuronKernelExecution&>(*dev_pending->kernel_execution),
              Schedule(_, _, _))
      .Times(0);
  EXPECT_EQ(stream_->ScheduleDeviceOperation(dev_pending.get(), KernelTypeEnum::kHLO),
            ScheduleResult::kNotReady);
}

// ── Multi-worker out-of-order completion stays monotonic ────────────────────
// Concurrent counterpart to the deterministic test above: a real 4-thread pool
// with ops sleeping for descending durations to force out-of-order completion.
// The real WorkerLoop advances completed_seq via the production path. We assert
// ONLY race-free values: each op sees completed_seq >= its own seq at completion
// (guaranteed by fetch_max), and final completed_seq == max stamped seq. We do
// NOT compare a cross-callback "last observed" value — those loads are unordered
// and would false-positive even though completed_seq itself never regresses.
TEST_F(HostCCSchedulingTest, MultiWorkerOutOfOrderCompletionMonotonic) {
  constexpr int N = 16;
  std::atomic<bool> own_seq_violation{false};

  // on_complete_ fires after the WorkerLoop's AdvanceHostCompletion for this
  // op, so completed_seq must be >= this op's seq. fetch_max guarantees this
  // regardless of other workers' interleaving — a race-free per-op check.
  HostWorkerPool pool(/*num_threads=*/4, [&](OperationContext* op, std::exception_ptr) {
    if (stream_->host_completion_state_.completed_seq_id.load(std::memory_order_acquire) <
        op->host_completion_seq_) {
      own_seq_violation.store(true);
    }
  });

  std::vector<std::shared_ptr<OperationContext>> ops;
  ops.reserve(N);
  for (int i = 0; i < N; ++i) {
    // Op i gets seq i+1; RunHost sleeps (N-i) ms so earlier-stamped ops finish
    // LATER, encouraging out-of-order completion across the 4 workers.
    int sleep_ms = (N - i);
    auto kernel = std::make_unique<HostKernelExecution>(
        "oo" + std::to_string(i), /*src_refs=*/std::vector<TensorDataRef>{},
        /*dst_refs=*/std::vector<TensorDataRef>{}, MakeSleepFn(sleep_ms), /*device_id=*/0);
    auto op = std::make_shared<OperationContext>(std::move(kernel));
    op->stream = stream_.get();
    op->stream_id = stream_->stream_id;
    op->host_completion_seq_ = static_cast<HostCompletionSeq>(i + 1);
    ops.push_back(op);
  }
  for (auto& op : ops) pool.Submit(op.get());
  pool.Shutdown();  // joins workers — all RunHost + AdvanceHostCompletion done

  EXPECT_FALSE(own_seq_violation.load())
      << "a kHost op's completion observed completed_seq below its own host_completion_seq_";
  // Final completed_seq is the max stamped seq regardless of completion order —
  // proves fetch_max landed on the maximum, never regressed to a lower seq.
  EXPECT_EQ(stream_->host_completion_state_.completed_seq_id.load(),
            static_cast<HostCompletionSeq>(N));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
