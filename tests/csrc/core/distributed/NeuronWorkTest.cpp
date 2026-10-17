#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronGuardImpl.h"
#include "torch_neuronx/csrc/distributed/NeuronWork.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch_neuronx {
namespace distributed {
namespace {

// =============================================================================
// TensorShelf Tests - No mocking needed, fully standalone
// =============================================================================

TEST(TensorShelfTest, DefaultConstruction) {
  TensorShelf shelf;
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, StashAndEmpty) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors = {
      at::zeros({2, 3}),
      at::ones({4, 5}),
  };

  EXPECT_TRUE(shelf.empty());
  shelf.stash(tensors);
  EXPECT_FALSE(shelf.empty());
}

TEST(TensorShelfTest, StashMultipleTimes) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors1 = {at::zeros({2, 3})};
  std::vector<at::Tensor> tensors2 = {at::ones({4, 5})};

  shelf.stash(tensors1);
  EXPECT_FALSE(shelf.empty());

  shelf.stash(tensors2);
  EXPECT_FALSE(shelf.empty());
}

TEST(TensorShelfTest, UnstashClearsShelf) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  shelf.stash(tensors);
  EXPECT_FALSE(shelf.empty());

  shelf.unstash();
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, ClearClearsShelf) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  shelf.stash(tensors);
  EXPECT_FALSE(shelf.empty());

  shelf.clear();
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, StashEmptyVector) {
  TensorShelf shelf;

  std::vector<at::Tensor> empty_tensors;
  shelf.stash(empty_tensors);
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, MultipleUnstashIsSafe) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  shelf.stash(tensors);

  shelf.unstash();
  EXPECT_TRUE(shelf.empty());

  // Second unstash should be safe
  EXPECT_NO_THROW(shelf.unstash());
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, MultipleClearIsSafe) {
  TensorShelf shelf;

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  shelf.stash(tensors);

  shelf.clear();
  EXPECT_TRUE(shelf.empty());

  // Second clear should be safe
  EXPECT_NO_THROW(shelf.clear());
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, ClearOnEmptyShelfIsSafe) {
  TensorShelf shelf;
  EXPECT_TRUE(shelf.empty());

  EXPECT_NO_THROW(shelf.clear());
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, ThreadSafetyStashAndEmpty) {
  TensorShelf shelf;
  std::atomic<int> stash_count{0};
  std::atomic<int> empty_check_count{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&shelf, &stash_count, &empty_check_count]() {
      for (int j = 0; j < 100; ++j) {
        std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
        shelf.stash(tensors);
        stash_count++;

        shelf.empty();
        empty_check_count++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(stash_count.load(), 400);
  EXPECT_EQ(empty_check_count.load(), 400);
  EXPECT_FALSE(shelf.empty());

  shelf.clear();
  EXPECT_TRUE(shelf.empty());
}

TEST(TensorShelfTest, ThreadSafetyStashAndClear) {
  TensorShelf shelf;
  std::atomic<bool> done{false};

  // One thread stashing
  std::thread stash_thread([&shelf, &done]() {
    while (!done.load()) {
      std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
      shelf.stash(tensors);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  // Another thread clearing
  std::thread clear_thread([&shelf, &done]() {
    while (!done.load()) {
      shelf.clear();
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  // Let them run for a bit
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  done.store(true);

  stash_thread.join();
  clear_thread.join();

  // Should not crash - final state doesn't matter
}

// =============================================================================
// NeuronWork Tests - Requires mocking infrastructure
// =============================================================================

class NeuronWorkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize NRT mocks
    nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    auto* nrt_mock = torch_neuronx::testing::MockNRT::GetInstance();

    ON_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
        .WillByDefault(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_allocate_tensor_set(_))
        .WillByDefault(DoAll(SetArgPointee<0>(reinterpret_cast<nrt::TensorSetHandle*>(0x1)),
                             Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_destroy_tensor_set(_)).WillByDefault(Return());
    ON_CALL(*nrt_mock, nrt_add_tensor_to_tensor_set(_, _, _)).WillByDefault(Return(NRT_SUCCESS));
    ON_CALL(*nrt_mock, nrt_execute(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    copy_utils_session_ = std::make_unique<torch_neuronx::utils::testing::MockCopyUtilsSession>();

    c10_neuron::reset_distributed_state();

    InitializeStreamPools();

    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();
    engine_ = mock_engine_wrapper_->GetEngine();
  }

  void TearDown() override {
    mock_engine_wrapper_.reset();
    copy_utils_session_.reset();
    nrt_session_.reset();
    CleanupStreamPools();
  }

  // Helper to create a NeuronWork instance for testing
  c10::intrusive_ptr<NeuronWork> createWork(const std::string& op_type = "test_op",
                                            std::vector<at::Tensor> outputs = {},
                                            float timeout_ms = 300000.0f,
                                            bool blocking_wait = false) {
    at::Device device(c10::DeviceType::PrivateUse1, 0);
    return c10::make_intrusive<NeuronWork>("test_pg_uid", "test_pg_desc", device,
                                           0,  // rank
                                           op_type, seq_num_++, std::move(outputs),
                                           false,  // enable_timing
                                           timeout_ms,
                                           nullptr,  // stream
                                           blocking_wait);
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuronx::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
  OperationExecutionEngine* engine_;
  uint64_t seq_num_ = 0;
};

// Test basic construction
TEST_F(NeuronWorkTest, BasicConstruction) {
  auto work = createWork();
  EXPECT_NE(work, nullptr);
}

// Test getters return correct values
TEST_F(NeuronWorkTest, GettersReturnCorrectValues) {
  auto work = createWork("all_reduce");

  EXPECT_EQ(work->getOpType(), "all_reduce");
  EXPECT_EQ(work->getDevice().type(), c10::DeviceType::PrivateUse1);
  EXPECT_EQ(work->getDevice().index(), 0);
  EXPECT_EQ(work->sourceRank(), -1);  // Not a receive operation
}

// Test sequence numbers increment
TEST_F(NeuronWorkTest, SequenceNumbersIncrement) {
  auto work1 = createWork();
  auto work2 = createWork();
  auto work3 = createWork();

  EXPECT_LT(work1->getSequenceNumber(), work2->getSequenceNumber());
  EXPECT_LT(work2->getSequenceNumber(), work3->getSequenceNumber());
}

// Test result returns output tensors
TEST_F(NeuronWorkTest, ResultReturnsOutputTensors) {
  std::vector<at::Tensor> outputs = {at::zeros({2, 3}), at::ones({4, 5})};
  auto work = createWork("all_reduce", outputs);

  auto result = work->result();
  EXPECT_EQ(result.size(), 2);
}

// Test isSuccess throws deprecation error
TEST_F(NeuronWorkTest, IsSuccessThrowsDeprecationError) {
  auto work = createWork();
  EXPECT_THROW(work->isSuccess(), c10::Error);
}

// Test stash and unstash
TEST_F(NeuronWorkTest, StashAndUnstash) {
  auto work = createWork();

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  EXPECT_NO_THROW(work->stash(tensors));
  EXPECT_NO_THROW(work->unstashTensors());
}

// Test stash multiple times
TEST_F(NeuronWorkTest, StashMultipleTimes) {
  auto work = createWork();

  std::vector<at::Tensor> tensors1 = {at::zeros({2, 3})};
  std::vector<at::Tensor> tensors2 = {at::ones({4, 5})};

  EXPECT_NO_THROW(work->stash(tensors1));
  EXPECT_NO_THROW(work->stash(tensors2));
  EXPECT_NO_THROW(work->unstashTensors());
}

// Test detachStashedTensorShelf transfers ownership
TEST_F(NeuronWorkTest, DetachStashedTensorShelfTransfersOwnership) {
  auto work = createWork();

  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  work->stash(tensors);

  auto shelf = work->detachStashedTensorShelf();
  EXPECT_NE(shelf, nullptr);

  // After detach, work's shelf should be null
  auto shelf2 = work->detachStashedTensorShelf();
  EXPECT_EQ(shelf2, nullptr);
}

// Test setException stores exception
TEST_F(NeuronWorkTest, SetExceptionStoresException) {
  auto work = createWork();

  auto exception = std::make_exception_ptr(std::runtime_error("test error"));
  work->setException(exception);

  // Work should now be "completed" with exception
  EXPECT_TRUE(work->isCompleted());
}

// Test getDuration throws when timing disabled
TEST_F(NeuronWorkTest, GetDurationThrowsWhenTimingDisabled) {
  auto work = createWork();
  EXPECT_THROW(work->getDuration(), c10::Error);
}

// Test checkTimeout returns false when not timed out
TEST_F(NeuronWorkTest, CheckTimeoutReturnsFalseInitially) {
  // Create work with very long timeout
  auto work = createWork("test_op", {}, 300000.0f);  // 5 minutes

  // Should not be timed out immediately
  EXPECT_FALSE(work->checkTimeout());
}

// Test checkTimeout returns true when timeout exceeded
TEST_F(NeuronWorkTest, CheckTimeoutReturnsTrueWhenExceeded) {
  // Create work with very short timeout
  auto work = createWork("test_op", {}, 1.0f);  // 1 ms

  // Wait for timeout to expire
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Should be timed out now
  EXPECT_TRUE(work->checkTimeout());
}

// Test isCompleted returns true after exception
TEST_F(NeuronWorkTest, IsCompletedReturnsTrueAfterException) {
  auto work = createWork();

  // Note: Without calling recordEndEvent(), end_event_ is never recorded.
  // NeuronEvent::query() returns true for unrecorded events (!is_recorded()),
  // so isCompleted() returns true. In production, backend.py calls
  // work.record_end_event() immediately after dispatching the collective.
  // This test verifies that setting an exception keeps it completed.
  auto exception = std::make_exception_ptr(std::runtime_error("test error"));
  work->setException(exception);

  EXPECT_TRUE(work->isCompleted());
}

// Test abort sets exception and completes
TEST_F(NeuronWorkTest, AbortSetsExceptionAndCompletes) {
  auto work = createWork();

  // Note: Without recordEndEvent(), isCompleted() returns true because
  // NeuronEvent::query() returns true for unrecorded events.
  // abort() should still work correctly - setting an exception.
  work->abort();

  EXPECT_TRUE(work->isCompleted());
}

class PublicEventWorkTest : public NeuronWorkTest {
 protected:
  void SetUp() override {
    NeuronWorkTest::SetUp();
    // This binary does not link the extension that registers the real device guard.
    static c10::impl::DeviceGuardImplRegistrar guard_registrar(c10::DeviceType::PrivateUse1,
                                                               new c10_neuron::NeuronGuardImpl());
    engine_->Shutdown();
    NeuronResourceManager::Instance().GetOperationExecutionEngine().Shutdown();
    scoped_engine_ = std::make_unique<ScopedTestEngine>(&compilation_cache_, &model_cache_);
    stream_impl_ = NeuronStream::getDefaultStream(0).get_stream_impl();
    auto kernel = std::make_unique<::testing::NiceMock<MockEventDirectKernelExecution>>(
        "record_blocker", EventDirectKernelExecution::EventAction::kSignal, 0);
    auto blocker = std::make_shared<OperationContext>(std::move(kernel));
    blocker->MarkExecutionReady();
    scoped_engine_->engine().InjectIntoRuntimeQueue(stream_impl_, std::move(blocker));
    const char* blocking = std::getenv("NEURON_LAUNCH_BLOCKING");
    if (blocking != nullptr && std::string(blocking) == "1") {
      GTEST_SKIP() << "Pending-record tests require NEURON_LAUNCH_BLOCKING=0";
    }
  }

  void TearDown() override {
    if (scoped_engine_) {
      scoped_engine_->engine().Shutdown();
      scoped_engine_->engine().ClearRuntimeQueue(stream_impl_);
      scoped_engine_->engine().dispatch_queue_.clear();
      scoped_engine_.reset();
    }
    NeuronWorkTest::TearDown();
  }

  CompilationCache compilation_cache_;
  ModelHandleCache model_cache_;
  std::unique_ptr<ScopedTestEngine> scoped_engine_;
  StreamImpl* stream_impl_{nullptr};
};

TEST_F(PublicEventWorkTest, AbortCompletesOwnRecordAfterPublicEndEventRerecord) {
  auto stream = NeuronStream::getDefaultStream(0);
  auto work = createWork();
  work->recordEndEvent(&stream);
  auto public_event = work->getEndEvent();
  auto original = public_event.capture();
  ASSERT_FALSE(original.query());
  ASSERT_FALSE(work->isCompleted());
  ASSERT_NE(work->getFuture(), nullptr);

  public_event.record(stream);
  scoped_engine_->engine().Shutdown();
  ASSERT_NE(original.get_impl(), public_event.get_impl());
  EXPECT_FALSE(original.query());
  EXPECT_FALSE(public_event.query());

  work->abort();
  EXPECT_TRUE(original.query());
  EXPECT_FALSE(public_event.query());
  EXPECT_EQ(work->getEndEvent().get_impl(), original.get_impl());
  EXPECT_NE(work->getEndEvent().get_impl(), public_event.get_impl());
  EXPECT_TRUE(work->getEndEvent().query());
  EXPECT_TRUE(work->isCompleted());
}

TEST_F(PublicEventWorkTest, PublicEndEventCompletionDoesNotCompleteWorkRecord) {
  auto stream = NeuronStream::getDefaultStream(0);
  auto work = createWork();
  work->recordEndEvent(&stream);
  auto public_event = work->getEndEvent();
  auto original = public_event.capture();
  public_event.record(stream);
  scoped_engine_->engine().Shutdown();
  ASSERT_NE(original.get_impl(), public_event.get_impl());

  EventDirectKernelExecution public_signal("public_signal", public_event,
                                           EventDirectKernelExecution::EventAction::kSignal, 0);
  public_signal.CompleteWithoutExecution();
  EXPECT_TRUE(public_event.query());
  EXPECT_FALSE(original.query());
  EXPECT_FALSE(work->isCompleted());

  EventDirectKernelExecution original_signal("original_signal", original,
                                             EventDirectKernelExecution::EventAction::kSignal, 0);
  original_signal.CompleteWithoutExecution();
  EXPECT_TRUE(work->isCompleted());
}

TEST_F(PublicEventWorkTest, ExplicitWorkRerecordPreservesEarlierSnapshot) {
  auto stream = NeuronStream::getDefaultStream(0);
  auto work = createWork();
  work->recordEndEvent(&stream);
  auto first = work->getEndEvent();
  work->recordEndEvent(&stream);
  auto second = work->getEndEvent();
  scoped_engine_->engine().Shutdown();
  ASSERT_NE(first.get_impl(), second.get_impl());
  ASSERT_FALSE(first.query());
  ASSERT_FALSE(second.query());

  EventDirectKernelExecution first_signal("first_signal", first,
                                          EventDirectKernelExecution::EventAction::kSignal, 0);
  first_signal.CompleteWithoutExecution();
  EXPECT_TRUE(first.query());
  EXPECT_FALSE(second.query());
  EXPECT_FALSE(work->isCompleted());

  EventDirectKernelExecution second_signal("second_signal", second,
                                           EventDirectKernelExecution::EventAction::kSignal, 0);
  second_signal.CompleteWithoutExecution();
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(work->getEndEvent().query());
}

TEST_F(PublicEventWorkTest, KernelSnapshotRerecordDoesNotRetargetWorkCompletion) {
  auto stream = NeuronStream::getDefaultStream(0);
  auto work = createWork();
  work->recordEndEvent(&stream);
  auto completion = work->getEndEvent();
  EventDirectKernelExecution signal("work_signal", completion,
                                    EventDirectKernelExecution::EventAction::kSignal, 0);
  auto snapshot = signal.GetEvent();
  snapshot.record(stream);
  scoped_engine_->engine().Shutdown();

  EXPECT_NE(snapshot.get_impl(), completion.get_impl());
  EXPECT_FALSE(work->isCompleted());
  signal.CompleteWithoutExecution();
  EXPECT_TRUE(work->isCompleted());
  EXPECT_TRUE(completion.query());
  EXPECT_EQ(work->getEndEvent().get_impl(), completion.get_impl());
  EXPECT_FALSE(snapshot.query());
}

// Test getFuture returns nullptr before recordEndEvent
TEST_F(NeuronWorkTest, GetFutureReturnsNullptrInitially) {
  auto work = createWork();
  EXPECT_EQ(work->getFuture(), nullptr);
}

// Test handleException rethrows
TEST_F(NeuronWorkTest, HandleExceptionRethrows) {
  auto work = createWork();

  auto exception = std::make_exception_ptr(std::runtime_error("test error"));
  work->setException(exception);

  EXPECT_THROW(work->handleException(), std::runtime_error);
}

// Test handleException does nothing without exception
TEST_F(NeuronWorkTest, HandleExceptionNoopWithoutException) {
  auto work = createWork();
  EXPECT_NO_THROW(work->handleException());
}

// Test concurrent detach and unstash is safe (regression test for shelf race condition)
// Previously, synchronize() calling unstashTensors() could race with watchdog's
// detachStashedTensorShelf(), causing use-after-free on the shelf's mutex (SIGSEGV).
TEST_F(NeuronWorkTest, ConcurrentDetachAndUnstashIsSafe) {
  for (int trial = 0; trial < 100; ++trial) {
    auto work = createWork("all_gather", {at::zeros({2, 3})});
    std::vector<at::Tensor> tensors = {at::zeros({4, 5})};
    work->stash(tensors);

    std::atomic<bool> start{false};

    // Simulate watchdog thread calling detachStashedTensorShelf
    std::thread watchdog_thread([&]() {
      while (!start.load()) {
      }
      auto shelf = work->detachStashedTensorShelf();
      if (shelf) {
        shelf->unstash();
      }
    });

    // Simulate main thread calling unstashTensors (as synchronize() does)
    std::thread main_thread([&]() {
      while (!start.load()) {
      }
      work->unstashTensors();
    });

    start.store(true);
    watchdog_thread.join();
    main_thread.join();
  }
}

// Test that unstashTensors is safe after detach (no crash on null shelf)
// This verifies the null check + mutex in unstashTensors works correctly
// when the watchdog has already detached the shelf.
TEST_F(NeuronWorkTest, UnstashAfterDetachIsSafe) {
  auto work = createWork("all_reduce", {at::zeros({2, 3})});
  std::vector<at::Tensor> tensors = {at::zeros({4, 5})};
  work->stash(tensors);

  // Watchdog detaches first
  auto shelf = work->detachStashedTensorShelf();
  EXPECT_NE(shelf, nullptr);

  // Main thread calls unstash after detach - must not crash
  EXPECT_NO_THROW(work->unstashTensors());
}

// Test that calling wait() after the watchdog has already processed the work
// (detached the shelf) is safe. This simulates the race where user is slow
// to call wait() and watchdog has already finished processing.
// Note: We don't call recordEndEvent() because that requires device registration
// which isn't available in unit tests. Without recordEndEvent(), isCompleted()
// returns true because unrecorded events return true for query().
TEST_F(NeuronWorkTest, WaitAfterWatchdogDetachIsSafe) {
  auto work = createWork("all_reduce", {at::zeros({2, 3})});
  std::vector<at::Tensor> tensors = {at::zeros({4, 5})};
  work->stash(tensors);

  // Simulate: work completes, user is slow (sleep), watchdog processes first
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Watchdog detaches shelf first (simulating watchdog thread processing)
  auto shelf = work->detachStashedTensorShelf();
  EXPECT_NE(shelf, nullptr);

  // User finally calls wait() AFTER watchdog already processed
  // This must still work without crash - synchronize() checks null shelf
  EXPECT_NO_THROW(work->wait());
  EXPECT_TRUE(work->isCompleted());
}

}  // namespace
}  // namespace distributed
}  // namespace torch_neuronx
