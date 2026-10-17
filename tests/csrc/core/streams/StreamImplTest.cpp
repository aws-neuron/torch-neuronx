#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/ProfilerMappingCollector.h"
#include "torch_neuronx/csrc/core/cache/PersistentCacheBackend.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamConfig.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"
#include "torch_neuronx/csrc/profiler/ProfilerEventBuffer.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace torch_neuronx::testing;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class StreamImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize NRT mocks
    nrt_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    auto* nrt_mock = torch_neuronx::testing::MockNRT::GetInstance();

    // Set up vnc_count mock before InitializeStreamPools
    ON_CALL(*nrt_mock, nrt_get_visible_vnc_count(_))
        .WillByDefault(DoAll(SetArgPointee<0>(1), Return(NRT_SUCCESS)));

    // Set up default NRT behaviors for successful execution
    ON_CALL(*nrt_mock, nrt_allocate_tensor_set(_))
        .WillByDefault(DoAll(SetArgPointee<0>(reinterpret_cast<nrt::TensorSetHandle*>(0x1)),
                             Return(NRT_SUCCESS)));

    ON_CALL(*nrt_mock, nrt_destroy_tensor_set(_)).WillByDefault(Return());

    ON_CALL(*nrt_mock, nrt_add_tensor_to_tensor_set(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    ON_CALL(*nrt_mock, nrt_execute(_, _, _)).WillByDefault(Return(NRT_SUCCESS));

    // Initialize copy utils mocks
    copy_utils_session_ = std::make_unique<torch_neuronx::utils::testing::MockCopyUtilsSession>();

    c10_neuron::reset_distributed_state();

    // Initialize stream pools before creating engine to prevent worker thread crashes
    InitializeStreamPools();

    // Use mock engine that skips device checks and actual execution
    mock_engine_wrapper_ = std::make_unique<MockOperationExecutionEngineWrapper>();
    engine_ = mock_engine_wrapper_->GetEngine();

    // Create stream with device 0, stream id 1
    stream_ = std::make_shared<StreamImpl>(0, 1);
  }

  void TearDown() override {
    if (stream_) {
      stream_.reset();
    }
    mock_engine_wrapper_.reset();
    copy_utils_session_.reset();
    nrt_session_.reset();

    // Cleanup stream pools
    CleanupStreamPools();
  }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> nrt_session_;
  std::unique_ptr<torch_neuronx::utils::testing::MockCopyUtilsSession> copy_utils_session_;
  std::unique_ptr<MockOperationExecutionEngineWrapper> mock_engine_wrapper_;
  OperationExecutionEngine* engine_;
  std::shared_ptr<StreamImpl> stream_;
};

// Test basic construction
TEST_F(StreamImplTest, BasicConstruction) {
  EXPECT_NE(stream_, nullptr);
  EXPECT_EQ(stream_->device_index, 0);
  EXPECT_EQ(stream_->stream_id, 1);
}

TEST_F(StreamImplTest, StreamLookupLazilyInitializesStateOutsideCompileOnly) {
  CleanupStreamPools();
  c10_neuron::set_compile_only(false);

  auto stream = getDefaultNeuronStream(0);
  EXPECT_NE(stream.get_stream_impl(), nullptr);
}

// Test Query on empty stream
TEST_F(StreamImplTest, QueryEmptyStream) { EXPECT_TRUE(stream_->Query()); }

// Test Synchronize on empty stream
TEST_F(StreamImplTest, SynchronizeEmptyStream) { EXPECT_NO_THROW(stream_->Synchronize()); }

// Test basic stream state
TEST_F(StreamImplTest, BasicStreamState) {
  EXPECT_TRUE(stream_->Query());  // Stream should be idle initially
  EXPECT_NO_THROW(stream_->Synchronize());
}

// Test stream on different devices
TEST_F(StreamImplTest, StreamOnDifferentDevices) {
  auto stream_device_0 = std::make_shared<StreamImpl>(0, 10);
  auto stream_device_1 = std::make_shared<StreamImpl>(1, 11);

  EXPECT_EQ(stream_device_0->device_index, 0);
  EXPECT_EQ(stream_device_1->device_index, 1);
}

// Test concurrent synchronization on empty stream
TEST_F(StreamImplTest, ConcurrentSynchronizationEmptyStream) {
  std::vector<std::thread> threads;

  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([this]() { stream_->Synchronize(); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(stream_->Query());
}

// Test error handler initialization
TEST_F(StreamImplTest, ErrorHandlerInitialization) { EXPECT_NE(stream_->error_handler_, nullptr); }

// Test runtime_queue_ starts empty
TEST_F(StreamImplTest, ActiveOperationsStartsEmpty) {
  // Query returns true when no active operations
  EXPECT_TRUE(stream_->Query());
}

// Test multiple Query calls are consistent
TEST_F(StreamImplTest, MultipleQueryCallsConsistent) {
  EXPECT_TRUE(stream_->Query());
  EXPECT_TRUE(stream_->Query());
  EXPECT_TRUE(stream_->Query());
}

// Test Synchronize is idempotent on empty stream
TEST_F(StreamImplTest, SynchronizeIdempotentOnEmptyStream) {
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_NO_THROW(stream_->Synchronize());
  EXPECT_TRUE(stream_->Query());
}

// Test stream with large stream ID
TEST_F(StreamImplTest, StreamWithLargeStreamId) {
  auto large_id_stream = std::make_shared<StreamImpl>(0, 999999);
  EXPECT_EQ(large_id_stream->stream_id, 999999);
  EXPECT_TRUE(large_id_stream->Query());
}

// Test stream with various device indices
TEST_F(StreamImplTest, StreamWithVariousDeviceIndices) {
  for (c10::DeviceIndex device = 0; device < 4; ++device) {
    auto stream = std::make_shared<StreamImpl>(device, device + 100);
    EXPECT_EQ(stream->device_index, device);
    EXPECT_EQ(stream->stream_id, device + 100);
  }
}

// Test concurrent Query calls
TEST_F(StreamImplTest, ConcurrentQueryCalls) {
  std::vector<std::thread> threads;
  std::atomic<int> true_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([this, &true_count]() {
      if (stream_->Query()) {
        true_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All should return true for empty stream
  EXPECT_EQ(true_count.load(), 10);
}

// Test stream destruction is safe
TEST_F(StreamImplTest, StreamDestructionIsSafe) {
  auto temp_stream = std::make_shared<StreamImpl>(0, 500);
  EXPECT_NE(temp_stream, nullptr);
  EXPECT_TRUE(temp_stream->Query());

  // Destruction should be safe
  EXPECT_NO_THROW(temp_stream.reset());
}

// Test multiple streams can coexist
TEST_F(StreamImplTest, MultipleStreamsCoexist) {
  std::vector<std::shared_ptr<StreamImpl>> streams;

  for (int i = 0; i < 5; ++i) {
    streams.push_back(std::make_shared<StreamImpl>(0, 300 + i));
  }

  // All streams should be queryable
  for (const auto& s : streams) {
    EXPECT_TRUE(s->Query());
  }

  // Verify each has correct stream_id
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(streams[i]->stream_id, 300 + i);
  }
}

// ============================================================================
// Async Execution State Tests
// ============================================================================

TEST_F(StreamImplTest, LastScheduledInfoInitiallyInvalid) {
  auto& info = stream_->GetLastScheduledInfo();
  EXPECT_FALSE(info.valid);
  EXPECT_EQ(info.sequence_id, 0);
  EXPECT_EQ(info.xu, NRTA_XU_COMPUTE);  // Default value
}

TEST_F(StreamImplTest, LastScheduledInfoModifiable) {
  auto& info = stream_->GetLastScheduledInfo();

  info.valid = true;
  info.sequence_id = 12345;
  info.xu = NRTA_XU_TENSOR_OP;

  // Verify changes persist
  const auto& const_info = stream_->GetLastScheduledInfo();
  EXPECT_TRUE(const_info.valid);
  EXPECT_EQ(const_info.sequence_id, 12345);
  EXPECT_EQ(const_info.xu, NRTA_XU_TENSOR_OP);
}

TEST_F(StreamImplTest, HasPendingAsyncOperationsEmptyStream) {
  // Empty stream should have no pending async operations
  EXPECT_FALSE(stream_->HasPendingAsyncOperations());
}

TEST_F(StreamImplTest, GetNextOperationToScheduleEmptyStream) {
  auto* result = stream_->GetNextOperationToSchedule();
  EXPECT_EQ(result, nullptr);
}

namespace {

// IsReady() defaults to false so the bypass path is exercised; execution is
// marked ready so IsSchedulable() depends only on IsReady().
std::shared_ptr<OperationContext> MakeEventOp(EventDirectKernelExecution::EventAction action,
                                              StreamImpl* recorded_on) {
  NeuronEvent event;
  event.get_impl()->recorded_stream_impl.store(recorded_on, std::memory_order_relaxed);
  auto kernel = std::make_unique<MockEventDirectKernelExecution>("test_event", event, action, 0);
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->MarkExecutionReady();
  return op;
}

}  // namespace

// With concat off, engine.SubmitOperation lands the shared_ptr directly into
// stream->runtime_queue_, so we exercise the real production submit path
// rather than test-only injection.
TEST_F(StreamImplTest, GetNextOperationToScheduleSameStreamKWaitBypass) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, stream_.get());
  EXPECT_TRUE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_.get()));
}

TEST_F(StreamImplTest, GetNextOperationToScheduleCrossStreamKWaitGated) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  auto other_stream = std::make_shared<StreamImpl>(0, 2);
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, other_stream.get());
  EXPECT_FALSE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_.get()));
}

TEST_F(StreamImplTest, GetNextOperationToScheduleKSignalNotBypassed) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kSignal, stream_.get());
  EXPECT_FALSE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_.get()));
}

// Regression test: pre-fix, the bypass compared a counter-encoded c10::StreamId
// (>= 32 for pool streams) against StreamImpl::stream_id (slot index, <= 8),
// which never matched. Test fails pre-fix, passes post-fix via StreamImpl*
// pointer compare.
TEST_F(StreamImplTest, BypassFiresForPoolStreamRecordedThroughProductionPath) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");

  // Counter-encoded pool-slot c10::StreamId, same shape as createStream emits.
  constexpr int kPoolIndex = 0;
  constexpr int kPriority = 0;
  constexpr uint32_t kCounter = 1;
  c10::StreamId encoded = (static_cast<c10::StreamId>(kCounter) << 5) |
                          ((static_cast<c10::StreamId>(kPriority) & 0x3) << 3) |
                          (static_cast<c10::StreamId>(kPoolIndex) & 0x7);
  ASSERT_GE(encoded, 32);
  NeuronStream pool_stream(encoded, /*device_index=*/0, NeuronStream::UNCHECKED);

  StreamImpl* pool_impl = pool_stream.get_stream_impl();
  ASSERT_NE(pool_impl, nullptr);
  // StreamImpl::stream_id is 1-based for pool slots (default is 0).
  EXPECT_EQ(pool_impl->stream_id, kPoolIndex + 1);
  // Sanity: the two ID spaces don't coincide for pool streams.
  EXPECT_NE(encoded, static_cast<c10::StreamId>(pool_impl->stream_id));

  NeuronEvent event;
  event.get_impl()->is_recorded.store(true, std::memory_order_relaxed);
  event.get_impl()->recorded_stream_impl.store(pool_impl, std::memory_order_relaxed);

  auto kernel = std::make_unique<MockEventDirectKernelExecution>(
      "regression_event_wait", event, EventDirectKernelExecution::EventAction::kWait,
      /*device_id=*/0);
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->MarkExecutionReady();

  // Same-stream wait must bypass.
  EXPECT_TRUE(StreamImpl::IsSameStreamWaitBypass(op.get(), pool_impl));

  // Cross-stream wait must still gate.
  StreamImpl* other_impl = at::neuron::GetDefaultStreamImpl(0);
  ASSERT_NE(other_impl, pool_impl);
  NeuronEvent cross_event;
  cross_event.get_impl()->is_recorded.store(true, std::memory_order_relaxed);
  cross_event.get_impl()->recorded_stream_impl.store(other_impl, std::memory_order_relaxed);
  auto cross_kernel = std::make_unique<MockEventDirectKernelExecution>(
      "regression_event_wait_cross", cross_event, EventDirectKernelExecution::EventAction::kWait,
      /*device_id=*/0);
  auto cross_op = std::make_shared<OperationContext>(std::move(cross_kernel));
  cross_op->MarkExecutionReady();
  EXPECT_FALSE(StreamImpl::IsSameStreamWaitBypass(cross_op.get(), pool_impl));
}

// NeuronEvent::record() must capture the StreamImpl identity used by the
// bypass. Guards the recording path against ID-space regressions.
TEST_F(StreamImplTest, EventRecordCapturesStreamImplIdentity) {
  NeuronStream default_stream = NeuronStream::getDefaultStream(/*device_index=*/0);
  NeuronEvent event;
  event.record(default_stream);
  EXPECT_EQ(event.recorded_stream_impl(), default_stream.get_stream_impl());
  EXPECT_NE(event.recorded_stream_impl(), nullptr);
}

TEST_F(StreamImplTest, WaitStreamAliasesDoNotSubmitEvents) {
  if (!config::AreStreamsEnabled() || config::IsHostCCEnabled()) {
    GTEST_SKIP() << "Requires streams enabled and HostCC disabled";
  }
  NeuronStream producer(32, 0, NeuronStream::UNCHECKED);
  NeuronStream consumer(64, 0, NeuronStream::UNCHECKED);
  ASSERT_NE(producer.id(), consumer.id());
  auto* impl = producer.get_stream_impl();
  ASSERT_EQ(impl, consumer.get_stream_impl());

  CompilationCache compilation_cache;
  ModelHandleCache model_handle_cache;
  ScopedTestEngine scoped_engine(&compilation_cache, &model_handle_cache);
  auto pending = MakeEventOp(EventDirectKernelExecution::EventAction::kSignal, impl);
  scoped_engine->InjectIntoRuntimeQueue(impl, pending);
  EXPECT_FALSE(producer.query());

  EXPECT_NO_THROW(consumer.wait_stream(producer));
  scoped_engine->Shutdown();
  EXPECT_EQ(scoped_engine->RuntimeQueueSize(impl), 1);
  EXPECT_TRUE(scoped_engine->dispatch_queue_.empty());
  EXPECT_EQ(pending->result_future.wait_for(0ms), std::future_status::timeout);
  scoped_engine->ClearRuntimeQueue(impl);
  scoped_engine->dispatch_queue_.clear();
}

TEST_F(StreamImplTest, CollectiveUsesDefaultFifoAndReportsFailure) {
  if (!config::AreStreamsEnabled() || config::IsHostCCEnabled()) {
    GTEST_SKIP() << "Requires streams enabled and HostCC disabled";
  }
  nrt_instance_info_t info{};
  info.family = NRT_INSTANCE_TRN3;
  ON_CALL(*MockNRT::GetInstance(), nrt_get_instance_info(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

  NeuronStream caller(32, /*device_index=*/0, NeuronStream::UNCHECKED);
  auto* default_impl = GetDefaultStreamImpl(0);
  auto kernel = std::make_unique<MockCompilableKernelExecution>("default_fifo_test", "", "",
                                                                /*has_collectives=*/true);
  auto* kernel_ptr = kernel.get();
  auto op = std::make_unique<OperationContext>(std::move(kernel));
  auto* op_ptr = op.get();
  EXPECT_CALL(*kernel_ptr, Schedule(_, _, 0))
      .WillOnce([&](NRT_STATUS*, nrt::SequenceId*, c10::StreamId) -> bool {
        EXPECT_EQ(op_ptr->stream, default_impl);
        EXPECT_EQ(op_ptr->stream_id, caller.id());
        throw std::runtime_error("collective execution failure");
      });
  SubmitOperationContext(caller, std::move(op));
  EXPECT_THROW(caller.get_stream_impl()->Synchronize(), std::runtime_error);
  EXPECT_NO_THROW(default_impl->Synchronize());
}

TEST_F(StreamImplTest, NoncollectiveKeepsCallerFifoAndLogicalId) {
  if (!config::AreStreamsEnabled()) GTEST_SKIP() << "Requires streams enabled";
  nrt_instance_info_t info{};
  info.family = NRT_INSTANCE_TRN3;
  ON_CALL(*MockNRT::GetInstance(), nrt_get_instance_info(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

  NeuronStream caller(32, /*device_index=*/0, NeuronStream::UNCHECKED);
  auto kernel = std::make_unique<MockCompilableKernelExecution>("pool_fifo_test", "", "",
                                                                /*has_collectives=*/false);
  auto* kernel_ptr = kernel.get();
  auto op = std::make_unique<OperationContext>(std::move(kernel));
  auto* op_ptr = op.get();
  EXPECT_CALL(*kernel_ptr, Schedule(_, _, caller.get_stream_impl()->stream_id))
      .WillOnce([&](NRT_STATUS*, nrt::SequenceId*, c10::StreamId) -> bool {
        EXPECT_EQ(op_ptr->stream, caller.get_stream_impl());
        EXPECT_EQ(op_ptr->stream_id, caller.id());
        throw std::runtime_error("noncollective execution failure");
      });
  SubmitOperationContext(caller, std::move(op));
  EXPECT_THROW(caller.get_stream_impl()->Synchronize(), std::runtime_error);
  EXPECT_NO_THROW(GetDefaultStreamImpl(0)->Synchronize());
}

TEST_F(StreamImplTest, DirectSubmissionFailureStaysOnExecutionStream) {
  nrt_instance_info_t info{};
  info.family = NRT_INSTANCE_TRN3;
  ON_CALL(*MockNRT::GetInstance(), nrt_get_instance_info(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));
  auto* execution_stream = NeuronStream(32, 0, NeuronStream::UNCHECKED).get_stream_impl();
  auto kernel = std::make_unique<MockCompilableKernelExecution>("direct_failure_test");
  auto* kernel_ptr = kernel.get();
  auto op = std::make_unique<OperationContext>(std::move(kernel));
  ASSERT_EQ(op->stream_id, 0);
  EXPECT_CALL(*kernel_ptr, Schedule(_, _, execution_stream->stream_id))
      .WillOnce([](NRT_STATUS*, nrt::SequenceId*, c10::StreamId) -> bool {
        throw std::runtime_error("direct submission failure");
      });
  execution_stream->SubmitOperationContext(std::move(op));
  EXPECT_THROW(execution_stream->Synchronize(), std::runtime_error);
  EXPECT_NO_THROW(GetDefaultStreamImpl(0)->Synchronize());
}

TEST_F(StreamImplTest, ProfilerUsesSelectedDefaultOrPoolStream) {
  StreamImpl default_stream(0, 0);
  auto& mappings = ProfilerMappingCollector::Instance();
  mappings.GetAndClear();
  mappings.SetEnabled(true);
  ProfilerEventBuffer buffer;
  buffer.activate();
  setActiveProfilerBuffer(&buffer);
  // Logical stream 96 represents a collective flattened onto the default implementation.
  for (c10::StreamId logical_id : {0, 32, 64, 96}) {
    auto* selected_stream = logical_id == 0 || logical_id == 96 ? &default_stream : stream_.get();
    auto op = std::make_shared<OperationContext>(std::make_unique<MockNeuronKernelExecution>());
    op->stream = selected_stream;
    op->stream_id = logical_id;
    op->nrt_async_state_.is_scheduled = true;
    op->nrt_async_state_.sequence_id = logical_id + 1;
    auto* engine = static_cast<MockOperationExecutionEngine*>(engine_);
    engine->InjectIntoRuntimeQueue(selected_stream, op);
    selected_stream->CompleteOperation(op.get());
  }
  setActiveProfilerBuffer(nullptr);
  mappings.SetEnabled(false);
  auto recorded_mappings = mappings.GetAndClear();
  ASSERT_EQ(recorded_mappings.size(), 4);
  for (c10::StreamId logical_id : {0, 32, 64, 96}) {
    const auto& entries = recorded_mappings.at(logical_id + 1);
    ASSERT_EQ(entries.size(), 1);
    EXPECT_EQ(entries[0].stream_id, logical_id);
  }
  auto events = buffer.drain();
  ASSERT_EQ(events.size(), 4);
  EXPECT_EQ(events[0].stream_id, 0);
  EXPECT_EQ(events[1].stream_id, 1);
  EXPECT_EQ(events[2].stream_id, 1);
  EXPECT_EQ(events[3].stream_id, 0);
}

namespace {

// A minimal device op of the given kernel type, execution-ready.
std::shared_ptr<OperationContext> MakeDeviceOp(KernelTypeEnum kt) {
  auto kernel = std::make_unique<at::neuron::testing::MockNeuronKernelExecution>();
  ON_CALL(*kernel, GetKernelType()).WillByDefault(Return(kt));
  auto op = std::make_shared<OperationContext>(std::move(kernel));
  op->MarkExecutionReady();
  return op;
}

}  // namespace

// (a) kHost following an unretired device-XU op must gate; once the prior op's
//     seq retires, it must not.
TEST_F(StreamImplTest, GateA_HostAfterUnretiredDeviceXU) {
  auto host_op = MakeDeviceOp(KernelTypeEnum::kHost);

  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kDeviceXU;
  info.xu = NRTA_XU_COMPUTE;
  info.sequence_id = 10;
  info.valid = true;

  // Prior device op (seq 10) NOT yet retired (completed watermark 5) → gated.
  int queue = nrt::GetHardwareQueue(NRTA_XU_COMPUTE, stream_->stream_id);
  stream_->xu_completed_seqs_.Set(NRTA_XU_COMPUTE, queue, 5);
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(host_op.get(), KernelTypeEnum::kHost,
                                                /*prior_ops_in_flight=*/false,
                                                /*refresh_completion=*/false));

  // Prior device op retired (watermark 10) → not gated.
  stream_->xu_completed_seqs_.Set(NRTA_XU_COMPUTE, queue, 10);
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(host_op.get(), KernelTypeEnum::kHost,
                                                 /*prior_ops_in_flight=*/false,
                                                 /*refresh_completion=*/false));
}

// (b) A device op following an incomplete kHost op must gate until the host
//     completion seq catches up.
TEST_F(StreamImplTest, GateB_DeviceAfterIncompleteHost) {
  auto dev_op = MakeDeviceOp(KernelTypeEnum::kHLO);

  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kHost;
  info.host_completion_seq = 7;
  info.valid = true;

  // Host not yet complete past seq 7 → gated.
  stream_->host_completion_state_.completed_seq_id.store(3, std::memory_order_release);
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO,
                                                /*prior_ops_in_flight=*/false,
                                                /*refresh_completion=*/false));

  // Host completed through seq 7 → not gated.
  stream_->host_completion_state_.completed_seq_id.store(7, std::memory_order_release);
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO,
                                                 /*prior_ops_in_flight=*/false,
                                                 /*refresh_completion=*/false));
}

// (c) A device op on a different XU than the prior device op must gate until
//     that prior op retires; same-XU never gates here.
TEST_F(StreamImplTest, GateC_CrossXUSync) {
  auto dev_op = MakeDeviceOp(KernelTypeEnum::kHLO);  // kHLO → COMPUTE XU
  const auto current_xu = static_cast<nrta_xu_t>(GetXUIndex(KernelTypeEnum::kHLO));

  auto& info = stream_->GetLastScheduledInfo();
  info.kind = LastKind::kDeviceXU;
  // Prior op on a *different* XU.
  info.xu = (current_xu == NRTA_XU_COMPUTE) ? NRTA_XU_TENSOR_OP : NRTA_XU_COMPUTE;
  info.sequence_id = 20;
  info.valid = true;

  int prior_queue = nrt::GetHardwareQueue(info.xu, stream_->stream_id);
  stream_->xu_completed_seqs_.Set(info.xu, prior_queue, 15);  // not retired
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO,
                                                /*prior_ops_in_flight=*/false,
                                                /*refresh_completion=*/false));

  stream_->xu_completed_seqs_.Set(info.xu, prior_queue, 20);  // retired
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO,
                                                 /*prior_ops_in_flight=*/false,
                                                 /*refresh_completion=*/false));

  // Same XU as prior → gate (c) does not apply.
  info.xu = current_xu;
  info.sequence_id = 99;
  stream_->xu_completed_seqs_.Set(current_xu, nrt::GetHardwareQueue(current_xu, stream_->stream_id),
                                  0);
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO,
                                                 /*prior_ops_in_flight=*/false,
                                                 /*refresh_completion=*/false));
}

// (d) A pending cross-stream event_wait gates while a prior op is in flight;
//     a same-stream wait never gates, and neither does a wait with no prior
//     in-flight op.
TEST_F(StreamImplTest, GateD_CrossStreamEventWait) {
  // Cross-stream wait, prior op in flight → gated.
  auto other_stream = std::make_shared<StreamImpl>(0, 2);
  auto cross = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, other_stream.get());
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(cross.get(), KernelTypeEnum::kEvent,
                                                /*prior_ops_in_flight=*/true,
                                                /*refresh_completion=*/false));

  // Cross-stream wait but nothing in flight → not gated.
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(cross.get(), KernelTypeEnum::kEvent,
                                                 /*prior_ops_in_flight=*/false,
                                                 /*refresh_completion=*/false));

  // Same-stream wait, even with prior in flight → not gated (FIFO implies order).
  auto same = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, stream_.get());
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(same.get(), KernelTypeEnum::kEvent,
                                                 /*prior_ops_in_flight=*/true,
                                                 /*refresh_completion=*/false));
}

TEST_F(StreamImplTest, CompletedCrossStreamWaitDoesNotDrainConsumer) {
  auto producer = std::make_shared<StreamImpl>(0, 2);
  NeuronEvent event;
  event.get_impl()->is_recorded.store(true, std::memory_order_relaxed);
  event.get_impl()->recorded_stream_impl.store(producer.get(), std::memory_order_relaxed);
  auto kernel = std::make_unique<EventDirectKernelExecution>(
      "event_wait", event, EventDirectKernelExecution::EventAction::kWait, 0);
  OperationContext op(std::move(kernel));
  op.MarkExecutionReady();

  EXPECT_FALSE(op.IsSchedulable());
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent,
                                                /*prior_ops_in_flight=*/true,
                                                /*refresh_completion=*/false));

  event.complete_event();
  EXPECT_TRUE(op.IsSchedulable());
  EXPECT_EQ(config::IsHostCCEnabled(),
            stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent,
                                              /*prior_ops_in_flight=*/true,
                                              /*refresh_completion=*/false));
}

TEST_F(StreamImplTest, QueuedComputeWaitPreservesEventCompletion) {
  auto producer = std::make_shared<StreamImpl>(0, 0);
  NeuronEvent event;
  event.get_impl()->is_recorded.store(true);
  event.get_impl()->recorded_stream_impl.store(producer.get());
  event.get_impl()->allow_queue_ordering = true;
  auto kernel = std::make_unique<EventDirectKernelExecution>(
      "event_wait", event, EventDirectKernelExecution::EventAction::kWait, 0);
  OperationContext op(std::move(kernel));
  op.MarkExecutionReady();
  auto& info = stream_->GetLastScheduledInfo();
  info.valid = true;
  info.kind = LastKind::kDeviceXU;
  info.xu = NRTA_XU_COMPUTE;
  info.sequence_id = 10;

  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
  event.get_impl()->queued_compute_sequence.store(20);
  EXPECT_EQ(config::IsHostCCEnabled(),
            stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
  EXPECT_FALSE(op.IsSchedulable());
  EXPECT_FALSE(event.query());
  event.complete_event();
  EXPECT_TRUE(op.IsSchedulable());
}

TEST_F(StreamImplTest, QueueWaitCannotHideConsumerHostWork) {
  auto producer = std::make_shared<StreamImpl>(0, 0);
  NeuronEvent event;
  event.get_impl()->is_recorded.store(true);
  event.get_impl()->recorded_stream_impl.store(producer.get());
  event.get_impl()->allow_queue_ordering = true;
  event.get_impl()->queued_compute_sequence.store(20);
  auto kernel = std::make_unique<EventDirectKernelExecution>(
      "event_wait", event, EventDirectKernelExecution::EventAction::kWait, 0);
  OperationContext op(std::move(kernel));
  op.MarkExecutionReady();
  auto& info = stream_->GetLastScheduledInfo();
  info.valid = true;
  info.kind = LastKind::kHost;
  info.host_completion_seq = 1;
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
  info.kind = LastKind::kDeviceXU;
  info.xu = NRTA_XU_TENSOR_OP;
  info.sequence_id = 1;
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
  info.xu = NRTA_XU_COMPUTE;
  EXPECT_EQ(config::IsHostCCEnabled(),
            stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
  event.get_impl()->allow_queue_ordering = false;
  EXPECT_TRUE(stream_->IsScheduleGatedByPriorOp(&op, KernelTypeEnum::kEvent, true, false));
}

// No prior op (last_scheduled_info_ invalid) → never gated, regardless of type.
TEST_F(StreamImplTest, NoPriorOpNeverGates) {
  ASSERT_FALSE(stream_->GetLastScheduledInfo().valid);
  auto host_op = MakeDeviceOp(KernelTypeEnum::kHost);
  auto dev_op = MakeDeviceOp(KernelTypeEnum::kHLO);
  EXPECT_FALSE(
      stream_->IsScheduleGatedByPriorOp(host_op.get(), KernelTypeEnum::kHost, false, false));
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO, false, false));
}

namespace {

// Stand-in for an eager kernel (carries HLO bytes, returns dummy NEFF bytes so
// the cache can be seeded without the real neuronx-cc backend).
class FakeEagerKernel
    : public CompilableKernelExecution,
      public NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  explicit FakeEagerKernel(const std::string& cache_key)
      : CompilableKernelExecution(CompilationCacheKey(cache_key, "", ""), "", "", false),
        NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                   c10::DeviceType::PrivateUse1>("fake_eager_op", {}, {}, 0),
        hlo_bytes_({'H', 'L', 'O', '_'}) {
    hlo_bytes_.insert(hlo_bytes_.end(), cache_key.begin(), cache_key.end());
    GetCompilationCacheKey().SetHloBytesPtr(&hlo_bytes_);
  }

  std::vector<uint8_t> CompileToNeff() const override {
    std::string neff = "NEFF_" + GetCompileKey();
    return std::vector<uint8_t>(neff.begin(), neff.end());
  }

  const std::vector<uint8_t>& GetHloBytes() const override { return hlo_bytes_; }

  // Required (pure virtual); never invoked in these tests.
  bool Schedule(NRT_STATUS* /*nrt_status*/, nrt::SequenceId* /*sequence_id*/,
                c10::StreamId /*stream_id*/) const override {
    return true;
  }

 protected:
  bool ValidateImpl() const override { return true; }

 private:
  std::vector<uint8_t> hlo_bytes_;
};

}  // namespace

// Real CompilationCache backed by temp dirs + mocked NRT, mirroring
// CompilationCacheTest.
class ResolvePersistentCacheKeyTest : public ::testing::Test {
 protected:
  MockNRTSession mock_session_;
  MockNRT* mock_nrt_ = MockNRT::GetInstance();

  void SetUp() override {
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    SaveEnvVar("TORCH_NEURONX_NEFF_CACHE_DIR");
    SaveEnvVar("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR");
    SaveEnvVar("TORCH_NEURONX_NEFF_DISABLE_CACHE");
    SaveEnvVar("NEURON_COMPILE_SERVICE");
    SaveEnvVar("TORCH_NEURONX_NEFF_KEY_FROM_HLO");

    // Force local backend so seeding the cache never calls real neuronx-cc.
    setenv("NEURON_COMPILE_SERVICE", "0", 1);

    temp_dir_ = std::make_unique<TempDirectory>("resolve_pk_test_");
    local_cache_dir_ = std::make_unique<TempDirectory>("resolve_pk_local_test_");
    setenv("TORCH_NEURONX_NEFF_CACHE_DIR", temp_dir_->path().c_str(), 1);
    setenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", local_cache_dir_->path().c_str(), 1);
    unsetenv("TORCH_NEURONX_NEFF_DISABLE_CACHE");

    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override {
    if (cache_) cache_->Clear();
    cache_.reset();
    temp_dir_.reset();
    local_cache_dir_.reset();
    RestoreEnvVars();
  }

  // Construct the cache after selecting the keying mode so it sees the env var.
  void InitCache(bool hlo_content_key) {
    setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", hlo_content_key ? "1" : "0", 1);
    cache_ = std::make_unique<CompilationCache>();
  }

  void SaveEnvVar(const char* name) {
    const char* value = std::getenv(name);
    if (value) saved_env_vars_[name] = value;
  }

  void RestoreEnvVars() {
    for (const char* name : {"TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
                             "TORCH_NEURONX_NEFF_DISABLE_CACHE", "NEURON_COMPILE_SERVICE",
                             "TORCH_NEURONX_NEFF_KEY_FROM_HLO"}) {
      auto it = saved_env_vars_.find(name);
      if (it != saved_env_vars_.end()) {
        setenv(name, it->second.c_str(), 1);
      } else {
        unsetenv(name);
      }
    }
  }

  std::unique_ptr<TempDirectory> temp_dir_;
  std::unique_ptr<TempDirectory> local_cache_dir_;
  std::unique_ptr<CompilationCache> cache_;
  std::map<std::string, std::string> saved_env_vars_;
};

// Kernel with HLO bytes recomputes its key directly; empty cache is not
// consulted.
TEST_F(ResolvePersistentCacheKeyTest, KernelWithHloComputesKeyWithoutCacheEntry) {
  InitCache(/*hlo_content_key=*/true);

  FakeEagerKernel kernel("eager_graph");
  const std::string* expected = kernel.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(expected, nullptr);
  ASSERT_FALSE(expected->empty());

  std::string resolved = ResolvePersistentCacheKey(kernel, *cache_);
  EXPECT_EQ(resolved, *expected);
}

// HLO-key mode: a NeffDirect kernel can't compute its own key (nullptr), so the
// helper recovers it from the CacheEntry the compile step stored.
TEST_F(ResolvePersistentCacheKeyTest, NeffDirectKernelReadsKeyFromCacheEntry) {
  InitCache(/*hlo_content_key=*/true);

  const std::string base_key = "shared_graph_key";

  // Seed the cache as the compile step would.
  FakeEagerKernel compile_kernel(base_key);
  cache_->GetOrCompileNeff(compile_kernel, /*stream_id=*/1);
  const std::string* compiled_pk = compile_kernel.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(compiled_pk, nullptr);
  ASSERT_FALSE(compiled_pk->empty());

  NeffDirectKernelExecution exec_kernel("shared_graph_key", base_key, {}, {}, {}, {}, 0, false);
  ASSERT_TRUE(exec_kernel.GetHloBytes().empty());
  EXPECT_EQ(exec_kernel.GetCompilationCacheKey().GetPersistentKey(), nullptr);

  std::string resolved = ResolvePersistentCacheKey(exec_kernel, *cache_);
  EXPECT_EQ(resolved, *compiled_pk);
}

// Legacy mode: a NeffDirect kernel's own key is non-null but WRONG (empty
// compiler config), so the helper must prefer the matching CacheEntry key.
TEST_F(ResolvePersistentCacheKeyTest, NeffDirectKernelLegacyModePrefersCacheEntry) {
  InitCache(/*hlo_content_key=*/false);

  const std::string base_key = "legacy_graph_key";

  FakeEagerKernel compile_kernel(base_key);
  cache_->GetOrCompileNeff(compile_kernel, /*stream_id=*/1);
  const std::string* compiled_pk = compile_kernel.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(compiled_pk, nullptr);
  ASSERT_FALSE(compiled_pk->empty());

  NeffDirectKernelExecution exec_kernel("legacy_graph_key", base_key, {}, {}, {}, {}, 0, false);

  const std::string* exec_own_pk = exec_kernel.GetCompilationCacheKey().GetPersistentKey();
  ASSERT_NE(exec_own_pk, nullptr);
  EXPECT_NE(*exec_own_pk, *compiled_pk);

  std::string resolved = ResolvePersistentCacheKey(exec_kernel, *cache_);
  EXPECT_EQ(resolved, *compiled_pk);
  EXPECT_NE(resolved, *exec_own_pk);
}

// put_neff_cache() stores a NEFF under a name the cache key cannot derive, because no
// compile step ran to agree on one. The recorded name must win in both keying modes: in
// legacy mode the key derives hash(base_key || config), where no NEFF was ever written.
class StagedNeffHintTest : public ResolvePersistentCacheKeyTest,
                           public ::testing::WithParamInterface<bool> {};

TEST_P(StagedNeffHintTest, StagedNeffResolvesThroughTheHint) {
  InitCache(/*hlo_content_key=*/GetParam());

  const std::string base_key = "staged_graph_key";
  const std::string staged_name = "staged_graph_key";  // put_neff_cache uses the base key
  const std::vector<uint8_t> neff_bytes{1, 2, 3, 4};

  at::neuron::PersistentCacheBackend backend;
  ASSERT_TRUE(backend.Put(staged_name, neff_bytes));
  cache_->RegisterPersistentKeyHint(base_key, staged_name);

  NeffDirectKernelExecution exec_kernel("staged_graph", base_key, {}, {}, {}, {}, 0, false);
  ASSERT_TRUE(exec_kernel.GetHloBytes().empty());

  // Without the hint this reaches NeffDirectKernelExecution::CompileToNeff(), which throws.
  auto entry = cache_->GetOrCompileNeff(exec_kernel, /*stream_id=*/1);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->neff_bytes, neff_bytes);
}

INSTANTIATE_TEST_SUITE_P(BothKeyingModes, StagedNeffHintTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& i) {
                           return i.param ? "HloContentKey" : "LegacyKey";
                         });

// No cache entry (e.g. evicted) → returns empty, not a bogus value.
TEST_F(ResolvePersistentCacheKeyTest, NeffDirectKernelCacheMissReturnsEmpty) {
  InitCache(/*hlo_content_key=*/true);

  NeffDirectKernelExecution exec_kernel("missing_graph", "missing_key", {}, {}, {}, {}, 0, false);

  std::string resolved = ResolvePersistentCacheKey(exec_kernel, *cache_);
  EXPECT_TRUE(resolved.empty());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
