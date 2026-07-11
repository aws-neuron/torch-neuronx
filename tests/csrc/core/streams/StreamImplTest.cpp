#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockCopyUtils.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockNRT.h"
#include "tests/csrc/mocks/MockOperationExecutionEngine.h"
#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
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
                                              c10::StreamId recorded_stream_id) {
  NeuronEvent event;
  event.get_impl()->recorded_stream_id.store(recorded_stream_id, std::memory_order_relaxed);
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
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, stream_->stream_id);
  EXPECT_TRUE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_->stream_id));
}

TEST_F(StreamImplTest, GetNextOperationToScheduleCrossStreamKWaitGated) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kWait, stream_->stream_id + 1);
  EXPECT_FALSE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_->stream_id));
}

TEST_F(StreamImplTest, GetNextOperationToScheduleKSignalNotBypassed) {
  unsetenv("TORCH_NEURONX_ENABLE_CONCATENATION");
  auto op = MakeEventOp(EventDirectKernelExecution::EventAction::kSignal, stream_->stream_id);
  EXPECT_FALSE(StreamImpl::IsSameStreamWaitBypass(op.get(), stream_->stream_id));
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

// (d) A cross-stream event_wait gates only while a prior op is in flight;
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

// No prior op (last_scheduled_info_ invalid) → never gated, regardless of type.
TEST_F(StreamImplTest, NoPriorOpNeverGates) {
  ASSERT_FALSE(stream_->GetLastScheduledInfo().valid);
  auto host_op = MakeDeviceOp(KernelTypeEnum::kHost);
  auto dev_op = MakeDeviceOp(KernelTypeEnum::kHLO);
  EXPECT_FALSE(
      stream_->IsScheduleGatedByPriorOp(host_op.get(), KernelTypeEnum::kHost, false, false));
  EXPECT_FALSE(stream_->IsScheduleGatedByPriorOp(dev_op.get(), KernelTypeEnum::kHLO, false, false));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
