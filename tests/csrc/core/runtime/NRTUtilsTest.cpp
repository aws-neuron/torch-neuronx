#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"

namespace nrt = at::neuron::nrt;
using namespace at::neuron::nrt;
using namespace torch_neuronx::testing;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class NRTUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<MockNRTSession>();
    engine_ = &at::neuron::NeuronResourceManager::Instance().GetOperationExecutionEngine();
    original_async_mode_ = engine_->IsNRTAsyncModeEnabled();
  }

  void TearDown() override {
    // Restore original async mode
    engine_->SetAsyncModeEnabled(original_async_mode_);
    mock_session_.reset();
  }

  std::unique_ptr<MockNRTSession> mock_session_;
  at::neuron::OperationExecutionEngine* engine_{nullptr};
  bool original_async_mode_{false};
};

// TensorSet Tests
TEST_F(NRTUtilsTest, TensorSetConstruction) {
  nrt::TensorSetHandle* fake_set = reinterpret_cast<nrt::TensorSetHandle*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(1);

  auto tensor_set = CreateTensorSet();
  EXPECT_NE(tensor_set, nullptr);
  EXPECT_EQ(tensor_set.get(), fake_set);
}

TEST_F(NRTUtilsTest, TensorSetConstructionFailure) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_)).WillOnce(Return(NRT_FAILURE));

  EXPECT_THROW({ auto tensor_set = CreateTensorSet(); }, std::runtime_error);
}

TEST_F(NRTUtilsTest, TensorSetMoveConstructor) {
  nrt::TensorSetHandle* fake_set = reinterpret_cast<nrt::TensorSetHandle*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(1);

  auto tensor_set1 = CreateTensorSet();
  EXPECT_NE(tensor_set1, nullptr);

  TensorSetPtr tensor_set2(std::move(tensor_set1));
  EXPECT_EQ(tensor_set1, nullptr);
  EXPECT_NE(tensor_set2, nullptr);
  EXPECT_EQ(tensor_set2.get(), fake_set);
}

TEST_F(NRTUtilsTest, TensorSetMoveAssignment) {
  nrt::TensorSetHandle* fake_set1 = reinterpret_cast<nrt::TensorSetHandle*>(0x1234);
  nrt::TensorSetHandle* fake_set2 = reinterpret_cast<nrt::TensorSetHandle*>(0x5678);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(2);

  auto tensor_set1 = CreateTensorSet();
  auto tensor_set2 = CreateTensorSet();

  tensor_set2 = std::move(tensor_set1);
  EXPECT_EQ(tensor_set1, nullptr);
  EXPECT_NE(tensor_set2, nullptr);
  EXPECT_EQ(tensor_set2.get(), fake_set1);
}

// Model Tests
TEST_F(NRTUtilsTest, ModelLoad) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};  // "NEFF"

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, neff_bytes.size(), 0, 1, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  NRT_STATUS status = LoadModel(model, neff_bytes, 0, 1);

  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(model.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelLoadFailure) {
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _)).WillOnce(Return(NRT_FAILURE));

  ModelPtr model;
  NRT_STATUS status = LoadModel(model, neff_bytes, 0, 1);
  EXPECT_NE(status, NRT_SUCCESS);
  EXPECT_EQ(model, nullptr);
}

TEST_F(NRTUtilsTest, ModelLoadCollectives) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load_collectives(_, neff_bytes.size(), 0, 1, 0, 2, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  NRT_STATUS status = LoadModelCollectives(model, neff_bytes, 0, 1, 0, 2);

  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(model.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelExecute) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::TensorSetHandle* fake_input = reinterpret_cast<nrt::TensorSetHandle*>(0x1111);
  nrt::TensorSetHandle* fake_output = reinterpret_cast<nrt::TensorSetHandle*>(0x2222);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_execute(fake_model, fake_input, fake_output))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);
  EXPECT_NO_THROW(DispatchExecution(model.get(), fake_input, fake_output));
}

TEST_F(NRTUtilsTest, ModelExecuteFailure) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::TensorSetHandle* fake_input = reinterpret_cast<nrt::TensorSetHandle*>(0x1111);
  nrt::TensorSetHandle* fake_output = reinterpret_cast<nrt::TensorSetHandle*>(0x2222);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_execute(fake_model, fake_input, fake_output))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);
  NRT_STATUS status = DispatchExecution(model.get(), fake_input, fake_output);
  EXPECT_EQ(status, NRT_FAILURE);
}

TEST_F(NRTUtilsTest, ModelExecuteWithoutLoad) {
  nrt::TensorSetHandle* fake_input = reinterpret_cast<nrt::TensorSetHandle*>(0x1111);
  nrt::TensorSetHandle* fake_output = reinterpret_cast<nrt::TensorSetHandle*>(0x2222);

  EXPECT_THROW(DispatchExecution(nullptr, fake_input, fake_output), std::runtime_error);
}

TEST_F(NRTUtilsTest, ModelExecuteWithNullTensorSets) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);

  EXPECT_THROW(DispatchExecution(model.get(), nullptr, nullptr), std::invalid_argument);
}

// GetLastCompletedRequest tests - verifies kernel_type_index to XU mapping

TEST_F(NRTUtilsTest, GetLastCompletedRequestHLO) {
  nrta_seq_t expected_seq = 42;
  // kHLO (index 0) -> NRTA_XU_COMPUTE
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(0, NRTA_XU_COMPUTE, 0, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(0, 0, 0, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestCollective) {
  nrta_seq_t expected_seq = 100;
  // kCollective (index 1) -> NRTA_XU_COLLECTIVES
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(1, NRTA_XU_COLLECTIVES, 0, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(1, 1, 0, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestCopy) {
  nrta_seq_t expected_seq = 200;
  // kCopy (index 2) -> NRTA_XU_TENSOR_OP
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(2, NRTA_XU_TENSOR_OP, 0, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(2, 2, 0, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestWrite) {
  nrta_seq_t expected_seq = 300;
  // kWrite (index 3) -> NRTA_XU_TENSOR_WRITE
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(0, NRTA_XU_TENSOR_WRITE, 0, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(0, 3, 0, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestRead) {
  nrta_seq_t expected_seq = 400;
  // kRead (index 4) -> NRTA_XU_TENSOR_READ
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(0, NRTA_XU_TENSOR_READ, 0, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(0, 4, 0, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestWithDifferentQueues) {
  nrta_seq_t expected_seq = 500;
  // Test with queue 1 instead of default 0
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(0, NRTA_XU_COMPUTE, 1, _))
      .WillOnce(DoAll(SetArgPointee<3>(expected_seq), Return(NRT_SUCCESS)));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(0, 0, 1, &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, GetLastCompletedRequestFailure) {
  // Test failure case
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_get_sequence(_, _, _, _)).WillOnce(Return(NRT_FAILURE));

  nrta_seq_t result_seq;
  NRT_STATUS status = GetLastCompletedRequest(0, 0, 0, &result_seq);
  EXPECT_EQ(status, NRT_FAILURE);
}

// ScheduleTensorWrite tests

TEST_F(NRTUtilsTest, ScheduleTensorWriteSuccess) {
  nrt_tensor_t* fake_tensor = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x5678);
  char buffer[64] = {0};
  nrta_seq_t expected_seq = 123;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(),
              nrta_tensor_write(fake_tensor, buffer, 0, 64, 0, fake_tracker_ptr, _))
      .WillOnce(DoAll(SetArgPointee<6>(expected_seq), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status =
      ScheduleTensorWrite(fake_tensor, buffer, 0, 64, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, ScheduleTensorWriteFailure) {
  nrt_tensor_t* fake_tensor = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x5678);
  char buffer[64] = {0};

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_tensor_write(_, _, _, _, _, _, _))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status =
      ScheduleTensorWrite(fake_tensor, buffer, 0, 64, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_FAILURE);
}

// ScheduleTensorRead tests

TEST_F(NRTUtilsTest, ScheduleTensorReadSuccess) {
  nrt_tensor_t* fake_tensor = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x5678);
  char buffer[64] = {0};
  nrta_seq_t expected_seq = 456;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(),
              nrta_tensor_read(buffer, fake_tensor, 0, 64, 0, fake_tracker_ptr, _))
      .WillOnce(DoAll(SetArgPointee<6>(expected_seq), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status = ScheduleTensorRead(buffer, fake_tensor, 0, 64, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, ScheduleTensorReadFailure) {
  nrt_tensor_t* fake_tensor = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x5678);
  char buffer[64] = {0};

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_tensor_read(_, _, _, _, _, _, _))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status = ScheduleTensorRead(buffer, fake_tensor, 0, 64, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_FAILURE);
}

// ScheduleTensorCopy tests

TEST_F(NRTUtilsTest, ScheduleTensorCopySuccess) {
  nrt_tensor_t* fake_src = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt_tensor_t* fake_dst = reinterpret_cast<nrt_tensor_t*>(0x5678);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0xABCD);
  nrta_seq_t expected_seq = 789;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(),
              nrta_tensor_copy(fake_src, 0, fake_dst, 0, 128, 0, fake_tracker_ptr, _))
      .WillOnce(DoAll(SetArgPointee<7>(expected_seq), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status =
      ScheduleTensorCopy(fake_src, 0, fake_dst, 0, 128, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, ScheduleTensorCopyFailure) {
  nrt_tensor_t* fake_src = reinterpret_cast<nrt_tensor_t*>(0x1234);
  nrt_tensor_t* fake_dst = reinterpret_cast<nrt_tensor_t*>(0x5678);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0xABCD);

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_tensor_copy(_, _, _, _, _, _, _, _))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status =
      ScheduleTensorCopy(fake_src, 0, fake_dst, 0, 128, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_FAILURE);
}

// IsNRTAsyncRequestCompleted tests

TEST_F(NRTUtilsTest, IsNRTAsyncRequestCompletedTrue) {
  nrta_seq_t seq_id = 100;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_is_completed(seq_id, _))
      .WillOnce(DoAll(SetArgPointee<1>(true), Return(NRT_SUCCESS)));

  bool is_completed = false;
  NRT_STATUS status = IsNRTAsyncRequestCompleted(seq_id, &is_completed);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_TRUE(is_completed);
}

TEST_F(NRTUtilsTest, IsNRTAsyncRequestCompletedFalse) {
  nrta_seq_t seq_id = 200;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_is_completed(seq_id, _))
      .WillOnce(DoAll(SetArgPointee<1>(false), Return(NRT_SUCCESS)));

  bool is_completed = true;
  NRT_STATUS status = IsNRTAsyncRequestCompleted(seq_id, &is_completed);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_FALSE(is_completed);
}

TEST_F(NRTUtilsTest, IsNRTAsyncRequestCompletedInvalidSeq) {
  nrta_seq_t invalid_seq = 999;

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_is_completed(invalid_seq, _))
      .WillOnce(Return(NRT_INVALID));

  bool is_completed = false;
  NRT_STATUS status = IsNRTAsyncRequestCompleted(invalid_seq, &is_completed);
  EXPECT_EQ(status, NRT_INVALID);
}

// ErrorTracker tests

TEST_F(NRTUtilsTest, ErrorTrackerConstruction) {
  nrt::ErrorTrackerHandle* fake_tracker = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker)).Times(1);

  auto tracker = CreateErrorTracker(0);
  EXPECT_NE(tracker, nullptr);
  EXPECT_EQ(tracker.get(), fake_tracker);
}

TEST_F(NRTUtilsTest, ErrorTrackerConstructionFailure) {
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(Return(NRT_FAILURE));

  EXPECT_THROW(CreateErrorTracker(0), std::runtime_error);
}

TEST_F(NRTUtilsTest, ErrorTrackerMoveConstructor) {
  nrt::ErrorTrackerHandle* fake_tracker = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker)).Times(1);

  auto tracker1 = CreateErrorTracker(0);
  EXPECT_NE(tracker1, nullptr);

  ErrorTrackerPtr tracker2(std::move(tracker1));
  EXPECT_EQ(tracker1, nullptr);
  EXPECT_NE(tracker2, nullptr);
  EXPECT_EQ(tracker2.get(), fake_tracker);
}

TEST_F(NRTUtilsTest, ErrorTrackerMoveAssignment) {
  nrt::ErrorTrackerHandle* fake_tracker1 = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x1234);
  nrt::ErrorTrackerHandle* fake_tracker2 = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x5678);

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker1), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(1, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(_)).Times(2);

  auto tracker1 = CreateErrorTracker(0);
  auto tracker2 = CreateErrorTracker(1);

  tracker2 = std::move(tracker1);
  EXPECT_EQ(tracker1, nullptr);
  EXPECT_NE(tracker2, nullptr);
  EXPECT_EQ(tracker2.get(), fake_tracker1);
}

TEST_F(NRTUtilsTest, ErrorTrackerGetAndClearErrorsEmpty) {
  nrt::ErrorTrackerHandle* fake_tracker = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x1234);

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_get_list(fake_tracker, _, _))
      .WillOnce(DoAll(SetArgPointee<1>(nullptr), SetArgPointee<2>(0), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker)).Times(1);

  auto tracker = CreateErrorTracker(0);
  auto errors = GetAndClearErrors(tracker);
  EXPECT_TRUE(errors.empty());
}

TEST_F(NRTUtilsTest, ErrorTrackerGetAndClearErrorsWithErrors) {
  nrt::ErrorTrackerHandle* fake_tracker = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x1234);

  // Create fake error list
  static nrta_error_t fake_errors[2] = {{100, 1005}, {200, 6}};

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_get_list(fake_tracker, _, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_errors), SetArgPointee<2>(2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker)).Times(1);

  auto tracker = CreateErrorTracker(0);
  auto errors = GetAndClearErrors(tracker);
  EXPECT_EQ(errors.size(), 2);
  EXPECT_EQ(errors[0].seq_id, 100);
  EXPECT_EQ(errors[0].error_code, 1005);
  EXPECT_EQ(errors[1].seq_id, 200);
  EXPECT_EQ(errors[1].error_code, 6);
}

// DispatchExecution async tests

TEST_F(NRTUtilsTest, ModelDispatchExecutionAsync) {
  // Enable async mode for this test
  engine_->SetAsyncModeEnabled(true);

  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::TensorSetHandle* fake_input = reinterpret_cast<nrt::TensorSetHandle*>(0x1111);
  nrt::TensorSetHandle* fake_output = reinterpret_cast<nrt::TensorSetHandle*>(0x2222);
  nrt::ErrorTrackerHandle* fake_tracker_ptr = reinterpret_cast<nrt::ErrorTrackerHandle*>(0x3333);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};
  nrta_seq_t expected_seq = 999;

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_create(0, _))
      .WillOnce(DoAll(SetArgPointee<1>(fake_tracker_ptr), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(),
              nrta_execute_schedule(fake_model, fake_input, fake_output, 0, fake_tracker_ptr, _))
      .WillOnce(DoAll(SetArgPointee<5>(expected_seq), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);
  EXPECT_CALL(*MockNRT::GetInstance(), nrta_error_tracker_destroy(fake_tracker_ptr)).Times(1);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);

  auto tracker = CreateErrorTracker(0);
  nrta_seq_t result_seq;
  NRT_STATUS status =
      DispatchExecution(model.get(), fake_input, fake_output, 0, tracker.get(), &result_seq);
  EXPECT_EQ(status, NRT_SUCCESS);
  EXPECT_EQ(result_seq, expected_seq);
}

TEST_F(NRTUtilsTest, ModelDispatchExecutionSync) {
  // Ensure async mode is disabled for this test
  engine_->SetAsyncModeEnabled(false);

  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::TensorSetHandle* fake_input = reinterpret_cast<nrt::TensorSetHandle*>(0x1111);
  nrt::TensorSetHandle* fake_output = reinterpret_cast<nrt::TensorSetHandle*>(0x2222);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  // When async mode is disabled, should call sync nrt_execute
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_execute(fake_model, fake_input, fake_output))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);

  NRT_STATUS status = DispatchExecution(model.get(), fake_input, fake_output, 0, nullptr, nullptr);
  EXPECT_EQ(status, NRT_SUCCESS);
}

TEST_F(NRTUtilsTest, ModelMoveConstructor) {
  nrt::ModelHandle* fake_model = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(fake_model)).Times(1);

  ModelPtr model1;
  LoadModel(model1, neff_bytes, 0, 1);
  EXPECT_NE(model1, nullptr);

  ModelPtr model2(std::move(model1));
  EXPECT_EQ(model1, nullptr);
  EXPECT_NE(model2, nullptr);
  EXPECT_EQ(model2.get(), fake_model);
}

TEST_F(NRTUtilsTest, ModelMoveAssignment) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  ModelPtr model1;
  LoadModel(model1, neff_bytes, 0, 1);
  ModelPtr model2;
  LoadModel(model2, neff_bytes, 1, 1);

  model2 = std::move(model1);
  EXPECT_EQ(model1, nullptr);
  EXPECT_NE(model2, nullptr);
  EXPECT_EQ(model2.get(), fake_model1);
}

TEST_F(NRTUtilsTest, ModelReassignment) {
  nrt::ModelHandle* fake_model1 = reinterpret_cast<nrt::ModelHandle*>(0xABCD);
  nrt::ModelHandle* fake_model2 = reinterpret_cast<nrt::ModelHandle*>(0xDCBA);
  std::vector<uint8_t> neff_bytes = {0x4E, 0x45, 0x46, 0x46};

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_load(_, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model1), Return(NRT_SUCCESS)))
      .WillOnce(DoAll(SetArgPointee<4>(fake_model2), Return(NRT_SUCCESS)));

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_unload(_)).Times(2);

  ModelPtr model;
  LoadModel(model, neff_bytes, 0, 1);
  EXPECT_EQ(model.get(), fake_model1);

  // Reassigning unloads the first model
  LoadModel(model, neff_bytes, 1, 1);
  EXPECT_EQ(model.get(), fake_model2);
}

TEST_F(NRTUtilsTest, AddTensorToSetSuccess) {
  nrt::TensorSetHandle* fake_set = reinterpret_cast<nrt::TensorSetHandle*>(0x1234);
  nrt_tensor_t* fake_tensor = reinterpret_cast<nrt_tensor_t*>(0x5678);

  EXPECT_CALL(*MockNRT::GetInstance(), nrt_allocate_tensor_set(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_set), Return(NRT_SUCCESS)));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_add_tensor_to_tensor_set(fake_set, _, fake_tensor))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*MockNRT::GetInstance(), nrt_destroy_tensor_set(_)).Times(1);

  auto set = CreateTensorSet();
  EXPECT_EQ(AddTensorToSet(set, "input0", fake_tensor), NRT_SUCCESS);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
