#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
#include "torch_neuronx/csrc/core/utils/ErrorRecovery.h"

using namespace at::neuron;
namespace mock = at::neuron::testing;

class ErrorRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear environment variables for consistent testing
    unsetenv("NEURON_FALLBACK_ENABLED");

    error_recovery_ = std::make_unique<ErrorRecovery>();
    mock_stream_.SetDefaultBehavior();

    // Set up default expectation for WaitForPriorOperationsToComplete
    EXPECT_CALL(mock_stream_, WaitForPriorOperationsToComplete(::testing::_))
        .WillRepeatedly(::testing::Return());
  }

  void TearDown() override {
    // Clean up environment
    unsetenv("NEURON_FALLBACK_ENABLED");
  }

  std::unique_ptr<ErrorRecovery> error_recovery_;
  mock::MockStreamImpl mock_stream_;
};

// Test ErrorRecovery constructor and configuration
TEST_F(ErrorRecoveryTest, ConstructorAndConfiguration) {
  // Test 1: Default configuration (fallback enabled)
  {
    ErrorRecovery recovery;
    EXPECT_TRUE(recovery.ShouldAttemptCpuFallback());
  }

  // Test 2: Fallback disabled with "0"
  {
    setenv("NEURON_FALLBACK_ENABLED", "0", 1);
    ErrorRecovery recovery;
    EXPECT_FALSE(recovery.ShouldAttemptCpuFallback());
    unsetenv("NEURON_FALLBACK_ENABLED");
  }

  // Test 3: Fallback enabled with "true"
  {
    setenv("NEURON_FALLBACK_ENABLED", "true", 1);
    ErrorRecovery recovery;
    EXPECT_TRUE(recovery.ShouldAttemptCpuFallback());
    unsetenv("NEURON_FALLBACK_ENABLED");
  }

  // Test 4: Fallback enabled with "1"
  {
    setenv("NEURON_FALLBACK_ENABLED", "1", 1);
    ErrorRecovery recovery;
    EXPECT_TRUE(recovery.ShouldAttemptCpuFallback());
    unsetenv("NEURON_FALLBACK_ENABLED");
  }
}

// Test should_attempt_cpu_fallback method
TEST_F(ErrorRecoveryTest, ShouldAttemptCpuFallback) {
  // Test 1: Enabled (default configuration)
  EXPECT_TRUE(error_recovery_->ShouldAttemptCpuFallback());

  // Test 2: Disabled configuration
  setenv("NEURON_FALLBACK_ENABLED", "0", 1);
  ErrorRecovery disabled_recovery;
  EXPECT_FALSE(disabled_recovery.ShouldAttemptCpuFallback());
  unsetenv("NEURON_FALLBACK_ENABLED");
}

// Test execute_cpu_fallback method
TEST_F(ErrorRecoveryTest, ExecuteCpuFallbackWithMockOperation) {
  // Create a mock operation for testing
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Execute CPU fallback - this will fail due to mock operation
  auto result = error_recovery_->ExecuteCpuFallback(&mock_op);

  // The result should indicate failure for mock operation
  EXPECT_FALSE(result.IsSuccess());
}

// Test RecoverConcatenationFailure function
TEST_F(ErrorRecoveryTest, RecoverConcatenationFailureWithNullOperation) {
  // Test that RecoverConcatenationFailure handles null operation gracefully
  RecoverConcatenationFailure(nullptr);
  // Should not crash - just debug log and return
}

TEST_F(ErrorRecoveryTest, RecoverConcatenationFailureWithNonConcatenatedOperation) {
  // Create a mock operation that is NOT concatenated
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);
  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();
  mock_op.stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Operation without concatenation state should handle gracefully
  EXPECT_FALSE(mock_op.HasConcatenatedOperation());
  EXPECT_FALSE(mock_op.IsConcatenatedOperation());

  // Should not crash
  RecoverConcatenationFailure(&mock_op);
}

TEST_F(ErrorRecoveryTest, RecoverConcatenationFailureWithConcatenatedOperation) {
  // Create cascading operations (individual ops that will be merged)
  std::vector<torch::Tensor> inputs1 = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs1 = {torch::ones({2, 3})};
  auto mock_kernel1 =
      std::make_unique<mock::MockNeuronKernelExecution>("cascading_op1", inputs1, outputs1);
  EXPECT_CALL(*mock_kernel1, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel1, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  auto cascading_op1 =
      std::make_unique<OperationContext>(std::move(mock_kernel1), "cascading_stack_trace1");
  cascading_op1->submit_time = std::chrono::steady_clock::now();
  cascading_op1->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  std::vector<torch::Tensor> inputs2 = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs2 = {torch::ones({2, 3})};
  auto mock_kernel2 =
      std::make_unique<mock::MockNeuronKernelExecution>("cascading_op2", inputs2, outputs2);
  EXPECT_CALL(*mock_kernel2, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel2, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  auto cascading_op2 =
      std::make_unique<OperationContext>(std::move(mock_kernel2), "cascading_stack_trace2");
  cascading_op2->submit_time = std::chrono::steady_clock::now();
  cascading_op2->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Create the concatenated (merged) operation
  std::vector<torch::Tensor> concat_inputs = {torch::ones({4, 3})};
  std::vector<torch::Tensor> concat_outputs = {torch::ones({4, 3})};
  auto concat_kernel =
      std::make_unique<mock::MockNeuronKernelExecution>("concat_op", concat_inputs, concat_outputs);
  EXPECT_CALL(*concat_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*concat_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  auto concat_op =
      std::make_unique<OperationContext>(std::move(concat_kernel), "concat_stack_trace");
  concat_op->submit_time = std::chrono::steady_clock::now();
  concat_op->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Store raw pointers for cascading ops vector
  std::vector<OperationContext*> cascading_ops = {cascading_op1.get(), cascading_op2.get()};

  // Track whether the failure callback was invoked
  bool callback_invoked = false;
  OperationContext* callback_received_op = nullptr;

  // Create ConcatenationState with failure callback
  auto concatenation_state = std::make_shared<ConcatenationState>(
      std::move(concat_op), cascading_ops, [&](OperationContext* failed_op) {
        callback_invoked = true;
        callback_received_op = failed_op;
      });

  // Set up the raw pointer for the concatenated operation (after move)
  OperationContext* concat_op_ptr = concatenation_state->GetConcatenatedOperation();
  concat_op_ptr->concatenation_state_raw_ = concatenation_state.get();

  // Set up shared_ptr for cascading operations
  cascading_op1->concatenation_state_ = concatenation_state;
  cascading_op2->concatenation_state_ = concatenation_state;

  // Verify concatenation state is set up correctly
  EXPECT_TRUE(cascading_op1->HasConcatenatedOperation());
  EXPECT_FALSE(cascading_op1->IsConcatenatedOperation());
  EXPECT_TRUE(cascading_op2->HasConcatenatedOperation());
  EXPECT_FALSE(cascading_op2->IsConcatenatedOperation());
  EXPECT_TRUE(concat_op_ptr->IsConcatenatedOperation());

  // Test 1: RecoverConcatenationFailure on cascading operation should invoke callback
  RecoverConcatenationFailure(cascading_op1.get());
  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(callback_received_op, concat_op_ptr);

  // Reset for next test
  callback_invoked = false;
  callback_received_op = nullptr;

  // Test 2: RecoverConcatenationFailure on concatenated operation itself should invoke callback
  RecoverConcatenationFailure(concat_op_ptr);
  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(callback_received_op, concat_op_ptr);
}

// Test attempt_recovery method (main entry point)
TEST_F(ErrorRecoveryTest, AttemptRecovery) {
  // Helper function to create mock operation
  auto create_mock_operation = []() {
    std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
    std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

    auto mock_kernel =
        std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);
    EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

    auto mock_op = std::make_unique<OperationContext>(std::move(mock_kernel), "test_stack_trace");
    mock_op->submit_time = std::chrono::steady_clock::now();
    return mock_op;
  };

  // Test 1: Fallback enabled (default) - should try CPU fallback
  {
    auto mock_op = create_mock_operation();
    mock_op->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

    torch_neuronx::ExecutionRuntimeException exec_error("Test execution error");
    auto result = error_recovery_->AttemptRecovery(mock_op.get(), exec_error);

    EXPECT_FALSE(result.IsSuccess());  // Mock operation will fail
  }

  // Test 2: Fallback disabled - should not try CPU fallback
  {
    setenv("NEURON_FALLBACK_ENABLED", "0", 1);
    ErrorRecovery disabled_recovery;

    auto mock_op = create_mock_operation();
    mock_op->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

    torch_neuronx::ExecutionRuntimeException exec_error("Test execution error");
    auto result = disabled_recovery.AttemptRecovery(mock_op.get(), exec_error);

    EXPECT_EQ(result.status, OperationContextResult::kPending);  // Default result

    unsetenv("NEURON_FALLBACK_ENABLED");
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
