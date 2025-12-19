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

  std::vector<TensorContext> input_contexts, output_contexts;
  for (const auto& t : inputs) input_contexts.push_back(TensorContext::FromTensor(t));
  for (const auto& t : outputs) output_contexts.push_back(TensorContext::FromTensor(t));
  OperationContext mock_op(std::move(mock_kernel), input_contexts, output_contexts,
                           "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Execute CPU fallback - this will fail due to mock operation
  auto result = error_recovery_->ExecuteCpuFallback(&mock_op);

  // The result should indicate failure for mock operation
  EXPECT_FALSE(result.IsSuccess());
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

    std::vector<TensorContext> input_contexts, output_contexts;
    for (const auto& t : inputs) input_contexts.push_back(TensorContext::FromTensor(t));
    for (const auto& t : outputs) output_contexts.push_back(TensorContext::FromTensor(t));
    auto mock_op = std::make_unique<OperationContext>(std::move(mock_kernel), input_contexts,
                                                      output_contexts, "test_stack_trace");
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
