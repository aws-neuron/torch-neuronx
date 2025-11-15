#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuronx/csrc/core/utils/ErrorRecovery.h"

using namespace at::neuron;
namespace mock = at::neuron::testing;

class ErrorRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear environment variables for consistent testing
    unsetenv("NEURON_FALLBACK_ENABLED");

    error_recovery_ = std::make_unique<ErrorRecovery>();
  }

  void TearDown() override {
    // Clean up environment
    unsetenv("NEURON_FALLBACK_ENABLED");
  }

  std::unique_ptr<ErrorRecovery> error_recovery_;
};

// Test ErrorRecovery constructor and configuration
TEST_F(ErrorRecoveryTest, ConstructorAndConfiguration) {
  // Test 1: Default configuration (fallback enabled)
  {
    ErrorRecovery recovery;
    EXPECT_TRUE(recovery.ShouldAttemptCpuFallback());

    auto stats = recovery.GetRecoveryStats();
    EXPECT_EQ(stats.total_fallback_attempts, 1);  // should_attempt_cpu_fallback increments this
    EXPECT_EQ(stats.successful_fallbacks, 0);
    EXPECT_EQ(stats.failed_fallbacks, 0);
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
  auto stats = error_recovery_->GetRecoveryStats();
  EXPECT_EQ(stats.total_fallback_attempts, 1);

  // Test 2: Disabled configuration
  setenv("NEURON_FALLBACK_ENABLED", "0", 1);
  ErrorRecovery disabled_recovery;
  EXPECT_FALSE(disabled_recovery.ShouldAttemptCpuFallback());

  auto disabled_stats = disabled_recovery.GetRecoveryStats();
  EXPECT_EQ(disabled_stats.total_fallback_attempts, 0);  // Should not increment when disabled

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

  // Verify that the fallback was attempted and stats were updated
  auto stats = error_recovery_->GetRecoveryStats();
  EXPECT_EQ(stats.total_fallback_attempts, 0);  // execute_cpu_fallback doesn't increment this
  EXPECT_EQ(stats.failed_fallbacks, 1);         // One of these should be incremented

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

    auto mock_op = std::make_unique<OperationContext>(std::move(mock_kernel), "test_stack_trace");
    mock_op->submit_time = std::chrono::steady_clock::now();
    return mock_op;
  };

  // Test 1: Fallback enabled (default) - should try CPU fallback
  {
    auto mock_op = create_mock_operation();
    auto result = error_recovery_->AttemptRecovery(mock_op.get());

    auto stats = error_recovery_->GetRecoveryStats();
    EXPECT_GT(stats.total_fallback_attempts, 0);
    EXPECT_FALSE(result.IsSuccess());  // Mock operation will fail
  }

  // Test 2: Fallback disabled - should not try CPU fallback
  {
    setenv("NEURON_FALLBACK_ENABLED", "0", 1);
    ErrorRecovery disabled_recovery;

    auto mock_op = create_mock_operation();
    auto result = disabled_recovery.AttemptRecovery(mock_op.get());

    auto stats = disabled_recovery.GetRecoveryStats();
    EXPECT_EQ(stats.total_fallback_attempts, 0);
    EXPECT_EQ(result.status, OperationContextResult::kPending);  // Default result

    unsetenv("NEURON_FALLBACK_ENABLED");
  }
}

// Test recovery statistics functionality
TEST_F(ErrorRecoveryTest, RecoveryStatistics) {
  // Test 1: Initial state - all counters should be zero
  auto stats = error_recovery_->GetRecoveryStats();
  EXPECT_EQ(stats.total_fallback_attempts, 0);
  EXPECT_EQ(stats.successful_fallbacks, 0);
  EXPECT_EQ(stats.failed_fallbacks, 0);
  EXPECT_EQ(stats.fallback_success_rate(), 0.0);

  // Test 2: Generate some statistics and test reset
  error_recovery_->ShouldAttemptCpuFallback();
  auto stats_before = error_recovery_->GetRecoveryStats();
  EXPECT_GT(stats_before.total_fallback_attempts, 0);

  error_recovery_->ResetRecoveryStats();
  auto stats_after = error_recovery_->GetRecoveryStats();
  EXPECT_EQ(stats_after.total_fallback_attempts, 0);
  EXPECT_EQ(stats_after.successful_fallbacks, 0);
  EXPECT_EQ(stats_after.failed_fallbacks, 0);

  // Test 3: RecoveryStats utility methods
  ErrorRecovery::RecoveryStats test_stats;

  // No attempts
  test_stats.total_fallback_attempts = 0;
  test_stats.successful_fallbacks = 0;
  test_stats.failed_fallbacks = 0;
  EXPECT_EQ(test_stats.fallback_success_rate(), 0.0);

  // Some attempts (70% success rate)
  test_stats.total_fallback_attempts = 10;
  test_stats.successful_fallbacks = 7;
  test_stats.failed_fallbacks = 3;
  EXPECT_DOUBLE_EQ(test_stats.fallback_success_rate(), 0.7);

  // All failures
  test_stats.total_fallback_attempts = 5;
  test_stats.successful_fallbacks = 0;
  test_stats.failed_fallbacks = 5;
  EXPECT_EQ(test_stats.fallback_success_rate(), 0.0);

  // All successes
  test_stats.total_fallback_attempts = 8;
  test_stats.successful_fallbacks = 8;
  test_stats.failed_fallbacks = 0;
  EXPECT_EQ(test_stats.fallback_success_rate(), 1.0);
}

// Test multiple recovery attempts to verify statistics tracking
TEST_F(ErrorRecoveryTest, MultipleRecoveryAttempts) {
  // Create multiple mock operations and attempt recovery
  for (int i = 0; i < 3; ++i) {
    std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
    std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

    auto mock_kernel =
        std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

    EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

    OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
    mock_op.submit_time = std::chrono::steady_clock::now();

    error_recovery_->AttemptRecovery(&mock_op);
  }

  // Verify statistics were accumulated
  auto stats = error_recovery_->GetRecoveryStats();
  EXPECT_EQ(stats.total_fallback_attempts, 3);
  EXPECT_EQ(stats.successful_fallbacks + stats.failed_fallbacks, 3);

  // Calculate success rate
  double expected_rate = double(stats.successful_fallbacks) / double(stats.total_fallback_attempts);
  EXPECT_EQ(stats.fallback_success_rate(), expected_rate);
}

// Test thread safety of statistics (basic test)
TEST_F(ErrorRecoveryTest, StatisticsThreadSafety) {
  // This is a basic test - in a real scenario you'd want more comprehensive threading tests

  // Get stats multiple times to ensure no race conditions in basic usage
  auto stats1 = error_recovery_->GetRecoveryStats();
  auto stats2 = error_recovery_->GetRecoveryStats();

  EXPECT_EQ(stats1.total_fallback_attempts, stats2.total_fallback_attempts);
  EXPECT_EQ(stats1.successful_fallbacks, stats2.successful_fallbacks);
  EXPECT_EQ(stats1.failed_fallbacks, stats2.failed_fallbacks);

  // Reset and verify
  error_recovery_->ResetRecoveryStats();
  auto stats3 = error_recovery_->GetRecoveryStats();

  EXPECT_EQ(stats3.total_fallback_attempts, 0);
  EXPECT_EQ(stats3.successful_fallbacks, 0);
  EXPECT_EQ(stats3.failed_fallbacks, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
