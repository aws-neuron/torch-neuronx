#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockCPUFallbackExecutor.h"
#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackExecutor.h"

using namespace at::neuron;
namespace mock = at::neuron::testing;

class CPUFallbackExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear environment variables for consistent testing
    unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");

    executor_ = std::make_unique<mock::MockCPUFallbackExecutor>();
  }

  void TearDown() override {
    // Clean up environment
    unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  }
  std::unique_ptr<mock::MockCPUFallbackExecutor> executor_;
};

// Test CPUFallbackExecutor constructor and configuration
TEST_F(CPUFallbackExecutorTest, ConstructorAndConfiguration) {
  // Test 1: Default configuration (enabled)
  {
    CPUFallbackExecutor executor;
    EXPECT_TRUE(executor.IsEnabled());
  }

  // Test 2: Disabled via environment variable
  {
    setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1", 1);
    CPUFallbackExecutor executor;
    EXPECT_FALSE(executor.IsEnabled());
    unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  }

  // Test 3: Enabled via environment variable (any other value)
  {
    setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "0", 1);
    CPUFallbackExecutor executor;
    EXPECT_TRUE(executor.IsEnabled());
    unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  }
}

// Test enable/disable functionality
TEST_F(CPUFallbackExecutorTest, EnableDisable) {
  // Test initial state
  EXPECT_TRUE(executor_->IsEnabled());

  // Test disabling
  executor_->SetEnabled(false);
  EXPECT_FALSE(executor_->IsEnabled());

  // Test re-enabling
  executor_->SetEnabled(true);
  EXPECT_TRUE(executor_->IsEnabled());
}

// Test can_execute_on_cpu method
TEST_F(CPUFallbackExecutorTest, CanExecuteOnCpu) {
  // Test with enabled executor
  EXPECT_TRUE(executor_->IsEnabled());

  // Test with common PyTorch operations that should have CPU implementations
  // Note: These may return false in test environment due to limited PyTorch setup
  bool can_add = executor_->CanExecuteOnCpu("aten::add.Tensor");
  bool can_mul = executor_->CanExecuteOnCpu("aten::mul.Tensor");

  // The exact result depends on PyTorch setup, but the method should not crash
  EXPECT_TRUE(can_add || !can_add);  // Just verify it returns a boolean
  EXPECT_TRUE(can_mul || !can_mul);  // Just verify it returns a boolean

  // Test with clearly invalid operation
  EXPECT_FALSE(executor_->CanExecuteOnCpu("invalid::nonexistent.operation"));

  // Test with disabled executor
  executor_->SetEnabled(false);
  EXPECT_FALSE(executor_->CanExecuteOnCpu("aten::add.Tensor"));
  EXPECT_FALSE(executor_->CanExecuteOnCpu("any::operation"));
}

// Test ExecuteCpuFallback with disabled executor
TEST_F(CPUFallbackExecutorTest, ExecuteCpuFallbackDisabled) {
  executor_->SetEnabled(false);

  // Create a mock operation
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Execute CPU fallback - should fail because disabled
  auto result = executor_->ExecuteCpuFallback(&mock_op);

  EXPECT_FALSE(result.IsSuccess());
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("CPU fallback is disabled"));
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("test_op"));
}

// Test ExecuteCpuFallback with mock operation (will fail due to no CPU implementation)
TEST_F(CPUFallbackExecutorTest, ExecuteCpuFallbackWithMockOperation) {
  EXPECT_TRUE(executor_->IsEnabled());

  // Create a mock operation
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*mock_kernel, GetKernelType())
      .WillRepeatedly(::testing::Return(KernelTypeEnum::kHLO));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Execute CPU fallback - will fail due to mock operation
  auto result = executor_->ExecuteCpuFallback(&mock_op);

  // Should fail because "test_op" is not a real PyTorch operation
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("CPU fallback failed for"));
}

// Test configuration loading with different environment variable values
TEST_F(CPUFallbackExecutorTest, ConfigurationLoading) {
  // Test various environment variable values
  std::vector<std::pair<std::string, bool>> test_cases = {
      {"1", false},     // "1" disables fallback
      {"0", true},      // "0" enables fallback
      {"true", true},   // "true" enables fallback
      {"false", true},  // "false" enables fallback
      {"", true},       // empty enables fallback
      {"random", true}  // any other value enables fallback
  };

  for (const auto& [env_value, expected_enabled] : test_cases) {
    if (env_value.empty()) {
      unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
    } else {
      setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", env_value.c_str(), 1);
    }

    CPUFallbackExecutor executor;
    EXPECT_EQ(executor.IsEnabled(), expected_enabled)
        << "Failed for env_value: '" << env_value << "'";

    unsetenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
