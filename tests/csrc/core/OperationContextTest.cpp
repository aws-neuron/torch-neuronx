#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuron/csrc/core/OperationContext.h"

using namespace at::neuron;
using namespace at::neuron::testing;
using namespace std::chrono_literals;
using ::testing::Return;

class OperationContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock kernel execution
    mock_kernel_ = std::make_unique<MockNeuronKernelExecution>();
    ON_CALL(*mock_kernel_, RequiresCompilation()).WillByDefault(Return(false));
    ON_CALL(*mock_kernel_, ValidateImpl()).WillByDefault(Return(true));
  }

  std::unique_ptr<MockNeuronKernelExecution> mock_kernel_;
};

// Test OperationContextResult
TEST_F(OperationContextTest, OperationContextResultDefaultConstruction) {
  OperationContextResult result;

  EXPECT_EQ(result.status, OperationContextResult::kPending);
  EXPECT_TRUE(result.error_message.empty());
  EXPECT_TRUE(result.IsPending());
  EXPECT_FALSE(result.IsSuccess());
}

TEST_F(OperationContextTest, OperationContextResultWithStatus) {
  OperationContextResult success_result(OperationContextResult::kCompleted);
  EXPECT_TRUE(success_result.IsSuccess());
  EXPECT_FALSE(success_result.IsPending());
  EXPECT_TRUE(success_result.GetError().empty());

  OperationContextResult error_result(OperationContextResult::kFailed, "Test error");
  EXPECT_FALSE(error_result.IsSuccess());
  EXPECT_FALSE(error_result.IsPending());
  EXPECT_EQ(error_result.GetError(), "Test error");
}

TEST_F(OperationContextTest, OperationContextResultStaticFactories) {
  auto success = OperationContextResult::CreateSuccess();
  EXPECT_TRUE(success.IsSuccess());
  EXPECT_EQ(success.status, OperationContextResult::kCompleted);

  auto error = OperationContextResult::CreateError("Custom error");
  EXPECT_FALSE(error.IsSuccess());
  EXPECT_EQ(error.status, OperationContextResult::kFailed);
  EXPECT_EQ(error.GetError(), "Custom error");
}

// Test OperationContext construction
TEST_F(OperationContextTest, DefaultConstruction) {
  OperationContext context;

  EXPECT_EQ(context.kernel_execution, nullptr);
  EXPECT_TRUE(context.python_stack_trace.empty());
  EXPECT_TRUE(context.events_to_complete.empty());
  EXPECT_TRUE(context.events_to_wait_for.empty());
}

TEST_F(OperationContextTest, ConstructionWithKernelExecution) {
  std::string stack_trace = "test_stack_trace";
  auto kernel = std::make_unique<MockNeuronKernelExecution>();

  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext context(std::move(kernel), stack_trace);

  EXPECT_NE(context.kernel_execution, nullptr);
  EXPECT_EQ(context.python_stack_trace, stack_trace);
  EXPECT_TRUE(context.IsValid());
}

TEST_F(OperationContextTest, ConstructionWithNullKernel) {
  OperationContext context(nullptr, "test_trace");

  // Cannot call IsValid() on null kernel_execution - would crash
  EXPECT_EQ(context.kernel_execution, nullptr);
}

TEST_F(OperationContextTest, TimingLogic) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  // Initially all times should be zero
  EXPECT_EQ(context.GetCompilationTime().count(), 0);
  EXPECT_EQ(context.GetExecutionTime().count(), 0);
  EXPECT_EQ(context.GetQueueTime().count(), 0);
  EXPECT_EQ(context.GetTotalTime().count(), 0);
  EXPECT_EQ(context.GetPipelineTime().count(), 0);

  // Mark submitted
  context.MarkSubmitted();
  std::this_thread::sleep_for(1ms);

  // Start compilation
  context.StartCompilation();
  std::this_thread::sleep_for(2ms);
  context.CompleteCompilation();

  // Start execution
  context.StartExecution();
  std::this_thread::sleep_for(3ms);
  context.execute_end = std::chrono::steady_clock::now();

  // Verify timing calculations
  EXPECT_GT(context.GetCompilationTime().count(), 0);
  EXPECT_GT(context.GetExecutionTime().count(), 0);
  EXPECT_GT(context.GetQueueTime().count(), 0);
  EXPECT_GT(context.GetTotalTime().count(), 0);
  EXPECT_GT(context.GetPipelineTime().count(), 0);
}

TEST_F(OperationContextTest, TimingWithoutSubmission) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  // Start execution without marking submitted
  context.StartExecution();

  // Queue time should be zero since submit_time wasn't set
  EXPECT_EQ(context.GetQueueTime().count(), 0);
  EXPECT_EQ(context.GetPipelineTime().count(), 0);
}

TEST_F(OperationContextTest, PartialTimingData) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  // Only set compilation start, not end
  context.StartCompilation();

  // Compilation time should be zero since end time not set
  EXPECT_EQ(context.GetCompilationTime().count(), 0);

  // Only set execution start, not end
  context.StartExecution();

  // Execution time should be zero since end time not set
  EXPECT_EQ(context.GetExecutionTime().count(), 0);
}

// Test kernel execution integration
TEST_F(OperationContextTest, KernelExecutionIntegration) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>("custom_op");

  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));
  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(Return(false));

  OperationContext context(std::move(kernel));

  EXPECT_TRUE(context.IsValid());
  EXPECT_EQ(context.GetOpName(), "custom_op");
  EXPECT_FALSE(context.RequiresCompilation());
}

TEST_F(OperationContextTest, CompilableKernelIntegration) {
  auto compilable = std::make_unique<MockCompilableKernelExecution>(
      "compilable_op", std::vector<torch::Tensor>(), std::vector<torch::Tensor>(), "test_key",
      false);

  EXPECT_CALL(*compilable, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext context(std::move(compilable));

  EXPECT_TRUE(context.IsValid());
  EXPECT_EQ(context.GetOpName(), "compilable_op");
  EXPECT_TRUE(context.RequiresCompilation());

  // Test getting compilable kernel
  auto& compilable_kernel = context.GetCompilableKernel();
  EXPECT_EQ(compilable_kernel.GetCacheKey(), "test_key");
  EXPECT_FALSE(compilable_kernel.HasCachedNeff());  // No cached NEFF initially
}

TEST_F(OperationContextTest, KernelTypeChecking) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();

  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(Return(false));

  OperationContext context(std::move(kernel));

  EXPECT_FALSE(context.RequiresCompilation());

  // Should be able to get direct kernel
  EXPECT_NO_THROW(context.GetDirectKernel());

  // Should throw when trying to get compilable kernel
  EXPECT_THROW(context.GetCompilableKernel(), std::logic_error);
}

TEST_F(OperationContextTest, CompilationRequirementChecking) {
  // Test with compilable kernel
  auto compilable = std::make_unique<MockCompilableKernelExecution>(
      "compilable_op", std::vector<torch::Tensor>(), std::vector<torch::Tensor>(), "test_key",
      false);

  EXPECT_CALL(*compilable, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext compilable_context(std::move(compilable));

  EXPECT_TRUE(compilable_context.RequiresCompilation());
  EXPECT_NO_THROW(compilable_context.GetCompilableKernel());
  EXPECT_THROW(compilable_context.GetDirectKernel(), std::logic_error);

  auto direct = std::make_unique<MockNeuronKernelExecution>("direct_op");

  EXPECT_CALL(*direct, RequiresCompilation()).WillRepeatedly(Return(false));

  OperationContext direct_context(std::move(direct));

  EXPECT_FALSE(direct_context.RequiresCompilation());
  EXPECT_NO_THROW(direct_context.GetDirectKernel());
  EXPECT_THROW(direct_context.GetCompilableKernel(), std::logic_error);
}

TEST_F(OperationContextTest, EventsToComplete) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  EXPECT_TRUE(context.events_to_complete.empty());
  EXPECT_TRUE(context.events_to_wait_for.empty());

  // TODO(rpsilva): Extend with more integration with events once ready.
  EXPECT_EQ(context.events_to_complete.size(), 0);
  EXPECT_EQ(context.events_to_wait_for.size(), 0);
}

TEST_F(OperationContextTest, BasicPromiseFunctionality) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  // Get future from promise
  auto future = context.promise.get_future();

  // Promise should be valid
  EXPECT_TRUE(future.valid());

  // Set result
  context.promise.set_value(OperationContextResult::CreateSuccess());

  // Future should now be ready
  EXPECT_EQ(future.wait_for(std::chrono::seconds(0)), std::future_status::ready);

  auto result = future.get();
  EXPECT_TRUE(result.IsSuccess());
}

TEST_F(OperationContextTest, PromiseWithError) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  auto future = context.promise.get_future();

  // Set error result
  context.promise.set_value(OperationContextResult::CreateError("Test error"));

  auto result = future.get();
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_EQ(result.GetError(), "Test error");
}

TEST_F(OperationContextTest, BasicMoveSemantics) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>("move_test");
  OperationContext context1(std::move(kernel), "trace1");

  // Move construct
  OperationContext context2 = std::move(context1);

  EXPECT_NE(context2.kernel_execution, nullptr);
  EXPECT_EQ(context2.GetOpName(), "move_test");
  EXPECT_EQ(context2.python_stack_trace, "trace1");

  // Original should be in moved-from state
  EXPECT_EQ(context1.kernel_execution, nullptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
