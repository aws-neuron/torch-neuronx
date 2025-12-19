#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"

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
  OperationContext context{nullptr};

  EXPECT_EQ(context.kernel_execution, nullptr);
  EXPECT_TRUE(context.python_stack_trace.empty());
}

TEST_F(OperationContextTest, ConstructionWithKernelExecution) {
  std::string stack_trace = "test_stack_trace";
  auto kernel = std::make_unique<MockNeuronKernelExecution>();

  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext context(std::move(kernel), {}, {}, stack_trace);

  EXPECT_NE(context.kernel_execution, nullptr);
  EXPECT_EQ(context.python_stack_trace, stack_trace);
  EXPECT_TRUE(context.IsValid());
  EXPECT_TRUE(context.input_tensors_context.empty());
  EXPECT_TRUE(context.output_tensors_context.empty());
}

TEST_F(OperationContextTest, ConstructionWithNullKernel) {
  OperationContext context(nullptr, {}, {}, "test_trace");

  // Cannot call IsValid() on null kernel_execution - would crash
  EXPECT_EQ(context.kernel_execution, nullptr);
}

TEST_F(OperationContextTest, ConstructionWithTensorContext) {
  auto input_tensor = torch::ones({2, 3}, torch::dtype(torch::kFloat32));
  auto output_tensor = torch::zeros({4, 5}, torch::dtype(torch::kInt32));

  std::vector<TensorContext> input_contexts = {TensorContext::FromTensor(input_tensor)};
  std::vector<TensorContext> output_contexts = {TensorContext::FromTensor(output_tensor)};

  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext context(std::move(kernel), input_contexts, output_contexts, "test_trace");

  EXPECT_EQ(context.input_tensors_context.size(), 1);
  EXPECT_EQ(context.output_tensors_context.size(), 1);

  // Verify input context - all fields
  const auto& input_ctx = context.input_tensors_context[0];
  EXPECT_EQ(input_ctx.get_numel(), static_cast<size_t>(input_tensor.numel()));
  EXPECT_EQ(input_ctx.get_size_bytes(), static_cast<size_t>(input_tensor.nbytes()));
  EXPECT_EQ(input_ctx.get_element_size(), static_cast<size_t>(input_tensor.element_size()));
  EXPECT_EQ(input_ctx.get_shape(), input_tensor.sizes().vec());
  EXPECT_EQ(input_ctx.get_storage_offset(), static_cast<size_t>(input_tensor.storage_offset()));
  EXPECT_EQ(input_ctx.get_dtype(), input_tensor.scalar_type());
  EXPECT_EQ(input_ctx.get_options().dtype(), input_tensor.options().dtype());
  EXPECT_EQ(input_ctx.get_options().layout(), input_tensor.options().layout());

  // Verify output context - all fields
  const auto& output_ctx = context.output_tensors_context[0];
  EXPECT_EQ(output_ctx.get_numel(), static_cast<size_t>(output_tensor.numel()));
  EXPECT_EQ(output_ctx.get_size_bytes(), static_cast<size_t>(output_tensor.nbytes()));
  EXPECT_EQ(output_ctx.get_element_size(), static_cast<size_t>(output_tensor.element_size()));
  EXPECT_EQ(output_ctx.get_shape(), output_tensor.sizes().vec());
  EXPECT_EQ(output_ctx.get_storage_offset(), static_cast<size_t>(output_tensor.storage_offset()));
  EXPECT_EQ(output_ctx.get_dtype(), output_tensor.scalar_type());
  EXPECT_EQ(output_ctx.get_options().dtype(), output_tensor.options().dtype());
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
  auto kernel = std::make_unique<MockNeuronKernelExecution>("compilable_op");

  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(Return(true));
  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  std::vector<TensorContext> input_contexts, output_contexts;
  OperationContext context(std::move(kernel), input_contexts, output_contexts, "test_stack");

  EXPECT_TRUE(context.IsValid());
  EXPECT_EQ(context.GetOpName(), "compilable_op");
  EXPECT_TRUE(context.RequiresCompilation());
}

TEST_F(OperationContextTest, KernelTypeChecking) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();

  EXPECT_CALL(*kernel, RequiresCompilation()).WillRepeatedly(Return(false));

  std::vector<TensorContext> input_contexts, output_contexts;
  OperationContext context(std::move(kernel), input_contexts, output_contexts, "test_stack");

  EXPECT_FALSE(context.RequiresCompilation());

  // Should throw when trying to get compilable kernel
  EXPECT_THROW(context.GetCompilableKernel(), c10::Error);
}

TEST_F(OperationContextTest, CompilationRequirementChecking) {
  // Test with compilable kernel
  auto compilable = std::make_unique<MockNeuronKernelExecution>("compilable_op");

  EXPECT_CALL(*compilable, RequiresCompilation()).WillRepeatedly(Return(true));
  EXPECT_CALL(*compilable, ValidateImpl()).WillRepeatedly(Return(true));

  std::vector<TensorContext> input_contexts, output_contexts;
  OperationContext compilable_context(std::move(compilable), input_contexts, output_contexts,
                                      "test_stack");

  EXPECT_TRUE(compilable_context.RequiresCompilation());

  auto direct = std::make_unique<MockNeuronKernelExecution>("direct_op");

  EXPECT_CALL(*direct, RequiresCompilation()).WillRepeatedly(Return(false));

  OperationContext direct_context(std::move(direct), input_contexts, output_contexts, "test_stack");

  EXPECT_FALSE(direct_context.RequiresCompilation());
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

TEST_F(OperationContextTest, ExecutionReadinessInitiallyFalse) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  // Initially not ready
  EXPECT_FALSE(context.IsExecutionReady());
}

TEST_F(OperationContextTest, MarkExecutionReadySetsState) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  EXPECT_FALSE(context.IsExecutionReady());

  context.MarkExecutionReady();

  EXPECT_TRUE(context.IsExecutionReady());
}

TEST_F(OperationContextTest, ExecutionReadinessIsIdempotent) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  context.MarkExecutionReady();
  EXPECT_TRUE(context.IsExecutionReady());

  // Multiple calls should be safe
  context.MarkExecutionReady();
  context.MarkExecutionReady();
  EXPECT_TRUE(context.IsExecutionReady());
}

TEST_F(OperationContextTest, StreamPointerInitiallyNull) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  OperationContext context(std::move(kernel));

  EXPECT_EQ(context.stream, nullptr);
}

TEST_F(OperationContextTest, CPUFallbackContextAccessor) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  CPUFallbackContext fallback_ctx;
  OperationContext context(std::move(kernel), {}, {}, "", std::move(fallback_ctx));

  // Should be able to access the fallback context
  const auto& ctx = context.GetCPUFallbackContext();
  // Just verify it doesn't crash - the context is default constructed
  (void)ctx;
}

TEST_F(OperationContextTest, OperationContextResultStatusValues) {
  // Verify the status enum values
  EXPECT_EQ(static_cast<int>(OperationContextResult::kPending), 1);
  EXPECT_EQ(static_cast<int>(OperationContextResult::kCompleted), 0);
  EXPECT_EQ(static_cast<int>(OperationContextResult::kFailed), -1);
}

TEST_F(OperationContextTest, OperationContextResultEmptyError) {
  OperationContextResult result(OperationContextResult::kFailed, "");
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_TRUE(result.GetError().empty());
}

TEST_F(OperationContextTest, MultipleTensorContexts) {
  auto input1 = torch::ones({2, 3});
  auto input2 = torch::zeros({4, 5});
  auto output1 = torch::ones({6, 7});

  std::vector<TensorContext> input_contexts = {TensorContext::FromTensor(input1),
                                               TensorContext::FromTensor(input2)};
  std::vector<TensorContext> output_contexts = {TensorContext::FromTensor(output1)};

  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  EXPECT_CALL(*kernel, ValidateImpl()).WillRepeatedly(Return(true));

  OperationContext context(std::move(kernel), input_contexts, output_contexts, "test");

  EXPECT_EQ(context.input_tensors_context.size(), 2);
  EXPECT_EQ(context.output_tensors_context.size(), 1);
}

TEST_F(OperationContextTest, IsSchedulableForNonEventKernel) {
  auto kernel = std::make_unique<MockNeuronKernelExecution>();
  // Mock kernel returns kHLO type by default (not kEvent)
  OperationContext context(std::move(kernel));

  // Initially not schedulable (not execution ready)
  EXPECT_FALSE(context.IsSchedulable());

  // Mark execution ready
  context.MarkExecutionReady();

  // Now should be schedulable
  EXPECT_TRUE(context.IsSchedulable());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
