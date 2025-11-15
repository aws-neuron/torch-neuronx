#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "torch_neuronx/csrc/core/utils/ErrorHandler.h"

using namespace at::neuron;
namespace mock = at::neuron::testing;

class ErrorHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear environment variables for consistent testing
    unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
    unsetenv("NEURON_DUMP_EXECUTION_ON_ERROR");
    unsetenv("NEURON_ERROR_DUMP_DIR");

    error_handler_ = std::make_unique<ErrorHandler>();
  }

  void TearDown() override {
    // Clean up environment
    unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
    unsetenv("NEURON_DUMP_EXECUTION_ON_ERROR");
    unsetenv("NEURON_ERROR_DUMP_DIR");
  }

  // Helper to create standard mock operation
  OperationContext create_mock_operation(
      const std::string& op_name = "test_op",
      const std::vector<torch::Tensor>& inputs = {torch::ones({2, 3})},
      const std::vector<torch::Tensor>& outputs = {torch::ones({2, 3})}) {
    auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>(op_name, inputs, outputs);

    EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

    OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
    mock_op.submit_time = std::chrono::steady_clock::now();
    return mock_op;
  }

  std::unique_ptr<ErrorHandler> error_handler_;
};

// Test ErrorHandler constructor and environment variable handling
TEST_F(ErrorHandlerTest, ConstructorDefaultConfiguration) {
  ErrorHandler handler;

  EXPECT_FALSE(handler.IsEnhancedErrorHandlingEnabled());
  EXPECT_FALSE(handler.IsExecutionDumpingEnabled());
}

TEST_F(ErrorHandlerTest, ConstructorWithEnhancedErrorHandling) {
  setenv("NEURON_ENHANCED_ERROR_HANDLING", "1", 1);
  ErrorHandler handler;

  EXPECT_TRUE(handler.IsEnhancedErrorHandlingEnabled());

  unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
}

TEST_F(ErrorHandlerTest, ConstructorWithExecutionDumping) {
  setenv("NEURON_DUMP_EXECUTION_ON_ERROR", "1", 1);
  ErrorHandler handler;

  EXPECT_TRUE(handler.IsExecutionDumpingEnabled());

  unsetenv("NEURON_DUMP_EXECUTION_ON_ERROR");
}

// Test static utility methods
TEST_F(ErrorHandlerTest, ExceptionToString) {
  std::runtime_error runtime_err("Runtime error message");
  std::string result = ErrorHandler::ExceptionToString(runtime_err);

  // The actual implementation returns "Exception: St13runtime_error - Runtime error message"
  EXPECT_THAT(result, ::testing::HasSubstr("Exception:"));
  EXPECT_THAT(result, ::testing::HasSubstr("St13runtime_error"));
  EXPECT_THAT(result, ::testing::HasSubstr("Runtime error message"));

  std::logic_error logic_err("Logic error message");
  result = ErrorHandler::ExceptionToString(logic_err);

  EXPECT_THAT(result, ::testing::HasSubstr("Exception:"));
  EXPECT_THAT(result, ::testing::HasSubstr("St11logic_error"));
  EXPECT_THAT(result, ::testing::HasSubstr("Logic error message"));
}

TEST_F(ErrorHandlerTest, CleanErrorMessage) {
  // Test removing stack traces - the actual implementation extracts the last meaningful line
  std::string messy_error =
      "Error occurred\nframe #0: some_function\nframe #1: another_function\nActual error message";
  std::string cleaned = ErrorHandler::CleanErrorMessage(messy_error);

  EXPECT_THAT(cleaned, ::testing::Not(::testing::HasSubstr("frame #")));
  EXPECT_THAT(cleaned, ::testing::HasSubstr("Actual error message"));

  // Test removing excessive whitespace - the implementation extracts the last line
  std::string whitespace_error = "Error   with    lots   of    spaces\n\n\nand newlines";
  cleaned = ErrorHandler::CleanErrorMessage(whitespace_error);

  EXPECT_THAT(cleaned, ::testing::HasSubstr("and newlines"));

  // Test with clean message (should remain unchanged)
  std::string clean_error = "Simple error message";
  cleaned = ErrorHandler::CleanErrorMessage(clean_error);

  EXPECT_EQ(cleaned, clean_error);
}

// Test extract_nrt_status method
TEST_F(ErrorHandlerTest, ExtractNrtStatusValidStatus) {
  EXPECT_EQ(error_handler_->ExtractNrtStatus("Error occurred with status: 1"), 1);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status= 2 - some error"), 2);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status= 1005 - some error"), 1005);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("Multiple errors status: -999"), -999);
}

TEST_F(ErrorHandlerTest, ExtractNrtStatusNoStatus) {
  EXPECT_EQ(error_handler_->ExtractNrtStatus("Error without status code"), -999);
  EXPECT_EQ(error_handler_->ExtractNrtStatus(""), -999);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status: invalid"), -999);
}

TEST_F(ErrorHandlerTest, ExtractNrtStatusEdgeCases) {
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status: 0"), 0);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status= -1"), -1);
  EXPECT_EQ(error_handler_->ExtractNrtStatus("status: 999999"), 999999);
}

// Test classify_nrt_error method with various status codes
TEST_F(ErrorHandlerTest, ClassifyNrtErrorStatuses) {
  ErrorContext context;

  // Test key status codes from different categories
  struct StatusTest {
    int status;
    std::string expected_substr;
  };

  std::vector<StatusTest> tests = {{1, "Non-specific failure"},
                                   {2, "Invalid NEFF, instruction, or input"},
                                   {5, "Operation timed out"},
                                   {6, "Hardware failure"},
                                   {1002, "Invalid input to exec()"},
                                   {1005, "Neuron core busy/locked"},
                                   {1201, "HBM encountered an unrepairable uncorrectable error"},
                                   {1203, "DMA engine encountered an unrecoverable error"}};

  for (const auto& test : tests) {
    auto [ctx, msg] =
        error_handler_->ClassifyNrtError("status: " + std::to_string(test.status), context);
    EXPECT_THAT(msg, ::testing::HasSubstr(test.expected_substr))
        << "Failed for status: " << test.status;
  }
}

TEST_F(ErrorHandlerTest, ClassifyNrtErrorUnknownStatus) {
  ErrorContext context;

  auto [ctx, msg] = error_handler_->ClassifyNrtError("status: 9999", context);
  // The actual message format is "status: 9999, " so let's check for that
  EXPECT_THAT(msg, ::testing::HasSubstr("status: 9999"));
}

TEST_F(ErrorHandlerTest, ClassifyNrtErrorNoStatus) {
  ErrorContext context;

  auto [ctx, msg] = error_handler_->ClassifyNrtError("No status in message", context);
  EXPECT_THAT(msg, ::testing::HasSubstr("Unknown status"));
  EXPECT_THAT(msg, ::testing::HasSubstr("-999"));
}

// Test create_error_context method
TEST_F(ErrorHandlerTest, CreateErrorContextWithNullOperation) {
  ErrorContext context =
      error_handler_->CreateErrorContext(nullptr, ErrorContext::Stage::COMPILATION);

  EXPECT_TRUE(context.operation_name.empty());
  EXPECT_EQ(context.current_stage, ErrorContext::Stage::COMPILATION);
  EXPECT_TRUE(context.input_shapes.empty());
  EXPECT_TRUE(context.output_shapes.empty());
}

TEST_F(ErrorHandlerTest, CreateErrorContextWithMockOperation) {
  // Create tensors for testing
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3}), torch::ones({4, 5})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 5})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  // Set up mock expectations
  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext op(std::move(mock_kernel), "test_stack_trace");
  op.submit_time = std::chrono::steady_clock::now();

  ErrorContext context = error_handler_->CreateErrorContext(&op, ErrorContext::Stage::COMPILATION);

  EXPECT_EQ(context.operation_name, "test_op");
  EXPECT_EQ(context.current_stage, ErrorContext::Stage::COMPILATION);
  EXPECT_EQ(context.input_shapes.size(), 2);
  EXPECT_EQ(context.output_shapes.size(), 1);
}

// Test handle_compilation_error method
TEST_F(ErrorHandlerTest, HandleCompilationError) {
  auto mock_op = create_mock_operation();

  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 123;

  std::string result =
      error_handler_->HandleCompilationError(&mock_op, "Compilation failed", context);

  EXPECT_THAT(result, ::testing::HasSubstr("Compilation error occurred on Neuron"));
  EXPECT_THAT(result, ::testing::HasSubstr("operation=test_op"));
  EXPECT_THAT(result, ::testing::HasSubstr("COMPILATION FAILED"));
  EXPECT_THAT(result, ::testing::HasSubstr("test_stack_trace"));
}

// Test handle_execution_error method with full flow (extract -> classify -> error_message)
TEST_F(ErrorHandlerTest, HandleExecutionErrorWithStatusCode) {
  // Create a mock operation for testing
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Test with a specific NRT status code to verify the full flow
  std::string error_with_status = "NRT execution failed with status: 1005 - core busy";

  // Step 1: Test ExtractNrtStatus
  int extracted_status = error_handler_->ExtractNrtStatus(error_with_status);
  EXPECT_EQ(extracted_status, 1005);

  // Step 2: Test ClassifyNrtError
  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 456;

  auto [classified_context, classified_message] =
      error_handler_->ClassifyNrtError(error_with_status, context);
  EXPECT_THAT(classified_message, ::testing::HasSubstr("Neuron core busy/locked"));
  EXPECT_THAT(classified_message, ::testing::HasSubstr("status: 1005"));

  // Step 3: Test HandleExecutionError with the classified message
  std::string result =
      error_handler_->HandleExecutionError(&mock_op, classified_message, classified_context);

  EXPECT_THAT(result, ::testing::HasSubstr("NRT Execution error occurred on Neuron"));
  EXPECT_THAT(result, ::testing::HasSubstr("operation=test_op"));
  EXPECT_THAT(result, ::testing::HasSubstr("NRT EXECUTION FAILED"));
  EXPECT_THAT(result, ::testing::HasSubstr("Neuron core busy/locked"));
  EXPECT_THAT(result, ::testing::HasSubstr("test_stack_trace"));
}

// Test handle_execution_error method with different status codes
TEST_F(ErrorHandlerTest, HandleExecutionErrorWithDifferentStatusCodes) {
  // Create a mock operation for testing
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Test with hardware error status
  {
    std::string error_with_status = "Hardware error occurred with status: 1201";
    auto [context, classified_message] =
        error_handler_->ClassifyNrtError(error_with_status, ErrorContext{});

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result,
                ::testing::HasSubstr("HBM encountered an unrepairable uncorrectable error"));
    EXPECT_THAT(result, ::testing::HasSubstr("NRT EXECUTION FAILED"));
  }

  // Test with timeout error status
  {
    std::string error_with_status = "Operation timed out with status: 5";
    auto [context, classified_message] =
        error_handler_->ClassifyNrtError(error_with_status, ErrorContext{});

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result, ::testing::HasSubstr("Operation timed out"));
    EXPECT_THAT(result, ::testing::HasSubstr("NRT EXECUTION FAILED"));
  }

  // Test with unknown status code
  {
    std::string error_with_status = "Unknown error with status: 9999";
    auto [context, classified_message] =
        error_handler_->ClassifyNrtError(error_with_status, ErrorContext{});

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result, ::testing::HasSubstr("status: 9999"));
    EXPECT_THAT(result, ::testing::HasSubstr("NRT EXECUTION FAILED"));
  }
}

// Test generate_error_message method
TEST_F(ErrorHandlerTest, GenerateErrorMessageBasic) {
  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 789;

  std::string result = error_handler_->GenerateErrorMessage("Base error", context);
  EXPECT_EQ(result, "Base error");
}

TEST_F(ErrorHandlerTest, GenerateErrorMessageEnhanced) {
  setenv("NEURON_ENHANCED_ERROR_HANDLING", "1", 1);
  ErrorHandler enhanced_handler;

  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 789;
  context.current_stage = ErrorContext::Stage::COMPILATION;
  context.input_shapes = {"input0: [2, 3]", "input1: [4, 5]"};
  context.output_shapes = {"output0: [2, 5]"};

  std::string result = enhanced_handler.GenerateErrorMessage("Base error", context);

  EXPECT_THAT(result, ::testing::HasSubstr("Base error"));
  EXPECT_THAT(result, ::testing::HasSubstr("--- Error Context ---"));
  EXPECT_THAT(result, ::testing::HasSubstr("Operation: test_op"));
  EXPECT_THAT(result, ::testing::HasSubstr("Stream ID: 789"));
  EXPECT_THAT(result, ::testing::HasSubstr("Stage: COMPILATION"));
  EXPECT_THAT(result, ::testing::HasSubstr("Input Shapes:"));
  EXPECT_THAT(result, ::testing::HasSubstr("input0: [2, 3]"));
  EXPECT_THAT(result, ::testing::HasSubstr("Output Shapes:"));
  EXPECT_THAT(result, ::testing::HasSubstr("output0: [2, 5]"));

  unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
}

// Test error statistics functionality
TEST_F(ErrorHandlerTest, ErrorStatistics) {
  ErrorStatistics stats = error_handler_->GetErrorStatistics();

  // Initially all counters should be zero
  EXPECT_EQ(stats.compilation_errors.load(), 0);
  EXPECT_EQ(stats.execution_errors.load(), 0);
  EXPECT_EQ(stats.timeout_errors.load(), 0);
  EXPECT_EQ(stats.get_total_errors(), 0);

  // Test utility methods
  EXPECT_EQ(stats.get_avg_error_handling_time_us(), 0);  // No calls yet
  EXPECT_EQ(stats.get_error_rate(0), 0.0);               // No operations

  // Test with some values
  stats.total_error_handling_time_us = 1000;
  stats.error_handling_calls = 10;
  stats.compilation_errors = 10;
  EXPECT_EQ(stats.get_avg_error_handling_time_us(), 100);
  EXPECT_EQ(stats.get_error_rate(100), 0.1);  // 10 errors out of 100 operations

  // Test reset functionality
  error_handler_->ResetErrorStatistics();
  stats = error_handler_->GetErrorStatistics();
  EXPECT_EQ(stats.get_total_errors(), 0);
  EXPECT_EQ(stats.retry_attempts.load(), 0);
}

// Test get_recent_errors method
TEST_F(ErrorHandlerTest, GetRecentErrors) {
  // Initially should have no recent errors
  auto recent_errors = error_handler_->GetRecentErrors();
  EXPECT_TRUE(recent_errors.empty());

  // Test with max_errors parameter
  auto limited_errors = error_handler_->GetRecentErrors(10);
  EXPECT_TRUE(limited_errors.empty());
}

// Test handle_operation_error method (main entry point) - comprehensive test
TEST_F(ErrorHandlerTest, HandleOperationError) {
  // Create a mock operation for testing
  std::vector<torch::Tensor> inputs = {torch::ones({2, 3})};
  std::vector<torch::Tensor> outputs = {torch::ones({2, 3})};

  auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>("test_op", inputs, outputs);

  EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
  EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

  OperationContext mock_op(std::move(mock_kernel), "test_stack_trace");
  mock_op.submit_time = std::chrono::steady_clock::now();

  // Test 1: Compilation error - should trigger CPU fallback attempt which will fail
  // The warning "CPU fallback failed for 'test_op' failed: No CPU implementation found" is expected
  auto compilation_result = error_handler_->HandleOperationError(
      &mock_op, "Compilation failed", ErrorContext::Stage::COMPILATION, 123);

  EXPECT_FALSE(compilation_result.IsSuccess());
  EXPECT_THAT(compilation_result.GetError(), ::testing::HasSubstr("COMPILATION FAILED"));

  // Test 2: Execution error with status code - should trigger CPU fallback attempt which will fail
  auto execution_result = error_handler_->HandleOperationError(
      &mock_op, "Execution failed with status: 1005", ErrorContext::Stage::EXECUTION, 456, 1005);

  EXPECT_FALSE(execution_result.IsSuccess());
  EXPECT_THAT(execution_result.GetError(), ::testing::HasSubstr("NRT EXECUTION FAILED"));
  EXPECT_THAT(execution_result.GetError(), ::testing::HasSubstr("Neuron core busy/locked"));

  // The CPU fallback failures are expected behavior for this mock operation
  // This test verifies that error handling works correctly when both Neuron execution
  // and CPU fallback fail (which is the expected scenario for Neuron-only operations)
}

// Test exception tracking methods
TEST_F(ErrorHandlerTest, ExceptionTracking) {
  uint32_t stream_id = 789;

  // Initially should have no pending exception
  EXPECT_FALSE(error_handler_->has_pending_exception_);

  // Record an exception
  auto exception = std::make_exception_ptr(std::runtime_error("Test exception"));
  error_handler_->RecordFirstException(stream_id, exception);

  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Recording another exception should not overwrite the first
  auto second_exception = std::make_exception_ptr(std::runtime_error("Second exception"));
  error_handler_->RecordFirstException(stream_id + 1, second_exception);

  EXPECT_TRUE(error_handler_->has_pending_exception_);
}

TEST_F(ErrorHandlerTest, CheckAndThrowPendingException) {
  uint32_t stream_id = 999;

  // Should not throw when no pending exception
  EXPECT_NO_THROW(error_handler_->CheckAndThrowPendingException(stream_id, false));

  // Record an exception and verify it gets thrown
  auto exception = std::make_exception_ptr(std::runtime_error("Pending exception"));
  error_handler_->RecordFirstException(stream_id, exception);

  EXPECT_THROW(error_handler_->CheckAndThrowPendingException(stream_id, true), std::runtime_error);

  // After throwing, should no longer have pending exception
  EXPECT_FALSE(error_handler_->has_pending_exception_);
}

// Test ErrorContext utility methods
TEST_F(ErrorHandlerTest, ErrorContextUtilities) {
  ErrorContext context;

  // Test get_stage_name
  context.current_stage = ErrorContext::Stage::SUBMISSION;
  EXPECT_EQ(context.get_stage_name(), "SUBMISSION");

  context.current_stage = ErrorContext::Stage::COMPILATION;
  EXPECT_EQ(context.get_stage_name(), "COMPILATION");

  context.current_stage = ErrorContext::Stage::EXECUTION;
  EXPECT_EQ(context.get_stage_name(), "EXECUTION");

  // Test get_elapsed_time with no submit_time
  auto elapsed = context.get_elapsed_time();
  EXPECT_EQ(elapsed.count(), 0);

  // Test get_elapsed_time with submit_time
  context.submit_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);
  context.error_time = std::chrono::steady_clock::now();
  elapsed = context.get_elapsed_time();
  EXPECT_GT(elapsed.count(), 50);  // Should be around 100ms, but allow some variance
}

// Test Environment variable combinations
TEST_F(ErrorHandlerTest, ConstructorWithMultipleEnvVars) {
  setenv("NEURON_ENHANCED_ERROR_HANDLING", "1", 1);
  setenv("NEURON_DUMP_EXECUTION_ON_ERROR", "1", 1);
  ErrorHandler handler;

  EXPECT_TRUE(handler.IsEnhancedErrorHandlingEnabled());
  EXPECT_TRUE(handler.IsExecutionDumpingEnabled());

  unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
  unsetenv("NEURON_DUMP_EXECUTION_ON_ERROR");
}

TEST_F(ErrorHandlerTest, ConstructorWithAllEnvVars) {
  setenv("NEURON_ENHANCED_ERROR_HANDLING", "1", 1);
  setenv("NEURON_DUMP_EXECUTION_ON_ERROR", "1", 1);
  setenv("NEURON_ERROR_DUMP_DIR", "/custom/dump/path", 1);
  ErrorHandler handler;

  EXPECT_TRUE(handler.IsEnhancedErrorHandlingEnabled());
  EXPECT_TRUE(handler.IsExecutionDumpingEnabled());

  unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
  unsetenv("NEURON_DUMP_EXECUTION_ON_ERROR");
  unsetenv("NEURON_ERROR_DUMP_DIR");
}

// Test Custom dump directory
TEST_F(ErrorHandlerTest, CustomDumpDirectory) {
  setenv("NEURON_ERROR_DUMP_DIR", "/custom/dump/path", 1);
  ErrorHandler handler;

  // The GetDebugDumpDirectory is private, but we can verify it was set
  // by checking that the handler was constructed successfully
  EXPECT_FALSE(handler.IsEnhancedErrorHandlingEnabled());

  unsetenv("NEURON_ERROR_DUMP_DIR");
}

TEST_F(ErrorHandlerTest, DefaultDumpDirectory) {
  // Ensure env var is not set
  unsetenv("NEURON_ERROR_DUMP_DIR");
  ErrorHandler handler;

  // Handler should use default /tmp/neuron_error_dumps
  // We can't directly test the private member, but we verify construction succeeds
  EXPECT_FALSE(handler.IsEnhancedErrorHandlingEnabled());
}

// Test Multiple error recording
TEST_F(ErrorHandlerTest, MultipleErrorRecording) {
  auto mock_op1 = create_mock_operation("op1");
  auto mock_op2 = create_mock_operation("op2");
  auto mock_op3 = create_mock_operation("op3");

  ErrorContext context1;
  context1.operation_name = "op1";
  context1.stream_id = 100;
  context1.current_stage = ErrorContext::Stage::COMPILATION;

  ErrorContext context2;
  context2.operation_name = "op2";
  context2.stream_id = 200;
  context2.current_stage = ErrorContext::Stage::EXECUTION;

  ErrorContext context3;
  context3.operation_name = "op3";
  context3.stream_id = 300;
  context3.current_stage = ErrorContext::Stage::COMPILATION;

  // Record multiple errors
  error_handler_->HandleCompilationError(&mock_op1, "Error 1", context1);
  error_handler_->HandleExecutionError(&mock_op2, "Error 2", context2);
  error_handler_->HandleCompilationError(&mock_op3, "Error 3", context3);

  // Check statistics
  ErrorStatistics stats = error_handler_->GetErrorStatistics();
  EXPECT_EQ(stats.compilation_errors.load(), 2);
  EXPECT_EQ(stats.execution_errors.load(), 1);
  EXPECT_EQ(stats.get_total_errors(), 3);

  // Check recent errors
  auto recent_errors = error_handler_->GetRecentErrors();
  EXPECT_EQ(recent_errors.size(), 3);
  EXPECT_EQ(recent_errors[0].operation_name, "op1");
  EXPECT_EQ(recent_errors[1].operation_name, "op2");
  EXPECT_EQ(recent_errors[2].operation_name, "op3");
}

TEST_F(ErrorHandlerTest, ErrorRecordingUpdatesStatistics) {
  auto mock_op = create_mock_operation();

  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 123;

  // Initial statistics
  ErrorStatistics initial_stats = error_handler_->GetErrorStatistics();
  EXPECT_EQ(initial_stats.compilation_errors.load(), 0);
  EXPECT_EQ(initial_stats.error_handling_calls.load(), 0);

  // Record a compilation error
  error_handler_->HandleCompilationError(&mock_op, "Compilation failed", context);

  // Check updated statistics
  ErrorStatistics updated_stats = error_handler_->GetErrorStatistics();
  EXPECT_EQ(updated_stats.compilation_errors.load(), 1);
  EXPECT_EQ(updated_stats.compilation_stage_errors.load(), 1);
  EXPECT_EQ(updated_stats.error_handling_calls.load(), 1);
  // Note: total_error_handling_time_us may be 0 if execution is very fast
  EXPECT_GE(updated_stats.total_error_handling_time_us.load(), 0);
}

// Test concurrent access (thread-safety)
TEST_F(ErrorHandlerTest, ConcurrentErrorRecording) {
  const int num_threads = 10;
  const int errors_per_thread = 5;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, t, errors_per_thread]() {
      for (int i = 0; i < errors_per_thread; ++i) {
        auto mock_op = create_mock_operation("op_" + std::to_string(t) + "_" + std::to_string(i));
        ErrorContext context;
        context.operation_name = "op_" + std::to_string(t) + "_" + std::to_string(i);
        context.stream_id = t * 100 + i;
        context.current_stage =
            (i % 2 == 0) ? ErrorContext::Stage::COMPILATION : ErrorContext::Stage::EXECUTION;

        if (i % 2 == 0) {
          error_handler_->HandleCompilationError(&mock_op, "Error", context);
        } else {
          error_handler_->HandleExecutionError(&mock_op, "Error", context);
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify statistics
  // With i % 2 == 0 for compilation: i=0,2,4 (3 per thread)
  // With i % 2 != 0 for execution: i=1,3 (2 per thread)
  ErrorStatistics stats = error_handler_->GetErrorStatistics();
  int expected_compilation = num_threads * 3;  // i=0,2,4 per thread
  int expected_execution = num_threads * 2;    // i=1,3 per thread
  EXPECT_EQ(stats.compilation_errors.load(), expected_compilation);
  EXPECT_EQ(stats.execution_errors.load(), expected_execution);
  EXPECT_EQ(stats.get_total_errors(), num_threads * errors_per_thread);
}

TEST_F(ErrorHandlerTest, ConcurrentStatisticsAccess) {
  const int num_threads = 20;
  std::vector<std::thread> threads;
  std::atomic<int> successful_reads{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, &successful_reads]() {
      // Half the threads read statistics
      for (int i = 0; i < 100; ++i) {
        auto stats = error_handler_->GetErrorStatistics();
        (void)stats;  // Intentionally unused - just testing thread safety
        successful_reads++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(successful_reads.load(), num_threads * 100);
}

// Test Edge cases for CleanErrorMessage
TEST_F(ErrorHandlerTest, CleanErrorMessageEmptyString) {
  std::string empty = "";
  std::string cleaned = ErrorHandler::CleanErrorMessage(empty);
  EXPECT_EQ(cleaned, "");
}

TEST_F(ErrorHandlerTest, CleanErrorMessageOnlyWhitespace) {
  std::string whitespace = "   \n\n\t\t  \n";
  std::string cleaned = ErrorHandler::CleanErrorMessage(whitespace);
  EXPECT_EQ(cleaned, whitespace);  // No meaningful line found, returns original
}

TEST_F(ErrorHandlerTest, CleanErrorMessagePythonTraceback) {
  std::string python_traceback = R"(Traceback (most recent call last):
  File "test.py", line 10, in <module>
    result = model(input)
  File "model.py", line 50, in forward
    return self.layer(x)
RuntimeError: Out of memory)";

  std::string cleaned = ErrorHandler::CleanErrorMessage(python_traceback);
  EXPECT_THAT(cleaned, ::testing::HasSubstr("RuntimeError: Out of memory"));
  EXPECT_THAT(cleaned, ::testing::Not(::testing::HasSubstr("Traceback")));
  EXPECT_THAT(cleaned, ::testing::Not(::testing::HasSubstr("File ")));
}

TEST_F(ErrorHandlerTest, CleanErrorMessageMultipleErrors) {
  std::string multiple_errors = R"(First error occurred
Second error: Failed to allocate
Third error: Timeout)";

  std::string cleaned = ErrorHandler::CleanErrorMessage(multiple_errors);
  // Should return the last meaningful line
  EXPECT_EQ(cleaned, "Third error: Timeout");
}

// Test CheckAndThrowPendingException with clear_exception parameter
TEST_F(ErrorHandlerTest, CheckAndThrowPendingExceptionWithoutClearing) {
  uint32_t stream_id = 888;

  // Record an exception
  auto exception = std::make_exception_ptr(std::runtime_error("Persistent exception"));
  error_handler_->RecordFirstException(stream_id, exception);

  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Check and throw without clearing (clear_exception = false)
  EXPECT_THROW(error_handler_->CheckAndThrowPendingException(stream_id, false), std::runtime_error);

  // Exception should still be pending
  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Throw again - should still work
  EXPECT_THROW(error_handler_->CheckAndThrowPendingException(stream_id, false), std::runtime_error);

  // Now clear it
  EXPECT_THROW(error_handler_->CheckAndThrowPendingException(stream_id, true), std::runtime_error);

  // Should no longer be pending
  EXPECT_FALSE(error_handler_->has_pending_exception_);
}

// Test Multiple calls to RecordFirstException with different scenarios
TEST_F(ErrorHandlerTest, RecordFirstExceptionMultipleStreams) {
  uint32_t stream_id_1 = 100;
  uint32_t stream_id_2 = 200;

  // Record first exception from stream 1
  auto exception1 = std::make_exception_ptr(std::runtime_error("Exception from stream 1"));
  error_handler_->RecordFirstException(stream_id_1, exception1);

  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Try to record exception from stream 2 - should be ignored
  auto exception2 = std::make_exception_ptr(std::runtime_error("Exception from stream 2"));
  error_handler_->RecordFirstException(stream_id_2, exception2);

  // Should still have the first exception
  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // When we throw, it should be the first exception
  try {
    error_handler_->CheckAndThrowPendingException(stream_id_1, true);
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "Exception from stream 1");
  }
}

TEST_F(ErrorHandlerTest, RecordFirstExceptionSameStreamMultipleTimes) {
  uint32_t stream_id = 300;

  // Record first exception
  auto exception1 = std::make_exception_ptr(std::runtime_error("First exception"));
  error_handler_->RecordFirstException(stream_id, exception1);

  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Try to record another exception from same stream - should be ignored
  auto exception2 = std::make_exception_ptr(std::runtime_error("Second exception"));
  error_handler_->RecordFirstException(stream_id, exception2);

  auto exception3 = std::make_exception_ptr(std::runtime_error("Third exception"));
  error_handler_->RecordFirstException(stream_id, exception3);

  // Should still have the first exception
  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Verify it's the first exception
  try {
    error_handler_->CheckAndThrowPendingException(stream_id, true);
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "First exception");
  }
}

TEST_F(ErrorHandlerTest, RecordFirstExceptionAfterClearing) {
  uint32_t stream_id = 400;

  // Record and clear first exception
  auto exception1 = std::make_exception_ptr(std::runtime_error("First exception"));
  error_handler_->RecordFirstException(stream_id, exception1);
  EXPECT_THROW(error_handler_->CheckAndThrowPendingException(stream_id, true), std::runtime_error);
  EXPECT_FALSE(error_handler_->has_pending_exception_);

  // Now record a new exception - should succeed
  auto exception2 = std::make_exception_ptr(std::runtime_error("Second exception"));
  error_handler_->RecordFirstException(stream_id, exception2);
  EXPECT_TRUE(error_handler_->has_pending_exception_);

  // Verify it's the second exception
  try {
    error_handler_->CheckAndThrowPendingException(stream_id, true);
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "Second exception");
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
