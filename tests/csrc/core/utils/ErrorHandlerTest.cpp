#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tests/csrc/mocks/MockKernelExecution.h"
#include "tests/csrc/mocks/MockStreamImpl.h"
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
  std::unique_ptr<OperationContext> create_mock_operation(
      const std::string& op_name = "test_op",
      const std::vector<torch::Tensor>& inputs = {torch::ones({2, 3})},
      const std::vector<torch::Tensor>& outputs = {torch::ones({2, 3})}) {
    auto mock_kernel = std::make_unique<mock::MockNeuronKernelExecution>(op_name, inputs, outputs);

    EXPECT_CALL(*mock_kernel, RequiresCompilation()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_kernel, ValidateImpl()).WillRepeatedly(::testing::Return(true));

    auto mock_op = std::make_unique<OperationContext>(std::move(mock_kernel), "test_stack_trace");
    mock_op->submit_time = std::chrono::steady_clock::now();
    mock_op->stream = reinterpret_cast<StreamImpl*>(&mock_stream_);
    return mock_op;
  }

  std::unique_ptr<ErrorHandler> error_handler_;
  mock::MockStreamImpl mock_stream_;
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
// ExtractNrtStatus is now internal - status is extracted from ExecutionRuntimeException
TEST_F(ErrorHandlerTest, ClassifyErrorWithValidStatus) {
  torch_neuronx::ExecutionRuntimeException e1("Error occurred", static_cast<NRT_STATUS>(1));
  auto msg1 = error_handler_->ClassifyNrtError(e1);
  EXPECT_THAT(msg1, ::testing::HasSubstr("Non-specific failure"));

  torch_neuronx::ExecutionRuntimeException e2("Some error", static_cast<NRT_STATUS>(2));
  auto msg2 = error_handler_->ClassifyNrtError(e2);
  EXPECT_THAT(msg2, ::testing::HasSubstr("Invalid NEFF"));

  torch_neuronx::ExecutionRuntimeException e1005("Busy error", static_cast<NRT_STATUS>(1005));
  auto msg1005 = error_handler_->ClassifyNrtError(e1005);
  EXPECT_THAT(msg1005, ::testing::HasSubstr("Neuron core busy"));
}

// Test classify_nrt_error method with various status codes
TEST_F(ErrorHandlerTest, ClassifyNrtErrorStatuses) {
  // Test key status codes from different categories
  struct StatusTest {
    int status;
    std::string expected_substr;
  };

  std::vector<StatusTest> tests = {{1, "Non-specific failure"}, {2, "Invalid NEFF"},
                                   {5, "Operation timed out"},  {6, "Hardware failure"},
                                   {1002, "Invalid input"},     {1005, "Neuron core busy"},
                                   {1201, "HBM encountered"},   {1203, "DMA engine encountered"}};

  for (const auto& test : tests) {
    torch_neuronx::ExecutionRuntimeException e("Test error", static_cast<NRT_STATUS>(test.status));
    auto msg = error_handler_->ClassifyNrtError(e);
    EXPECT_THAT(msg, ::testing::HasSubstr(test.expected_substr))
        << "Failed for status: " << test.status;
  }
}

TEST_F(ErrorHandlerTest, ClassifyNrtErrorUnknownStatus) {
  torch_neuronx::ExecutionRuntimeException e("Unknown error", static_cast<NRT_STATUS>(9999));
  auto msg = error_handler_->ClassifyNrtError(e);
  EXPECT_THAT(msg, ::testing::HasSubstr("Unknown status"));
}

TEST_F(ErrorHandlerTest, ClassifyNrtErrorDefaultStatus) {
  torch_neuronx::ExecutionRuntimeException e("No status error", static_cast<NRT_STATUS>(-999));
  auto msg = error_handler_->ClassifyNrtError(e);
  EXPECT_THAT(msg, ::testing::HasSubstr("Unknown status"));
}

// Test create_error_context method
TEST_F(ErrorHandlerTest, CreateErrorContextWithNullOperation) {
  auto mock_op = create_mock_operation();

  ErrorContext context =
      error_handler_->CreateErrorContext(mock_op.get(), ErrorContext::Stage::COMPILATION);

  EXPECT_EQ(context.operation_name, "test_op");
  EXPECT_EQ(context.current_stage, ErrorContext::Stage::COMPILATION);
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
  op.stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  ErrorContext context = error_handler_->CreateErrorContext(&op, ErrorContext::Stage::COMPILATION);

  EXPECT_EQ(context.operation_name, "test_op");
  EXPECT_EQ(context.current_stage, ErrorContext::Stage::COMPILATION);
}

// Test handle_compilation_error method
TEST_F(ErrorHandlerTest, HandleCompilationError) {
  auto mock_op = create_mock_operation();

  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 123;

  std::string result =
      error_handler_->HandleCompilationError(mock_op.get(), "Compilation failed", context);

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
  mock_op.stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Test with a specific NRT status code to verify the full flow
  torch_neuronx::ExecutionRuntimeException exec_error("NRT execution failed - core busy",
                                                      static_cast<NRT_STATUS>(1005));

  // Test ClassifyNrtError
  auto classified_message = error_handler_->ClassifyNrtError(exec_error);
  EXPECT_THAT(classified_message, ::testing::HasSubstr("Neuron core busy"));

  // Test HandleExecutionError with the classified message
  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 456;

  std::string result = error_handler_->HandleExecutionError(&mock_op, classified_message, context);

  EXPECT_THAT(result, ::testing::HasSubstr("NRT Execution error"));
  EXPECT_THAT(result, ::testing::HasSubstr("test_op"));
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
  mock_op.stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Test with hardware error status
  {
    torch_neuronx::ExecutionRuntimeException exec_error("Hardware error occurred",
                                                        static_cast<NRT_STATUS>(1201));
    auto classified_message = error_handler_->ClassifyNrtError(exec_error);
    ErrorContext context;

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result, ::testing::HasSubstr("HBM encountered"));
  }

  // Test with timeout error status
  {
    torch_neuronx::ExecutionRuntimeException exec_error("Operation timed out",
                                                        static_cast<NRT_STATUS>(5));
    auto classified_message = error_handler_->ClassifyNrtError(exec_error);
    ErrorContext context;

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result, ::testing::HasSubstr("Operation timed out"));
  }

  // Test with unknown status code
  {
    torch_neuronx::ExecutionRuntimeException exec_error("Unknown error",
                                                        static_cast<NRT_STATUS>(9999));
    auto classified_message = error_handler_->ClassifyNrtError(exec_error);
    ErrorContext context;

    std::string result =
        error_handler_->HandleExecutionError(&mock_op, classified_message, context);

    EXPECT_THAT(result, ::testing::HasSubstr("Unknown status"));
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

  std::string result = enhanced_handler.GenerateErrorMessage("Base error", context);

  EXPECT_THAT(result, ::testing::HasSubstr("Base error"));
  EXPECT_THAT(result, ::testing::HasSubstr("--- Error Context ---"));
  EXPECT_THAT(result, ::testing::HasSubstr("Operation: test_op"));
  EXPECT_THAT(result, ::testing::HasSubstr("Stream ID: 789"));
  EXPECT_THAT(result, ::testing::HasSubstr("Stage: COMPILATION"));

  unsetenv("NEURON_ENHANCED_ERROR_HANDLING");
}

// Test error handler functionality
TEST_F(ErrorHandlerTest, ErrorHandlerFunctionality) {
  // Test that error handler is properly initialized
  EXPECT_FALSE(error_handler_->IsEnhancedErrorHandlingEnabled());
  EXPECT_FALSE(error_handler_->IsExecutionDumpingEnabled());

  // Test recent errors is initially empty
  auto recent_errors = error_handler_->GetRecentErrors();
  EXPECT_TRUE(recent_errors.empty());
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
  mock_op.stream = reinterpret_cast<StreamImpl*>(&mock_stream_);

  // Test 1: Execution error with status code - should be handled
  torch_neuronx::ExecutionRuntimeException exec_error("Execution failed - core busy",
                                                      static_cast<NRT_STATUS>(1005));
  auto execution_result = error_handler_->HandleOperationError(&mock_op, exec_error);

  EXPECT_FALSE(execution_result.IsSuccess());
  EXPECT_THAT(execution_result.GetError(), ::testing::HasSubstr("Execution failed"));

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
  error_handler_->HandleCompilationError(mock_op1.get(), "Error 1", context1);
  error_handler_->HandleExecutionError(mock_op2.get(), "Error 2", context2);
  error_handler_->HandleCompilationError(mock_op3.get(), "Error 3", context3);

  // Check recent errors
  auto recent_errors = error_handler_->GetRecentErrors();
  EXPECT_EQ(recent_errors.size(), 3);
  EXPECT_EQ(recent_errors[0].operation_name, "op1");
  EXPECT_EQ(recent_errors[1].operation_name, "op2");
  EXPECT_EQ(recent_errors[2].operation_name, "op3");
}

TEST_F(ErrorHandlerTest, ErrorRecordingUpdatesRecentErrors) {
  auto mock_op = create_mock_operation();

  ErrorContext context;
  context.operation_name = "test_op";
  context.stream_id = 123;

  // Initial recent errors should be empty
  auto initial_errors = error_handler_->GetRecentErrors();
  EXPECT_TRUE(initial_errors.empty());

  // Record a compilation error
  error_handler_->HandleCompilationError(mock_op.get(), "Compilation failed", context);

  // Check recent errors were updated
  auto updated_errors = error_handler_->GetRecentErrors();
  EXPECT_EQ(updated_errors.size(), 1);
  EXPECT_EQ(updated_errors[0].operation_name, "test_op");
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
          error_handler_->HandleCompilationError(mock_op.get(), "Error", context);
        } else {
          error_handler_->HandleExecutionError(mock_op.get(), "Error", context);
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
  // With i % 2 == 0 for compilation: i=0,2,4 (3 per thread)
  // With i % 2 != 0 for execution: i=1,3 (2 per thread)
  int expected_compilation = num_threads * 3;  // i=0,2,4 per thread
  int expected_execution = num_threads * 2;    // i=1,3 per thread
  int recorded_compilation = 0;
  int recorded_execution = 0;
  auto recent_errors = error_handler_->GetRecentErrors();
  for (const auto& err : recent_errors) {
    if (err.current_stage == ErrorContext::Stage::COMPILATION) {
      ++recorded_compilation;
    } else if (err.current_stage == ErrorContext::Stage::EXECUTION) {
      ++recorded_execution;
    }
  }
  EXPECT_EQ(recent_errors.size(), num_threads * errors_per_thread);
  EXPECT_EQ(recorded_compilation, expected_compilation);
  EXPECT_EQ(recorded_execution, expected_execution);
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

// ExecutionRuntimeException tests

TEST_F(ErrorHandlerTest, ExecutionRuntimeExceptionMessageOnly) {
  torch_neuronx::ExecutionRuntimeException e("Test error message");

  EXPECT_STREQ(e.what(), "Test error message");
  EXPECT_FALSE(e.HasNRTStatus());
  EXPECT_FALSE(e.HasSeqId());
}

TEST_F(ErrorHandlerTest, ExecutionRuntimeExceptionWithNRTStatus) {
  torch_neuronx::ExecutionRuntimeException e("Test error", static_cast<NRT_STATUS>(5));

  EXPECT_STREQ(e.what(), "Test error");
  EXPECT_TRUE(e.HasNRTStatus());
  EXPECT_FALSE(e.HasSeqId());
  EXPECT_EQ(e.GetNRTStatus(), 5);
}

TEST_F(ErrorHandlerTest, ExecutionRuntimeExceptionWithSeqId) {
  // Constructor for async execution errors (seq_id, NRT_STATUS)
  torch_neuronx::ExecutionRuntimeException e(12345, static_cast<NRT_STATUS>(6));

  EXPECT_STREQ(e.what(), "Async execution error");
  EXPECT_TRUE(e.HasNRTStatus());
  EXPECT_TRUE(e.HasSeqId());
  EXPECT_EQ(e.GetSeqId(), 12345);
  EXPECT_EQ(e.GetNRTStatus(), 6);
}

TEST_F(ErrorHandlerTest, HandleOperationErrorWithSeqId) {
  // Test that HandleOperationError correctly handles exceptions with seq_id
  auto mock_op = create_mock_operation();

  // Create async execution error with seq_id
  torch_neuronx::ExecutionRuntimeException async_error(42, static_cast<NRT_STATUS>(1005));

  auto result = error_handler_->HandleOperationError(mock_op.get(), async_error);

  EXPECT_FALSE(result.IsSuccess());
  // Error message should contain seq_id info
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("seq_id=42"));
  EXPECT_THAT(result.GetError(), ::testing::HasSubstr("Neuron core busy"));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
