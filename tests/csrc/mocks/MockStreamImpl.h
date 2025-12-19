#pragma once

#include <c10/core/Stream.h>
#include <gmock/gmock.h>

#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/ErrorHandler.h"

namespace at::neuron::testing {

class MockStreamImpl {
 public:
  c10::StreamId stream_id = 1;

  MOCK_METHOD(void, handle_error_with_cleanup,
              (OperationContext * op, const std::string& error, ErrorContext::Stage stage,
               int error_code),
              ());

  MOCK_METHOD(void, complete_operation, (OperationContext * op), ());

  MOCK_METHOD(void, WaitForPriorOperationsToComplete, (const OperationContext* op), ());

  // For tests that don't need mocking, provide simple implementations
  void SetDefaultBehavior() {
    ON_CALL(*this,
            handle_error_with_cleanup(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke([this](OperationContext* op, const std::string& error,
                                                ErrorContext::Stage stage, int error_code) {
          error_called = true;
          last_error = error;
          last_stage = stage;
        }));

    ON_CALL(*this, complete_operation(::testing::_))
        .WillByDefault(::testing::Invoke([this](OperationContext* op) { complete_called = true; }));

    ON_CALL(*this, WaitForPriorOperationsToComplete(::testing::_))
        .WillByDefault(::testing::Return());
  }

  // Test state tracking
  bool error_called = false;
  bool complete_called = false;
  std::string last_error;
  ErrorContext::Stage last_stage;
};

}  // namespace at::neuron::testing
