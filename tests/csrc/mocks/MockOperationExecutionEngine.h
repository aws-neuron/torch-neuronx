#pragma once

#include <gmock/gmock.h>

#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/runtime/ModelHandleCache.h"
#include "torch_neuronx/csrc/core/streams/StreamImpl.h"

namespace at::neuron::testing {

// Mock OperationExecutionEngine that skips actual device execution
// This allows testing stream management without needing real device tensors or NRT.
class MockOperationExecutionEngine : public OperationExecutionEngine {
 public:
  MockOperationExecutionEngine(CompilationCache* compilation_cache,
                               ModelHandleCache* model_handle_cache)
      : OperationExecutionEngine(compilation_cache, model_handle_cache) {}

  using OperationExecutionEngine::execution_queue_depth_;
  using OperationExecutionEngine::pending_async_ops_count_;
  using OperationExecutionEngine::ProcessCompletions;

 protected:
  void ProcessExecutionTask(OperationContext* operation) noexcept override {
    auto* stream = operation->stream;
    operation->StartExecution();
    operation->execute_end = std::chrono::steady_clock::now();
    stream->CompleteOperation(operation);
  }
};

// Wrapper for backward compatibility
class MockOperationExecutionEngineWrapper {
 public:
  MockOperationExecutionEngineWrapper()
      : compilation_cache_(),
        model_handle_cache_(),
        engine_(&compilation_cache_, &model_handle_cache_) {}

  ~MockOperationExecutionEngineWrapper() { engine_.Shutdown(); }

  OperationExecutionEngine* GetEngine() { return &engine_; }

  // Manually complete an operation (called by test after submission)
  void CompleteOperation(OperationContext* op, StreamImpl* stream) {
    if (op && stream) {
      stream->CompleteOperation(op);
    }
  }

 private:
  CompilationCache compilation_cache_;
  ModelHandleCache model_handle_cache_;
  MockOperationExecutionEngine engine_;
};

}  // namespace at::neuron::testing
